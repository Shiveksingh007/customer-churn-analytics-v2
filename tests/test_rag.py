"""Unit tests for document RAG chunking (no Chroma/Gemini required)."""

from __future__ import annotations

import pytest

from rag.ingest import (
    DocumentChunk,
    chunk_text,
    fetch_github_documents,
    parse_upload,
)
from rag.query import Citation, ask_documents


def test_chunk_text_respects_overlap():
    words = " ".join(f"w{i}" for i in range(1200))
    chunks = chunk_text(words, source="test.txt", chunk_size=500, overlap=50)
    assert len(chunks) >= 2
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert chunks[0].source == "test.txt"
    assert chunks[0].text.split()[-50:] == chunks[1].text.split()[:50]


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (10, -1), (10, 10)])
def test_chunk_text_rejects_invalid_configuration(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("one two", "test.txt", chunk_size=chunk_size, overlap=overlap)


def test_parse_markdown_upload():
    content = b"# Title\n\nSome policy text about warranty coverage.\n"
    chunks = parse_upload("manual.md", content)
    assert len(chunks) >= 1
    assert "warranty" in chunks[0].text.lower()


def test_fetch_github_documents_uses_repo_default_branch(monkeypatch):
    calls = []

    def fake_http_get(url, headers=None):
        calls.append(url)
        if url == "https://api.github.com/repos/acme/handbook":
            return b'{"default_branch": "master"}'
        if "/git/trees/master?recursive=1" in url:
            return b'{"tree": [{"type": "blob", "path": "docs/guide.md"}]}'
        if url.endswith("/master/docs/guide.md"):
            return b"# Guide\n\nRetention policy details."
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("rag.ingest._http_get", fake_http_get)

    documents = fetch_github_documents("acme/handbook")

    assert documents == [("github:acme/handbook/docs/guide.md", "# Guide\n\nRetention policy details.")]
    assert calls[0] == "https://api.github.com/repos/acme/handbook"


def test_ask_documents_passes_full_chunk_text_to_model(monkeypatch):
    full_text = "A" * 600
    citation = Citation("manual.pdf", 3, 2, full_text, 0.9)
    captured = {}

    monkeypatch.setattr("rag.query.retrieve_chunks", lambda *args, **kwargs: [citation])
    monkeypatch.setattr(
        "rag.query.call_llm",
        lambda system, prompt, max_output_tokens: captured.update({"prompt": prompt})
        or "Answer [1]",
    )

    result = ask_documents("What does the manual say?")

    assert result.citations[0].snippet == full_text[:400]
    assert full_text in captured["prompt"]
    assert result.answer == "Answer [1]"
