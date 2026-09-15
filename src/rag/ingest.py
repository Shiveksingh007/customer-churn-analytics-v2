"""
===========================================================
Module  : RAG Ingestion
Scope   : SEPARATE from churn analytics — general document Q&A
===========================================================

Parse PDF / Markdown / text (local upload or GitHub repo), chunk,
embed with sentence-transformers, and store in ChromaDB.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from config.settings import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE, RAG_STORE_DIR

GITHUB_REPO_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/(?P<owner>[\w\-.]+)/(?P<repo>[\w\-.]+)",
    re.IGNORECASE,
)
SUPPORTED_DOC_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".rst"}


class RAGDependencyError(RuntimeError):
    """Raised when optional Document RAG packages are not installed."""


def _dependency_error(package: str) -> RAGDependencyError:
    return RAGDependencyError(
        f"{package} is required for Document Q&A. Install optional dependencies with: "
        "pip install -r requirements-rag.txt"
    )


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int | None
    chunk_index: int

    @property
    def chunk_id(self) -> str:
        digest = hashlib.sha1(
            f"{self.source}|{self.page}|{self.chunk_index}|{self.text[:120]}".encode()
        ).hexdigest()[:16]
        return f"{Path(self.source).stem}_{self.chunk_index}_{digest}"


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def chunk_text(
    text: str,
    source: str,
    *,
    page: int | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[DocumentChunk]:
    """
    Split text into ~500-token chunks with overlap (word-based approximation).
    """
    chunk_size = RAG_CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = RAG_CHUNK_OVERLAP if overlap is None else overlap
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or greater and smaller than chunk_size.")
    words = text.split()
    if not words:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(
                DocumentChunk(text=piece, source=source, page=page, chunk_index=index)
            )
            index += 1
        if end >= len(words):
            break
        start = max(0, end - overlap)

    return chunks


def _read_pdf_pages(content: bytes) -> list[tuple[str, int]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise _dependency_error("pypdf") from exc

    reader = PdfReader(BytesIO(content))
    pages: list[tuple[str, int]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((text, i))
    if not pages:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "Use a text-based PDF or run OCR before uploading it."
        )
    return pages


def _read_text_file(content: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def parse_upload(filename: str, content: bytes) -> list[DocumentChunk]:
    """Parse an uploaded document into chunks."""
    suffix = Path(filename).suffix.lower()
    source = filename

    if suffix == ".pdf":
        all_chunks: list[DocumentChunk] = []
        for text, page in _read_pdf_pages(content):
            all_chunks.extend(chunk_text(text, source=source, page=page))
        return all_chunks

    if suffix in {".md", ".markdown", ".txt", ".rst"}:
        text = _read_text_file(content)
        return chunk_text(text, source=source, page=None)

    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_DOC_EXTENSIONS))}"
    )


def _http_get(url: str, headers: dict | None = None) -> bytes:
    req = Request(url, headers=headers or {"User-Agent": "Customer-Churn-Platform-RAG/1.0"})
    with urlopen(req, timeout=60) as response:
        return response.read()


def _parse_github_repo(url_or_slug: str) -> tuple[str, str, str]:
    text = url_or_slug.strip().rstrip("/")
    match = GITHUB_REPO_RE.search(text)
    if match:
        owner, repo = match.group("owner"), match.group("repo")
    elif re.fullmatch(r"[\w\-.]+/[\w\-.]+", text):
        owner, repo = text.split("/", 1)
    else:
        raise ValueError("Expected GitHub URL or owner/repo slug.")

    branch = ""
    if "/tree/" in text:
        branch = text.split("/tree/", 1)[1].split("/")[0]
    return owner, repo, branch


def _get_default_github_branch(owner: str, repo: str) -> str:
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        payload = json.loads(_http_get(api_url, headers={"Accept": "application/vnd.github+json"}))
        branch = str(payload.get("default_branch", "")).strip()
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not determine the default branch for {owner}/{repo}.") from exc
    if not branch:
        raise RuntimeError(f"GitHub did not return a default branch for {owner}/{repo}.")
    return branch


def fetch_github_documents(
    repo_url: str,
    *,
    max_files: int = 25,
) -> list[tuple[str, str]]:
    """
    Fetch markdown/text files from a public GitHub repository.

    Returns list of (relative_path, file_text).
    """
    owner, repo, branch = _parse_github_repo(repo_url)
    branch = branch or _get_default_github_branch(owner, repo)
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/"
        f"{quote(branch, safe='')}?recursive=1"
    )

    try:
        tree_payload = json.loads(_http_get(api_url, headers={"Accept": "application/vnd.github+json"}))
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not list GitHub repo {owner}/{repo}@{branch}. "
            "Ensure the repo is public or provide documents via upload."
        ) from exc

    paths = [
        item["path"]
        for item in tree_payload.get("tree", [])
        if item.get("type") == "blob"
        and Path(item["path"]).suffix.lower() in SUPPORTED_DOC_EXTENSIONS
        and item["path"].lower() != "license"
    ]
    paths = sorted(paths)[:max_files]

    documents: list[tuple[str, str]] = []
    for path in paths:
        raw_url = (
            f"https://raw.githubusercontent.com/{owner}/{repo}/{quote(branch, safe='')}/"
            f"{quote(path, safe='/')}"
        )
        try:
            text = _read_text_file(_http_get(raw_url))
        except (HTTPError, URLError):
            continue
        if text.strip():
            documents.append((f"github:{owner}/{repo}/{path}", text))

    if not documents:
        # Fallback: README only
        for readme in ("README.md", "readme.md", "README.MD"):
            raw_url = (
                f"https://raw.githubusercontent.com/{owner}/{repo}/{quote(branch, safe='')}/"
                f"{readme}"
            )
            try:
                text = _read_text_file(_http_get(raw_url))
                if text.strip():
                    documents.append((f"github:{owner}/{repo}/{readme}", text))
                    break
            except (HTTPError, URLError):
                continue

    if not documents:
        raise RuntimeError(f"No readable docs found in {owner}/{repo}@{branch}.")

    return documents


def github_to_chunks(repo_url: str) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for source, text in fetch_github_documents(repo_url):
        chunks.extend(chunk_text(text, source=source, page=None))
    return chunks


@lru_cache(maxsize=1)
def _get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise _dependency_error("sentence-transformers") from exc

    model_name = "all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)


def _get_chroma_collection(collection_name: str):
    try:
        import chromadb
    except ImportError as exc:
        raise _dependency_error("chromadb") from exc

    store = Path(RAG_STORE_DIR)
    store.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_chunks(
    chunks: Iterable[DocumentChunk],
    *,
    collection_name: str = "documents",
    reset: bool = False,
) -> int:
    """Embed chunks and upsert into Chroma. Returns number of chunks stored."""
    chunk_list = list(chunks)
    if not chunk_list:
        return 0

    if reset:
        try:
            import chromadb
        except ImportError as exc:
            raise _dependency_error("chromadb") from exc

        store = Path(RAG_STORE_DIR)
        store.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(store))
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = _get_chroma_collection(collection_name)
    embedder = _get_embedder()

    texts = [c.text for c in chunk_list]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    ids = [c.chunk_id for c in chunk_list]
    metadatas = [
        {
            "source": c.source,
            "page": c.page if c.page is not None else -1,
            "chunk_index": c.chunk_index,
        }
        for c in chunk_list
    ]

    # Upsert in batches for large uploads
    batch = 64
    for i in range(0, len(chunk_list), batch):
        sl = slice(i, i + batch)
        collection.upsert(
            ids=ids[sl],
            documents=texts[sl],
            embeddings=embeddings[sl],
            metadatas=metadatas[sl],
        )

    return len(chunk_list)


def ingest_uploads(
    files: Iterable[tuple[str, bytes]],
    *,
    collection_name: str = "documents",
    reset: bool = False,
) -> int:
    all_chunks: list[DocumentChunk] = []
    for name, content in files:
        all_chunks.extend(parse_upload(name, content))
    return ingest_chunks(all_chunks, collection_name=collection_name, reset=reset)


def ingest_github_repo(
    repo_url: str,
    *,
    collection_name: str = "documents",
    reset: bool = False,
) -> int:
    return ingest_chunks(github_to_chunks(repo_url), collection_name=collection_name, reset=reset)


def collection_stats(collection_name: str = "documents") -> dict:
    try:
        collection = _get_chroma_collection(collection_name)
        count = collection.count()
        return {"collection": collection_name, "chunks": count}
    except Exception:
        return {"collection": collection_name, "chunks": 0}
