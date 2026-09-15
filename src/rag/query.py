"""
===========================================================
Module  : RAG Query
Scope   : SEPARATE from churn analytics — grounded document Q&A
===========================================================

Retrieve top-k chunks from Chroma and answer with Gemini + citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genai.llm_client import call_llm
from rag.ingest import _get_chroma_collection, _get_embedder


@dataclass
class Citation:
    source: str
    page: int | None
    chunk_index: int
    text: str
    score: float

    @property
    def snippet(self) -> str:
        """Short display-only preview; the answer model receives the full chunk."""
        return self.text[:400]

    def label(self) -> str:
        page_part = f", p.{self.page}" if self.page and self.page > 0 else ""
        return f"{self.source}{page_part} · chunk {self.chunk_index}"


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[Citation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [
                {
                    "source": c.source,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "snippet": c.snippet,
                    "score": c.score,
                    "label": c.label(),
                }
                for c in self.citations
            ],
        }


_RAG_SYSTEM = """\
You answer questions using ONLY the provided document excerpts.

Rules:
- If the excerpts do not contain enough information, say so clearly.
- Cite sources inline using [1], [2], etc. matching the excerpt numbers.
- Be concise and practical (2–5 sentences unless the user asks for detail).
- Do not invent facts not supported by the excerpts.
"""


def retrieve_chunks(
    question: str,
    *,
    collection_name: str = "documents",
    top_k: int = 5,
) -> list[Citation]:
    collection = _get_chroma_collection(collection_name)
    if collection.count() == 0:
        return []

    embedder = _get_embedder()
    query_embedding = embedder.encode([question], show_progress_bar=False).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    citations: list[Citation] = []
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        page = meta.get("page")
        page = None if page is None or int(page) < 0 else int(page)
        citations.append(
            Citation(
                source=str(meta.get("source", "unknown")),
                page=page,
                chunk_index=int(meta.get("chunk_index", 0)),
                text=str(doc),
                score=float(1.0 - dist) if dist is not None else 0.0,
            )
        )

    return citations


def ask_documents(
    question: str,
    *,
    collection_name: str = "documents",
    top_k: int = 5,
) -> RAGAnswer:
    """
    Retrieve relevant chunks and produce a grounded answer with citations.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    citations = retrieve_chunks(question, collection_name=collection_name, top_k=top_k)
    if not citations:
        return RAGAnswer(
            question=question,
            answer="No documents indexed yet. Upload files or ingest a GitHub repo first.",
            citations=[],
        )

    context_blocks = []
    for i, cite in enumerate(citations, start=1):
        page_note = f" (page {cite.page})" if cite.page else ""
        context_blocks.append(
            f"[{i}] Source: {cite.source}{page_note}\n{cite.text}"
        )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Document excerpts:\n\n"
        + "\n\n".join(context_blocks)
    )

    answer = call_llm(_RAG_SYSTEM, user_prompt, max_output_tokens=1024)
    return RAGAnswer(question=question, answer=answer, citations=citations)
