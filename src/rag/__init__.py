"""
General-purpose document RAG — intentionally separate from churn analytics.

Use for manuals, runbooks, and GitHub README/docs Q&A.
Churn tabular Q&A lives in the GenAI layer (Phase 3).
"""

from rag.ingest import (
    collection_stats,
    ingest_github_repo,
    ingest_uploads,
)
from rag.query import RAGAnswer, ask_documents

__all__ = [
    "RAGAnswer",
    "ask_documents",
    "collection_stats",
    "ingest_github_repo",
    "ingest_uploads",
]
