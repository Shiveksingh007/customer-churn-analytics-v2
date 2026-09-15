"""Streamlit UI for the separate Document RAG module (Phase 7)."""

from __future__ import annotations

import streamlit as st

from rag import ask_documents, collection_stats, ingest_github_repo, ingest_uploads


def section_ask_documents() -> None:
    st.header("Ask Your Documents")
    st.caption(
        "Separate module · General document Q&A (PDF, Markdown, GitHub docs). "
        "Not part of the churn analytics pipeline."
    )

    st.info(
        "This is a **general-purpose RAG tool** bundled alongside the churn platform. "
        "For questions about your **customer dataset**, use **AI Assistant** instead."
    )

    stats = collection_stats()
    st.metric("Indexed chunks", stats["chunks"])

    tab_upload, tab_github, tab_chat = st.tabs(["Upload documents", "GitHub repo", "Ask"])

    with tab_upload:
        st.subheader("Upload PDF / Markdown / text")
        uploads = st.file_uploader(
            "Choose files",
            type=["pdf", "md", "markdown", "txt", "rst"],
            accept_multiple_files=True,
            key="rag_uploads",
        )
        reset = st.checkbox("Replace existing index (clear before ingest)", value=False)
        if st.button("Index uploaded files", type="primary", disabled=not uploads):
            with st.spinner("Parsing, chunking, embedding…"):
                files = [(f.name, f.getvalue()) for f in uploads]
                try:
                    count = ingest_uploads(files, reset=reset)
                except Exception as exc:
                    st.error(f"Could not index the uploaded files: {exc}")
                else:
                    st.success(f"Indexed **{count}** chunks from {len(files)} file(s).")
                    st.rerun()

    with tab_github:
        st.subheader("Ingest from a public GitHub repo")
        repo = st.text_input(
            "Repo URL or owner/name",
            placeholder="https://github.com/owner/repo or owner/repo",
        )
        if st.button("Index GitHub docs", disabled=not repo.strip()):
            with st.spinner("Fetching markdown/text from GitHub…"):
                try:
                    count = ingest_github_repo(repo.strip(), reset=True)
                    st.success(f"Indexed **{count}** chunks from `{repo.strip()}`.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with tab_chat:
        st.subheader("Ask a question")
        if stats["chunks"] == 0:
            st.warning("Index at least one document first (Upload or GitHub tab).")
            return

        if "rag_history" not in st.session_state:
            st.session_state.rag_history = []

        question = st.text_input(
            "Your question",
            placeholder="What does the manual say about warranty coverage?",
            key="rag_question",
        )
        top_k = st.slider("Context chunks (top-k)", 3, 10, 5)

        if st.button("Ask documents", type="primary", disabled=not question.strip()):
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    result = ask_documents(question.strip(), top_k=top_k)
                    st.session_state.rag_history.append(result)
                except Exception as exc:
                    st.error(f"RAG error: {exc}")

        for turn in reversed(st.session_state.rag_history[-10:]):
            with st.chat_message("user"):
                st.write(turn.question)
            with st.chat_message("assistant"):
                st.markdown(turn.answer)
                if turn.citations:
                    st.markdown("**Sources**")
                    for i, cite in enumerate(turn.citations, start=1):
                        page = f" · page {cite.page}" if cite.page else ""
                        st.markdown(f"**[{i}]** `{cite.source}`{page} (chunk {cite.chunk_index})")
                        with st.expander(f"Excerpt [{i}]"):
                            st.write(cite.snippet)
