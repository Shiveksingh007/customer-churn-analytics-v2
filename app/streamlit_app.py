"""
Customer Churn Analytics Platform — Streamlit dashboard.

Walks employees through the same stages as notebooks 01–09:
Overview → Data → Insights → Features → Predictions → Model → AI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
SRC = ROOT / "src"

for path in (str(APP_DIR), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.env_loader import load_project_env

load_project_env()

def _get_runtime_api_key() -> str:
    """Return a user-entered session key, then environment/Streamlit secrets."""
    session_key = str(st.session_state.get("gemini_api_key", "")).strip()
    if session_key:
        return session_key
    try:
        from utils.env_loader import get_gemini_api_key
        return get_gemini_api_key()
    except Exception:
        return ""


from analytics import build_demo_dataset, enrich_pipeline
from ui_rag import section_ask_documents
from ui_sections import (
    section_ai,
    section_business_insights,
    section_data_overview,
    section_features,
    section_model,
    section_overview,
    section_predictions,
)

st.set_page_config(
    page_title="Customer Churn Analytics Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# V2 Product Theme
# Keep the application light by default with strong contrast.  The previous
# OS-level dark-mode fallback could make light backgrounds inherit light text.
# --------------------------------------------------------------------------

_PALETTE = {
    "page_bg": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_alt": "#F5F8FC",
    "border": "#D8E1EA",
    "text": "#172033",
    "muted": "#526173",
    "primary": "#1565D8",
    "sidebar_bg": "#F7F9FC",
    "sidebar_text": "#172033",
    "hero_from": "#0B2545",
    "hero_to": "#1565D8",
    "hero_text": "#FFFFFF",
}

def render_theme_css() -> None:
    c = _PALETTE
    st.markdown(
        f"""
<style>
    .stApp {{
        background: {c["page_bg"]};
        color: {c["text"]};
    }}
    .block-container {{
        padding-top: 1.35rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }}

    /* High-contrast typography */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {{
        color: {c["text"]} !important;
    }}
    h1, h2, h3 {{
        letter-spacing: -0.02em;
    }}
    [data-testid="stCaptionContainer"] {{
        color: {c["muted"]} !important;
    }}

    /* Light, readable sidebar */
    [data-testid="stSidebar"] {{
        background: {c["sidebar_bg"]} !important;
        border-right: 1px solid {c["border"]};
    }}
    [data-testid="stSidebar"] * {{
        color: {c["sidebar_text"]} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
        background: {c["surface"]} !important;
        border: 1px dashed #B8C7D8;
    }}
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {{
        background: #FFFFFF !important;
        color: {c["text"]} !important;
        border: 1px solid #B8C7D8 !important;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: {c["surface_alt"]};
        border: 1px solid {c["border"]};
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 2px rgba(23,32,51,0.04);
    }}
    div[data-testid="stMetric"] * {{
        color: {c["text"]} !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {c["muted"]} !important;
    }}

    /* Tables / expanders */
    div[data-testid="stExpander"] {{
        background: {c["surface"]};
        border: 1px solid {c["border"]};
        border-radius: 10px;
    }}
    div[data-testid="stExpander"] * {{
        color: {c["text"]} !important;
    }}

    /* Hero */
    .hero-band {{
        background: linear-gradient(120deg, {c["hero_from"]} 0%, {c["hero_to"]} 100%);
        color: {c["hero_text"]};
        padding: 1.35rem 1.55rem;
        border-radius: 14px;
        margin-bottom: 1.35rem;
        box-shadow: 0 4px 14px rgba(21,101,216,0.12);
    }}
    .hero-band h1 {{
        color: {c["hero_text"]} !important;
        margin: 0 0 0.35rem 0;
        font-size: 1.9rem;
    }}
    .hero-band p {{
        color: {c["hero_text"]} !important;
        margin: 0;
        opacity: 0.92;
    }}

    /* API status card */
    .api-status {{
        border: 1px solid #B8E0C5;
        background: #F0FBF3;
        border-radius: 9px;
        padding: 0.55rem 0.7rem;
        margin-top: 0.45rem;
        font-size: 0.82rem;
        color: #1E6B35 !important;
    }}
    .api-status.missing {{
        border-color: #E5C98B;
        background: #FFF9EA;
        color: #7A5A00 !important;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


render_theme_css()

SECTIONS = {
    "Executive Overview": section_overview,
    "Data Overview": section_data_overview,
    "Business Insights": section_business_insights,
    "Feature Engineering": section_features,
    "Risk Predictions": section_predictions,
    "Machine Learning": section_model,
    "AI Assistant": section_ai,
}

RAG_PAGE = "Ask Your Documents"
ALL_PAGES = list(SECTIONS.keys()) + [RAG_PAGE]


def _load_upload(uploaded) -> "pd.DataFrame | None":
    import pandas as pd

    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded)
    if name.endswith(".json"):
        return pd.read_json(uploaded)
    st.error("Supported uploads: CSV, Parquet, JSON")
    return None


with st.sidebar:
    st.markdown("### ◈ Customer Churn")
    st.caption("Analytics Platform · V2")
    st.divider()

    page = st.radio(
        "Navigate",
        ALL_PAGES,
        index=0,
    )

    st.divider()

    if page == RAG_PAGE:
        st.markdown("#### Document RAG")
        st.caption("PDF · Markdown · GitHub docs")
    else:
        st.markdown("#### Dataset")
        uploaded = st.file_uploader(
            "Upload customer data",
            type=["csv", "parquet", "json"],
            help="Upload a CSV, Parquet, or JSON customer dataset.",
        )
        use_demo = st.checkbox(
            "Use demo Telco-style dataset",
            value=uploaded is None,
        )
        demo_size = (
            st.slider("Demo customers", 100, 2000, 500, 100)
            if use_demo and uploaded is None
            else 500
        )

    # API settings belong next to the dataset controls so a business user can
    # configure AI without touching .env files or deployment code.
    st.divider()
    st.markdown("#### 🔑 API Settings")
    st.caption("Optional · used by AI Assistant and document Q&A")

    existing_key = _get_runtime_api_key()
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=existing_key,
        type="password",
        placeholder="Paste your Gemini API key",
        help="The key is kept in this browser session and is not written to the project files.",
        key="gemini_api_key_input",
    )

    if api_key_input.strip():
        st.session_state["gemini_api_key"] = api_key_input.strip()
        st.markdown(
            '<div class="api-status">✓ Gemini API key configured for this session.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear API key", use_container_width=True, key="clear_gemini_key"):
            st.session_state.pop("gemini_api_key", None)
            st.session_state.pop("gemini_api_key_input", None)
            st.rerun()
    else:
        st.markdown(
            '<div class="api-status missing">○ No API key — AI features remain unavailable.</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### App Guide")
    st.caption("1–3 Data · 4 Insights · 5–6 Features")
    st.caption("7–9 Predictions & Model · GenAI → AI Assistant")
    st.caption("The V1 ML pipeline remains the backend foundation.")


# Initialize sidebar-only vars when on RAG page
if page != RAG_PAGE:
    pass  # uploaded, use_demo, demo_size set in sidebar branch above
else:
    uploaded = None
    use_demo = False
    demo_size = 500

# Make the session key available to existing GenAI modules without writing it
# to disk. Existing modules that read GEMINI_API_KEY will therefore continue
# to work unchanged.
_runtime_key = str(st.session_state.get("gemini_api_key", "")).strip()
if _runtime_key:
    import os
    os.environ["GEMINI_API_KEY"] = _runtime_key


st.markdown(
    """
<div class="hero-band">
  <h1>Customer Churn Analytics Platform</h1>
  <p>End-to-end walkthrough for teams — from raw data quality to risk scores, revenue at risk, and AI-assisted action.</p>
</div>
""",
    unsafe_allow_html=True,
)

if page == RAG_PAGE:
    section_ask_documents()
else:
    if uploaded is not None:
        raw = _load_upload(uploaded)
        if raw is None:
            st.stop()
        df = enrich_pipeline(raw)
        st.success(
            f"Loaded upload · {len(df):,} rows · {df.shape[1]} columns "
            "(Gold features + risk scores applied when needed)."
        )
    elif use_demo:
        df = build_demo_dataset(n=demo_size)
        st.info(
            f"Demo mode · {len(df):,} synthetic Telco-style customers "
            "(heuristic risk scores for local demos)."
        )
    else:
        st.warning("Upload a dataset or enable demo mode.")
        st.stop()

    SECTIONS[page](df)