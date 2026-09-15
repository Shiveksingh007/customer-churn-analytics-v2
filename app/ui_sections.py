"""
Streamlit UI sections mirroring notebooks 01–09 for employee walkthroughs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analytics import (
    FEATURE_IMPORTANCE,
    MODEL_METRICS,
    available_insight_columns,
    churn_by_segment,
    contract_risk_matrix,
    data_quality_report,
    executive_kpis,
    feature_distribution,
    pipeline_steps,
    risk_distribution,
    risk_profile_summary,
    top_high_risk,
)


def _metric_row(items: list[tuple[str, object]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        display = "—" if value is None else value
        col.metric(label, display)


def section_overview(df: pd.DataFrame) -> None:
    st.header("Executive Overview")
    st.caption(
        "Notebook 08 · One-screen summary for managers — risk mix, revenue at risk, and pipeline status."
    )

    kpis = executive_kpis(df)
    _metric_row(
        [
            ("Customers", f"{kpis['total_customers']:,}"),
            ("Churn rate", f"{kpis['churn_rate_pct']}%" if kpis["churn_rate_pct"] is not None else "—"),
            ("High risk", kpis["high_risk"]),
            ("Monthly $ at risk", f"${kpis['monthly_revenue_at_risk']:,.0f}"),
            ("Annual $ at risk", f"${kpis['annual_revenue_at_risk']:,.0f}"),
        ]
    )

    left, right = st.columns(2)
    with left:
        dist = risk_distribution(df)
        if not dist.empty:
            fig = px.pie(
                dist,
                names="Risk_Level",
                values="count",
                title="Customer risk mix",
                color="Risk_Level",
                color_discrete_map={"High": "#C44B3C", "Medium": "#D4A017", "Low": "#2F6F4E"},
            )
            st.plotly_chart(fig, use_container_width=True)
    with right:
        profile = risk_profile_summary(df)
        if not profile.empty:
            st.subheader("Profile by risk level")
            st.dataframe(profile, use_container_width=True, hide_index=True)

    st.subheader("How the platform was built")
    steps = pd.DataFrame(pipeline_steps())
    st.dataframe(steps, use_container_width=True, hide_index=True)

    with st.expander("Personas & recommended playbooks"):
        st.markdown(
            """
| Persona | Typical profile | Action |
|---------|-----------------|--------|
| **High risk** | Month-to-month, often Fiber, shorter tenure, higher charges | Immediate retention call |
| **Medium risk** | Mixed contracts, mid charges | Discount / bundle / add Tech Support |
| **Low risk** | One/Two year contracts, longer tenure | Loyalty program |
"""
        )


def section_data_overview(df: pd.DataFrame) -> None:
    st.header("Data Overview")
    st.caption("Notebooks 01–03 · Ingestion, validation, and cleaning health checks.")

    report = data_quality_report(df)
    _metric_row(
        [
            ("Rows", f"{report['rows']:,}"),
            ("Columns", report["columns"]),
            ("Duplicate rows", report["duplicate_rows"]),
            ("Cols with nulls", report["columns_with_nulls"]),
            ("PK unique (customerID)", "Yes" if report["primary_key_unique"] else ("No" if report["primary_key_unique"] is False else "n/a")),
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sample rows (Bronze preview)")
        st.dataframe(df.head(15), use_container_width=True)
    with c2:
        st.subheader("Schema")
        schema = pd.DataFrame(
            {"column": df.columns, "dtype": [str(t) for t in df.dtypes]}
        )
        st.dataframe(schema, use_container_width=True, hide_index=True)

    if report["nulls_by_column"]:
        null_df = (
            pd.Series(report["nulls_by_column"])
            .rename("null_count")
            .reset_index()
            .rename(columns={"index": "column"})
        )
        fig = px.bar(null_df, x="column", y="null_count", title="Nulls by column")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No null values detected in the current dataset.")

    if "Churn" in df.columns:
        churn_dist = (
            df["Churn"].astype(str).value_counts().rename_axis("Churn").reset_index(name="count")
        )
        fig = px.bar(churn_dist, x="Churn", y="count", title="Target distribution (Churn)")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Cleaning note (Notebook 03): blank `TotalCharges` values are coerced to numeric — "
        f"blank/null TotalCharges count = **{report['blank_or_null_TotalCharges']}**."
    )


def section_business_insights(df: pd.DataFrame) -> None:
    st.header("Business Insights")
    st.caption("Notebooks 04 & 06 · Where churn concentrates — segments that retention teams should watch.")

    kpis = executive_kpis(df)
    _metric_row(
        [
            ("Total", f"{kpis['total_customers']:,}"),
            ("Active", kpis["active_customers"]),
            ("Churned", kpis["churned_customers"]),
            ("Churn rate", f"{kpis['churn_rate_pct']}%" if kpis["churn_rate_pct"] is not None else "—"),
            ("Avg monthly $", kpis["avg_monthly_charges"]),
            ("Avg tenure", kpis["avg_tenure"]),
        ]
    )

    if "Churn" not in df.columns:
        st.warning("No `Churn` column found — upload a labeled dataset for historical insights.")
        return

    options = available_insight_columns(df)
    extra = [c for c in ("HasFamily", "TenureGroup", "MonthlyChargeLevel", "IsLongTermContract") if c in df.columns]
    options = list(dict.fromkeys(options + extra))
    if not options:
        st.warning("No categorical segment columns available.")
        return

    selected = st.multiselect(
        "Segments to analyze",
        options=options,
        default=[c for c in ("Contract", "InternetService", "TechSupport", "HasFamily") if c in options][:4],
    )

    for col in selected:
        rates = churn_by_segment(df, col)
        if rates.empty:
            continue
        st.subheader(f"Churn rate by {col}")
        fig = px.bar(
            rates,
            x=col,
            y="Churn_Rate",
            text="Churn_Rate",
            hover_data=["Total_Customers", "Churned_Customers"],
            title=f"Churn % by {col}",
            labels={"Churn_Rate": "Churn rate (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rates, use_container_width=True, hide_index=True)

    with st.expander("Business takeaways (from the notebook analysis)"):
        st.markdown(
            """
- **Month-to-month** contracts drive most churn vs one/two-year commitments.
- **Fiber optic** customers churn more than DSL or no-internet.
- Missing **Tech Support** / **Online Security** correlates with higher churn.
- Customers **with family** (partner/dependents) churn less than those without.
- Prioritize retention for **Senior + Month-to-month + Fiber** combinations.
"""
        )


def section_features(df: pd.DataFrame) -> None:
    st.header("Feature Engineering (Gold Layer)")
    st.caption("Notebook 05 · Business features created for modeling and segmentation.")

    gold_cols = [
        c
        for c in (
            "HasFamily",
            "TenureGroup",
            "MonthlyChargeLevel",
            "IsLongTermContract",
            "HasInternet",
            "TotalServicesSubscribed",
        )
        if c in df.columns
    ]
    if not gold_cols:
        st.warning("Gold features not present. Re-load the demo dataset or upload Telco-style data.")
        return

    st.markdown(
        """
| Feature | Meaning |
|---------|---------|
| **HasFamily** | Partner or Dependents = Yes |
| **TenureGroup** | New / Regular / Loyal / Very Loyal |
| **MonthlyChargeLevel** | Low (<35) / Medium (<70) / High |
| **IsLongTermContract** | One year or Two year |
| **HasInternet** | Internet service present |
| **TotalServicesSubscribed** | Count of Yes add-on services |
"""
    )

    picks = st.multiselect("Feature distributions", gold_cols, default=gold_cols[:4])
    for col in picks:
        dist = feature_distribution(df, col)
        if dist.empty:
            continue
        fig = px.bar(dist, x=col, y="count", title=f"Distribution — {col}")
        st.plotly_chart(fig, use_container_width=True)

    if "HasFamily" in df.columns and "Churn" in df.columns:
        st.subheader("Feature validation — HasFamily vs churn (Notebook 06)")
        rates = churn_by_segment(df, "HasFamily")
        fig = px.bar(rates, x="HasFamily", y="Churn_Rate", title="Churn % by HasFamily")
        st.plotly_chart(fig, use_container_width=True)


def section_predictions(df: pd.DataFrame) -> None:
    st.header("Risk Predictions")
    st.caption(
        "Notebooks 07–09 · Customer-level risk, recommended actions, and validation filters. "
        "Local demos use a transparent heuristic score when Databricks RF predictions are not loaded."
    )

    if "Risk_Level" not in df.columns:
        st.error("No Risk_Level column. Upload a prediction report or use the demo dataset.")
        return

    kpis = executive_kpis(df)
    _metric_row(
        [
            ("High", kpis["high_risk"]),
            ("Medium", kpis["medium_risk"]),
            ("Low", kpis["low_risk"]),
            ("Avg churn % score", kpis["avg_churn_percentage"]),
        ]
    )

    left, right = st.columns(2)
    with left:
        matrix = contract_risk_matrix(df)
        if not matrix.empty:
            fig = px.bar(
                matrix,
                x="Contract",
                y="count",
                color="Risk_Level",
                barmode="group",
                title="Contract × Risk Level",
                color_discrete_map={"High": "#C44B3C", "Medium": "#D4A017", "Low": "#2F6F4E"},
            )
            st.plotly_chart(fig, use_container_width=True)
    with right:
        profile = risk_profile_summary(df)
        if not profile.empty:
            fig = px.bar(
                profile,
                x="Risk_Level",
                y="Avg_MonthlyCharges" if "Avg_MonthlyCharges" in profile.columns else "Customers",
                title="Average monthly charges by risk",
                color="Risk_Level",
                color_discrete_map={"High": "#C44B3C", "Medium": "#D4A017", "Low": "#2F6F4E"},
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filter prediction report")
    risks = st.multiselect(
        "Risk levels",
        options=sorted(df["Risk_Level"].astype(str).unique()),
        default=["High"],
    )
    filtered = df[df["Risk_Level"].astype(str).isin(risks)].copy()
    if "Churn_Percentage" in filtered.columns:
        filtered = filtered.sort_values("Churn_Percentage", ascending=False)

    show_cols = [
        c
        for c in (
            "customerID",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "InternetService",
            "TechSupport",
            "Churn_Percentage",
            "Risk_Level",
            "Recommended_Action",
        )
        if c in filtered.columns
    ]
    st.dataframe(filtered[show_cols].head(200), use_container_width=True, hide_index=True)

    st.subheader("Top high-risk customers")
    st.dataframe(top_high_risk(df, limit=20), use_container_width=True, hide_index=True)

    if "Recommended_Action" in df.columns:
        action_mix = (
            df.groupby(["Risk_Level", "Recommended_Action"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        st.subheader("Risk → recommended action mapping")
        st.dataframe(action_mix, use_container_width=True, hide_index=True)


def section_model(df: pd.DataFrame) -> None:
    st.header("Machine Learning Model")
    st.caption("Notebook 07 · Random Forest performance and why features matter.")

    _metric_row(
        [
            ("Accuracy", f"{MODEL_METRICS['accuracy']}%"),
            ("F1 Score", f"{MODEL_METRICS['f1']}%"),
            ("ROC-AUC", f"{MODEL_METRICS['roc_auc']}%"),
        ]
    )

    st.markdown(
        """
**Pipeline stages (Databricks / Spark MLlib)**  
Gold features → StringIndexer → OneHotEncoder → VectorAssembler → Train/Test split → Random Forest → Evaluation → Risk banding  

**Risk bands used in production**  
- High: churn probability ≥ 80%  
- Medium: ≥ 60%  
- Low: otherwise  
"""
    )

    imp = pd.DataFrame(FEATURE_IMPORTANCE, columns=["Feature", "Importance"])
    fig = px.bar(
        imp.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top feature importance (from trained Random Forest)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Business validation checklist (Notebook 09)")
    checks = [
        "High-risk customers are mostly Month-to-month.",
        "High-risk customers show higher average monthly charges.",
        "High-risk customers show lower average tenure.",
        "Long-term contracts concentrate in Low risk.",
    ]
    for item in checks:
        st.markdown(f"- {item}")

    profile = risk_profile_summary(df)
    if not profile.empty:
        st.dataframe(profile, use_container_width=True, hide_index=True)


def section_ai(df: pd.DataFrame) -> None:
    st.header("AI Assistant")
    st.caption("GenAI layer · Ask questions in plain English, generate executive insights, draft retention emails.")

    from genai.insight_generator import generate_auto_insights
    from genai.query_engine import ask
    from genai.retention_copywriter import generate_retention_emails

    tabs = st.tabs(["Chat with data", "Executive AI summary", "Retention emails"])

    with tabs[0]:
        st.write(
            "Questions are translated into pandas code, executed in a sandbox, then answered in business language."
        )
        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []

        # chat_input only works once at page level in Streamlit; use text_input inside tab
        question = st.text_input(
            "Ask a question",
            placeholder="e.g. What is the churn rate by Contract?",
            key="ai_question",
        )
        if st.button("Ask", type="primary", key="ask_btn") and question.strip():
            with st.spinner("Thinking…"):
                try:
                    response = ask(question.strip(), df)
                    st.session_state.qa_history.append(response)
                except Exception as exc:
                    st.error(f"GenAI error: {exc}")

        for turn in reversed(st.session_state.qa_history[-8:]):
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn.get("answer") or turn.get("explanation") or "No answer.")
                if turn.get("error"):
                    st.warning(f"Execution issue: {turn['error']}")
                with st.expander("Generated code"):
                    st.code(turn.get("code", ""), language="python")
                preview = turn.get("result_preview")
                if isinstance(preview, list):
                    st.dataframe(pd.DataFrame(preview), use_container_width=True)
                elif preview is not None:
                    st.json(preview)
                figure = turn.get("figure")
                if figure is not None:
                    try:
                        st.plotly_chart(figure, use_container_width=True)
                    except Exception:
                        st.pyplot(figure)

    with tabs[1]:
        if st.button("Generate executive summary", key="insights_btn"):
            with st.spinner("Running statistical checks…"):
                insights = generate_auto_insights(df, narrate=True)
            for line in insights:
                st.markdown(f"- {line}")

    with tabs[2]:
        if "Risk_Level" not in df.columns:
            st.warning("Risk_Level required for retention drafts.")
        else:
            limit = st.slider("Number of high-risk customers", 1, 15, 5)
            if st.button("Draft retention emails", key="email_btn"):
                with st.spinner("Writing outreach…"):
                    drafts = generate_retention_emails(df, limit=limit, use_llm=True)
                for draft in drafts:
                    st.markdown(f"**{draft.get('customerID')}** — {draft.get('subject')}")
                    st.text(draft.get("body", ""))
                    st.divider()
