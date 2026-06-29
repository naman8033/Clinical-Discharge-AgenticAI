"""
Final Report page — shows the discharge summary, risk assessment,
discrepancies, and lets the clinician download JSON / HTML reports.
Only rendered for auto-approved (low-risk) cases or after HITL approval.
"""

import os
import json
import streamlit as st
from langgraph_workflows.graph import graph

REPORTS_DIR = "data/reports"


def _get_report_from_state(thread_id: str) -> dict:
    """Pull the final_report dict from the LangGraph state."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if snapshot is None:
        return {}
    return snapshot.values.get("final_report", {})


def _risk_badge(risk_level: str) -> str:
    colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    return colors.get(risk_level, "⚪")


def final_report_page():
    st.title("📋 Discharge Summary & Final Report")

    thread_id = st.session_state.get("active_thread_id", "patient_01")
    report = _get_report_from_state(thread_id)

    if not report:
        st.warning(
            "No report found for this session. "
            "Please complete the discharge pipeline first."
        )
        return

    patient_id = report.get("patient_id", "Unknown")
    risk_level  = report.get("risk_level", "unknown")
    risk_score  = report.get("risk_score", 0)
    recommendation = report.get("recommendation", "")
    generated_at   = report.get("generated_at", "")

    st.subheader(f"Patient: {patient_id}")
    st.caption(f"Report generated at: {generated_at}")
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", risk_score)
    col2.metric("Risk Level", f"{_risk_badge(risk_level)} {risk_level.upper()}")
    col3.metric("HITL Required", "Yes" if report.get("hitl_required") else "No")

    st.info(f"**Recommendation:** {recommendation}")

    guardrails = report.get("hard_guardrail_triggered", [])
    if guardrails:
        st.error(f"⚠ Hard Guardrails Triggered: {', '.join(guardrails)}")

    st.divider()

    with st.expander("🌐 Translation Confidence Scores", expanded=False):
        conf = report.get("translation_confidence", {})
        langs = conf.get("detected_languages", {})
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Doctor Report", f"{conf.get('doctor', 'N/A')}", langs.get('doctor', ''))
        tc2.metric("Lab Report",    f"{conf.get('lab', 'N/A')}",    langs.get('lab', ''))
        tc3.metric("Bill",          f"{conf.get('bill', 'N/A')}",   langs.get('bill', ''))

    discrepancies = report.get("discrepancies", [])
    with st.expander(
        f"⚠ EHR Discrepancies ({len(discrepancies)} found)",
        expanded=bool(discrepancies)
    ):
        if not discrepancies:
            st.success("✅ No discrepancies found.")
        else:
            for d in discrepancies:
                st.markdown(f"**{d.get('type','').replace('_',' ').title()}** (weight: {d.get('weight',0)})")
                st.write(d.get("description", ""))
                if d.get("evidence"):
                    st.caption(f"Evidence: {d['evidence']}")
                st.divider()

    missing = report.get("missing_fields", [])
    if missing:
        with st.expander(f"❌ Missing Fields ({len(missing)})", expanded=False):
            for f in missing:
                st.warning(f)

    st.subheader("📄 Patient Discharge Summary")
    summary = report.get("patient_summary", "")
    if summary:
        st.text_area(
            label="Discharge Summary (plain English)",
            value=summary,
            height=500,
            disabled=True,
            key="summary_display",
        )
    else:
        st.info("Summary not yet generated.")

    st.divider()
    st.subheader("⬇ Download Reports")

    dl_col1, dl_col2, dl_col3 = st.columns(3)

    json_path = report.get("json_report_path", "")
    if json_path and os.path.exists(json_path):
        with open(json_path, "rb") as jf:
            dl_col1.download_button(
                label="📥 Download JSON Report",
                data=jf.read(),
                file_name=os.path.basename(json_path),
                mime="application/json",
            )
    else:
        dl_col1.download_button(
            label="📥 Download JSON Report",
            data=json.dumps(report, indent=2).encode(),
            file_name=f"{patient_id}_report.json",
            mime="application/json",
        )

    html_path = report.get("html_report_path", "")
    if html_path and os.path.exists(html_path):
        with open(html_path, "rb") as hf:
            dl_col2.download_button(
                label="🌐 Download HTML Report",
                data=hf.read(),
                file_name=os.path.basename(html_path),
                mime="text/html",
            )

    if summary:
        dl_col3.download_button(
            label="📝 Download Summary (TXT)",
            data=summary.encode(),
            file_name=f"{patient_id}_discharge_summary.txt",
            mime="text/plain",
        )

    with st.expander("🗂 Raw Patient Record (JSON)", expanded=False):
        st.json(report.get("patient_record", {}))
