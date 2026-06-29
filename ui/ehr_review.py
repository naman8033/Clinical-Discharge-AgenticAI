"""
EHR Review page — HITL for medium/high risk cases after EHR validation.
Shows discrepancies, risk score, guardrails. Clinician can approve, edit, or reject.
"""

import streamlit as st
from langgraph.types import Command
from langgraph_workflows.graph import graph


def _risk_color(risk_level: str) -> str:
    return {"low": "green", "medium": "orange", "high": "red"}.get(risk_level, "gray")


def ehr_review_page():
    st.title("🔍 EHR Validation Review")
    st.caption("Clinician review of EHR cross-validation findings before discharge approval.")

    thread_id = st.session_state.get("active_thread_id", "patient_01")
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = graph.get_state(config)
    if snapshot is None:
        st.error("No active workflow found.")
        st.stop()

    state = snapshot.values
    final_report   = state.get("final_report", {})
    patient_record = state.get("patient_record", {})
    ehr_parsed     = state.get("ehr_parsed", {})

    if not final_report:
        st.info("EHR validation has not completed yet, or no discrepancies require review.")
        return

    patient_id   = final_report.get("patient_id", "Unknown")
    risk_level   = final_report.get("risk_level", "unknown")
    risk_score   = final_report.get("risk_score", 0)
    recommendation = final_report.get("recommendation", "")
    guardrails   = final_report.get("hard_guardrail_triggered", [])
    discrepancies = final_report.get("discrepancies", [])

    st.subheader(f"Patient: {patient_id}")
    st.info(f"Thread: `{thread_id}`")
    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Score",    risk_score)
    m2.metric("Risk Level",    risk_level.upper())
    m3.metric("Discrepancies", len(discrepancies))

    st.markdown(f"**Recommendation:** _{recommendation}_")

    if guardrails:
        st.error(f"⚠ Hard Guardrails: {', '.join(guardrails)}")

    st.divider()

    st.subheader("📋 Discrepancy Details")
    if not discrepancies:
        st.success("✅ No discrepancies found.")
    else:
        for d in discrepancies:
            with st.expander(
                f"**{d.get('type','').replace('_',' ').title()}** — weight {d.get('weight',0)}",
                expanded=True
            ):
                st.write(d.get("description", ""))
                if d.get("evidence"):
                    st.info(f"Evidence: {d['evidence']}")

    st.divider()

    st.subheader("📄 Patient Record (read-only)")
    with st.expander("View full patient record"):
        st.json(patient_record)

    notes = st.text_area(
        "📝 Clinician Notes / Override Reason",
        placeholder="Enter your clinical justification for approval or rejection...",
        key="ehr_notes",
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    def _resume_and_navigate(payload: dict):
        with st.spinner("Processing — please wait..."):
            graph.invoke(Command(resume=payload), config=config)
        snap = graph.get_state(config)
        if snap and snap.next:
            st.session_state.page = "Document Upload"
        elif snap and snap.values.get("final_report"):
            st.session_state.page = "Final Report"
        else:
            st.session_state.page = "Document Upload"
        st.rerun()

    with col1:
        if st.button("✅ Approve (Override)", use_container_width=True):
            _resume_and_navigate({"action": "approve", "notes": notes})

    with col2:
        if st.button("❌ Reject", use_container_width=True):
            st.session_state.page = "Document Upload"
            graph.invoke(Command(resume={"action": "reject", "notes": notes}), config=config)
            st.rerun()

    with col3:
        if st.button("🔁 Acknowledge & Continue", use_container_width=True):
            _resume_and_navigate({
                "action": "edit",
                "notes": notes,
                "patient_record": patient_record,
            })
