"""
Document Upload page.

Flow:
  UI saves files → Data/incoming/<folder>/ → Monitor Agent detects all 3 files
  → Extractor Agent extracts text → graph.invoke() starts the pipeline

The UI ONLY saves files and polls graph state.
It never calls extract_file() or graph.invoke() directly.
"""

import os
import re
import streamlit as st
from langgraph_workflows.graph import graph

PIPELINE_STAGES = [
    ("translation",      "🌐 Translation & Normalization"),
    ("completeness",     "✅ Completeness Check"),
    ("human_review",     "👤 Human Review — Missing Fields"),
    ("ehr_validation",   "🏥 EHR Validation"),
    ("reporting",        "📋 Discharge Report Generation"),
    ("human_ehr_review", "👨‍⚕️ Clinician EHR Review"),
    ("rag_indexing",     "🔍 RAG Indexing for Q&A"),
]


def _save_file(uploaded_file, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def _extract_patient_id(filename: str) -> str:
    """Parse patient ID from filename: P1019_thomas_wright.pdf → P1019."""
    match = re.match(r"^([A-Za-z0-9]+)_", filename)
    return match.group(1).upper() if match else "PATIENT_UNKNOWN"


def _get_completed_nodes(thread_id: str) -> list:
    """
    Walk the LangGraph checkpoint history to find which nodes have finished.
    Each checkpoint's metadata.writes contains {node_name: state_updates}.
    """
    config = {"configurable": {"thread_id": thread_id}}
    completed = []
    try:
        for snapshot in graph.get_state_history(config):
            writes = (snapshot.metadata or {}).get("writes") or {}
            for node_name in writes:
                if node_name not in ("__start__", "__input__") and node_name not in completed:
                    completed.append(node_name)
    except Exception:
        pass
    return completed


def _show_pipeline_progress(thread_id: str):
    """Render a live step-by-step pipeline progress indicator."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot       = graph.get_state(config)
        completed_nodes = _get_completed_nodes(thread_id)
    except Exception:
        st.info("⏳ Waiting for Monitor Agent to start the pipeline...")
        return

    if not snapshot or not snapshot.values:
        st.info("⏳ Waiting for Monitor Agent to detect files and start the pipeline...")
        st.caption("Make sure `python main.py` is running in a separate terminal.")
        return

    waiting_node = snapshot.next[0] if snapshot.next else None
    is_complete  = (not snapshot.next) and bool(snapshot.values.get("final_report"))

    st.subheader("📊 Pipeline Progress")

    stage_keys = [k for k, _ in PIPELINE_STAGES]
    last_done_idx = max(
        (stage_keys.index(n) for n in completed_nodes if n in stage_keys),
        default=-1
    )

    for idx, (node_key, label) in enumerate(PIPELINE_STAGES):

        if node_key in completed_nodes:
            st.success(f"✅ {label}")

        elif node_key == waiting_node:
            st.warning(f"⏸ {label}  ← **Action required**")
            if node_key == "human_review":
                st.session_state.page = "Validation"
            elif node_key == "human_ehr_review":
                st.session_state.page = "EHR Review"

        elif (not is_complete
              and not waiting_node
              and idx == last_done_idx + 1):
            st.info(f"🔄 {label}  ← Running...")

        elif is_complete and node_key not in completed_nodes:
            st.markdown(f"➖ {label}  *(skipped)*")

        else:
            st.markdown(f"⬜ {label}")

    if is_complete:
        final_report = snapshot.values.get("final_report", {})
        risk  = final_report.get("risk_level", "unknown")
        score = final_report.get("risk_score", 0)
        icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        st.divider()
        st.success(
            f"{icons.get(risk, '⚪')} **Pipeline complete** — "
            f"Risk: **{risk.upper()}** (score {score})"
        )
        st.session_state.page = "Final Report"
    elif waiting_node:
        st.divider()
        action_page = {
            "human_review":     "Validation",
            "human_ehr_review": "EHR Review",
        }.get(waiting_node, "")
        if action_page:
            st.warning(
                f"⚠ Pipeline is paused and needs your input. "
                f"Go to the **{action_page}** page."
            )


def document_upload_page():
    st.title("📄 Clinical Document Upload")
    st.write(
        "Upload all three patient documents. "
        "The **Monitor Agent** (`python main.py`) will detect them and start the pipeline automatically."
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # These are the exact folders the monitor agent watches
    bills_folder  = os.path.join(project_root, "Data", "incoming", "bills")
    doctor_folder = os.path.join(project_root, "Data", "incoming", "doctor_reports")
    lab_folder    = os.path.join(project_root, "Data", "incoming", "lab_reports")

    col1, col2 = st.columns(2)

    with col1:
        bill_file = st.file_uploader(
            "💰 Upload Bill",
            type=["pdf", "png", "jpg", "json", "txt"],
            key="bill",
        )
        doctor_file = st.file_uploader(
            "👨‍⚕ Upload Doctor Report",
            type=["pdf", "png", "jpg", "docx", "txt", "json"],
            key="doctor",
        )

    with col2:
        lab_file = st.file_uploader(
            "🧪 Upload Lab Report",
            type=["pdf", "png", "jpg", "txt", "json"],
            key="lab",
        )

    st.divider()

    if st.button("📁 Save Documents", use_container_width=True):
        if not bill_file:
            st.error("Please upload a Bill.")
            st.stop()
        if not doctor_file:
            st.error("Please upload a Doctor Report.")
            st.stop()
        if not lab_file:
            st.error("Please upload a Lab Report.")
            st.stop()

        # Derive patient_id from the doctor report filename
        patient_id = _extract_patient_id(doctor_file.name)
        st.session_state["active_thread_id"] = patient_id

        with st.spinner("Saving documents to incoming folders..."):
            _save_file(bill_file,   bills_folder)
            _save_file(doctor_file, doctor_folder)
            _save_file(lab_file,    lab_folder)

        st.success(
            f"✅ All three files saved for patient **{patient_id}**.\n\n"
            "The Monitor Agent will detect them and start extraction automatically."
        )

    # ── Live pipeline progress ────────────────────────────────────────────────
    thread_id = st.session_state.get("active_thread_id")
    if thread_id:
        st.divider()
        st.info(f"Active Patient Thread: `{thread_id}`")
        _show_pipeline_progress(thread_id)
