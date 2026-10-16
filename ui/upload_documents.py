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


def _get_completed_nodes(thread_id: str, snapshot=None) -> list:
    """
    Determine completed nodes from the graph state values and execution status.
    """
    if snapshot is None:
        try:
            snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        except Exception:
            return []

    if not snapshot or not snapshot.values:
        return []

    values = snapshot.values
    completed = []

    # 1. Translation: completed if translated_data has text
    translated = values.get("translated_data")
    if translated and isinstance(translated, dict) and any(bool(v) for v in translated.values()):
        completed.append("translation")

    # 2. Completeness: completed if patient_record exists
    if values.get("patient_record") is not None:
        completed.append("completeness")

    # 3. Human Review: completed if status is approved or subsequent stages have completed
    status = values.get("status")
    ehr_done = values.get("ehr_validation_done")
    final_report = values.get("final_report")
    if status == "approved" or ehr_done or final_report:
        completed.append("human_review")

    # 4. EHR Validation: completed if ehr_validation_done or final_report exists
    if ehr_done or final_report:
        completed.append("ehr_validation")

    # 5. Reporting: completed if final_report dictionary is present
    if final_report and isinstance(final_report, dict) and final_report.get("patient_id"):
        completed.append("reporting")

    # 6. Clinician EHR Review: completed if clinician override or approval in final_report
    if final_report and (final_report.get("clinician_override") or final_report.get("hitl_action") == "approved"):
        completed.append("human_ehr_review")

    # 7. RAG Indexing: completed if report is generated and workflow has finished
    if bool(final_report) and (not snapshot.next or "rag_indexing" not in snapshot.next):
        completed.append("rag_indexing")

    return completed


def _get_task_errors(snapshot) -> list:
    """Extract any runtime task errors from the LangGraph snapshot."""
    errors = []
    if snapshot and snapshot.tasks:
        for t in snapshot.tasks:
            err = getattr(t, "error", None)
            if err:
                errors.append((getattr(t, "name", "node"), str(err)))
    return errors


def _show_pipeline_progress(thread_id: str):
    """Render a live step-by-step pipeline progress indicator."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot        = graph.get_state(config)
        completed_nodes = _get_completed_nodes(thread_id, snapshot)
        task_errors     = _get_task_errors(snapshot)
    except Exception:
        st.info("⏳ Waiting for Monitor Agent to start the pipeline...")
        return

    if not snapshot or not snapshot.values:
        st.info("⏳ Waiting for Monitor Agent to detect files and start the pipeline...")
        st.caption("Make sure `python main.py` is running in a separate terminal.")
        return

    # Check for execution errors first
    if task_errors:
        for task_name, err_msg in task_errors:
            stage_label = dict(PIPELINE_STAGES).get(task_name, task_name)
            st.error(f"❌ **Pipeline Error in {stage_label}:**\n\n`{err_msg}`")
            if "AuthenticationError" in err_msg or "invalid_api_key" in err_msg:
                st.warning("🔑 **Fix:** Check your API key in `.env` or OmniRouter/LiteLLM Gateway configuration.")
            elif "Connection" in err_msg or "refused" in err_msg or "500" in err_msg or "503" in err_msg or "unreachable" in err_msg.lower():
                st.warning("🔌 **Fix:** Check that your local LLM gateway (`python main.py` or OmniRouter/Ollama) is running and reachable.")

    waiting_node = snapshot.next[0] if snapshot.next else None
    is_complete  = (not snapshot.next) and bool(snapshot.values.get("final_report"))

    # Auto-navigate if the pipeline just reached human review while user was on upload page
    if waiting_node == "human_review" and st.session_state.get("pipeline_started"):
        st.session_state["pipeline_started"] = False
        st.session_state.page = "Validation"
        st.rerun()

    st.subheader("📊 Pipeline Progress")

    stage_keys = [k for k, _ in PIPELINE_STAGES]
    last_done_idx = max(
        (stage_keys.index(n) for n in completed_nodes if n in stage_keys),
        default=-1
    )

    for idx, (node_key, label) in enumerate(PIPELINE_STAGES):
        # If this node has a task error
        node_has_error = any(t_name == node_key for t_name, _ in task_errors)

        if node_has_error:
            st.error(f"❌ {label}  *(failed)*")

        elif node_key in completed_nodes:
            st.success(f"✅ {label}")

        elif node_key == waiting_node:
            st.warning(f"⏸ {label}  ← **Action required**")

        elif (not is_complete
              and not waiting_node
              and not task_errors
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
        col_f1, _ = st.columns([1, 2])
        with col_f1:
            if st.button("📋 View Final Report ➔", type="primary", use_container_width=True):
                st.session_state.page = "Final Report"
                st.rerun()

    elif waiting_node and not task_errors:
        st.divider()
        action_map = {
            "human_review":     ("Validation", "🏥 Open Human Review Dashboard"),
            "human_ehr_review": ("EHR Review", "👨‍⚕️ Open Clinician EHR Review"),
        }
        if waiting_node in action_map:
            page_name, btn_label = action_map[waiting_node]
            st.warning(
                f"⚠️ **Action Required:** Pipeline is paused at **{page_name}**."
            )
            col_act, _ = st.columns([1, 2])
            with col_act:
                if st.button(f"{btn_label} ➔", type="primary", use_container_width=True):
                    st.session_state.page = page_name
                    st.rerun()


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
        st.session_state["pipeline_started"] = True

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
