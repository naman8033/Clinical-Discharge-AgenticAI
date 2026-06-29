import streamlit as st
from langgraph.types import Command

from langgraph_workflows.graph import graph
from langgraph_workflows.state import GraphState
import json

def normalize(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return value

    if isinstance(value, list):
        return [normalize(v) for v in value]

    return value

def validation_page():

    THREAD_ID = st.session_state.get("active_thread_id", "patient_01")

    config = {
        "configurable": {
            "thread_id": THREAD_ID
        }
    }

    snapshot = graph.get_state(config)

    if snapshot is None:
        st.error("No workflow found.")
        st.stop()

    state = snapshot.values
    print("stream lit state: ", state)
    patient_record = state.get("patient_record", {})
    missing_fields = state.get("missing_fields", [])

    st.title("🏥 Human Review Dashboard")
    st.caption("Review extracted patient information before continuing the workflow.")

    st.divider()

    m1, m2, m3 = st.columns(3)

    m1.metric("Total Fields", len(patient_record))
    m2.metric("Missing Fields", len(missing_fields))
    m3.metric("Workflow Status", "Pending Review")

    st.divider()

    left, right = st.columns([3, 1])

    edited_record = {}


    with left:
        st.subheader("📋 Patient Record")

        with st.expander("📄 Extracted Information", expanded=True):

            for key, value in patient_record.items():

                if isinstance(value, list):
                    continue

                is_missing = key in missing_fields

                edited_record[key] = st.text_input(
                    label=f"🔴 {key}" if is_missing else key,
                    value="" if value is None else str(value),
                    key=f"field_{key}"
                )

        extra_missing = [
            field
            for field in missing_fields
            if field not in patient_record
        ]

        if extra_missing:

            with st.expander("⚠ Missing Fields", expanded=True):

                st.warning(
                    "The following fields could not be extracted."
                )

                for field in extra_missing:

                    edited_record[field] = st.text_input(
                        label=f"🔴 {field}",
                        value="",
                        key=f"missing_{field}"
                    )

        for key, value in patient_record.items():

            if not isinstance(value, list):
                continue

            with st.expander(
                f"📦 {key.replace('_',' ').title()}",
                expanded=True
            ):

                edited_record[key] = []

                if len(value) == 0:
                    st.info("No records available.")
                    continue

                if all(isinstance(item, str) for item in value):

                    for i, item in enumerate(value):

                        edited_record[key].append(
                            st.text_input(
                                label=f"{key.replace('_',' ').title()} {i+1}",
                                value=item,
                                key=f"{key}_{i}"
                            )
                        )

                elif all(isinstance(item, dict) for item in value):

                    for i, row in enumerate(value):

                        st.markdown(f"#### Record {i+1}")

                        new_row = {}

                        cols = st.columns(3)

                        for j, (k, v) in enumerate(row.items()):

                            with cols[j % 3]:

                                new_row[k] = st.text_input(
                                    label=k.replace("_", " ").title(),
                                    value="" if v is None else str(v),
                                    key=f"{key}_{i}_{k}"
                                )

                        edited_record[key].append(new_row)

                else:
                    st.warning(
                        f"Unsupported data format for '{key}'"
                    )
                    st.write(value)            
                    
    with right:
        st.subheader("✅ Validation")
        if missing_fields:
            st.error(f"{len(missing_fields)} Missing Fields")
            for field in missing_fields:
                st.warning(field)
        else:
            st.success("No Missing Fields")
        st.divider()
        st.subheader("Workflow")
        st.info(f"Thread ID\n\n{THREAD_ID}")

    st.divider()

    notes = st.text_area(
        "📝 Reviewer Notes",
        placeholder="Enter any reviewer comments..."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    def _resume_and_navigate(payload: dict):
        """Invoke graph with HITL response, check where it ended up, and navigate."""
        with st.spinner("Processing — please wait..."):
            graph.invoke(Command(resume=payload), config=config)
        snap = graph.get_state(config)
        if snap and snap.next:
            next_node = snap.next[0]
            if next_node == "human_ehr_review":
                st.session_state.page = "EHR Review"
            else:
                st.session_state.page = "Document Upload"
        elif snap and snap.values.get("final_report"):
            st.session_state.page = "Final Report"
        else:
            st.session_state.page = "Document Upload"
        st.rerun()

    with col1:
        if st.button("✅ Approve", use_container_width=True):
            _resume_and_navigate({
                "action": "approve",
                "patient_record": edited_record,
                "notes": notes,
            })

    with col2:
        if st.button("❌ Reject", use_container_width=True):
            _resume_and_navigate({"action": "reject", "notes": notes})

    with col3:
        if st.button("💾 Save & Continue", use_container_width=True):
            _resume_and_navigate({
                "action": "edit",
                "patient_record": edited_record,
                "notes": notes,
            })