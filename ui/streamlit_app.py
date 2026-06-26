import streamlit as st
from langgraph.types import Command

from langgraph_workflows.graph import graph
from langgraph_workflows.state import GraphState

# PAGE CONFIG
st.set_page_config(
    page_title="Patient Review Dashboard",
    page_icon="🏥",
    layout="wide"
)
# THREAD ID
THREAD_ID = "patient_001"

config = {
    "configurable": {
        "thread_id": THREAD_ID
    }
}

# LOAD GRAPH STATE
snapshot = graph.get_state(config)

if snapshot is None:
    st.error("No workflow found.")
    st.stop()

state = snapshot.values

patient_record = state.get("patient_record", {"name":"sajid","age":23,"city":"hyderabad"})
missing_fields = state.get("missing_fields", ["medicine","patient_id"])

# HEADER
st.title("🏥 Human Review Dashboard")
st.caption("Review extracted patient information before continuing the workflow.")

st.divider()

m1, m2, m3 = st.columns(3)

m1.metric("Total Fields", len(patient_record))
m2.metric("Missing Fields", len(missing_fields))
m3.metric("Workflow Status", "Pending Review")

st.divider()

# EDIT TOGGLE
# enable_edit = st.toggle(
#     "✏️ Enable Editing",
#     value=False,
#     help="Turn on editing mode to modify extracted data."
# )

# MAIN LAYOUT
left, right = st.columns([3, 1])

edited_record = {}

####################################################
# PATIENT RECORD
####################################################

with left:
    st.subheader("📋 Patient Record")
    # Dictionary to store edited values
    edited_record = {}
    with st.expander("📄 Extracted Information", expanded=True):
        for key, value in patient_record.items():
            # Skip list fields here
            if isinstance(value, list):
                continue
            is_missing = key in missing_fields
            edited_record[key] = st.text_input(
                label=f"🔴 {key}" if is_missing else key,
                value="" if value is None else str(value),
                key=f"field_{key}"
            )
    extra_missing = [
        field for field in missing_fields
        if field not in patient_record
    ]
    if extra_missing:
        with st.expander("⚠ Missing Fields", expanded=True):
            st.warning(
                "The following fields could not be extracted. "
                "You may fill them manually if the information is available."
            )
            for field in extra_missing:
                edited_record[field] = st.text_input(
                    label=f"🔴 {field}",
                    value="",
                    placeholder=f"Enter {field}",
                    key=f"missing_{field}"
                )
    for key, value in patient_record.items():
        if not isinstance(value, list):
            continue
        with st.expander(f"📦 {key.replace('_',' ').title()}", expanded=True):
            edited_record[key] = []
            if len(value) == 0:
                st.info("No records available.")
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

# VALIDATION PANEL
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
    # if enable_edit:
    #     st.success("Editing Enabled")
    # else:
    #     st.info("Read Only Mode")

# REVIEWER NOTES
st.divider()

notes = st.text_area(
    "📝 Reviewer Notes",
    placeholder="Enter any reviewer comments..."
)

# ACTION BUTTONS
st.divider()

col1, col2, col3 = st.columns(3)

# APPROVE
with col1:
    if st.button(
        "✅ Approve",
        use_container_width=True
    ):
        graph.invoke(
            Command(
                resume={
                    "action": "approve",
                    "patient_record": edited_record,
                    "notes": notes
                }
            ),
            config=config
        )

        st.success("Workflow Continued Successfully.")
        st.balloons()

# REJECT
with col2:
    if st.button(
        "❌ Reject",
        use_container_width=True
    ):

        graph.invoke(
            Command(
                resume={
                    "action": "reject",
                    "notes": notes
                }
            ),
            config=config
        )

        st.error("Workflow Rejected.")

# SAVE & CONTINUE
with col3:
    if st.button(
        "💾 Save & Continue",
        use_container_width=True,
        # disabled=not enable_edit
    ):
        graph.invoke(
            Command(
                resume={
                    "action": "edit",
                    "patient_record": edited_record,
                    "notes": notes
                }
            ),
            config=config
        )
        st.success("Changes Saved Successfully.")
        st.toast("Patient record updated.")