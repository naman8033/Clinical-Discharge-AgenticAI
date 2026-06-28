import streamlit as st

from upload_documents import document_upload_page
from validation import validation_page
from langgraph_workflows.graph import graph
# from qa_chatbot import qa_chatbot_page
# from final_report import final_report_page
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=5000, key="poll")

st.set_page_config(
    page_title="Hospital Agentic AI",
    page_icon="🏥",
    layout="wide"
)

st.sidebar.title("🏥 Agentic AI")
if "page" not in st.session_state:
    st.session_state.page = "Document Upload"

page = st.sidebar.radio(
    "Select Module",
    [
        "Document Upload",
        "Validation",
        "QA Chatbot",
        "Final Report"
    ],
    index=[
        "Document Upload",
        "Validation",
        "QA Chatbot",
        "Final Report"
    ].index(st.session_state.page)
)

st.session_state.page = page
config = {
    "configurable":{
        "thread_id":"patient_01"
    }
}
snapshot = graph.get_state(config)
print("snap shot of ui : ",snapshot,"page ",page)
if snapshot.next == ("human_review",):
    # Show validation page
    st.session_state.page = "Validation"
    page = "Validation"
if page == "Document Upload":
    document_upload_page()

elif page == "Validation":
    print("validation calling")
    validation_page()

# elif page == "QA Chatbot":
#     # qa_chatbot_page()

# elif page == "Final Report":
    # final_report_page()