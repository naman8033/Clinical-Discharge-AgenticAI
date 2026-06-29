"""
Main Streamlit application — Agentic AI Hospital Discharge System.
Pages:
  1. Document Upload    — upload + trigger pipeline
  2. Validation         — completeness HITL (missing fields)
  3. EHR Review         — EHR validation HITL (discrepancies / risk)
  4. QA Chatbot         — RAG-based clinical Q&A
  5. Final Report       — discharge summary, risk, downloads
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from upload_documents import document_upload_page
from validation import validation_page
from ehr_review import ehr_review_page
from qa_assistant import qa_chatbot_page
from final_report import final_report_page
from langgraph_workflows.graph import graph

st.set_page_config(
    page_title="Hospital Agentic AI",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
html, body, [class*="css"], .main, .stApp {
    background-color: #0e1117 !important;
    color: #ffffff !important;
}
.main .block-container {
    background-color: #0e1117 !important;
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    background: #060d18 !important;
    border-right: 1px solid #1e2d45 !important;
}
[data-testid="stSidebar"] * { color: #c9d8ec !important; }
[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 1.35rem !important;
    letter-spacing: 0.03em;
    border-bottom: 2px solid #00d4aa;
    padding-bottom: 0.4rem;
}
[data-testid="stSidebar"] hr { border-color: #1e2d45 !important; }
[data-testid="stSidebar"] .stAlert { background: rgba(255,255,255,0.06) !important; }

h1 {
    color: #ffffff !important;
    font-weight: 800 !important;
    border-bottom: 3px solid #00d4aa !important;
    padding-bottom: 0.35rem;
    margin-bottom: 0.7rem !important;
}
h2 { color: #e2e8f0 !important; font-weight: 700 !important; }
h3 { color: #cbd5e1 !important; font-weight: 600 !important; }
p, li, span, label { color: #e2e8f0 !important; }

[data-testid="stMetric"] {
    background: #151d2e !important;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    border-left: 4px solid #00d4aa;
    transition: box-shadow 0.2s, transform 0.2s;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 24px rgba(0,212,170,0.15);
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] {
    color: #7fa8cc !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

.stButton > button,
[data-testid="stDownloadButton"] > button,
.stDownloadButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.52rem 1.2rem !important;
    transition: all 0.18s ease !important;
    border: 1px solid #00d4aa !important;
    background: linear-gradient(135deg, #0a2540 0%, #0d3460 100%) !important;
    color: #ffffff !important;
    letter-spacing: 0.01em;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,212,170,0.25) !important;
    background: linear-gradient(135deg, #0d3460 0%, #00d4aa22 100%) !important;
    border-color: #00d4aa !important;
}
.stButton > button:active,
[data-testid="stDownloadButton"] > button:active,
.stDownloadButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
}

[data-testid="stExpander"],
[data-testid="stExpander"] > div,
[data-testid="stExpander"] details,
[data-testid="stExpander"] details > div {
    background: #151d2e !important;
    border-radius: 10px !important;
    border: 1px solid #1e2d45 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.3) !important;
    margin-bottom: 0.7rem;
}
[data-testid="stExpander"] .streamlit-expanderContent,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #151d2e !important;
    color: #e2e8f0 !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] summary p {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    background: #151d2e !important;
}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:hover * {
    background: #1a2540 !important;
    color: #ffffff !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
.stTextInput input,
.stTextArea textarea {
    background: #151d2e !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    border: 1.5px solid #1e3a5f !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
}
[data-testid="stTextArea"] textarea:disabled,
.stTextArea textarea:disabled,
textarea[disabled] {
    background: #111827 !important;
    color: #d1e8ff !important;
    border-color: #1e2d45 !important;
    opacity: 1 !important;          /* browsers dim disabled by default */
    -webkit-text-fill-color: #d1e8ff !important; /* overrides Chrome's dimming */
    cursor: default !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 3px rgba(0,212,170,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: #4a6080 !important;
}

[data-testid="stFileUploader"] {
    background: #151d2e !important;
    border-radius: 10px !important;
    border: 2px dashed #1e3a5f !important;
    padding: 0.5rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00d4aa !important;
}

[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label { color: #c9d8ec !important; }
[data-testid="stRadio"] div[role="radiogroup"] label {
    color: #e2e8f0 !important;
    padding: 0.3rem 0;
}

[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-width: 1px !important;
}
.stSuccess  { background: #0d2b1e !important; border-color: #00d4aa !important; color: #a7f3d0 !important; }
.stWarning  { background: #2b1e08 !important; border-color: #f59e0b !important; color: #fde68a !important; }
.stError    { background: #2b0e0e !important; border-color: #ef4444 !important; color: #fca5a5 !important; }
.stInfo     { background: #0d1e2b !important; border-color: #3b82f6 !important; color: #93c5fd !important; }

[data-testid="stJson"],
[data-testid="stJson"] > div,
[data-testid="stJson"] ul,
[data-testid="stJson"] li,
[data-testid="stJson"] span,
[data-testid="stJson"] * {
    background: #111827 !important;
    color: #a5f3c9 !important;
    border-radius: 8px !important;
}
/* key names */
[data-testid="stJson"] .key,
[data-testid="stJson"] [class*="key"] { color: #7dd3fc !important; }
/* string values */
[data-testid="stJson"] [class*="string"] { color: #86efac !important; }
/* number values */
[data-testid="stJson"] [class*="number"] { color: #fdba74 !important; }
/* null / bool */
[data-testid="stJson"] [class*="null"],
[data-testid="stJson"] [class*="bool"]  { color: #c084fc !important; }
/* collapse / expand arrows */
[data-testid="stJson"] svg,
[data-testid="stJson"] button {
    background: transparent !important;
    color: #7dd3fc !important;
}
[data-testid="stJson"] button:hover { background: #1e2d45 !important; }

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #151d2e !important;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1e2d45;
}

[data-testid="stSpinner"] * { color: #00d4aa !important; }

.stCaption, [data-testid="stCaptionContainer"] {
    color: #6b88a8 !important;
    font-size: 0.82rem !important;
}

hr { border-color: #1e2d45 !important; }

code {
    background: #151d2e !important;
    color: #00d4aa !important;
    border-radius: 4px;
    padding: 0.1rem 0.3rem;
    font-size: 0.85em;
}

[data-testid="stToast"] {
    background: #151d2e !important;
    color: #ffffff !important;
    border: 1px solid #00d4aa !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

st_autorefresh(interval=2000, key="poll")

st.sidebar.title("🏥 Agentic AI")

PAGES = [
    "Document Upload",
    "Validation",
    "EHR Review",
    "QA Chatbot",
    "Final Report",
]

if "page" not in st.session_state:
    st.session_state.page = "Document Upload"

page = st.sidebar.radio(
    "Navigate",
    PAGES,
    index=PAGES.index(st.session_state.page),
)
st.session_state.page = page

thread_id = st.session_state.get("active_thread_id", "patient_01")
config    = {"configurable": {"thread_id": thread_id}}
snapshot  = graph.get_state(config)

if snapshot and snapshot.next:
    next_node = snapshot.next[0]
    if next_node == "human_review" and page not in ("Validation", "Document Upload"):
        st.session_state.page = "Validation"
        page = "Validation"
    elif next_node == "human_ehr_review" and page not in ("EHR Review", "Document Upload"):
        st.session_state.page = "EHR Review"
        page = "EHR Review"

elif snapshot and snapshot.values.get("final_report") and page == "Document Upload":
    # Pipeline just completed — auto-navigate away from upload page
    st.session_state.page = "Final Report"
    page = "Final Report"

st.sidebar.divider()
st.sidebar.caption(f"Active Thread: `{thread_id}`")

if snapshot and snapshot.next:
    waiting = snapshot.next[0]
    label   = {"human_review": "Missing Fields", "human_ehr_review": "EHR Review"}.get(waiting, waiting)
    st.sidebar.warning(f"⏳ Paused: **{label}**")

elif snapshot and snapshot.values.get("final_report"):
    risk  = snapshot.values["final_report"].get("risk_level", "")
    score = snapshot.values["final_report"].get("risk_score", 0)
    icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    st.sidebar.success(
        f"{icons.get(risk,'⚪')} Complete\n\n"
        f"Risk: **{risk.upper()}** (score {score})"
    )

elif snapshot and snapshot.values:
    st.sidebar.info("🔄 Pipeline running...")

else:
    st.sidebar.caption("No active pipeline.")

if page == "Document Upload":
    document_upload_page()

elif page == "Validation":
    validation_page()

elif page == "EHR Review":
    ehr_review_page()

elif page == "QA Chatbot":
    qa_chatbot_page()

elif page == "Final Report":
    final_report_page()
