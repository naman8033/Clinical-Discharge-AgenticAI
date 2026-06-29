import sqlite3
from langgraph.graph import StateGraph, END

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _USE_SQLITE = True
except (ImportError, ModuleNotFoundError):
    from langgraph.checkpoint.memory import MemorySaver
    _USE_SQLITE = False

from langgraph_workflows.state import GraphState
from agents.translation_normalization_agent import translation_agent
from agents.completenss_agent import completeness_agent
from agents.human_review import human_review
from agents.ehr_validation_agent import ehr_validation_agent
from agents.human_ehr_review import human_ehr_review
from agents.reporting_agent import reporting_agent
from agents.rag_agents.indexing_agent import IndexingAgent

_indexing_agent = IndexingAgent()

def rag_indexing_node(state: GraphState) -> GraphState:
    """Index the final discharge report into FAISS for Q&A."""
    final_report = state.get("final_report", {})
    summary = final_report.get("patient_summary", "")
    patient_id = state.get("patient_id", "unknown")

    if summary:
        print(f"[RAG Indexing] Indexing report for {patient_id}...")
        _indexing_agent.run(summary)
    else:
        print("[RAG Indexing] No summary to index, skipping.")

    return state

def route_after_completeness(state: GraphState) -> str:
    print("routing function after completeness agent")
    missing = state.get("missing_fields", [])
    if len(missing) > 0:
        return "human_review"
    return "ehr_validation"


def route_after_validation(state:GraphState)-> str:
    print("route function after human review validation")
    if state['status'] =='rejected':
        return END
    return "ehr_validation"

def route_after_ehr(state:GraphState)-> str:
    print("route funciton after ehr human validation")
    if state['final_report']['hitl_required'] == True:
        return "human_ehr_review"
    return "reporting"
def route_after_ehr_review(state:GraphState)->str:
    print("human ehr routing funciton")
    if state['status'] == 'rejected':
        return END
    return "reporting"

def route_after_reporting(state: GraphState) -> str:
    """If risk is medium or high AND not already clinician-approved → EHR HITL."""
    report = state.get("final_report", {})
    risk_level = report.get("risk_level", "low")
    hitl_required = report.get("hitl_required", False)
    clinician_override = report.get("clinician_override", False)

    if hitl_required and not clinician_override and risk_level in ("medium", "high"):
        return "human_ehr_review"
    return "rag_indexing"

builder = StateGraph(GraphState)

builder.add_node("translation",      translation_agent)
builder.add_node("completeness",     completeness_agent)
builder.add_node("human_review",     human_review)
builder.add_node("ehr_validation",   ehr_validation_agent)
builder.add_node("reporting",        reporting_agent)
builder.add_node("human_ehr_review", human_ehr_review)
builder.add_node("rag_indexing",     rag_indexing_node)

builder.set_entry_point("translation")

builder.add_edge("translation", "completeness")

builder.add_conditional_edges(
    "completeness",
    route_after_completeness,
    {
        "human_review":  "human_review",
        "ehr_validation": "ehr_validation",
    }
)
builder.add_conditional_edges(
    "human_review",
    route_after_validation,
    {
        END:END,
        "ehr_validation":"ehr_validation"
    }
)
builder.add_conditional_edges(
    "ehr_validation",
    route_after_ehr,
    {
        "reporting":"reporting",
        "human_ehr_review":"human_ehr_review"
    }
)
builder.add_conditional_edges(
    "human_ehr_review",
    route_after_ehr_review,
    {
        END:END,
        "reporting":"reporting"
    }
)
# builder.add_edge("human_ehr_review","reporting")
builder.add_edge("reporting","rag_indexing")

builder.add_edge("human_ehr_review", "rag_indexing")
builder.add_edge("rag_indexing", END)

if _USE_SQLITE:
    conn = sqlite3.connect("hospital_graph.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
else:
    checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)