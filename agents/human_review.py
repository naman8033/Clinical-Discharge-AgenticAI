import streamlit as st
from langgraph_workflows.state import GraphState
from langgraph.types import interrupt
from agents.completenss_agent import validate_patient_record

def human_review(state):
    review = interrupt({
        "patient_record": state["patient_record"],
        "missing_fields": state["missing_fields"]
    })

    action = review["action"]

    if action == "approve":
        state["patient_record"] = review["patient_record"]
        state["missing_fields"] = []

    elif action == "edit":
        state["patient_record"] = review["patient_record"]
        # Re-run validation after edits
        state["missing_fields"] = validate_patient_record(state["patient_record"])

    elif action == "reject":
        state["status"] = "rejected"

    return state