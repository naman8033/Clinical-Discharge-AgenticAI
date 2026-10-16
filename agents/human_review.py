from langgraph_workflows.state import GraphState
from langgraph.types import interrupt
from agents.completenss_agent import validate_patient_record

def human_review(state):
    print("In human review")
    
    review = interrupt({
        "patient_record": state["patient_record"],
        "missing_fields": state["missing_fields"]
    })

    action = review["action"]

    if action == "approve":
        state["patient_record"] = review["patient_record"]
        state["missing_fields"] = []
        state['status']="approved"

    elif action == "edit":
        print("inside backend edit ,graph invokation")
        state["patient_record"] = review["patient_record"]
        state["missing_fields"] = validate_patient_record(state["patient_record"])
        state["status"]="approved"

    elif action == "reject":
        state["status"] = "rejected"

    return state