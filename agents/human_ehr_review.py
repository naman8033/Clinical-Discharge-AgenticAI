"""
Human-in-the-Loop node for EHR validation results.
Triggered when risk_level is 'medium' or 'high'.
Clinician can approve, override discrepancies, or reject.
"""

from langgraph.types import interrupt
from langgraph_workflows.state import GraphState


def human_ehr_review(state: GraphState) -> GraphState:
    print("ehr hitl...")

    ehr_parsed = state.get("ehr_parsed", {})
    final_report = state.get("final_report", {})

    review = interrupt({
        "patient_id": state.get("patient_id"),
        "risk_score": final_report.get("risk_score"),
        "risk_level": final_report.get("risk_level"),
        "recommendation": final_report.get("recommendation"),
        "hard_guardrail_triggered": ehr_parsed.get("hard_guardrail_triggered", []),
        "discrepancies": ehr_parsed.get("discrepancies", []),
        "patient_record": state.get("patient_record", {}),
    })

    action = review.get("action", "approve")

    if action == "approve":
        updated_report = {
            **final_report,
            "hitl_action": "approved",
            "hitl_notes": review.get("notes", ""),
            "clinician_override": True,
            "risk_level": "low",
            "recommendation": "Approve — Clinician Override",
        }
        return {**state, "final_report": updated_report, "status": "active"}

    elif action == "reject":
        updated_report = {
            **final_report,
            "hitl_action": "rejected",
            "hitl_notes": review.get("notes", ""),
        }
        return {**state, "final_report": updated_report, "status": "rejected"}

    else:
        updated_record = review.get("patient_record", state.get("patient_record", {}))
        updated_report = {
            **final_report,
            "hitl_action": "edited",
            "hitl_notes": review.get("notes", ""),
            "patient_record": updated_record,
        }
        return {**state, "patient_record": updated_record, "final_report": updated_report, "status": "active"}
