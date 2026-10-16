import os
import json
import yaml
import requests
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agents.llm import gateway_kwargs, get_default_model
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langgraph_workflows.state import GraphState
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_persona() -> str:
    """Load the ehr_validation_agent persona from persona_clinical_agent.yaml."""
    path = os.path.join(_ROOT, "configs", "persona_clinical_agent.yaml")
    with open(path, "r") as f:
        personas = yaml.safe_load(f)
    p = personas.get("ehr_validation_agent", {})
    expertise = "\n".join(f"  - {e}" for e in p.get("expertise", []))
    responsibilities = "\n".join(f"  - {r}" for r in p.get("responsibilities", []))
    behavioral_guidelines = "\n".join(f"  - {b}" for b in p.get("behavioral_guidelines", []))
    return (
        f"## Agent Identity\n"
        f"Name: {p.get('name', '')}\n"
        f"Role: {p.get('role', '').strip()}\n\n"
        f"## Expertise\n{expertise}\n\n"
        f"## Responsibilities\n{responsibilities}\n\n"
        f"## Behavioral Guidelines\n{behavioral_guidelines}\n"
    )

BASE_URL = "http://localhost:8001"

@tool
def get_patient(patient_id: str) -> dict:
    """Fetch demographic and admission data for a patient from the mock EHR."""
    return requests.get(f"{BASE_URL}/ehr/patient/{patient_id}").json()  

@tool
def get_medicine_orders(patient_id: str) -> dict:
    """Fetch the current medication orders from the mock EHR for a patient."""
    return requests.get(f"{BASE_URL}/ehr/med_orders/{patient_id}").json()

@tool
def get_lab_data(patient_id: str) -> dict:
    """Fetch lab results from the mock EHR. Flag any abnormal values."""
    return requests.get(f"{BASE_URL}/ehr/labs/{patient_id}").json()

@tool
def get_careplan(patient_id: str) -> dict:
    """Fetch the active care plan from the mock EHR, including follow-up requirements."""
    return requests.get(f"{BASE_URL}/ehr/careplan/{patient_id}").json()

@tool
def get_allergies(patient_id: str) -> dict:
    """Fetch the recorded allergy list from the mock EHR."""
    return requests.get(f"{BASE_URL}/ehr/allergies/{patient_id}").json()

@tool
def get_pathway(diagnosis_code: str) -> dict:
    """Fetch the clinical guideline pathway for a given ICD-10 diagnosis code."""
    return requests.get(f"{BASE_URL}/guidelines/pathway/{diagnosis_code}").json()

def _build_system_prompt(rules: dict, patient_record: dict) -> str:
    """
    rules.yaml drives the agent in three distinct ways:

    1. clinical_validation_policies  → WHAT to check
       e.g. allergy_must_not_match_prescription = true  means the agent must
       call get_allergies() and cross-check every prescribed drug.
       high_risk_meds_need_counseling lists which drugs need extra scrutiny.

    2. risk_scoring_matrix.weights   → HOW MUCH each violation costs
       e.g. allergy_contradiction = 8  means finding an allergy conflict
       adds 8 points to the total risk score.

    3. risk_scoring_matrix.hitl_hard_guardrails  → IMMEDIATE escalation
       These violations set hitl_required=true regardless of the total score.
       e.g. allergy_contradiction is both a weight AND a hard guardrail,
       so it always blocks auto-approval even if score is somehow ≤ 2.
    """
    persona    = _load_persona()
    policies   = rules["clinical_validation_policies"]
    weights    = rules["risk_scoring_matrix"]["weights"]
    thresholds = rules["risk_scoring_matrix"]["thresholds"]
    guardrails = rules["risk_scoring_matrix"]["hitl_hard_guardrails"]
    high_risk  = policies["high_risk_meds_need_counseling"]

    return f"""{persona}

---

You are an EHR Validation Agent — a cross-reference auditor.

## Patient discharge record to validate
{patient_record}

## Your validation checklist (from clinical_validation_policies)
Use the EHR tools to check ALL of the following:

1. MEDICATION RECONCILIATION
   - Call get_medicine_orders() and compare every drug in the discharge record
     against EHR history.
   - A drug in EHR but missing from discharge → medication_omission (weight: {weights['medication_omission']})
   - A drug in discharge not in EHR → medication_added (weight: {weights['medication_added']})
   - High-risk drugs requiring counseling notes: {high_risk}
     Missing counseling for these → high_risk_med_no_counseling (weight: {weights['high_risk_med_no_counseling']})
   - If any high-risk drug is completely absent from EHR → IMMEDIATE ESCALATE
     (weight: {weights['high_risk_med_missing_in_ehr']} — always triggers HITL)

2. ALLERGY CHECK  (policy: allergy_must_not_match_prescription = {policies['allergy_must_not_match_prescription']})
   - Call get_allergies() and check every prescribed drug against the allergy list.
   - Any match → allergy_contradiction (weight: {weights['allergy_contradiction']}) — ALWAYS triggers HITL

3. DIAGNOSIS CONSISTENCY
   - Call get_careplan() and compare discharge_diagnosis with the EHR care plan.
   - Mismatch → diagnosis_mismatch (weight: {weights['diagnosis_mismatch']})

4. FOLLOW-UP REQUIREMENTS
   - If care plan requires follow-up but discharge record has none →
     followup_missing (weight: {weights['followup_missing']})

5. ABNORMAL LAB RESOLUTION
   - Call get_lab_data() and find any abnormal values.
   - If abnormal lab has no documented action in the discharge record →
     abnormal_lab_unresolved (weight: {weights['abnormal_lab_unresolved']})

6. DISCHARGE APPROVAL
   - Check that discharge_ok field is true in patient record.
   - Doctor's approval must be present before any release.

7. BILLING
   - Check that bill payment status is settled.
   - Unpaid bill with discharge approved → bill_unpaid_with_discharge_ok
     (weight: {weights['bill_unpaid_with_discharge_ok']})

## Scoring rules
After checking everything, sum the weights of every violation found.
- score ≤ {thresholds['low_max']}  → risk_level: "low",    hitl_required: false
- score ≤ {thresholds['medium_max']} → risk_level: "medium", hitl_required: true
- score >  {thresholds['medium_max']} → risk_level: "high",   hitl_required: true

## Hard guardrails — ALWAYS set hitl_required=true if any of these fire:
{guardrails}
These override the score — even a score of 0 must trigger HITL if a guardrail fires.

## Output format — return ONLY this JSON, no prose:
{{
  "patient_id": "...",
  "discrepancies": [
    {{
      "type": "medication_omission | allergy_contradiction | diagnosis_mismatch | ...",
      "description": "human-readable explanation",
      "evidence": "what the EHR says vs what the discharge says",
      "weight": <int>
    }}
  ],
  "risk_score": <int>,
  "risk_level": "low | medium | high",
  "hitl_required": <bool>,
  "hard_guardrail_triggered": ["list of guardrail names if any"],
  "recommendation": "Approve — Auto-release | Needs Review — Standard HITL | Urgent Attention — Block release",
  "ehr_validation_passed": <bool>
}}
"""

def _parse_ehr_result(raw: Any) -> dict:
    """EHR validation agent returns a JSON string; parse it safely."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {}

def _calc_risk_score(ehr_parsed: dict, translation_confidence: dict, rules: dict) -> tuple[int, str, str]:
    """
    Recalculate risk score from ehr_parsed discrepancies + translation confidence.
    Returns (score, risk_level, recommendation).
    """
    weights = rules["risk_scoring_matrix"]["weights"]
    thresholds = rules["risk_scoring_matrix"]["thresholds"]
    reporting = rules.get("reporting", {}).get("recommendations", {})

    # start from ehr agent score if available
    score = int(ehr_parsed.get("risk_score", 0))

    # add low-confidence translation penalty
    min_conf = min(translation_confidence.get(k, 1.0) for k in ("bill", "doctor", "lab"))
    quality_min = rules.get("quality_thresholds", {}).get("translation_confidence_min", 0.70)
    if min_conf < quality_min:
        score += weights.get("low_translation_confidence", 3)

    # determine level
    low_max = thresholds.get("low_max", 2)
    medium_max = thresholds.get("medium_max", 8)

    guardrails = ehr_parsed.get("hard_guardrail_triggered", [])
    hitl_required = ehr_parsed.get("hitl_required", False) or bool(guardrails)

    if score <= low_max and not hitl_required:
        risk_level = "low"
        recommendation = reporting.get("low", "Approve — Auto-release")
    elif score <= medium_max or (hitl_required and score <= medium_max):
        risk_level = "medium"
        recommendation = reporting.get("medium", "Needs Review — Standard HITL")
    else:
        risk_level = "high"
        recommendation = reporting.get("high", "Urgent Attention — Block release")

    return score, risk_level, recommendation

def ehr_validation_agent(state: GraphState) -> GraphState:
    print("[EHR Validation agent] Starting...")

    with open(os.path.join(_ROOT, "configs", "rules.yaml")) as f:
        rules = yaml.safe_load(f)

    tools = [
        get_patient, get_medicine_orders, get_allergies,
        get_lab_data, get_careplan, get_pathway
    ]

    llm = ChatOpenAI(
        model=get_default_model(),                          
        **gateway_kwargs(),
    )

    system_prompt = _build_system_prompt(rules, state["patient_record"])

    agent = create_react_agent(                        
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    result = agent.invoke({
        "messages": [("user", "Validate this patient's discharge record against the EHR.")]
    })

    last_message = result["messages"][-1].content
    ehr_raw = _parse_ehr_result(last_message)

    ehr_parsed = _parse_ehr_result(ehr_raw)

    score, risk_level, recommendation = _calc_risk_score(
        ehr_parsed, state.get('translation_confidence',{}), rules
    )
    report={
        "risk_score": score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "hitl_required": ehr_parsed.get("hitl_required", risk_level != "low"),
        "hard_guardrail_triggered": ehr_parsed.get("hard_guardrail_triggered", []),
        "discrepancies": ehr_parsed.get("discrepancies", []),
        "ehr_validation_passed": ehr_parsed.get("ehr_validation_passed", False),
    }

    return {
        **state,
        "final_report":report,
        "ehr_validation_result": last_message,
        "ehr_validation_done": True,
    }