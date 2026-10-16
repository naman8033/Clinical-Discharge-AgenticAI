"""
Discharge Summary Generation Agent (Reporting Agent)
=====================================================
- Loads EHR validation result + patient record from state
- Calculates final risk score using rules.yaml weights
- Generates patient-friendly discharge summary (plain English, no jargon)
- Outputs:  state['final_report'] dict
            data/reports/<patient_id>_report.json
            data/reports/<patient_id>_report.html
"""

import os
import json
import yaml
import datetime
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agents.llm import gateway_kwargs, get_reasoning_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph_workflows.state import GraphState

load_dotenv()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(_ROOT, "data", "reports")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_persona() -> str:
    """Load the reporting_agent persona from persona_clinical_agent.yaml."""
    path = os.path.join(_ROOT, "configs", "persona_clinical_agent.yaml")
    with open(path, "r") as f:
        personas = yaml.safe_load(f)
    p = personas.get("reporting_agent", {})
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

def _llm():
    return ChatOpenAI(
        model=get_reasoning_model(),
        **gateway_kwargs(),
    )

def _parse_ehr_result(raw: Any) -> dict:
    """EHR validation agent returns a JSON string; parse it safely. """
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



def _load_rules() -> dict:
    with open(os.path.join(_ROOT, "configs", "rules.yaml")) as f:
        return yaml.safe_load(f)


def _generate_patient_summary(patient_record: dict, ehr_parsed: dict) -> str:
    """Ask LLM to produce a plain-English discharge summary."""
    persona = _load_persona()
    prompt = ChatPromptTemplate.from_template(persona + """

---

You are a clinical documentation specialist writing a patient-friendly discharge summary.

Rules:
- Replace ALL medical abbreviations with plain English (BID → twice daily, QD → once daily, TID → three times daily, QHS → at bedtime, PRN → as needed, PO → by mouth, etc.)
- Write in simple language that any patient can understand
- Include ALL sections below, even if some data is null (write "Not recorded" for those)
- For medications table, include all prescription fields: medicine name, strength, dose, frequency (plain English), route, duration, remarks, total quantity
- Include lab results with vendor and reference ranges
- Include full bill details and payment status

Patient Record:
{patient_record}

EHR Validation Notes:
{discrepancies}

Generate a comprehensive discharge summary with these sections:
1. Patient Information (name, ID, age, gender, address)
2. Admission & Discharge Details (dates, ward, bed)
3. Treating Doctors (attending physician + consulting doctors)
4. Discharge Diagnosis
5. Discharge Instructions
6. Follow-up Appointment
7. Medications / Prescriptions (as a table: #, Medicine, Strength, Dose, Frequency, Route, Duration, Remarks, Total Qty)
8. Lab Reports (test name, value, status, vendor)
9. Bill Summary (items, total amount, payment status)
10. Discharge Approval Status

Return the summary as plain text — no markdown, no headers with #, just clear sections with labels.
""")
    llm = _llm()
    chain = prompt | llm
    result = chain.invoke({
        "patient_record": json.dumps(patient_record, indent=2),
        "discrepancies": json.dumps(ehr_parsed.get("discrepancies", []), indent=2),
    })
    return result.content


def _write_json_report(report: dict, patient_id: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{patient_id}_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def _write_html_report(report: dict, patient_id: str) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{patient_id}_report.html")

    risk_colors = {"low": "#28a745", "medium": "#ffc107", "high": "#dc3545"}
    risk_level = report.get("risk_level", "unknown")
    color = risk_colors.get(risk_level, "#6c757d")

    discrepancies_html = ""
    for d in report.get("discrepancies", []):
        discrepancies_html += f"""
        <tr>
          <td>{d.get('type','')}</td>
          <td>{d.get('description','')}</td>
          <td>{d.get('evidence','')}</td>
          <td><strong>{d.get('weight',0)}</strong></td>
        </tr>"""

    summary_text = report.get("patient_summary", "").replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Discharge Report — {patient_id}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #333; }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
  h2 {{ color: #2e4057; margin-top: 30px; }}
  .badge {{ display: inline-block; padding: 6px 18px; border-radius: 20px;
            color: #fff; font-weight: bold; background: {color}; font-size: 1.1em; }}
  .score-box {{ border: 2px solid {color}; border-radius: 8px; padding: 16px;
                margin: 16px 0; background: #fafafa; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th {{ background: #2e4057; color: #fff; padding: 8px; text-align: left; }}
  td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
  .summary {{ background: #f0f4f8; padding: 20px; border-radius: 8px;
              border-left: 4px solid #1a5276; white-space: pre-wrap; line-height: 1.7; }}
  .footer {{ font-size: 0.85em; color: #888; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 12px; }}
</style>
</head>
<body>
<h1>🏥 Patient Discharge Report</h1>
<p><strong>Patient ID:</strong> {patient_id} &nbsp;|&nbsp;
   <strong>Generated:</strong> {report.get('generated_at','')}</p>

<div class="score-box">
  <h2 style="margin-top:0">Risk Assessment</h2>
  <p>Risk Score: <strong>{report.get('risk_score', 0)}</strong> &nbsp;&nbsp;
     Risk Level: <span class="badge">{risk_level.upper()}</span></p>
  <p><strong>Recommendation:</strong> {report.get('recommendation','')}</p>
  {"<p><strong>⚠ Hard Guardrails Triggered:</strong> " + ", ".join(report.get('hard_guardrail_triggered',[])) + "</p>" if report.get('hard_guardrail_triggered') else ""}
</div>

<h2>Translation Confidence</h2>
<table>
  <tr><th>Document</th><th>Confidence</th><th>Language Detected</th></tr>
  <tr><td>Doctor Report</td><td>{report.get('translation_confidence',{}).get('doctor','N/A')}</td>
      <td>{report.get('translation_confidence',{}).get('detected_languages',{}).get('doctor','N/A')}</td></tr>
  <tr><td>Lab Report</td><td>{report.get('translation_confidence',{}).get('lab','N/A')}</td>
      <td>{report.get('translation_confidence',{}).get('detected_languages',{}).get('lab','N/A')}</td></tr>
  <tr><td>Bill</td><td>{report.get('translation_confidence',{}).get('bill','N/A')}</td>
      <td>{report.get('translation_confidence',{}).get('detected_languages',{}).get('bill','N/A')}</td></tr>
</table>

<h2>EHR Discrepancies</h2>
{"<p>✅ No discrepancies found.</p>" if not report.get('discrepancies') else f"""
<table>
  <tr><th>Type</th><th>Description</th><th>Evidence</th><th>Weight</th></tr>
  {discrepancies_html}
</table>"""}

<h2>Patient-Friendly Discharge Summary</h2>
<div class="summary">{summary_text}</div>

<div class="footer">
  <p>Generated by Agentic AI System for Automated Discharge Summaries &nbsp;|&nbsp;
     Rules Version: {report.get('rules_version','N/A')} &nbsp;|&nbsp;
     Thread: {report.get('thread_id','N/A')}</p>
</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def reporting_agent(state: GraphState) -> GraphState:
    print("[Reporting] Starting discharge summary generation...")

    rules = _load_rules()
    patient_id = state.get("patient_id", "UNKNOWN")
    patient_record = state.get("patient_record", {})
    translation_confidence = state.get("translation_confidence", {})

    # parse EHR result
    ehr_raw = state.get("ehr_validation_result", {})
    ehr_parsed = _parse_ehr_result(ehr_raw)

    # # recalculate / finalise risk score
    # score, risk_level, recommendation = _calc_risk_score(
    #     ehr_parsed, translation_confidence, rules
    # )

    # generate patient-friendly narrative summary
    patient_summary = _generate_patient_summary(patient_record, ehr_parsed)

    # build final report dict
    report = {
        "patient_id": patient_id,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "thread_id": state.get("patient_id", "unknown"),
        "rules_version": "1.0",
        # "risk_score": score,
        # "risk_level": risk_level,
        # "recommendation": recommendation,
        # "hitl_required": ehr_parsed.get("hitl_required", risk_level != "low"),
        # "hard_guardrail_triggered": ehr_parsed.get("hard_guardrail_triggered", []),
        "translation_confidence": translation_confidence,
        "missing_fields": state.get("missing_fields", []),
        # "discrepancies": ehr_parsed.get("discrepancies", []),
        # "ehr_validation_passed": ehr_parsed.get("ehr_validation_passed", False),
        "patient_record": patient_record,
        "patient_summary": patient_summary,
        **state['final_report']
    }

    # write reports to disk
    json_path = _write_json_report(report, patient_id)
    html_path = _write_html_report(report, patient_id)
    report["json_report_path"] = json_path
    report["html_report_path"] = html_path

    # print(f"[Reporting] Done. Risk: {risk_level} (score={score}). Report → {html_path}")

    return {**state, "final_report": report, "ehr_parsed": ehr_parsed}
