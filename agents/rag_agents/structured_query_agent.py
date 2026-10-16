import os
import json
import glob
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agents.llm import gateway_kwargs, get_default_model

load_dotenv()

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "reports")

_llm = ChatOpenAI(
    model=get_default_model(),
    temperature=0,
    **gateway_kwargs(),
)

_FIELDS = [
    "patient_id", "patient_name", "age", "gender", "admission_date",
    "discharge_date", "ward", "attending_physician",
]

_PROMPT = """You are a hospital operations assistant. Answer the question using ONLY the patient \
records table below. Count, filter, or list rows exactly as needed — do not guess or include \
patients not present in the table. If the table doesn't contain enough information to answer, say so.

Patient records ({count} total):
{table}

Question:
{query}
"""


class StructuredQueryAgent:
    def _load_records(self):
        records = []
        for path in glob.glob(os.path.join(REPORTS_DIR, "*_report.json")):
            with open(path, "r") as f:
                report = json.load(f)
            patient_record = report.get("patient_record", {})
            row = {field: patient_record.get(field, report.get(field, "")) for field in _FIELDS}
            row["risk_level"] = report.get("risk_level", "")
            diagnoses = patient_record.get("discharge_diagnosis", []) or []
            row["discharge_diagnosis"] = ", ".join(
                d if isinstance(d, str) else d.get("diagnosis", str(d)) for d in diagnoses
            )
            records.append(row)
        return records

    def _to_table(self, records):
        header = list(records[0].keys()) if records else _FIELDS + ["risk_level", "discharge_diagnosis"]
        lines = [" | ".join(header)]
        for row in records:
            lines.append(" | ".join(str(row.get(col, "")) for col in header))
        return "\n".join(lines)

    def run(self, query: str) -> dict:
        records = self._load_records()
        print(f"[StructuredQueryAgent] loaded {len(records)} patient records from {REPORTS_DIR}")
        table = self._to_table(records)

        if not records:
            return {
                "answer": "No discharged patient records are available yet.",
                "context_table": table,
                "record_count": 0,
            }

        prompt = _PROMPT.format(count=len(records), table=table, query=query)
        response = _llm.invoke(prompt)
        print(f"[StructuredQueryAgent] answer: {response.content}")

        return {
            "answer": response.content.strip(),
            "context_table": table,
            "record_count": len(records),
        }
