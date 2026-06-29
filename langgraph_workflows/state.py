from typing import TypedDict, Optional, List, Dict, Any

class GraphState(TypedDict):
    patient_id: str
    raw_data: Dict
    translated_data: Dict
    translation_confidence: Dict        # {bill: float, doctor: float, lab: float}
    patient_record: Dict
    missing_fields: List[str]
    status: str                         # active | rejected
    ehr_validation_done: bool
    ehr_validation_result: Any          # raw JSON string from EHR agent
    ehr_parsed: Dict                    # parsed dict from ehr_validation_result
    final_report: Dict                  # structured report from reporting agent