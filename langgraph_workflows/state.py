from typing import TypedDict, Optional,List,Dict

class GraphState(TypedDict):
    patient_id:str
    raw_data:Dict
    translated_data:Dict
    # missing_fields: List[str]
    ehr_validation_result:Dict
    final_report:Dict