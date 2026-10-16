import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph_workflows.state import GraphState
from langchain_openai import ChatOpenAI
from agents.llm import gateway_kwargs, get_default_model
from dotenv import load_dotenv
import yaml

load_dotenv()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

with open(os.path.join(_ROOT, "configs", "rules.json"), "r") as f:
    rules = json.load(f)
json_schema = rules['json_keys']


def _load_persona() -> str:
    """Load the completeness_agent persona from persona_clinical_agent.yaml."""
    path = os.path.join(_ROOT, "configs", "persona_clinical_agent.yaml")
    with open(path, "r") as f:
        personas = yaml.safe_load(f)
    p = personas.get("completeness_agent", {})
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

def build_patient_record(translated_text: dict) -> dict:
    global json_schema
    """
        Uses LLM to consolidate doctor, lab and bill data
        into one canonical patient record.
    """
    persona = _load_persona()
    llm = ChatOpenAI(model=get_default_model(),
                     **gateway_kwargs())

    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        persona + """

---

        You are a clinical document consolidation assistant.
        You will receive translated clinical documents from multiple sources.
        Sources:
        1. Doctor Report
        2. Lab Report
        3. Hospital Bill
        Your task is to merge all information into ONE canonical patient record.
        Rules:

        - Return ONLY JSON.
        - Never hallucinate.
        - Never invent values.
        - If a field is unavailable, return null.
        - Merge duplicate information.
        - Prefer Doctor > Lab > Bill when conflicting.
        - Preserve medication lists.
        - Preserve laboratory tables.
        - Preserve billing information.

        Output Schema should be strict json and must contain all keys present in the input array:
        input array:{json_schema}
        Translated Documents:
        {translated_text}
        {format_instructions}
        """
    )
    chain = (
        prompt
        | llm
        | parser
    )
    patient_record = chain.invoke(
        {
            "translated_text": translated_text,
            "format_instructions": parser.get_format_instructions(),
            "json_schema":json_schema
        }
    )
    return patient_record


def validate_patient_record(patient_record: dict):
    global json_schema
    missing_fields = []
    for key, schema_value in json_schema.items():
        if key == "medications":
            medications = patient_record.get("medications")
            if not medications:
                missing_fields.append("medications")
                continue

            required_med_fields = schema_value[0].keys()
            for idx, medication in enumerate(medications):
                for field in required_med_fields:
                    if (
                        field not in medication
                        or medication[field] is None
                        or medication[field] == ""
                    ):
                        missing_fields.append(
                            f"medications[{idx}].{field}"
                        )
        else:
            if (
                key not in patient_record
                or patient_record[key] is None
                or patient_record[key] == ""
            ):
                missing_fields.append(key)

    return missing_fields

def completeness_agent(state:GraphState):
    print("completeness agent statrted")
    translated_text = state["translated_data"]
    patient_record = build_patient_record(
        translated_text
    )
    state["patient_record"] = patient_record
    state['missing_fields'] = validate_patient_record(patient_record)

    return state