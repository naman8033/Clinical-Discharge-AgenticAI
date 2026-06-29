import os
import json
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph_workflows.state import GraphState

load_dotenv()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _build_llm():
    return ChatOpenAI(
        model="gpt-5.4-mini",
        api_key=os.getenv("FOUNDATION_MODEL_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

def _load_abbrev_map() -> str:
    with open(os.path.join(_ROOT, "configs", "rules.yaml"), "r") as f:
        rules = yaml.safe_load(f)
    abbrev = rules.get("normalization_standards", {}).get("abbreviation_map", {})
    return "\n".join(f"  {k} → {v}" for k, v in abbrev.items())

def translate_with_llm(text: str) -> dict:
    """Translate + normalize one document. Returns {text, confidence, detected_language}."""
    llm = _build_llm()
    abbrev_examples = _load_abbrev_map()

    prompt = f"""You are a medical translation and normalization assistant.

Your job:
1. Detect the source language of the input text.
2. Translate everything to English.
3. Preserve medical meaning exactly.
4. Expand ALL medical abbreviations using this map:
{abbrev_examples}
5. Convert structured tables into natural readable text.
6. Remove OCR artifacts (======, ------, |, extra spaces, broken line wrapping).

After translating, estimate a confidence score (0.0 to 1.0) for:
- 1.0 = source was already English, clear text
- 0.85-0.99 = translation confident, medical terms clear
- 0.70-0.84 = some ambiguity in medical terms or language
- below 0.70 = low confidence (handwriting OCR issues, rare dialect, heavy jargon)

Return ONLY valid JSON in this exact format:
{{
  "translated_text": "<clean English text here>",
  "detected_language": "<ISO 639-1 code, e.g. en, es, hi, de>",
  "confidence": <float between 0.0 and 1.0>
}}

Input text:
{text}
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        parsed = json.loads(raw)
        return {
            "text": parsed.get("translated_text", raw),
            "confidence": float(parsed.get("confidence", 0.85)),
            "detected_language": parsed.get("detected_language", "en"),
        }
    except (json.JSONDecodeError, ValueError):
        return {"text": raw, "confidence": 0.6, "detected_language": "unknown"}


def translation_agent(state: GraphState) -> GraphState:
    print("[Translation] Starting...")

    bill_result   = translate_with_llm(state["raw_data"]["bill"])
    doctor_result = translate_with_llm(state["raw_data"]["doctor"])
    lab_result    = translate_with_llm(state["raw_data"]["lab"])

    return {
        **state,
        "translated_data": {
            "bill":   bill_result["text"],
            "doctor": doctor_result["text"],
            "lab":    lab_result["text"],
        },
        "translation_confidence": {
            "bill":   bill_result["confidence"],
            "doctor": doctor_result["confidence"],
            "lab":    lab_result["confidence"],
            "detected_languages": {
                "bill":   bill_result["detected_language"],
                "doctor": doctor_result["detected_language"],
                "lab":    lab_result["detected_language"],
            },
        },
    }

