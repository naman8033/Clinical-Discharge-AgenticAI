import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agents.llm import gateway_kwargs, get_fast_model

load_dotenv()

_llm = ChatOpenAI(
    model=get_fast_model(),
    temperature=0,
    **gateway_kwargs(),
)

_PROMPT = """Classify the clinical question below into exactly one category:

- "aggregate": questions that need counting, filtering, or listing across MULTIPLE patients \
(e.g. "how many patients were discharged on 30 June", "list all patients with high risk", \
"which patients have diabetes").
- "patient_specific": questions about the clinical details of ONE specific patient's record \
(e.g. medications, diagnosis, allergies, bill, lab results for a particular patient).

Question: {query}

Reply with ONLY one word: aggregate or patient_specific"""


def classify_query(query: str) -> str:
    """Returns 'aggregate' or 'patient_specific'."""
    response = _llm.invoke(_PROMPT.format(query=query))
    label = response.content.strip().lower()
    print(f"[Router] query classified as: {label}")
    return "aggregate" if "aggregate" in label else "patient_specific"
