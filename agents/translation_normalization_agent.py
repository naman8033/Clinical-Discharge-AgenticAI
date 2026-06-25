from langgraph_workflows.state import GraphState
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json

load_dotenv()
# raw,translated -> {
# bill : {text:str,confidence:float}
# doctor : str
# lab:str
# }
def translate_with_llm(text):
    llm = ChatOpenAI(model="gpt-5.4-mini",
                     api_key=os.getenv("FOUNDATION_MODEL_API_KEY"),
                     base_url=os.getenv("LLM_BASE_URL"))
    
    with open("configs/rules.json", "r") as f:
        rules = json.load(f)

    prompt = f"""
    You are a medical translation and normalization assistant.

    Your job:
        1. Detect source language.
        2. Translate everything to English.
        3. Preserve medical meaning exactly.
        4. Expand medical abbreviations.
        Examples:{rules}

        5. Convert structured tables into natural readable text.
        6. Remove OCR artifacts such as:
        - ======
        - ------
        - |
        - extra spaces
        - broken line wrapping

        IMPORTANT:
        Return ONLY clean plain English text.
        Do NOT return JSON.
        Do NOT return markdown.
        Do NOT include explanations.

        Example conversion:

        Input:
        Glucosa en Ayuno | 92 | mg/dL | NORMAL

        Output:
        Fasting glucose is 92 mg/dL, which is within the normal range.

        Input text:
        {text}
    """
    response = llm.invoke(prompt)
    return response.content
    
def translation_agent(state:GraphState):
    print("translation agent started first node")

    bill_text = state['raw_data']['bill']
    doctor_text = state['raw_data']['doctor']
    lab_text = state['raw_data']['lab']

    state['translated_data'] = {}
    state['translated_data']['bill'] = translate_with_llm(bill_text)
    state['translated_data']['doctor'] = translate_with_llm(doctor_text)
    state['translated_data']['lab'] = translate_with_llm(lab_text)

    return state

