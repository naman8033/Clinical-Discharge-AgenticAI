from typing import TypedDict
from langgraph.graph import StateGraph, END

from agents.rag_agents.indexing_agent import IndexingAgent
from agents.rag_agents.retrieval_agent import RetrievalAgent
from agents.rag_agents.augmentation_agent import AugmentationAgent
from agents.rag_agents.generation_agent import GenerationAgent
from agents.rag_agents.reflection_agent import ReflectionAgent


class RAGState(TypedDict):
    report_text: str
    query: str

    vector_db: object
    retrieved_chunks: list
    augmented_context: str
    generated_answer: str

    groundedness_score: float
    answer_relevance_score: float
    context_relevance_score: float

    safety_label: str
    grounded: bool
    final_answer: str


indexing = IndexingAgent()
retrieval = RetrievalAgent()
augmentation = AugmentationAgent()
generation = GenerationAgent()
reflection = ReflectionAgent()


def indexing_node(state):
    vector_db = indexing.run(state["report_text"])
    return {"vector_db": vector_db}


def retrieval_node(state):
    return retrieval.run(
        state["vector_db"],
        state["query"]
    )


def augmentation_node(state):
    return augmentation.run(state)


def generation_node(state):
    return generation.run(state)


def reflection_node(state):
    return reflection.run(state)


graph = StateGraph(RAGState)

graph.add_node("Indexing", indexing_node)
graph.add_node("Retrieval", retrieval_node)
graph.add_node("Augmentation", augmentation_node)
graph.add_node("Generation", generation_node)
graph.add_node("Reflection", reflection_node)

graph.set_entry_point("Indexing")

graph.add_edge("Indexing", "Retrieval")
graph.add_edge("Retrieval", "Augmentation")
graph.add_edge("Augmentation", "Generation")
graph.add_edge("Generation", "Reflection")
graph.add_edge("Reflection", END)

app = graph.compile()

data = """ 

    # Patient Medical Summary



## Patient Information

- **Name:** Thomas Wright

- **Patient ID:** P1019`

- **Age:** 53 years

- **Sex:** Male

- **Hospital:** St. Marian Regional Medical Center

- **Admission Date:** 26-May-2026

- **Discharge Date:** 29-May-2026

- **Length of Stay:** 3 days



## Primary Diagnoses

1. Type 2 Diabetes Mellitus (T2DM)

2. Essential Hypertension (High Blood Pressure)



**Allergies:** No Known Drug Allergies (NKDA)



## Key Laboratory Results

| Test | Result | Reference Range | Interpretation |

|--------|--------|----------------|----------------|

| HbA1c | 6.9% | 4.0–5.6% | Elevated |

| Fasting Glucose | 122 mg/dL | 70–99 mg/dL | Elevated |

| LDL Cholesterol | 118 mg/dL | <100 mg/dL | Elevated |

| Creatinine | 0.9 mg/dL | 0.6–1.1 mg/dL | Normal |

| Potassium | 4.2 mmol/L | 3.5–5.1 mmol/L | Normal |



### Lab Assessment

- Diabetes remains above target control based on elevated HbA1c and fasting glucose levels.

- LDL cholesterol is higher than recommended and requires continued cholesterol-lowering therapy.

- Kidney function and electrolyte levels are within normal limits.



## Medications at Discharge

| Medication | Dose | Frequency | Purpose |

|------------|------|-----------|---------|

| Metformin | 500 mg | Twice daily | Diabetes management |

| Lisinopril | 10 mg | Once daily | Blood pressure control |

| Atorvastatin | 20 mg | Nightly | Cholesterol management |

| Aspirin | 81 mg | Once daily | Cardiovascular protection |



## Discharge Instructions

- Continue a diabetic-friendly diet.

- Perform at least 30 minutes of walking, 5 times per week.

- Monitor blood glucose and blood pressure daily and record readings.

- Take medications exactly as prescribed.

- Seek urgent medical attention for chest pain, severe shortness of breath, or blood glucose levels above 300 mg/dL.



## Follow-Up Appointments

- **Primary Care Physician:** 05-Jun-2026

- **Endocrinology Clinic:** 12-Jun-2026



## Billing Summary

- **Total Hospital Charges:** $1,903.07

- **Major Charges Included:**

  - 3-day ward stay

  - Physician services

  - HbA1c and lipid panel testing

  - Discharge medications

- **Payment Status:** Paid in Full

- **Payment Method:** Insurance (co-pay collected at discharge)



## Overall Summary

Thomas Wright was hospitalized for management of Type 2 Diabetes Mellitus and Hypertension. He remained clinically stable and was discharged after a 3-day stay. Laboratory findings showed moderately elevated blood glucose and LDL cholesterol levels, while kidney function and potassium levels were normal. The patient was discharged on Metformin, Lisinopril, Atorvastatin, and Aspirin, with recommendations for lifestyle modifications, daily self-monitoring, and scheduled follow-up appointments to optimize diabetes, blood pressure, and cholesterol control.



"""









if __name__ == "__main__":

    state = {
        "report_text": data,
        "query": "What are the patient's primary diagnoses?"
    }

    result = app.invoke(state)

    print("\n========== FINAL ANSWER ==========")
    print(result["final_answer"])

    print("\n========== RAG TRIAD ==========")
    print(
        "Groundedness Score:",
        result["groundedness_score"]
    )

    print(
        "Answer Relevance Score:",
        result["answer_relevance_score"]
    )

    print(
        "Context Relevance Score:",
        result["context_relevance_score"]
    )

    print(
        "Safety Label:",
        result["safety_label"]
    )

    print(
        "Grounded:",
        result["grounded"]
    )