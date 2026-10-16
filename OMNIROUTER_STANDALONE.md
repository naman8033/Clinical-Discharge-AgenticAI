# OmniRouter Setup & Integration Guide

OmniRouter is fully integrated as the primary LLM provider for the **Agentic Discharge AI** project.

---

## 1. Connection Configuration

The project connects to your local OmniRouter server:
- **Base URL:** `http://localhost:20128/v1`
- **API Key:** `your-omnirouter-api-key`
- **Server Status:** Running with free-tier providers configured

These settings are defined in [`.env`](file:///.env) and managed globally via [`agents/llm.py`](file:///agents/llm.py).

```env
OMNIROUTER_BASE_URL=http://localhost:20128/v1
OMNIROUTER_API_KEY=your-omnirouter-api-key
LLM_BASE_URL=http://localhost:20128/v1
LLM_API_KEY=your-omnirouter-api-key
OPENAI_API_KEY=your-omnirouter-api-key

# Configured Models
OMNIROUTER_DEFAULT_MODEL=vertex/gemini-2.0-flash
OMNIROUTER_FAST_MODEL=vertex/gemini-2.0-flash
OMNIROUTER_REASONING_MODEL=vertex/gemini-2.5-flash
OMNIROUTER_PRO_MODEL=vertex/gemini-3.1-pro-preview
OMNIROUTER_EMBEDDING_MODEL=gemini/gemini-embedding-001
```

---

## 2. Models Assigned to Agents

| Agent / Component | File | Model | Role |
| :--- | :--- | :--- | :--- |
| **Translation & Normalization** | [`agents/translation_normalization_agent.py`](file:///agents/translation_normalization_agent.py) | `vertex/gemini-2.0-flash` | Multilingual document translation & medical abbreviation expansion |
| **Completeness Check** | [`agents/completenss_agent.py`](file:///agents/completenss_agent.py) | `vertex/gemini-2.0-flash` | Canonical patient record consolidation & missing-field checks |
| **EHR Validation (HITL)** | [`agents/ehr_validation_agent.py`](file:///agents/ehr_validation_agent.py) | `vertex/gemini-2.0-flash` | ReAct tool-calling agent validating records against EHR server |
| **Clinical Reporting** | [`agents/reporting_agent.py`](file:///agents/reporting_agent.py) | `vertex/gemini-2.5-flash` | Discharge summary generation & clinical risk assessment |
| **RAG Classification** | [`agents/rag_agents/router_agent.py`](file:///agents/rag_agents/router_agent.py) | `vertex/gemini-2.0-flash` | Question categorization (aggregate vs. patient-specific) |
| **Structured Query** | [`agents/rag_agents/structured_query_agent.py`](file:///agents/rag_agents/structured_query_agent.py) | `vertex/gemini-2.0-flash` | SQL/Tabular operations over discharged patients |
| **QA Generation** | [`agents/rag_agents/generation_agent.py`](file:///agents/rag_agents/generation_agent.py) | `vertex/gemini-2.0-flash` | Grounded clinical Q&A generation |
| **Vector Embeddings** | [`agents/rag_agents/indexing_agent.py`](file:///agents/rag_agents/indexing_agent.py) & [`ui/qa_assistant.py`](file:///ui/qa_assistant.py) | `gemini/gemini-embedding-001` | 3072-dimensional embeddings for FAISS vector store |

---

## 3. How Agents Access OmniRouter

All agents import helper utilities from [`agents/llm.py`](file:///agents/llm.py):

```python
from agents.llm import gateway_kwargs, get_default_model, get_embeddings, get_chat_llm
from langchain_openai import ChatOpenAI

# Standard Chat Model (e.g. Translation, Validation)
llm = ChatOpenAI(
    model=get_default_model(),
    **gateway_kwargs()
)

# Embeddings Model (FAISS vector store)
embeddings = get_embeddings()
```

---

## 4. Verification & Testing

To test and verify the entire OmniRouter integration at any time:

```bash
python test_omnirouter.py
```

### Verified Functionality:
1. **Connectivity**: Discovers all registered models on `http://localhost:20128/v1`.
2. **Chat Completions**: Fast, high-accuracy completion via `vertex/gemini-2.0-flash`.
3. **Function / Tool Calling**: Validated with LangGraph ReAct agent tools.
4. **Embeddings**: Generated 3072-dimensional vector representations with `gemini/gemini-embedding-001`.
5. **FAISS Vector Search**: Document indexing and semantic retrieval tested.
6. **Clinical Pipeline Execution**: End-to-end tests completed on sample medical reports.

---

## 5. Running the Application

- **Start Main Pipeline Monitor:**
  ```bash
  python main.py
  ```

- **Launch Streamlit Web UI:**
  ```bash
  streamlit run ui/streamlit_app.py
  ```
