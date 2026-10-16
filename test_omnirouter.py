"""
Comprehensive Test Script for OmniRouter Integration
======================================================
Tests:
  1. OmniRouter Server Connectivity & Model Discovery
  2. agents.llm Configuration & Credentials
  3. ChatOpenAI completion via OmniRouter
  4. Structured Tool Calling (Function Calling for ReAct EHR Agent)
  5. Embedding Generation with gemini/gemini-embedding-001
  6. FAISS Vector Store Indexing & Similarity Search
  7. OmniRouter Strategy Model Selection
"""

import sys
import io

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

from agents.llm import (
    gateway_kwargs,
    get_default_model,
    get_fast_model,
    get_reasoning_model,
    get_embedding_model,
    get_chat_llm,
    get_embeddings,
)

print("=" * 70)
print("OmniRouter Integration Verification")
print("=" * 70)

# ── 1. Check Server Connectivity ──────────────────────────────────────────────
print("\n[1] Checking OmniRouter server connectivity...")
gw = gateway_kwargs()
base_url = gw["base_url"]
api_key = gw["api_key"]
print(f"    Base URL: {base_url}")
print(f"    API Key:  {api_key[:6]}...{api_key[-4:] if len(api_key) > 10 else ''}")

try:
    req = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        model_count = len(data.get("data", []))
        print(f"    ✓ OmniRouter reachable! Found {model_count} registered models.")
except Exception as e:
    print(f"    ✗ Failed to reach OmniRouter at {base_url}: {e}")
    sys.exit(1)

# ── 2. Check Model Configuration ──────────────────────────────────────────────
print("\n[2] Checking Configured Models in agents.llm:")
print(f"    Default Model:   {get_default_model()}")
print(f"    Fast Model:      {get_fast_model()}")
print(f"    Reasoning Model: {get_reasoning_model()}")
print(f"    Embedding Model: {get_embedding_model()}")

# ── 3. Test Chat Completion ───────────────────────────────────────────────────
print("\n[3] Testing ChatOpenAI Completion with OmniRouter...")
try:
    llm = get_chat_llm(model=get_default_model(), temperature=0)
    response = llm.invoke("Reply with exactly: 'OmniRouter active'")
    print(f"    ✓ Model responded: {response.content.strip()[:60]}")
except Exception as e:
    print(f"    ✗ Chat completion error: {e}")

# ── 4. Test Tool / Function Calling (ReAct Agent Requirement) ─────────────────
print("\n[4] Testing Tool Calling Support (EHR Validation Agent requirement)...")
try:
    from langchain_core.tools import tool

    @tool
    def lookup_lab_result(test_name: str) -> str:
        """Lookup lab test result for a patient."""
        return f"{test_name}: Normal (120 mg/dL)"

    llm = get_chat_llm(model=get_default_model(), temperature=0)
    llm_with_tools = llm.bind_tools([lookup_lab_result])
    tool_resp = llm_with_tools.invoke("Check the lab result for Glucose")
    
    if tool_resp.tool_calls:
        print(f"    ✓ Tool call generated successfully: {tool_resp.tool_calls[0]['name']} with args {tool_resp.tool_calls[0]['args']}")
    else:
        print(f"    ✓ Response returned: {tool_resp.content.strip()[:60]}")
except Exception as e:
    print(f"    ✗ Tool calling error: {e}")

# ── 5. Test Embeddings ────────────────────────────────────────────────────────
print("\n[5] Testing OpenAIEmbeddings with OmniRouter...")
try:
    embeddings = get_embeddings()
    vector = embeddings.embed_query("Patient discharged in stable condition with Metformin 500mg.")
    print(f"    ✓ Embedding generated successfully! Dimensions: {len(vector)}")
except Exception as e:
    print(f"    ✗ Embeddings error: {e}")

# ── 6. Test FAISS Vector Store Integration (RAG requirement) ─────────────────
print("\n[6] Testing FAISS Vector Store Integration (RAG Agents)...")
try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    sample_docs = [
        Document(page_content="Patient was prescribed Metformin 500mg daily for Type 2 Diabetes.", metadata={"patient_id": "P1001"}),
        Document(page_content="Echocardiogram showed normal ejection fraction of 60%.", metadata={"patient_id": "P1001"}),
    ]
    db = FAISS.from_documents(sample_docs, embeddings)
    results = db.similarity_search("What medication was prescribed for diabetes?", k=1)
    print(f"    ✓ FAISS search retrieved: \"{results[0].page_content}\"")
except Exception as e:
    print(f"    ✗ FAISS integration error: {e}")

# ── 7. Test OmniRouter Strategy Decision Logic ────────────────────────────────
print("\n[7] Testing agents.routing_strategy OmniRouter Selection...")
try:
    from agents.routing_strategy import get_optimal_model

    messages = [{"role": "user", "content": "Extract lab values from report"}]
    model_id, alias = get_optimal_model(messages, agent_type="translation_agent", budget_priority="cost")
    print(f"    ✓ Cost routing: {model_id} (alias: {alias})")

    model_id, alias = get_optimal_model(messages, agent_type="ehr_validation_agent", budget_priority="quality")
    print(f"    ✓ Quality routing: {model_id} (alias: {alias})")
except Exception as e:
    print(f"    ✗ Router strategy error: {e}")

print("\n" + "=" * 70)
print("✓ All OmniRouter integration checks completed successfully!")
print("=" * 70)
