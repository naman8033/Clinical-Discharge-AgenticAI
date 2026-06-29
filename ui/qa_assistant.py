"""
QA Assistant page — Agentic RAG-based clinical question answering.
Uses the 5 RAG agents: Indexing → Retrieval → Augmentation → Generation → Reflection.
The FAISS index is built automatically after the discharge pipeline finishes.
"""

import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

_api_key  = os.getenv("FOUNDATION_MODEL_API_KEY")
_base_url = os.getenv("LLM_BASE_URL")

FAISS_INDEX_PATH = "faiss_index"


@st.cache_resource(show_spinner=False)
def _load_vector_db():
    """Load the FAISS index from disk (created by the indexing agent after the pipeline)."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=_api_key,
        base_url=_base_url,
    )
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _run_rag_pipeline(query: str, vector_db) -> dict:
    """Run all 5 RAG agents and return the full result dict."""
    from agents.rag_agents.retrieval_agent import RetrievalAgent
    from agents.rag_agents.augmentation_agent import AugmentationAgent
    from agents.rag_agents.generation_agent import GenerationAgent
    from agents.rag_agents.reflection_agent import ReflectionAgent

    retrieval_result   = RetrievalAgent().run(vector_db, query)
    augmentation_result = AugmentationAgent().run(retrieval_result)
    generation_result  = GenerationAgent().run({**augmentation_result, "query": query})
    reflection_result  = ReflectionAgent().run({
        **generation_result,
        **augmentation_result,
        "query": query,
    })
    return {
        "retrieved_chunks":       retrieval_result["retrieved_chunks"],
        "augmented_context":      augmentation_result["augmented_context"],
        "generated_answer":       generation_result["generated_answer"],
        "final_answer":           reflection_result["final_answer"],
        "groundedness_score":     reflection_result["groundedness_score"],
        "answer_relevance_score": reflection_result["answer_relevance_score"],
        "context_relevance_score":reflection_result["context_relevance_score"],
        "safety_label":           reflection_result["safety_label"],
        "grounded":               reflection_result["grounded"],
    }


def qa_chatbot_page():
    st.title("🤖 Clinical Q&A Assistant")
    st.caption("Ask questions about the patient's discharge record. Answers are grounded in the indexed report.")

    # ---- example queries ----
    st.info(
        "**Example questions:**\n"
        "- Was the patient discharged with appropriate follow-up for diabetes?\n"
        "- List medication discrepancies for this patient.\n"
        "- What is the patient's discharge diagnosis?\n"
        "- Is the bill settled before discharge?\n"
        "- What are the discharge instructions?"
    )

    st.divider()

    vector_db = _load_vector_db()

    if vector_db is None:
        st.warning(
            "⚠ No indexed report found. Please complete the discharge pipeline first "
            "(upload documents → validation → EHR check → report generation)."
        )
        return

    # ---- chat history ----
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # ---- render history ----
    for turn in st.session_state.qa_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            cols = st.columns(3)
            cols[0].metric("Groundedness", f"{turn['groundedness']:.2f}")
            cols[1].metric("Relevance",    f"{turn['relevance']:.2f}")
            cols[2].metric("Safety",       turn["safety"])

    # ---- input ----
    query = st.chat_input("Ask a clinical question about this patient...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching discharge records..."):
                result = _run_rag_pipeline(query, vector_db)

            final_answer = result["final_answer"]
            safety_label = result["safety_label"]

            if safety_label == "Unsafe":
                st.error("⚠ Unable to provide a safe answer from the available records.")
                st.write(final_answer)
            else:
                st.write(final_answer)

            cols = st.columns(3)
            cols[0].metric("Groundedness", f"{result['groundedness_score']:.2f}")
            cols[1].metric("Relevance",    f"{result['answer_relevance_score']:.2f}")
            cols[2].metric("Safety",       safety_label)

            with st.expander("📄 Retrieved Context Chunks"):
                for i, chunk in enumerate(result["retrieved_chunks"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk)

            st.session_state.qa_history.append({
                "question":   query,
                "answer":     final_answer,
                "groundedness": result["groundedness_score"],
                "relevance":  result["answer_relevance_score"],
                "safety":     safety_label,
            })

    # ---- clear history ----
    if st.session_state.qa_history:
        st.divider()
        if st.button("🗑 Clear Chat History"):
            st.session_state.qa_history = []
            st.rerun()
