"""Shared client settings for OmniRouter and LLM providers."""

import os
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()


def gateway_kwargs() -> dict[str, str]:
    """Return client settings configured for OmniRouter."""
    api_key = (
        os.getenv("OMNIROUTER_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LITELLM_MASTER_KEY")
        or "none"
    )
    base_url = (
        os.getenv("OMNIROUTER_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or "http://localhost:20128/v1"
    )
    # Ensure /v1 suffix is present for OpenAI client compatibility
    if not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
        base_url = f"{base_url.rstrip('/')}/v1"

    return {
        "api_key": api_key,
        "base_url": base_url,
    }


def get_default_model() -> str:
    """Return default chat model for standard tasks."""
    return os.getenv("OMNIROUTER_DEFAULT_MODEL", "vertex/gemini-2.0-flash")


def get_fast_model() -> str:
    """Return fast model for routing, classification, and quick checks."""
    return os.getenv("OMNIROUTER_FAST_MODEL", "vertex/gemini-2.0-flash")


def get_reasoning_model() -> str:
    """Return high-capability model for clinical review and reporting."""
    return os.getenv("OMNIROUTER_REASONING_MODEL", "vertex/gemini-2.5-flash")


def get_embedding_model() -> str:
    """Return embedding model supported by OmniRouter."""
    return os.getenv("OMNIROUTER_EMBEDDING_MODEL", "gemini/gemini-embedding-001")


def get_chat_llm(model: Optional[str] = None, temperature: float = 0.7, **kwargs: Any):
    """Factory function for LangChain ChatOpenAI configured with OmniRouter."""
    from langchain_openai import ChatOpenAI

    chosen_model = model or get_default_model()
    return ChatOpenAI(
        model=chosen_model,
        temperature=temperature,
        **gateway_kwargs(),
        **kwargs,
    )


def get_embeddings(model: Optional[str] = None, **kwargs: Any):
    """Factory function for LangChain OpenAIEmbeddings configured with OmniRouter."""
    from langchain_openai import OpenAIEmbeddings

    chosen_model = model or get_embedding_model()
    return OpenAIEmbeddings(
        model=chosen_model,
        check_embedding_ctx_length=False,
        **gateway_kwargs(),
        **kwargs,
    )
