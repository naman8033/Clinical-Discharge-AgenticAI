"""
LiteLLM Gateway — Agentic Discharge AI
========================================
Starts a local OpenAI-compatible proxy server using LiteLLM.

All agents point their `base_url` here (http://localhost:4000/v1) and use
their existing `ChatOpenAI` calls unchanged.  The gateway handles:

  - Model alias routing   (gpt-5.4-mini → actual provider model)
  - Automatic fallback    (OpenAI → Gemini → Ollama)
  - Retry / back-off      (configurable in litellm_config.yaml)
  - Token usage logging   (printed to console)
  - Budget guard-rails    (daily USD cap, optional)

Usage
-----
  python gateway/gateway.py              # start gateway on port 4000
  python gateway/gateway.py --port 4001  # custom port

The main app (main.py) automatically starts this as a background thread
before launching any agents.
"""

import os
import sys
import time
import argparse
import logging
import threading
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

_HERE = Path(__file__).parent
_CONFIG_PATH = _HERE / "litellm_config.yaml"
_DEFAULT_PORT = int(os.getenv("LITELLM_GATEWAY_PORT", "4000"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("litellm_gateway")

# ---------------------------------------------------------------------------
# Import LiteLLM proxy app
# ---------------------------------------------------------------------------
try:
    from litellm.proxy.proxy_server import app as litellm_app, initialize
    _LITELLM_PROXY_AVAILABLE = True
except ImportError:
    _LITELLM_PROXY_AVAILABLE = False
    logger.warning(
        "litellm[proxy] not installed — falling back to lightweight shim.\n"
        "  Run:  python -m pip install 'litellm[proxy]'"
    )

# ---------------------------------------------------------------------------
# Lightweight shim (used if litellm[proxy] is not installed)
# ---------------------------------------------------------------------------
if not _LITELLM_PROXY_AVAILABLE:
    import json
    import httpx
    import litellm
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    shim_app = FastAPI(title="LiteLLM Gateway (shim mode)")

    @shim_app.get("/health")
    async def health():
        return {"status": "ok", "mode": "shim"}

    @shim_app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Proxy /v1/chat/completions → OmniRouter / LiteLLM."""
        body = await request.json()
        model_alias = body.get("model", "gpt-5.4-mini")

        # Map alias → real model
        model_map = {
            "gpt-5.4-mini":   os.getenv("OMNIROUTER_DEFAULT_MODEL", "vertex/gemini-2.0-flash"),
            "gpt-5.4-turbo":  os.getenv("OMNIROUTER_REASONING_MODEL", "vertex/gemini-2.5-flash"),
            "gpt-4o-mini":    os.getenv("OMNIROUTER_DEFAULT_MODEL", "vertex/gemini-2.0-flash"),
            "gpt-4o":         os.getenv("OMNIROUTER_REASONING_MODEL", "vertex/gemini-2.5-flash"),
        }
        real_model = model_map.get(model_alias, model_alias)

        omnirouter_base = os.getenv("OMNIROUTER_BASE_URL", "http://localhost:20128/v1").rstrip("/")
        omnirouter_key = os.getenv("OMNIROUTER_API_KEY", "")

        # Direct HTTP forward to OmniRouter if available
        try:
            headers = {
                "Authorization": f"Bearer {omnirouter_key}",
                "Content-Type": "application/json",
            }
            forward_body = dict(body)
            forward_body["model"] = real_model
            async with httpx.AsyncClient(timeout=120.0) as client:
                res = await client.post(f"{omnirouter_base}/chat/completions", json=forward_body, headers=headers)
                return JSONResponse(status_code=res.status_code, content=res.json())
        except Exception as exc:
            logger.error("OmniRouter call failed: %s, falling back to litellm", exc)

        # Fallback to LiteLLM if direct forward fails
        kwargs = {
            "model": f"openai/{real_model}",
            "messages": body.get("messages", []),
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 8192),
            "stream": body.get("stream", False),
            "api_base": omnirouter_base,
            "api_key": omnirouter_key,
        }
        try:
            response = await litellm.acompletion(**kwargs)
            return JSONResponse(content=response.model_dump())
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": "gateway_error"}},
            )

    @shim_app.post("/v1/embeddings")
    async def embeddings(request: Request):
        """Proxy /v1/embeddings → OmniRouter."""
        body = await request.json()
        model_alias = body.get("model", "text-embedding-3-small")
        embedding_model = os.getenv("OMNIROUTER_EMBEDDING_MODEL", "gemini/gemini-embedding-001")

        omnirouter_base = os.getenv("OMNIROUTER_BASE_URL", "http://localhost:20128/v1").rstrip("/")
        omnirouter_key = os.getenv("OMNIROUTER_API_KEY", "")

        try:
            headers = {
                "Authorization": f"Bearer {omnirouter_key}",
                "Content-Type": "application/json",
            }
            forward_body = {
                "model": embedding_model,
                "input": body.get("input", ""),
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(f"{omnirouter_base}/embeddings", json=forward_body, headers=headers)
                return JSONResponse(status_code=res.status_code, content=res.json())
        except Exception as exc:
            logger.error("OmniRouter embeddings call failed: %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": "gateway_error"}},
            )

    @shim_app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "gpt-5.4-mini",  "object": "model"},
                {"id": "gpt-5.4-turbo", "object": "model"},
            ],
        }

    litellm_app = shim_app


# ---------------------------------------------------------------------------
# Gateway startup helpers
# ---------------------------------------------------------------------------

def _resolve_api_keys():
    """
    Ensure the correct environment variables are set.
    The gateway prefers OPENAI_API_KEY; falls back to GEMINI_API_KEY.
    Agents may set FOUNDATION_MODEL_API_KEY — we copy it over so LiteLLM
    picks it up automatically.
    """
    foundation_key = os.getenv("FOUNDATION_MODEL_API_KEY")
    if foundation_key:
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = foundation_key
            logger.info("FOUNDATION_MODEL_API_KEY → OPENAI_API_KEY")
        # Do not copy Conduit key (sk-cdt-) to GEMINI_API_KEY, as it is OpenAI-compatible only
        if not os.getenv("GEMINI_API_KEY") and not foundation_key.startswith("sk-cdt-"):
            os.environ["GEMINI_API_KEY"] = foundation_key
            logger.info("FOUNDATION_MODEL_API_KEY → GEMINI_API_KEY")


async def _init_litellm_proxy(port: int):
    """Initialise the full LiteLLM proxy with our YAML config."""
    if _LITELLM_PROXY_AVAILABLE:
        logger.info("Initialising LiteLLM proxy with config: %s", _CONFIG_PATH)
        await initialize(
            config=str(_CONFIG_PATH),
            port=port,
            host="0.0.0.0",
            debug=False,
        )


def start_gateway(port: int = _DEFAULT_PORT, block: bool = True):
    """
    Start the LiteLLM gateway.

    Parameters
    ----------
    port  : TCP port to listen on (default: 4000)
    block : If True, runs forever (use False to start as background thread)
    """
    _resolve_api_keys()

    if not _LITELLM_PROXY_AVAILABLE:
        logger.warning("Running in SHIM mode — install litellm[proxy] for full features.")

    if block:
        logger.info("[GATEWAY] LiteLLM Gateway starting on http://localhost:%d", port)
        uvicorn.run(
            litellm_app,
            host="0.0.0.0",
            port=port,
            log_level="warning",
        )
    else:
        # Non-blocking: run in a daemon thread so main.py can continue
        def _run():
            import asyncio
            if _LITELLM_PROXY_AVAILABLE:
                asyncio.run(_init_litellm_proxy(port))
            uvicorn.run(
                litellm_app,
                host="0.0.0.0",
                port=port,
                log_level="warning",
            )

        t = threading.Thread(target=_run, daemon=True, name="litellm-gateway")
        t.start()
        logger.info("[GATEWAY] LiteLLM Gateway started on http://localhost:%d (background thread)", port)

        # Wait until the gateway accepts connections
        _wait_for_gateway(port)
        return t


def _wait_for_gateway(port: int, timeout: int = 30):
    """Block until the gateway's /health endpoint responds."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    url = f"http://localhost:{port}/health"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            logger.info("[GATEWAY] Ready at http://localhost:%d", port)
            return
        except Exception:
            time.sleep(0.5)
    logger.warning("[GATEWAY] Did not respond within %ds -- agents may fail!", timeout)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiteLLM Gateway for Agentic Discharge AI")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help="Port to listen on")
    args = parser.parse_args()

    start_gateway(port=args.port, block=True)
