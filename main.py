"""
Agentic Discharge AI — Main Entry Point
=========================================
Start order:
  1. LiteLLM Gateway  (background thread, port 4000)
  2. Monitor Agent    (watches `incoming/` folder for new patient files)
"""
import os
from dotenv import load_dotenv

# Load .env FIRST so gateway + agents pick up all env vars
load_dotenv()

# ── 1. Start the LiteLLM Gateway ─────────────────────────────────────────────
from gateway.gateway import start_gateway

gateway_thread = start_gateway(
    port=int(os.getenv("LITELLM_GATEWAY_PORT", "4000")),
    block=False,   # non-blocking: runs in daemon thread
)

# ── 2. Start the Monitor Agent ───────────────────────────────────────────────
from agents.monitor_agent import start_monitor


def main():
    print("=" * 60)
    print("  Agentic Discharge AI")
    print(f"  LiteLLM Gateway : http://localhost:{os.getenv('LITELLM_GATEWAY_PORT', '4000')}/v1")
    print("=" * 60)
    start_monitor()


if __name__ == "__main__":
    main()