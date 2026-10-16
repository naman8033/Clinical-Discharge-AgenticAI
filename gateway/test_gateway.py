"""
Quick smoke-test for the LiteLLM Gateway.

Usage:
    py gateway/test_gateway.py
"""
# -*- coding: utf-8 -*-
import os
import sys
import time
import urllib.request

# Force UTF-8 output on Windows to avoid cp1252 issues
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

GATEWAY_PORT = int(os.getenv("LITELLM_GATEWAY_PORT", "4000"))
GATEWAY_URL = f"http://localhost:{GATEWAY_PORT}"


def test_gateway_health():
    print(f"\n[1] Checking gateway health -> {GATEWAY_URL}/health")
    try:
        with urllib.request.urlopen(f"{GATEWAY_URL}/health", timeout=5) as r:
            body = r.read().decode()
            print(f"    [OK]  HTTP {r.status}: {body}")
            return True
    except Exception as e:
        print(f"    [FAIL]  Not reachable: {e}")
        return False


def test_chat_completion():
    print(f"\n[2] Testing /v1/chat/completions ...")
    import json, urllib.error

    payload = json.dumps({
        "model": "gpt-5.4-mini",
        "messages": [{"role": "user", "content": "Say 'LiteLLM gateway OK' in exactly 5 words."}],
        "max_tokens": 30,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{GATEWAY_URL}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('LITELLM_MASTER_KEY', 'test')}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode())
            answer = body["choices"][0]["message"]["content"]
            model_used = body.get("model", "unknown")
            usage = body.get("usage", {})
            print(f"    [OK]  Response   : {answer}")
            print(f"    [OK]  Model used : {model_used}")
            print(f"    [OK]  Tokens     : {usage}")
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"    [FAIL]  HTTP {e.code}: {err}")
    except Exception as e:
        print(f"    [FAIL]  Error: {e}")


if __name__ == "__main__":
    print("=" * 55)
    print("  LiteLLM Gateway -- Smoke Test")
    print("=" * 55)

    # Start the gateway (blocking=False -> background thread)
    from gateway.gateway import start_gateway
    start_gateway(port=GATEWAY_PORT, block=False)
    time.sleep(2)

    ok = test_gateway_health()
    if ok:
        test_chat_completion()
    else:
        print("\n    Gateway not running. Start it first:")
        print(f"    py gateway/gateway.py --port {GATEWAY_PORT}")

    print("\nDone.")
