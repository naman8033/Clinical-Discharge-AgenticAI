# LiteLLM Gateway — Agentic Discharge AI

A **local OpenAI-compatible proxy** that sits between your agents and any LLM provider.
All four agents (`translation`, `completeness`, `ehr_validation`, `reporting`) continue using their existing `ChatOpenAI` calls — they just hit `http://localhost:4000/v1` instead of OpenAI directly.

```
Agents (ChatOpenAI)
      │  model="gpt-5.4-mini"
      │  base_url="http://localhost:4000/v1"
      ▼
┌─────────────────────────────────────────┐
│       LiteLLM Gateway  :4000            │
│  • Model alias routing                  │
│  • Automatic provider fallback          │
│  • Retry / back-off                     │
│  • Token usage logging                  │
│  • Budget guard-rails                   │
└───────┬──────────────┬──────────────────┘
        │              │
  OpenAI gpt-4o-mini   Gemini 2.0 Flash  …
```

## Quick Start

### 1. Add your API key to `.env`
```env
OPENAI_API_KEY=sk-...         # primary provider
GEMINI_API_KEY=...            # fallback (optional)
FOUNDATION_MODEL_API_KEY=sk-... # used by agents, copied by gateway
```

### 2. Install dependencies
```powershell
py -m pip install -r requirements.txt
```

### 3a. Run everything together (recommended)
```powershell
py main.py
```
`main.py` starts the gateway as a **background thread** then launches the monitor agent.

### 3b. Run gateway standalone (for debugging)
```powershell
py gateway/gateway.py              # port 4000
py gateway/gateway.py --port 4001  # custom port
```

### 3c. Run the smoke test
```powershell
py gateway/test_gateway.py
```

---

## File Layout

```
gateway/
├── __init__.py
├── gateway.py           # FastAPI proxy server (shim + full LiteLLM proxy)
├── litellm_config.yaml  # Model routing rules, retry, budget
└── test_gateway.py      # Smoke test
```

## Model Routing (`litellm_config.yaml`)

| Virtual alias used by agents | Real provider model    | Priority |
|------------------------------|------------------------|----------|
| `gpt-5.4-mini`               | `openai/gpt-4o-mini`   | Primary  |
| `gpt-5.4-mini`               | `gemini/gemini-3.6-flash` | Fallback |
| `gpt-5.4-turbo`              | `openai/gpt-4o`        | Heavy    |

To swap to a different model, **only edit `litellm_config.yaml`** — no agent code changes needed.

## Modes

| Mode | Condition | Features |
|------|-----------|----------|
| **Full proxy** | `litellm[proxy]` installed | All LiteLLM features + admin UI |
| **Shim mode** | only `litellm` installed | Basic routing + fallback |

Install the full proxy:
```powershell
py -m pip install "litellm[proxy]"
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | OpenAI provider key |
| `GEMINI_API_KEY` | — | Gemini fallback key |
| `FOUNDATION_MODEL_API_KEY` | — | Copied → both keys above |
| `LLM_BASE_URL` | `http://localhost:4000/v1` | Used by agents |
| `LITELLM_GATEWAY_PORT` | `4000` | Gateway listen port |
| `LITELLM_MASTER_KEY` | `discharge-ai-gateway-secret` | Admin API key |
