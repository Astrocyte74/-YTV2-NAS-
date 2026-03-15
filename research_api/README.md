# Portable Research API

This folder is a self-contained extraction of the research pipeline from TTS Hub.
It is designed to be copied into another repo or run as a standalone service on an always-on machine.

## What is included

- Neutral HTTP endpoints:
  - `GET /health`
  - `GET /api/research/status`
  - `POST /api/research`
  - `POST /api/research/stream`
- Relative imports only inside `research_service/`
- Hybrid planner/synthesis controls:
  - `RESEARCH_PLANNER_PROVIDER=auto|inception|openrouter`
  - `RESEARCH_SYNTH_PROVIDER=auto|inception|openrouter`
- Brave and Tavily retrieval
- Mercury/OpenRouter planning and synthesis with fallback and synthesis retries

## Folder layout

```text
portable/research_api/
├── app.py
├── requirements.txt
├── .env.example
└── research_service/
    ├── __init__.py
    ├── config.py
    ├── executor.py
    ├── llm.py
    ├── models.py
    ├── planner.py
    ├── service.py
    ├── synthesizer.py
    └── providers/
        ├── __init__.py
        ├── base.py
        ├── brave.py
        └── tavily.py
```

## Run standalone

```bash
cd /path/to/research_api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 app.py
```

Default port is `8090`.

## Quick validation

Before debugging provider behavior or HTTP integration, run the included smoke test:

```bash
cd /path/to/research_api
python3 smoke_test.py --check-only
python3 smoke_test.py
```

See [INTEL_INTEGRATION.md](./INTEL_INTEGRATION.md) for the recommended validation order and failure signals.

## Example request

```bash
curl -sS http://127.0.0.1:8090/api/research \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Compare recent pricing and feature differences between Cursor and Windsurf with sources.",
    "provider_mode": "tavily",
    "tool_mode": "auto",
    "depth": "quick",
    "compare": true
  }' | jq
```

## Streaming request

```bash
curl -N http://127.0.0.1:8090/api/research/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Summarize recent Apple Intelligence rollout updates with sources.",
    "provider_mode": "auto",
    "tool_mode": "auto",
    "depth": "balanced",
    "compare": false
  }'
```

## Response shape

Sync and streamed final results both return:

```json
{
  "status": "ok",
  "response": "Grounded answer text...",
  "sources": [
    {
      "name": "Example Source",
      "url": "https://example.com/story",
      "domain": "example.com",
      "tier": "expert-review",
      "providers": ["tavily"],
      "tools": ["search"]
    }
  ],
  "meta": {
    "objective": "Research objective",
    "queries": ["..."],
    "planner_llm_provider": "openrouter",
    "planner_llm_model": "google/gemini-3.1-flash-lite-preview",
    "synth_llm_provider": "inception",
    "synth_llm_model": "mercury-2"
  },
  "error": null
}
```

## Incorporating into another app

The simplest integration is backend-to-backend:

1. Run this service on the always-on Mac.
2. Keep API keys only in that environment.
3. Call it from your other webapp's backend using HTTP.
4. Pass the response through to the UI or adapt it there.

If you want to embed it directly into another Python app instead of running it as a separate service, import from `research_service` and call `run_research(...)`.
