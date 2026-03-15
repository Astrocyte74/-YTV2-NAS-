# Intel App Integration

This folder is ready for the Intel app to use in two ways:

1. Direct import inside the Intel app backend
2. Standalone HTTP service that the Intel app calls locally

The first option is the main recommendation if the Intel app is Python-based.

## Recommended: direct import

Copy `portable/research_api/research_service/` into the Intel app repo, install the dependencies from `requirements.txt`, and call `run_research(...)` directly.

Example:

```python
from research_service import run_research

run = run_research(
    message="Compare recent Cursor and Windsurf pricing with sources.",
    history=[],
    provider_mode="auto",
    tool_mode="auto",
    depth="balanced",
    compare=True,
    manual_tools={},
)

print(run.status)
print(run.answer)
print(run.meta.get("planner_llm_provider"), run.meta.get("planner_llm_model"))
print(run.meta.get("synth_llm_provider"), run.meta.get("synth_llm_model"))
```

## Alternate: run as a local service

If you want the Intel app to call it over HTTP instead:

```bash
cd portable/research_api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Available endpoints:

- `GET /health`
- `GET /api/research/status`
- `POST /api/research`
- `POST /api/research/stream`

## Environment variables

The local `.env` in this folder is intended as a temporary source-of-truth for moving the research configuration into the Intel app.

Suggested workflow:

1. Copy the needed entries from `portable/research_api/.env`
2. Paste them into the Intel app's local `.env`
3. Delete `portable/research_api/.env` after migration if you do not want it kept here

Important:

- `portable/research_api/.env` is ignored by git via the repo-wide `.env` ignore rules
- do not commit live keys to source control

## Minimum required keys

- `INCEPTION_API_KEY`
- `OPENROUTER_API_KEY` if fallback or Gemini planning is used
- `BRAVE_API_KEY`
- `TAVILY_API_KEY`

## Current recommended research mode

These settings preserve the hybrid setup you preferred:

- `RESEARCH_PLANNER_PROVIDER=openrouter`
- `RESEARCH_SYNTH_PROVIDER=inception`

That yields:

- Gemini planning
- Mercury synthesis

## Dependencies

- `Flask`
- `python-dotenv`
- `requests`

If the Intel app already has its own web framework, Flask is only needed if you also want to run the standalone API wrapper.
