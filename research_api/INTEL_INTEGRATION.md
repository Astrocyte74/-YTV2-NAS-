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

## Recommended test order

Do not start by debugging the Flask wrapper and the provider stack at the same time.
Validate the system in this order:

1. import the package successfully
2. verify `.env` loading and required keys
3. verify stage overrides
4. run one direct `run_research(...)` call
5. only then test the standalone HTTP wrapper
6. only after that test `provider_mode=auto` and Brave behavior

The shortest path is:

```bash
cd portable/research_api
python3 smoke_test.py --check-only
python3 smoke_test.py
```

The default smoke test intentionally uses:

- `provider_mode=tavily`
- `depth=quick`

That avoids Brave rate limiting as a first debugging variable.

## What success looks like

With the preferred hybrid config:

- planner should be `openrouter (google/gemini-3.1-flash-lite-preview)`
- synthesis should be `inception (mercury-2)`

If that is not what you see, stop and fix configuration before testing more prompts.

## Common failure signals

If planner and synthesis both show Mercury:

- the stage override env vars were not loaded

If planner is Gemini but synthesis is `fallback (deterministic)`:

- Mercury synthesis failed
- or the Inception key/config is wrong

If the query list is too narrow or one-sided on compare prompts:

- planner override is probably not active

If the answer preview starts with plain sections like `Research summary` and `Top findings`:

- deterministic synthesis fired

## After the smoke test passes

Once direct import works, then test:

1. `python3 app.py`
2. `GET /health`
3. `GET /api/research/status`
4. `POST /api/research`

If `/api/research/status` does not show the expected `llm_stage_overrides`, fix env loading before anything else.
