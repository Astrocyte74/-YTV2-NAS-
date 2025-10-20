# Ollama Integration (via NAS Hub)

This repo can talk to your Mac‑hosted Ollama through the hub you already run for TTS. The NAS/bot only needs `TTSHUB_API_BASE`; the hub proxies to local Ollama and handles streaming.

## Base URL
- `TTSHUB_API_BASE` (NAS/bot): `http://<WG_IP_OF_MAC>:7860/api`
- `OLLAMA_URL` (hub on the Mac): defaults to `http://127.0.0.1:11434` (set only on the hub if non‑default)

## Endpoints (served by the hub)
- `GET /ollama/tags`
  - Lists models (raw of upstream `/api/tags`).
  - Example: `curl -sS "$TTSHUB_API_BASE/ollama/tags" | jq`
- `POST /ollama/generate`
  - Body: `{ "model": "phi3:latest", "prompt": "...", "stream": false }`
  - `stream=false`: single JSON. `stream=true`: SSE token stream; first event is `{ "status":"starting" }`.
  - Example (SSE):
    ```bash
    curl -N -sS -X POST "$TTSHUB_API_BASE/ollama/generate" \
      -H 'Content-Type: application/json' \
      -d '{"model":"tinyllama:latest","prompt":"Stream one line","stream":true}'
    ```
- `POST /ollama/chat`
  - Body: `{ "model": "phi3:latest", "messages": [{"role":"user","content":"..."}], "stream": false }`
  - Streaming identical to `/ollama/generate`.
- `POST /ollama/pull`
  - Body: `{ "model": "llama3:8b", "stream": true }` (defaults to `true`)
  - SSE progress events until `{ "status":"success" }`.
- `GET /ollama/ps` — runtime status (raw of upstream `/api/ps`).
- `GET|POST /ollama/show` — model details (`?model=name` or body).
- `GET|POST /ollama/delete` — removes model; normalizes “already missing” to 200.

## Streaming (SSE) Notes
- Transport: HTTP/1.1 with `Content-Type: text/event-stream`.
- Event framing: the hub wraps each upstream NDJSON line as a single SSE message: `data: {json}\n\n`.
- Liveness: the hub sends `data: {"status":"starting"}` immediately.
- End of stream: chat/generate emit `{ "done": true }`; pull emits `{ "status": "success" }`.

## Python usage (quick)
```python
import os, json, requests
BASE = os.environ["TTSHUB_API_BASE"].rstrip("/")

# Non-stream chat
body = {"model": "tinyllama:latest", "messages": [{"role":"user","content":"hello"}], "stream": False}
print(requests.post(f"{BASE}/ollama/chat", json=body, timeout=60).json())

# Stream chat
body["stream"] = True
with requests.post(f"{BASE}/ollama/chat", json=body, stream=True, timeout=None) as r:
    acc = []
    for raw in r.iter_lines(decode_unicode=True):
        if raw == "":
            if not acc: 
                continue
            payload = "".join(acc); acc.clear()
            if payload.startswith("b'") and payload.endswith("'"):
                import ast
                payload = ast.literal_eval(payload).decode("utf-8", errors="ignore")
            data = json.loads(payload)
            if data.get("status") == "starting":
                continue
            if isinstance(data.get("response"), str):
                print(data["response"], end="", flush=True)
            elif isinstance((data.get("message") or {}).get("content"), str):
                print(data["message"]["content"], end="", flush=True)
            if data.get("done"):
                break
```

## Telegram Bot Integration
- Commands: `/ollama`, `/o`, `/o_stop` (alias `/stop`), `/chat`
- Model picker:
  - Streaming is **on by default** (tokens stream into a single message). Override with `OLLAMA_STREAM_DEFAULT=0` if you prefer JSON responses by default.
  - Top row toggles between **Single AI Chat** and **AI↔AI Chat**. Single mode shows a 4×3 grid of installed models; tap a model and type a prompt to start.
  - Single chat auto-selects the first available model (override with `OLLAMA_DEFAULT_MODEL=<model>`), so you can begin typing immediately.
  - Single chat also exposes a **Models / Personas** toggle: choose a persona category, pick a persona, and the bot will role-play that identity (first reply introduces itself and invites you to introduce yourself).
- AI↔AI mode:
  - Pick model **A** and **B** in the integrated picker (model B list filters out A by default; allow same model with `OLLAMA_AI2AI_ALLOW_SAME=1`).
  - The first two available models are pre-selected automatically (override with `OLLAMA_AI2AI_DEFAULT_MODELS=ModelA,ModelB`).
  - Tap **Personas** under Model A or B to browse persona categories sourced from the environment (`OLLAMA_PERSONA_*`). Choose a category, then pick the persona name for that side.
  - Message headers show the current turn (e.g. `A · phi3:latest · Turn 3/10`) so you can follow the exchange at a glance.
  - Type a topic prompt—AI↔AI automatically runs the configured number of combined turns (default `OLLAMA_AI2AI_TURNS`, e.g. 10). Each turn is streamed with labels `A · <model>` / `B · <model>`.
  - Use `/chat <message>` at any time to inject a fresh prompt and immediately run another AI↔AI turn with the existing models/personas.
  - When the cycle completes, you’ll see “Continue AI↔AI” (runs another block) and “Options” (adjust turn count) as inline buttons, plus “Clear AI↔AI” to return to single chat.
  - Personas default to a random pair from your configured lists; once selected (or changed), each opening response introduces the persona and prompts the counterpart to introduce themselves.
- Personas for AI↔AI fall back to `OLLAMA_PERSONA` (comma-separated, e.g. `Albert Einstein,Isaac Newton`) if no category is chosen; unset variables use built-in defaults.
- Single chat responses are labelled `🤖 <model>` so you can tell which model answered.
- Implementation:
  - Client: `modules/ollama_client.py` (non-stream + SSE streaming helpers, handles SSE payloads such as `data: b'…'`)
  - Telegram: `modules/telegram_handler.py` (model picker, streaming updates, AI↔AI automation)

## Environment
- NAS/bot:
  - `TTSHUB_API_BASE` (required)
  - `OLLAMA_STREAM_DEFAULT` (`1` by default → streaming ON; set `0` to default OFF)
  - `OLLAMA_AI2AI_TURNS` (default number of combined turns when AI↔AI runs automatically, e.g. `10`)
  - `OLLAMA_AI2AI_ALLOW_SAME` (`0` by default; set `1` to allow the same model for both A and B)
  - `OLLAMA_DEFAULT_MODEL` (optional default model for single chat mode)
  - `OLLAMA_AI2AI_DEFAULT_MODELS` (optional comma-separated defaults for AI↔AI; first entry is model A, second entry model B)
  - `OLLAMA_PERSONA` (optional comma-separated fallback personas for AI↔AI, first entry used for A, second for B)
  - `OLLAMA_PERSONA_<CATEGORY>` (optional comma-separated persona lists grouped by category; the `<CATEGORY>` suffix becomes the label in the Telegram picker, e.g. `OLLAMA_PERSONA_ARTISTS`)
- Hub on Mac (optional):
  - `OLLAMA_URL` if Ollama is not on `http://127.0.0.1:11434`
  - `OLLAMA_ALLOW_CLI` to control delete fallback via the local `ollama` CLI

## Status & Errors
- Non‑stream returns `200` JSON; stream returns SSE events.
- Upstream failures surface as `503 {"error":"..."}` and close the stream.
- Delete normalizes “not installed” to 200 with a note.

## Tips
- For small/fast models use `tinyllama:latest` to validate end‑to‑end streaming.
- Streaming edits are throttled (≈2–3/s) to keep within Telegram edit limits.
- If you only list installed models in the picker, pull‑on‑demand can remain disabled on the bot.
