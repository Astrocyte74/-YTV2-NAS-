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
- Command: `/ollama`
  - Shows available models (installed on hub). Pick a model to start chatting.
- Options:
  - Streaming: live‑edit a single message as tokens stream in (on/off).
  - AI↔AI mode: chat between two personas.
    - Pick A model (first tap), then B model (second tap) directly from the main picker; start AI↔AI, then “Continue exchange” to generate turns.
- Implementation:
  - Client: `modules/ollama_client.py` (non‑stream + SSE streaming helpers)
  - Telegram: `modules/telegram_handler.py` (model picker, sessions, streaming edits, AI↔AI scaffolding)

## Environment
- NAS/bot: `TTSHUB_API_BASE` only.
- Hub on Mac: (optional) `OLLAMA_URL` if not default; `OLLAMA_ALLOW_CLI` for delete fallback.

## Status & Errors
- Non‑stream returns `200` JSON; stream returns SSE events.
- Upstream failures surface as `503 {"error":"..."}` and close the stream.
- Delete normalizes “not installed” to 200 with a note.

## Tips
- For small/fast models use `tinyllama:latest` to validate end‑to‑end streaming.
- Streaming edits are throttled (≈2–3/s) to keep within Telegram edit limits.
- If you only list installed models in the picker, pull‑on‑demand can remain disabled on the bot.

