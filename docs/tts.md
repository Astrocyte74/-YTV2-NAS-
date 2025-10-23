# NAS TTS Integration Overview

## Current Components

- `YouTubeSummarizer.generate_tts_audio(...)`
  - Acts as the single entry point for audio generation.
  - Supports two providers: `local` (WireGuard-accessible TTS hub) and `openai` (cloud fallback).
  - Retains chunking + stitching for long texts.

- `modules/tts_hub.py`
  - Async client for the local TTS hub (`/api/meta`, `/voices_catalog`, `/favorites`, `/synthesise`).
  - Exposes accent-family helpers (`available_accent_families`, `filter_catalog_voices`).
  - Raises `LocalTTSUnavailable` when the hub cannot be reached.

- `modules/tts_queue.py`
  - Lightweight file-based queue (`data/tts_queue/`).
  - Stores jobs for deferred local synthesis when the hub is offline.

- `modules/telegram_handler.py`
  - Orchestrates Telegram prompts and UI for both one‑off TTS (`/tts`) and YouTube summary audio.
  - Provider picker: “Local TTS hub” vs “OpenAI TTS”. On local failure: “Queue for later” or “Use OpenAI”.
  - Voice picker: Favorites‑first by default (if any). Toggle between Favorites and All voices, switch engines (Kokoro/XTTS, etc.), and filter by gender or accent family.
  - UX details: Shows a status bubble when a voice is selected (⏳ Generating … → ✅ Generated …), keeps the picker open for rapid A/B tests, and includes the voice name in audio captions.

## Typical Flow

1. User requests either:
   - One‑off preview: `/tts your text here` (no DB/Render sync), or
   - Summary audio: choose an audio variant from the YouTube flow (DB/Render sync enabled).
2. The bot prompts for provider:
   - Local hub → synthesize via the NAS TTS hub (preferred).
   - OpenAI → synthesize via OpenAI as fallback.
   - If local hub is offline/unreachable → offer to queue job or switch to OpenAI.
3. Voice selection:
   - Favorites are shown/selected by default when available. If no favorite matches the active engine, the picker auto-switches to one that does.
   - Engine chips start on `ALL` (aggregated view) and let you narrow to a specific engine; voice buttons show a short prefix like `[K]` or `[X]` so you know which catalog they belong to.
   - Optionally switch engines manually, flip to All voices, and filter by gender and accent family.
4. When you click a voice, the bot posts a “Generating …” status bubble, then replies with the audio and updates the bubble to “Generated …”. The picker remains open for more tests.
5. Delivery and sync:
   - One‑off preview (`/tts`): sends audio with a compact caption and does NOT sync to DB/Render.
  - Summary audio (YouTube flow): sends audio and syncs to Postgres (dashboard playback reads from the synced exports plus Postgres metadata).

## Modes and Captions

- One‑off TTS (`/tts`)
  - Mode key: `oneoff_tts`
  - Caption: `TTS Preview • {voice} • {provider}`
  - No DB/Render upload, intended for quick voice auditioning. Picker stays open after each run.

- Summary Audio (YouTube flow)
  - Mode key: `summary_audio`
  - Caption: `Audio Summary: {Title} • {voice}` on the first line; provider on the next line.
  - Postgres ingest enabled (controls appear on the dashboard; Listen chips stream the saved MP3 from the exports share).

## Favorites Resolution and Defaults

- When provider `local` is selected, the voice picker defaults to Favorites if any favorites exist.
- Favorites are fetched from the hub in one call; starred entries drive the picker (with an automatic fallback to all voices if the favorites list would be empty for the current catalog).
- Engine chips surface both an `ALL` tab (aggregated) and per-engine tabs; the picker auto-selects the first engine that contains your favorites, shows a hint when it switches, and prefixes each button with the engine code.
- The picker checkmarks reflect the current mode: “Favorites” or “All voices”.

## Environment

- Local hub base URL: `TTSHUB_API_BASE` (required for `local` provider)
- OpenAI key: `OPENAI_API_KEY` (required for `openai` provider)
- Dashboard sync (summary audio only):
  - `DATABASE_URL` – Postgres connection for direct upserts
  - `AUDIO_PUBLIC_BASE` – Base used to construct public audio URLs (e.g., `https://your-host` → `${AUDIO_PUBLIC_BASE}/exports/<file>.mp3`)

## Local Hub Testing (curl one‑liners)

Use these commands to verify your TTS hub favorites and synth endpoints.

Setup (pick your hub)

```bash
export HUB=http://10.0.4.2:7860
export API=$HUB/api
# Optional (only if your hub requires favorites auth)
# export TTSHUB_API_KEY=your_key
```

List favorites (shape and fields)

```bash
curl -sS ${API}/favorites | jq '.profiles[] | {label,engine,voiceId,slug,id,tags}'
# Filter examples
curl -sS "${API}/favorites?tag=ai2ai" | jq '.profiles[] | {label,engine,voiceId,slug,id,tags}'
curl -sS "${API}/favorites?engine=kokoro" | jq '.profiles[] | {label,engine,voiceId,slug,id,tags}'
```

Single favorite

```bash
curl -sS "${API}/favorites/<favorite_id>" | jq
```

Call formats (two valid bodies)

```bash
# By favorite (recommended)
{"text":"Hello","favoriteSlug":"favorite--af-heart"}
# or
{"text":"Hello","favoriteId":"fav_6644b291e1dd"}

# By engine + voice
{"text":"Hello","engine":"kokoro","voice":"af_heart"}
```

Synthesize examples

```bash
# By slug (JSON back)
curl -sS -X POST "${API}/synthesise" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from hub","favoriteSlug":"favorite--af-heart"}' | jq

# By id
curl -sS -X POST "${API}/synthesise" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello","favoriteId":"fav_6644b291e1dd"}' | jq

# By engine + voice
curl -sS -X POST "${API}/synthesise" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello","engine":"kokoro","voice":"af_heart"}' | jq
```

Download the returned audio

```bash
resp=$(curl -sS -X POST "${API}/synthesise" -H 'Content-Type: application/json' -d '{"text":"Hello","favoriteSlug":"favorite--af-heart"}')
u=$(printf %s "$resp" | jq -r '.url // .path // .filename // .file')
curl -sS "${HUB}${u#/}" -o out.wav && echo "Saved: out.wav"
```

Notes

- Favorites JSON includes: `label`, `engine`, `voiceId`, `slug`, `id`, `tags`, `notes`.
- If the hub enforces auth on favorites, add the header to favorites/CRUD calls:
  - `-H "Authorization: Bearer $TTSHUB_API_KEY"`
- The audio path resolves under `${HUB}` (no `/api` prefix): combine as shown above.

## In‑Container Testing (docker exec)

Run these inside your NAS (host) against the running bot container to validate local hub connectivity and the AI↔AI recap flow with gendered voice selection.

Prereqs

- Container name: `youtube-summarizer-bot` (adjust if yours differs)
- Hub env (on host): `export HUB=http://10.0.4.2:7860; export API=$HUB/api`

Check hub from inside the container

```bash
docker exec \
  -e TTSHUB_API_BASE=$API \
  youtube-summarizer-bot \
  python3 - <<'PY'
import asyncio
from modules.tts_hub import TTSHubClient

async def main():
    c = TTSHubClient.from_env()
    favs = await c.fetch_favorites()
    cat = await c.fetch_catalog()
    print('favorites=', len(favs), 'voices=', len((cat or {}).get('voices') or []))
    if favs:
        slug = favs[0].get('slug')
        r = await c.synthesise('hello from NAS', favorite_slug=slug)
        print('synth_ok=', bool(r.get('audio_bytes')), 'path/url=', r.get('path') or r.get('url'))
asyncio.run(main())
PY
```

End‑to‑end AI↔AI recap (gender mapping, collision avoidance)

```bash
docker exec \
  -e TTSHUB_API_BASE=$API \
  -e OLLAMA_AI2AI_TTS_GENDER_FROM_PERSONA=1 \
  youtube-summarizer-bot \
  python3 - <<'PY'
import os, asyncio
from pathlib import Path
from modules.tts_hub import TTSHubClient, filter_catalog_voices
from modules.telegram_handler import YouTubeTelegramBot

async def pick_gender_envs():
    c = TTSHubClient.from_env()
    cat = await c.fetch_catalog()
    m = (filter_catalog_voices(cat, gender='male') or [])
    f = (filter_catalog_voices(cat, gender='female') or [])
    if m:
        os.environ['OLLAMA_AI2AI_TTS_VOICE_MALE'] = m[0].get('id','')
    if f:
        os.environ['OLLAMA_AI2AI_TTS_VOICE_FEMALE'] = f[0].get('id','')

async def run():
    await pick_gender_envs()
    bot = YouTubeTelegramBot('dummy', [0])
    session = {
        'persona_a': 'Sun Tzu (M)',
        'persona_b': 'Cleopatra (F)',
        'ai2ai_transcript': [
            {'speaker':'A','text':'Strategy begins with information.'},
            {'speaker':'B','text':'And leadership requires action.'},
        ],
        'ai2ai_model_a': 'testA',
        'ai2ai_model_b': 'testB',
    }
    p = await bot._ollama_ai2ai_generate_audio(12345, session)
    print('mp3:', p, 'exists:', Path(p).exists())

asyncio.run(run())
PY
```

View selection logs and collision handling

```bash
docker logs -f youtube-summarizer-bot | rg 'AI↔AI TTS voice|skip collision|Gendered TTS'
```

Expected

- An MP3 at `exports/ai2ai_chat_*.mp3` created inside the container
- Logs like `A resolved: male via env -> …`, `B resolved: female via env -> …`, and `skip collision …` only if the same voice was about to be re‑used

## Queue Worker (Auto‑run)

- The image now starts both the Telegram bot and a background TTS queue watcher by default.
- Entrypoint: `/app/entrypoint.sh` runs:
  - `python3 telegram_bot.py`
  - `python3 tools/drain_tts_queue.py --watch`
- Environment toggles:
  - `ENABLE_TTS_QUEUE_WORKER=1` (default) – set to `0` to disable the watcher
  - `TTS_QUEUE_INTERVAL=30` – poll interval in seconds
  - `POSTGRES_ONLY=true` – worker skips any SQLite metadata update
- Manual usage inside the container:
  - One‑shot: `python3 tools/drain_tts_queue.py`
  - Watcher: `python3 tools/drain_tts_queue.py --watch --interval 15`

## Deployment Notes

- Because the entrypoint and Dockerfile changed to include the watcher, you must rebuild the image once:
  - `docker build -t ytv2-nas:web-ingest .` (or a new tag like `ytv2-nas:web-ingest-watcher`)
  - Recreate the container in Portainer using the rebuilt tag.
- No compose change is required; a single container runs both processes.

## Future Considerations

- Build a queue worker to drain `data/tts_queue/` when the hub comes online.
- Add CLI or dashboard controls for queued jobs (view, retry, purge).
- Extract provider selection from Telegram into a reusable function for CLI/batch jobs.
- Expose config for default voices/accents (currently hard-coded fallback voice when using OpenAI).
- Consider caching hub catalog responses to avoid fetching on every `/tts` invocation.
 - Optional env toggles to select between Minimal (favorites‑only), Compact (reduced header), and Full catalog UI for `/tts`.
 - Add a simple “Close” button to dismiss the picker explicitly when testing is done.

## Troubleshooting

- Clicks don’t register
  - Ensure the picker was created after the latest restart (old keyboards won’t use compact callback keys).
  - Check logs for `🔔 Callback received:` and `🎛️ TTS callback:` lines.

- Local hub unreachable
  - Confirm `TTSHUB_API_BASE` is set in the container env and that the NAS can reach the hub URL.
  - Use the fallback prompt to queue or switch to OpenAI.

- Favorites toggle does nothing
  - If no favorites match the active catalog, the picker now flips to “All voices” automatically and shows an informational hint.
- Queue not draining
  - Confirm `ENABLE_TTS_QUEUE_WORKER=1` and that logs show the worker PID.
  - Verify jobs appear in `/app/data/tts_queue/` inside the container.
  - If local hub was down when jobs were queued, ensure `TTSHUB_API_BASE` is reachable again.
