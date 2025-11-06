# YouTube Pipeline – Transcripts, Metadata, Limits, and WG Proxy

This note captures the current, working setup for YouTube processing on the NAS, how we avoid 429 throttling, and how to test/operate the flow. Use this as the reference when new Codex agents or contributors come online.

## Overview

- Components
  - Transcripts: `youtube-transcript-api` (no API key).
  - Metadata (title/description/duration/stats): YouTube Data API v3 (official; API key).
  - Fallbacks: `yt-dlp` only when transcript API fails or subtitles are explicitly needed (SAFE client + pacing).
- 429 strategy
  - The NAS IPv4 lane was throttled for transcript endpoints; we proxy ONLY transcript calls via the i9 Mac over WireGuard.
  - A process-level transcript rate limiter + circuit breaker reduces burst and prevents re-triggering throttles.

## Transcripts (primary path)

- Library: `youtube-transcript-api` (instance API on the NAS).
- Proxy (transcript-only)
  - Env: `YT_TRANSCRIPT_PROXY=http://10.0.4.3:8888`
  - The app temporarily sets HTTPS_PROXY/HTTP_PROXY only while calling `youtube-transcript-api`; all other calls remain direct.
- Rate limiter + circuit breaker
  - Env knobs:
    - `YT_TRANSCRIPT_MIN_INTERVAL` (seconds, default `20`): enforced wait between transcript fetches.
    - `YT_TRANSCRIPT_CIRCUIT_THRESHOLD` (default `3`): repeated 429s before pausing.
    - `YT_TRANSCRIPT_COOLDOWN` (seconds, default `600`): pause window after threshold is hit.
  - Behavior: if repeated 429s occur, we auto-pause transcript attempts for the cooldown window to let the lane cool.
- Language order (first match): `en`, `en-GB`, `en-US`, `en-AU`, `en-CA`, `es`, `es-ES`, `es-MX`, `fr`, `fr-FR`, `fr-CA`.

### i9 WireGuard + tinyproxy (transcript proxy)

- WireGuard peer on NAS: `10.0.4.3/32` for the i9.
- i9 proxy: `tinyproxy` bound to `10.0.4.3:8888`.
  - Config (key lines):
    - `Port 8888`
    - `Listen 10.0.4.3`
    - ACL: `Allow 127.0.0.1`, `Allow 10.0.4.0/24` (optionally LAN `Allow 192.168.4.0/24`)
    - `ConnectPort 443`, `ConnectPort 563`
    - Optional: `DisableViaHeader Yes`, `BasicAuth <user> <pass>`
  - Start on i9: `brew services start tinyproxy`
  - Test from NAS container: `HTTPS_PROXY=http://10.0.4.3:8888 python3 tools/test_youtube_transcripts.py --ids dQw4w9WgXcQ --repeat 1`

## Metadata (official API)

- Use the YouTube Data API v3 for metadata (stable, quotaed, key-restricted).
- Env
  - `YT_METADATA_SOURCE=data_api`
  - `YT_API_KEY=<restricted-key>` (IP-restricted to NAS; API-restricted to YouTube Data API v3).
- Quotas: `videos.list` part=snippet,contentDetails,statistics is 1 unit per call.
- This path is separate from transcript endpoints and will not inherit the 429 lane; it’s key/quota-based.

### Testing tools

- Transcript probe (429 detector):
  - `tools/test_youtube_transcripts.py`
  - Example:
    - `python3 tools/test_youtube_transcripts.py --ids dQw4w9WgXcQ,9bZkp7q19f0 --repeat 2 --sleep 1.5 --jitter 0.5`
- Data API probe:
  - `tools/test_youtube_data_api.py`
  - Example:
    - `python3 tools/test_youtube_data_api.py --ids dQw4w9WgXcQ,9bZkp7q19f0`

## yt-dlp fallback (use sparingly)

- Only when transcript API fails or explicit subtitles fetch is needed.
- Recommended envs:
  - `YTDLP_SAFE_MODE=1` (android client)
  - `YTDLP_SLEEP_REQUESTS=2`, `YTDLP_RETRIES=1`
  - `YTDLP_FORCE_STACK=ipv4` (or `ipv6` if egress supports it)
  - Optional: `YTDLP_COOKIES_FILE=/app/config/youtube_cookies.txt`

## Ops toggles

- `YOUTUBE_ACCESS=true|false` — cool-down switch for Telegram YouTube handling.
- Transcript limiter:
  - `YT_TRANSCRIPT_MIN_INTERVAL=20`
  - `YT_TRANSCRIPT_CIRCUIT_THRESHOLD=3`
  - `YT_TRANSCRIPT_COOLDOWN=600`
- Transcript proxy:
  - `YT_TRANSCRIPT_PROXY=http://10.0.4.3:8888` (plain) or `http://user:pass@10.0.4.3:8888` if BasicAuth enabled.

## Why separating metadata and transcripts is safe

- Metadata calls hit `googleapis.com` (key/quota-based). Transcripts fetch timedtext on `youtube.com` (no key). They are independent lanes with independent controls.
- Using the Data API for metadata reduces scraping noise and is more stable than watch-page fallbacks.
- With the transcript proxy, transcripts egress via the i9 (and often IPv6), while metadata continues from the NAS with your restricted key. No practical “flag linking” risk between key usage and non-key transcript lane.

## Troubleshooting

- Transcript OK on Mac, 429 on NAS: likely IPv6 vs IPv4 lane; container had no IPv6 route. Use proxy (above) or enable Docker IPv6 (more invasive).
- Proxy health: tail tinyproxy logs, or run `curl -v --proxy 10.0.4.3:8888 https://example.com/ -I` from NAS.
- Circuit breaker active: app logs show “Transcript fetch paused by circuit breaker (cooldown active)”; wait or reduce `YT_TRANSCRIPT_MIN_INTERVAL` once it cools.
- If BasicAuth is enabled on tinyproxy, update `YT_TRANSCRIPT_PROXY` to include credentials.

## WireGuard: add new peers on NAS (ADM)

- NAS (ADM): VPN Server → Settings → WireGuard → enable + keypair.
- Add peer: Privilege → WireGuard Peer → Create
  - Public key: client’s public key from its WireGuard app.
  - Allowed IPs: unique `/32` (e.g., `10.0.4.3/32`).
  - Persistent keepalive: 25.
- Client (macOS):
  - [Interface] PrivateKey=…, Address=`10.0.4.x/32`, DNS=1.1.1.1
  - [Peer] PublicKey=<NAS public key>, Endpoint=`<WAN-IP>:51820`, AllowedIPs=`10.0.4.0/24, 192.168.4.0/24`, PersistentKeepalive=25

## Next improvements

- Wire Data API metadata into the main pipeline (use `YT_METADATA_SOURCE=data_api`; avoid watch-page scraping unless strictly needed).
- Add a small per-video transcript cache (reduce duplicate fetches on reprocess).
- Optional: set up a tiny proxy auth and `DisableViaHeader Yes` for extra hardening.

