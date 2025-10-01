# Remote Access & Frontend Bridge Notes

## WireGuard Personal VPN
- 2025-09-30: WireGuard server enabled on AS5304T (UDP 51820). Port forwarded on Eero.
- Clients can connect using configs stored under NAS VPN Server app (Privilege → WireGuard Peer).
- Default tunnel subnet: `10.0.4.0/24`. Client config example:
  ```
  [Interface]
  Address = 10.0.4.2/32
  DNS = 1.1.1.1

  [Peer]
  PublicKey = <NAS public key>
  Endpoint = 24.66.251.193:51820
  AllowedIPs = 10.0.4.0/24, 192.168.4.0/24
  PersistentKeepalive = 25
  ```
- SSH available via `ssh -p 22 user@192.168.4.54` when tunnel active.

## Ngrok Bridge for Render (2025-10-01)
- Container run on NAS: `ngrok/ngrok:latest` with reserved domain `chief-inspired-lab.ngrok-free.app`.
- Command (detached) installed via:
  ```
  docker run -d \
    --name ngrok-dashboard \
    --add-host=host.docker.internal:host-gateway \
    --restart unless-stopped \
    -e NGROK_AUTHTOKEN=<token> \
    ngrok/ngrok:latest \
    http \
      --domain=chief-inspired-lab.ngrok-free.app \
  http://host.docker.internal:6452
  ```
- Basic auth removed; reprocess still guarded by `X-Reprocess-Token`.
- Added CORS support in `telegram_bot.py`:
  - `send_cors_headers()` now covers `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods`, and headers `Authorization`, `Content-Type`, `ngrok-skip-browser-warning`.
  - `do_OPTIONS` returns 204, satisfying ngrok/browser preflight.
  - Applied to `/api/metrics`, `/api/report-events`, `/api/reprocess`, `/api/reports`, `/api/reports?latest=true`, `/api/report/<id>`, and `/api/config`.
  - `/api/reports?latest=true` now returns `{ "report": ... }` (single item) for the polling fallback.

## Frontend Integration (Render)
- Environment variables in Render:
  - `NGROK_BASE_URL = https://chief-inspired-lab.ngrok-free.app`
  - `NGROK_BASIC_USER`, `NGROK_BASIC_PASS` retained for parity (not used now but safe).
- In `dashboard_v3_template.html` from YTV2-Dashboard repo, `window.NAS_CONFIG` populated with env vars.
- `static/dashboard_v3.js` updates:
  - `initNasConfig()` loads base URL and credentials.
  - `nasFetch()` prefixes `/api/...` routes with ngrok base, sets `Authorization` header (if supplied) and `ngrok-skip-browser-warning` header.
  - `buildSseUrl()` appends `ngrok-skip-browser-warning=true` to query for EventSource.
  - Metrics polling (`fetchMetrics`), reports latest fetch, reprocess POST now routed via helper.
  - Audio playback reverted to use relative `/exports/...` paths so MP3s stream from Render.
  - SSE reuses same base; toasts fire for `report-synced`, `audio-synced`, `reprocess-*` events.

## Testing Checklist (2025-10-01)
- `curl https://chief-inspired-lab.ngrok-free.app/api/metrics` → JSON + CORS headers.
- Dashboard pills populate after reload (no CORS errors).
- SSE stream continues over ngrok (EventSource connected).
- Reprocess action prompts user for X-Reprocess-Token; request returns 202, SSE emits lifecycle events.
- Audio playback works both on NAS (LAN/WireGuard) and Render (ngrok not used for audio).

## Ongoing Maintenance
- Rotate ngrok token/password periodically (update Render env + rerun container).
- Monitor ngrok usage (free tier 1 GB/month; metrics + SSE are lightweight).
- Future option: replace ngrok with Cloudflare Tunnel or similar for long-term public access.
