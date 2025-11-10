#!/usr/bin/env python3
"""
Backfill original image prompts (AI1/AI2) for existing cards.

Usage examples:

  # Dry-run, both modes, 500 rows (inside container)
  DEBUG_TOKEN=... python scripts/backfill_original_prompts.py --limit 500

  # Apply changes
  DEBUG_TOKEN=... python scripts/backfill_original_prompts.py --limit 500 --apply

  # Target specific video_ids
  DEBUG_TOKEN=... python scripts/backfill_original_prompts.py --ids gZ7QHnBvxEQ,abc123 --apply

You can override the endpoint with --url or set DASHBOARD_BASE_URL.
When running inside the Render container, the script will default to
http://127.0.0.1:$PORT if available.
"""

import argparse
import json
import os
import sys
import urllib.request


def resolve_base_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url.rstrip('/')
    env_url = os.environ.get('DASHBOARD_BASE_URL')
    if env_url:
        return env_url.rstrip('/')
    port = os.environ.get('PORT')
    if port:
        return f"http://127.0.0.1:{port}"
    # Fallback to public URL if known; otherwise require --url
    return os.environ.get('DASHBOARD_PUBLIC_URL', '').rstrip('/') or ''


def call(endpoint: str, token: str, payload: dict, base_url: str) -> dict:
    url = f"{base_url}{endpoint}"
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
        },
        method='POST',
    )
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode('utf-8')
        try:
            return json.loads(body)
        except Exception:
            return {'status': resp.status, 'raw': body}


def main():
    ap = argparse.ArgumentParser(description='Backfill original prompts for AI1/AI2')
    ap.add_argument('--mode', choices=['ai1', 'ai2', 'both'], default='both')
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--ids', type=str, default='', help='Comma-separated video_ids')
    ap.add_argument('--apply', action='store_true', help='Apply changes (default is dry-run)')
    ap.add_argument('--url', type=str, help='Base URL override (e.g., http://127.0.0.1:10000)')
    args = ap.parse_args()

    token = (os.environ.get('DEBUG_TOKEN') or os.environ.get('DASHBOARD_DEBUG_TOKEN') or '').strip()
    if not token:
        print('ERROR: Set DEBUG_TOKEN or DASHBOARD_DEBUG_TOKEN in the environment', file=sys.stderr)
        return 2

    base_url = resolve_base_url(args.url)
    if not base_url:
        print('ERROR: Could not resolve base URL. Set --url, DASHBOARD_BASE_URL or PORT.', file=sys.stderr)
        return 2

    payload: dict = {
        'mode': args.mode,
        'dry_run': (not args.apply),
    }
    if args.ids:
        payload['video_ids'] = [s.strip() for s in args.ids.split(',') if s.strip()]
    else:
        payload['limit'] = args.limit

    try:
        result = call('/api/admin/backfill-original-prompts', token, payload, base_url)
        print(json.dumps(result, indent=2))
        return 0
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8')
        except Exception:
            body = ''
        print(f'HTTP {e.code}: {body}', file=sys.stderr)
        return 1
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
