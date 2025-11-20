#!/usr/bin/env python3
"""Delete dashboard images with no matching content row."""
import os
import psycopg
import requests

DB = os.environ['DATABASE_URL']
DASH = os.environ['RENDER_DASHBOARD_URL'].rstrip('/')
TOKEN = os.environ.get('SYNC_SECRET') or os.environ.get('INGEST_TOKEN')

with psycopg.connect(DB) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT video_id, summary_image_url FROM content WHERE summary_image_url IS NOT NULL")
        used = cur.fetchall()

used_paths = set()
for _, url in used:
    if url:
        if url.startswith('http'):
            used_paths.add(url.split('/exports/images/')[-1])
        else:
            used_paths.add(url.split('/')[-1])

res = requests.get(f"{DASH}/api/admin/list-images", headers={'Authorization': f'Bearer {TOKEN}'}, timeout=10)
res.raise_for_status()
files = res.json().get('files', [])

deleted = 0
for name in files:
    if name not in used_paths:
        requests.delete(f"{DASH}/api/admin/delete-image/{name}", headers={'Authorization': f'Bearer {TOKEN}'}, timeout=10)
        deleted += 1

print(f"Deleted {deleted} orphaned images")
