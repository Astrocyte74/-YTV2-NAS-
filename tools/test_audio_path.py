#!/usr/bin/env python3
from pathlib import Path
import sys

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.services.audio_path import build_audio_path

cases = [
    ({"video_id": "1nx65a1", "id": "yt:1nx65a1"}, "/exports/by_video/1nx65a1.mp3"),
    ({"id": "web:abc123"}, "/exports/by_video/abc123.mp3"),
    ({"content_source": "reddit", "file_stem": "t3_xyz"}, "/exports/audio/redditt3_xyz.mp3"),
]

for payload, expect in cases:
    got = build_audio_path(payload)
    print(payload, "->", got, "ok" if got == expect else f"mismatch (expected {expect})")
