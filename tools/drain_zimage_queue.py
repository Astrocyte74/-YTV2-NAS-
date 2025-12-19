#!/usr/bin/env python3
"""
Drain queued Z-Image jobs.

Reads jobs from data/zimage_queue/, posts to Z-Image Turbo API, and sends the
resulting PNG to Telegram.

Usage:
  python tools/drain_zimage_queue.py               # process all pending once
  python tools/drain_zimage_queue.py --limit 5     # process up to 5
  python tools/drain_zimage_queue.py --watch --interval 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.zimage_queue import QUEUE_DIR  # noqa: E402


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - drain_zimage_queue - %(levelname)s - %(message)s",
    )


def _env_default(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


def _env_bool(name: str, default: str = "false") -> bool:
    return _env_default(name, default).lower() in {"1", "true", "yes", "on"}


def load_job(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logging.error("Failed to read job %s: %s", path.name, exc)
        # Try to salvage by stripping the trailing comma/None markers
        try:
            text = path.read_text(encoding="utf-8")
            # Drop anything after the last closing brace
            fixed = text.rsplit("}", 1)[0] + "}"
            return json.loads(fixed)
        except Exception:
            return None
    except Exception as exc:
        logging.error("Failed to read job %s: %s", path.name, exc)
        return None


def move_job(path: Path, outcome: str) -> None:
    dest = QUEUE_DIR / outcome
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists():
            path.rename(dest / path.name)
    except Exception as exc:
        logging.warning("Could not move job %s to %s: %s", path.name, outcome, exc)


def _style_default() -> str:
    raw = _env_default("ZIMAGE_DEFAULT_STYLE", "Cinematic photo")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts[0] if parts else "Cinematic photo"


def _resolution_default() -> tuple[int, int]:
    width, height = 512, 512
    raw = _env_default("ZIMAGE_DEFAULT_RESOLUTION", "")
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    val = parts[0] if parts else ""
    if "x" in val:
        try:
            w_str, h_str = val.split("x", 1)
            width = int(w_str) or width
            height = int(h_str) or height
        except Exception:
            pass
    else:
        try:
            width = int(_env_default("ZIMAGE_DEFAULT_WIDTH", str(width)))
            height = int(_env_default("ZIMAGE_DEFAULT_HEIGHT", str(height)))
        except Exception:
            width, height = 512, 512
    return width, height


def _zimage_generate_endpoint_order() -> List[str]:
    raw = _env_default("ZIMAGE_GENERATE_ENDPOINT", "generate_ephemeral,generate")
    candidates = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not candidates:
        candidates = ["generate_ephemeral", "generate"]
    order: List[str] = []
    for cand in candidates:
        if cand in {"generate_ephemeral", "ephemeral"}:
            order.append("generate_ephemeral")
        elif cand in {"generate", "persistent", "normal"}:
            order.append("generate")
    if not order:
        order = ["generate_ephemeral", "generate"]
    return list(dict.fromkeys(order))


def _as_int(val: Any) -> Optional[int]:
    try:
        if val is None:
            return None
        if isinstance(val, int):
            return val
        s = str(val).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _as_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


async def process_job(path: Path) -> bool:
    job = load_job(path)
    if not job:
        move_job(path, "failed")
        return False

    prompt = (job.get("prompt") or "").strip()
    chat_id = job.get("chat_id")
    if not prompt or chat_id is None:
        logging.error("Invalid job %s: missing prompt or chat_id", path.name)
        move_job(path, "failed")
        return False

    base = (job.get("base") or _env_default("ZIMAGE_BASE_URL", "")).rstrip("/")
    if not base:
        logging.warning("ZIMAGE_BASE_URL not set; leaving job pending")
        return False

    seed = int(job.get("seed", -1))
    style = (job.get("style") or _style_default()).strip() or "Cinematic photo"
    steps = int(job.get("steps", int(_env_default("ZIMAGE_DEFAULT_STEPS", "7"))))
    try:
        cfg_scale = float(job.get("cfg_scale", _env_default("ZIMAGE_DEFAULT_CFG", "0.0")))
    except Exception:
        cfg_scale = 0.0
    width = int(job.get("width", 0)) or _resolution_default()[0]
    height = int(job.get("height", 0)) or _resolution_default()[1]
    lora_id = job.get("lora_id")
    try:
        lora_scale = float(job.get("lora_scale", 1.0))
    except Exception:
        lora_scale = 1.0
    enhance = bool(job.get("enhance"))

    payload = {
        "prompt": prompt,
        "negative_prompt": job.get("negative_prompt") or "blurry, distorted, watermark, logo",
        "style_preset": style,
        "advanced": {
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "use_lora": bool(lora_id),
            "lora_id": lora_id,
            "lora_scale": lora_scale if lora_id else 1.0,
        },
        "stealth": False,
        "model": None,
    }

    post_timeout = httpx.Timeout(120.0)
    get_timeout = httpx.Timeout(60.0)
    data: Dict[str, Any] = {}
    image_bytes: Optional[bytes] = None
    used_endpoint: str = "unknown"
    async with httpx.AsyncClient() as client:
        enable_health = _env_bool("ENABLE_ZIMAGE_HEALTHCHECK", "1")
        try:
            health_timeout = float(_env_default("ZIMAGE_HEALTHCHECK_TIMEOUT", "5"))
        except Exception:
            health_timeout = 5.0
        if enable_health:
            try:
                h = await client.get(f"{base}/health", timeout=health_timeout)
                h_ok = h.status_code == 200 and isinstance(h.json(), dict) and h.json().get("status") == "ok"
                if not h_ok:
                    logging.info("Z-Image health failed; leaving job pending")
                    return False
            except Exception as exc:
                logging.info("Z-Image health unreachable: %s; leaving job pending", exc)
                return False
        if enhance:
            try:
                enh_resp = await client.post(
                    f"{base}/api/enhance_prompt",
                    json={
                        "prompt": prompt,
                        "negative_prompt": payload.get("negative_prompt"),
                        "style_preset": style,
                        "lora_id": lora_id,
                    },
                    timeout=20.0,
                )
                if enh_resp.status_code == 200:
                    enh_json = enh_resp.json()
                    new_prompt = enh_json.get("prompt")
                    if isinstance(new_prompt, str) and new_prompt.strip():
                        payload["prompt"] = new_prompt.strip()
                        prompt = new_prompt.strip()
            except Exception as exc:
                logging.info("Z-Image enhance failed: %s", exc)

        last_error: Optional[str] = None
        for endpoint in _zimage_generate_endpoint_order():
            if endpoint == "generate_ephemeral":
                gen_url = f"{base}/api/generate_ephemeral"
                logging.info("zimage queue: generating via %s", gen_url)
                try:
                    resp = await client.post(gen_url, json=payload, timeout=post_timeout)
                except Exception as exc:
                    logging.warning("POST %s failed: %s", gen_url, exc)
                    return False
                if resp.status_code == 200:
                    image_bytes = resp.content
                    used_endpoint = "generate_ephemeral"
                    data = {
                        "seed": resp.headers.get("X-Zimage-Seed"),
                        "duration_sec": resp.headers.get("X-Zimage-Duration-Sec"),
                        "id": resp.headers.get("X-Zimage-Image-Id"),
                        "width": width,
                        "height": height,
                    }
                    break
                if resp.status_code in (404, 405):
                    last_error = f"{gen_url} returned {resp.status_code}"
                    continue
                last_error = f"{gen_url} returned {resp.status_code}: {resp.text[:200]}"
                break

            gen_url = f"{base}/api/generate"
            logging.info("zimage queue: generating via %s", gen_url)
            try:
                resp = await client.post(gen_url, json=payload, timeout=post_timeout)
            except Exception as exc:
                logging.warning("POST %s failed: %s", gen_url, exc)
                return False
            if resp.status_code != 200:
                last_error = f"{gen_url} returned {resp.status_code}: {resp.text[:200]}"
                break
            try:
                data = resp.json() or {}
            except Exception as exc:
                logging.warning("Invalid JSON from Z-Image: %s", exc)
                move_job(path, "failed")
                return False
            image_url = data.get("image_url")
            if not image_url:
                logging.warning("Z-Image response missing image_url")
                move_job(path, "failed")
                return False
            full_url = image_url if image_url.startswith("http") else f"{base}{image_url}"
            try:
                img_resp = await client.get(full_url, timeout=get_timeout)
                if img_resp.status_code != 200:
                    last_error = f"GET {full_url} returned {img_resp.status_code}"
                    break
                image_bytes = img_resp.content
                used_endpoint = "generate"
                break
            except Exception as exc:
                last_error = f"GET {full_url} failed: {exc}"
                break

        if not image_bytes:
            if last_error:
                logging.warning("Z-Image generation failed: %s", last_error)
            return False

    token = _env_default("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logging.error("TELEGRAM_BOT_TOKEN not set; cannot deliver image")
        return False

    seed_used = seed if seed != -1 else _as_int(data.get("seed"))
    w_used = _as_int(data.get("width")) or width
    h_used = _as_int(data.get("height")) or height
    dur = _as_float(data.get("duration_sec"))
    cap_parts = ["Z-Image"]
    if lora_id:
        cap_parts.append(f"LoRA {lora_id}")
    if seed_used is not None:
        cap_parts.append(f"Seed {seed_used}")
    if w_used and h_used:
        cap_parts.append(f"{w_used}×{h_used}")
    if isinstance(dur, float):
        cap_parts.append(f"{dur:.1f}s")
    if used_endpoint == "generate_ephemeral":
        img_id = str(data.get("id") or "").strip()
        if img_id:
            cap_parts.append(f"ID {img_id}")
    caption = " • ".join(cap_parts)

    tg_url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {"photo": ("zimage.png", image_bytes, "image/png")}
    data_form = {"chat_id": chat_id, "caption": caption}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(tg_url, data=data_form, files=files, timeout=30.0)
            if resp.status_code != 200:
                logging.warning("sendPhoto failed: %s %s", resp.status_code, resp.text[:200])
                return False
    except Exception as exc:
        logging.warning("sendPhoto request failed: %s", exc)
        return False

    move_job(path, "processed")
    logging.info("Delivered Z-Image job %s to chat %s", path.name, chat_id)
    return True


async def drain_once(limit: Optional[int] = None) -> None:
    processed = 0
    for job_path in sorted(QUEUE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if limit is not None and processed >= limit:
            break
        ok = await process_job(job_path)
        if ok:
            processed += 1


async def watch(interval: int) -> None:
    while True:
        await drain_once()
        await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drain Z-Image queue")
    parser.add_argument("--watch", action="store_true", help="Watch mode (poll every interval seconds)")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Max jobs to process once")
    args = parser.parse_args()

    setup_logging()

    loop = asyncio.get_event_loop()
    if args.watch:
        loop.run_until_complete(watch(args.interval))
    else:
        loop.run_until_complete(drain_once(args.limit))


if __name__ == "__main__":
    main()
