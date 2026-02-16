"""
Automatic1111 Image Generation Service

Provides image generation via Automatic1111's Stable Diffusion WebUI API.
Designed as a fallback provider when Draw Things (M4 Mac) is unavailable.

Default configuration optimized for SDXL Base + Lightning LoRA (8-step).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Default settings - read from environment or use hardcoded defaults
# SDXL + Lightning LoRA optimized for 8-step generation
DEFAULT_MODEL = os.getenv("AUTO1111_MODEL", "sd_xl_base_1.0.safetensors")
DEFAULT_SAMPLER = os.getenv("AUTO1111_SAMPLER", "DPM++ SDE Karras")
DEFAULT_STEPS = int(os.getenv("AUTO1111_STEPS", "8"))
DEFAULT_CFG_SCALE = float(os.getenv("AUTO1111_CFG_SCALE", "2.5"))
DEFAULT_WIDTH = int(os.getenv("AUTO1111_WIDTH", "384"))
DEFAULT_HEIGHT = int(os.getenv("AUTO1111_HEIGHT", "384"))
DEFAULT_LORA = os.getenv("AUTO1111_LORA", "sdxl_lightning_8step_lora")

# Health check cache
_HEALTH_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_HEALTH_CACHE_TTL_SECONDS = 30

# Model info cache
_MODELS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_MODELS_CACHE_TTL_SECONDS = 300


class Auto1111GenerationError(RuntimeError):
    """Raised when Automatic1111 generation fails."""
    pass


async def fetch_auto1111_health(
    base_url: str,
    *,
    ttl: int = _HEALTH_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Check Automatic1111 API health.
    Returns dict with 'reachable' boolean and optional details.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return {"reachable": False, "error": "URL not configured"}

    now = time.monotonic()
    cache_key = base

    if not force_refresh:
        cached = _HEALTH_CACHE.get(cache_key)
        if cached:
            cached_ts, cached_data = cached
            if now - cached_ts <= ttl:
                return cached_data

    # Try a lightweight API call to check health
    url = f"{base}/sdapi/v1/sd-models"
    loop = asyncio.get_running_loop()

    def _call() -> Dict[str, Any]:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return {"reachable": True, "status": "ok"}
            return {"reachable": False, "status": f"http_{resp.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"reachable": False, "error": "connection_refused"}
        except requests.exceptions.Timeout:
            return {"reachable": False, "error": "timeout"}
        except Exception as exc:
            return {"reachable": False, "error": str(exc)[:100]}

    data = await loop.run_in_executor(None, _call)
    _HEALTH_CACHE[cache_key] = (now, data)
    return data


async def fetch_available_models(
    base_url: str,
    *,
    ttl: int = _MODELS_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fetch available models from Automatic1111.
    Returns dict with model list and default model.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return {"models": [], "error": "URL not configured"}

    now = time.monotonic()
    cache_key = base

    if not force_refresh:
        cached = _MODELS_CACHE.get(cache_key)
        if cached:
            cached_ts, cached_data = cached
            if now - cached_ts <= ttl:
                return cached_data

    url = f"{base}/sdapi/v1/sd-models"
    loop = asyncio.get_running_loop()

    def _call() -> Dict[str, Any]:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            models = resp.json() or []
            model_names = [m.get("title", "") for m in models if m.get("title")]
            return {
                "models": model_names,
                "count": len(model_names),
                "default": DEFAULT_MODEL,
            }
        except Exception as exc:
            return {"models": [], "error": str(exc)[:100]}

    data = await loop.run_in_executor(None, _call)
    _MODELS_CACHE[cache_key] = (now, data)
    return data


async def switch_model(
    base_url: str,
    model_name: str,
) -> bool:
    """
    Switch the currently loaded model in Automatic1111.
    Returns True if successful.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return False

    url = f"{base}/sdapi/v1/options"
    loop = asyncio.get_running_loop()

    def _call() -> bool:
        try:
            resp = requests.post(
                url,
                json={"sd_model_checkpoint": model_name},
                timeout=30,
            )
            return resp.status_code == 200
        except Exception:
            return False

    return await loop.run_in_executor(None, _call)


async def generate_image(
    base_url: str,
    prompt: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    sampler: Optional[str] = None,
    cfg_scale: Optional[float] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate an image via Automatic1111's txt2img API.

    Returns dict with:
        - image_bytes: raw PNG bytes
        - seed: the seed used
        - width, height: actual dimensions
        - model: model used
        - duration_sec: generation time
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise Auto1111GenerationError("Automatic1111 base URL is not configured.")

    # CPU mode needs longer timeout (default 10 min for 8-step generation)
    actual_timeout = timeout or float(os.getenv("AUTO1111_TIMEOUT", "600"))

    # Build payload with defaults
    # Add LoRA trigger to prompt if configured (Lightning LoRA needs this)
    lora_name = os.getenv("AUTO1111_LORA", DEFAULT_LORA)
    final_prompt = prompt
    if lora_name:
        # Lightning LoRA works best with low CFG (1-3) and specific trigger
        final_prompt = f"<lora:{lora_name}:1> {prompt}"

    payload: Dict[str, Any] = {
        "prompt": final_prompt,
        "width": width or DEFAULT_WIDTH,
        "height": height or DEFAULT_HEIGHT,
        "steps": steps or DEFAULT_STEPS,
        "cfg_scale": cfg_scale or DEFAULT_CFG_SCALE,
        "sampler_name": sampler or DEFAULT_SAMPLER,
        "seed": seed if seed is not None else -1,
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    else:
        # Default negative prompt for quality
        payload["negative_prompt"] = (
            "blurry, low quality, distorted, watermark, text, logo, "
            "ugly, duplicate, morbid, mutilated, extra fingers, "
            "bad anatomy, bad proportions, missing limbs"
        )

    # Optionally switch model before generation
    model_to_use = model or os.getenv("AUTO1111_MODEL", DEFAULT_MODEL)
    if model_to_use:
        await switch_model(base, model_to_use)

    url = f"{base}/sdapi/v1/txt2img"
    loop = asyncio.get_running_loop()
    start_time = time.monotonic()

    def _call() -> Dict[str, Any]:
        try:
            resp = requests.post(url, json=payload, timeout=actual_timeout)
        except requests.exceptions.RequestException as exc:
            raise Auto1111GenerationError(f"Request to Automatic1111 failed: {exc}") from exc

        if resp.status_code >= 400:
            snippet = resp.text.strip()
            snippet = snippet[:400] if snippet else f"HTTP {resp.status_code}"
            raise Auto1111GenerationError(
                f"Automatic1111 returned {resp.status_code}: {snippet}"
            )

        try:
            return resp.json() or {}
        except ValueError as exc:
            raise Auto1111GenerationError(f"Automatic1111 returned invalid JSON: {exc}") from exc

    data = await loop.run_in_executor(None, _call)
    duration = time.monotonic() - start_time

    # Extract image from response
    images = data.get("images", [])
    if not images:
        raise Auto1111GenerationError("Automatic1111 returned no images")

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(images[0])
    except Exception as exc:
        raise Auto1111GenerationError(f"Failed to decode image: {exc}") from exc

    # Extract metadata from info
    info = data.get("info", {})
    if isinstance(info, str):
        try:
            import json
            info = json.loads(info)
        except Exception:
            info = {}

    return {
        "image_bytes": image_bytes,
        "seed": info.get("seed"),
        "width": payload["width"],
        "height": payload["height"],
        "model": model_to_use,
        "duration_sec": round(duration, 2),
        "steps": payload["steps"],
        "cfg_scale": payload["cfg_scale"],
        "sampler": payload["sampler_name"],
    }


def clear_caches() -> None:
    """Clear all caches."""
    _HEALTH_CACHE.clear()
    _MODELS_CACHE.clear()


__all__ = [
    "Auto1111GenerationError",
    "fetch_auto1111_health",
    "fetch_available_models",
    "switch_model",
    "generate_image",
    "clear_caches",
    "DEFAULT_MODEL",
    "DEFAULT_SAMPLER",
    "DEFAULT_STEPS",
    "DEFAULT_CFG_SCALE",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
]
