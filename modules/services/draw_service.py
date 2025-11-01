from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests

from llm_config import llm_config, get_quick_cloud_env_model
from modules import ollama_client
from modules.services import cloud_service

_PROMPT_ENHANCER_SYSTEM_GENERAL = (
    "You are an expert prompt engineer for Stable Diffusion–style models. "
    "Rewrite the concept as a concise, comma-separated prompt optimised for Draw Things. "
    "Include subject, context, style or medium, lighting, composition, and key adjectives in short descriptive fragments. "
    "Avoid full sentences, commentary, markdown, or negative prompts; return a single line of fragments separated by commas."
)

_PROMPT_ENHANCER_SYSTEM_FLUX = (
    "You are an expert prompt engineer optimising prompts for Flux.1 [schnell]. "
    "Rewrite the concept as one natural-language sentence under 60 words that clearly describes the entire scene. "
    "Explicitly cover foreground, midground, and background (or depth cues); specify mood, lighting, colour palette, and camera perspective. "
    "Mention important materials or special visual effects (for example rain, reflections, glass, or motion blur) when implied. "
    "Return plain text without quotation marks or any markdown."
)


def _prompt_enhancer_system_for_family(family: Optional[str]) -> str:
    if isinstance(family, str) and family.strip().lower() == "flux":
        return _PROMPT_ENHANCER_SYSTEM_FLUX
    return _PROMPT_ENHANCER_SYSTEM_GENERAL


class DrawGenerationError(RuntimeError):
    """Raised when Draw Things generation via the hub fails."""


def _strip_api_suffix(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/api"):
        return url[:-4]
    return url


def _extract_ollama_text(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""

    response = payload.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()

    message = payload.get("message")
    if isinstance(message, dict):
        for key in ("content", "text"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for seg in content:
                if not isinstance(seg, dict):
                    continue
                text = seg.get("text") or seg.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            if parts:
                return "\n".join(parts).strip()

    messages = payload.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if not isinstance(item, dict):
                continue
            if item.get("role") != "assistant":
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

    return ""


def _resolve_local_model() -> Optional[str]:
    env_model = (os.getenv("DRAW_LOCAL_MODEL") or os.getenv("QUICK_LOCAL_MODEL") or "").strip()
    if env_model:
        return env_model
    try:
        provider, model, _ = llm_config.get_model_config("ollama", None)
        if provider == "ollama" and model:
            return model
    except Exception as exc:
        logging.debug("draw_service: local model resolution failed: %s", exc)
    return None


def _resolve_cloud_model() -> Optional[Tuple[str, str]]:
    slug = get_quick_cloud_env_model()
    try:
        if slug:
            provider, model, _ = llm_config.get_model_config(None, slug)
        else:
            provider, model, _ = llm_config.get_model_config("cloud", None)
        if provider and provider != "ollama" and model:
            return provider, model
    except Exception as exc:
        logging.debug("draw_service: cloud model resolution failed: %s", exc)
    return None


def _humanize_label(key: str) -> str:
    if not key:
        return ""
    if key.startswith("flux_"):
        tail = key.split("_", 1)[1] if "_" in key else key
        pretty_tail = _humanize_label(tail)
        return f"Flux · {pretty_tail}"
    segments = [seg.strip() for seg in key.replace("-", "_").split("_") if seg.strip()]
    words: List[str] = []
    for seg in segments:
        lower = seg.lower()
        if lower == "nsfw":
            words.append("NSFW")
        elif seg.isupper():
            words.append(seg)
        else:
            words.append(seg.capitalize())
    return " ".join(words) if words else key


def _normalize_presets(raw: Dict[str, Any]) -> Dict[str, Any]:
    presets_dict = raw.get("presets") or {}
    styles_dict = raw.get("stylePresets") or {}
    negative_dict = raw.get("negativePresets") or {}
    drawthings_info = raw.get("drawthings") if isinstance(raw, dict) else None

    presets_list: List[Dict[str, Any]] = []
    preset_map: Dict[str, Dict[str, Any]] = {}
    for key, value in presets_dict.items():
        entry = dict(value or {})
        entry["key"] = key
        label = entry.get("label")
        if not isinstance(label, str) or not label.strip():
            label = _humanize_label(key)
        entry["label"] = label
        family = entry.get("family") or ("flux" if key.lower().startswith("flux") else "general")
        entry["group"] = str(family).strip().lower() or "general"
        default_size = entry.get("defaultSize") or {}
        entry["default_width"] = default_size.get("width")
        entry["default_height"] = default_size.get("height")
        preset_map[key] = entry

    style_list: List[Dict[str, Any]] = []
    style_map: Dict[str, Dict[str, Any]] = {}
    for key, value in styles_dict.items():
        if isinstance(value, dict):
            label = value.get("label") or _humanize_label(key)
            prompt = value.get("tags") or value.get("prompt") or value.get("value")
        else:
            label = _humanize_label(key)
            prompt = value
        entry = {
            "key": key,
            "label": label,
            "prompt": prompt,
            "desc": value.get("desc") if isinstance(value, dict) else None,
        }
        style_map[key] = entry

    negative_list: List[Dict[str, Any]] = []
    negative_map: Dict[str, Dict[str, Any]] = {}
    for key, value in negative_dict.items():
        if isinstance(value, dict):
            label = value.get("label") or _humanize_label(key)
            prompt = value.get("tags") or value.get("prompt") or value.get("value")
        else:
            label = _humanize_label(key)
            prompt = value
        entry = {
            "key": key,
            "label": label,
            "prompt": prompt,
        }
        negative_map[key] = entry

    orders = raw.get("order") or {}

    def _ordered_list(mapping: Dict[str, Dict[str, Any]], order_keys: Optional[List[str]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen: set = set()
        if order_keys:
            for key in order_keys:
                entry = mapping.get(key)
                if entry:
                    result.append(entry)
                    seen.add(key)
        for key, entry in mapping.items():
            if key not in seen:
                result.append(entry)
        return result

    presets_list = _ordered_list(preset_map, orders.get("presets"))
    style_list = _ordered_list(style_map, orders.get("stylePresets"))
    negative_list = _ordered_list(negative_map, orders.get("negativePresets"))

    defaults = raw.get("defaults") or {}

    return {
        "presets": presets_list,
        "style_presets": style_list,
        "negative_presets": negative_list,
        "defaults": defaults,
        "orders": orders,
        "maps": {
            "preset": preset_map,
            "style": style_map,
            "negative": negative_map,
        },
        "drawthings": drawthings_info if isinstance(drawthings_info, dict) else None,
        "raw": raw,
    }


_PRESET_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_PRESET_CACHE_TTL_SECONDS = 300

_HEALTH_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_HEALTH_CACHE_TTL_SECONDS = 30

async def fetch_presets(base_api_url: str, *, ttl: int = _PRESET_CACHE_TTL_SECONDS, force_refresh: bool = False) -> Dict[str, Any]:
    base = (base_api_url or "").strip()
    if not base:
        raise RuntimeError("Hub base URL is not configured.")

    now = time.time()
    cached = _PRESET_CACHE.get(base)
    if not force_refresh and cached and now - cached[0] <= ttl:
        return cached[1]

    url = f"{base.rstrip('/')}/telegram/presets"
    loop = asyncio.get_running_loop()

    def _call() -> Dict[str, Any]:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        raw = resp.json() or {}
        return _normalize_presets(raw)

    data = await loop.run_in_executor(None, _call)
    _PRESET_CACHE[base] = (now, data)
    return data


async def fetch_drawthings_health(
    base_api_url: str,
    *,
    ttl: int = _HEALTH_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    base = (base_api_url or "").strip()
    if not base:
        raise RuntimeError("Hub base URL is not configured.")

    now = time.time()
    cached = _HEALTH_CACHE.get(base)
    if not force_refresh and cached and now - cached[0] <= ttl:
        return cached[1]

    url = f"{base.rstrip('/')}/drawthings/health"
    loop = asyncio.get_running_loop()

    def _call() -> Dict[str, Any]:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        return resp.json() or {}

    data = await loop.run_in_executor(None, _call)
    _HEALTH_CACHE[base] = (now, data)
    return data


def clear_preset_cache() -> None:
    _PRESET_CACHE.clear()


async def enhance_prompt_local(
    concept: str,
    *,
    model: Optional[str] = None,
    family: Optional[str] = None,
    style_hint: Optional[str] = None,
) -> str:
    model_slug = model or _resolve_local_model()
    if not model_slug:
        raise RuntimeError("No local model configured for prompt enhancement.")

    system_prompt = _prompt_enhancer_system_for_family(family)
    messages = [{"role": "system", "content": system_prompt}]
    if style_hint:
        messages.append({"role": "user", "content": f"Style preference: {style_hint}"})
    messages.append({"role": "user", "content": concept})

    loop = asyncio.get_running_loop()

    def _call() -> str:
        resp = ollama_client.chat(messages, model_slug, stream=False)
        text = _extract_ollama_text(resp if isinstance(resp, dict) else {})
        if text:
            return text
        raise RuntimeError("Local prompt enhancement returned no text.")

    return await loop.run_in_executor(None, _call)


async def enhance_prompt_cloud(
    concept: str,
    *,
    family: Optional[str] = None,
    style_hint: Optional[str] = None,
) -> str:
    resolved = _resolve_cloud_model()
    if not resolved:
        raise RuntimeError("No cloud model configured for prompt enhancement.")
    provider, model = resolved

    system_prompt = _prompt_enhancer_system_for_family(family)
    messages = [{"role": "system", "content": system_prompt}]
    if style_hint:
        messages.append({"role": "user", "content": f"Style preference: {style_hint}"})
    messages.append({"role": "user", "content": concept})

    loop = asyncio.get_running_loop()

    def _call() -> str:
        return cloud_service.chat(messages, provider=provider, model=model).strip()

    text = await loop.run_in_executor(None, _call)
    if not text:
        raise RuntimeError("Cloud prompt enhancement returned no text.")
    return text


async def generate_image(
    base_api_url: str,
    prompt: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    negative: Optional[str] = None,
    sampler: Optional[str] = None,
    cfg_scale: Optional[float] = None,
    preset: Optional[str] = None,
    style_preset: Optional[str] = None,
    negative_preset: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    base = (base_api_url or "").strip()
    if not base:
        raise RuntimeError("Hub base URL is not configured.")

    payload: Dict[str, Any] = {
        "prompt": prompt,
    }
    if width is not None:
        payload["width"] = width
    if height is not None:
        payload["height"] = height
    if steps is not None:
        payload["steps"] = steps
    if seed is not None:
        payload["seed"] = seed
    if negative:
        payload["negative"] = negative
    if sampler:
        payload["sampler"] = sampler
    if cfg_scale is not None:
        payload["cfgScale"] = cfg_scale
    if preset:
        payload["preset"] = preset
    if style_preset:
        payload["stylePreset"] = style_preset
    if negative_preset:
        payload["negativePreset"] = negative_preset
    if model:
        payload["model"] = model

    url = f"{base.rstrip('/')}/telegram/draw"
    loop = asyncio.get_running_loop()

    def _call() -> Dict[str, Any]:
        try:
            resp = requests.post(url, json=payload, timeout=120)
        except requests.exceptions.RequestException as exc:
            raise DrawGenerationError(f"Request to hub failed: {exc}") from exc

        if resp.status_code >= 400:
            snippet = resp.text.strip()
            snippet = snippet[:400] if snippet else f"HTTP {resp.status_code}"
            raise DrawGenerationError(
                f"Hub returned {resp.status_code}: {snippet}"
            )
        try:
            return resp.json() or {}
        except ValueError as exc:
            raise DrawGenerationError(f"Hub returned invalid JSON: {exc}") from exc

    data = await loop.run_in_executor(None, _call)
    raw_url = data.get("url")
    if isinstance(raw_url, str) and raw_url.strip():
        image_base = _strip_api_suffix(base)
        absolute = urllib.parse.urljoin(image_base.rstrip("/") + "/", raw_url.lstrip("/"))
        data["absolute_url"] = absolute
    return data


__all__ = [
    "DrawGenerationError",
    "clear_preset_cache",
    "fetch_presets",
    "fetch_drawthings_health",
    "enhance_prompt_local",
    "enhance_prompt_cloud",
    "generate_image",
]
