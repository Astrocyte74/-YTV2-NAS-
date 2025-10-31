from __future__ import annotations

import asyncio
import logging
import os
import urllib.parse
from typing import Any, Dict, Optional, Tuple

import requests

from llm_config import llm_config, get_quick_cloud_env_model
from modules import ollama_client
from modules.services import cloud_service

_PROMPT_ENHANCER_SYSTEM = (
    "You are an expert prompt engineer for latent diffusion models. "
    "Rewrite the user's concept as a vivid, concise Draw Things prompt. "
    "Include style, lighting, composition, and important details. "
    "Return a single line with no commentary or markdown."
)


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


async def enhance_prompt_local(concept: str, *, model: Optional[str] = None) -> str:
    model_slug = model or _resolve_local_model()
    if not model_slug:
        raise RuntimeError("No local model configured for prompt enhancement.")

    messages = [
        {"role": "system", "content": _PROMPT_ENHANCER_SYSTEM},
        {"role": "user", "content": concept},
    ]

    loop = asyncio.get_running_loop()

    def _call() -> str:
        resp = ollama_client.chat(messages, model_slug, stream=False)
        text = _extract_ollama_text(resp if isinstance(resp, dict) else {})
        if text:
            return text
        raise RuntimeError("Local prompt enhancement returned no text.")

    return await loop.run_in_executor(None, _call)


async def enhance_prompt_cloud(concept: str) -> str:
    resolved = _resolve_cloud_model()
    if not resolved:
        raise RuntimeError("No cloud model configured for prompt enhancement.")
    provider, model = resolved

    messages = [
        {"role": "system", "content": _PROMPT_ENHANCER_SYSTEM},
        {"role": "user", "content": concept},
    ]

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
    width: int,
    height: int,
    steps: int = 20,
    seed: Optional[int] = None,
    negative: Optional[str] = None,
    sampler: Optional[str] = None,
    cfg_scale: Optional[float] = None,
) -> Dict[str, Any]:
    base = (base_api_url or "").strip()
    if not base:
        raise RuntimeError("Hub base URL is not configured.")

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
    }
    if seed is not None:
        payload["seed"] = seed
    if negative:
        payload["negative"] = negative
    if sampler:
        payload["sampler"] = sampler
    if cfg_scale is not None:
        payload["cfgScale"] = cfg_scale

    url = f"{base.rstrip('/')}/telegram/draw"
    loop = asyncio.get_running_loop()

    def _call() -> Dict[str, Any]:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json() or {}

    data = await loop.run_in_executor(None, _call)
    raw_url = data.get("url")
    if isinstance(raw_url, str) and raw_url.strip():
        image_base = _strip_api_suffix(base)
        absolute = urllib.parse.urljoin(image_base.rstrip("/") + "/", raw_url.lstrip("/"))
        data["absolute_url"] = absolute
    return data


__all__ = [
    "enhance_prompt_local",
    "enhance_prompt_cloud",
    "generate_image",
]
