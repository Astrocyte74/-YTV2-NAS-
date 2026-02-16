"""
Flux.2 Klein 4B Image Generation Service via OpenRouter API

Provides image generation via Black Forest Labs' Flux.2 Klein 4B model.
Cost: ~$0.014 per image

Environment variables:
- FLUX2_ENABLED: Set to "1" or "true" to enable (default: disabled)
- OPENROUTER_API_KEY: Your OpenRouter API key
- FLUX2_MODEL: Model name (default: black-forest-labs/flux.2-klein-4b)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "black-forest-labs/flux.2-klein-4b"
DEFAULT_TIMEOUT = 120


class Flux2GenerationError(RuntimeError):
    """Raised when Flux.2 generation fails."""
    pass


def is_enabled() -> bool:
    """Check if Flux.2 provider is enabled."""
    val = os.getenv("FLUX2_ENABLED", "").strip().lower()
    return val in ("1", "true", "yes", "enabled")


async def fetch_flux2_health() -> Dict[str, Any]:
    """
    Check if Flux.2 is available.
    Returns dict with 'reachable' boolean.
    """
    if not is_enabled():
        return {"reachable": False, "error": "Flux.2 provider is disabled"}
    
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {"reachable": False, "error": "OPENROUTER_API_KEY not set"}
    
    # OpenRouter doesn't have a health endpoint, so just check config
    return {"reachable": True, "status": "configured"}


async def generate_image(
    prompt: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Generate an image via Flux.2 Klein 4B through OpenRouter API.
    
    Returns dict with:
        - image_bytes: raw PNG bytes
        - seed: the seed used (if any)
        - width, height: actual dimensions
        - model: model used
        - duration_sec: generation time
        - cost: cost in dollars
    """
    if not is_enabled():
        raise Flux2GenerationError("Flux.2 provider is disabled (FLUX2_ENABLED not set)")
    
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise Flux2GenerationError("OPENROUTER_API_KEY not configured")
    
    model = os.getenv("FLUX2_MODEL", DEFAULT_MODEL)
    
    # Add NO TEXT instruction to prompt
    no_text_prefix = "IMPORTANT: Generate this image with ABSOLUTELY NO TEXT, NO LETTERS, NO WORDS, NO WRITING, NO TYPOGRAPHY of any kind. "
    final_prompt = no_text_prefix + prompt
    
    loop = asyncio.get_running_loop()
    start_time = time.monotonic()
    
    def _call() -> Dict[str, Any]:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": final_prompt}],
            },
            timeout=timeout,
        )
        
        if response.status_code != 200:
            raise Flux2GenerationError(f"OpenRouter error {response.status_code}: {response.text[:300]}")
        
        return response.json()
    
    try:
        data = await loop.run_in_executor(None, _call)
    except requests.exceptions.RequestException as exc:
        raise Flux2GenerationError(f"Request failed: {exc}") from exc
    
    duration = time.monotonic() - start_time
    
    # Extract image from response
    message = data.get("choices", [{}])[0].get("message", {})
    images = message.get("images", [])
    
    if not images:
        raise Flux2GenerationError("No images in response")
    
    # Get image URL (data URL with base64)
    url = images[0].get("image_url", {}).get("url", "") if isinstance(images[0], dict) else ""
    
    if not url or not url.startswith("data:image"):
        raise Flux2GenerationError("Unexpected image format in response")
    
    # Decode base64
    try:
        base64_data = url.split(",", 1)[1]
        image_bytes = base64.b64decode(base64_data)
    except Exception as exc:
        raise Flux2GenerationError(f"Failed to decode image: {exc}") from exc
    
    # Get cost from usage
    usage = data.get("usage", {})
    cost = usage.get("cost", 0.0)
    
    return {
        "image_bytes": image_bytes,
        "seed": seed,
        "width": width or 1024,
        "height": height or 1024,
        "model": model,
        "duration_sec": round(duration, 2),
        "cost": cost,
        "provider": "flux2",
    }


__all__ = [
    "Flux2GenerationError",
    "is_enabled",
    "fetch_flux2_health",
    "generate_image",
]
