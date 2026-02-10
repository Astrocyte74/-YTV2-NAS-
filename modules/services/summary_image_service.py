"""
Summary Image Generation Service

Builds prompt templates, optionally leverages the local prompt enhancer, and
invokes an image provider (Draw Things hub or Z-Image) to render a square
illustration for a summary.

Designed to be non-blocking and fail-safe: if anything goes wrong we simply
return None and the caller continues without a generated image.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
import urllib.parse
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from modules.services import draw_service

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


SUMMARY_IMAGE_ENABLED = _env_flag("SUMMARY_IMAGE_ENABLED", "false")
EXPORTS_DIR = Path(os.getenv("SUMMARY_IMAGE_EXPORT_DIR", "exports/images")).resolve()
DEFAULT_SIZE = int(os.getenv("SUMMARY_IMAGE_SIZE", "384"))
DEFAULT_PRESET = os.getenv("SUMMARY_IMAGE_PRESET", "flux_balanced")
DEFAULT_STYLE = os.getenv("SUMMARY_IMAGE_STYLE", "cinematic_warm")

_SUMMARY_IMAGE_ZIMAGE_DEFAULT_STYLE = "Cinematic photo"
_SUMMARY_IMAGE_ZIMAGE_DEFAULT_STEPS = 7
_SUMMARY_IMAGE_ZIMAGE_DEFAULT_CFG_SCALE = 0.0
_SUMMARY_IMAGE_ZIMAGE_NEGATIVE_PROMPT = "blurry, distorted, watermark, logo, text, lettering"

_ZIMAGE_HEALTH_CACHE: Dict[str, Tuple[float, bool]] = {}


def _csv_list_env(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    value = str(raw or "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _summary_image_providers() -> List[str]:
    """
    Return provider preference order for summary images.
    Supported providers: drawthings, zimage
    """
    raw = _csv_list_env("SUMMARY_IMAGE_PROVIDERS", "drawthings")
    normalized: List[str] = []
    for entry in raw:
        key = entry.strip().lower()
        if key in {"draw", "drawthings", "hub"}:
            key = "drawthings"
        elif key in {"zimage", "z-image"}:
            key = "zimage"
        else:
            continue
        if key not in normalized:
            normalized.append(key)
    return normalized or ["drawthings"]


def _map_style_preset_to_zimage(style_preset: Optional[str]) -> str:
    s = (style_preset or "").strip().lower()
    if not s or s == "none":
        return _SUMMARY_IMAGE_ZIMAGE_DEFAULT_STYLE
    if "anime" in s or "manga" in s:
        return "Anime"
    if "product" in s or "render" in s:
        return "Product render"
    if any(
        token in s
        for token in (
            "illustration",
            "watercolor",
            "pastel",
            "retro",
            "poster",
            "pixar",
            "cartoon",
            "comic",
            "sketch",
            "vector",
        )
    ):
        return "Digital illustration"
    return "Cinematic photo"


def _zimage_generate_endpoint_order() -> List[str]:
    raw = _csv_list_env("ZIMAGE_GENERATE_ENDPOINT", "generate_ephemeral,generate")
    order: List[str] = []
    for entry in raw:
        key = entry.strip().lower()
        if key in {"generate_ephemeral", "ephemeral"}:
            order.append("generate_ephemeral")
        elif key in {"generate", "persistent", "normal"}:
            order.append("generate")
    if not order:
        order = ["generate_ephemeral", "generate"]
    return list(dict.fromkeys(order))


async def _zimage_is_reachable(
    base_url: str,
    *,
    ttl: int = 30,
    timeout: float = 5.0,
) -> bool:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return False

    now = time.monotonic()
    cached = _ZIMAGE_HEALTH_CACHE.get(base)
    if cached and ttl > 0:
        cached_ts, cached_ok = cached
        if now - cached_ts <= ttl:
            return cached_ok

    # Try /health endpoint (Z-Image has this at root, not /api/health)
    # Remove /api suffix if present for health check
    health_base = base[:-4] if base.endswith("/api") else base
    url = f"{health_base}/health"
    loop = asyncio.get_running_loop()

    def _call() -> bool:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                try:
                    data = resp.json() or {}
                    status = str(data.get("status") or "").strip().lower()
                    return status == "ok"
                except Exception:
                    return False
        except Exception:
            return False

    ok = await loop.run_in_executor(None, _call)
    _ZIMAGE_HEALTH_CACHE[base] = (now, ok)
    return ok


async def _zimage_generate_image(
    base_url: str,
    prompt: str,
    *,
    width: int,
    height: int,
    seed: int = -1,
    steps: int = _SUMMARY_IMAGE_ZIMAGE_DEFAULT_STEPS,
    cfg_scale: float = _SUMMARY_IMAGE_ZIMAGE_DEFAULT_CFG_SCALE,
    style_preset: str = _SUMMARY_IMAGE_ZIMAGE_DEFAULT_STYLE,
    negative_prompt: str = _SUMMARY_IMAGE_ZIMAGE_NEGATIVE_PROMPT,
    recipe_id: Optional[str] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("Z-Image base URL is not configured.")

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "style_preset": style_preset,
        "advanced": {
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "use_lora": False,
            "lora_id": None,
            "lora_scale": 1.0,
        },
        "stealth": False,
        "model": None,
    }

    # Add recipe_id if provided (overrides other settings)
    if recipe_id:
        payload["recipe_id"] = recipe_id
        logger.info("Z-Image using recipe: %s", recipe_id)

    loop = asyncio.get_running_loop()

    def _call() -> Dict[str, Any]:
        last_error: Optional[str] = None
        for endpoint in _zimage_generate_endpoint_order():
            if endpoint == "generate_ephemeral":
                url = f"{base}/api/generate_ephemeral"
                resp = requests.post(url, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    seed_val: Optional[int] = None
                    dur_val: Optional[float] = None
                    try:
                        seed_hdr = (resp.headers.get("X-Zimage-Seed") or "").strip()
                        seed_val = int(seed_hdr) if seed_hdr else None
                    except Exception:
                        seed_val = None
                    try:
                        dur_hdr = (resp.headers.get("X-Zimage-Duration-Sec") or "").strip()
                        dur_val = float(dur_hdr) if dur_hdr else None
                    except Exception:
                        dur_val = None
                    img_id = (resp.headers.get("X-Zimage-Image-Id") or "").strip() or None
                    return {
                        "endpoint": "generate_ephemeral",
                        "source_url": url,
                        "image_bytes": resp.content,
                        "seed": seed_val,
                        "duration_sec": dur_val,
                        "id": img_id,
                        "width": int(width),
                        "height": int(height),
                        "model": None,
                    }
                if resp.status_code in (404, 405):
                    last_error = f"{url} returned {resp.status_code}"
                    continue
                snippet = (resp.text or "").strip()
                snippet = snippet[:400] if snippet else f"HTTP {resp.status_code}"
                raise RuntimeError(f"Z-Image returned {resp.status_code}: {snippet}")

            url = f"{base}/api/generate"
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                snippet = (resp.text or "").strip()
                snippet = snippet[:400] if snippet else f"HTTP {resp.status_code}"
                raise RuntimeError(f"Z-Image returned {resp.status_code}: {snippet}")
            data = resp.json() or {}
            out = dict(data)
            out["endpoint"] = "generate"
            return out

        raise RuntimeError(last_error or "Z-Image did not attempt any generate endpoint")

    data = await loop.run_in_executor(None, _call)
    if isinstance(data, dict) and data.get("endpoint") == "generate_ephemeral":
        return data

    image_url = data.get("image_url")
    if not isinstance(image_url, str) or not image_url.strip():
        raise RuntimeError("Z-Image did not return image_url")

    absolute = urllib.parse.urljoin(base.rstrip("/") + "/", image_url.lstrip("/"))
    out = dict(data)
    out["url"] = image_url
    out["absolute_url"] = absolute
    out.setdefault("width", width)
    out.setdefault("height", height)
    return out


# Removed CATEGORY_KEYWORDS - no longer needed with universal templates
@dataclass
class PromptVariant:
    key: str
    prompt_template: Optional[str] = None
    prompt_suffix: Optional[str] = None
    style_preset: Optional[str] = None
    preset: Optional[str] = None
    enhance_mode: Optional[str] = None
    style_hint: Optional[str] = None
    concept_instructions: Optional[str] = None


@dataclass
class PromptTemplate:
    key: str
    preset: str = DEFAULT_PRESET
    style_preset: str = DEFAULT_STYLE
    width: int = DEFAULT_SIZE
    height: int = DEFAULT_SIZE
    enhance_mode: str = "none"  # none|local|cloud
    style_hint: Optional[str] = None
    family: Optional[str] = None
    concept_instructions: Optional[str] = None
    variants: List[PromptVariant] = field(default_factory=list)
    prompt_template: str = (
        'Stylized 1:1 summary illustration for "{title}". '
        "Capture the core idea: {headline}. "
        "Highlight motifs such as {motifs}. "
        "{enhanced_sentence}"
        "Use a cinematic collage with modern gradients and layered lighting. "
        "No text or lettering anywhere."
    )


# Universal templates - simplified approach replacing 30+ category-specific templates
# AI1 randomly selects between photorealistic and symbolic for variety
# AI2 uses ai2_freestyle for creative/editorial style
# Universal templates - simplified approach replacing 30+ category-specific templates
# AI1 randomly selects between photorealistic and symbolic for variety
# AI2 uses ai2_freestyle for creative/editorial style
PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "photorealistic": PromptTemplate(
        key="photorealistic",
        style_preset="cinematic",
        prompt_template=(
            'Photorealistic image capturing the essence of "{title}". '
            "The scene conveys: {headline} "
            "Use natural lighting, realistic textures, and authentic atmosphere. "
            "No text, logos, or writing in the image."
        ),
    ),
    "symbolic": PromptTemplate(
        key="symbolic",
        style_preset="cinematic",
        prompt_template=(
            'Symbolic visualization of "{title}". '
            "The image represents the essence of: {headline} "
            "Use metaphor, visual storytelling, and conceptual imagery. "
            "Abstract but accessible. No text or letters."
        ),
    ),
    "ai2_freestyle": PromptTemplate(
        key="ai2_freestyle",
        style_preset="cinematic",
        prompt_template=(
            "Create an imaginative editorial-style illustration representing '{title}'. "
            "Capture the essence of {headline} using symbolic storytelling, bold shapes, and expressive lighting. "
            "{enhanced_sentence}"
            "No text or logos; focus on mood and creative freedom."
        ),
        variants=[
            PromptVariant(
                key="surreal_collage",
                prompt_suffix=" Blend surreal collage elements with layered textures and unexpected juxtapositions.",
            ),
            PromptVariant(
                key="minimal_vector",
                style_preset="illustration",
                prompt_suffix=" Render as a clean vector poster with geometric forms and limited colour palette.",
            ),
            PromptVariant(
                key="cinematic_smoke",
                style_preset="cinematic_dark",
                prompt_suffix=" Use dramatic chiaroscuro, drifting smoke, and volumetric light for a moody tableau.",
            ),
        ],
    ),
}






def _select_template_key(
    summary_text: str,
    analysis: Dict[str, Any],
    *,
    source_url: Optional[str] = None,
) -> str:
    """
    Universal template selection - randomly choose between photorealistic and symbolic.
    No complex routing needed - these universal templates work for any content.
    """
    import random
    return random.choice(["photorealistic", "symbolic"])

def _resolve_template_variant(template: PromptTemplate, content_id: str) -> Tuple[PromptTemplate, str, Optional[str]]:
    variant_suffix = ""
    variant_key: Optional[str] = None
    if template.variants:
        choice = random.choice(template.variants)
        variant_key = choice.key
        template = replace(
            template,
            style_preset=choice.style_preset or template.style_preset,
            preset=choice.preset or template.preset,
            enhance_mode=choice.enhance_mode or template.enhance_mode,
            style_hint=choice.style_hint or template.style_hint,
            concept_instructions=choice.concept_instructions or template.concept_instructions,
            prompt_template=choice.prompt_template or template.prompt_template,
        )
        variant_suffix = (choice.prompt_suffix or "").strip()
    return template, variant_suffix, variant_key


def _summary_excerpt(summary_text: str, limit: int = 400) -> str:
    clean = re.sub(r"\s+", " ", summary_text).strip()
    return clean[:limit]


def _first_sentence(summary_text: str) -> str:
    clean = re.sub(r"\s+", " ", summary_text).strip()
    if not clean:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return parts[0] if parts else clean[:200]


def _top_topics(analysis: Dict[str, Any], summary_text: str, max_items: int = 3) -> List[str]:
    topics: List[str] = []
    if isinstance(analysis, dict):
        key_topics = analysis.get("key_topics")
        if isinstance(key_topics, list):
            for item in key_topics:
                if isinstance(item, str):
                    topics.append(item.replace("-", " "))
                elif isinstance(item, dict) and item.get("topic"):
                    topics.append(str(item["topic"]))
                if len(topics) >= max_items:
                    break
    if not topics:
        words = [w.strip(",. ").lower() for w in summary_text.split() if len(w) > 4]
        seen = set()
        for word in words:
            if word not in seen:
                topics.append(word)
                seen.add(word)
            if len(topics) >= max_items:
                break
    return topics


async def _enhance_sentence(template: PromptTemplate, context: Dict[str, Any]) -> Optional[str]:
    if template.enhance_mode == "none":
        return None
    concept = ""
    if template.concept_instructions:
        concept = template.concept_instructions.format(**context)
    else:
        concept = (
            f"Create a short illustration description for {context['title']}. "
            f"Key idea: {context['headline']}. "
            f"Motifs: {context['motifs']}. "
            "Return one sentence, <=75 words, natural language, no lists, no text in scene."
        )
    try:
        if template.enhance_mode == "cloud":
            return await draw_service.enhance_prompt_cloud(
                concept,
                family=template.family,
                style_hint=template.style_hint,
            )
        return await draw_service.enhance_prompt_local(
            concept,
            family=template.family,
            style_hint=template.style_hint,
        )
    except Exception as exc:
        logger.debug("summary image prompt enhancement failed: %s", exc)
        return None


async def _download_image(url: str) -> bytes:
    loop = asyncio.get_running_loop()

    def _fetch() -> bytes:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        return response.content

    return await loop.run_in_executor(None, _fetch)


def _sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]", "_", value or "")
    return cleaned.strip("_") or "summary"


def _derive_public_url(path: Path) -> Optional[str]:
    base = os.getenv("AUDIO_PUBLIC_BASE") or os.getenv("POSTGRES_DASHBOARD_URL")
    if not base:
        return None
    try:
        relative = path.relative_to(Path.cwd())
    except ValueError:
        relative = path
    return f"{base.rstrip('/')}/{str(relative).replace(os.sep, '/')}"


_LAST_IMAGE_GEN: Dict[str, float] = {}
_LAST_IMAGE_ENQ: Dict[str, float] = {}

def _suppress_enqueue() -> bool:
    try:
        return str(os.getenv("SUMMARY_IMAGE_QUEUE_SUPPRESS","0")).strip().lower() in ("1","true","yes","on")
    except Exception:
        return False

def _should_enqueue(cid: str, ttl: float = 300.0) -> bool:
    if not cid:
        return True
    try:
        import time as _t
        now = _t.time()
        last = _LAST_IMAGE_ENQ.get(cid, 0.0)
        if (now - last) < ttl:
            return False
        _LAST_IMAGE_ENQ[cid] = now
        return True
    except Exception:
        return True

def _pending_job_exists(cid: str) -> bool:
    if not cid:
        return False
    try:
        from pathlib import Path
        import json
        qdir = Path('data/image_queue')
        if not qdir.exists():
            return False
        for p in qdir.glob('*.json'):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            content = d.get('content') or {}
            existing = str(content.get('id') or content.get('video_id') or '')
            if existing == cid:
                return True
    except Exception:
        return False
    return False


def _select_zimage_recipe(content: Dict[str, Any], summary_data: Any, analysis: Dict[str, Any]) -> Optional[str]:
    """
    Intelligently select a Z-Image thumbnail recipe based on content analysis.

    Available recipes in the 'thumbnails' group (actual IDs from Z-Image API):
    - Bold & Eye-catching: neon-pop-7e6006b7-thumb, breaking-the-feed-thumb, energy-burst-hero-illustration-thumb
    - Product Reviews: recipe_1768973569544-thumb (Sleek Product), recipe_1767928648857-thumb (Dynamic), recipe_1768631936696-thumb (Luxury), recipe_1768026749167-thumb (Tech Heartbeat)
    - Interviews/Profiles: hero-spotlight-portrait-thumb, ethereal-energy-portrait-thumb
    - Historical/Tribute: recipe_1767928545975-thumb (Tribute Poster)
    """
    import re

    if not isinstance(content, dict):
        return None

    analysis = analysis or {}
    title = (content.get("title") or content.get("metadata", {}).get("title") or "").lower()
    summary_text = ""
    if isinstance(summary_data, dict):
        summary_text = summary_data.get("summary") or summary_data.get("text") or ""
    elif isinstance(summary_data, str):
        summary_text = summary_data

    text = f"{title} {summary_text}".lower()

    # Extract topics from analysis
    topics = []
    if isinstance(analysis, dict):
        if isinstance(analysis.get("topics_json"), dict):
            topics_obj = analysis.get("topics_json", {})
            if isinstance(topics_obj.get("topics"), list):
                topics = [str(t).lower() for t in topics_obj.get("topics", [])[:10]]
        elif isinstance(analysis.get("topics"), list):
            topics = [str(t).lower() for t in analysis.get("topics", [])[:10]]

    topics_text = " ".join(topics)
    combined_text = f"{text} {topics_text}"

    # Product Review detection
    product_keywords = [
        "review", "test", "vs", "versus", "comparison", "unboxing", "hands-on",
        "product", "gadget", "device", "shoe", "camera", "phone", "laptop",
        "rating", "score", "verdict", "recommend", "buy", "price", "value"
    ]
    if any(kw in combined_text for kw in product_keywords):
        # Tech/sleek product
        if any(kw in combined_text for kw in ["tech", "gadget", "device", "hardware", "chip", "cpu"]):
            return "recipe_1768026749167-thumb"  # Visible Tech Heartbeat
        # Luxury product
        if any(kw in combined_text for kw in ["luxury", "premium", "high-end", "expensive", "exclusive"]):
            return "recipe_1768631936696-thumb"  # Luxury Product Unboxing Reveal
        # Dynamic/action product
        if any(kw in combined_text for kw in ["performance", "speed", "action", "dynamic", "fast"]):
            return "recipe_1767928648857-thumb"  # Dynamic Product Showcase
        # Default sleek product
        return "recipe_1768973569544-thumb"  # Sleek Product Advertising

    # Interview/Profile detection
    interview_keywords = [
        "interview", "profile", "portrait", "spotlight", "feature", "conversation",
        "talks with", "sits down", "discusses", "reveals", "shares", "personal"
    ]
    if any(kw in combined_text for kw in interview_keywords):
        # Ethereal/spiritual
        if any(kw in combined_text for kw in ["spiritual", "meditation", "energy", "ethereal", "peaceful"]):
            return "ethereal-energy-portrait-thumb"
        # Hero spotlight
        return "hero-spotlight-7e6006b7-thumb"

    # Historical/Tribute detection
    historical_keywords = [
        "tribute", "historical", "history", "documentary", "wwii", "ww2", "world war",
        "memorial", "legacy", "remember", "anniversary", "commemorate", "in memory"
    ]
    if any(kw in combined_text for kw in historical_keywords):
        return "recipe_1767928545975-thumb"  # Conceptual Tribute Poster Design

    # Bold/Eye-catching detection (default for most content)
    bold_keywords = [
        "breaking", "shocking", "surprising", "revealed", "exposed", "truth",
        "secret", "hidden", "urgent", "alert", "warning", "energy", "burst"
    ]
    if any(kw in combined_text for kw in bold_keywords):
        return "breaking-the-feed-7e6006b7-thumb"

    # Default: Neon Pop Art (works for most content)
    return "neon-pop-7e6006b7-thumb"


async def maybe_generate_summary_image(
    content: Dict[str, Any],
    *,
    mode: str = "ai1",
) -> Optional[Dict[str, Any]]:
    """
    Attempt to generate a summary illustration. Returns metadata about the image
    or None if generation is disabled or fails.
    """
    if not SUMMARY_IMAGE_ENABLED:
        return None

    # Safety: never generate for test scaffolding
    try:
        cid = str(content.get("id") or content.get("video_id") or "")
        title = (content.get("title") or content.get("metadata", {}).get("title") or "").strip()
        if cid.upper().startswith("TEST") or title.lower() == "test":
            logger.debug("summary image skipped: test content (%s / %s)", cid, title)
            return None
    except Exception:
        pass

    mode_key = (mode or "ai1").strip().lower()
    freestyle_mode = mode_key in {"ai2", "freestyle", "ai2_freestyle", "free"}
    image_mode = "ai2" if freestyle_mode else "ai1"

    # Provider selection (priority list): drawthings,zimage
    providers = _summary_image_providers()
    tts_base = (os.getenv("TTSHUB_API_BASE") or "").strip()
    zimage_base = (os.getenv("ZIMAGE_BASE_URL") or "").strip()

    selected_provider: Optional[str] = None
    any_configured = False

    # Quick health probe; if offline, optionally enqueue and return gracefully.
    # Allow drain context to bypass per-call health when a preflight has passed.
    bypass_health = str(os.getenv("SUMMARY_IMAGE_HEALTH_BYPASS","0")).strip().lower() in ("1","true","yes","on")
    health_ttl = 0 if bypass_health else int(os.getenv("SUMMARY_IMAGE_HEALTH_TTL","120") or "120")
    probe_errors: List[str] = []
    for provider in providers:
        if provider == "drawthings":
            if not tts_base:
                continue
            any_configured = True
            if bypass_health:
                selected_provider = "drawthings"
                break
            try:
                health = await draw_service.fetch_drawthings_health(tts_base, ttl=health_ttl)
                if bool((health or {}).get("reachable", False)):
                    selected_provider = "drawthings"
                    break
                probe_errors.append("drawthings:not_reachable")
            except Exception as exc:
                probe_errors.append(f"drawthings:{type(exc).__name__}")
        elif provider == "zimage":
            if not zimage_base:
                continue
            any_configured = True
            if bypass_health:
                selected_provider = "zimage"
                break
            ok = await _zimage_is_reachable(zimage_base, ttl=health_ttl)
            if ok:
                selected_provider = "zimage"
                break
            probe_errors.append("zimage:offline")

    if not selected_provider:
        if not any_configured:
            logger.debug(
                "summary image skipped: no providers configured (set SUMMARY_IMAGE_PROVIDERS and TTSHUB_API_BASE and/or ZIMAGE_BASE_URL)"
            )
            return None
        if _suppress_enqueue():
            logger.info("summary image: all providers offline; enqueue suppressed (drain context)")
            return None
        try:
            from modules import image_queue

            job = {
                "mode": "summary_image",
                "image_mode": image_mode,
                "content": content,
                "reason": "providers_offline:" + ",".join(probe_errors[:4]),
                "providers": providers,
            }
            cid = str(content.get("id") or content.get("video_id") or "")
            if not _pending_job_exists(cid) and _should_enqueue(cid):
                path = image_queue.enqueue(job)
                logger.info("summary image queued (providers offline): %s", path.name)
            else:
                logger.info("summary image enqueue suppressed (recent enqueue) for %s", cid)
        except Exception as qexc:
            logger.warning("summary image could not be queued: %s", qexc)
        return None

    try:
        logger.info(
            "summary image provider selected: %s (order=%s)",
            selected_provider,
            ",".join(providers),
        )
    except Exception:
        pass

    summary_data = content.get("summary") or {}
    analysis = content.get("analysis")
    if not isinstance(analysis, dict):
        analysis = content.get("analysis_json")
        if not isinstance(analysis, dict):
            analysis = {}
    cid = str(content.get("id") or content.get("video_id") or "")
    summary_text = ""
    if isinstance(summary_data, dict):
        summary_text = summary_data.get("summary") or summary_data.get("text") or ""
    elif isinstance(summary_data, str):
        summary_text = summary_data
    source_url = _extract_source_url(content, analysis)
    override_prompt = _extract_override_prompt(content, mode="ai2" if freestyle_mode else "ai1")
    override_requested = bool(override_prompt)

    # Cooldown to avoid repeated generations for the same content id
    if image_mode != "ai2" and not override_requested:
        try:
            import time as _t
            cid = str(content.get("id") or content.get("video_id") or "")
            now = _t.time()
            last = _LAST_IMAGE_GEN.get(cid, 0.0)
            cooldown = float(os.getenv("SUMMARY_IMAGE_COOLDOWN_SECONDS", "900") or "900")  # 15 min default
            if cid and (now - last) < cooldown:
                logger.info("summary image skipped: cooldown active for %s (%.0fs remaining)", cid, cooldown - (now - last))
                return None
        except Exception:
            pass

    if not summary_text and not override_prompt:
        logger.debug("summary image skipped: no summary text available")
        return None
    variant_key: Optional[str] = None
    variant_suffix = ""
    if override_prompt:
        prompt = override_prompt.strip()
        prompt_source = "override"
        template_key = None
        template = PROMPT_TEMPLATES.get("default")
        enhanced_sentence = ""
    else:
        if freestyle_mode:
            template_key = "ai2_freestyle"
        else:
            template_key = _select_template_key(summary_text, analysis, source_url=source_url)
        base_template = PROMPT_TEMPLATES.get(template_key) or PROMPT_TEMPLATES["default"]
        template, variant_suffix, variant_key = _resolve_template_variant(base_template, cid)

        headline = _first_sentence(summary_text)
        topics = _top_topics(analysis, summary_text)
        motifs = ", ".join(dict.fromkeys(topics)) if topics else "key ideas from the summary"
        summary_excerpt = _summary_excerpt(summary_text)
        context = {
            "title": content.get("title") or content.get("metadata", {}).get("title") or "Summary",
            "headline": headline,
            "motifs": motifs,
            "summary_excerpt": summary_excerpt,
        }

        enhanced_sentence = await _enhance_sentence(template, context)
        context["enhanced_sentence"] = (enhanced_sentence or "").strip()

        prompt = template.prompt_template.format(**context).strip()
        if variant_suffix:
            prompt = f"{prompt} {variant_suffix}".strip()
        variant_label = variant_key or "base"
        prompt_source = f"ai2_{variant_label}" if freestyle_mode else template.key
        logger.info(
            "summary image prompt (template=%s variant=%s): %s",
            template.key,
            variant_label,
            prompt,
        )

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if selected_provider == "zimage":
            # Select appropriate recipe based on content analysis
            recipe_id = _select_zimage_recipe(content, summary_data, analysis)
            logger.info("Z-Image recipe selected: %s", recipe_id)

            generation = await _zimage_generate_image(
                zimage_base,
                prompt,
                width=template.width,
                height=template.height,
                style_preset=_map_style_preset_to_zimage(template.style_preset),
                recipe_id=recipe_id,
            )
        else:
            generation = await draw_service.generate_image(
                tts_base,
                prompt,
                width=template.width,
                height=template.height,
                preset=template.preset,
                style_preset=template.style_preset,
            )
    except Exception as exc:
        # Enqueue for later if generation path failed (likely offline), unless suppressed
        if _suppress_enqueue():
            logger.info("summary image: generation failed; enqueue suppressed (drain context)")
        else:
            try:
                from modules import image_queue
                job = {
                    "mode": "summary_image",
                    "image_mode": image_mode,
                    "content": content,
                    "reason": f"{selected_provider or 'provider'}_gen_failed:{str(exc)[:120]}",
                    "providers": providers,
                }
                cid = str(content.get("id") or content.get("video_id") or "")
                if not _pending_job_exists(cid) and _should_enqueue(cid):
                    image_queue.enqueue(job)
                    logger.info("summary image queued after failure: %s", exc)
                else:
                    logger.info("summary image enqueue suppressed (recent enqueue) for %s", cid)
            except Exception:
                logger.warning("summary image generation failed and queueing also failed: %s", exc)
        return None

    image_bytes: Optional[bytes] = None
    if isinstance(generation, dict):
        raw_bytes = generation.get("image_bytes")
        if isinstance(raw_bytes, (bytes, bytearray)):
            image_bytes = bytes(raw_bytes)

    absolute_url = generation.get("absolute_url") or generation.get("url")
    source_url = absolute_url or generation.get("source_url") or ""
    if image_bytes is None:
        if not absolute_url:
            logger.warning("summary image failed: hub did not return an image URL")
            # Enqueue for later if hub produced no URL (transient failure), unless suppressed
            if not _suppress_enqueue():
                try:
                    from modules import image_queue
                    job = {
                        "mode": "summary_image",
                        "image_mode": image_mode,
                        "content": content,
                        "reason": "no_image_url",
                    }
                    cid = str(content.get("id") or content.get("video_id") or "")
                    if not _pending_job_exists(cid) and _should_enqueue(cid):
                        image_queue.enqueue(job)
                        logger.info("summary image queued (no URL returned)")
                except Exception:
                    pass
            return None

        try:
            image_bytes = await _download_image(absolute_url)
        except Exception as exc:
            logger.warning("summary image download failed: %s", exc)
            return None

    content_id = content.get("id") or content.get("video_id") or "summary"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = _sanitize_slug(str(content_id).split(":", 1)[-1])
    prefix = "AI2_" if image_mode == "ai2" else ""
    name_token = variant_key or template.key
    filename = f"{prefix}{slug}_{timestamp}_{name_token}{template.width}.png"
    output_path = EXPORTS_DIR / filename
    output_path.write_bytes(image_bytes)

    public_url = _derive_public_url(output_path)
    created_at = _now_iso()
    model_name = (
        generation.get("model")
        or generation.get("engine")
        or generation.get("device")
        or (template.key if template else "default")
    )
    metadata = {
        "provider": selected_provider,
        "provider_endpoint": generation.get("endpoint"),
        "path": str(output_path),
        "relative_path": str(output_path.relative_to(Path.cwd())),
        "public_url": public_url,
        "seed": generation.get("seed"),
        "prompt": prompt,
        "prompt_source": prompt_source,
        "template": template.key,
        "variant_key": variant_key,
        "image_mode": image_mode,
        "preset": template.preset,
        "style_preset": template.style_preset,
        "width": template.width,
        "height": template.height,
        "source_url": source_url,
        "model": model_name,
        "created_at": created_at,
        "analysis_variant": build_analysis_variant(
            {
                "prompt": prompt,
                "prompt_source": prompt_source,
                "image_mode": image_mode,
                "template": template.key,
                "preset": template.preset,
                "style_preset": template.style_preset,
                "width": template.width,
                "height": template.height,
                "seed": generation.get("seed"),
                "model": model_name,
                "created_at": created_at,
            },
            public_url,
        ),
    }
    logger.info(
        "summary image saved to %s (provider=%s model=%s size=%sx%s template=%s)",
        output_path,
        selected_provider,
        model_name,
        template.width,
        template.height,
        template.key,
    )
    try:
        if cid:
            _LAST_IMAGE_GEN[cid] = __import__('time').time()
    except Exception:
        pass
    return metadata


def _coerce_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_analysis(content: Dict[str, Any]) -> Dict[str, Any]:
    analysis = _coerce_dict(content.get("analysis"))
    if not analysis:
        analysis = _coerce_dict(content.get("analysis_json"))
    return analysis


def _extract_override_prompt(content: Dict[str, Any], *, mode: str = "ai1") -> Optional[str]:
    analysis = _coerce_analysis(content)
    key = "summary_image_prompt" if mode != "ai2" else "summary_image_ai2_prompt"
    prompt = analysis.get(key)
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    prompt = content.get(key)
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def _extract_source_url(content: Dict[str, Any], analysis: Optional[Dict[str, Any]]) -> Optional[str]:
    metadata = _coerce_dict(content.get("metadata"))
    analysis_dict = analysis if isinstance(analysis, dict) else {}
    candidates = [
        content.get("source_url"),
        content.get("url"),
        content.get("video_url"),
        metadata.get("source_url"),
        metadata.get("url"),
        analysis_dict.get("source_url"),
        analysis_dict.get("canonical_url"),
        analysis_dict.get("origin_url"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str):
            trimmed = candidate.strip()
            if trimmed:
                return trimmed
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_analysis_variant(metadata: Dict[str, Any], public_url: Optional[str]) -> Dict[str, Any]:
    variant = {
        "url": public_url or metadata.get("relative_path") or metadata.get("path"),
        "prompt": metadata.get("prompt"),
        "prompt_source": metadata.get("prompt_source"),
        "variant_key": metadata.get("variant_key"),
        "image_mode": metadata.get("image_mode"),
        "template": metadata.get("template"),
        "preset": metadata.get("preset"),
        "style_preset": metadata.get("style_preset"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "seed": metadata.get("seed"),
        "model": metadata.get("model"),
        "created_at": metadata.get("created_at") or _now_iso(),
    }
    # Remove keys with None to keep JSON concise
    return {k: v for k, v in variant.items() if v is not None}


def apply_analysis_variant(
    analysis: Optional[Dict[str, Any]],
    variant_entry: Dict[str, Any],
    *,
    selected_url: Optional[str] = None,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    version: int = 3,
) -> Dict[str, Any]:
    base = _coerce_dict(analysis)
    variants = base.get("summary_image_variants")
    if not isinstance(variants, list):
        variants = []
    # Remove duplicates with same url
    new_variants: List[Dict[str, Any]] = [
        entry for entry in variants if not (isinstance(entry, dict) and entry.get("url") == variant_entry.get("url"))
    ]
    new_variants.append(variant_entry)
    base["summary_image_variants"] = new_variants
    image_mode = variant_entry.get("image_mode")
    if prompt:
        if image_mode == "ai2":
            base["summary_image_ai2_prompt_last_used"] = prompt
        else:
            base["summary_image_prompt_last_used"] = prompt
    if model:
        base["summary_image_model"] = model
    base["summary_image_version"] = version
    if selected_url:
        base["summary_image_selected_url"] = selected_url
    if image_mode == "ai2":
        url = variant_entry.get("url")
        if url:
            base["summary_image_ai2_url"] = url
    return base


__all__ = [
    "maybe_generate_summary_image",
    "SUMMARY_IMAGE_ENABLED",
    "build_analysis_variant",
    "apply_analysis_variant",
]
