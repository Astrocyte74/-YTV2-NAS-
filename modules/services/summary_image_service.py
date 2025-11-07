"""
Summary Image Generation Service

Builds prompt templates, optionally leverages the local prompt enhancer, and
invokes the Draw Things hub to render a square illustration for a summary.

Designed to be non-blocking and fail-safe: if anything goes wrong we simply
return None and the caller continues without a generated image.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "spiritual": [
        "christ",
        "god",
        "gospel",
        "faith",
        "prophet",
        "scripture",
        "holy",
        "integrity",
        "divine",
        "temple",
        "testament",
        "devotional",
    ],
    "tech": [
        "technology",
        "ai",
        "machine learning",
        "neural",
        "robot",
        "chip",
        "semiconductor",
        "gadget",
        "hardware",
        "software",
        "developer",
        "coding",
        "api",
        "electronics",
        "dirac",
    ],
    "innovation": [
        "innovation",
        "invent",
        "prototype",
        "breakthrough",
        "lab",
        "research",
        "futuristic",
    ],
    "science": [
        "science",
        "biology",
        "physics",
        "chemistry",
        "astronomy",
        "space",
        "galaxy",
        "research",
        "experiment",
        "laboratory",
        "climate",
        "ecology",
        "nature",
        "genetics",
        "microscopy",
        "discovery",
    ],
    "business": [
        "business",
        "market",
        "startup",
        "revenue",
        "finance",
        "strategy",
        "economy",
        "investment",
        "company",
        "sales",
        "leadership",
        "enterprise",
        "merger",
        "growth",
        "valuation",
        "board",
    ],
    "maker": [
        "how-to",
        "tutorial",
        "diy",
        "build",
        "hands-on",
        "workshop",
        "craft",
        "repair",
        "setup",
        "solder",
        "forge",
        "makerspace",
        "installation",
        "guide",
        "project",
    ],
    "education": [
        "education",
        "lecture",
        "class",
        "course",
        "seminar",
        "professor",
        "learning",
        "student",
        "university",
        "school",
        "curriculum",
        "academic",
        "lesson",
    ],
    "history": [
        "history",
        "historic",
        "ancient",
        "timeline",
        "empire",
        "battle",
        "revolution",
        "wwii",
        "ww2",
        "wwi",
        "world war",
        "cold war",
    ],
    "news": [
        "breaking",
        "headline",
        "policy",
        "politics",
        "election",
        "government",
        "news",
        "report",
        "journalism",
        "analysis",
        "world news",
        "press",
    ],
    "health": [
        "health",
        "wellness",
        "fitness",
        "exercise",
        "nutrition",
        "mental health",
        "therapy",
        "medicine",
        "clinical",
        "healthcare",
        "mindfulness",
    ],
    "entertainment": [
        "entertainment",
        "music",
        "film",
        "movie",
        "series",
        "tv",
        "art",
        "creative",
        "performance",
        "culture",
        "festival",
        "animation",
        "vlog",
        "podcast",
        "streaming",
        "cinema",
    ],
    "sports": [
        "sport",
        "team",
        "match",
        "game",
        "season",
        "championship",
        "player",
        "athlete",
        "league",
        "score",
        "tournament",
    ],
}


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
    prompt_template: str = (
        'Stylized 1:1 summary illustration for "{title}". '
        "Capture the core idea: {headline}. "
        "Highlight motifs such as {motifs}. "
        "{enhanced_sentence}"
        "Use a cinematic collage with modern gradients and layered lighting. "
        "No text or lettering anywhere."
    )


# Base templates we support today. Additional categories can be appended later
# without touching the core service code.
PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "default": PromptTemplate(
        key="default",
        prompt_template=(
            'Stylized 1:1 summary illustration for "{title}". '
            "Capture the core idea: {headline}. "
            "Highlight motifs such as {motifs}. "
            "Show a cinematic collage of scenes connected to the topic, rich lighting, layered depth. "
            "No text, no lettering."
        ),
    ),
    "tech": PromptTemplate(
        key="tech",
        style_preset="cinematic_warm",
        prompt_template=(
            'Stylized 1:1 tech briefing illustration for "{title}". '
            "Visualize {headline}. "
            "Highlight motifs such as {motifs}. "
            "Use a bold cinematic collage with neon accents, glass reflections, and crisp HUD overlays. "
            "No text or logos."
        ),
    ),
    "innovation": PromptTemplate(
        key="innovation",
        style_preset="cinematic",
        prompt_template=(
            'Stylized 1:1 innovation collage for "{title}". '
            "Blend {headline} across science and technology. "
            "Highlight motifs such as {motifs}. "
            "Layer luminous circuitry, molecular diagrams, and cosmic gradients with futuristic depth. "
            "No text or logos."
        ),
    ),
    "science": PromptTemplate(
        key="science",
        style_preset="cinematic",
        prompt_template=(
            'Stylized 1:1 science briefing illustration for "{title}". '
            "Visualize {headline}. "
            "Highlight motifs such as {motifs}. "
            "Layer luminous diagrams, molecular structures, and research instrumentation with balanced composition and neutral palette. "
            "No text or equations."
        ),
    ),
    "business": PromptTemplate(
        key="business",
        style_preset="studio",
        prompt_template=(
            'Stylized 1:1 business insight collage for "{title}". '
            "Translate {headline} into visuals. "
            "Highlight motifs such as {motifs}. "
            "Combine skyline silhouettes, data charts, and collaborative workspaces using clean geometry, glass, and chrome materials. "
            "No text or logos."
        ),
    ),
    "maker": PromptTemplate(
        key="maker",
        style_preset="illustration",
        prompt_template=(
            'Stylized 1:1 workshop illustration for "{title}". '
            "Turn {headline} into a hands-on scene. "
            "Highlight motifs such as {motifs}. "
            "Show tools, blueprints, and step-by-step components on a workbench with warm practical lighting and tactile textures. "
            "No text or lettering."
        ),
    ),
    "education": PromptTemplate(
        key="education",
        style_preset="studio",
        prompt_template=(
            'Stylized 1:1 academic insight illustration for "{title}". '
            "Translate {headline} into a thoughtful study scene. "
            "Highlight motifs such as {motifs}. "
            "Show books, chalkboards, diagrams, and study spaces with soft natural light and orderly composition. "
            "No text or lettering."
        ),
    ),
    "history": PromptTemplate(
        key="history",
        style_preset="retro_film",
        prompt_template=(
            'Stylized 1:1 historical timeline collage for "{title}". '
            "Depict {headline}. "
            "Highlight motifs such as {motifs}. "
            "Blend vintage maps, archival artifacts, and layered silhouettes with aged film grain and sepia tones. "
            "No text or banners."
        ),
    ),
    "news": PromptTemplate(
        key="news",
        style_preset="cinematic_warm",
        prompt_template=(
            'Stylized 1:1 news briefing illustration for "{title}". '
            "Convey {headline}. "
            "Highlight motifs such as {motifs}. "
            "Assemble photojournalistic elements—press lights, microphones, city skylines—with bold contrasts and dynamic depth. "
            "No text or captions."
        ),
    ),
    "health": PromptTemplate(
        key="health",
        style_preset="portrait",
        prompt_template=(
            'Stylized 1:1 wellness illustration for "{title}". '
            "Express {headline}. "
            "Highlight motifs such as {motifs}. "
            "Use calming natural light, abstract body forms, and organic gradients to suggest balance and recovery. "
            "No text or medical icons."
        ),
    ),
    "entertainment": PromptTemplate(
        key="entertainment",
        style_preset="pixar",
        prompt_template=(
            'Stylized 1:1 entertainment spotlight illustration for "{title}". '
            "Translate {headline} into stage-ready visuals. "
            "Highlight motifs such as {motifs}. "
            "Layer spotlights, instruments, film reels, and expressive color bursts with cinematic energy. "
            "No text or logos."
        ),
    ),
    "sports": PromptTemplate(
        key="sports",
        style_preset="cinematic",
        prompt_template=(
            'Stylized 1:1 sports highlight illustration for "{title}". '
            "Capture {headline}. "
            "Highlight motifs such as {motifs}. "
            "Freeze dynamic motion trails, stadium lighting, and athletes in action with high-energy contrast and dramatic shadows. "
            "No text or team logos."
        ),
    ),
    "spiritual": PromptTemplate(
        key="spiritual",
        style_preset="watercolor",
        enhance_mode="local",
        family="spiritual",
        style_hint="Watercolor serenity",
        concept_instructions=(
            "You craft cinematic illustration prompts in natural sentences.\n"
            "Title: {title}\n"
            "Key message: {summary_excerpt}\n"
            "Write one sentence (<=75 words) describing a peaceful 1x1 scene that symbolises integrity and divine identity using nature motifs.\n"
            "Mention light, landscape, and palette. No people, no explicit religious icons, no text."
        ),
        prompt_template=(
            "{enhanced_sentence} "
            "Keep it serene with watercolor washes and diffuse light. "
            "Absolutely no human figures, crosses, or lettering."
        ),
    ),
    "spiritual_pastel": PromptTemplate(
        key="spiritual_pastel",
        style_preset="pastel",
        enhance_mode="local",
        family="spiritual",
        style_hint="Minimalist pastel symbolism",
        concept_instructions=(
            "You craft minimalist landscape prompts.\n"
            "Title: {title}\n"
            "Key message: {summary_excerpt}\n"
            "Return a short phrase (<=60 words) for a calm abstract scene that symbolises integrity, devotion, and divine purpose with nature shapes. "
            "Reference light, forms, and palette. No people or overt religious symbols, no text."
        ),
        prompt_template=(
            "{enhanced_sentence} "
            "Rendered as layered pastel shapes with soft grain, warm golds and muted blues. "
            "No figures, no text."
        ),
    ),
}


def _select_template_key(summary_text: str, analysis: Dict[str, Any]) -> str:
    text = summary_text.lower()
    for key, keywords in CATEGORY_KEYWORDS.items():
        if any(token in text for token in keywords):
            if key == "spiritual":
                # prefer pastel variant if content feels devotional, fallback to watercolor otherwise
                return "spiritual"
            return key
    topics = analysis.get("key_topics") if isinstance(analysis, dict) else None
    if isinstance(topics, list):
        lowered_topics = " ".join(str(t).lower() for t in topics if t)
        for key, keywords in CATEGORY_KEYWORDS.items():
            if any(token in lowered_topics for token in keywords):
                return "spiritual" if key == "spiritual" else key
    return "default"


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

async def maybe_generate_summary_image(content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    # Cooldown to avoid repeated generations for the same content id
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

    tts_base = (os.getenv("TTSHUB_API_BASE") or "").strip()
    if not tts_base:
        logger.debug("summary image skipped: TTSHUB_API_BASE not configured")
        return None

    # Quick health probe; if offline, enqueue a job and return gracefully
    # Allow drain context to bypass per-call health when a preflight has passed
    bypass_health = str(os.getenv("SUMMARY_IMAGE_HEALTH_BYPASS","0")).strip().lower() in ("1","true","yes","on")
    health_ttl = 0 if bypass_health else int(os.getenv("SUMMARY_IMAGE_HEALTH_TTL","30") or "30")
    try:
        health = await draw_service.fetch_drawthings_health(tts_base, ttl=health_ttl)
        if not isinstance(health, dict) or not health:
            raise RuntimeError("empty health")
        # Treat non-reachable hub as offline and enqueue for later
        if not bool(health.get("reachable", False)):
            if _suppress_enqueue():
                logger.info("summary image: hub not reachable; enqueue suppressed (drain context)")
                return None
            try:
                from modules import image_queue
                job = {
                    "mode": "summary_image",
                    "content": content,
                    "reason": "hub_not_reachable",
                }
                cid = str(content.get("id") or content.get("video_id") or "")
                if _should_enqueue(cid):
                    path = image_queue.enqueue(job)
                    logger.info("summary image queued (hub not reachable): %s", path.name)
                else:
                    logger.info("summary image enqueue suppressed (recent enqueue) for %s", cid)
            except Exception as qexc:
                logger.warning("summary image could not be queued (not reachable): %s", qexc)
            return None
    except Exception as exc:
        if _suppress_enqueue():
            logger.info("summary image: hub offline; enqueue suppressed (drain context)")
            return None
        try:
            from modules import image_queue
            job = {
                "mode": "summary_image",
                "content": content,
                "reason": f"hub_offline:{str(exc)[:80]}",
            }
            cid = str(content.get("id") or content.get("video_id") or "")
            if _should_enqueue(cid):
                path = image_queue.enqueue(job)
                logger.info("summary image queued (hub offline): %s", path.name)
            else:
                logger.info("summary image enqueue suppressed (recent enqueue) for %s", cid)
        except Exception as qexc:
            logger.warning("summary image could not be queued: %s", qexc)
        return None

    summary_data = content.get("summary") or {}
    analysis = content.get("analysis") or {}
    summary_text = ""
    if isinstance(summary_data, dict):
        summary_text = summary_data.get("summary") or summary_data.get("text") or ""
    elif isinstance(summary_data, str):
        summary_text = summary_data
    if not summary_text:
        logger.debug("summary image skipped: no summary text available")
        return None

    template_key = _select_template_key(summary_text, analysis if isinstance(analysis, dict) else {})
    template = PROMPT_TEMPLATES.get(template_key) or PROMPT_TEMPLATES["default"]

    headline = _first_sentence(summary_text)
    topics = _top_topics(analysis if isinstance(analysis, dict) else {}, summary_text)
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
    logger.info("summary image prompt (template=%s): %s", template.key, prompt)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        generation = await draw_service.generate_image(
            tts_base,
            prompt,
            width=template.width,
            height=template.height,
            preset=template.preset,
            style_preset=template.style_preset,
        )
    except Exception as exc:
        # Enqueue for later if generation path failed (likely offline)
        try:
            from modules import image_queue
            job = {
                "mode": "summary_image",
                "content": content,
                "reason": f"gen_failed:{str(exc)[:120]}",
            }
            image_queue.enqueue(job)
            logger.info("summary image queued after failure: %s", exc)
        except Exception:
            logger.warning("summary image generation failed and queueing also failed: %s", exc)
        return None

    absolute_url = generation.get("absolute_url") or generation.get("url")
    if not absolute_url:
        logger.warning("summary image failed: hub did not return an image URL")
        # Enqueue for later if hub produced no URL (transient failure), unless suppressed
        if not _suppress_enqueue():
            try:
                from modules import image_queue
                job = {
                    "mode": "summary_image",
                    "content": content,
                    "reason": "no_image_url",
                }
                cid = str(content.get("id") or content.get("video_id") or "")
                if _should_enqueue(cid):
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
    filename = f"{slug}_{timestamp}_{template.key}{template.width}.png"
    output_path = EXPORTS_DIR / filename
    output_path.write_bytes(image_bytes)

    public_url = _derive_public_url(output_path)
    metadata = {
        "path": str(output_path),
        "relative_path": str(output_path.relative_to(Path.cwd())),
        "public_url": public_url,
        "seed": generation.get("seed"),
        "prompt": prompt,
        "template": template.key,
        "preset": template.preset,
        "style_preset": template.style_preset,
        "width": template.width,
        "height": template.height,
        "source_url": absolute_url,
    }
    logger.info("summary image saved to %s (template=%s)", output_path, template.key)
    try:
        if cid:
            _LAST_IMAGE_GEN[cid] = __import__('time').time()
    except Exception:
        pass
    return metadata


__all__ = ["maybe_generate_summary_image", "SUMMARY_IMAGE_ENABLED"]
