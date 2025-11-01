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
    ],
    "tech": [
        "ai",
        "machine learning",
        "robot",
        "chip",
        "gadget",
        "hardware",
        "software",
        "electronics",
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
            "Use a bold cinematic collage with neon accents, floating UI diagrams, and depth of field. "
            "No text or logos."
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
        if any(token in lowered_topics for token in CATEGORY_KEYWORDS["spiritual"]):
            return "spiritual"
        if any(token in lowered_topics for token in CATEGORY_KEYWORDS["tech"]):
            return "tech"
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


async def maybe_generate_summary_image(content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Attempt to generate a summary illustration. Returns metadata about the image
    or None if generation is disabled or fails.
    """
    if not SUMMARY_IMAGE_ENABLED:
        return None

    tts_base = (os.getenv("TTSHUB_API_BASE") or "").strip()
    if not tts_base:
        logger.debug("summary image skipped: TTSHUB_API_BASE not configured")
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
        logger.warning("summary image generation failed: %s", exc)
        return None

    absolute_url = generation.get("absolute_url") or generation.get("url")
    if not absolute_url:
        logger.warning("summary image failed: hub did not return an image URL")
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
    return metadata


__all__ = ["maybe_generate_summary_image", "SUMMARY_IMAGE_ENABLED"]
