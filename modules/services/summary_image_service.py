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
from datetime import datetime, timezone
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
    "spiritual_conference": PromptTemplate(
        key="spiritual_conference",
        style_preset="cinematic_warm",
        enhance_mode="local",
        family="spiritual",
        style_hint="Warm conference reverence",
        concept_instructions=(
            "You craft cinematic prompts evoking a sacred gathering.\n"
            "Title: {title}\n"
            "Key message: {summary_excerpt}\n"
            "Describe a warm, reverent interior scene suggesting a spiritual conference or sacred assembly—soft stage lighting, golden tones, light through windows, or rays from above. Subtle architectural hints like a pulpit, hall, or temple-inspired geometry may appear but not literal depictions. No people, no text, no overt religious symbols."
        ),
        prompt_template=(
            "{enhanced_sentence} Rendered in cinematic warm tones with diffused golden lighting and a calm atmosphere. Allow temple-inspired geometry or sacred architectural light motifs, but avoid literal structures or crosses. No figures, lettering, or icons."
        ),
    ),
    # --- New templates ---
    "tech_hardware": PromptTemplate(
        key="tech_hardware",
        style_preset="studio",
        prompt_template=(
            "Studio-lit macro photo of computer components—circuits, chips, cables—rendered with precise reflections and metallic microtextures for '{title}'. "
            "Visualize {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Use glowing accent lights and crisp depth of field. No text or branding."
        ),
    ),
    "reviews_products": PromptTemplate(
        key="reviews_products",
        style_preset="studio",
        prompt_template=(
            "Elegant studio product composition under soft key lighting for '{title}'. "
            "Depict {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Emphasize materials, contours, and reflections with minimal background. No logos or packaging."
        ),
    ),
    "hobbies_creativity": PromptTemplate(
        key="hobbies_creativity",
        style_preset="illustration",
        prompt_template=(
            "Warm tabletop scene of creative tools—brushes, notebooks, crafts—lit with golden hour tones and tactile textures for '{title}'. "
            "Express {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " No people or text."
        ),
    ),
    "home_theater": PromptTemplate(
        key="home_theater",
        style_preset="cinematic_dark",
        prompt_template=(
            "Dark cinematic interior scene with glowing projector light, lens flares, and ambient neon reflections for '{title}'. "
            "Capture {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " No logos, faces, or text."
        ),
    ),
    "vlog_personal": PromptTemplate(
        key="vlog_personal",
        style_preset="portrait",
        prompt_template=(
            "Soft-focus aesthetic of an everyday moment framed with natural light for '{title}'. "
            "Express {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Use shallow depth of field to suggest personality and reflection. No text or direct portraits."
        ),
    ),
    "gaming_cinematic": PromptTemplate(
        key="gaming_cinematic",
        style_preset="cinematic",
        prompt_template=(
            "Dynamic cinematic scene with dramatic lighting, digital armor, glowing UI elements, and particle effects for '{title}'. "
            "Depict {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Rendered as game concept art. No text or logos."
        ),
    ),
    "tech_ai": PromptTemplate(
        key="tech_ai",
        style_preset="cinematic_cool",
        prompt_template=(
            "Surreal macro composition showing neural networks and human silhouettes merging in luminous data flow for '{title}'. "
            "Visualize {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Use glowing synapses and ethereal lighting. No text."
        ),
    ),
    "science_ecology": PromptTemplate(
        key="science_ecology",
        style_preset="watercolor",
        prompt_template=(
            "Vivid ecological collage of forests, oceans, and atmosphere for '{title}'. "
            "Depict {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Use bioluminescent greens and blues, symbolising regeneration. No text or logos."
        ),
    ),
    "health_mindfulness": PromptTemplate(
        key="health_mindfulness",
        style_preset="pastel",
        prompt_template=(
            "Abstract light waves and floating shapes symbolizing peace and emotional healing for '{title}'. "
            "Express {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Rendered in watercolor gradients of mint and lavender. No text or icons."
        ),
    ),
    "history_war": PromptTemplate(
        key="history_war",
        style_preset="cinematic",
        prompt_template=(
            "Cinematic war montage showing dramatic contrast between chaos and courage for '{title}'. "
            "Depict {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence} "
            "Rendered with moody, desaturated lighting, smoke, and faint atmospheric haze. No text, weapons, or gore."
        ),
    ),
    "history_modern_conflict": PromptTemplate(
        key="history_modern_conflict",
        style_preset="retro_poster",
        prompt_template=(
            "Stylized retro poster aesthetic depicting ideological tension and power symbolism for '{title}'. "
            "Convey {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence} "
            "Use muted reds, blues, and textured film grain. No faces, flags, or text."
        ),
    ),
    "default": PromptTemplate(
        key="default",
        prompt_template=(
            'Layered cinematic montage with crisp contrast and depth of field, resembling a magazine editorial photo composite for "{title}". '
            "Capture the core idea: {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "Use dynamic lighting, modern gradients, and rich visual depth. No text or lettering anywhere."
        ),
    ),
    "tech": PromptTemplate(
        key="tech",
        style_preset="cinematic_warm",
        prompt_template=(
            'Neon-lit macro world of glass circuits and glowing grids, with crisp HDR reflections and subtle motion-blur energy trails for "{title}". '
            "Visualize {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "No text or logos."
        ),
    ),
    "innovation": PromptTemplate(
        key="innovation",
        style_preset="cinematic",
        prompt_template=(
            'Mixed-media collage merging sketches, prototypes, and digital holograms under dual warm-cool lighting for "{title}". '
            "Blend {headline} across science and technology. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "No text or logos."
        ),
    ),
    "science": PromptTemplate(
        key="science",
        style_preset="cinematic",
        prompt_template=(
            'Scientific visualization collage combining fluorescent microscopy and cosmic scale imagery, rendered with crisp diagrammatic overlays for "{title}". '
            "Visualize {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "No text or equations."
        ),
    ),
    "business": PromptTemplate(
        key="business",
        style_preset="studio",
        prompt_template=(
            'Architectural abstraction of commerce: glass towers and dynamic charts fused through isometric perspective and reflective copper tones for "{title}". '
            "Translate {headline} into visuals. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "No text or logos."
        ),
    ),
    "maker": PromptTemplate(
        key="maker",
        style_preset="illustration",
        prompt_template=(
            'Warm workshop bench covered with sketches, woodgrain textures, and softly lit metallic tools in cinematic macro focus for "{title}". '
            "Turn {headline} into a hands-on scene. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "No text or lettering."
        ),
    ),
    "education": PromptTemplate(
        key="education",
        style_preset="studio",
        prompt_template=(
            'Abstract visualization of learning: glowing diagrams emerging from open books, rendered in soft natural light and minimalist scholastic design for "{title}". '
            "Translate {headline} into a thoughtful study scene. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "No text or lettering."
        ),
    ),
    "history": PromptTemplate(
        key="history",
        style_preset="retro_film",
        prompt_template=(
            'Cinematic timeline montage blending ancient relics and modern echoes in bronze and indigo hues, illuminated like aged parchment under warm light for "{title}". '
            "Depict {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "No text or banners."
        ),
    ),
    "news": PromptTemplate(
        key="news",
        style_preset="cinematic_warm",
        prompt_template=(
            'Photojournalistic collage with strong diagonals, selective red-blue accents, and cinematic press-light contrast evoking urgency for "{title}". '
            "Convey {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "No text or captions."
        ),
    ),
    "health": PromptTemplate(
        key="health",
        style_preset="portrait",
        prompt_template=(
            'Organic watercolor abstraction with flowing light and botanical translucency, evoking balance and restoration for "{title}". '
            "Express {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "No text or medical icons."
        ),
    ),
    "entertainment": PromptTemplate(
        key="entertainment",
        style_preset="pixar",
        prompt_template=(
            'Vivid cinematic spotlight scene with stage haze, vibrant flares, and playful depth of colour celebrating performance for "{title}". '
            "Translate {headline} into stage-ready visuals. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            "No text or logos."
        ),
    ),
    "sports": PromptTemplate(
        key="sports",
        style_preset="cinematic",
        prompt_template=(
            'Dynamic kinetic illustration freezing athletic motion in streaks of light and dust, rendered with high-contrast chiaroscuro and cool-warm energy for "{title}". '
            "Capture {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
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
    "spiritual_cosmic": PromptTemplate(
        key="spiritual_cosmic",
        style_preset="cinematic_cool",
        enhance_mode="local",
        family="spiritual",
        style_hint="Celestial light and stars",
        concept_instructions=(
            "You craft cosmic illustration prompts.\n"
            "Title: {title}\n"
            "Key message: {summary_excerpt}\n"
            "Describe a peaceful celestial scene with glowing light, stars, clouds, or constellations symbolising divine purpose—no figures or text."
        ),
        prompt_template=(
            "{enhanced_sentence} "
            "Focus on glowing light, celestial gradients, and cosmic calm. No figures, no text."
        ),
    ),
    "spiritual_geometry": PromptTemplate(
        key="spiritual_geometry",
        style_preset="studio",
        enhance_mode="local",
        family="spiritual",
        style_hint="Geometric light symbolism",
        concept_instructions=(
            "You craft abstract geometric prompts.\n"
            "Title: {title}\n"
            "Key message: {summary_excerpt}\n"
            "Describe an abstract geometric composition symbolising divine order through shapes and light, no people or icons."
        ),
        prompt_template=(
            "{enhanced_sentence} "
            "Emphasize radiant geometry, harmonious symmetry, and soft gradients of gold and blue. No figures, no text."
        ),
    ),
}


def _select_template_key(summary_text: str, analysis: Dict[str, Any]) -> str:
    text = summary_text.lower()
    url_hint = ""
    if isinstance(analysis, dict):
        url_hint = str(
            analysis.get("canonical_url")
            or analysis.get("url")
            or analysis.get("source_url")
            or ""
        ).lower()
    # --- Custom routing for new templates ---
    # Tech hardware
    if "tech" in text or any(token in text for token in CATEGORY_KEYWORDS.get("tech", [])):
        tech_hw_terms = ("hardware", "cpu", "gpu", "processor", "motherboard", "ssd", "ram")
        if any(term in text for term in tech_hw_terms):
            return "tech_hardware"
        ai_terms = ("ai", "artificial intelligence", "neural network", "machine learning")
        if any(term in text for term in ai_terms):
            return "tech_ai"
    # Business reviews/products
    if "business" in text or any(token in text for token in CATEGORY_KEYWORDS.get("business", [])):
        reviews_terms = ("review", "product", "comparison", "gadget", "test", "hands-on")
        if any(term in text for term in reviews_terms):
            return "reviews_products"
    # Maker/entertainment hobbies/creativity
    if "maker" in text or "entertainment" in text or any(token in text for token in CATEGORY_KEYWORDS.get("maker", [])) or any(token in text for token in CATEGORY_KEYWORDS.get("entertainment", [])):
        hobbies_terms = ("hobby", "craft", "creative", "art", "painting", "cooking", "baking")
        if any(term in text for term in hobbies_terms):
            return "hobbies_creativity"
    # Entertainment home theater
    if "entertainment" in text or any(token in text for token in CATEGORY_KEYWORDS.get("entertainment", [])):
        home_theater_terms = ("home theater", "projector", "sound system", "av receiver", "dolby", "cinema room")
        if any(term in text for term in home_theater_terms):
            return "home_theater"
        gaming_terms = ("game", "gamer", "rpg", "esport", "console", "controller", "boss fight")
        if any(term in text for term in gaming_terms):
            return "gaming_cinematic"
    # General/personal vlog
    if "general" in text or "personal" in text:
        vlog_terms = ("vlog", "daily life", "personal", "reflection", "journal", "thoughts")
        if any(term in text for term in vlog_terms):
            return "vlog_personal"
    # Fallback to original category heuristics
    for key, keywords in CATEGORY_KEYWORDS.items():
        if any(token in text for token in keywords):
            # --- Custom logic for entertainment: gaming_cinematic
            if key == "entertainment":
                gaming_terms = ("game", "gamer", "rpg", "esport", "console", "controller", "boss fight")
                if any(term in text for term in gaming_terms):
                    return "gaming_cinematic"
            # --- Custom logic for tech: tech_ai
            if key == "tech":
                ai_terms = ("ai", "artificial intelligence", "neural network", "machine learning")
                if any(term in text for term in ai_terms):
                    return "tech_ai"
            # --- Custom logic for science: science_ecology
            if key == "science":
                ecology_terms = ("ecology", "climate", "environment", "sustainability", "forest", "wildlife", "earth")
                if any(term in text for term in ecology_terms):
                    return "science_ecology"
            # --- Custom logic for health: health_mindfulness
            if key == "health":
                mind_terms = ("mental health", "mindfulness", "therapy", "anxiety", "depression", "meditation")
                if any(term in text for term in mind_terms):
                    return "health_mindfulness"
            if key == "spiritual":
                # Conference detection
                conference_terms = (
                    "conference", "general conference", "elder", "apostle", "talk", "address", "devotional", "byu", "ensign"
                )
                if any(term in text for term in conference_terms) or "study/general-conference" in url_hint:
                    return "spiritual_conference"
                # Check for style field in analysis
                style_val = ""
                if isinstance(analysis, dict):
                    style_val = str(analysis.get("style", "")).lower()
                if "cosmic" in style_val:
                    return "spiritual_cosmic"
                if "geometry" in style_val:
                    return "spiritual_geometry"
                # Keyword heuristics for cosmic and geometry
                cosmic_kw = ("heaven", "light", "stars", "celestial", "sky", "eternal", "cosmos")
                geometry_kw = ("principle", "law", "truth", "order", "structure", "balance", "pattern")
                if any(kw in text for kw in cosmic_kw):
                    return "spiritual_cosmic"
                if any(kw in text for kw in geometry_kw):
                    return "spiritual_geometry"
                return "spiritual"
            if key == "history":
                # Subcategory detection for history
                war_keywords = ("ww1", "ww2", "world war", "battle", "revolution", "invasion", "conflict")
                modern_conflict_keywords = ("cold war", "arms race", "nuclear", "propaganda")
                if any(wk in text for wk in war_keywords):
                    return "history_war"
                if any(mk in text for mk in modern_conflict_keywords):
                    return "history_modern_conflict"
                return "history"
            return key
    topics = analysis.get("key_topics") if isinstance(analysis, dict) else None
    if isinstance(topics, list):
        lowered_topics = " ".join(str(t).lower() for t in topics if t)
        # --- Custom routing for new templates (topics) ---
        if "tech" in lowered_topics or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("tech", [])):
            tech_hw_terms = ("hardware", "cpu", "gpu", "processor", "motherboard", "ssd", "ram")
            if any(term in lowered_topics for term in tech_hw_terms):
                return "tech_hardware"
            ai_terms = ("ai", "artificial intelligence", "neural network", "machine learning")
            if any(term in lowered_topics for term in ai_terms):
                return "tech_ai"
        if "business" in lowered_topics or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("business", [])):
            reviews_terms = ("review", "product", "comparison", "gadget", "test", "hands-on")
            if any(term in lowered_topics for term in reviews_terms):
                return "reviews_products"
        if "maker" in lowered_topics or "entertainment" in lowered_topics or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("maker", [])) or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("entertainment", [])):
            hobbies_terms = ("hobby", "craft", "creative", "art", "painting", "cooking", "baking")
            if any(term in lowered_topics for term in hobbies_terms):
                return "hobbies_creativity"
        if "entertainment" in lowered_topics or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("entertainment", [])):
            home_theater_terms = ("home theater", "projector", "sound system", "av receiver", "dolby", "cinema room")
            if any(term in lowered_topics for term in home_theater_terms):
                return "home_theater"
            gaming_terms = ("game", "gamer", "rpg", "esport", "console", "controller", "boss fight")
            if any(term in lowered_topics for term in gaming_terms):
                return "gaming_cinematic"
        if "general" in lowered_topics or "personal" in lowered_topics:
            vlog_terms = ("vlog", "daily life", "personal", "reflection", "journal", "thoughts")
            if any(term in lowered_topics for term in vlog_terms):
                return "vlog_personal"
        for key, keywords in CATEGORY_KEYWORDS.items():
            if any(token in lowered_topics for token in keywords):
                # --- Custom logic for entertainment: gaming_cinematic
                if key == "entertainment":
                    gaming_terms = ("game", "gamer", "rpg", "esport", "console", "controller", "boss fight")
                    if any(term in lowered_topics for term in gaming_terms):
                        return "gaming_cinematic"
                # --- Custom logic for tech: tech_ai
                if key == "tech":
                    ai_terms = ("ai", "artificial intelligence", "neural network", "machine learning")
                    if any(term in lowered_topics for term in ai_terms):
                        return "tech_ai"
                # --- Custom logic for science: science_ecology
                if key == "science":
                    ecology_terms = ("ecology", "climate", "environment", "sustainability", "forest", "wildlife", "earth")
                    if any(term in lowered_topics for term in ecology_terms):
                        return "science_ecology"
                # --- Custom logic for health: health_mindfulness
                if key == "health":
                    mind_terms = ("mental health", "mindfulness", "therapy", "anxiety", "depression", "meditation")
                    if any(term in lowered_topics for term in mind_terms):
                        return "health_mindfulness"
                if key == "spiritual":
                    # Conference detection in topics
                    conference_terms = (
                        "conference", "general conference", "elder", "apostle", "talk", "address", "devotional", "byu", "ensign"
                    )
                    if any(term in lowered_topics for term in conference_terms):
                        return "spiritual_conference"
                    style_val = ""
                    if isinstance(analysis, dict):
                        style_val = str(analysis.get("style", "")).lower()
                    if "cosmic" in style_val:
                        return "spiritual_cosmic"
                    if "geometry" in style_val:
                        return "spiritual_geometry"
                    cosmic_kw = ("heaven", "light", "stars", "celestial", "sky", "eternal", "cosmos")
                    geometry_kw = ("principle", "law", "truth", "order", "structure", "balance", "pattern")
                    if any(kw in lowered_topics for kw in cosmic_kw):
                        return "spiritual_cosmic"
                    if any(kw in lowered_topics for kw in geometry_kw):
                        return "spiritual_geometry"
                    return "spiritual"
                if key == "history":
                    war_keywords = ("ww1", "ww2", "world war", "battle", "revolution", "invasion", "conflict")
                    modern_conflict_keywords = ("cold war", "arms race", "nuclear", "propaganda")
                    if any(wk in lowered_topics for wk in war_keywords):
                        return "history_war"
                    if any(mk in lowered_topics for mk in modern_conflict_keywords):
                        return "history_modern_conflict"
                    return "history"
                return key
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
                if not _pending_job_exists(cid) and _should_enqueue(cid):
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
            if not _pending_job_exists(cid) and _should_enqueue(cid):
                path = image_queue.enqueue(job)
                logger.info("summary image queued (hub offline): %s", path.name)
            else:
                logger.info("summary image enqueue suppressed (recent enqueue) for %s", cid)
        except Exception as qexc:
            logger.warning("summary image could not be queued: %s", qexc)
        return None

    summary_data = content.get("summary") or {}
    analysis = content.get("analysis")
    if not isinstance(analysis, dict):
        analysis = content.get("analysis_json")
        if not isinstance(analysis, dict):
            analysis = {}
    summary_text = ""
    if isinstance(summary_data, dict):
        summary_text = summary_data.get("summary") or summary_data.get("text") or ""
    elif isinstance(summary_data, str):
        summary_text = summary_data
    override_prompt = _extract_override_prompt(content)
    if not summary_text and not override_prompt:
        logger.debug("summary image skipped: no summary text available")
        return None
    if override_prompt:
        prompt = override_prompt.strip()
        prompt_source = "override"
        template_key = None
        template = PROMPT_TEMPLATES.get("default")
        enhanced_sentence = ""
    else:
        template_key = _select_template_key(summary_text, analysis)
        template = PROMPT_TEMPLATES.get(template_key) or PROMPT_TEMPLATES["default"]

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
        prompt_source = template.key
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
        # Enqueue for later if generation path failed (likely offline), unless suppressed
        if _suppress_enqueue():
            logger.info("summary image: generation failed; enqueue suppressed (drain context)")
        else:
            try:
                from modules import image_queue
                job = {
                    "mode": "summary_image",
                    "content": content,
                    "reason": f"gen_failed:{str(exc)[:120]}",
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
    filename = f"{slug}_{timestamp}_{template.key}{template.width}.png"
    output_path = EXPORTS_DIR / filename
    output_path.write_bytes(image_bytes)

    public_url = _derive_public_url(output_path)
    created_at = _now_iso()
    model_name = generation.get("model") or generation.get("engine") or (template.key if template else "default")
    metadata = {
        "path": str(output_path),
        "relative_path": str(output_path.relative_to(Path.cwd())),
        "public_url": public_url,
        "seed": generation.get("seed"),
        "prompt": prompt,
        "prompt_source": prompt_source,
        "template": template.key,
        "preset": template.preset,
        "style_preset": template.style_preset,
        "width": template.width,
        "height": template.height,
        "source_url": absolute_url,
        "model": model_name,
        "created_at": created_at,
        "analysis_variant": build_analysis_variant(
            {
                "prompt": prompt,
                "prompt_source": prompt_source,
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
    logger.info("summary image saved to %s (template=%s)", output_path, template.key)
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


def _extract_override_prompt(content: Dict[str, Any]) -> Optional[str]:
    analysis = _coerce_analysis(content)
    prompt = analysis.get("summary_image_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    prompt = content.get("summary_image_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_analysis_variant(metadata: Dict[str, Any], public_url: Optional[str]) -> Dict[str, Any]:
    variant = {
        "url": public_url or metadata.get("relative_path") or metadata.get("path"),
        "prompt": metadata.get("prompt"),
        "prompt_source": metadata.get("prompt_source"),
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
    if prompt:
        base["summary_image_prompt_last_used"] = prompt
    if model:
        base["summary_image_model"] = model
    base["summary_image_version"] = version
    if selected_url:
        base["summary_image_selected_url"] = selected_url
    return base


__all__ = [
    "maybe_generate_summary_image",
    "SUMMARY_IMAGE_ENABLED",
    "build_analysis_variant",
    "apply_analysis_variant",
]
