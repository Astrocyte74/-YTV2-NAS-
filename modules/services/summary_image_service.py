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
import random
import re
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
    "programming": [
        "code",
        "coding",
        "programming",
        "software",
        "devops",
        "tutorial",
        "script",
        "scripting",
        "python",
        "javascript",
        "typescript",
        "golang",
        "rust",
        "kotlin",
        "java",
        "ci/cd",
        "docker",
        "kubernetes",
        "sdk",
        "api",
        "framework",
        "cli",
        "ide",
    ],
}

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


# Base templates we support today. Additional categories can be appended later
# without touching the core service code.
PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "entertainment_lordoftherings": PromptTemplate(
        key="entertainment_lordoftherings",
        style_preset="cinematic_dark",
        prompt_template=(
            "Epic fantasy composition inspired by The Lord of the Rings for '{title}', depicting ancient landscapes, glowing runes, and heroic silhouettes. Depict {headline}. Highlight motifs such as {motifs}. {enhanced_sentence} Rendered with misty mountains, golden light, and mythic depth. No text or logos."
        ),
    ),
    # --- Lifestyle/Food/Travel templates ---
    "lifestyle_food": PromptTemplate(
        key="lifestyle_food",
        style_preset="cinematic_warm",
        prompt_template=(
            "Artful culinary composition for '{title}', showing fresh ingredients, plated dishes, and warm natural light. "
            "Depict {headline}. Highlight motifs such as {motifs}. {enhanced_sentence} "
            "Rendered with shallow depth of field, soft shadows, and rich textures. No text or brand logos."
        ),
    ),
    "lifestyle_travel": PromptTemplate(
        key="lifestyle_travel",
        style_preset="cinematic_cool",
        prompt_template=(
            "Cinematic travel scene for '{title}', showing sweeping landscapes, iconic architecture, and natural light. "
            "Depict {headline}. Highlight motifs such as {motifs}. {enhanced_sentence} "
            "Use warm skies, soft haze, and gentle color grading to evoke wanderlust. No text or signs."
        ),
        variants=[
            PromptVariant(
                key="coastal_sunrise",
                prompt_suffix=" Highlight sunrise peach light over coastal cliffs with long-lens compression and atmospheric haze.",
            ),
            PromptVariant(
                key="urban_blue_hour",
                style_preset="cinematic_dark",
                prompt_suffix=" Focus on city streets at blue hour with neon reflections on wet pavement and light trails.",
            ),
        ],
    ),
    "travel_france": PromptTemplate(
        key="travel_france",
        style_preset="pastel",
        prompt_template=(
            "Impressionist travel illustration for '{title}', evoking France through pastel tones, vineyards, cafés, and elegant architecture. "
            "Depict {headline}. Highlight motifs such as {motifs}. {enhanced_sentence} "
            "Rendered in soft morning light with warm highlights and gentle brush texture. No text or signage."
        ),
    ),
    "travel_japan": PromptTemplate(
        key="travel_japan",
        style_preset="watercolor",
        prompt_template=(
            "Serene travel composition for '{title}', evoking Japan with temples, cherry blossoms, and lantern-lit streets. "
            "Depict {headline}. Highlight motifs such as {motifs}. {enhanced_sentence} "
            "Rendered in delicate watercolor tones of pink, indigo, and misty grey. No text or signage."
        ),
    ),
    # --- New templates for specific domains ---
    "tech_datacenter": PromptTemplate(
        key="tech_datacenter",
        style_preset="cinematic_cool",
        prompt_template=(
            "Futuristic data infrastructure scene with rows of glowing servers, intricate fiber-optic cables, and ambient blue light for '{title}'. "
            "Visualize {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Use reflections, subtle mist, and a sense of scale. No people, no text."
        ),
    ),
    "business_finance": PromptTemplate(
        key="business_finance",
        style_preset="studio",
        prompt_template=(
            "Studio-lit composition of illuminated financial charts, digital market visuals, and glowing trend lines for '{title}'. "
            "Depict {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Use sharp contrasts, metallic accents, and vibrant color gradients. No text or logos."
        ),
        variants=[
            PromptVariant(
                key="night_market",
                style_preset="cinematic_dark",
                prompt_suffix=" Frame luminous skyscrapers at night with neon market tickers reflected across glass surfaces.",
            ),
            PromptVariant(
                key="sunrise_boardroom",
                style_preset="cinematic_warm",
                prompt_suffix=" Bathe translucent charts in sunrise golds and rose tones, hinting at executive boardrooms.",
            ),
            PromptVariant(
                key="holographic_dashboard",
                prompt_suffix=" Introduce holographic dashboards suspended in mid-air with cool cyan accents and floating data glyphs.",
            ),
        ],
    ),
    "science_bio": PromptTemplate(
        key="science_bio",
        style_preset="cinematic_cool",
        prompt_template=(
            "Cinematic close-up of molecular structures, DNA helices, and neurons with fluorescent highlights for '{title}'. "
            "Visualize {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Use glowing blues and greens, crisp focus, and scientific abstraction. No text or labels."
        ),
    ),
    "entertainment_music": PromptTemplate(
        key="entertainment_music",
        style_preset="cinematic_warm",
        prompt_template=(
            "Dynamic stage scene with musical instruments, warm spotlights, and stylized sound waves for '{title}'. "
            "Express {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Use golden lighting, vibrant color, and a sense of rhythm. No people or text."
        ),
    ),
    "entertainment_starwars": PromptTemplate(
        key="entertainment_starwars",
        style_preset="cinematic_dark",
        prompt_template=(
            "Epic galactic vista with starships, glowing lightsabers, and cosmic nebulae inspired by Star Wars for '{title}'. "
            "Depict {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Use dramatic lighting, deep shadows, and sci-fi energy. No text or logos."
        ),
    ),
    "entertainment_harrypotter": PromptTemplate(
        key="entertainment_harrypotter",
        style_preset="cinematic_warm",
        prompt_template=(
            "Magical fantasy scene with candlelit halls, enchanted books, and mystical symbols inspired by Harry Potter for '{title}'. "
            "Express {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Use warm golden light, mysterious ambiance, and subtle magical effects. No text or faces."
        ),
    ),
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
        variants=[
            PromptVariant(
                key="pulpit_rays",
                prompt_suffix=" Spotlight a subtle pulpit silhouette with golden volumetric rays and floating dust motes.",
            ),
            PromptVariant(
                key="skylight_bloom",
                prompt_suffix=" Introduce skylight beams cascading over soft seating rows with gentle lens bloom.",
            ),
            PromptVariant(
                key="temple_geometry",
                prompt_suffix=" Hint at temple-inspired arch geometry with layered translucent panels and glowing outlines.",
            ),
        ],
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
        variants=[
            PromptVariant(
                key="macro_copper",
                prompt_suffix=" Emphasize copper and amber reflections with cinematic lens flares and razor-thin depth of field.",
            ),
            PromptVariant(
                key="blueprint_azure",
                style_preset="cinematic_cool",
                prompt_suffix=" Overlay translucent blueprint schematics and electric-blue rim lights to suggest engineering precision.",
            ),
            PromptVariant(
                key="vaporwave_magenta",
                style_preset="cinematic",
                prompt_suffix=" Flood the scene with magenta/teal vaporwave gradients and suspended dust particles for retro-futuristic energy.",
            ),
        ],
    ),
    "tech_programming": PromptTemplate(
        key="tech_programming",
        style_preset="cinematic_dark",
        prompt_template=(
            "Immersive software engineering scene for '{title}', mixing glowing terminals, layered diagrams, and tactile desk elements. "
            "Visualize {headline}. "
            "Highlight motifs such as {motifs}. "
            "{enhanced_sentence}"
            " Emphasize elegant code blocks, whiteboard sketches, and flowing neon signal paths. No people or text."
        ),
        variants=[
            PromptVariant(
                key="dark_terminal",
                prompt_suffix=" Focus on dark-mode editors with teal syntax glow, glass reflections, and shallow depth of field.",
            ),
            PromptVariant(
                key="whiteboard_flow",
                style_preset="studio",
                prompt_suffix=" Capture wide-angle whiteboards with colorful sticky notes, flowcharts, and sunlit studio ambiance.",
            ),
            PromptVariant(
                key="neon_pipeline",
                style_preset="cinematic",
                prompt_suffix=" Introduce holographic pipelines and magenta/cyan data streams weaving between transparent screens.",
            ),
        ],
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
        variants=[
            PromptVariant(
                key="neon_network",
                prompt_suffix=" Emphasize magenta/cyan neural threads weaving through translucent human silhouettes.",
            ),
            PromptVariant(
                key="holographic_brain",
                prompt_suffix=" Form a holographic brain lattice with softly pulsating nodes hovering above metallic surfaces.",
            ),
            PromptVariant(
                key="data_stream",
                style_preset="cinematic",
                prompt_suffix=" Surround the subject with cascading glyph streams and volumetric light tunnels to suggest data flow.",
            ),
        ],
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
            'Layered cinematic montage blending concept and emotion for "{title}". '
            "Capture the essence of {headline} with symbolic visual cues from {motifs}. "
            "{enhanced_sentence}"
            "Use modern lighting, balanced composition, and subtle atmosphere. No text or lettering anywhere."
        ),
        variants=[
            PromptVariant(
                key="prism_warm",
                prompt_suffix=" Lean into warm amber rim lights with prismatic glass shards to convey optimism.",
            ),
            PromptVariant(
                key="noir_contrast",
                style_preset="cinematic_dark",
                prompt_suffix=" Use high-contrast chiaroscuro with cool cyan accents for a more dramatic tone.",
            ),
        ],
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
    "maker_3d_printing": PromptTemplate(
        key="maker_3d_printing",
        style_preset="studio",
        prompt_template=(
            "High-detail studio scene of a modern 3D printer fabricating a component for '{title}'. "
            "Illustrate {headline} with motifs such as {motifs}. "
            "{enhanced_sentence}"
            "Show glowing nozzle light, translucent filament spools, and precise mechanical motion. No people, no text."
        ),
        variants=[
            PromptVariant(
                key="filament_macro",
                prompt_suffix=" Zoom close on the molten filament bead, suspended particles, and spark-like reflections.",
            ),
            PromptVariant(
                key="workspace_wide",
                prompt_suffix=" Pull back to show the printer on a tidy workbench with labelled tools and colorful filament racks.",
            ),
        ],
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
        variants=[
            PromptVariant(
                key="radiant_starscape",
                prompt_suffix=" Scatter radiant constellations and golden dust streams across deep midnight blues.",
            ),
            PromptVariant(
                key="nebula_glow",
                style_preset="cinematic",
                prompt_suffix=" Swirl magenta and cyan nebula clouds with subtle aurora ribbons encircling the focal light.",
            ),
        ],
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
        variants=[
            PromptVariant(
                key="golden_ratio",
                prompt_suffix=" Arrange golden-ratio arcs and concentric halos in matte gold over deep indigo background.",
            ),
            PromptVariant(
                key="crystal_grid",
                prompt_suffix=" Form crystalline lattices and prism-like beams that intersect to suggest divine order.",
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
    analysis = analysis or {}
    text = (summary_text or "").lower()

    source_hint = (source_url or "").strip()
    if not source_hint and isinstance(analysis, dict):
        for key in ("source_url", "canonical_url", "origin_url", "url"):
            candidate = analysis.get(key)
            if isinstance(candidate, str) and candidate.strip():
                source_hint = candidate.strip()
                break
    url = source_hint.lower()

    def url_contains(*snippets: str) -> bool:
        return bool(url) and any(snippet and snippet in url for snippet in snippets)

    if url_contains(
        "churchofjesuschrist.org/study/general-conference",
        "churchofjesuschrist.org/general-conference",
    ):
        return "spiritual_conference"

    # --- Custom routing for new templates ---
    # Lifestyle/Food
    food_terms = (
        "recipe",
        "cook",
        "cooking",
        "ingredient",
        "ingredients",
        "dish",
        "cuisine",
        "chef",
        "culinary",
        "baking",
        "bakery",
        "pastry",
        "dessert",
        "flavour",
        "flavor",
        "meal prep",
        "meal-prep",
        "café",
        "cafe",
    )
    food_url_snippets = (
        "/food","/recipe","/recipes","/cook","/kitchen","/dining","/meal","/cuisine","/baking","bonappetit","foodnetwork","seriouseats","allrecipes","epicurious","tasty.co"
    )
    if any(_text_has_term(text, term) for term in food_terms) or url_contains(*food_url_snippets):
        return "lifestyle_food"
    # Lifestyle/Travel
    travel_terms = (
        "travel","trip","journey","vacation","wanderlust","adventure","destination","tourism","explore","itinerary"
    )
    travel_url_snippets = (
        "/travel","/trip","/journey","/tourism","/destinations","/itinerary","travel-guide","lonelyplanet","expedia","tripadvisor"
    )
    if any(_text_has_term(text, term) for term in travel_terms) or url_contains(*travel_url_snippets):
        return "lifestyle_travel"
    # Travel France
    france_terms = (
        "france","paris","french","eiffel","louvre","provence","bordeaux","café","cafe","versailles","lyon"
    )
    france_url_snippets = (
        "france","/fr/","/france","paris","provence","bordeaux","versailles","lyon","normandy"
    )
    if any(_text_has_term(text, term) for term in france_terms) or url_contains(*france_url_snippets):
        return "travel_france"
    # Travel Japan
    japan_terms = (
        "japan","tokyo","kyoto","osaka","japanese","shrine","temple","cherry blossom","sakura","nara","hokkaido"
    )
    japan_url_snippets = (
        "japan","/jp/","/japan",".jp/","tokyo","kyoto","osaka","sapporo","nara","hokkaido"
    )
    if any(_text_has_term(text, term) for term in japan_terms) or url_contains(*japan_url_snippets):
        return "travel_japan"
    # Programming / software
    programming_terms = (
        "code","coding","programming","software","devops","tutorial","script","scripting","cli","api","sdk","pipeline","microservice","lambda","kubernetes","docker","golang","python","javascript","typescript","terraform","ansible","ci/cd"
    )
    if any(_text_has_term(text, term) for term in programming_terms):
        return "tech_programming"
    # Maker 3D printing
    three_d_terms = (
        "3d print","3d printing","3d-printer","3d-printers","additive manufacturing","filament printer","resin printer","fdm printer","sla printer","bambu lab","prusa","ender 3","gcode","build plate","nozzle","extruder"
    )
    three_d_url_snippets = (
        "/3d-print","/3dprinting","/3d-printer","/additive","3dprinting","all3dp","printables.com","thingiverse","makerworld"
    )
    if any(_text_has_term(text, term) for term in three_d_terms) or url_contains(*three_d_url_snippets):
        return "maker_3d_printing"
    # Tech datacenter
    if "tech" in text or any(token in text for token in CATEGORY_KEYWORDS.get("tech", [])):
        tech_datacenter_terms = ("datacenter", "cloud", "network", "server", "router", "infrastructure")
        if any(term in text for term in tech_datacenter_terms):
            return "tech_datacenter"
        tech_hw_terms = ("hardware", "cpu", "gpu", "processor", "motherboard", "ssd", "ram")
        if any(term in text for term in tech_hw_terms):
            return "tech_hardware"
        ai_terms = ("ai", "artificial intelligence", "neural network", "machine learning")
        if any(term in text for term in ai_terms):
            return "tech_ai"
    # Business finance
    if "business" in text or any(token in text for token in CATEGORY_KEYWORDS.get("business", [])):
        business_finance_terms = ("finance", "market", "economy", "stocks", "investment", "valuation")
        if any(term in text for term in business_finance_terms):
            return "business_finance"
        reviews_terms = ("review", "product", "comparison", "gadget", "test", "hands-on")
        if any(term in text for term in reviews_terms):
            return "reviews_products"
    # Science bio
    if "science" in text or any(token in text for token in CATEGORY_KEYWORDS.get("science", [])):
        science_bio_terms = ("dna", "molecule", "neuron", "cell", "biotech", "lab", "medical", "genetic")
        if any(term in text for term in science_bio_terms):
            return "science_bio"
    # Maker/entertainment hobbies/creativity
    if "maker" in text or "entertainment" in text or any(token in text for token in CATEGORY_KEYWORDS.get("maker", [])) or any(token in text for token in CATEGORY_KEYWORDS.get("entertainment", [])):
        hobbies_terms = ("hobby", "craft", "creative", "art", "painting", "cooking", "baking")
        if any(_text_has_term(text, term) for term in three_d_terms):
            return "maker_3d_printing"
        if any(_text_has_term(text, term) for term in hobbies_terms):
            return "hobbies_creativity"
    # Entertainment subcategories
    if "entertainment" in text or any(token in text for token in CATEGORY_KEYWORDS.get("entertainment", [])):
        # Lord of the Rings
        entertainment_lotr_terms = ("lord of the rings", "middle-earth", "frodo", "gandalf", "mordor", "aragorn", "shire")
        if any(term in text for term in entertainment_lotr_terms):
            return "entertainment_lordoftherings"
        # Music
        entertainment_music_terms = ("music", "song", "album", "band", "concert", "composer", "soundtrack")
        if any(term in text for term in entertainment_music_terms):
            return "entertainment_music"
        # Star Wars
        entertainment_starwars_terms = ("star wars", "jedi", "lightsaber", "galactic empire", "the force")
        if any(term in text for term in entertainment_starwars_terms):
            return "entertainment_starwars"
        # Harry Potter
        entertainment_harrypotter_terms = ("harry potter", "hogwarts", "wizard", "magic", "spell")
        if any(term in text for term in entertainment_harrypotter_terms):
            return "entertainment_harrypotter"
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
        if any(_text_has_term(text, token) for token in keywords):
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
                if any(term in text for term in conference_terms):
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
            if key == "programming":
                return "tech_programming"
            return key
    topics = analysis.get("key_topics") if isinstance(analysis, dict) else None
    if isinstance(topics, list):
        lowered_topics = " ".join(str(t).lower() for t in topics if t)
        if any(_text_has_term(lowered_topics, term) for term in food_terms):
            return "lifestyle_food"
        if any(_text_has_term(lowered_topics, term) for term in travel_terms):
            return "lifestyle_travel"
        if any(_text_has_term(lowered_topics, term) for term in france_terms):
            return "travel_france"
        if any(_text_has_term(lowered_topics, term) for term in japan_terms):
            return "travel_japan"
        if any(_text_has_term(lowered_topics, term) for term in programming_terms):
            return "tech_programming"
        # --- Custom routing for new templates (topics) ---
        # Tech datacenter
        if "tech" in lowered_topics or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("tech", [])):
            tech_datacenter_terms = ("datacenter", "cloud", "network", "server", "router", "infrastructure")
            if any(term in lowered_topics for term in tech_datacenter_terms):
                return "tech_datacenter"
            tech_hw_terms = ("hardware", "cpu", "gpu", "processor", "motherboard", "ssd", "ram")
            if any(term in lowered_topics for term in tech_hw_terms):
                return "tech_hardware"
            ai_terms = ("ai", "artificial intelligence", "neural network", "machine learning")
            if any(term in lowered_topics for term in ai_terms):
                return "tech_ai"
        # Business finance
        if "business" in lowered_topics or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("business", [])):
            business_finance_terms = ("finance", "market", "economy", "stocks", "investment", "valuation")
            if any(term in lowered_topics for term in business_finance_terms):
                return "business_finance"
            reviews_terms = ("review", "product", "comparison", "gadget", "test", "hands-on")
            if any(term in lowered_topics for term in reviews_terms):
                return "reviews_products"
        # Science bio
        if "science" in lowered_topics or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("science", [])):
            science_bio_terms = ("dna", "molecule", "neuron", "cell", "biotech", "lab", "medical", "genetic")
            if any(term in lowered_topics for term in science_bio_terms):
                return "science_bio"
        # Maker/entertainment hobbies/creativity
        if "maker" in lowered_topics or "entertainment" in lowered_topics or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("maker", [])) or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("entertainment", [])):
            hobbies_terms = ("hobby", "craft", "creative", "art", "painting", "cooking", "baking")
            if any(_text_has_term(lowered_topics, term) for term in three_d_terms):
                return "maker_3d_printing"
            if any(_text_has_term(lowered_topics, term) for term in hobbies_terms):
                return "hobbies_creativity"
        # Entertainment subcategories
        if "entertainment" in lowered_topics or any(token in lowered_topics for token in CATEGORY_KEYWORDS.get("entertainment", [])):
            # Lord of the Rings
            entertainment_lotr_terms = ("lord of the rings", "middle-earth", "frodo", "gandalf", "mordor", "aragorn", "shire")
            if any(term in lowered_topics for term in entertainment_lotr_terms):
                return "entertainment_lordoftherings"
            entertainment_music_terms = ("music", "song", "album", "band", "concert", "composer", "soundtrack")
            if any(term in lowered_topics for term in entertainment_music_terms):
                return "entertainment_music"
            entertainment_starwars_terms = ("star wars", "jedi", "lightsaber", "galactic empire", "the force")
            if any(term in lowered_topics for term in entertainment_starwars_terms):
                return "entertainment_starwars"
            entertainment_harrypotter_terms = ("harry potter", "hogwarts", "wizard", "magic", "spell")
            if any(term in lowered_topics for term in entertainment_harrypotter_terms):
                return "entertainment_harrypotter"
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

    # Cooldown to avoid repeated generations for the same content id
    if image_mode != "ai2":
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
                    "image_mode": image_mode,
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
                "image_mode": image_mode,
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
    cid = str(content.get("id") or content.get("video_id") or "")
    summary_text = ""
    if isinstance(summary_data, dict):
        summary_text = summary_data.get("summary") or summary_data.get("text") or ""
    elif isinstance(summary_data, str):
        summary_text = summary_data
    source_url = _extract_source_url(content, analysis)
    override_prompt = _extract_override_prompt(content, mode="ai2" if freestyle_mode else "ai1")
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
    model_name = generation.get("model") or generation.get("engine") or (template.key if template else "default")
    metadata = {
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
        "source_url": absolute_url,
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


def _text_has_term(text: str, term: str) -> bool:
    term = (term or "").strip().lower()
    if not term:
        return False
    text = text or ""
    if all(ch.isalpha() for ch in term):
        pattern = rf"\b{re.escape(term)}\b"
        return re.search(pattern, text) is not None
    return term in text


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
