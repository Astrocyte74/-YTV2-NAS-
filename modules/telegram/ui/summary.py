from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import urllib.parse


def _short_id(video_id: str) -> str:
    """Generate a short hash for video_id that fits in Telegram's 64-byte callback_data limit."""
    # Use first 8 chars of MD5 hash - gives us 64M unique values, more than enough
    return hashlib.md5(video_id.encode()).hexdigest()[:8]


def friendly_variant_label(variant: str, variant_labels: Dict[str, str]) -> str:
    base, _, suffix = variant.partition(':')
    base_label = variant_labels.get(base, base.replace('-', ' ').title())
    if suffix:
        suffix_clean = suffix.replace('-', ' ').replace('_', ' ').title()
        return f"{base_label} ({suffix_clean})"
    return base_label


def build_summary_callback(action: str, content_id: Optional[str] = None) -> str:
    callback = f"summarize_{action}"
    if content_id:
        callback += f"|{content_id}"
    return callback


def build_summary_keyboard(
    variant_labels: Dict[str, str],
    existing_variants: Optional[List[str]] = None,
    video_id: Optional[str] = None,
    dashboard_url: Optional[str] = None,
    show_delete_button: bool = False,
) -> tuple[InlineKeyboardMarkup, Dict[str, str]]:
    """
    Build summary keyboard with variant buttons.

    Returns:
        tuple: (keyboard markup, delete_id_map mapping short_id -> video_id)
    """
    existing_variants = existing_variants or []
    existing_bases = {variant.split(':', 1)[0] for variant in existing_variants}

    # Mapping for delete button: short hash -> full video_id
    delete_id_map: Dict[str, str] = {}

    def label_for(variant_key: str) -> str:
        label = variant_labels.get(variant_key, variant_key.replace('-', ' ').title())
        return f"{label} ✅" if variant_key in existing_bases else label

    # Retire "Key Points" (bullet-points) from the UI, but keep support internally.
    # Arrange primary options compactly: Comprehensive + Insights on row 1, Audio on row 2.
    keyboard: List[List[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(label_for('comprehensive'), callback_data=build_summary_callback("comprehensive", video_id)),
            InlineKeyboardButton(label_for('key-insights'), callback_data=build_summary_callback("key-insights", video_id)),
        ],
        [
            InlineKeyboardButton(label_for('bullet-points'), callback_data=build_summary_callback("bullet-points", video_id)),
        ],
        [
            InlineKeyboardButton(label_for('audio'), callback_data=build_summary_callback("audio", video_id)),
        ],
        [
            InlineKeyboardButton(label_for('audio-fr'), callback_data=build_summary_callback("audio-fr", video_id)),
            InlineKeyboardButton(label_for('audio-es'), callback_data=build_summary_callback("audio-es", video_id)),
        ],
    ]

    if existing_bases and video_id and dashboard_url:
        report_id_encoded = urllib.parse.quote(video_id, safe='')
        keyboard.append([
            InlineKeyboardButton("📄 Open summary", url=f"{dashboard_url}#report={report_id_encoded}"),
        ])

    # Add delete button when showing summary results (not in initial selection)
    # Use short hash to fit within Telegram's 64-byte callback_data limit
    if show_delete_button and video_id:
        short_id = _short_id(video_id)
        delete_id_map[short_id] = video_id  # Store mapping for handler to resolve
        keyboard.append([
            InlineKeyboardButton("🗑️ Delete", callback_data=f"summary_delete:{short_id}"),
        ])

    return InlineKeyboardMarkup(keyboard), delete_id_map


def existing_variants_message(
    variant_labels: Dict[str, str],
    content_id: str,
    variants: List[str],
    source: str = "youtube",
) -> str:
    if not variants:
        prompts = {
            "youtube": "🎬 Processing YouTube video...\n\nChoose your summary type:",
            "reddit": "🧵 Processing Reddit thread...\n\nChoose your summary type:",
            "web": "📰 Processing web article...\n\nChoose your summary type:",
        }
        return prompts.get(source, prompts["youtube"])

    variants_sorted = sorted(variants)
    noun = {
        "youtube": "video",
        "reddit": "thread",
        "web": "article",
    }.get(source, "item")
    lines = [f"✅ Existing summaries for this {noun}:"]
    lines.extend(f"• {friendly_variant_label(variant, variant_labels)}" for variant in variants_sorted)
    lines.append("\nRe-run a variant below or open the summary card.")
    return "\n".join(lines)


def build_summary_provider_keyboard(
    cloud_label: str,
    *,
    local_label: Optional[str] = None,
    content_id: Optional[str] = None,
) -> InlineKeyboardMarkup:
    """Render provider selection keyboard for summary generation."""
    rows: List[List[InlineKeyboardButton]] = [
        [InlineKeyboardButton(cloud_label, callback_data="summary_provider:cloud")],
    ]
    if local_label:
        rows.append([InlineKeyboardButton(local_label, callback_data="summary_provider:ollama")])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data=build_summary_callback("back_to_main", content_id))])
    return InlineKeyboardMarkup(rows)


def build_summary_model_keyboard(
    provider_key: str,
    model_options: List[Dict[str, Any]],
    *,
    per_row: int = 2,
) -> InlineKeyboardMarkup:
    """Render model selection keyboard for the chosen provider."""
    rows: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    for idx, option in enumerate(model_options):
        label = option.get("button_label") or option.get("label") or f"Model {idx + 1}"
        callback = f"summary_model:{provider_key}:{idx}"
        row.append(InlineKeyboardButton(label, callback_data=callback))
        if len(row) == per_row:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="summary_model:back")])
    return InlineKeyboardMarkup(rows)
