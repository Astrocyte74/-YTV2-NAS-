from __future__ import annotations

from typing import Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import urllib.parse


def friendly_variant_label(variant: str, variant_labels: Dict[str, str]) -> str:
    base, _, suffix = variant.partition(':')
    base_label = variant_labels.get(base, base.replace('-', ' ').title())
    if suffix:
        suffix_clean = suffix.replace('-', ' ').replace('_', ' ').title()
        return f"{base_label} ({suffix_clean})"
    return base_label


def build_summary_keyboard(
    variant_labels: Dict[str, str],
    existing_variants: Optional[List[str]] = None,
    video_id: Optional[str] = None,
    dashboard_url: Optional[str] = None,
) -> InlineKeyboardMarkup:
    existing_variants = existing_variants or []
    existing_bases = {variant.split(':', 1)[0] for variant in existing_variants}

    def label_for(variant_key: str) -> str:
        label = variant_labels.get(variant_key, variant_key.replace('-', ' ').title())
        return f"{label} ✅" if variant_key in existing_bases else label

    keyboard: List[List[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(label_for('comprehensive'), callback_data="summarize_comprehensive"),
            InlineKeyboardButton(label_for('bullet-points'), callback_data="summarize_bullet-points"),
        ],
        [
            InlineKeyboardButton(label_for('key-insights'), callback_data="summarize_key-insights"),
            InlineKeyboardButton(label_for('audio'), callback_data="summarize_audio"),
        ],
        [
            InlineKeyboardButton(label_for('audio-fr'), callback_data="summarize_audio-fr"),
            InlineKeyboardButton(label_for('audio-es'), callback_data="summarize_audio-es"),
        ],
    ]

    if existing_bases and video_id and dashboard_url:
        report_id_encoded = urllib.parse.quote(video_id, safe='')
        keyboard.append([
            InlineKeyboardButton("📄 Open summary", url=f"{dashboard_url}#report={report_id_encoded}"),
        ])

    return InlineKeyboardMarkup(keyboard)


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
