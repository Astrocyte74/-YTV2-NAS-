"""Utility functions for working with summary variants.

These helpers normalize variant identifiers, merge newly generated
summaries with previously stored variants, and provide light HTML
formatting for text variants. The goal is to keep variant handling
consistent across the NAS reprocessing pipeline and the Postgres
ingest client.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import html
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Canonical variant ordering used for consistent presentation
_VARIANT_ORDER = {
    "comprehensive": 0,
    "bullet-points": 1,
    "key-points": 1,  # legacy alias, normalize later
    "key-insights": 2,
    "executive": 3,
    "adaptive": 4,
}

# Known aliases that should collapse to a canonical id
_VARIANT_ALIASES = {
    "summary": "comprehensive",
    "key_points": "bullet-points",
    "key-points": "bullet-points",
    "key_insights": "key-insights",
    "audio_summary": "audio",
    "audio-summary": "audio",
    "audio_fr": "audio-fr",
    "audio_es": "audio-es",
}


def normalize_variant_id(value: Optional[str]) -> str:
    """Return a canonical, kebab-cased variant id."""
    if not value:
        return ""

    text = str(value).strip().lower()

    # Preserve suffixes such as `audio-fr:beginner`
    if ":" in text:
        prefix, suffix = text.split(":", 1)
        normalized_prefix = normalize_variant_id(prefix)
        normalized_suffix = suffix.replace("_", "-").strip()
        return f"{normalized_prefix}:{normalized_suffix}" if normalized_suffix else normalized_prefix

    text = text.replace("_", "-").replace(" ", "-")
    text = re.sub(r"[^a-z0-9:-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")

    if not text:
        return ""

    return _VARIANT_ALIASES.get(text, text)


def variant_kind(variant_id: str) -> str:
    """Return the content kind for a variant (text vs audio)."""
    normalized = normalize_variant_id(variant_id)
    if normalized.startswith("audio"):
        return "audio"
    return "text"


def format_summary_html(text: str) -> str:
    """Convert plain text summary into minimalist HTML.

    Enhancements:
    - Strip code fence markers (```lang) while keeping content lines
    - Detect single-line heading blocks ending with ':' → <h3 class="kp-heading">
    - Detect 'Bottom line:' → <p class="kp-takeaway"><strong>Bottom line:</strong> …</p>
    - Map bullet lines (•, -, *) → <ul class="kp-list"><li>…</li></ul>
    """
    if not isinstance(text, str):
        return ""

    normalized = text.replace("\r", "\n").strip()
    if not normalized:
        return ""

    # Remove code fence markers to avoid literal ``` blocks in HTML
    normalized = re.sub(r"^```.*$", "", normalized, flags=re.MULTILINE)
    normalized = normalized.replace("```", "")

    out: List[str] = []
    in_list = False
    pending_heading: Optional[str] = None

    def flush_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    heading_re = re.compile(r"^[A-Za-z][\w \-/]{1,60}:$")
    bottom_re = re.compile(r"^bottom\s+line:\s*(.*)$", flags=re.IGNORECASE)

    def clean_heading(text: str) -> str:
        s = text.strip()
        if s.endswith(":"):
            s = s[:-1].strip()
        # Remove common markdown heading/bold/italic wrappers
        s = re.sub(r"^#{1,6}\s+", "", s)  # strip leading markdown '#'
        s = re.sub(r"\s+#{1,6}$", "", s)  # strip trailing markdown '#'
        # Bold/italic
        # Extract bold label in numbered patterns like "1) **Title**"
        m = re.match(r"^\d+\)\s+(\*\*|__)(.+?)(\*\*|__)", s)
        if m:
            return m.group(2).strip()
        m = re.match(r"^(\*\*|__)(.+?)(\*\*|__)$", s)
        if m:
            s = m.group(2).strip()
        else:
            m = re.match(r"^\*(.+)\*$", s)
            if m:
                s = m.group(1).strip()
            else:
                m = re.match(r"^_(.+)_$", s)
                if m:
                    s = m.group(1).strip()
        return s

    for raw in normalized.splitlines():
        line = raw.strip()
        if not line:
            flush_list()
            # Keep pending_heading across blank lines
            continue

        # Heading line with colon
        if heading_re.match(line):
            flush_list()
            heading = clean_heading(line)
            out.append(f"<h3 class=\"kp-heading\">{html.escape(heading)}</h3>")
            pending_heading = None
            continue

        # Bottom line
        m = bottom_re.match(line)
        if m:
            flush_list()
            rest = (m.group(1) or "").strip()
            label = "Bottom line:"
            content = f"<strong>{html.escape(label)}</strong> {html.escape(rest)}" if rest else f"<strong>{html.escape(label)}</strong>"
            out.append(f"<p class=\"kp-takeaway\">{content}</p>")
            continue

        # Markdown-style bold/italic heading line (e.g., **Title** or _Title_)
        if re.match(r"^(\*\*|__).+(\*\*|__)$", line) or re.match(r"^(\*|_).+(\*|_)$", line):
            flush_list()
            out.append(f"<h3 class=\"kp-heading\">{html.escape(clean_heading(line))}</h3>")
            pending_heading = None
            continue

        # If we have a pending heading and now see bullets, emit heading
        if pending_heading and re.match(r"^[\-*•]", line):
            flush_list()
            out.append(f"<h3 class=\"kp-heading\">{html.escape(clean_heading(pending_heading))}</h3>")
            pending_heading = None

        # Bullet line
        if re.match(r"^[\-*•]", line):
            if not in_list:
                out.append("<ul class=\"kp-list\">")
                in_list = True
            cleaned = re.sub(r"^[\-*•]\s*", "", line).strip()
            out.append(f"<li>{html.escape(cleaned)}</li>")
            continue

        # Potential heading without colon: short line, next block is a list (handled above)
        if len(line) <= 60 and not re.search(r"[.!?]$", line) and not re.match(r"^[a-z]", line):
            # Defer decision until we know what follows
            # If another non-bullet line comes next, we'll emit as paragraph then
            if pending_heading:
                # Previous pending wasn't followed by bullets; emit as paragraph
                flush_list()
                out.append(f"<p>{html.escape(pending_heading)}</p>")
            pending_heading = line
            continue

        # Paragraph (and flush any pending heading as paragraph)
        if pending_heading:
            flush_list()
            out.append(f"<p>{html.escape(pending_heading)}</p>")
            pending_heading = None
        flush_list()
        out.append(f"<p>{html.escape(line)}</p>")

    flush_list()
    # If file ended with a pending non-bullet heading, emit as paragraph
    if pending_heading:
        out.append(f"<p>{html.escape(pending_heading)}</p>")
    return "\n".join(out)


def _clone_variant_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sanitized copy of a variant entry."""
    clone = deepcopy(entry)

    # Harmonize field names
    text_value = clone.get("text") or clone.get("summary") or clone.get("content")
    if isinstance(text_value, str):
        clone["text"] = text_value.strip()
    else:
        clone.pop("text", None)

    for remove_key in ("summary", "content"):
        clone.pop(remove_key, None)

    variant_id = normalize_variant_id(clone.get("variant") or clone.get("summary_type") or clone.get("type"))
    if variant_id:
        clone["variant"] = variant_id
    else:
        clone.pop("variant", None)

    if "kind" not in clone and variant_id:
        clone["kind"] = variant_kind(variant_id)

    # Normalize proficiency naming
    if "proficiency_level" in clone and "proficiency" not in clone:
        clone["proficiency"] = clone.pop("proficiency_level")

    # Ensure generated_at is ISO string if present
    if "generated_at" in clone and isinstance(clone["generated_at"], datetime):
        clone["generated_at"] = clone["generated_at"].isoformat()

    return clone


def _sort_variants(variants: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return variants sorted by priority, then lexicographically."""
    def sort_key(entry: Dict[str, Any]) -> Tuple[int, str]:
        variant_id = normalize_variant_id(entry.get("variant"))
        base = variant_id.split(":", 1)[0]
        priority = _VARIANT_ORDER.get(base, 100)
        return priority, variant_id

    return sorted(variants, key=sort_key)


def merge_summary_variants(
    *,
    new_report: Dict[str, Any],
    requested_variant: Optional[str],
    existing_report: Optional[Dict[str, Any]] = None,
    prefer_default: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge newly generated summary data with prior variants.

    The function mutates and returns ``new_report`` with ``summary.variants``
    populated, default/active variants aligned, and metadata normalized.
    """
    summary_section = new_report.setdefault("summary", {})
    requested_id = normalize_variant_id(requested_variant or summary_section.get("summary_type") or "comprehensive")

    # Build variant catalog from existing report (if any)
    variant_catalog: Dict[str, Dict[str, Any]] = {}
    default_variant = normalize_variant_id(summary_section.get("default_variant"))

    if existing_report:
        existing_summary = existing_report.get("summary") or {}
        default_variant = default_variant or normalize_variant_id(
            existing_summary.get("default_variant") or existing_summary.get("summary_type")
        )

        existing_variants = existing_summary.get("variants")
        if isinstance(existing_variants, list):
            for entry in existing_variants:
                if not isinstance(entry, dict):
                    continue
                cloned = _clone_variant_entry(entry)
                variant_id = cloned.get("variant")
                text_value = cloned.get("text")
                if variant_id and isinstance(text_value, str) and text_value.strip():
                    variant_catalog[variant_id] = cloned
        else:
            # Promote legacy single summary into a variant entry
            legacy_text = existing_summary.get("summary")
            legacy_type = normalize_variant_id(existing_summary.get("summary_type"))
            if isinstance(legacy_text, str) and legacy_text.strip() and legacy_type:
                variant_catalog[legacy_type] = {
                    "variant": legacy_type,
                    "text": legacy_text.strip(),
                    "headline": existing_summary.get("headline"),
                    "language": existing_summary.get("language"),
                    "generated_at": existing_summary.get("generated_at"),
                    "kind": variant_kind(legacy_type),
                }

    # Incorporate freshly generated variant
    new_variant_entry = {
        "variant": requested_id,
        "text": (summary_section.get("summary") or "").strip(),
        "headline": summary_section.get("headline"),
        "language": summary_section.get("language") or new_report.get("summary_language"),
        "generated_at": summary_section.get("generated_at") or new_report.get("processed_at"),
        "kind": variant_kind(requested_id),
    }

    proficiency = summary_section.get("proficiency") or summary_section.get("proficiency_level")
    if proficiency:
        new_variant_entry["proficiency"] = proficiency

    audio_url = summary_section.get("audio_url")
    if audio_url:
        new_variant_entry["audio_url"] = audio_url

    if requested_id and new_variant_entry["text"]:
        variant_catalog[requested_id] = new_variant_entry

    # Determine default variant
    base_default = prefer_default or default_variant
    base_default = normalize_variant_id(base_default)

    if requested_id == "comprehensive":
        active_default = "comprehensive"
    elif base_default and base_default in variant_catalog:
        active_default = base_default
    elif "comprehensive" in variant_catalog:
        active_default = "comprehensive"
    elif requested_id and requested_id in variant_catalog:
        active_default = requested_id
    elif variant_catalog:
        active_default = next(iter(_sort_variants(variant_catalog.values()))).get("variant", "comprehensive")
    else:
        active_default = "comprehensive"

    # Finalize variant list
    variant_list = _sort_variants(variant_catalog.values())
    summary_section["variants"] = variant_list
    summary_section["default_variant"] = active_default
    summary_section["latest_variant"] = requested_id

    active_entry = variant_catalog.get(active_default)
    if active_entry:
        summary_section["summary"] = active_entry.get("text", "")
        summary_section["summary_type"] = active_default
        if active_entry.get("headline"):
            summary_section["headline"] = active_entry["headline"]
        if active_entry.get("language"):
            summary_section["language"] = active_entry["language"]
        if active_entry.get("generated_at"):
            summary_section["generated_at"] = active_entry["generated_at"]

    # Keep top-level languages aligned with the active variant when possible
    if active_entry and active_entry.get("language"):
        new_report["summary_language"] = active_entry["language"]
        new_report["audio_language"] = new_report.get("audio_language") or active_entry["language"]

    return new_report


__all__ = [
    "format_summary_html",
    "merge_summary_variants",
    "normalize_variant_id",
    "variant_kind",
]
