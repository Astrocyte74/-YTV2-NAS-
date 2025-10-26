#!/usr/bin/env python3
"""
CJCLDS speaker classifier

Detects General Conference talks from churchofjesuschrist.org and assigns:
- category: CJCLDS
- subcategory: canonical speaker name (Apostles) or "Other"

Integration: call `classify_and_apply_cjclds(report_dict, url)` before saving
JSON or posting to ingest. The function merges with any existing categories.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse


APOSTLES_LASTNAME_TO_CANONICAL: Dict[str, str] = {
    "oaks": "Dallin H. Oaks",
    "eyring": "Henry B. Eyring",
    "holland": "Jeffrey R. Holland",
    "uchtdorf": "Dieter F. Uchtdorf",
    "bednar": "David A. Bednar",
    "cook": "Quentin L. Cook",
    "christofferson": "D. Todd Christofferson",
    "andersen": "Neil L. Andersen",
    "rasband": "Ronald A. Rasband",
    "stevenson": "Gary E. Stevenson",
    "renlund": "Dale G. Renlund",
    "gong": "Gerrit W. Gong",
    "soares": "Ulisses Soares",
    "kearon": "Patrick Kearon",
}


def _is_church_site(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    return "churchofjesuschrist.org" in host


def _is_cjclds_talk(url: Optional[str]) -> bool:
    if not _is_church_site(url):
        return False
    try:
        parsed = urlparse(url or "")
    except Exception:
        return False
    path = (parsed.path or "")
    return "/general-conference/" in path


def _extract_slug_lastname(url: str) -> Optional[str]:
    """Extract a likely last name token from the last path segment.

    Examples:
    - /2025/10/41holland -> holland
    - /2025/10/35andersen?lang=eng -> andersen
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    segment = (parsed.path or "").rstrip("/").split("/")[-1]
    if not segment:
        return None
    # Drop query-like suffixes and non-letters
    segment = re.sub(r"[^A-Za-z]+", "", segment)
    segment = segment.strip().lower()
    return segment or None


def _merge_categories(existing: Optional[Dict[str, Any]], category: str, sub: str) -> Dict[str, Any]:
    categories: List[Dict[str, Any]] = []
    if isinstance(existing, dict):
        cats = existing.get("categories")
        if isinstance(cats, list):
            categories = [c for c in cats if isinstance(c, dict)]

    # Find or create CJCLDS entry
    cj = None
    for entry in categories:
        if (entry.get("category") or "").strip().lower() == category.lower():
            cj = entry
            break
    if cj is None:
        cj = {"category": category, "subcategories": []}
        categories.append(cj)

    subs = cj.get("subcategories")
    if not isinstance(subs, list):
        subs = []
        cj["subcategories"] = subs

    if sub and sub not in subs:
        subs.append(sub)

    return {"categories": categories}


def classify_and_apply_cjclds(report: Dict[str, Any], url: Optional[str]) -> Dict[str, Any]:
    """Augment a universal schema dict with CJCLDS categorization when applicable.

    - Adds/merges subcategories_json with { category: CJCLDS, subcategories: [speaker or Other] }
    - Adds analysis_json fields: speaker, speaker_role
    """
    if not _is_church_site(url):
        return report

    if _is_cjclds_talk(url):
        lastname = _extract_slug_lastname(url or "") or ""
        canonical = APOSTLES_LASTNAME_TO_CANONICAL.get(lastname)
        speaker = canonical or "Other"
        role = "apostle" if canonical else "other"
    else:
        # Non‑conference church content → tag as CJCLDS: Non GC
        speaker = "Non GC"
        role = "non-gc"

    # Merge categories
    existing = report.get("subcategories_json") if isinstance(report, dict) else None
    report["subcategories_json"] = _merge_categories(existing, "CJCLDS", speaker)

    # Add analysis helpers
    analysis = report.get("analysis_json")
    if not isinstance(analysis, dict):
        analysis = {}
    analysis.setdefault("speaker", speaker)
    analysis.setdefault("speaker_role", role)
    report["analysis_json"] = analysis

    # Normalize channel name for church site
    try:
        report["channel_name"] = "ChurchofJesusChrist"
        meta = report.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("uploader", "ChurchofJesusChrist")
        report["metadata"] = meta
    except Exception:
        pass

    return report


__all__ = [
    "classify_and_apply_cjclds",
]
