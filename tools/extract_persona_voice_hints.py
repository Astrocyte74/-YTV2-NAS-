#!/usr/bin/env python3
"""
Utility: extract persona voice requirements from persona_data.json.

Produces a concise table (or JSON) of persona names, categories, gender,
voice hints, and derived accent keywords so front-end voice pickers can
be aligned with the stored metadata.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_PERSONA_PATH = Path("data/persona_data.json")

# Simple keyword map to normalise accent references found in voice_hint strings.
ACCENT_KEYWORDS: Dict[str, str] = {
    "african": "african",
    "american": "american",
    "arabic": "arabic",
    "argentine": "argentine",
    "australian": "australian",
    "british": "british",
    "canadian": "canadian",
    "caribbean": "caribbean",
    "chinese": "chinese",
    "dutch": "dutch",
    "english": "english",
    "french": "french",
    "german": "german",
    "greek": "greek",
    "hindi": "hindi",
    "indian": "indian",
    "irish": "irish",
    "italian": "italian",
    "japanese": "japanese",
    "korean": "korean",
    "latin american": "latin-american",
    "middle eastern": "middle-eastern",
    "nigerian": "nigerian",
    "norwegian": "norwegian",
    "polish": "polish",
    "portuguese": "portuguese",
    "russian": "russian",
    "scandinavian": "scandinavian",
    "scottish": "scottish",
    "south african": "south-african",
    "spanish": "spanish",
    "swedish": "swedish",
    "turkish": "turkish",
}


def accent_keywords(voice_hint: str) -> List[str]:
    """Return a sorted list of canonical accent keywords detected in the hint."""
    if not voice_hint:
        return []
    text = voice_hint.lower()
    hits = {value for key, value in ACCENT_KEYWORDS.items() if key in text}
    return sorted(hits)


def build_display_name(persona: Dict[str, object]) -> str:
    """Create a helpful display string including era or archetype when available."""
    name = str(persona.get("name", "")).strip()
    era = str(persona.get("era", "") or "").strip()
    archetype = str(persona.get("archetype", "") or "").strip()

    if era and archetype:
        return f"{name} — {archetype} ({era})"
    if era:
        return f"{name} ({era})"
    if archetype:
        return f"{name} — {archetype}"
    return name


def collect_voice_rows(persona_data: Dict[str, object]) -> List[Dict[str, object]]:
    """Flatten persona entries to the subset needed for voice assignment."""
    rows: List[Dict[str, object]] = []
    categories: Sequence[Dict[str, object]] = persona_data.get("categories", [])  # type: ignore[assignment]

    for block in categories:
        category_name = str(block.get("category", "")).strip()
        for persona in block.get("personas", []) or []:
            if not isinstance(persona, dict):
                continue

            row = {
                "name": persona.get("name"),
                "display_name": build_display_name(persona),
                "category": category_name,
                "gender": persona.get("gender"),
                "voice_hint": persona.get("voice_hint"),
                "accent_keywords": accent_keywords(str(persona.get("voice_hint", ""))),
                "archetype": persona.get("archetype"),
                "temperament": persona.get("temperament"),
                "curiosity_level": persona.get("curiosity_level"),
            }
            rows.append(row)
    return rows


def print_table(rows: Sequence[Dict[str, object]]) -> None:
    """Pretty-print a simple table to stdout."""
    if not rows:
        print("No persona entries found.", file=sys.stderr)
        return

    headers = ["Name", "Category", "Gender", "Accent Keywords", "Voice Hint", "Display Name"]
    # Compute column widths with sensible caps.
    widths = {
        "Name": 24,
        "Category": 14,
        "Gender": 6,
        "Accent Keywords": 22,
        "Voice Hint": 48,
        "Display Name": 48,
    }

    def clamp(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[: limit - 1] + "…"

    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))

    for row in rows:
        line_parts = [
            clamp(str(row.get("name", "")), widths["Name"]).ljust(widths["Name"]),
            clamp(str(row.get("category", "")), widths["Category"]).ljust(widths["Category"]),
            clamp(str(row.get("gender", "") or ""), widths["Gender"]).ljust(widths["Gender"]),
            clamp(", ".join(row.get("accent_keywords") or []), widths["Accent Keywords"]).ljust(
                widths["Accent Keywords"]
            ),
            clamp(str(row.get("voice_hint", "") or ""), widths["Voice Hint"]).ljust(widths["Voice Hint"]),
            clamp(str(row.get("display_name", "") or ""), widths["Display Name"]).ljust(widths["Display Name"]),
        ]
        print(" | ".join(line_parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="List persona voice expectations.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PERSONA_PATH,
        help=f"Path to persona_data.json (default: {DEFAULT_PERSONA_PATH})",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format: human-readable table or raw JSON (default: table).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="When using JSON output, indent for readability.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Persona data file not found: {args.input}")

    try:
        persona_payload = json.loads(args.input.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        parser.error(f"Failed to parse persona JSON ({args.input}): {exc}")

    rows = collect_voice_rows(persona_payload)

    if args.format == "json":
        indent = 2 if args.pretty else None
        json.dump(rows, sys.stdout, indent=indent)
        if indent is not None:
            sys.stdout.write("\n")
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
