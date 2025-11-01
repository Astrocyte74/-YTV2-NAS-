#!/usr/bin/env python3
"""
Quick comparison harness for draw prompt enhancement.

Runs the current local model plus optional overrides and the configured cloud model
against a handful of sample prompts, printing the enhanced outputs for review.
"""

import asyncio
import os
from typing import List, Tuple

from modules.services import draw_service


SAMPLE_PROMPTS: List[str] = [
    "cozy living room with sunlight streaming through stained glass windows",
    "cyberpunk alley cat wearing neon goggles at night in the rain",
    "ancient forest temple ruins covered in moss and glowing runes",
]

# Default to whatever the environment resolves today (e.g., gemma3:12b)
CURRENT_LOCAL = draw_service._resolve_local_model()

# Additional local models to evaluate explicitly.
EXTRA_LOCAL_MODELS: List[str] = [
    "brxce/stable-diffusion-prompt-generator:latest",
    "ALIENTELLIGENCE/imagepromptengineer:latest",
]

# Prepare (label, model_slug) tuples, skipping duplicates and empty slots.
local_models: List[Tuple[str, str]] = []
seen = set()

if CURRENT_LOCAL:
    local_models.append(("current_env", CURRENT_LOCAL))
    seen.add(CURRENT_LOCAL)

for slug in EXTRA_LOCAL_MODELS:
    slug = (slug or "").strip()
    if not slug or slug in seen:
        continue
    local_models.append((slug, slug))
    seen.add(slug)


async def run() -> None:
    print("== Prompt Enhancement Comparison ==")
    print(f"Resolved current local model: {CURRENT_LOCAL or '<unset>'}")
    print(f"QUICK_CLOUD_MODEL: {os.getenv('QUICK_CLOUD_MODEL', '').strip() or '<unset>'}")
    print()

    for concept in SAMPLE_PROMPTS:
        print("=" * 80)
        print(f"Original prompt: {concept}")
        print("-" * 80)

        for label, model_slug in local_models:
            try:
                enhanced = await draw_service.enhance_prompt_local(concept, model=model_slug)
            except Exception as exc:
                print(f"[local:{label}] ERROR: {exc}")
                continue
            print(f"[local:{label}] {enhanced}")

        try:
            enhanced_cloud = await draw_service.enhance_prompt_cloud(concept)
        except Exception as exc:
            print(f"[cloud] ERROR: {exc}")
        else:
            print(f"[cloud] {enhanced_cloud}")

        print()


if __name__ == "__main__":
    asyncio.run(run())
