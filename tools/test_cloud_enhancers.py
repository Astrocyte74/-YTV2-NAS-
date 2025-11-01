#!/usr/bin/env python3
import asyncio
import os
from typing import Iterable, List, Tuple

from modules.services import draw_service

MODELS: List[Tuple[str, str]] = [
    ("google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-lite"),
    ("openai/gpt-5-nano", "openai/gpt-5-nano"),
    ("minimax/minimax-m2:free", "minimax/minimax-m2:free"),
    ("x-ai/grok-4-fast", "x-ai/grok-4-fast"),
]

PROMPTS: Iterable[str] = [
    "sleepy cat curled on a sunlit windowsill",
    "massive space pirate ship landing on a desert planet",
    "lush rainforest waterfall hiding an ancient stone statue",
    "retro neon diner interior at midnight in the rain",
    "cyberpunk samurai portrait with neon rim lighting",
    "steampunk airship skyline at sunrise",
    "mystical library with floating books and candles",
    "stormy coastal lighthouse during lightning strike",
]


async def run() -> None:
    for label, slug in MODELS:
        print("=" * 80)
        print(f"Cloud model: {label}")
        os.environ["QUICK_CLOUD_MODEL"] = slug
        for prompt in PROMPTS:
            try:
                result = await draw_service.enhance_prompt_cloud(prompt)
            except Exception as exc:
                result = f"ERROR: {exc}"
            print(f"Prompt: {prompt}")
            print(f" -> {result}\n")


if __name__ == "__main__":
    asyncio.run(run())
