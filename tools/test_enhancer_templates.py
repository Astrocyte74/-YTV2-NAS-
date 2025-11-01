#!/usr/bin/env python3
"""
Quick sanity-checker for prompt enhancer instructions.

Runs both the local Ollama model (gemma3:12b by default) and the configured cloud
model (OpenRouter → google/gemini-2.5-flash-lite) against a list of sample prompts,
using separate system instructions for Flux-style vs general SD-style outputs.
"""

import os
from typing import Iterable, Tuple

from modules import ollama_client
from modules.services import cloud_service

FLUX_SYSTEM = (
    "You are an expert prompt engineer optimising prompts for Flux.1 [schnell]. "
    "Rewrite the concept as one natural-language sentence (<60 words) that clearly describes the entire scene. "
    "Explicitly cover foreground, midground, and background or depth cues; specify mood, lighting, colour palette, and camera perspective. "
    "Mention important materials or special visual effects (e.g., rain, reflections, glass, motion blur) when implied. "
    "Return plain text without quotation marks or any markdown (no bold, italics, bullet points)."
)

GENERAL_SYSTEM = (
    "You are an expert prompt engineer for Stable Diffusion–style models. "
    "Rewrite the concept as a concise, comma-separated prompt optimised for Draw Things. "
    "Include subject, context, style or medium, lighting, composition, and key adjectives in short descriptive fragments. "
    "Avoid full sentences, commentary, markdown, or negative prompts; return a single line of fragments separated by commas."
)

PROMPTS: Iterable[str] = (
    "sleepy cat curled on a sunlit windowsill",
    "massive space pirate ship landing on a desert planet",
    "lush rainforest waterfall hiding an ancient stone statue",
    "retro neon diner interior at midnight in the rain",
    "cyberpunk samurai portrait with neon rim lighting",
)

LOCAL_MODEL = os.getenv("DRAW_LOCAL_MODEL") or os.getenv("QUICK_LOCAL_MODEL") or "gemma3:12b"
CLOUD_MODEL = "google/gemini-2.5-flash-lite"


def run_local(system: str, concept: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": concept},
    ]
    resp = ollama_client.chat(messages, LOCAL_MODEL, stream=False)
    text = resp.get("response") if isinstance(resp, dict) else None
    if not text and isinstance(resp, dict):
        message = resp.get("message") or {}
        text = message.get("content") or message.get("text")
    return (text or "").strip()


def run_cloud(system: str, concept: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": concept},
    ]
    return cloud_service.chat(messages, provider="openrouter", model=CLOUD_MODEL).strip()


def test_label(label: str, system: str) -> None:
    print("=" * 80)
    print(f"Instruction set: {label}")
    for prompt in PROMPTS:
        print(f"\nConcept: {prompt}")
        try:
            local = run_local(system, prompt)
        except Exception as exc:
            local = f"[LOCAL ERROR] {exc}"
        try:
            cloud = run_cloud(system, prompt)
        except Exception as exc:
            cloud = f"[CLOUD ERROR] {exc}"
        print(f"  local → {local}")
        print(f"  cloud → {cloud}")


if __name__ == "__main__":
    test_label("Flux / Schnell", FLUX_SYSTEM)
    test_label("General SD", GENERAL_SYSTEM)
