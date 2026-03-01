#!/usr/bin/env python3
"""
Real-world LLM comparison test using actual Nike Pegasus 42 transcript.

Tests: mercury-2-instant, mercury-2, gemini-2.5-flash-lite, gemma3:12b
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if not os.getenv('INCEPTION_API_KEY'):
    raise ValueError("INCEPTION_API_KEY environment variable not set")
if not os.getenv('OPENROUTER_API_KEY'):
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

from youtube_summarizer import YouTubeSummarizer
from langchain_core.messages import HumanMessage

# Load the actual transcript
with open('data/reports/nike_pegasus_42_design_analysis_SueGkP8au70.json', 'r') as f:
    report = json.load(f)

TRANSCRIPT = report.get('transcript', '')
VIDEO_TITLE = report.get('title', 'Nike Pegasus 42')

print(f"Testing with: {VIDEO_TITLE}")
print(f"Transcript: {len(TRANSCRIPT)} chars, {len(TRANSCRIPT.split())} words")
print()

# The actual summarization prompt used by the system
SUMMARY_PROMPT = f"""You are a video summarization AI. Create a comprehensive summary of the following video transcript.

Title: {VIDEO_TITLE}

Transcript:
{TRANSCRIPT}

Please provide:
1. A concise headline (10-15 words)
2. Key topics covered (3-5 bullet points)
3. Main insights or takeaways (3-5 bullet points)
4. Brief conclusion

Format your response clearly with headings."""


async def test_llm_async(provider: str, model: str) -> dict:
    """Test a single LLM asynchronously."""
    full_name = f"{provider}/{model}"
    print(f"\n{'='*70}")
    print(f"Testing: {full_name}")
    print('='*70)

    try:
        # Initialize summarizer
        start_init = time.time()
        summarizer = YouTubeSummarizer(llm_provider=provider, model=model)
        init_time = time.time() - start_init

        # Make the call
        start_call = time.time()

        # Use the robust_llm_call method like the real system does
        result = await summarizer._robust_llm_call(
            [HumanMessage(content=SUMMARY_PROMPT)],
            operation_name=f"{full_name} summary"
        )

        if not result:
            raise Exception("No response from LLM")

        call_time = time.time() - start_call
        word_count = len(result.split())

        # Estimate cost
        # Rough estimate: prompt ~4000 tokens, response ~500 tokens
        pricing = {
            "inception/mercury-2-instant": {"input": 0.00025, "output": 0.001},
            "inception/mercury-2": {"input": 0.00025, "output": 0.001},
            "openrouter/google/gemini-2.5-flash-lite": {"input": 0.000075, "output": 0.0001},
            "ollama/gemma3:12b": {"input": 0, "output": 0},
        }

        est_input = 4000
        est_output = word_count * 1.3
        p = pricing.get(full_name, {"input": 0, "output": 0})
        est_cost = (est_input / 1000 * p["input"] + est_output / 1000 * p["output"])

        return {
            "provider": provider,
            "model": model,
            "full_name": full_name,
            "success": True,
            "call_time": round(call_time, 2),
            "content": result,
            "word_count": word_count,
            "est_cost": round(est_cost, 5),
            "error": None
        }

    except Exception as e:
        import traceback
        return {
            "provider": provider,
            "model": model,
            "full_name": full_name,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "content": None,
            "call_time": None,
            "est_cost": None
        }


def print_summary_preview(content: str, max_lines: int = 15):
    """Print a preview of the summary."""
    lines = content.split('\n')
    if len(lines) <= max_lines:
        print(content)
    else:
        print('\n'.join(lines[:max_lines]))
        print(f"\n... [{len(lines) - max_lines} more lines]")


async def main():
    models_to_test = [
        ("inception", "mercury-2-instant"),
        ("inception", "mercury-2"),
        ("openrouter", "google/gemini-2.5-flash-lite"),
        ("ollama", "gemma3:12b"),
    ]

    results = []

    for provider, model in models_to_test:
        result = await test_llm_async(provider, model)
        results.append(result)
        await asyncio.sleep(1)  # Brief pause between tests

    # Print comparison table
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"\n{'Model':<35} | {'Time':<8} | {'Words':<6} | {'Est Cost':<10}")
    print("-"*70)

    for r in results:
        if r["success"]:
            print(f"{r['full_name']:<35} | {r['call_time']:<8} | {r['word_count']:<6} | ${r['est_cost']:<10}")
        else:
            print(f"{r['full_name']:<35} | FAILED")

    # Print summaries
    print("\n" + "="*70)
    print("GENERATED SUMMARIES (Preview)")
    print("="*70)

    for r in results:
        if r["success"]:
            print(f"\n{'='*70}")
            print(f"📌 {r['full_name']} ({r['call_time']}s, {r['word_count']} words)")
            print('='*70)
            print_summary_preview(r["content"])
            print()
        else:
            print(f"\n{'='*70}")
            print(f"❌ {r['full_name']} - FAILED")
            print(f"Error: {r['error']}")
            print('='*70)

    # Speed ranking
    print("\n" + "="*70)
    print("SPEED RANKING")
    print("="*70)
    successful = [r for r in results if r["success"]]
    successful.sort(key=lambda x: x["call_time"])
    for i, r in enumerate(successful, 1):
        print(f"{i}. {r['full_name']}: {r['call_time']}s ({r['word_count']} words)")

    # Cost ranking
    print("\n" + "="*70)
    print("COST COMPARISON")
    print("="*70)
    for r in successful:
        cost_str = "FREE" if r['est_cost'] == 0 else f"${r['est_cost']}"
        print(f"{r['full_name']}: {cost_str}")


if __name__ == "__main__":
    asyncio.run(main())
