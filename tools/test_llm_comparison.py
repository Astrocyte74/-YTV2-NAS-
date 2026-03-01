#!/usr/bin/env python3
"""
Compare 4 LLM options on the same summary:

1. Inception • mercury-2-instant (Fast)
2. Inception • mercury-2 (Deep)
3. OpenRouter • gemini-2.5-flash-lite
4. Local • gemma3:12b

Measures: quality, speed, price
"""

import os
import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if not os.getenv('INCEPTION_API_KEY'):
    raise ValueError("INCEPTION_API_KEY environment variable not set")

from youtube_summarizer import YouTubeSummarizer
from langchain_core.messages import HumanMessage

# Pricing (as of 2026)
PRICING = {
    "inception/mercury-2-instant": {
        "input": 0.00025,  # per 1K tokens (estimated)
        "output": 0.001,   # per 1K tokens (estimated)
    },
    "inception/mercury-2": {
        "input": 0.00025,
        "output": 0.001,
    },
    "openrouter/gemini-2.5-flash-lite": {
        "input": 0.000075,  # $0.075 per 1M input
        "output": 0.0001,   # $0.10 per 1M output
    },
    "ollama/gemma3:12b": {
        "input": 0,  # Free local
        "output": 0,
    }
}


# Test transcript - short but meaningful (first paragraph of a tech article)
TEST_TRANSCRIPT = """
In this video, we're going to explore the fascinating world of quantum computing.
Quantum computers use quantum bits or qubits, which unlike classical bits that can only be
0 or 1, can exist in a state of superposition, being both 0 and 1 at the same time. This
allows quantum computers to process multiple possibilities simultaneously, making them
potentially much more powerful for certain types of problems.

One of the most famous quantum algorithms is Shor's algorithm, which can factor large
numbers exponentially faster than the best known classical algorithms. This has
significant implications for cryptography, as many encryption schemes rely on the
difficulty of factoring large numbers.

However, quantum computers are still in their early stages. Current quantum processors
have limited qubits and suffer from noise and decoherence, which means quantum states
are fragile and can be easily disturbed. Companies like IBM, Google, and startups
are racing to build more stable quantum computers with more qubits.

Another key concept is quantum entanglement, where two qubits become correlated in
such a way that the state of one instantly influences the state of the other, no
matter how far apart they are. Einstein called this "spooky action at a distance."

In summary, quantum computing is a revolutionary technology that could transform
fields from cryptography to drug discovery, but significant engineering challenges
remain before we see practical, large-scale quantum computers.
"""

TEST_PROMPT = f"""Summarize the following transcript in 3-4 bullet points:

{TEST_TRANSCRIPT}

Keep each bullet point concise (under 20 words)."""


def test_llm(provider: str, model: str) -> dict:
    """Test a single LLM and return results."""
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
        response = summarizer.llm.ainvoke([HumanMessage(content=TEST_PROMPT)])
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(response)
        call_time = time.time() - start_init

        # Extract content
        content = result.content if hasattr(result, 'content') else str(result)

        # Calculate estimated price (we don't have actual token counts)
        # Rough estimate: prompt ~250 tokens, response ~100 tokens
        est_input_tokens = 250
        est_output_tokens = len(content.split()) * 1.3  # rough estimate

        pricing = PRICING.get(full_name, {"input": 0, "output": 0})
        est_cost = (est_input_tokens / 1000 * pricing["input"] +
                   est_output_tokens / 1000 * pricing["output"])

        return {
            "provider": provider,
            "model": model,
            "full_name": full_name,
            "success": True,
            "init_time": round(init_time, 2),
            "call_time": round(call_time, 2),
            "content": content,
            "word_count": len(content.split()),
            "est_cost": round(est_cost, 5),
            "error": None
        }

    except Exception as e:
        return {
            "provider": provider,
            "model": model,
            "full_name": full_name,
            "success": False,
            "error": str(e),
            "content": None,
            "call_time": None,
            "est_cost": None
        }


def print_results(results: list):
    """Print comparison results."""
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    print(f"\n{'Model':<35} | {'Time':<6} | {'Words':<6} | {'Est Cost':<10}")
    print("-"*70)

    for r in results:
        if r["success"]:
            print(f"{r['full_name']:<35} | {r['call_time']:<6} | {r['word_count']:<6} | ${r['est_cost']:<10}")
        else:
            print(f"{r['full_name']:<35} | FAILED: {r['error']}")

    print("\n" + "="*70)
    print("GENERATED SUMMARIES")
    print("="*70)

    for r in results:
        if r["success"]:
            print(f"\n📌 {r['full_name']}")
            print("-"*70)
            print(r["content"])
            print()


def main():
    # Define the 4 models to test
    models_to_test = [
        ("inception", "mercury-2-instant"),
        ("inception", "mercury-2"),
        ("openrouter", "google/gemini-2.5-flash-lite"),
        ("ollama", "gemma3:12b"),
    ]

    results = []

    for provider, model in models_to_test:
        result = test_llm(provider, model)
        results.append(result)
        time.sleep(1)  # Brief pause between tests

    print_results(results)

    # Speed comparison
    print("="*70)
    print("SPEED RANKING (fastest to slowest)")
    print("="*70)
    successful = [r for r in results if r["success"]]
    successful.sort(key=lambda x: x["call_time"])
    for i, r in enumerate(successful, 1):
        print(f"{i}. {r['full_name']}: {r['call_time']}s")

    # Cost comparison
    print("\n" + "="*70)
    print("COST RANKING (cheapest to most expensive)")
    print("="*70)
    successful_with_cost = [r for r in successful if r['est_cost'] > 0]
    successful_with_cost.sort(key=lambda x: x["est_cost"])
    for i, r in enumerate(successful_with_cost, 1):
        print(f"{i}. {r['full_name']}: ${r['est_cost']}")
    free_models = [r for r in successful if r['est_cost'] == 0]
    if free_models:
        print(f"FREE: {', '.join([r['full_name'] for r in free_models])}")


if __name__ == "__main__":
    main()
