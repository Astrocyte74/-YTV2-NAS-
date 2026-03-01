#!/usr/bin/env python3
"""
Test to verify Mercury-2 instant vs default modes actually differ.

This compares:
1. Instant mode (reasoning_effort="instant") - should be faster with fewer reasoning tokens
2. Default mode - should be slower with more reasoning tokens
"""

import os
import time
import requests

INCEPTION_API_KEY = os.getenv("INCEPTION_API_KEY")
if not INCEPTION_API_KEY:
    raise ValueError("INCEPTION_API_KEY environment variable not set")
INCEPTION_ENDPOINT = "https://api.inceptionlabs.ai/v1/chat/completions"

TEST_PROMPT = "Explain the difference between a list and a tuple in Python. Keep it brief."


def test_mercury_mode(mode: str):
    """Test Mercury-2 in specified mode."""
    headers = {
        "Authorization": f"Bearer {INCEPTION_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "mercury-2",
        "messages": [{"role": "user", "content": TEST_PROMPT}]
    }

    if mode == "instant":
        body["reasoning_effort"] = "instant"
    else:
        body["max_tokens"] = 1000

    start = time.time()
    response = requests.post(INCEPTION_ENDPOINT, headers=headers, json=body, timeout=30)
    elapsed = time.time() - start

    if response.status_code != 200:
        return {
            "mode": mode,
            "error": f"HTTP {response.status_code}",
            "response": response.text
        }

    data = response.json()
    usage = data.get("usage", {})

    # Extract reasoning tokens if available
    reasoning_tokens = None
    if "choices" in data and len(data["choices"]) > 0:
        reasoning_tokens = data["choices"][0].get("reasoning_tokens")

    return {
        "mode": mode,
        "elapsed_seconds": round(elapsed, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "content_preview": data.get("choices", [{}])[0].get("message", {}).get("content", "")[:100]
    }


def main():
    print("=" * 70)
    print("Mercury-2 Mode Comparison Test")
    print("=" * 70)
    print(f"Test prompt: {TEST_PROMPT}")
    print()

    # Test instant mode
    print("Testing INSTANT mode (reasoning_effort='instant')...")
    instant_result = test_mercury_mode("instant")
    print(f"  Response time: {instant_result.get('elapsed_seconds')}s")
    print(f"  Reasoning tokens: {instant_result.get('reasoning_tokens', 'N/A')}")
    print(f"  Total tokens: {instant_result.get('total_tokens', 'N/A')}")
    print()

    # Wait a moment between requests
    time.sleep(1)

    # Test default mode
    print("Testing DEFAULT mode (no reasoning_effort)...")
    default_result = test_mercury_mode("default")
    print(f"  Response time: {default_result.get('elapsed_seconds')}s")
    print(f"  Reasoning tokens: {default_result.get('reasoning_tokens', 'N/A')}")
    print(f"  Total tokens: {default_result.get('total_tokens', 'N/A')}")
    print()

    # Comparison
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    instant_time = instant_result.get('elapsed_seconds', 0)
    default_time = default_result.get('elapsed_seconds', 0)
    instant_reasoning = instant_result.get('reasoning_tokens')
    default_reasoning = default_result.get('reasoning_tokens')

    print(f"Response time:")
    print(f"  Instant: {instant_time}s")
    print(f"  Default: {default_time}s")
    print(f"  Difference: {abs(instant_time - default_time):.2f}s ({default_time/instant_time:.1f}x slower)")

    print(f"\nReasoning tokens:")
    print(f"  Instant: {instant_reasoning}")
    print(f"  Default: {default_reasoning}")

    # Verify instant mode is actually working
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    if instant_reasoning is not None and default_reasoning is not None:
        if instant_reasoning < default_reasoning:
            print("✅ CONFIRMED: Instant mode uses FEWER reasoning tokens")
        elif instant_reasoning == default_reasoning:
            print("⚠️  WARNING: Same reasoning tokens in both modes")
            print("    The reasoning_effort parameter may not be working as expected!")
        else:
            print("❌ UNEXPECTED: Instant mode has MORE reasoning tokens")
    else:
        print("⚠️  Could not compare - reasoning tokens not returned in response")

    if instant_time < default_time:
        print("✅ CONFIRMED: Instant mode is FASTER")
    else:
        print("⚠️  WARNING: Instant mode was not faster in this test")


if __name__ == "__main__":
    main()
