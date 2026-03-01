#!/usr/bin/env python3
"""
Test script for Inception Labs Mercury-2 API.

Mercury-2 is a diffusion reasoning model that generates tokens in parallel,
enabling 2-6x faster responses than traditional autoregressive models.

Usage:
    python tools/test_mercury.py              # Test instant mode
    python tools/test_mercury.py --mode default   # Test default mode
    python tools/test_mercury.py --prompt "Your custom prompt"
"""

import os
import sys
import time
import argparse
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
except ImportError:
    print("ERROR: requests not installed")
    print("  pip install requests")
    sys.exit(1)


# Inception Labs Mercury-2 config
INCEPTION_API_KEY = os.getenv("INCEPTION_API_KEY")
if not INCEPTION_API_KEY:
    raise ValueError("INCEPTION_API_KEY environment variable not set")
INCEPTION_ENDPOINT = "https://api.inceptionlabs.ai/v1/chat/completions"
MODEL = "mercury-2"


def test_mercury(prompt: str, mode: str = "instant", max_tokens: int = 1000):
    """Test Mercury-2 API with given prompt and mode."""

    print(f"\n{'='*60}")
    print(f"Mercury-2 Test ({mode} mode)")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n")

    headers = {
        "Authorization": f"Bearer {INCEPTION_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    if mode == "instant":
        body["reasoning_effort"] = "instant"
    else:
        body["max_tokens"] = max_tokens

    print(f"Request body: {json.dumps(body, indent=2)}\n")

    start_time = time.time()

    try:
        response = requests.post(
            INCEPTION_ENDPOINT,
            headers=headers,
            json=body,
            timeout=30
        )

        elapsed = time.time() - start_time

        print(f"Response time: {elapsed:.2f}s")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\n{'='*60}")
            print("Response:")
            print(f"{'='*60}")

            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                print(content)

            # Show usage stats if available
            if "usage" in data:
                usage = data["usage"]
                print(f"\n{'='*60}")
                print("Token Usage:")
                print(f"{'='*60}")
                print(f"  Prompt tokens:      {usage.get('prompt_tokens', 'N/A')}")
                print(f"  Completion tokens:  {usage.get('completion_tokens', 'N/A')}")
                print(f"  Total tokens:       {usage.get('total_tokens', 'N/A')}")

            # Show reasoning tokens if available
            if "choices" in data and len(data["choices"]) > 0:
                reasoning = data["choices"][0].get("reasoning_tokens")
                if reasoning:
                    print(f"\n  Reasoning tokens:  {reasoning}")

        else:
            print(f"\nERROR: {response.status_code}")
            print(response.text)

    except requests.exceptions.Timeout:
        print(f"\nERROR: Request timed out after {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"\nERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Inception Labs Mercury-2 API"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["instant", "default"],
        default="instant",
        help="Mode: instant (fast, ~0.3s) or default (deeper, ~0.5-3s)"
    )
    parser.add_argument(
        "--prompt", "-p",
        default="What is quantum computing? Explain in 2-3 sentences.",
        help="Prompt to send to Mercury-2"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Max tokens for default mode (default: 1000)"
    )

    args = parser.parse_args()

    test_mercury(args.prompt, args.mode, args.max_tokens)


if __name__ == "__main__":
    main()
