#!/usr/bin/env python3
"""Test JSON parsing reliability across different LLM models."""

import sys
import os
sys.path.insert(0, '/app')

from research_api.research_service.llm import chat_json_schema, _extract_json_object
import json

# Test prompt for follow-up suggestions
TEST_MESSAGES = [
    {
        "role": "system",
        "content": "You are a research assistant that generates follow-up questions in JSON format."
    },
    {
        "role": "user",
        "content": """Based on this summary, generate 2 follow-up research questions.

Summary: "Scientists have discovered evidence that humans may be much older than previously thought, with new fossil findings pushing back the timeline of human evolution by several hundred thousand years."

Return JSON in this exact format:
{
  "should_suggest": true,
  "explanation": "Brief explanation",
  "suggestions": [
    {
      "id": "q1",
      "label": "Short label",
      "question": "The actual research question?",
      "reason": "Why this is relevant",
      "kind": "background",
      "priority": 0.8,
      "default_selected": true
    }
  ]
}

Return ONLY valid JSON, no markdown."""
    }
]

SCHEMA = {
    "type": "object",
    "properties": {
        "should_suggest": {"type": "boolean"},
        "explanation": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "question": {"type": "string"},
                    "reason": {"type": "string"},
                    "kind": {"type": "string"},
                    "priority": {"type": "number"},
                    "default_selected": {"type": "boolean"}
                }
            }
        }
    },
    "required": ["should_suggest", "suggestions"]
}

# Models to test
MODELS = [
    ("openrouter", "google/gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite"),
    ("openrouter", "google/gemini-2.5-flash", "gemini-2.5-flash"),
    ("openrouter", "google/gemini-3-flash-preview", "gemini-3-flash-preview"),
    ("inception", None, "mercury-2"),
]

def test_model(provider, model_override, display_name):
    """Test a single model for JSON parsing reliability."""
    print(f"\n{'='*60}")
    print(f"Testing: {display_name}")
    print(f"Provider: {provider}, Model: {model_override or 'default'}")
    print('='*60)

    try:
        parsed, actual_provider, actual_model = chat_json_schema(
            messages=TEST_MESSAGES,
            schema_name="TestSuggestions",
            schema=SCHEMA,
            max_tokens=500,
            reasoning_effort="low",
            temperature=0.1,
            timeout=30,
            provider=provider,
            model_override=model_override,
        )

        # Validate structure
        has_should_suggest = "should_suggest" in parsed
        has_suggestions = "suggestions" in parsed
        suggestions_count = len(parsed.get("suggestions", []))

        print(f"✅ SUCCESS")
        print(f"   Provider: {actual_provider}")
        print(f"   Model: {actual_model}")
        print(f"   has should_suggest: {has_should_suggest}")
        print(f"   has suggestions: {has_suggestions}")
        print(f"   suggestions count: {suggestions_count}")

        if suggestions_count > 0:
            first_q = parsed["suggestions"][0].get("question", "")[:80]
            print(f"   First question: {first_q}...")

        return True, parsed

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, None


def main():
    print("JSON Parsing Model Comparison Test")
    print("="*60)
    print(f"Running {len(MODELS)} models, 2 trials each\n")

    results = {}

    for provider, model_override, display_name in MODELS:
        success_count = 0
        for trial in range(2):
            print(f"\n--- Trial {trial + 1}/2 ---")
            success, parsed = test_model(provider, model_override, display_name)
            if success:
                success_count += 1

        results[display_name] = {
            "success_rate": f"{success_count}/2",
            "pct": success_count * 50
        }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, result in sorted(results.items(), key=lambda x: x[1]["pct"], reverse=True):
        status = "✅" if result["pct"] == 100 else "⚠️" if result["pct"] >= 50 else "❌"
        print(f"{status} {name}: {result['success_rate']} ({result['pct']}%)")


if __name__ == "__main__":
    main()
