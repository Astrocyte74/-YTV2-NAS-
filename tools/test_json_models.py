#!/usr/bin/env python3
"""Test JSON parsing reliability and quality across different LLM models."""

import sys
import os
sys.path.insert(0, '/app')

from research_api.research_service.llm import chat_json_schema
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

# Models to test - grouped by price tier
MODELS = [
    # Budget tier (~$0.01-0.02 per 1M tokens)
    ("openrouter", "google/gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite", "budget"),
    ("openrouter", "openai/gpt-5.4-nano", "gpt-5.4-nano", "budget"),

    # Mid tier (~$0.05-0.10 per 1M tokens)
    ("openrouter", "google/gemini-3-flash-preview", "gemini-3-flash-preview", "mid"),
    ("openrouter", "openai/gpt-5.4-mini", "gpt-5.4-mini", "mid"),

    # Inception (free local)
    ("inception", None, "mercury-2", "free"),
]


def assess_quality(parsed):
    """Assess the quality of generated questions."""
    score = 0
    max_score = 100
    details = []

    suggestions = parsed.get("suggestions", [])

    # Check suggestions count (up to 20 points)
    if len(suggestions) >= 2:
        score += 20
        details.append("✓ 2+ suggestions")
    elif len(suggestions) == 1:
        score += 10
        details.append("△ Only 1 suggestion")
    else:
        details.append("✗ No suggestions")

    # Check each suggestion (up to 40 points per suggestion, normalized to 2)
    for i, s in enumerate(suggestions[:2]):
        q_score = 0
        question = s.get("question", "")
        reason = s.get("reason", "")
        label = s.get("label", "")

        # Question quality (20 points)
        if question:
            # Check if it's actually a question
            if "?" in question:
                q_score += 5
            # Check length (not too short, not too long)
            if 20 <= len(question) <= 200:
                q_score += 5
            # Check for research-worthy keywords
            research_keywords = ["how", "what", "why", "when", "which", "compare", "impact", "evidence", "current", "recent"]
            if any(kw in question.lower() for kw in research_keywords):
                q_score += 5
            # Check specificity (not generic)
            if any(w in question.lower() for w in ["specific", "exactly", "precise", "method", "technique"]):
                q_score += 3
            # Penalize vague questions
            if question.lower().startswith("what is") and len(question.split()) < 8:
                q_score -= 3
        details.append(f"  Q{i+1} quality: {q_score}/20")

        # Reason quality (10 points)
        r_score = 0
        if reason:
            if len(reason) >= 10:
                r_score += 5
            if len(reason) >= 30:
                r_score += 3
            # Check for relevance keywords
            if any(w in reason.lower() for w in ["relevant", "important", "because", "helps", "understand"]):
                r_score += 2
        details.append(f"  Q{i+1} reason: {r_score}/10")

        # Required fields present (10 points)
        f_score = 0
        if s.get("id"): f_score += 2
        if s.get("label"): f_score += 2
        if s.get("kind"): f_score += 2
        if s.get("priority") is not None: f_score += 2
        if s.get("default_selected") is not None: f_score += 2
        details.append(f"  Q{i+1} fields: {f_score}/10")

        score += q_score + r_score + f_score

    # Normalize to 100
    score = min(100, score)

    return score, details


def test_model(provider, model_override, display_name, tier):
    """Test a single model for JSON parsing reliability and quality."""
    print(f"\n{'='*60}")
    print(f"Testing: {display_name} ({tier})")
    print(f"Provider: {provider}, Model: {model_override or 'default'}")
    print('='*60)

    try:
        parsed, actual_provider, actual_model = chat_json_schema(
            messages=TEST_MESSAGES,
            schema_name="TestSuggestions",
            schema=SCHEMA,
            max_tokens=800,
            reasoning_effort="low",
            temperature=0.1,
            timeout=45,
            provider=provider,
            model_override=model_override,
        )

        # Validate structure
        has_should_suggest = "should_suggest" in parsed
        has_suggestions = "suggestions" in parsed
        suggestions_count = len(parsed.get("suggestions", []))

        # Assess quality
        quality_score, quality_details = assess_quality(parsed)

        print(f"✅ JSON PARSED")
        print(f"   Provider: {actual_provider}")
        print(f"   Model: {actual_model}")
        print(f"   Suggestions: {suggestions_count}")
        print(f"   Quality Score: {quality_score}/100")

        if suggestions_count > 0:
            print(f"\n   Generated Questions:")
            for i, s in enumerate(parsed.get("suggestions", [])[:2]):
                q = s.get("question", "")[:100]
                print(f"   {i+1}. {q}{'...' if len(s.get('question',''))>100 else ''}")

        print(f"\n   Quality Details:")
        for d in quality_details:
            print(f"   {d}")

        return True, quality_score, parsed

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, 0, None


def main():
    print("="*70)
    print("JSON + QUALITY Model Comparison Test")
    print("="*70)
    print(f"Testing {len(MODELS)} models, 2 trials each")
    print("Measuring: JSON parse success + Question quality")
    print("="*70)

    results = {}

    for provider, model_override, display_name, tier in MODELS:
        success_count = 0
        quality_scores = []

        for trial in range(2):
            print(f"\n--- Trial {trial + 1}/2 ---")
            success, quality, parsed = test_model(provider, model_override, display_name, tier)
            if success:
                success_count += 1
                quality_scores.append(quality)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        results[display_name] = {
            "tier": tier,
            "success_rate": success_count,
            "avg_quality": avg_quality,
            "combined": success_count * 50 + avg_quality * 0.5,  # 50% success, 50% quality
        }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (sorted by combined score)")
    print("="*70)
    print(f"{'Model':<25} {'Tier':<8} {'JSON':<8} {'Quality':<10} {'Combined'}")
    print("-"*70)

    for name, r in sorted(results.items(), key=lambda x: x[1]["combined"], reverse=True):
        json_status = f"{r['success_rate']}/2"
        quality = f"{r['avg_quality']:.0f}/100"
        combined = f"{r['combined']:.1f}"
        tier = r['tier']

        status = "✅" if r['success_rate'] == 2 and r['avg_quality'] >= 70 else "⚠️" if r['success_rate'] >= 1 else "❌"
        print(f"{status} {name:<23} {tier:<8} {json_status:<8} {quality:<10} {combined}")

    print("\n" + "="*70)
    print("PRICE TIER COMPARISON")
    print("="*70)

    # Budget tier comparison
    budget = [(n, r) for n, r in results.items() if r['tier'] == 'budget']
    if budget:
        best_budget = max(budget, key=lambda x: x[1]['combined'])
        print(f"Budget Tier Winner: {best_budget[0]} (quality: {best_budget[1]['avg_quality']:.0f})")

    # Mid tier comparison
    mid = [(n, r) for n, r in results.items() if r['tier'] == 'mid']
    if mid:
        best_mid = max(mid, key=lambda x: x[1]['combined'])
        print(f"Mid Tier Winner: {best_mid[0]} (quality: {best_mid[1]['avg_quality']:.0f})")

    # Free tier
    free = [(n, r) for n, r in results.items() if r['tier'] == 'free']
    if free:
        best_free = max(free, key=lambda x: x[1]['combined'])
        print(f"Free Tier: {best_free[0]} (quality: {best_free[1]['avg_quality']:.0f})")


if __name__ == "__main__":
    main()
