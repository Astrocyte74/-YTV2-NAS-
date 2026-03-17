#!/usr/bin/env python3
"""Rigorous JSON + Quality comparison across LLM models."""

import sys
import os
sys.path.insert(0, '/app')

from research_api.research_service.llm import chat_json_schema
import time

# Test cases with different topics and complexity
TEST_CASES = [
    {
        "name": "Science - Galaxy counting",
        "summary": "Astronomers estimate there are 2 trillion galaxies in the observable universe, based on deep-field imaging from Hubble and JWST telescopes. This count has increased 10x from earlier estimates of 200 billion due to better detection of faint, distant galaxies.",
        "expected_topics": ["telescopes", "methods", "accuracy", "JWST"],
    },
    {
        "name": "Technology - AI agents",
        "summary": "AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Modern agents use large language models for reasoning and can interact with APIs, browse the web, and execute code.",
        "expected_topics": ["capabilities", "limitations", "safety", "applications"],
    },
    {
        "name": "Health - Sleep research",
        "summary": "New research shows that during sleep, the brain's glymphatic system clears toxic proteins 10x faster than when awake. This discovery may explain the link between poor sleep and neurodegenerative diseases like Alzheimer's.",
        "expected_topics": ["mechanisms", "diseases", "prevention", "research"],
    },
    {
        "name": "Business - Remote work",
        "summary": "A 2024 Stanford study found that fully remote workers are 18% less productive than office workers, but hybrid workers (3 days office) showed no productivity difference. Employee satisfaction was highest in hybrid arrangements.",
        "expected_topics": ["productivity", "satisfaction", "best practices", "trends"],
    },
    {
        "name": "Short/Unclear summary",
        "summary": "Something about new batteries.",
        "expected_topics": ["clarification"],
    },
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

MODELS = [
    ("openrouter", "google/gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite", "budget"),
    ("openrouter", "openai/gpt-5.4-nano", "gpt-5.4-nano", "budget"),
    ("openrouter", "google/gemini-3-flash-preview", "gemini-3-flash-preview", "mid"),
    ("openrouter", "openai/gpt-5.4-mini", "gpt-5.4-mini", "mid"),
    ("inception", None, "mercury-2", "free"),
]

QUALITY_CRITERIA = {
    "relevance": "Does the question directly relate to the summary topic?",
    "specificity": "Is the question specific and actionable, not vague?",
    "depth": "Does the question go beyond surface-level information?",
    "research_value": "Would answering this question add real knowledge?",
    "clarity": "Is the question clearly worded and unambiguous?",
}


def assess_question_quality(question: str, reason: str, expected_topics: list) -> dict:
    """Assess a single question across multiple quality dimensions."""
    scores = {}
    q_lower = question.lower()
    r_lower = reason.lower() if reason else ""

    # Relevance (0-20): Does it relate to expected topics?
    topic_matches = sum(1 for t in expected_topics if t.lower() in q_lower)
    scores["relevance"] = min(20, 10 + topic_matches * 5)

    # Specificity (0-20): Specific words indicate focused questions
    specific_words = ["specific", "exactly", "how many", "which", "what method", "compare", "vs", "percentage", "rate"]
    specificity_count = sum(1 for w in specific_words if w in q_lower)
    scores["specificity"] = min(20, 10 + specificity_count * 5)

    # Avoid vague questions
    vague_starts = ["what is", "tell me about", "describe"]
    if any(q_lower.startswith(v) for v in vague_starts) and len(question.split()) < 10:
        scores["specificity"] -= 5

    # Depth (0-20): Complex question indicators
    depth_words = ["implications", "why", "how does", "what causes", "relationship", "impact", "consequences"]
    depth_count = sum(1 for w in depth_words if w in q_lower)
    scores["depth"] = min(20, 10 + depth_count * 5)

    # Research value (0-20): Would this add knowledge?
    research_words = ["recent", "current", "latest", "evidence", "studies", "data", "compare", "versus"]
    research_count = sum(1 for w in research_words if w in q_lower)
    scores["research_value"] = min(20, 10 + research_count * 5)

    # Clarity (0-20): Well-formed question
    has_question_mark = "?" in question
    word_count = len(question.split())
    if has_question_mark and 8 <= word_count <= 30:
        scores["clarity"] = 20
    elif has_question_mark and 5 <= word_count <= 40:
        scores["clarity"] = 15
    else:
        scores["clarity"] = 10

    # Reason quality bonus (0-10)
    if reason and len(reason) >= 20:
        scores["reason_quality"] = 10
    elif reason and len(reason) >= 10:
        scores["reason_quality"] = 5
    else:
        scores["reason_quality"] = 0

    scores["total"] = sum(scores.values())
    scores["max_possible"] = 100
    return scores


def test_model_on_case(provider, model_override, test_case):
    """Test a model on a single test case."""
    messages = [
        {
            "role": "system",
            "content": "You are a research assistant. Generate follow-up research questions in JSON format."
        },
        {
            "role": "user",
            "content": f"""Based on this summary, generate 2-3 follow-up research questions.

Summary: "{test_case['summary']}"

Return JSON:
{{
  "should_suggest": true,
  "explanation": "Brief explanation of why follow-up is useful",
  "suggestions": [
    {{
      "id": "q1",
      "label": "Short label",
      "question": "The actual research question?",
      "reason": "Why this question is relevant",
      "kind": "background",
      "priority": 0.8,
      "default_selected": true
    }}
  ]
}}

Return ONLY valid JSON."""
        }
    ]

    try:
        start = time.time()
        parsed, actual_provider, actual_model = chat_json_schema(
            messages=messages,
            schema_name="TestSuggestions",
            schema=SCHEMA,
            max_tokens=800,
            reasoning_effort="low",
            temperature=0.1,
            timeout=45,
            provider=provider,
            model_override=model_override,
        )
        elapsed = time.time() - start

        suggestions = parsed.get("suggestions", [])
        quality_scores = []
        for s in suggestions:
            q_scores = assess_question_quality(
                s.get("question", ""),
                s.get("reason", ""),
                test_case["expected_topics"]
            )
            quality_scores.append(q_scores)

        avg_quality = sum(s["total"] for s in quality_scores) / len(quality_scores) if quality_scores else 0

        return {
            "success": True,
            "suggestions_count": len(suggestions),
            "avg_quality": avg_quality,
            "quality_scores": quality_scores,
            "elapsed": elapsed,
            "parsed": parsed,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "suggestions_count": 0,
            "avg_quality": 0,
            "elapsed": 0,
        }


def main():
    print("=" * 70)
    print("RIGOROUS JSON + QUALITY Model Comparison")
    print("=" * 70)
    print(f"Testing {len(MODELS)} models across {len(TEST_CASES)} topics")
    print("=" * 70)

    all_results = {}

    for provider, model_override, display_name, tier in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {display_name} ({tier})")
        print("=" * 70)

        model_results = {
            "tier": tier,
            "cases": {},
            "total_success": 0,
            "total_quality": 0,
            "total_time": 0,
        }

        for test_case in TEST_CASES:
            case_name = test_case["name"]
            print(f"\n  Topic: {case_name}")

            result = test_model_on_case(provider, model_override, test_case)
            model_results["cases"][case_name] = result

            if result["success"]:
                model_results["total_success"] += 1
                model_results["total_quality"] += result["avg_quality"]
                model_results["total_time"] += result["elapsed"]
                print(f"    ✅ Quality: {result['avg_quality']:.0f}/100 | Time: {result['elapsed']:.1f}s")
                if result.get("parsed", {}).get("suggestions"):
                    q1 = result["parsed"]["suggestions"][0].get("question", "")[:60]
                    print(f"    Q1: {q1}...")
            else:
                print(f"    ❌ Failed: {result.get('error', 'Unknown')[:50]}")

        # Calculate averages
        if model_results["total_success"] > 0:
            model_results["avg_quality"] = model_results["total_quality"] / model_results["total_success"]
            model_results["avg_time"] = model_results["total_time"] / model_results["total_success"]
        else:
            model_results["avg_quality"] = 0
            model_results["avg_time"] = 0

        model_results["success_rate"] = model_results["total_success"] / len(TEST_CASES) * 100
        all_results[display_name] = model_results

    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL RANKINGS")
    print("=" * 70)

    # Sort by combined score (quality * success_rate)
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["avg_quality"] * x[1]["success_rate"] / 100,
        reverse=True
    )

    print(f"\n{'Rank':<5} {'Model':<25} {'Tier':<8} {'Success':<10} {'Quality':<10} {'Speed':<10}")
    print("-" * 70)

    for i, (name, r) in enumerate(sorted_results, 1):
        success = f"{r['success_rate']:.0f}%"
        quality = f"{r['avg_quality']:.0f}/100"
        speed = f"{r['avg_time']:.1f}s" if r['avg_time'] > 0 else "N/A"
        tier = r['tier']

        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i:<2} {name:<23} {tier:<8} {success:<10} {quality:<10} {speed:<10}")

    # Detailed breakdown
    print("\n" + "=" * 70)
    print("QUALITY BY TOPIC")
    print("=" * 70)

    for test_case in TEST_CASES:
        case_name = test_case["name"]
        print(f"\n{case_name}:")
        case_scores = []
        for name, r in all_results.items():
            case_result = r["cases"].get(case_name, {})
            if case_result.get("success"):
                case_scores.append((name, case_result["avg_quality"]))
            else:
                case_scores.append((name, 0))

        case_scores.sort(key=lambda x: x[1], reverse=True)
        for name, score in case_scores:
            bar = "█" * int(score / 5)
            print(f"  {name:<23} {score:>5.0f} {bar}")

    # Tier winners
    print("\n" + "=" * 70)
    print("TIER WINNERS")
    print("=" * 70)

    for tier in ["budget", "mid", "free"]:
        tier_models = [(n, r) for n, r in all_results.items() if r["tier"] == tier]
        if tier_models:
            winner = max(tier_models, key=lambda x: x[1]["avg_quality"] * x[1]["success_rate"])
            print(f"\n{tier.upper()} TIER: {winner[0]}")
            print(f"  Success Rate: {winner[1]['success_rate']:.0f}%")
            print(f"  Avg Quality: {winner[1]['avg_quality']:.0f}/100")
            print(f"  Avg Speed: {winner[1]['avg_time']:.1f}s")


if __name__ == "__main__":
    main()
