#!/usr/bin/env python3
"""Rigorous benchmark for follow-up suggestion models.

This tool is intentionally stricter than the earlier ad-hoc comparison:
- uses the real production prompt and schema
- validates responses with jsonschema against the production schema
- measures repeated trials with randomized case order
- scores recommendation calibration separately from schema compliance

It is still a heuristic benchmark, but it is materially closer to runtime
behavior than the earlier single-pass keyword test.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import jsonschema

sys.path.insert(0, "/app")

from research_api.research_service.follow_up import (  # noqa: E402
    SUGGESTIONS_SCHEMA,
    SUGGESTIONS_SYSTEM_PROMPT,
)
from research_api.research_service.llm import chat_json_schema  # noqa: E402


MODELS = [
    ("openrouter", "google/gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite", "budget"),
    ("openrouter", "openai/gpt-5.4-nano", "gpt-5.4-nano", "budget"),
    ("openrouter", "google/gemini-3-flash-preview", "gemini-3-flash-preview", "mid"),
    ("openrouter", "openai/gpt-5.4-mini", "gpt-5.4-mini", "mid"),
    ("inception", None, "mercury-2", "free"),
]

GENERIC_STARTS = (
    "what is",
    "tell me about",
    "describe",
    "more about",
)
FRESHNESS_TERMS = (
    "current",
    "latest",
    "recent",
    "pricing",
    "changed",
    "update",
    "today",
    "compare",
    "vs",
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    summary: str
    source_context: dict[str, Any]
    expected_should_suggest: bool
    expected_topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    notes: str = ""


CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        name="Pricing review (time-sensitive)",
        summary=(
            "This 2024 review compares Cursor and other AI coding tools, focusing on "
            "pricing, free-tier limits, and how well each tool handles large codebases."
        ),
        source_context={
            "title": "Cursor AI review and pricing breakdown",
            "type": "youtube",
            "url": "https://youtube.com/watch?v=abc123def45",
            "published_at": "2024-06-01T00:00:00Z",
        },
        expected_should_suggest=True,
        expected_topics=["pricing", "limits", "comparison", "alternatives"],
        entities=["Cursor", "AI coding tools"],
        notes="Should reward current-state and comparison questions.",
    ),
    BenchmarkCase(
        name="Business study (time-sensitive)",
        summary=(
            "A 2024 Stanford study found fully remote workers were 18% less productive "
            "than office workers, while hybrid workers showed no measurable difference."
        ),
        source_context={
            "title": "Remote work productivity study",
            "type": "web",
            "url": "https://example.com/remote-work-study",
            "published_at": "2024-09-15T00:00:00Z",
        },
        expected_should_suggest=True,
        expected_topics=["productivity", "hybrid", "replication", "current"],
        entities=["Stanford", "remote work"],
    ),
    BenchmarkCase(
        name="Health claim (needs follow-up)",
        summary=(
            "New sleep research suggests the glymphatic system clears toxic proteins much "
            "faster during sleep, which may help explain links between poor sleep and "
            "neurodegenerative disease."
        ),
        source_context={
            "title": "Sleep and glymphatic system research",
            "type": "web",
            "url": "https://example.com/sleep-glymphatic-study",
            "published_at": "2024-11-10T00:00:00Z",
        },
        expected_should_suggest=True,
        expected_topics=["mechanism", "evidence", "replication", "disease"],
        entities=["glymphatic system", "Alzheimer's"],
    ),
    BenchmarkCase(
        name="Evergreen explainer (should not suggest)",
        summary=(
            "This lesson explains how photosynthesis works: plants use sunlight, water, "
            "and carbon dioxide to produce glucose and oxygen through light-dependent and "
            "light-independent reactions."
        ),
        source_context={
            "title": "Photosynthesis explained",
            "type": "youtube",
            "url": "https://youtube.com/watch?v=photosynth01",
            "published_at": "2026-03-10T00:00:00Z",
        },
        expected_should_suggest=False,
        expected_topics=["photosynthesis"],
        notes="Production prompt says evergreen educational content should usually not suggest.",
    ),
    BenchmarkCase(
        name="Opinion piece (should not suggest)",
        summary=(
            "The author argues that deep work matters more than hustle culture and reflects "
            "on personal writing habits, focus rituals, and why boredom can be productive."
        ),
        source_context={
            "title": "Why I stopped chasing hustle culture",
            "type": "web",
            "url": "https://example.com/deep-work-opinion",
            "published_at": "2026-02-10T00:00:00Z",
        },
        expected_should_suggest=False,
        expected_topics=["writing", "focus"],
    ),
    BenchmarkCase(
        name="Vague summary (should not suggest)",
        summary="Something about new batteries.",
        source_context={
            "title": "Battery update",
            "type": "web",
            "url": "https://example.com/battery-update",
            "published_at": "2026-03-01T00:00:00Z",
        },
        expected_should_suggest=False,
        expected_topics=["clarification"],
        notes="Good models should avoid pretending they have enough context.",
    ),
]


@dataclass
class RunResult:
    success: bool
    provider: str
    model: str
    latency_seconds: float
    schema_valid: bool
    should_suggest: bool | None
    calibration_correct: bool
    utility_score: float
    composite_score: float
    error: str | None = None
    suggestions_count: int = 0
    parsed: dict[str, Any] | None = None
    schema_errors: list[str] = field(default_factory=list)


def _collect_missing_additional_properties(schema: dict[str, Any], path: str = "$") -> list[str]:
    missing: list[str] = []
    if not isinstance(schema, dict):
        return missing

    schema_type = schema.get("type")
    if schema_type == "object" and schema.get("properties") and schema.get("additionalProperties") is not False:
        missing.append(path)

    for key, value in (schema.get("properties") or {}).items():
        missing.extend(_collect_missing_additional_properties(value, f"{path}.properties.{key}"))

    items = schema.get("items")
    if isinstance(items, dict):
        missing.extend(_collect_missing_additional_properties(items, f"{path}.items"))

    for key in ("allOf", "anyOf", "oneOf"):
        for idx, value in enumerate(schema.get(key) or []):
            missing.extend(_collect_missing_additional_properties(value, f"{path}.{key}[{idx}]"))

    return missing


def _build_messages(case: BenchmarkCase, max_suggestions: int) -> list[dict[str, str]]:
    ctx = case.source_context
    user_context = [
        f"Title: {ctx.get('title', 'Untitled')}",
        f"Type: {ctx.get('type', 'unknown')}",
    ]
    if ctx.get("url"):
        user_context.append(f"URL: {ctx['url']}")
    if ctx.get("published_at"):
        user_context.append(f"Published at: {ctx['published_at']}")
    if case.entities:
        user_context.append(f"Identified entities: {', '.join(case.entities[:5])}")

    return [
        {"role": "system", "content": SUGGESTIONS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n".join(
                [
                    *user_context,
                    "",
                    "Summary:",
                    case.summary[:3000],
                    "",
                    f"Generate up to {max_suggestions} follow-up research suggestions.",
                ]
            ),
        },
    ]


def _normalize_question(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _score_utility(parsed: dict[str, Any], case: BenchmarkCase) -> float:
    suggestions = parsed.get("suggestions") or []
    should_suggest = bool(parsed.get("should_suggest"))
    if not case.expected_should_suggest:
        return 100.0 if not should_suggest and not suggestions else 0.0
    if not suggestions:
        return 0.0

    normalized_questions = [_normalize_question(str(item.get("question") or "")) for item in suggestions]
    unique_questions = len({q for q in normalized_questions if q})

    topic_matches = 0
    for topic in case.expected_topics:
        topic_lower = topic.lower()
        if any(topic_lower in question for question in normalized_questions):
            topic_matches += 1
    coverage_ratio = topic_matches / len(case.expected_topics) if case.expected_topics else 1.0

    question_form_scores: list[float] = []
    reason_scores: list[float] = []
    freshness_scores: list[float] = []
    for item in suggestions:
        question = str(item.get("question") or "").strip()
        reason = str(item.get("reason") or "").strip()
        q_lower = question.lower()
        word_count = len(question.split())

        form = 0.0
        if "?" in question:
            form += 0.4
        if 8 <= word_count <= 28:
            form += 0.3
        if not any(q_lower.startswith(prefix) for prefix in GENERIC_STARTS):
            form += 0.3
        question_form_scores.append(min(1.0, form))

        reason_score = 0.0
        if len(reason) >= 12:
            reason_score += 0.5
        if len(reason) >= 30:
            reason_score += 0.3
        if any(token in reason.lower() for token in ("because", "helps", "important", "useful", "relevant")):
            reason_score += 0.2
        reason_scores.append(min(1.0, reason_score))

        freshness_score = 1.0 if any(token in q_lower for token in FRESHNESS_TERMS) else 0.0
        freshness_scores.append(freshness_score)

    diversity_ratio = unique_questions / max(1, len(suggestions))
    avg_form = statistics.mean(question_form_scores) if question_form_scores else 0.0
    avg_reason = statistics.mean(reason_scores) if reason_scores else 0.0

    freshness_weight = 0.15 if case.expected_should_suggest else 0.0
    freshness_component = (
        statistics.mean(freshness_scores) * freshness_weight if freshness_scores and freshness_weight else 0.0
    )

    score = (
        coverage_ratio * 0.40
        + avg_form * 0.25
        + avg_reason * 0.15
        + diversity_ratio * 0.20
        + freshness_component
    )
    return min(100.0, round(score * 100.0, 2))


def _validate_schema(parsed: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(SUGGESTIONS_SCHEMA)
    errors = []
    for err in validator.iter_errors(parsed):
        path = ".".join(str(seg) for seg in err.absolute_path) or "$"
        errors.append(f"{path}: {err.message}")
    return errors


def _run_single(
    *,
    case: BenchmarkCase,
    provider: str,
    model_override: str | None,
    max_suggestions: int,
    max_tokens: int,
    timeout: float,
) -> RunResult:
    started = time.time()
    try:
        parsed, actual_provider, actual_model = chat_json_schema(
            messages=_build_messages(case, max_suggestions),
            schema_name="FollowUpSuggestionsBenchmark",
            schema=SUGGESTIONS_SCHEMA,
            max_tokens=max_tokens,
            reasoning_effort="low",
            temperature=0.1,
            timeout=timeout,
            provider=provider,
            model_override=model_override,
        )
        latency = time.time() - started
    except Exception as exc:
        return RunResult(
            success=False,
            provider=provider,
            model=model_override or "default",
            latency_seconds=time.time() - started,
            schema_valid=False,
            should_suggest=None,
            calibration_correct=False,
            utility_score=0.0,
            composite_score=0.0,
            error=str(exc),
        )

    schema_errors = _validate_schema(parsed)
    schema_valid = not schema_errors
    should_suggest = bool(parsed.get("should_suggest")) if "should_suggest" in parsed else None
    calibration_correct = should_suggest is case.expected_should_suggest
    utility_score = _score_utility(parsed, case)

    composite = 0.0
    composite += 40.0 if schema_valid else 0.0
    composite += 25.0 if calibration_correct else 0.0
    composite += utility_score * 0.35

    return RunResult(
        success=True,
        provider=actual_provider,
        model=actual_model,
        latency_seconds=latency,
        schema_valid=schema_valid,
        should_suggest=should_suggest,
        calibration_correct=calibration_correct,
        utility_score=utility_score,
        composite_score=round(composite, 2),
        suggestions_count=len(parsed.get("suggestions") or []),
        parsed=parsed,
        schema_errors=schema_errors,
    )


def _summarize_model(results: list[RunResult]) -> dict[str, Any]:
    successful = [r for r in results if r.success]
    schema_valid = [r for r in successful if r.schema_valid]
    calibrations = [r for r in successful if r.calibration_correct]
    latencies = [r.latency_seconds for r in successful]
    utility = [r.utility_score for r in successful]
    composite = [r.composite_score for r in successful]

    return {
        "runs": len(results),
        "successful_runs": len(successful),
        "success_rate": len(successful) / len(results) if results else 0.0,
        "schema_valid_rate": len(schema_valid) / len(successful) if successful else 0.0,
        "calibration_rate": len(calibrations) / len(successful) if successful else 0.0,
        "avg_latency": statistics.mean(latencies) if latencies else None,
        "median_latency": statistics.median(latencies) if latencies else None,
        "avg_utility": statistics.mean(utility) if utility else 0.0,
        "avg_composite": statistics.mean(composite) if composite else 0.0,
        "stdev_composite": statistics.pstdev(composite) if len(composite) > 1 else 0.0,
    }


def _print_schema_findings() -> None:
    missing = _collect_missing_additional_properties(SUGGESTIONS_SCHEMA)
    print("=" * 78)
    print("PRODUCTION SCHEMA CHECK")
    print("=" * 78)
    if not missing:
        print("Production suggestion schema already looks OpenAI strict-compatible.")
    else:
        print("Production suggestion schema is not strict-compatible yet.")
        print("Object nodes missing `additionalProperties: false`:")
        for path in missing:
            print(f"  - {path}")
    print()


def _print_case_list(cases: list[BenchmarkCase]) -> None:
    print("Benchmark cases:")
    for case in cases:
        expectation = "suggest" if case.expected_should_suggest else "do-not-suggest"
        print(f"  - {case.name}: {expectation}")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3, help="Number of randomized passes per model")
    parser.add_argument("--seed", type=int, default=20260317, help="Random seed for case order")
    parser.add_argument("--max-suggestions", type=int, default=3, help="Maximum suggestions requested per prompt")
    parser.add_argument("--max-tokens", type=int, default=420, help="Token cap per request")
    parser.add_argument("--timeout", type=float, default=45.0, help="Per-request timeout in seconds")
    parser.add_argument("--model", action="append", default=[], help="Limit to one or more display names")
    parser.add_argument("--case", action="append", default=[], help="Limit to one or more case names")
    parser.add_argument("--output-json", type=str, default="", help="Optional path for raw results JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    selected_models = [m for m in MODELS if not args.model or m[2] in set(args.model)]
    selected_cases = [c for c in CASES if not args.case or c.name in set(args.case)]

    if not selected_models:
        print("No models matched --model filters.")
        return 2
    if not selected_cases:
        print("No cases matched --case filters.")
        return 2

    _print_schema_findings()
    _print_case_list(selected_cases)

    rng = random.Random(args.seed)
    raw_results: dict[str, dict[str, Any]] = {}

    print("=" * 78)
    print("FOLLOW-UP SUGGESTION BENCHMARK")
    print("=" * 78)
    print(f"Trials per model: {args.trials}")
    print(f"Cases per trial: {len(selected_cases)}")
    print(f"Total runs per model: {args.trials * len(selected_cases)}")
    print()

    for provider, model_override, display_name, tier in selected_models:
        print(f"[MODEL] {display_name} ({tier})")
        model_runs: list[RunResult] = []

        for trial_index in range(args.trials):
            case_order = selected_cases[:]
            rng.shuffle(case_order)
            print(f"  Trial {trial_index + 1}/{args.trials}: {[case.name for case in case_order]}")
            for case in case_order:
                result = _run_single(
                    case=case,
                    provider=provider,
                    model_override=model_override,
                    max_suggestions=args.max_suggestions,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                model_runs.append(result)

                if result.success:
                    schema_label = "schema-ok" if result.schema_valid else "schema-fail"
                    calibration_label = "decision-ok" if result.calibration_correct else "decision-miss"
                    print(
                        f"    {case.name}: {schema_label}, {calibration_label}, "
                        f"utility={result.utility_score:.1f}, latency={result.latency_seconds:.1f}s"
                    )
                    if result.schema_errors:
                        print(f"      schema errors: {result.schema_errors[0]}")
                else:
                    print(f"    {case.name}: failed ({result.error})")

        summary = _summarize_model(model_runs)
        raw_results[display_name] = {
            "provider": provider,
            "model_override": model_override,
            "tier": tier,
            "summary": summary,
            "runs": [
                {
                    "success": run.success,
                    "provider": run.provider,
                    "model": run.model,
                    "latency_seconds": run.latency_seconds,
                    "schema_valid": run.schema_valid,
                    "should_suggest": run.should_suggest,
                    "calibration_correct": run.calibration_correct,
                    "utility_score": run.utility_score,
                    "composite_score": run.composite_score,
                    "suggestions_count": run.suggestions_count,
                    "error": run.error,
                    "schema_errors": run.schema_errors,
                    "parsed": run.parsed,
                }
                for run in model_runs
            ],
        }
        print()

    rankings = sorted(
        raw_results.items(),
        key=lambda item: (
            item[1]["summary"]["avg_composite"],
            item[1]["summary"]["schema_valid_rate"],
            item[1]["summary"]["calibration_rate"],
            -(item[1]["summary"]["median_latency"] or 999.0),
        ),
        reverse=True,
    )

    print("=" * 78)
    print("FINAL RANKINGS")
    print("=" * 78)
    print(
        f"{'Rank':<5} {'Model':<24} {'Tier':<8} {'Success':<9} "
        f"{'Schema':<9} {'Decision':<9} {'Utility':<8} {'Latency':<8} {'Composite':<10}"
    )
    print("-" * 78)
    for index, (name, payload) in enumerate(rankings, start=1):
        summary = payload["summary"]
        print(
            f"{index:<5} {name:<24} {payload['tier']:<8} "
            f"{summary['success_rate']*100:>6.0f}%   "
            f"{summary['schema_valid_rate']*100:>6.0f}%   "
            f"{summary['calibration_rate']*100:>6.0f}%   "
            f"{summary['avg_utility']:>6.1f}   "
            f"{(summary['median_latency'] or 0):>6.1f}s  "
            f"{summary['avg_composite']:>8.2f}"
        )

    print()
    print("Composite score weights:")
    print("  - 40% production schema validity")
    print("  - 25% should_suggest calibration")
    print("  - 35% utility heuristic")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(raw_results, handle, indent=2, ensure_ascii=False)
        print()
        print(f"Saved raw results to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
