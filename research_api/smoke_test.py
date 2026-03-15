"""Smoke test for the portable research service."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

from research_service import get_research_capabilities, run_research
from research_service.config import (
    BRAVE_API_KEY,
    INCEPTION_API_KEY,
    OPENROUTER_API_KEY,
    RESEARCH_PLANNER_PROVIDER,
    RESEARCH_SYNTH_PROVIDER,
    TAVILY_API_KEY,
)


def _bool_label(value: bool) -> str:
    return "yes" if value else "no"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the portable research service.")
    parser.add_argument(
        "--message",
        default="Compare recent pricing and feature differences between Cursor and Windsurf with sources.",
        help="Prompt to run through the research stack.",
    )
    parser.add_argument("--provider-mode", default="tavily", help="Research provider_mode.")
    parser.add_argument("--tool-mode", default="auto", help="Research tool_mode.")
    parser.add_argument("--depth", default="quick", help="Research depth.")
    parser.add_argument(
        "--compare",
        dest="compare",
        action="store_true",
        default=True,
        help="Enable compare mode.",
    )
    parser.add_argument(
        "--no-compare",
        dest="compare",
        action="store_false",
        help="Disable compare mode.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate imports, env loading, and capabilities without making provider calls.",
    )
    args = parser.parse_args()

    print("Portable Research smoke test")
    print(f"cwd: {ROOT}")
    print()

    print("Config")
    print(f"- INCEPTION_API_KEY present: {_bool_label(bool(INCEPTION_API_KEY))}")
    print(f"- OPENROUTER_API_KEY present: {_bool_label(bool(OPENROUTER_API_KEY))}")
    print(f"- BRAVE_API_KEY present: {_bool_label(bool(BRAVE_API_KEY))}")
    print(f"- TAVILY_API_KEY present: {_bool_label(bool(TAVILY_API_KEY))}")
    print(f"- planner provider override: {RESEARCH_PLANNER_PROVIDER}")
    print(f"- synth provider override: {RESEARCH_SYNTH_PROVIDER}")
    print(f"- dotenv file loaded: {_bool_label((ROOT / '.env').exists())}")
    print()

    caps = get_research_capabilities()
    print("Capabilities")
    print(f"- enabled: {caps.get('enabled')}")
    print(f"- llm_configured: {caps.get('llm_configured')}")
    print(f"- llm_primary: {caps.get('llm_primary')} ({caps.get('llm_primary_model')})")
    print(f"- stage overrides: {caps.get('llm_stage_overrides')}")
    print(f"- brave enabled: {caps.get('providers', {}).get('brave', {}).get('enabled')}")
    print(f"- tavily enabled: {caps.get('providers', {}).get('tavily', {}).get('enabled')}")
    print()

    if args.check_only:
        print("Check-only mode complete.")
        return 0

    print("Running research")
    print(f"- message: {args.message}")
    print(f"- provider_mode: {args.provider_mode}")
    print(f"- tool_mode: {args.tool_mode}")
    print(f"- depth: {args.depth}")
    print(f"- compare: {args.compare}")
    print()

    run = run_research(
        message=args.message,
        history=[],
        provider_mode=args.provider_mode,
        tool_mode=args.tool_mode,
        depth=args.depth,
        compare=args.compare,
        manual_tools={},
    )

    meta = run.meta or {}
    print("Result")
    print(f"- status: {run.status}")
    print(f"- planner: {meta.get('planner_llm_provider')} ({meta.get('planner_llm_model')})")
    print(f"- synthesis: {meta.get('synth_llm_provider')} ({meta.get('synth_llm_model')})")
    print(f"- queries: {meta.get('queries')}")
    print(f"- source_count: {meta.get('source_count')}")
    print(f"- provider_chain: {meta.get('provider_chain')}")
    if meta.get("errors"):
        print(f"- errors: {meta.get('errors')}")
    print()

    answer_preview = (run.answer or "").strip()
    if len(answer_preview) > 900:
        answer_preview = f"{answer_preview[:900].rstrip()}..."
    print("Answer preview")
    print(answer_preview)
    print()

    if meta.get("planner_llm_provider") == "unknown" or meta.get("synth_llm_provider") == "unknown":
        print("Warning: provider metadata was not populated as expected.")
        return 1

    if meta.get("synth_llm_provider") == "fallback":
        print("Warning: synthesis fell back to deterministic mode.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
