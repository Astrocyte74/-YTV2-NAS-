"""Grounded answer synthesis for research results."""

from __future__ import annotations

import logging
from collections import defaultdict
import re
import time

from .config import (
    OPENROUTER_API_KEY,
    RESEARCH_FALLBACK_ENABLED,
    RESEARCH_SYNTH_PROVIDER,
    SYNTH_RETRY_ATTEMPTS,
    SYNTH_RETRY_DELAY_SECONDS,
    SYNTH_CONTINUATION_MAX_TOKENS,
    SYNTH_MAX_CONTINUATIONS,
    SYNTH_MAX_TOKENS,
    SYNTH_TIMEOUT_SECONDS,
)
from .llm import chat_text_with_metadata, stream_inception_chat
from .models import ResearchBatchResult, ResearchSource

logger = logging.getLogger(__name__)


def _call_synth_llm(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    provider: str,
) -> dict[str, str | None]:
    attempts = max(1, SYNTH_RETRY_ATTEMPTS)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return chat_text_with_metadata(
                messages=messages,
                max_tokens=max_tokens,
                reasoning_effort="low",
                temperature=0.1,
                timeout=timeout,
                provider=provider,
            )
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                logger.warning(
                    "Synthesis attempt %s/%s failed via %s: %s",
                    attempt,
                    attempts,
                    provider,
                    exc,
                )
                time.sleep(SYNTH_RETRY_DELAY_SECONDS * attempt)

    if (
        provider == "inception"
        and RESEARCH_FALLBACK_ENABLED
        and OPENROUTER_API_KEY
    ):
        logger.warning("Synthesis failed via inception after %s attempts; trying OpenRouter fallback", attempts)
        return chat_text_with_metadata(
            messages=messages,
            max_tokens=max_tokens,
            reasoning_effort="low",
            temperature=0.1,
            timeout=timeout,
            provider="openrouter",
        )

    if last_error is not None:
        raise last_error
    raise RuntimeError("Synthesis call failed without an exception")


def _build_evidence_block(batches: list[ResearchBatchResult], max_items: int = 30) -> str:
    lines: list[str] = []
    count = 0
    for batch in batches:
        lines.append(f"[Provider={batch.provider} Tool={batch.tool} Query={batch.query}]")
        for item in batch.results:
            lines.append(f"- title: {item.title}")
            lines.append(f"  url: {item.url}")
            lines.append(f"  snippet: {item.snippet}")
            if item.published_at:
                lines.append(f"  published_at: {item.published_at}")
            count += 1
            if count >= max_items:
                break
        if count >= max_items:
            break
    return "\n".join(lines)


def deterministic_fallback_answer(
    *,
    user_message: str,
    batches: list[ResearchBatchResult],
    sources: list[ResearchSource],
    compare: bool,
) -> tuple[str, dict[str, str]]:
    """Generate a deterministic fallback answer when LLM synthesis fails.

    Returns:
        Tuple of (answer_text, llm_info) where llm_info indicates fallback.
    """
    if not batches:
        return (
            "I couldn't retrieve enough web results right now to answer reliably. "
            "Please try again or refine the query.",
            {"llm_provider": "fallback", "llm_model": "deterministic"},
        )

    grouped: dict[str, list[ResearchBatchResult]] = defaultdict(list)
    for batch in batches:
        grouped[batch.provider].append(batch)

    lines = [f"Research summary for: {user_message}"]

    if compare:
        lines.append("")
        lines.append("Provider comparison:")
        for provider, provider_batches in grouped.items():
            item_count = sum(len(b.results) for b in provider_batches)
            lines.append(f"- {provider}: {item_count} results across {len(provider_batches)} tool runs")

    lines.append("")
    lines.append("Top findings:")
    seen = 0
    for batch in batches:
        for item in batch.results:
            lines.append(f"- {item.title}: {item.snippet[:220]} ({item.url})")
            seen += 1
            if seen >= 8:
                break
        if seen >= 8:
            break

    if sources:
        lines.append("")
        lines.append("Sources:")
        for src in sources[:12]:
            lines.append(f"- {src.url}")

    return "\n".join(lines), {"llm_provider": "fallback", "llm_model": "deterministic"}


def _looks_truncated(text: str) -> bool:
    clean = (text or "").rstrip()
    if not clean:
        return False
    if clean.endswith((":", ",", ";", "-", "(", "[", "{", "with the", "with a", "and the", "and a")):
        return True
    if clean.count("|") % 2 == 1:
        return True
    last_line = clean.splitlines()[-1].strip().lower()
    if last_line and len(last_line.split()) <= 4 and last_line not in {"sources:", "source attribution:"}:
        return True
    return False


def _message_needs_history(user_message: str) -> bool:
    text = (user_message or "").strip().lower()
    if not text:
        return False

    referential_markers = (
        " it ",
        " its ",
        " they ",
        " them ",
        " their ",
        " that ",
        " those ",
        " these ",
        " this ",
        " same ",
        " previous ",
        " above ",
        " earlier ",
    )
    padded = f" {text} "
    if any(marker in padded for marker in referential_markers):
        return True

    return text.startswith((
        "what about",
        "how about",
        "and ",
        "also ",
        "compare that",
        "compare those",
        "compare them",
        "research that",
        "research those",
        "research them",
    ))


def _build_history_block(
    *,
    user_message: str,
    history: list[dict[str, str]] | None,
) -> str:
    if not history or not _message_needs_history(user_message):
        return ""

    user_turns: list[str] = []
    for turn in history[-6:]:
        role = turn.get("role", "")
        content = str(turn.get("content", "")).strip()
        if role == "user" and content:
            user_turns.append(content)

    if not user_turns:
        return ""

    recent_user_turns = user_turns[-3:]
    return "\n".join(f"user: {turn}" for turn in recent_user_turns)


_URL_RE = re.compile(r"https?://[^\s)>\]]+")


def _is_sources_heading(line: str) -> bool:
    normalized = (
        (line or "")
        .strip()
        .replace("#", "", 1) if (line or "").strip().startswith("#") else (line or "").strip()
    )
    normalized = re.sub(r"^#+\s*", "", normalized)
    normalized = normalized.replace("*", "").replace("_", "").replace("`", "")
    normalized = re.sub(r"\s+", " ", normalized).rstrip(":").strip().lower()
    return normalized in {"sources", "source attribution"}


def _normalize_sources_section(text: str, sources: list[ResearchSource]) -> str:
    lines = text.splitlines()
    body_lines: list[str] = []
    collected_urls: list[str] = [src.url for src in sources if getattr(src, "url", "")]

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if _is_sources_heading(stripped):
            i += 1
            while i < len(lines):
                current = lines[i]
                current_stripped = current.strip()
                if _is_sources_heading(current_stripped):
                    i += 1
                    continue
                if current_stripped:
                    collected_urls.extend(_URL_RE.findall(current_stripped))
                if (
                    current_stripped
                    and not _URL_RE.search(current_stripped)
                    and not re.match(r"^[-*]\s+", current_stripped)
                    and not re.match(r"^\d+\.\s+", current_stripped)
                ):
                    break
                i += 1
            continue
        body_lines.append(lines[i])
        i += 1

    deduped_urls: list[str] = []
    seen: set[str] = set()
    for url in collected_urls:
        clean = url.rstrip(".,);]")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped_urls.append(clean)

    body = "\n".join(body_lines).strip()
    if deduped_urls:
        source_block = "Sources:\n" + "\n".join(f"- {url}" for url in deduped_urls[:20])
        return f"{body}\n\n{source_block}" if body else source_block
    return body


def synthesize_answer(
    *,
    user_message: str,
    history: list[dict[str, str]] | None,
    batches: list[ResearchBatchResult],
    sources: list[ResearchSource],
    compare: bool,
) -> tuple[str, dict[str, str]]:
    """Synthesize a grounded answer from research batches.

    Returns:
        Tuple of (answer_text, llm_info) where llm_info contains provider/model.
    """
    if not batches:
        return deterministic_fallback_answer(
            user_message=user_message,
            batches=batches,
            sources=sources,
            compare=compare,
        )

    evidence = _build_evidence_block(batches)
    source_urls = "\n".join(f"- {s.url}" for s in sources[:20])

    history_block = _build_history_block(user_message=user_message, history=history)

    compare_instruction = (
        "The user requested comparison mode. Include a short provider/tool comparison section."
        if compare
        else ""
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research synthesis assistant. Use ONLY the provided evidence. "
                "Do not fabricate facts. If evidence is thin/conflicting, say so clearly. "
                "Always include a final 'Sources' section with raw URLs. "
                "Use recent context only to resolve references in the current user request. "
                "Never continue, restate, or elaborate on a prior answer unless the current request explicitly asks for that."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User request: {user_message}\n\n"
                f"Recent context:\n{history_block or '(none)'}\n\n"
                f"Evidence:\n{evidence}\n\n"
                f"Known source URLs:\n{source_urls}\n\n"
                f"{compare_instruction}"
            ),
        },
    ]

    try:
        result = _call_synth_llm(
            messages=messages,
            max_tokens=SYNTH_MAX_TOKENS,
            timeout=SYNTH_TIMEOUT_SECONDS,
            provider=RESEARCH_SYNTH_PROVIDER,
        )
        out = str(result.get("content") or "")
        if not out.strip():
            raise RuntimeError("empty synthesis output")

        # Capture LLM provider info
        llm_provider = str(result.get("llm_provider") or "unknown")
        llm_model = str(result.get("llm_model") or "unknown")
        logger.info("Synthesis completed via %s (%s)", llm_provider, llm_model)

        finish_reason = str(result.get("finish_reason") or "")
        continuations_used = 0
        while continuations_used < SYNTH_MAX_CONTINUATIONS and (
            finish_reason == "length" or _looks_truncated(out)
        ):
            continuation_messages = [
                messages[0],
                messages[1],
                {"role": "assistant", "content": out},
                {
                    "role": "user",
                    "content": (
                        "Continue exactly where you left off. Do not restart, repeat prior sections, "
                        "or emit another Sources heading if one already exists. "
                        "Finish the report cleanly and include the final Sources section only once."
                    ),
                },
            ]
            continuation = _call_synth_llm(
                messages=continuation_messages,
                max_tokens=SYNTH_CONTINUATION_MAX_TOKENS,
                timeout=SYNTH_TIMEOUT_SECONDS,
                provider=RESEARCH_SYNTH_PROVIDER,
            )
            chunk = str(continuation.get("content") or "").strip()
            finish_reason = str(continuation.get("finish_reason") or "")
            if not chunk:
                break
            out = f"{out.rstrip()}\n{chunk.lstrip()}"
            continuations_used += 1

        out = _normalize_sources_section(out, sources)
        if "http" not in out:
            out = f"{out}\n\nSources:\n{source_urls}"
        return out, {"llm_provider": llm_provider, "llm_model": llm_model}
    except Exception as e:
        logger.warning("Synthesis LLM call failed, using fallback: %s", e)
        return deterministic_fallback_answer(
            user_message=user_message,
            batches=batches,
            sources=sources,
            compare=compare,
        )


def synthesize_follow_up(
    *,
    source_context: dict,
    summary: str,
    approved_questions: list[str],
    question_kinds: list[str],
    batches: list[ResearchBatchResult],
    sources: list[ResearchSource],
    compare: bool = False,
) -> tuple[str, dict[str, str]]:
    """Synthesize a follow-up research report with sectioned answers per question.

    Unlike synthesize_answer which answers a single query, this function
    generates a structured report with explicit sections for each approved
    follow-up question.

    Args:
        source_context: Information about the original content (url, type, title, etc.)
        summary: The existing summary text
        approved_questions: User-approved research directions (max 3)
        question_kinds: Question category for each approved question
        batches: Research execution results from follow-up queries
        sources: Deduped source list from all batches
        compare: Whether to include provider comparison

    Returns:
        Tuple of (report_text, llm_info) where llm_info contains provider/model.
    """
    if not batches:
        # Fallback for no research results
        lines = [
            "# Follow-Up Research Report",
            "",
            f"**Source:** {source_context.get('title', 'Unknown')}",
            f"**URL:** {source_context.get('url', 'N/A')}",
            "",
            "## Research Questions",
            *(f"- {q}" for q in approved_questions),
            "",
            "I couldn't retrieve enough current web results to answer these questions reliably. "
            "Please try again or refine the questions.",
        ]
        if sources:
            lines.extend(["", "Sources:", *(f"- {s.url}" for s in sources[:10])])
        return "\n".join(lines), {"llm_provider": "fallback", "llm_model": "deterministic"}

    evidence = _build_evidence_block(batches)
    source_urls = "\n".join(f"- {s.url}" for s in sources[:20])

    # Build question sections
    question_sections = "\n".join(
        f"## {i+1}. {question}\n"
        for i, question in enumerate(approved_questions)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a follow-up research synthesis assistant. Your job is to write a clear, "
                "well-structured report that answers EACH of the user's approved follow-up questions "
                "using ONLY the provided web research evidence.\n\n"
                "Required format:\n"
                "# Follow-Up Research Report\n"
                "## Executive Summary\n"
                "Brief 2-3 sentence overview of findings.\n\n"
                "## 1. First Approved Question\n"
                "Detailed answer with inline citations.\n\n"
                "## 2. Second Approved Question\n"
                "Detailed answer with inline citations.\n\n"
                "(Continue for all approved questions)\n\n"
                "## Sources\n"
                "Full list of URLs.\n\n"
                "Rules:\n"
                "- Answer EVERY approved question in its own section\n"
                "- Use ONLY the provided evidence (do not fabricate)\n"
                "- Be specific about what changed or what's current\n"
                "- Include timestamps/dates when available\n"
                "- If evidence is thin for a question, say so clearly"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original Content:\n"
                f"- Title: {source_context.get('title', 'Unknown')}\n"
                f"- URL: {source_context.get('url', 'N/A')}\n"
                f"- Published: {source_context.get('published_at', 'Unknown')}\n"
                f"- Type: {source_context.get('type', 'unknown')}\n\n"
                f"Existing Summary:\n{summary[:2000]}\n\n"
                f"Approved Follow-Up Questions:\n"
                f"{question_sections}\n\n"
                f"Research Evidence:\n{evidence}\n\n"
                f"Source URLs:\n{source_urls}\n\n"
                "Generate the follow-up research report."
            ),
        },
    ]

    compare_instruction = (
        "\n\nInclude a brief 'Research Methodology' section comparing provider/tool results."
        if compare
        else ""
    )

    if compare_instruction:
        messages[1]["content"] += compare_instruction

    try:
        result = _call_synth_llm(
            messages=messages,
            max_tokens=SYNTH_MAX_TOKENS,
            timeout=SYNTH_TIMEOUT_SECONDS,
            provider=RESEARCH_SYNTH_PROVIDER,
        )
        out = str(result.get("content") or "")
        if not out.strip():
            raise RuntimeError("empty follow-up synthesis output")

        llm_provider = str(result.get("llm_provider") or "unknown")
        llm_model = str(result.get("llm_model") or "unknown")
        logger.info("Follow-up synthesis completed via %s (%s)", llm_provider, llm_model)

        finish_reason = str(result.get("finish_reason") or "")
        continuations_used = 0
        while continuations_used < SYNTH_MAX_CONTINUATIONS and (
            finish_reason == "length" or _looks_truncated(out)
        ):
            continuation_messages = [
                messages[0],
                messages[1],
                {"role": "assistant", "content": out},
                {
                    "role": "user",
                    "content": (
                        "Continue exactly where you left off. Do not restart or repeat prior sections. "
                        "Finish the report cleanly, ensuring all approved questions are answered "
                        "and the Sources section appears only once at the end."
                    ),
                },
            ]
            continuation = _call_synth_llm(
                messages=continuation_messages,
                max_tokens=SYNTH_CONTINUATION_MAX_TOKENS,
                timeout=SYNTH_TIMEOUT_SECONDS,
                provider=RESEARCH_SYNTH_PROVIDER,
            )
            chunk = str(continuation.get("content") or "").strip()
            finish_reason = str(continuation.get("finish_reason") or "")
            if not chunk:
                break
            out = f"{out.rstrip()}\n{chunk.lstrip()}"
            continuations_used += 1

        out = _normalize_sources_section(out, sources)
        if "http" not in out:
            out = f"{out}\n\nSources:\n{source_urls}"
        return out, {"llm_provider": llm_provider, "llm_model": llm_model}
    except Exception as e:
        logger.warning("Follow-up synthesis LLM call failed, using fallback: %s", e)
        return deterministic_fallback_answer(
            user_message="; ".join(approved_questions),
            batches=batches,
            sources=sources,
            compare=compare,
        )


def answer_report_chat(
    *,
    source_context: dict,
    report_answer: str,
    report_sources: list[ResearchSource],
    user_question: str,
    history: list[dict[str, str]] | None = None,
    thread_turns: list[dict] | None = None,
) -> tuple[str, dict[str, str]]:
    """Answer a lightweight follow-up question using only existing report context.

    This path does not run fresh web research. It is intended for report-grounded
    clarification and drill-down questions inside the WebUI chat surface.
    """
    report_text = str(report_answer or "").strip()
    if not report_text:
        return (
            "I don't have an existing Deep Research report to answer from. Run Deep Research first.",
            {"llm_provider": "fallback", "llm_model": "deterministic"},
        )

    history_lines: list[str] = []
    for turn in (history or [])[-8:]:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines) or "(none)"

    thread_lines: list[str] = []
    for index, turn in enumerate((thread_turns or [])[-6:], start=max(1, len(thread_turns or []) - 5)):
        approved_questions = [str(item).strip() for item in (turn.get("approved_questions") or []) if str(item).strip()]
        if approved_questions:
            thread_lines.append(f"Turn {index} questions: {'; '.join(approved_questions)}")
        excerpt = str(turn.get("answer") or "").strip()
        if excerpt:
            thread_lines.append(f"Turn {index} excerpt:\n{excerpt[:1500]}")
    thread_block = "\n\n".join(thread_lines) or "(none)"

    source_urls = "\n".join(f"- {src.url}" for src in report_sources[:20]) or "(none)"
    truncated_report = report_text[:12000]

    # Detect whether we have a full research report or just a summary
    has_research = bool(report_sources) or "research" in (source_context.get("type") or "").lower()
    context_label = "Deep Research report" if has_research else "content summary"

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a grounded assistant answering questions about an existing {context_label}. "
                "Use ONLY the provided context text, prior thread context, and source URLs. "
                "Do not claim to have checked the live web. If the user asks for current/latest/fresh information "
                "or for facts not supported by the context, say that a fresh Deep Research run is needed. "
                "Answer directly, stay concise, and include a short 'Sources' section only when useful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source title: {source_context.get('title', 'Unknown')}\n"
                f"Source URL: {source_context.get('url', 'N/A')}\n"
                f"Source type: {source_context.get('type', 'unknown')}\n\n"
                f"Persisted research thread context:\n{thread_block}\n\n"
                f"Current {context_label}:\n{truncated_report}\n\n"
                f"Known source URLs:\n{source_urls}\n\n"
                f"Recent chat:\n{history_block}\n\n"
                f"User question: {user_question}"
            ),
        },
    ]

    try:
        result = _call_synth_llm(
            messages=messages,
            max_tokens=min(SYNTH_MAX_TOKENS, 1400),
            timeout=SYNTH_TIMEOUT_SECONDS,
            provider=RESEARCH_SYNTH_PROVIDER,
        )
        out = str(result.get("content") or "").strip()
        if not out:
            raise RuntimeError("empty report-chat synthesis output")
        out = _normalize_sources_section(out, report_sources)
        return out, {
            "llm_provider": str(result.get("llm_provider") or "unknown"),
            "llm_model": str(result.get("llm_model") or "unknown"),
        }
    except Exception as e:
        logger.warning("Report chat synthesis failed, using fallback: %s", e)
        source_tail = f"\n\nSources:\n{source_urls}" if report_sources else ""
        return (
            "I couldn't answer that confidently from the current Deep Research report alone. "
            "Run fresh Deep Research if you need newer evidence or broader coverage."
            f"{source_tail}",
            {"llm_provider": "fallback", "llm_model": "deterministic"},
        )


def stream_report_chat(
    *,
    source_context: dict,
    report_answer: str,
    report_sources: list[ResearchSource],
    user_question: str,
    history: list[dict[str, str]] | None = None,
    thread_turns: list[dict] | None = None,
):
    """Streaming version of answer_report_chat. Yields (event_type, data) tuples."""
    report_text = str(report_answer or "").strip()
    if not report_text:
        yield ("error", "No report or summary content available.")
        return

    history_lines: list[str] = []
    for turn in (history or [])[-8:]:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines) or "(none)"

    thread_lines: list[str] = []
    for index, turn in enumerate((thread_turns or [])[-6:], start=max(1, len(thread_turns or []) - 5)):
        approved_questions = [str(item).strip() for item in (turn.get("approved_questions") or []) if str(item).strip()]
        if approved_questions:
            thread_lines.append(f"Turn {index} questions: {'; '.join(approved_questions)}")
        excerpt = str(turn.get("answer") or "").strip()
        if excerpt:
            thread_lines.append(f"Turn {index} excerpt:\n{excerpt[:1500]}")
    thread_block = "\n\n".join(thread_lines) or "(none)"

    source_urls = "\n".join(f"- {src.url}" for src in report_sources[:20]) or "(none)"
    truncated_report = report_text[:12000]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded assistant answering questions about an existing report. "
                "Use ONLY the provided report text, prior thread context, and source URLs. "
                "Do not claim to have checked the live web. If the user asks for current/latest/fresh information "
                "or for facts not supported by the report, say that a fresh Deep Research run is needed. "
                "Answer directly, stay concise, and include a short 'Sources' section only when useful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source title: {source_context.get('title', 'Unknown')}\n"
                f"Source URL: {source_context.get('url', 'N/A')}\n"
                f"Source type: {source_context.get('type', 'unknown')}\n\n"
                f"Research thread context:\n{thread_block}\n\n"
                f"Current report:\n{truncated_report}\n\n"
                f"Known source URLs:\n{source_urls}\n\n"
                f"Recent chat:\n{history_block}\n\n"
                f"User question: {user_question}"
            ),
        },
    ]

    try:
        yield from stream_inception_chat(
            messages=messages,
            max_tokens=1400,
            reasoning_effort="low",
            temperature=0.1,
            timeout=SYNTH_TIMEOUT_SECONDS,
            diffusing=True,
            reasoning_summary=True,
        )
    except Exception as e:
        yield ("error", str(e))
