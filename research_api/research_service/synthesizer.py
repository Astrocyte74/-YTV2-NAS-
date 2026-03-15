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
from .llm import chat_text_with_metadata
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
