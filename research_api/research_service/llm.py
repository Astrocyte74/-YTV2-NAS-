"""LLM client helpers for planning and synthesis with OpenRouter fallback."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

from .config import (
    INCEPTION_API_KEY,
    INCEPTION_MODEL,
    INCEPTION_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_TITLE,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_URL,
    RESEARCH_FALLBACK_ENABLED,
    RESEARCH_FALLBACK_MODEL,
)

logger = logging.getLogger(__name__)
ProviderMode = str


@dataclass
class LLMResponse:
    """Response from LLM with provider metadata."""
    data: dict[str, Any]
    provider: str  # "inception" or "openrouter"
    model: str


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        logger.debug("Direct JSON parse failed: %s", e)

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.debug("Fenced JSON parse failed: %s", e)

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        try:
            parsed = json.loads(text[first : last + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.debug("Extracted JSON parse failed (first=%d, last=%d): %s", first, last, e)
            return None

    return None


def _post_inception(payload: dict[str, Any], timeout: float, model_override: str | None = None) -> LLMResponse:
    """Call Inception Labs Mercury API."""
    if not INCEPTION_API_KEY:
        raise RuntimeError("INCEPTION_API_KEY is not configured")

    model = (model_override or payload.get("model") or INCEPTION_MODEL)
    request_payload = dict(payload)
    request_payload["model"] = model
    headers = {
        "Authorization": f"Bearer {INCEPTION_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(INCEPTION_URL, headers=headers, json=request_payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected LLM response")
    logger.info("LLM request completed via Inception (model=%s)", model)
    return LLMResponse(data=data, provider="inception", model=str(model))


def _post_openrouter(payload: dict[str, Any], timeout: float, model_override: str | None = None) -> LLMResponse:
    """Call OpenRouter API as fallback."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    # OpenRouter doesn't support reasoning_effort - remove it
    clean_payload = {k: v for k, v in payload.items() if k != "reasoning_effort"}
    model = (model_override or RESEARCH_FALLBACK_MODEL)
    clean_payload["model"] = model

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_APP_TITLE,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=clean_payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected LLM response from OpenRouter")
    logger.info("LLM request completed via OpenRouter fallback (model=%s)", model)
    return LLMResponse(data=data, provider="openrouter", model=str(model))


def _post_chat(
    payload: dict[str, Any],
    timeout: float,
    provider: ProviderMode = "auto",
    model_override: str | None = None,
) -> LLMResponse:
    """Call LLM with fallback support.

    Tries Inception Labs first, falls back to OpenRouter if:
    - Inception API key is missing
    - Inception request fails (rate limit, outage, etc.)
    - Fallback is enabled and configured

    Returns LLMResponse with provider metadata.
    """
    if provider not in {"auto", "inception", "openrouter"}:
        provider = "auto"

    if provider == "inception":
        return _post_inception(payload, timeout, model_override=model_override)
    if provider == "openrouter":
        return _post_openrouter(payload, timeout, model_override=model_override)

    errors: list[str] = []

    # Try Inception first
    if INCEPTION_API_KEY:
        try:
            return _post_inception(payload, timeout, model_override=model_override)
        except Exception as e:
            errors.append(f"Inception: {e}")
            logger.warning("Inception API call failed: %s", e)

    # Fallback to OpenRouter
    if RESEARCH_FALLBACK_ENABLED and OPENROUTER_API_KEY:
        try:
            logger.info(
                "Falling back to OpenRouter with model: %s",
                model_override or RESEARCH_FALLBACK_MODEL,
            )
            return _post_openrouter(payload, timeout, model_override=model_override)
        except Exception as e:
            errors.append(f"OpenRouter: {e}")
            logger.error("OpenRouter fallback also failed: %s", e)

    # Both failed or not configured
    if not INCEPTION_API_KEY and not OPENROUTER_API_KEY:
        raise RuntimeError("No LLM API key configured (INCEPTION_API_KEY or OPENROUTER_API_KEY)")

    raise RuntimeError(f"All LLM providers failed: {'; '.join(errors)}")


def chat_json_schema(
    *,
    messages: list[dict[str, str]],
    schema_name: str,
    schema: dict[str, Any],
    max_tokens: int,
    reasoning_effort: str,
    temperature: float,
    timeout: float,
    provider: ProviderMode = "auto",
    model_override: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Call the configured LLM expecting structured JSON.

    Uses json_schema response_format and includes robust parse fallback in case
    provider returns wrapped content.

    Returns:
        Tuple of (parsed_json, provider, model)
    """
    payload: dict[str, Any] = {
        "model": INCEPTION_MODEL,
        "messages": messages,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    }

    llm_resp = _post_chat(payload, timeout, provider=provider, model_override=model_override)
    choices = llm_resp.data.get("choices") or []
    if not choices:
        raise RuntimeError("No choices in LLM response")

    content = (choices[0].get("message", {}) or {}).get("content") or ""
    parsed = _extract_json_object(content)
    if parsed is None:
        logger.warning("JSON parse failed. Full content (%d chars): %s", len(content), content[:2000] if content else "(empty)")
        raise RuntimeError("Unable to parse structured JSON response")
    return parsed, llm_resp.provider, llm_resp.model


def chat_tool_call(
    *,
    messages: list[dict[str, str]],
    function_name: str,
    schema: dict[str, Any],
    max_tokens: int,
    reasoning_effort: str,
    temperature: float,
    timeout: float,
    provider: ProviderMode = "auto",
) -> tuple[dict[str, Any], str, str]:
    """Call the configured LLM tool-use path and parse JSON arguments.

    Returns:
        Tuple of (parsed_json, provider, model)
    """
    payload: dict[str, Any] = {
        "model": INCEPTION_MODEL,
        "messages": messages,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": "Select routing decision.",
                    "parameters": schema,
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": function_name},
        },
    }

    llm_resp = _post_chat(payload, timeout, provider=provider)
    choices = llm_resp.data.get("choices") or []
    if not choices:
        raise RuntimeError("No choices in LLM response")

    message = (choices[0].get("message", {}) or {})
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        function = (tool_calls[0].get("function", {}) or {})
        arguments = function.get("arguments") or ""
        parsed = _extract_json_object(str(arguments))
        if parsed is not None:
            return parsed, llm_resp.provider, llm_resp.model

    # Fallback: some providers return content instead of tool_calls
    content = message.get("content") or ""
    parsed = _extract_json_object(str(content))
    if parsed is not None:
        return parsed, llm_resp.provider, llm_resp.model

    raise RuntimeError("Unable to parse tool-call arguments")


def chat_text(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    reasoning_effort: str,
    temperature: float,
    timeout: float,
    provider: ProviderMode = "auto",
) -> tuple[str, str, str]:
    """Call LLM for text response.

    Returns:
        Tuple of (text_content, provider, model)
    """
    payload = {
        "model": INCEPTION_MODEL,
        "messages": messages,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    llm_resp = _post_chat(payload, timeout, provider=provider)
    choices = llm_resp.data.get("choices") or []
    if not choices:
        return "", llm_resp.provider, llm_resp.model
    content = str((choices[0].get("message", {}) or {}).get("content") or "").strip()
    return content, llm_resp.provider, llm_resp.model


def chat_text_with_metadata(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    reasoning_effort: str,
    temperature: float,
    timeout: float,
    provider: ProviderMode = "auto",
) -> dict[str, str | None]:
    payload = {
        "model": INCEPTION_MODEL,
        "messages": messages,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    llm_resp = _post_chat(payload, timeout, provider=provider)
    choices = llm_resp.data.get("choices") or []
    if not choices:
        return {"content": "", "finish_reason": None, "llm_provider": llm_resp.provider, "llm_model": llm_resp.model}
    choice = choices[0] or {}
    return {
        "content": str((choice.get("message", {}) or {}).get("content") or "").strip(),
        "finish_reason": str(choice.get("finish_reason") or "").strip() or None,
        "llm_provider": llm_resp.provider,
        "llm_model": llm_resp.model,
    }
