from __future__ import annotations

import logging
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from llm_config import llm_config


def _build_llm(provider: str, model: str):
    """Return a LangChain chat model for the requested cloud provider/model."""
    prov = (provider or "").strip().lower()
    if prov == "openai":
        return ChatOpenAI(model=model, api_key=llm_config.openai_key)
    if prov == "anthropic":
        return ChatAnthropic(model=model, api_key=llm_config.anthropic_key)
    if prov == "openrouter":
        # OpenRouter via OpenAI-compatible endpoint
        return ChatOpenAI(
            model=model,
            api_key=llm_config.openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/astrocyte74/stuff",
                "X-Title": "YouTube Summarizer",
            },
        )
    raise ValueError(f"Unsupported cloud provider: {provider}")


def _to_langchain_messages(messages: List[Dict[str, str]]):
    """Convert simple {role, content} dicts to LangChain messages."""
    out = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def chat(messages: List[Dict[str, str]], *, provider: Optional[str], model: Optional[str]) -> str:
    """Run a single non-streaming cloud chat completion and return text."""
    # Resolve provider/model using existing llm_config logic
    resolved_provider, resolved_model, _api_key = llm_config.get_model_config(provider, model)
    if resolved_provider == "ollama":
        raise ValueError("Cloud chat requested but resolved to local (ollama)")
    llm = _build_llm(resolved_provider, resolved_model)
    lcm = _to_langchain_messages(messages)
    try:
        resp = llm.invoke(lcm)
    except Exception as exc:
        logging.exception("Cloud chat error: %s", exc)
        raise
    text = getattr(resp, "content", None) or ""
    if isinstance(text, list):
        # LangChain sometimes returns a structured content list; join text parts
        parts = []
        for seg in text:
            if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                parts.append(seg["text"]) 
        text = "\n".join(parts)
    return text or ""


__all__ = ["chat"]

