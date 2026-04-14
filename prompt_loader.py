"""
Shared prompt loader for YTV2.

Loads prompts from ``backend/config/all_prompts.json`` (symlinked from
``prompt_test_samples/all_prompts.json``).  Every runtime caller should go
through this module so that prompt text, context assembly, and LLM defaults
have a single source of truth.

Usage
-----
    from prompt_loader import render_prompt, get_llm_config

    prompt_text = render_prompt(
        "primary_summaries.comprehensive",
        variables={
            "title": "...",
            "uploader": "...",
            "upload_date": "...",
            "duration": "0",
            "url": "...",
            "transcript": "...",
            "lang_instruction": "...",
        },
    )

Config precedence (documented for callers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **explicit code override**  – caller passes value directly
2. **env-driven override**     – os.getenv("SOME_VAR")
3. **prompt JSON default**     – stored alongside the prompt in all_prompts.json
4. **loader fallback default** – hardcoded in get_llm_config()
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_PROMPTS_CACHE: Optional[Dict[str, Any]] = None
_PROMPTS_PATH: Optional[Path] = None

# Regex to detect unresolved placeholders like {foo} or {bar_baz}.
_UNRESOLVED_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _default_prompts_path() -> Path:
    """Return the canonical path to all_prompts.json."""
    return Path(__file__).resolve().parent / "config" / "all_prompts.json"


def _load_prompts(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and cache the prompts JSON."""
    global _PROMPTS_CACHE, _PROMPTS_PATH
    target = Path(path) if path else _default_prompts_path()
    if _PROMPTS_CACHE is not None and _PROMPTS_PATH == target:
        return _PROMPTS_CACHE
    if not target.exists():
        raise FileNotFoundError(f"Prompt registry not found: {target}")
    with open(target, "r", encoding="utf-8") as fh:
        _PROMPTS_CACHE = json.load(fh)
    _PROMPTS_PATH = target
    return _PROMPTS_CACHE


def reload_prompts(path: Optional[Path] = None) -> None:
    """Force reload on next access (useful after editing JSON)."""
    global _PROMPTS_CACHE
    _PROMPTS_CACHE = None
    if path:
        global _PROMPTS_PATH
        _PROMPTS_PATH = Path(path)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_entry(dotpath: str, prompts: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a dot-path like ``primary_summaries.comprehensive`` to its dict.

    Also supports single-part paths (e.g. ``headline``) when the top-level
    key maps directly to a dict containing a ``prompt`` key.

    Raises ``KeyError`` with a helpful message for unknown paths.
    """
    parts = dotpath.split(".")
    if len(parts) == 1:
        # Single-part path: check if it's a top-level prompt entry
        top = prompts.get(dotpath)
        if isinstance(top, dict) and "prompt" in top:
            return top
        raise KeyError(
            f"Prompt path must be 'section.name', got '{dotpath}'. "
            f"Example: 'primary_summaries.comprehensive'"
        )

    section_name = parts[0]
    entry_name = parts[1]

    section = prompts.get(section_name)
    if section is None:
        available = [k for k in prompts if not k.startswith("_") and isinstance(prompts[k], dict)]
        raise KeyError(
            f"Unknown prompt section '{section_name}'. Available: {available}"
        )

    if not isinstance(section, dict):
        raise KeyError(
            f"Section '{section_name}' is not a dict — cannot look up '{entry_name}'"
        )

    entry = section.get(entry_name)
    if entry is None:
        available = [k for k in section if isinstance(section.get(k), dict) and "prompt" in section[k]]
        raise KeyError(
            f"Unknown prompt '{dotpath}'. Available in '{section_name}': {available}"
        )

    if not isinstance(entry, dict):
        raise KeyError(f"Entry '{dotpath}' is not a dict — expected a prompt object")

    if "prompt" not in entry:
        raise KeyError(f"Entry '{dotpath}' has no 'prompt' key")

    return entry


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def render_template(template: str, variables: Dict[str, str]) -> str:
    """Safely replace ``{key}`` placeholders in *template*.

    Uses simple string replacement (not ``str.format``) so that curly braces
    in transcript/source text do not cause errors.

    Raises ``ValueError`` if unresolved ``{placeholder}`` patterns remain
    after substitution, unless ``allow_unresolved=True``.
    """
    result = template
    for key, value in variables.items():
        result = result.replace("{" + key + "}", str(value))

    return result


def _check_unresolved(rendered: str, source_label: str) -> None:
    """Raise ValueError if unresolved placeholders remain."""
    matches = _UNRESOLVED_RE.findall(rendered)
    if matches:
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        raise ValueError(
            f"Unresolved placeholders in {source_label}: {unique}. "
            f"Either supply values or mark them as optional."
        )


# ---------------------------------------------------------------------------
# Context resolution
# ---------------------------------------------------------------------------


def _resolve_context(context_name: str, prompts: Dict[str, Any]) -> str:
    """Look up a shared context template by name.

    Returns the raw template string (still with placeholders).
    Raises ``KeyError`` for unknown context names.
    """
    shared = prompts.get("shared_context", {})
    if context_name == "none" or not context_name:
        return ""
    ctx_entry = shared.get(context_name)
    if ctx_entry is None:
        available = list(shared.keys())
        raise KeyError(
            f"Unknown context block '{context_name}'. Available: {available}"
        )
    if isinstance(ctx_entry, dict) and "template" in ctx_entry:
        return ctx_entry["template"]
    raise KeyError(
        f"Context block '{context_name}' has no 'template' key"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_prompt_entry(dotpath: str) -> dict:
    """Return the raw prompt dict for *dotpath* (e.g. ``primary_summaries.comprehensive``).

    Includes keys like ``prompt``, ``context``, ``model``, etc.
    """
    prompts = _load_prompts()
    return _resolve_entry(dotpath, prompts)


def render_prompt(
    dotpath: str,
    variables: Dict[str, str],
    *,
    allow_unresolved: bool = False,
) -> str:
    """Render a full prompt (context block + prompt template) from JSON.

    Parameters
    ----------
    dotpath :
        Dotted path like ``primary_summaries.comprehensive``.
    variables :
        Values for ``{key}`` placeholders.  Must include variables required
        by both the context block and the prompt template.
    allow_unresolved :
        If True, silently allow unresolved placeholders (useful for chunked
        prompts where some variables are only known at call time).

    Returns
    -------
    str
        The fully rendered prompt text, ready to send to an LLM.

    Raises
    ------
    KeyError
        On unknown prompt path or context name.
    ValueError
        On unresolved placeholders (when ``allow_unresolved=False``).
    """
    prompts = _load_prompts()
    entry = _resolve_entry(dotpath, prompts)

    # 1. Render context block
    context_name = entry.get("context", "none")
    context_template = _resolve_context(context_name, prompts)
    context_rendered = render_template(context_template, variables)

    # 2. Render the prompt template itself
    prompt_template = entry["prompt"]
    prompt_rendered = render_template(prompt_template, variables)

    # 3. Combine context + prompt
    if context_rendered:
        combined = context_rendered + "\n\n" + prompt_rendered
    else:
        combined = prompt_rendered

    # 4. Check for unresolved placeholders
    if not allow_unresolved:
        _check_unresolved(combined, dotpath)

    return combined


def render_prompt_only(
    dotpath: str,
    variables: Dict[str, str],
    *,
    allow_unresolved: bool = False,
) -> str:
    """Render just the prompt template (no context block) from JSON.

    Useful for prompts whose context is assembled separately (e.g.
    chunked extractors that operate on an already-summarized text).

    Parameters
    ----------
    dotpath :
        Dotted path like ``chunked_extractors.extract_bullet_points``.
    variables :
        Values for ``{key}`` placeholders.
    allow_unresolved :
        If True, silently allow unresolved placeholders.

    Returns
    -------
    str
        The rendered prompt text (no context prefix).

    Raises
    ------
    KeyError
        On unknown prompt path.
    ValueError
        On unresolved placeholders (when ``allow_unresolved=False``).
    """
    prompts = _load_prompts()
    entry = _resolve_entry(dotpath, prompts)

    prompt_template = entry["prompt"]
    rendered = render_template(prompt_template, variables)

    if not allow_unresolved:
        _check_unresolved(rendered, dotpath)

    return rendered


def get_llm_config(dotpath: str) -> Dict[str, Any]:
    """Return LLM defaults for a prompt entry.

    Merges prompt-local config from JSON with sensible fallback defaults.
    Callers should still be able to override via env or code.

    Config precedence (highest wins):
        1. explicit code override
        2. env-driven override
        3. prompt JSON default (this function returns this level)
        4. loader fallback default

    Returns
    -------
    dict
        Keys: ``model``, ``reasoning_effort``, ``max_tokens``, ``temperature``.
    """
    FALLBACK = {
        "model": None,
        "reasoning_effort": None,
        "max_tokens": None,
        "temperature": None,
    }
    try:
        entry = get_prompt_entry(dotpath)
    except KeyError:
        return dict(FALLBACK)

    result = dict(FALLBACK)
    for key in FALLBACK:
        if key in entry and entry[key] is not None:
            result[key] = entry[key]

    return result


# ---------------------------------------------------------------------------
# Convenience: list available prompts
# ---------------------------------------------------------------------------


def list_prompts() -> Dict[str, list]:
    """Return a mapping of section → list of prompt names."""
    prompts = _load_prompts()
    result: Dict[str, list] = {}
    for section_name, section_data in prompts.items():
        if section_name.startswith("_"):
            continue
        if not isinstance(section_data, dict):
            continue
        names = [
            name for name, entry in section_data.items()
            if isinstance(entry, dict) and "prompt" in entry
        ]
        if names:
            result[section_name] = names
    return result
