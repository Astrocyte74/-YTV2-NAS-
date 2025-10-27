"""Client helpers to talk to a local Ollama endpoint via the hub proxy or
directly, with a unified API.

Priority:
- If ``TTSHUB_API_BASE`` is set, use the hub (WireGuard‑reachable) and call
  ``/ollama/*`` endpoints under that base.
- Else if ``OLLAMA_URL`` or ``OLLAMA_HOST`` is set, call the direct Ollama
  server using ``/api/*`` endpoints.

Streaming helpers yield raw SSE data lines (JSON strings without the
"data: " prefix).
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import ast

import requests


class OllamaClientError(RuntimeError):
    """Raised when local (hub/direct) Ollama access fails or is unavailable."""
    pass


class LocalUnavailable(OllamaClientError):
    """Connectivity/timeouts/5xx from local (hub/direct) paths."""
    pass


class NotFound(OllamaClientError):
    """Requested resource/model not found (404)."""
    pass


class InvalidRequest(OllamaClientError):
    """4xx client errors such as bad prompt/context too large, etc."""
    pass


def _resolve() -> Tuple[str, str]:
    """Resolve which base to use and return a tuple (mode, base).

    mode is one of:
    - 'hub'     → use hub proxy paths under '/ollama/*'
    - 'direct'  → use direct Ollama paths under '/api/*'
    """
    hub = (os.environ.get("TTSHUB_API_BASE") or "").strip()
    if hub:
        base = hub.rstrip("/")
        logging.info("ollama_client: provider=hub base=%s", base)
        return ("hub", base)
    direct = (os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST") or "").strip()
    if direct:
        base = direct.rstrip("/")
        logging.info("ollama_client: provider=direct base=%s", base)
        return ("direct", base)
    raise OllamaClientError("No local Ollama base configured (set TTSHUB_API_BASE or OLLAMA_URL/OLLAMA_HOST)")


def _translate(path: str) -> str:
    """Translate logical 'ollama/<endpoint>' path to the correct URL.

    For hub mode, paths are passed through as-is under '<hub>/ollama/...'.
    For direct mode, 'ollama/<ep>' is mapped to '<direct>/api/<ep>'.
    """
    mode, base = _resolve()
    p = path.lstrip("/")
    if mode == "hub":
        # Expect paths like 'ollama/chat', 'ollama/tags', ...
        return f"{base}/{p}"
    # direct → map 'ollama/<ep>' to '/api/<ep>'
    if p.startswith("ollama/"):
        p = p.split("/", 1)[1]
    return f"{base}/api/{p}"


def _get(path: str, *, timeout: float = 15.0) -> Dict:
    url = _translate(path)
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json() or {}
    except requests.exceptions.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 404:
            raise NotFound(f"GET {url} 404 Not Found")
        if status and 400 <= status < 500:
            raise InvalidRequest(f"GET {url} {status}")
        raise LocalUnavailable(f"GET {url} {status or ''} {exc}")
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise LocalUnavailable(f"GET {url} failed: {exc}")
    except Exception as exc:
        raise OllamaClientError(f"GET {url} failed: {exc}")


def _post(path: str, body: Dict, *, timeout: Optional[float] = 120.0) -> Dict:
    url = _translate(path)
    try:
        r = requests.post(url, json=body or {}, timeout=timeout)
        r.raise_for_status()
        return r.json() or {}
    except requests.exceptions.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 404:
            raise NotFound(f"POST {url} 404 Not Found")
        if status and 400 <= status < 500:
            raise InvalidRequest(f"POST {url} {status}")
        raise LocalUnavailable(f"POST {url} {status or ''} {exc}")
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise LocalUnavailable(f"POST {url} failed: {exc}")
    except Exception as exc:
        raise OllamaClientError(f"POST {url} failed: {exc}")


def get_models() -> Dict:
    return _get("ollama/tags")


def ps() -> Dict:
    return _get("ollama/ps")


def show(model: str) -> Dict:
    return _post("ollama/show", {"model": model})


def delete(model: str) -> Dict:
    return _post("ollama/delete", {"model": model}, timeout=30)


def pull(model: str, *, stream: bool = True) -> Dict | Iterable[str]:
    if not stream:
        return _post("ollama/pull", {"model": model, "stream": False}, timeout=None)
    return pull_stream(model)


def _sse_events(resp: requests.Response) -> Iterable[bytes]:
    acc: List[bytes] = []
    for raw in resp.iter_lines(decode_unicode=False):
        if raw is None:
            continue
        if raw == b"":
            if acc:
                payload = b"".join(acc)
                acc.clear()
                yield payload
            continue
        if raw.startswith(b"data: "):
            acc.append(raw[6:])


def pull_stream(model: str) -> Iterable[Dict]:
    url = _translate("ollama/pull")
    body = {"model": model, "stream": True}
    with requests.post(url, json=body, stream=True, timeout=None) as r:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                raise NotFound(f"POST {url} 404 Not Found")
            if status and 400 <= status < 500:
                raise InvalidRequest(f"POST {url} {status}")
            raise LocalUnavailable(f"POST {url} {status or ''} {exc}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            raise LocalUnavailable(f"POST {url} failed: {exc}")
        for payload in _sse_events(r):
            txt: str
            if payload.startswith(b"b'") and payload.endswith(b"'"):
                try:
                    by = ast.literal_eval(payload.decode("utf-8", errors="ignore"))  # type: ignore
                    txt = by.decode("utf-8", errors="ignore")
                except Exception:
                    continue
            else:
                txt = payload.decode("utf-8", errors="ignore")
            try:
                yield requests.models.complexjson.loads(txt)
            except Exception:
                continue


def generate(prompt: str, model: str, *, stream: bool = False) -> Dict | Iterable[str]:
    if not stream:
        return _post("ollama/generate", {"model": model, "prompt": prompt, "stream": False})
    return generate_stream(prompt, model)


def generate_stream(prompt: str, model: str) -> Iterable[Dict]:
    url = _translate("ollama/generate")
    body = {"model": model, "prompt": prompt, "stream": True}
    with requests.post(url, json=body, stream=True, timeout=None) as r:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                raise NotFound(f"POST {url} 404 Not Found")
            if status and 400 <= status < 500:
                raise InvalidRequest(f"POST {url} {status}")
            raise LocalUnavailable(f"POST {url} {status or ''} {exc}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            raise LocalUnavailable(f"POST {url} failed: {exc}")
        for payload in _sse_events(r):
            txt: str
            if payload.startswith(b"b'") and payload.endswith(b"'"):
                try:
                    by = ast.literal_eval(payload.decode("utf-8", errors="ignore"))  # type: ignore
                    txt = by.decode("utf-8", errors="ignore")
                except Exception:
                    continue
            else:
                txt = payload.decode("utf-8", errors="ignore")
            try:
                yield requests.models.complexjson.loads(txt)
            except Exception:
                continue


def chat(messages: List[Dict[str, str]], model: str, *, stream: bool = False) -> Dict | Iterable[str]:
    if not stream:
        return _post("ollama/chat", {"model": model, "messages": messages, "stream": False})
    return chat_stream(messages, model)


def chat_stream(messages: List[Dict[str, str]], model: str) -> Iterable[Dict]:
    url = _translate("ollama/chat")
    body = {"model": model, "messages": messages, "stream": True}
    with requests.post(url, json=body, stream=True, timeout=None) as r:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                raise NotFound(f"POST {url} 404 Not Found")
            if status and 400 <= status < 500:
                raise InvalidRequest(f"POST {url} {status}")
            raise LocalUnavailable(f"POST {url} {status or ''} {exc}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            raise LocalUnavailable(f"POST {url} failed: {exc}")
        for payload in _sse_events(r):
            txt: str
            if payload.startswith(b"b'") and payload.endswith(b"'"):
                try:
                    by = ast.literal_eval(payload.decode("utf-8", errors="ignore"))  # type: ignore
                    txt = by.decode("utf-8", errors="ignore")
                except Exception:
                    continue
            else:
                txt = payload.decode("utf-8", errors="ignore")
            try:
                yield requests.models.complexjson.loads(txt)
            except Exception:
                continue


__all__ = [
    "OllamaClientError",
    "get_models",
    "ps",
    "show",
    "delete",
    "pull",
    "pull_stream",
    "generate",
    "generate_stream",
    "chat",
    "chat_stream",
]


# ------------------------ Health / Self-test helpers ------------------------
_HEALTH_CACHE: Dict[str, Any] = {"ts": 0.0, "ok": None, "summary": None}


def health_summary(cache_ttl: int = 30) -> Dict[str, Any]:
    """Quick classification of local availability.

    Returns a dict: { provider: 'hub'|'direct', base: str, reachable: bool,
    models: int|None, notes: str }
    """
    now = time.time()
    if _HEALTH_CACHE["ts"] and (now - _HEALTH_CACHE["ts"]) < max(0, cache_ttl):
        return dict(_HEALTH_CACHE.get("summary") or {})
    try:
        mode, base = _resolve()
    except Exception as exc:
        summ = {"provider": None, "base": None, "reachable": False, "models": None, "notes": str(exc)}
        _HEALTH_CACHE.update({"ts": now, "ok": False, "summary": summ})
        return summ
    try:
        tags = get_models() or {}
        models = tags.get("models") or []
        model_count = len(models) if isinstance(models, list) else None
        # Try a tiny non-stream chat if any model is known
        note = ""
        if model_count and isinstance(models, list):
            m = None
            # common shapes: {'name': 'llama3.2:3b'} or {'model': '...'}
            for entry in models:
                if isinstance(entry, dict):
                    m = entry.get("name") or entry.get("model")
                if m:
                    break
            if m:
                try:
                    resp = chat([{"role": "user", "content": "Hi"}], model=m, stream=False)
                    _ = bool(resp)
                except NotFound:
                    note = f"model {m} not found"
                except InvalidRequest:
                    note = "invalid request"
        summ = {"provider": mode, "base": base, "reachable": True, "models": model_count, "notes": note}
        _HEALTH_CACHE.update({"ts": now, "ok": True, "summary": summ})
        return summ
    except (LocalUnavailable, OllamaClientError) as exc:
        summ = {"provider": mode, "base": base, "reachable": False, "models": None, "notes": str(exc)}
        _HEALTH_CACHE.update({"ts": now, "ok": False, "summary": summ})
        return summ


def is_available(cache_ttl: int = 30) -> bool:
    """Return True if local (hub/direct) appears reachable (cached)."""
    return bool((health_summary(cache_ttl) or {}).get("reachable"))
