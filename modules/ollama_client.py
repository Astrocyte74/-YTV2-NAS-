"""Client helpers to talk to the hub's Ollama proxy.

Uses TTSHUB_API_BASE and exposes simple sync methods for tags, chat, generate,
pull, ps, show, and delete. Streaming helpers yield raw SSE data lines (JSON
strings without the "data: " prefix).
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Iterator, List, Optional, Union
import ast

import requests


class OllamaClientError(RuntimeError):
    pass


def _base() -> str:
    base = os.environ.get("TTSHUB_API_BASE")
    if not base:
        raise OllamaClientError("TTSHUB_API_BASE is not set")
    return base.rstrip("/")


def _get(path: str, *, timeout: float = 15.0) -> Dict:
    url = f"{_base()}/{path.lstrip('/')}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json() or {}


def _post(path: str, body: Dict, *, timeout: Optional[float] = 120.0) -> Dict:
    url = f"{_base()}/{path.lstrip('/')}"
    r = requests.post(url, json=body or {}, timeout=timeout)
    r.raise_for_status()
    return r.json() or {}


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
    url = f"{_base()}/ollama/pull"
    body = {"model": model, "stream": True}
    with requests.post(url, json=body, stream=True, timeout=None) as r:
        r.raise_for_status()
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
    url = f"{_base()}/ollama/generate"
    body = {"model": model, "prompt": prompt, "stream": True}
    with requests.post(url, json=body, stream=True, timeout=None) as r:
        r.raise_for_status()
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
    url = f"{_base()}/ollama/chat"
    body = {"model": model, "messages": messages, "stream": True}
    with requests.post(url, json=body, stream=True, timeout=None) as r:
        r.raise_for_status()
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
