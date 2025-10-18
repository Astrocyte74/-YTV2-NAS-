"""Client helpers for interacting with the external TTS hub service.

This module centralises HTTP access to the TTS hub plus small utilities
for working with voice metadata (accent families, filtering, etc.).
Both the Telegram bot and other NAS components can import these helpers
instead of duplicating request logic.
"""

from __future__ import annotations

import asyncio
import os
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional, Set

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

DEFAULT_ENGINE = "kokoro"


class LocalTTSUnavailable(RuntimeError):
    """Raised when the local TTS hub is unavailable or errors."""



def _strip_api_suffix(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/api"):
        return url[:-4]
    return url


class TTSHubClient:
    """Minimal asynchronous wrapper around the TTS hub REST API."""

    def __init__(self, base_url: Optional[str]):
        self.base_api_url = base_url.rstrip("/") if base_url else None
        self._audio_base_url = (
            _strip_api_suffix(self.base_api_url) if self.base_api_url else None
        )

    @classmethod
    def from_env(cls) -> "TTSHubClient":
        return cls(os.getenv("TTSHUB_API_BASE"))

    @property
    def audio_base_url(self) -> Optional[str]:
        return self._audio_base_url

    def _ensure_requests(self) -> None:
        if requests is None:
            raise RuntimeError("requests library is required for TTS hub access.")

    def _ensure_base(self) -> None:
        if not self.base_api_url:
            raise RuntimeError("TTS hub base URL is not configured.")

    async def fetch_catalog(
        self, engine: str = DEFAULT_ENGINE, timeout: float = 12.0
    ) -> Optional[Dict[str, Any]]:
        self._ensure_requests()
        self._ensure_base()
        loop = asyncio.get_running_loop()

        def _call() -> Optional[Dict[str, Any]]:
            resp = requests.get(  # type: ignore
                f"{self.base_api_url}/voices_catalog",
                params={"engine": engine} if engine else None,
                timeout=timeout,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()

        return await loop.run_in_executor(None, _call)

    async def fetch_favorites(
        self, tag: Optional[str] = None, timeout: float = 10.0
    ) -> List[Dict[str, Any]]:
        self._ensure_requests()
        self._ensure_base()
        loop = asyncio.get_running_loop()

        def _call() -> List[Dict[str, Any]]:
            params = {"tag": tag} if tag else None
            resp = requests.get(  # type: ignore
                f"{self.base_api_url}/favorites", params=params, timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json() or {}
            profiles = data.get("profiles") or []
            return [p for p in profiles if isinstance(p, dict)]

        return await loop.run_in_executor(None, _call)

    async def synthesise(
        self,
        text: str,
        *,
        favorite_slug: Optional[str] = None,
        voice_id: Optional[str] = None,
        engine: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        self._ensure_requests()
        self._ensure_base()
        audio_base = self.audio_base_url
        if not audio_base:
            raise RuntimeError("TTS hub audio base URL could not be determined.")

        payload: Dict[str, Any]
        if favorite_slug:
            payload = {"favoriteSlug": favorite_slug, "text": text}
        elif voice_id:
            payload = {"voice": voice_id, "text": text}
            if engine:
                payload["engine"] = engine
        else:
            raise ValueError("Either favorite_slug or voice_id must be provided.")

        loop = asyncio.get_running_loop()

        def _call() -> Dict[str, Any]:
            try:
                resp = requests.post(  # type: ignore
                    f"{self.base_api_url}/synthesise", json=payload, timeout=timeout
                )
                resp.raise_for_status()
                data = resp.json() or {}
                audio_path = data.get("path")
                if not audio_path:
                    raise ValueError("TTS hub response missing audio path.")
                audio_url = urllib.parse.urljoin(audio_base.rstrip("/") + "/", audio_path.lstrip("/"))
                audio_resp = requests.get(audio_url, timeout=60)  # type: ignore
                audio_resp.raise_for_status()
                data["audio_bytes"] = audio_resp.content
                data["audio_url"] = audio_url
                return data
            except requests.exceptions.RequestException as exc:  # type: ignore
                raise LocalTTSUnavailable(str(exc))

        try:
            return await loop.run_in_executor(None, _call)
        except (ValueError, LocalTTSUnavailable) as exc:
            raise LocalTTSUnavailable(str(exc))


def normalize_accent_family(accent_id: Optional[str]) -> str:
    accent_id = (accent_id or "").lower()
    if accent_id.startswith("us"):
        return "us"
    if accent_id.startswith("uk"):
        return "uk"
    return "other"


def filter_catalog_voices(
    catalog: Dict[str, Any],
    *,
    gender: Optional[str] = None,
    family: Optional[str] = None,
    allowed_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    voices = (catalog or {}).get("voices") or []
    allowed_set: Optional[Set[str]] = set(allowed_ids) if allowed_ids is not None else None
    result: List[Dict[str, Any]] = []
    for voice in voices:
        voice_id = voice.get("id")
        if not voice_id:
            continue
        if allowed_set is not None and voice_id not in allowed_set:
            continue
        if gender and voice.get("gender") != gender:
            continue
        family_id = normalize_accent_family((voice.get("accent") or {}).get("id"))
        if family and family_id != family:
            continue
        result.append(voice)
    result.sort(key=lambda v: (v.get("label") or v.get("id") or "").lower())
    return result


def available_accent_families(
    catalog: Dict[str, Any],
    *,
    gender: Optional[str] = None,
    allowed_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    voices = filter_catalog_voices(catalog, gender=gender, allowed_ids=allowed_ids)
    counts: Dict[str, int] = {}
    for voice in voices:
        accent_id = (voice.get("accent") or {}).get("id")
        family_id = normalize_accent_family(accent_id)
        counts[family_id] = counts.get(family_id, 0) + 1

    families_meta = ((catalog or {}).get("filters") or {}).get("accentFamilies", {}).get("any", [])
    meta_lookup = {entry.get("id"): entry for entry in families_meta if entry.get("id")}

    results: List[Dict[str, Any]] = []
    for family_id, count in counts.items():
        meta = meta_lookup.get(family_id, {"label": family_id.title(), "flag": ""})
        results.append(
            {
                "id": family_id,
                "label": meta.get("label") or family_id.title(),
                "flag": meta.get("flag") or "",
                "count": count,
            }
        )
    results.sort(key=lambda item: (-item["count"], item["label"].lower()))
    return results


def accent_family_label(catalog: Dict[str, Any], family_id: Optional[str]) -> str:
    if not family_id:
        return "All accents"
    families_meta = ((catalog or {}).get("filters") or {}).get("accentFamilies", {}).get("any", [])
    for entry in families_meta:
        if entry.get("id") == family_id:
            label = entry.get("label") or family_id.title()
            flag = entry.get("flag") or ""
            return f"{flag} {label}".strip()
    return family_id.title()


__all__ = [
    "DEFAULT_ENGINE",
    "TTSHubClient",
    "LocalTTSUnavailable",
    "normalize_accent_family",
    "filter_catalog_voices",
    "available_accent_families",
    "accent_family_label",
]
