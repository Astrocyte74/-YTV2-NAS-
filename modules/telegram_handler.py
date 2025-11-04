"""
Telegram Bot Handler Module

This module contains the YouTubeTelegramBot class extracted from the monolithic file.
It handles all Telegram bot interactions without embedded HTML generation.
"""

import asyncio
import json
import io
import logging
import os
import re
import subprocess
import time
import urllib.parse
from datetime import datetime
import random
from typing import List, Dict, Any, Optional, Callable, Tuple, Awaitable, Set
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import RetryAfter, BadRequest
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.constants import ParseMode

from export_utils import SummaryExporter
from modules.report_generator import JSONReportGenerator, create_report_from_youtube_summarizer
from modules import ledger
from modules.metrics import metrics
from modules.event_stream import emit_report_event
from modules.summary_variants import merge_summary_variants, normalize_variant_id

from youtube_summarizer import YouTubeSummarizer
from llm_config import llm_config, get_quick_cloud_env_model
from modules.tts_hub import (
    TTSHubClient,
    LocalTTSUnavailable,
    accent_family_label,
    available_accent_families,
    filter_catalog_voices,
    DEFAULT_ENGINE,
)
from modules.telegram.ui.formatting import (
    escape_markdown as ui_escape_markdown,
    split_text_into_chunks as ui_split_chunks,
)
from modules.telegram.handlers.captions import build_ai2ai_audio_caption
from modules.telegram.handlers import ai2ai as ai2ai_handler
from modules.telegram.handlers import single as single_handler
from modules.telegram.ui.keyboards import build_ollama_models_keyboard as ui_build_models_keyboard
from modules.telegram.ui.summary import (
    build_summary_keyboard as ui_build_summary_keyboard,
    existing_variants_message as ui_existing_variants_message,
    friendly_variant_label as ui_friendly_variant_label,
    build_summary_provider_keyboard as ui_build_summary_provider_keyboard,
    build_summary_model_keyboard as ui_build_summary_model_keyboard,
)
from modules.telegram.ui.tts import (
    build_local_failure_keyboard as ui_build_local_failure_keyboard,
    build_tts_catalog_keyboard as ui_build_tts_catalog_keyboard,
    build_tts_keyboard as ui_build_tts_keyboard,
    gender_label as ui_gender_label,
    short_engine_label,
    strip_favorite_label,
    tts_prompt_text as ui_tts_prompt_text,
    tts_voice_label as ui_tts_voice_label,
)
from modules.ollama_client import (
    OllamaClientError,
    get_models as ollama_get_models,
    pull as ollama_pull,
    health_summary as ollama_health_summary,
)
from modules.services import cloud_service
from modules.services import draw_service
from modules.services import ollama_service, summary_service, tts_service
from modules.services.reachability import hub_ok as reach_hub_ok, hub_ollama_ok as reach_hub_ollama_ok
import hashlib
from pydub import AudioSegment


def _clean_label(value: Optional[str]) -> str:
    if not value:
        return ""
    parts = [seg.strip() for seg in value.replace("-", "_").split("_") if seg.strip()]
    words: List[str] = []
    for seg in parts:
        lower = seg.lower()
        if lower == "nsfw":
            words.append("NSFW")
        else:
            words.append(seg.capitalize())
    return " ".join(words) if words else value


def _family_from_name(name: Optional[str]) -> str:
    if not isinstance(name, str) or not name:
        return "general"
    lowered = name.lower()
    if "flux" in lowered:
        return "flux"
    if any(token in lowered for token in ("hidream", "i1", "sdxl")):
        return "hidream"
    return "general"


def _draw_family_label(family: Optional[str]) -> str:
    mapping = {
        "flux": "Flux",
        "hidream": "HiDream SDXL",
        "general": "General",
    }
    return mapping.get((family or "").strip().lower(), "General")


def _default_style_key(
    style_map: Dict[str, Dict[str, Any]],
    orders: Optional[Dict[str, Any]],
    family: Optional[str],
) -> Optional[str]:
    if not style_map:
        return None
    preferred: List[str] = []
    fam = (family or "").strip().lower()
    if fam in ("flux", "hidream"):
        preferred = ["photoreal", "cinematic"]
    else:
        preferred = ["photoreal"]
    preferred.extend(["photorealistic", "photo"])
    for key in preferred:
        if key in style_map:
            return key
    order_keys = (orders or {}).get("stylePresets") or []
    for key in order_keys:
        if key in style_map:
            return key
    # fallback to first available key
    return next(iter(style_map.keys()), None)


def _draw_health_warning(health: Dict[str, Any]) -> Optional[str]:
    if not health:
        return None
    if not health.get("reachable", True):
        return "📴 Draw Things appears offline."
    probe = health.get("probe") or {}
    if not probe.get("ok", True):
        return "⚠️ Draw Things is not ready yet; open the app and ensure the model is loaded."
    elapsed = probe.get("elapsedMs")
    if isinstance(elapsed, (int, float)) and elapsed > 5000:
        seconds = elapsed / 1000.0
        return f"🐢 Draw Things is responding slowly (~{seconds:.1f}s)."
    return None


_PRESET_HINTS = {
    "flux_ultra": "⚡ 3-step turbo",
    "flux_fast": "⚡ 4-step quick",
    "flux_balanced": "🎯 6-step crisp",
    "flux_photoreal": "🖼️ 8-step detail",
    "hidream_fast": "⚡ 24-step sharp",
    "hidream_balanced": "🎯 28-step detail",
    "hidream_photoreal": "🖼️ 32-step glossy",
}


def _preset_with_hint(entry: Dict[str, Any]) -> str:
    key = entry.get("key") or ""
    base = entry.get("label") or _clean_label(key)
    hint = _PRESET_HINTS.get(key)
    if hint:
        return f"{base} — {hint}"
    steps = entry.get("steps")
    if isinstance(steps, (int, float)) and steps > 0:
        return f"{base} — {int(steps)} steps"
    return base


def _friendly_model_name(raw: Optional[str], overrides: List[Dict[str, str]]) -> Optional[str]:
    if not isinstance(raw, str) or not raw:
        return None
    lowered = raw.lower()
    lowered_clean = lowered.replace("_", " ").replace("-", " ")
    for opt in overrides:
        name = opt.get("name")
        if not name:
            continue
        name_lower = name.lower()
        name_clean = name_lower.replace("_", " ").replace("-", " ")
        if (
            name_lower == lowered
            or name_clean == lowered_clean
            or name_lower in lowered
            or lowered in name_lower
            or name_clean in lowered_clean
            or lowered_clean in name_clean
        ):
            return name
    # fallback: replace separators for readability
    display = raw.replace("_", " ").replace("-", " ").strip()
    return display or raw


def _friendly_engine_label(raw: Optional[str], overrides: List[Dict[str, str]]) -> Optional[str]:
    if not isinstance(raw, str) or not raw:
        return None
    friendly = _friendly_model_name(raw, overrides)
    if friendly:
        return friendly
    label = raw
    for suffix in (".ckpt", ".safetensors"):
        if label.lower().endswith(suffix):
            label = label[: -len(suffix)]
            break
    label = label.replace("_", " ").replace("-", " ").strip()
    return label or raw


def _load_draw_models() -> List[Dict[str, str]]:
    raw = (os.getenv("DRAW_MODELS") or "").strip()
    models: List[Dict[str, str]] = []
    if raw:
        for chunk in raw.split(","):
            part = chunk.strip()
            if not part:
                continue
            if ":" in part:
                name, group = part.split(":", 1)
            else:
                name, group = part, "general"
            name = name.strip()
            group = group.strip().lower() or "general"
            if not name:
                continue
            models.append({"name": name, "group": group})
    if not models:
        models = [
            {"name": "Flux.1 [schnell]", "group": "flux"},
            {"name": "HiDream I1 fast", "group": "hidream"},
        ]
    # Deduplicate while preserving order
    seen = set()
    deduped: List[Dict[str, str]] = []
    for entry in models:
        key = (entry["name"], entry["group"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


class YouTubeTelegramBot:
    """Telegram bot for YouTube video summarization."""
    VARIANT_LABELS = {
        'comprehensive': "📝 Comprehensive",
        'bullet-points': "🎯 Key Points",
        'key-insights': "💡 Insights",
        'audio': "🎙️ Audio Summary",
        'audio-fr': "🎙️ Audio français 🇫🇷",
        'audio-es': "🎙️ Audio español 🇪🇸",
    }
    DRAW_SIZE_PRESETS = {
        "small": {"width": 512, "height": 512, "label": "Small • 512²"},
        "medium": {"width": 768, "height": 768, "label": "Medium • 768²"},
        "large": {"width": 1024, "height": 1024, "label": "Large • 1024²"},
    }

    def __init__(self, token: str, allowed_user_ids: List[int]):
        """
        Initialize the Telegram bot.
        
        Args:
            token: Telegram bot token
            allowed_user_ids: List of user IDs allowed to use the bot
        """
        self.token = token
        self.allowed_user_ids = set(allowed_user_ids)
        self.application = None
        self.summarizer = None
        self.current_item: Optional[Dict[str, Any]] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Initialize exporters and ensure directories exist
        Path("./data/reports").mkdir(parents=True, exist_ok=True)
        Path("./exports").mkdir(parents=True, exist_ok=True)
        Path("./data/tts_cache").mkdir(parents=True, exist_ok=True)
        self.html_exporter = SummaryExporter("./exports")
        self.json_exporter = JSONReportGenerator("./data/reports")
        # In-memory cache for one-off TTS playback keyed by (chat_id, message_id)
        self.tts_cache: Dict[tuple, Dict[str, Any]] = {}
        self.tts_client: Optional[TTSHubClient] = None
        self.started_at = time.time()
        # Interactive TTS sessions keyed by (chat_id, message_id)
        self.tts_sessions: Dict[tuple, Dict[str, Any]] = {}
        # Content-anchored preselect for TTS combos keyed by normalized video_id
        self.tts_preselect_by_content: Dict[str, Dict[str, Any]] = {}
        # Interactive summary sessions keyed by (chat_id, message_id)
        self.summary_sessions: Dict[tuple, Dict[str, Any]] = {}
        # Interactive Draw Things sessions keyed by (chat_id, message_id)
        self.draw_sessions: Dict[tuple, Dict[str, Any]] = {}
        # Cache of instantiated summarizers keyed by provider/model
        self.summarizer_cache: Dict[str, YouTubeSummarizer] = {}
        # Ollama chat sessions keyed by chat_id
        self.ollama_sessions: Dict[int, Dict[str, Any]] = {}
        # Auto-process scheduler for idle URLs (keyed by (chat_id, message_id))
        self._auto_tasks: Dict[tuple, asyncio.Task] = {}

        # Persisted user preferences (e.g., last-used cloud/local models for quick picks)
        self.user_prefs_path = Path("./data/user_prefs.json")
        self.user_prefs: Dict[str, Any] = {}
        try:
            self.user_prefs_path.parent.mkdir(parents=True, exist_ok=True)
            if self.user_prefs_path.exists():
                self.user_prefs = json.loads(self.user_prefs_path.read_text(encoding="utf-8")) or {}
        except Exception:
            self.user_prefs = {}
        
        # YouTube URL regex pattern (supports www./m., watch, embed, v, shorts, live, and youtu.be)
        self.youtube_url_pattern = re.compile(
            r'(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/|shorts/|live/)|youtu\.be/)([a-zA-Z0-9_-]{11})'
        )
        # Reddit URL regex pattern (support subdomains like www., old., m.)
        self.reddit_url_pattern = re.compile(
            r'(https?://(?:[a-z]+\.)?(?:reddit\.com|redd\.it)/[^\s]+)',
            re.IGNORECASE,
        )
        self.web_url_pattern = re.compile(r'(https?://[^\s<>]+)', re.IGNORECASE)

        # Telegram message length limit
        self.MAX_MESSAGE_LENGTH = 4096

        # Cache for URLs
        self.url_cache = {}
        self.CACHE_TTL = 3600  # 1 hour TTL for cached URLs

        self.draw_model_switch_enabled = (
            (os.getenv("DRAW_MODEL_SWITCH_ENABLED") or "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self.draw_models_overrides = _load_draw_models()  # cache for later use

        # Initialize summarizer
        try:
            llm_config.load_environment()
            self.summarizer = YouTubeSummarizer()
            self._cache_summarizer_instance(self.summarizer)
            logging.info(f"✅ YouTube summarizer initialized with {self.summarizer.llm_provider}/{self.summarizer.model}")
        except Exception as e:
            logging.error(f"Failed to initialize YouTubeSummarizer: {e}")

    # ------------------------------------------------------------------
    # Utility helpers for reprocessing and internal orchestration
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_content_id(content_id: str) -> str:
        content_id = (content_id or '').strip()
        if ':' in content_id:
            content_id = content_id.split(':', 1)[1]
        return content_id

    @staticmethod
    def _extract_video_id(youtube_url: str) -> str:
        """Extract plain video id from a YouTube URL."""
        youtube_url = (youtube_url or '').strip()
        if 'v=' in youtube_url:
            return youtube_url.split('v=', 1)[1].split('&')[0]
        return youtube_url.rstrip('/').split('/')[-1]

    def _extract_reddit_url(self, text: str) -> Optional[str]:
        """Extract the first Reddit URL from the provided text."""
        if not text:
            return None
        match = self.reddit_url_pattern.search(text)
        if not match:
            return None
        url = match.group(1)
        return url.strip('<>') if url else None

    @staticmethod
    def _extract_reddit_id(reddit_url: str) -> Optional[str]:
        """Extract the submission ID from a Reddit URL."""
        try:
            parsed = urllib.parse.urlparse(reddit_url)
        except Exception:
            return None

        path = (parsed.path or "").strip("/")
        if not path:
            return None

        parts = path.split("/")
        # /r/sub/comments/<id>/...
        if len(parts) >= 4 and parts[0].lower() == "r" and parts[2].lower() == "comments":
            return parts[3]
        # /r/sub/s/<token> (Reddit share links)
        if len(parts) >= 3 and parts[-2].lower() == 's':
            return parts[-1]
        # redd.it/<id>
        if parsed.netloc.endswith("redd.it") and parts:
            return parts[0]
        return None

    def _extract_web_url(self, text: str) -> Optional[str]:
        """Extract the first supported generic web URL from the text."""
        if not text:
            return None
        for match in self.web_url_pattern.finditer(text):
            candidate = match.group(1)
            candidate = candidate.strip('<>') if candidate else None
            if candidate and self._is_supported_web_domain(candidate):
                return candidate
        return None

    @staticmethod
    def _is_supported_web_domain(url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            return False
        host = (parsed.netloc or "").lower()
        if not host:
            return False
        blocked_domains = (
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
            "reddit.com",
            "www.reddit.com",
            "old.reddit.com",
            "redd.it",
        )
        return host not in blocked_domains

    def _current_source(self) -> str:
        return (self.current_item or {}).get("source", "youtube")

    def _current_content_id(self) -> Optional[str]:
        return (self.current_item or {}).get("content_id")

    def _current_normalized_id(self) -> Optional[str]:
        content_id = self._current_content_id()
        return self._normalize_content_id(content_id) if content_id else None

    def _current_url(self) -> Optional[str]:
        return (self.current_item or {}).get("url")

    def _friendly_variant_label(self, variant: str) -> str:
        return ui_friendly_variant_label(variant, self.VARIANT_LABELS)

    def _build_summary_keyboard(
        self,
        existing_variants: Optional[List[str]] = None,
        video_id: Optional[str] = None,
    ) -> InlineKeyboardMarkup:
        dashboard_url = None
        if video_id:
            dashboard_url = (
                os.getenv('DASHBOARD_URL')
                or os.getenv('POSTGRES_DASHBOARD_URL')
                or 'https://ytv2-dashboard-postgres.onrender.com'
            )
        return ui_build_summary_keyboard(
            self.VARIANT_LABELS,
            existing_variants=existing_variants,
            video_id=video_id,
            dashboard_url=dashboard_url,
        )

    def _existing_variants_message(self, content_id: str, variants: List[str], source: str = "youtube") -> str:
        return ui_existing_variants_message(self.VARIANT_LABELS, content_id, variants, source)

    async def _send_existing_summary_notice(self, query, video_id: str, summary_type: str):
        await summary_service.send_existing_summary_notice(self, query, video_id, summary_type)

    def _discover_summary_types(self, video_id: str) -> List[str]:
        """Infer summary types available for a video from reports or ledger."""
        if not video_id:
            return []

        video_id = self._normalize_content_id(video_id)
        summary_types: set[str] = set()
        reports_dir = Path('./data/reports')

        for path in reports_dir.glob(f'*{video_id}*.json'):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                summary_meta = data.get('summary') or {}
                summary_type = normalize_variant_id(summary_meta.get('summary_type') or summary_meta.get('type'))
                if summary_type:
                    summary_types.add(summary_type)

                variants_meta = summary_meta.get('variants')
                if isinstance(variants_meta, list):
                    for entry in variants_meta:
                        if not isinstance(entry, dict):
                            continue
                        variant_id = normalize_variant_id(entry.get('variant') or entry.get('summary_type') or entry.get('type'))
                        if variant_id:
                            summary_types.add(variant_id)
                # Also consider top-level summary_variants used for Postgres ingest (if present)
                top_variants = data.get('summary_variants')
                if isinstance(top_variants, list):
                    for entry in top_variants:
                        if isinstance(entry, dict):
                            vid = normalize_variant_id(entry.get('variant') or entry.get('summary_type') or entry.get('type'))
                            if vid:
                                summary_types.add(vid)
            except Exception:
                continue

        if summary_types:
            found_local = sorted(summary_types)
        else:
            found_local = []

        ledger_data = ledger.list_all()
        for key in ledger_data.keys():
            try:
                ledger_video, summary_type = key.rsplit(':', 1)
            except ValueError:
                continue
            if self._normalize_content_id(ledger_video) == video_id:
                normalized = normalize_variant_id(summary_type)
                if normalized:
                    summary_types.add(normalized)

        # Optional: query dashboard API for authoritative variants if available
        # Prefer Postgres dashboard URL; also support legacy env var names
        dash_url = (
            os.getenv('DASHBOARD_URL')
            or os.getenv('POSTGRES_DASHBOARD_URL')
            or os.getenv('RENDER_DASHBOARD_URL')
        )
        if dash_url and requests is not None:
            try:
                url = f"{dash_url.rstrip('/')}/api/reports/{video_id}"
                headers = {}
                token = os.getenv('DASHBOARD_TOKEN')
                if token:
                    headers['Authorization'] = f"Bearer {token}"
                r = requests.get(url, headers=headers, timeout=6)
                if r.status_code == 200:
                    payload = r.json() or {}
                    # New Postgres payload shape
                    sv = payload.get('summary_variants')
                    if isinstance(sv, list):
                        for entry in sv:
                            if isinstance(entry, dict):
                                vid = normalize_variant_id(entry.get('variant') or entry.get('summary_type') or entry.get('type'))
                                if vid:
                                    summary_types.add(vid)
                    # Legacy payloads
                    s = payload.get('summary') or {}
                    if isinstance(s, dict):
                        v = s.get('variants')
                        if isinstance(v, list):
                            for entry in v:
                                if isinstance(entry, dict):
                                    vid = normalize_variant_id(entry.get('variant') or entry.get('summary_type') or entry.get('type'))
                                    if vid:
                                        summary_types.add(vid)
                        st = normalize_variant_id(s.get('summary_type') or s.get('type'))
                        if st:
                            summary_types.add(st)
            except Exception:
                # Network or parsing errors should not block local detection
                pass

        return sorted(summary_types)

    def _resolve_video_url(self, video_id: str, provided_url: Optional[str] = None) -> Optional[str]:
        """Find the best URL for a given video id."""
        if provided_url:
            return provided_url

        video_id = self._normalize_content_id(video_id)
        reports_dir = Path('./data/reports')

        for path in reports_dir.glob(f'*{video_id}*.json'):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                url = data.get('canonical_url') or data.get('video', {}).get('url')
                if url:
                    return url
            except Exception:
                continue

        ledger_data = ledger.list_all()
        for key, entry in ledger_data.items():
            try:
                ledger_video, _ = key.split(':', 1)
            except ValueError:
                continue
            if ledger_video != video_id:
                continue
            json_path = entry.get('json')
            if json_path:
                try:
                    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
                    url = data.get('canonical_url') or data.get('video', {}).get('url')
                    if url:
                        return url
                except Exception:
                    continue

        # Fallback if nothing else found
        if len(video_id) == 11:
            return f"https://www.youtube.com/watch?v={video_id}"
        return None

    async def _generate_tts_audio_file(self, summary_text: str, video_id: str, json_path: Path) -> Optional[str]:
        return await summary_service.generate_tts_audio_file(self, summary_text, video_id, json_path)

    async def _reprocess_single_summary(
        self,
        video_id: str,
        video_url: str,
        summary_type: str,
        ledger_entry: Optional[Dict[str, Any]] = None,
        force: bool = False,
        regenerate_audio: bool = True,
    ) -> Dict[str, Any]:
        return await summary_service.reprocess_single_summary(
            self,
            video_id,
            video_url,
            summary_type,
            ledger_entry=ledger_entry,
            force=force,
            regenerate_audio=regenerate_audio,
        )

    async def reprocess_video(
        self,
        video_id: str,
        summary_types: Optional[List[str]] = None,
        force: bool = False,
        regenerate_audio: bool = True,
        video_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await summary_service.reprocess_video(
            self,
            video_id,
            summary_types=summary_types,
            force=force,
            regenerate_audio=regenerate_audio,
            video_url=video_url,
        )

    def setup_handlers(self):
        """Set up bot command and message handlers."""
        if not self.application:
            return
            
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        # Alias: /s → /status
        self.application.add_handler(CommandHandler("s", self.status_command))
        self.application.add_handler(CommandHandler("restart", self.restart_command))
        self.application.add_handler(CommandHandler("r", self.restart_command))
        self.application.add_handler(CommandHandler("logs", self.logs_command))
        self.application.add_handler(CommandHandler("diag", self.diag_command))
        self.application.add_handler(CommandHandler("tts", self.tts_command))
        self.application.add_handler(CommandHandler("draw", self.draw_command))
        self.application.add_handler(CommandHandler("d", self.draw_command))
        # Ollama chat commands (aliases: /o, /o_stop, /stop)
        self.application.add_handler(CommandHandler("ollama", self.ollama_command))
        self.application.add_handler(CommandHandler("o", self.ollama_command))
        self.application.add_handler(CommandHandler("o_stop", self.ollama_stop_command))
        self.application.add_handler(CommandHandler("stop", self.ollama_stop_command))
        self.application.add_handler(CommandHandler("ollama_stop", self.ollama_stop_command))
        self.application.add_handler(CommandHandler("chat", self.ollama_ai2ai_chat_command))
        # (moved above)
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Callback query handler for inline keyboards
        self.application.add_handler(CallbackQueryHandler(self.handle_callback_query))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)

    async def restart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Restart the container by exiting the process (Docker restarts it)."""
        user_id = update.effective_user.id
        if not self._is_user_allowed(user_id):
            msg = getattr(update, 'message', None) or getattr(getattr(update, 'callback_query', None), 'message', None)
            if msg:
                await msg.reply_text("❌ You are not authorized to use this bot.")
            return
        # Send confirmation message regardless of whether this came from a slash or a button
        msg = getattr(update, 'message', None) or getattr(getattr(update, 'callback_query', None), 'message', None)
        if msg:
            await msg.reply_text("♻️ Restarting the bot container…")
        elif getattr(update, 'effective_chat', None):
            try:
                await context.bot.send_message(update.effective_chat.id, "♻️ Restarting the bot container…")
            except Exception:
                pass
        # Persist notify target so we can confirm after restart
        try:
            from pathlib import Path as _P
            import json as _J
            _P('data').mkdir(exist_ok=True)
            chat_id = (getattr(getattr(update, 'effective_chat', None), 'id', None))
            if chat_id:
                (_P('data')/ 'restart_notify.json').write_text(_J.dumps({
                    'chat_id': chat_id,
                    'ts': int(time.time())
                }), encoding='utf-8')
        except Exception:
            pass
        await asyncio.sleep(0.5)
        # Attempt to terminate PID 1 (entrypoint) to ensure full container exit; fall back to self-exit
        import os, signal, time as _t
        try:
            os.kill(1, signal.SIGTERM)
            _t.sleep(1.0)
        except Exception:
            pass
        os._exit(0)

    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show tail of bot.log. Usage: /logs [lines] (default 80)."""
        user_id = update.effective_user.id
        target = getattr(update, 'message', None) or getattr(getattr(update, 'callback_query', None), 'message', None)
        if not self._is_user_allowed(user_id):
            if target:
                await target.reply_text("❌ You are not authorized to use this bot.")
            return
        try:
            n = 80
            if context.args:
                try:
                    n = max(10, min(500, int(context.args[0])))
                except Exception:
                    pass
            path = Path('bot.log')
            if not path.exists():
                if target:
                    await target.reply_text("⚠️ bot.log not found.")
                return
            lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
            tail = lines[-n:]
            text = "\n".join(tail)
            if len(text) > 3500:
                text = text[-3500:]
                text = "…\n" + text
            if target:
                await target.reply_text(f"```\n{text}\n```", parse_mode=ParseMode.MARKDOWN)
        except Exception as exc:
            if target:
                await target.reply_text(f"❌ logs error: {exc}")

    async def diag_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Run quick diagnostics inside the container."""
        user_id = update.effective_user.id
        target = getattr(update, 'message', None) or getattr(getattr(update, 'callback_query', None), 'message', None)
        if not self._is_user_allowed(user_id):
            if target:
                await target.reply_text("❌ You are not authorized to use this bot.")
            return
        import sys, platform, shutil, subprocess, json
        # System
        py = sys.version.split()[0]
        plat = platform.platform()
        # Tools
        def _ver(cmd, arg='--version'):
            try:
                r = subprocess.run([cmd, arg], capture_output=True, text=True, timeout=4)
                if r.returncode == 0:
                    out = (r.stdout or r.stderr or '').strip().splitlines()[0]
                    return out[:120]
                return f"{cmd}: rc={r.returncode}"
            except FileNotFoundError:
                return f"{cmd}: not found"
            except Exception as e:
                return f"{cmd}: {e}"
        ytdlp = _ver('yt-dlp')
        ffmpeg = _ver('ffmpeg')
        # Disk
        try:
            du = shutil.disk_usage('/')
            gb = 1024**3
            disk_line = f"{du.used//gb}G used / {du.total//gb}G total (free {du.free//gb}G)"
        except Exception:
            disk_line = "unknown"
        # Hub health
        try:
            hs = ollama_health_summary(cache_ttl=5) or {}
        except Exception as exc:
            hs = {"provider": None, "base": None, "reachable": False, "models": None, "notes": str(exc)}
        # Postgres
        db = os.getenv('DATABASE_URL')
        db_line = 'unset'
        if db:
            try:
                import psycopg
                with psycopg.connect(db, connect_timeout=3) as conn:
                    with conn.cursor() as cur:
                        cur.execute('SELECT 1')
                        cur.fetchone()
                db_line = '✅ ok'
            except Exception as e:
                db_line = f"❌ {str(e).splitlines()[0][:200]}"
        # Uptime
        up_secs = int(max(0, time.time() - (self.started_at or time.time())))
        hrs = up_secs // 3600; mins = (up_secs % 3600) // 60; secs = up_secs % 60

        lines = [
            "🧪 Diagnostics:",
            f"• Python: {py}",
            f"• Platform: {plat}",
            f"• yt-dlp: {ytdlp}",
            f"• ffmpeg: {ffmpeg}",
            f"• Disk: {disk_line}",
            f"• Uptime: {hrs}h {mins}m {secs}s",
            "",
            "🧩 Local LLM:",
            f"• Provider: {hs.get('provider') or 'none'}",
            f"• Base: {hs.get('base') or 'unset'}",
            f"• Reachable: {'✅' if hs.get('reachable') else '❌'}",
            f"• Models: {hs.get('models') if hs.get('models') is not None else 'unknown'}",
        ]
        note = (hs.get('notes') or '').strip()
        if note:
            lines.append(f"• Notes: {note}")
        lines.extend([
            "",
            "🗄️ Postgres:",
            f"• DATABASE_URL: {'set' if db else 'unset'}",
            f"• Connect: {db_line}",
        ])
        out = "\n".join(lines)
        if len(out) > 3500:
            out = out[:3500] + "\n…"
        if target:
            await target.reply_text(out)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user_id = update.effective_user.id
        user_name = update.effective_user.first_name or "Unknown"
        
        if not self._is_user_allowed(user_id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            logging.warning(f"Unauthorized access attempt by user {user_id} ({user_name})")
            return
        
        welcome_message = (
            f"🎬 Welcome to the YouTube Summarizer Bot, {user_name}!\n\n"
            "Send me a YouTube URL and I'll provide:\n"
            "• 🤖 AI-powered summary\n"
            "• 🎯 Key insights and takeaways\n"
            "• 📊 Content analysis\n\n"
            "Use /help for more commands."
        )
        
        await update.message.reply_text(welcome_message)
        logging.info(f"User {user_id} ({user_name}) started the bot")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        user_id = update.effective_user.id
        
        if not self._is_user_allowed(user_id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return
        
        help_message = (
            "🤖 YouTube Summarizer Bot Commands:\n\n"
            "/start - Start using the bot\n"
            "/help - Show this help message\n"
            "/status (/s) - Check bot and API status\n\n"
            "📝 How to use:\n"
            "1. Send a YouTube URL\n"
            "2. Choose summary type\n"
            "3. Get AI-powered insights\n\n"
            "Supported formats:\n"
            "• youtube.com/watch?v=...\n"
            "• youtu.be/...\n"
            "• m.youtube.com/..."
        )
        
        await update.message.reply_text(help_message)
    
    async def _check_api_credentials(self) -> Dict[str, Dict[str, str]]:
        """
        Perform lightweight probes against supported cloud providers to verify API keys.
        Returns provider keyed dict with state/detail metadata.
        """
        if requests is None:  # type: ignore[name-defined]
            return {
                "openai": {"state": "skipped", "detail": "requests library unavailable"},
                "anthropic": {"state": "skipped", "detail": "requests library unavailable"},
                "openrouter": {"state": "skipped", "detail": "requests library unavailable"},
            }

        # Refresh environment-backed credentials so we pick up recent changes
        try:
            llm_config.load_environment()
        except Exception as exc:
            logging.warning("Failed to refresh LLM config before status probe: %s", exc)

        timeout = float(os.getenv("STATUS_API_PROBE_TIMEOUT", "6"))

        async def run_probe(func: Callable[[], Dict[str, str]]) -> Dict[str, str]:
            try:
                return await asyncio.to_thread(func)
            except Exception as exc:  # Defensive catch for unexpected threading errors
                return {"state": "error", "detail": f"{type(exc).__name__}: {exc}"}

        def truncate(payload: str) -> str:
            payload = (payload or "").strip()
            return payload if len(payload) <= 80 else f"{payload[:77]}..."

        def probe_openai() -> Dict[str, str]:
            key = getattr(llm_config, "openai_key", None)
            if not key:
                return {"state": "missing", "detail": "not set"}
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            url = "https://api.openai.com/v1/models?limit=1"
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)  # type: ignore[name-defined]
            except requests.RequestException as exc:  # type: ignore[name-defined]
                return {"state": "error", "detail": f"{type(exc).__name__}: {exc}"}
            if resp.status_code == 200:
                return {"state": "ok", "detail": "OK (200)"}
            reason = truncate(resp.reason or "")
            snippet = truncate(resp.text)
            msg = reason or f"HTTP {resp.status_code}"
            if snippet:
                msg = f"{msg} – {snippet}"
            return {"state": "error", "detail": msg}

        def probe_anthropic() -> Dict[str, str]:
            key = getattr(llm_config, "anthropic_key", None)
            if not key:
                return {"state": "missing", "detail": "not set"}
            headers = {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            }
            url = "https://api.anthropic.com/v1/models"
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)  # type: ignore[name-defined]
            except requests.RequestException as exc:  # type: ignore[name-defined]
                return {"state": "error", "detail": f"{type(exc).__name__}: {exc}"}
            if resp.status_code == 200:
                return {"state": "ok", "detail": "OK (200)"}
            reason = truncate(resp.reason or "")
            snippet = truncate(resp.text)
            msg = reason or f"HTTP {resp.status_code}"
            if snippet:
                msg = f"{msg} – {snippet}"
            return {"state": "error", "detail": msg}

        def probe_openrouter() -> Dict[str, str]:
            key = getattr(llm_config, "openrouter_key", None)
            if not key:
                return {"state": "missing", "detail": "not set"}
            headers = {
                "Authorization": f"Bearer {key}",
            }
            url = "https://openrouter.ai/api/v1/auth/key"
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)  # type: ignore[name-defined]
            except requests.RequestException as exc:  # type: ignore[name-defined]
                return {"state": "error", "detail": f"{type(exc).__name__}: {exc}"}
            if resp.status_code == 200:
                return {"state": "ok", "detail": "OK (200)"}
            reason = truncate(resp.reason or "")
            snippet = truncate(resp.text)
            msg = reason or f"HTTP {resp.status_code}"
            if snippet:
                msg = f"{msg} – {snippet}"
            return {"state": "error", "detail": msg}

        results = await asyncio.gather(
            run_probe(probe_openai),
            run_probe(probe_anthropic),
            run_probe(probe_openrouter),
        )
        summary = {
            "openai": results[0],
            "anthropic": results[1],
            "openrouter": results[2],
        }
        try:
            logging.debug(
                "Status API probes: openai=%s, anthropic=%s, openrouter=%s",
                summary["openai"].get("state"),
                summary["anthropic"].get("state"),
                summary["openrouter"].get("state"),
            )
        except Exception:
            pass
        return summary

    @staticmethod
    def _render_api_key_line(label: str, result: Optional[Dict[str, str]]) -> str:
        if not result:
            return f"• {label}: ⚠️ not checked"
        state = result.get("state", "error")
        detail = result.get("detail", "").strip()
        emoji = {
            "ok": "✅",
            "missing": "⚠️",
            "skipped": "⚠️",
            "error": "❌",
        }.get(state, "⚠️")
        if not detail:
            detail = {
                "ok": "OK",
                "missing": "not set",
                "skipped": "skipped",
                "error": "error",
            }.get(state, "unknown")
        return f"• {label}: {emoji} {detail}"

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        user_id = update.effective_user.id
        
        if not self._is_user_allowed(user_id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return
        
        # Check summarizer status
        summarizer_status = "✅ Ready" if self.summarizer else "❌ Not initialized"
        
        # Check LLM configuration
        try:
            llm_status = f"✅ {self.summarizer.llm_provider}/{self.summarizer.model}" if self.summarizer else "❌ Not configured"
        except Exception:
            llm_status = "❌ LLM not configured"

        # Local LLM (hub/direct) health
        hub_env = (os.getenv('TTSHUB_API_BASE') or '').strip()
        direct_env = (os.getenv('OLLAMA_URL') or os.getenv('OLLAMA_HOST') or '').strip()
        try:
            hs = ollama_health_summary(cache_ttl=20) or {}
        except Exception as exc:
            hs = {"provider": None, "base": None, "reachable": False, "models": None, "notes": str(exc)}

        def _mask(url: str) -> str:
            if not url:
                return 'unset'
            try:
                return url.split('://',1)[-1]
            except Exception:
                return url

        lines = [
            "📊 Bot Status:",
            f"🤖 Telegram Bot: ✅ Running",
            f"🔍 Summarizer: {summarizer_status}",
            f"🧠 LLM: {llm_status}",
            f"👥 Authorized Users: {len(self.allowed_user_ids)}",
            "",
            "🧩 Local LLM (hub/direct):",
            f"• TTSHUB_API_BASE: {_mask(hub_env)}",
            f"• OLLAMA_URL/HOST: {_mask(direct_env)}",
            f"• Selected: {hs.get('provider') or 'none'} → {_mask(str(hs.get('base') or ''))}",
            f"• Reachable: {'✅' if hs.get('reachable') else '❌'}",
            f"• Models: {hs.get('models') if hs.get('models') is not None else 'unknown'}",
        ]
        note = (hs.get('notes') or '').strip()
        if note:
            lines.append(f"• Notes: {note}")

        # Try to list installed model names (list all)
        try:
            tags = ollama_get_models() or {}
            raw = tags.get('models') or []
            names: List[str] = []
            for entry in raw:
                if isinstance(entry, dict):
                    name = (entry.get('name') or entry.get('model') or '').strip()
                    if name:
                        names.append(name)
            if names:
                joined = ", ".join(names)
                lines.append(f"• Installed Models: {joined}")
        except Exception:
            pass

        # Show key defaults/env for quick reference (no secrets)
        def env_or(key: str, default: str = 'unset') -> str:
            v = (os.getenv(key) or '').strip()
            return v if v else default

        lines.extend([
            "",
            "🎛️ Defaults (env):",
            f"• QUICK_LOCAL_MODEL: {env_or('QUICK_LOCAL_MODEL')}",
            f"• QUICK_CLOUD_MODEL: {env_or('QUICK_CLOUD_MODEL')}",
            f"• TTS_QUICK_FAVORITE: {env_or('TTS_QUICK_FAVORITE')}",
            f"• TTS_CLOUD_VOICE: {env_or('TTS_CLOUD_VOICE')}",
            f"• OLLAMA_DEFAULT_MODEL: {env_or('OLLAMA_DEFAULT_MODEL')}",
            f"• LLM_PROVIDER: {env_or('LLM_PROVIDER')}",
            f"• LLM_MODEL: {env_or('LLM_MODEL')}",
        ])

        # Cloud API credentials: probe actual validity when possible
        try:
            probe_results = await self._check_api_credentials()
        except Exception as exc:
            probe_results = {
                "openai": {"state": "error", "detail": f"probe failed: {exc}"},
                "anthropic": {"state": "error", "detail": f"probe failed: {exc}"},
                "openrouter": {"state": "error", "detail": f"probe failed: {exc}"},
            }
        lines.extend([
            self._render_api_key_line("OPENAI_API_KEY", probe_results.get("openai")),
            self._render_api_key_line("ANTHROPIC_API_KEY", probe_results.get("anthropic")),
            self._render_api_key_line("OPENROUTER_API_KEY", probe_results.get("openrouter")),
        ])

        # Auto-process snapshot
        ap_provider = env_or('AUTO_PROCESS_PROVIDER')
        ap_summary = env_or('AUTO_PROCESS_SUMMARY')
        ap_delay = env_or('AUTO_PROCESS_DELAY_SECONDS')
        lines.extend([
            "",
            "⚙️ Auto-Process:",
            f"• Provider(s): {ap_provider}",
            f"• Summary type(s): {ap_summary}",
            f"• Delay (s): {ap_delay}",
        ])

        # TTS Hub snapshot
        if hub_env:
            try:
                client = self.tts_client or TTSHubClient.from_env()
            except Exception:
                client = None
            if client:
                fav_labels: List[str] = []
                try:
                    favs = await client.fetch_favorites()
                    fav_labels = [str(f.get('label') or f.get('slug') or '') for f in (favs or []) if (f.get('label') or f.get('slug'))]
                except Exception:
                    favs = []
                engines: List[str] = []
                try:
                    cat = await client.fetch_catalog()
                    voices = (cat or {}).get('voices') or []
                    engines = sorted({(v.get('engine') or v.get('provider') or '').strip() for v in voices if (v.get('engine') or v.get('provider'))})
                except Exception:
                    engines = []
                lines.extend([
                    "",
                    "🎧 TTS Hub:",
                    f"• Favorites: {len(fav_labels)}",
                    f"• Favorite labels: {', '.join(fav_labels[:5]) if fav_labels else 'none'}",
                    f"• Engines: {', '.join([e for e in engines if e]) or 'unknown'}",
                ])

        # Queue snapshot
        try:
            q_enabled = env_or('ENABLE_TTS_QUEUE_WORKER')
            q_interval = env_or('TTS_QUEUE_INTERVAL')
            from pathlib import Path as _P
            qdir = _P('data/tts_queue')
            qcount = sum(1 for p in qdir.iterdir() if p.is_file()) if qdir.exists() else 0
            lines.extend([
                "",
                "📦 Queue:",
                f"• Worker enabled: {q_enabled}",
                f"• Interval (s): {q_interval}",
                f"• Jobs queued: {qcount}",
            ])
        except Exception:
            pass

        # Version/context: uptime + git rev
        try:
            up_secs = int(max(0, time.time() - (self.started_at or time.time())))
            hrs = up_secs // 3600; mins = (up_secs % 3600) // 60; secs = up_secs % 60
            from subprocess import run, PIPE
            rev = run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
            rev_str = (rev.stdout or '').strip() or 'unknown'
            lines.extend([
                "",
                "🧾 Context:",
                f"• Uptime: {hrs}h {mins}m {secs}s",
                f"• Git: {rev_str}",
            ])
        except Exception:
            pass

        # Dashboard status (version + storage health if token present)
        try:
            base = (os.getenv('RENDER_DASHBOARD_URL') or os.getenv('RENDER_API_URL') or '').strip().rstrip('/')
            if base and requests:
                # Version
                ver_txt = ''
                try:
                    vr = requests.get(f"{base}/api/version", timeout=8)
                    if vr.ok:
                        vj = vr.json() if vr.headers.get('content-type','').startswith('application/json') else {}
                        commit = (vj.get('deployment_commit') or '').strip()
                        branch = (vj.get('branch') or '').strip()
                        ver_txt = f"commit={commit or 'unknown'} branch={branch or 'unknown'}"
                    else:
                        ver_txt = f"HTTP {vr.status_code}"
                except Exception as exc:
                    ver_txt = f"error: {exc}"

                # Storage health (gated)
                health_txt = "unauthorized"
                tok = (os.getenv('DASHBOARD_DEBUG_TOKEN') or '').strip()
                if tok:
                    try:
                        hr = requests.get(
                            f"{base}/api/health/storage",
                            headers={"Authorization": f"Bearer {tok}"},
                            timeout=8,
                        )
                        if hr.ok:
                            hj = hr.json() if hr.headers.get('content-type','').startswith('application/json') else {}
                            used_pct = int(hj.get('used_pct') or 0)
                            free = int(hj.get('free_bytes') or 0)
                            total = int(hj.get('total_bytes') or 0)
                            def _fmt_bytes(n: int) -> str:
                                try:
                                    for unit in ['B','KB','MB','GB','TB']:
                                        if n < 1024:
                                            return f"{n}{unit}"
                                        n //= 1024
                                    return f"{n}PB"
                                except Exception:
                                    return str(n)
                            health_txt = f"used={used_pct}% free={_fmt_bytes(free)} total={_fmt_bytes(total)}"
                        else:
                            health_txt = f"HTTP {hr.status_code}"
                    except Exception as exc:
                        health_txt = f"error: {exc}"

                lines.extend([
                    "",
                    "🌐 Dashboard:",
                    f"• Base: {base}",
                    f"• Version: {ver_txt}",
                    f"• Storage: {health_txt}",
                ])
        except Exception:
            pass

        # Admin command hints
        lines.extend([
            "",
            "🔧 Admin shortcuts:",
            "• /diag  • /logs 120  • /restart",
        ])

        # Inline buttons for quick access
        try:
            kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Diagnostics", callback_data="status:diag"),
                    InlineKeyboardButton("Logs (120)", callback_data="status:logs:120"),
                ],
                [
                    InlineKeyboardButton("Restart", callback_data="status:restart"),
                ],
            ])
        except Exception:
            kb = None

        await update.message.reply_text("\n".join(lines), reply_markup=kb)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages with YouTube URLs."""
        user_id = update.effective_user.id
        user_name = update.effective_user.first_name or "Unknown"
        
        if not self._is_user_allowed(user_id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return
        
        message_text = update.message.text.strip()
        # If Ollama chat session active for this chat, route text to chat (unless it's a command)
        if message_text and not message_text.startswith("/"):
            chat_id = update.effective_chat.id
            session = self.ollama_sessions.get(chat_id)
            if session and (session.get("active") or session.get("mode") == "ai-ai"):
                await self._ollama_handle_user_text(update, session, message_text)
                return
            # Early dashboard storage gate (optional; only when token + base present)
            try:
                base = (os.getenv('RENDER_DASHBOARD_URL') or os.getenv('RENDER_API_URL') or '').strip().rstrip('/')
                tok = (os.getenv('DASHBOARD_DEBUG_TOKEN') or '').strip()
                if base and tok and requests:
                    hr = requests.get(
                        f"{base}/api/health/storage",
                        headers={"Authorization": f"Bearer {tok}"},
                        timeout=float(os.getenv('STATUS_API_PROBE_TIMEOUT', '6')),
                    )
                    if hr.ok and hr.headers.get('content-type','').startswith('application/json'):
                        used_pct = int((hr.json() or {}).get('used_pct') or 0)
                        # Informational thresholds
                        if used_pct >= 99:
                            await update.message.reply_text(
                                "🚫 Storage is critically full on the dashboard (≥99%). "
                                "Pausing new processing until space is freed. Try again soon."
                            )
                            return
                        elif used_pct >= 95:
                            await update.message.reply_text(
                                f"⚠️ Storage very high on dashboard (~{used_pct}%). "
                                "Uploads may be throttled or delayed."
                            )
                        elif used_pct >= 90:
                            await update.message.reply_text(
                                f"⚠️ Heads up: dashboard storage at ~{used_pct}%."
                            )
            except Exception:
                # Never block on probe errors
                pass
        logging.info(f"Received message from {user_name} ({user_id}): {message_text[:100]}...")
        
        # Check for YouTube links first (primary flow)
        youtube_match = self.youtube_url_pattern.search(message_text)
        if youtube_match:
            # OFF switch: allow disabling YouTube handling via env (e.g., cooldown for 429s)
            yt_access_raw = os.getenv("YOUTUBE_ACCESS", "true").strip().lower()
            yt_enabled = yt_access_raw in ("1", "true", "yes", "on")
            if not yt_enabled:
                await update.message.reply_text(
                    "⏸️ YouTube summaries are temporarily paused.\n"
                    "Please try again later, or send a Reddit or web article link in the meantime."
                )
                return
            video_url = self._extract_youtube_url(message_text)
            if not video_url:
                await update.message.reply_text("❌ Could not extract a valid YouTube URL from your message.")
                return

            raw_video_id = self._extract_video_id(video_url)
            normalized_id = self._normalize_content_id(raw_video_id)

            self.current_item = {
                "source": "youtube",
                "url": video_url,
                "content_id": normalized_id,
                "raw_id": raw_video_id,
                "normalized_id": normalized_id,
            }

            existing_variants = self._discover_summary_types(normalized_id)
            reply_markup = self._build_summary_keyboard(existing_variants, normalized_id)
            message_text = self._existing_variants_message(normalized_id, existing_variants, source="youtube")

            reply_msg = await update.message.reply_text(
                message_text,
                reply_markup=reply_markup
            )
            # Auto-process if configured
            await self._maybe_schedule_auto_process(
                reply_msg,
                source="youtube",
                url=video_url,
                content_id=normalized_id,
                user_name=user_name,
            )
            return

        # Check for Reddit submission URLs
        reddit_url = self._extract_reddit_url(message_text)
        if reddit_url:
            reddit_url = reddit_url.strip()
            reddit_id = self._extract_reddit_id(reddit_url)
            if not reddit_id:
                await update.message.reply_text("❌ Could not determine Reddit thread ID from that link.")
                return

            content_id = f"reddit:{reddit_id}"
            self.current_item = {
                "source": "reddit",
                "url": reddit_url,
                "content_id": content_id,
                "raw_id": reddit_id,
                "normalized_id": reddit_id,
            }

            existing_variants = self._discover_summary_types(content_id)
            reply_markup = self._build_summary_keyboard(existing_variants, content_id)
            message_text = self._existing_variants_message(content_id, existing_variants, source="reddit")

            reply_msg = await update.message.reply_text(
                message_text,
                reply_markup=reply_markup
            )
            await self._maybe_schedule_auto_process(
                reply_msg,
                source="reddit",
                url=reddit_url,
                content_id=content_id,
                user_name=user_name,
            )
            return

        # Check for generic web URLs
        web_url = self._extract_web_url(message_text)
        if web_url:
            web_url = web_url.strip()
            # Resolve shorteners (e.g., flip.it) to the final URL for better summaries and dedupe
            resolved = self._resolve_redirects(web_url)
            if isinstance(resolved, str) and resolved.strip():
                web_url = resolved.strip()
            hashed = hashlib.sha256(web_url.encode('utf-8')).hexdigest()[:24]
            content_id = f"web:{hashed}"
            self.current_item = {
                "source": "web",
                "url": web_url,
                "content_id": content_id,
                "raw_id": hashed,
                "normalized_id": hashed,
            }

            existing_variants = self._discover_summary_types(content_id)
            reply_markup = self._build_summary_keyboard(existing_variants, hashed)
            message_text = self._existing_variants_message(content_id, existing_variants, source="web")

            reply_msg = await update.message.reply_text(
                message_text,
                reply_markup=reply_markup
            )
            await self._maybe_schedule_auto_process(
                reply_msg,
                source="web",
                url=web_url,
                content_id=content_id,
                user_name=user_name,
            )
            return

        await update.message.reply_text(
            "🔍 Please send a YouTube or Reddit URL to get started.\n\n"
            "Supported YouTube formats:\n"
            "• https://youtube.com/watch?v=...\n"
            "• https://youtu.be/...\n"
            "• https://m.youtube.com/watch?v=...\n\n"
            "Supported Reddit formats:\n"
            "• https://www.reddit.com/r/<sub>/comments/<id>/...\n"
            "• https://redd.it/<id>\n\n"
            "Supported Web articles:\n"
            "• Any https:// link (except YouTube/Reddit)"
        )

    # ------------------------- Summary provider helpers -------------------------

    def _summary_session_key(self, chat_id: int, message_id: int) -> tuple:
        return (chat_id, message_id)

    def _store_summary_session(self, chat_id: int, message_id: int, payload: Dict[str, Any]) -> None:
        self.summary_sessions[self._summary_session_key(chat_id, message_id)] = payload

    def _get_summary_session(self, chat_id: int, message_id: int) -> Optional[Dict[str, Any]]:
        return self.summary_sessions.get(self._summary_session_key(chat_id, message_id))

    def _remove_summary_session(self, chat_id: int, message_id: int) -> None:
        self.summary_sessions.pop(self._summary_session_key(chat_id, message_id), None)

    def _summarizer_cache_key(self, provider: Optional[str], model: Optional[str]) -> str:
        provider_key = (provider or "unknown").strip().lower()
        model_key = (model or "").strip()
        return f"{provider_key}::{model_key}"

    def _cache_summarizer_instance(self, summarizer: Optional[YouTubeSummarizer]) -> None:
        if not summarizer:
            return
        key = self._summarizer_cache_key(
            getattr(summarizer, "llm_provider", None),
            getattr(summarizer, "model", None),
        )
        self.summarizer_cache[key] = summarizer

    def _cached_ollama_summarizer(self) -> Optional[YouTubeSummarizer]:
        for cached in self.summarizer_cache.values():
            if getattr(cached, "llm_provider", "").lower() == "ollama":
                return cached
        return None

    # ------------------------- Quick pick helpers -------------------------
    def _save_user_prefs(self) -> None:
        try:
            self.user_prefs_path.write_text(json.dumps(self.user_prefs, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _remember_last_model(self, user_id: int, provider_key: str, model_slug: Optional[str]) -> None:
        if not model_slug:
            return
        uid = str(user_id)
        prefs = self.user_prefs.get(uid) or {}
        if provider_key == "ollama":
            prefs["last_local_model"] = model_slug
        else:
            prefs["last_cloud_model"] = model_slug
        self.user_prefs[uid] = prefs
        self._save_user_prefs()

    def _short_model_name(self, model: Optional[str]) -> str:
        if not model:
            return ""
        return model.split("/", 1)[1] if "/" in model else model

    def _quick_pick_candidates(self, provider_options: Dict[str, Dict[str, Any]], user_id: int) -> Dict[str, Optional[str]]:
        mode = (os.getenv("QUICK_PICK_MODE") or "auto").strip().lower()
        env_cloud = get_quick_cloud_env_model()
        env_local = (os.getenv("QUICK_LOCAL_MODEL") or "").strip()

        uid = str(user_id)
        last = self.user_prefs.get(uid) or {}
        last_cloud = (last.get("last_cloud_model") or "").strip()
        last_local = (last.get("last_local_model") or "").strip()

        def first_slug(key: str) -> Optional[str]:
            opt = provider_options.get(key) or {}
            models = opt.get("models") or []
            if not models:
                return None
            return (models[0].get("model") or "").strip() or None

        pick_cloud: Optional[str] = None
        pick_local: Optional[str] = None

        if provider_options.get("cloud"):
            if mode == "env" and env_cloud:
                pick_cloud = env_cloud
            elif mode == "last" and last_cloud:
                pick_cloud = last_cloud
            else:  # auto or fallback
                pick_cloud = env_cloud or last_cloud or first_slug("cloud")

        if provider_options.get("ollama"):
            if mode == "env" and env_local:
                pick_local = env_local
            elif mode == "last" and last_local:
                pick_local = last_local
            else:  # auto or fallback
                pick_local = env_local or last_local or first_slug("ollama")

        return {"cloud": pick_cloud, "ollama": pick_local}

    # ------------------------- Draw Things helpers -------------------------
    def _draw_session_key(self, chat_id: int, message_id: int) -> tuple:
        return (chat_id, message_id)

    def _store_draw_session(self, chat_id: int, message_id: int, payload: Dict[str, Any]) -> None:
        self.draw_sessions[self._draw_session_key(chat_id, message_id)] = payload

    def _get_draw_session(self, chat_id: int, message_id: int) -> Optional[Dict[str, Any]]:
        return self.draw_sessions.get(self._draw_session_key(chat_id, message_id))

    def _remove_draw_session(self, chat_id: int, message_id: int) -> None:
        self.draw_sessions.pop(self._draw_session_key(chat_id, message_id), None)

    def _format_draw_prompt(self, session: Dict[str, Any]) -> str:
        original = (session.get("original_prompt") or "").strip()
        active = (session.get("active_prompt") or original).strip()
        source = session.get("active_source")
        negative = session.get("negative_prompt")
        last = session.get("last_generation")

        lines: List[str] = ["🎨 *Draw Things Prompt*"]
        lines.append("")
        lines.append(f"*Base:* {self._escape_markdown(original)}" if original else "*Base:* _(empty)_")
        if active and active != original:
            source_label = "Local" if source == "local" else "Cloud" if source == "cloud" else "Manual"
            lines.append(f"*Enhanced ({source_label}):* {self._escape_markdown(active)}")
        else:
            lines.append(f"*Current:* {self._escape_markdown(active)}" if active else "*Current:* _(empty)_")
        if negative:
            lines.append(f"*Negative:* {self._escape_markdown(str(negative))}")
        if isinstance(last, dict):
            label = last.get("label") or ""
            steps = last.get("steps")
            seed = last.get("seed")
            meta: List[str] = []
            if isinstance(steps, int) and steps > 0:
                meta.append(f"{steps} steps")
            if seed is not None:
                meta.append(f"seed {seed}")
            meta_str = f" ({', '.join(meta)})" if meta else ""
            lines.append(f"Last image: {label}{meta_str}".strip())
        preset_info = session.get("preset_info")
        if preset_info:
            preset_key = session.get("selected_preset") or self._draw_default_preset_key(session)
            preset_label = self._draw_choice_label(session, "preset", preset_key, default="Default")
            lines.append(f"*Preset:* {self._escape_markdown(preset_label)}")
            style_label = self._draw_choice_label(session, "style", session.get("selected_style"))
            negative_label = self._draw_choice_label(session, "negative", session.get("selected_negative"))
            lines.append(f"*Style:* {self._escape_markdown(style_label)}")
            lines.append(f"*Negative:* {self._escape_markdown(negative_label)}")
        banner = session.get("status_banner")
        if banner:
            lines.append("")
            lines.append(self._escape_markdown(str(banner)))
        status_line = session.get("status_message")
        if status_line:
            lines.append(self._escape_markdown(str(status_line)))
        lines.append("")
        if session.get("buttons_disabled"):
            lines.append("Working…")
        else:
            lines.append("Select an action:")
        return "\n".join(lines)

    def _draw_mapping(self, session: Dict[str, Any], kind: str) -> Dict[str, Any]:
        info = session.get("preset_info") or {}
        return ((info.get("maps") or {}).get(kind) or {}) if isinstance(info, dict) else {}

    def _draw_choice_label(self, session: Dict[str, Any], kind: str, key: Optional[str], default: str = "None") -> str:
        if not key:
            if default.lower().startswith("auto"):
                inferred = self._draw_default_preset_key(session, session.get("selected_model_group")) if kind == "preset" else None
                if inferred:
                    inner = self._draw_choice_label(session, kind, inferred, default="Default")
                    return f"Auto ({inner})"
            return default
        mapping = self._draw_mapping(session, kind)
        entry = mapping.get(key)
        if isinstance(entry, dict):
            label = entry.get("label")
            if isinstance(label, str) and label.strip():
                return label
        return _clean_label(key) if key else default

    def _draw_default_preset_key(self, session: Dict[str, Any], group: Optional[str] = None) -> Optional[str]:
        info = session.get("preset_info") or {}
        defaults = info.get("defaults") or {}
        preset_key = defaults.get("preset")
        mapping = self._draw_mapping(session, "preset")
        if preset_key and (not group or (mapping.get(preset_key, {}).get("group") == group)):
            return preset_key
        preferred = self._health_preferred_preset(session, mapping, group=group)
        if preferred:
            return preferred
        presets = info.get("presets") or []
        if presets:
            first = presets[0]
            if isinstance(first, dict):
                if not group or first.get("group") == group:
                    return first.get("key")
        if group:
            for entry in presets:
                if isinstance(entry, dict) and entry.get("group") == group:
                    return entry.get("key")
        if mapping:
            return next(iter(mapping.keys()))
        return None

    def _draw_presets_for_group(self, session: Dict[str, Any], group: Optional[str]) -> List[Dict[str, Any]]:
        mapping = self._draw_mapping(session, "preset")
        if not mapping:
            return []
        result: List[Dict[str, Any]] = []
        for entry in mapping.values():
            if not isinstance(entry, dict):
                continue
            entry_group = entry.get("group") or "general"
            if group and entry_group != group:
                continue
            result.append(entry)
        return result

    def _health_preferred_preset(
        self,
        session: Dict[str, Any],
        mapping: Dict[str, Dict[str, Any]],
        *,
        group: Optional[str] = None,
    ) -> Optional[str]:
        group_name = (group or session.get("selected_model_group") or "general").strip().lower()
        preferences: Dict[str, Tuple[str, ...]] = {
            "flux": ("flux_balanced", "flux_fast", "flux_ultra"),
            "hidream": ("hidream_fast", "hidream_balanced", "hidream_photoreal"),
        }
        for candidate in preferences.get(group_name, ()):
            if mapping.get(candidate):
                return candidate
        return None

    def _draw_fallback_generation(self, session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        mapping = self._draw_mapping(session, "preset")
        group = (session.get("selected_model_group") or "general").strip().lower()
        if group == "flux":
            preset = "flux_ultra" if mapping.get("flux_ultra") else ("flux_fast" if mapping.get("flux_fast") else None)
            return {
                "preset": preset,
                "steps": 4,
                "sampler": "Euler a",
                "cfg_scale": 7.5,
                "width": 512,
                "height": 512,
                "label": "Flux fallback 512²",
                "message": "⚠️ Generation failed; retrying with Flux fallback (4-step, 512²).",
            }
        if group == "hidream":
            preset = "hidream_fast" if mapping.get("hidream_fast") else None
            return {
                "preset": preset,
                "steps": 24,
                "sampler": "Euler a",
                "cfg_scale": 7.5,
                "width": 512,
                "height": 512,
                "label": "HiDream fallback 512²",
                "message": "⚠️ Generation failed; retrying with HiDream fallback (24-step, 512²).",
            }
        return {
            "preset": session.get("selected_preset"),
            "steps": None,
            "sampler": None,
            "cfg_scale": None,
            "width": 512,
            "height": 512,
            "label": "Fallback 512²",
            "message": "⚠️ Generation failed; retrying with simplified settings.",
        }

    def _update_draw_status_banner(self, session: Dict[str, Any]) -> None:
        overrides = session.get("model_options") or []
        model_raw = session.get("selected_model")
        model_label = _friendly_engine_label(model_raw, overrides) if model_raw else "Draw Things"
        banner_bits: List[str] = [f"🗂️ Model: {model_label}"]

        dt_status = session.get("drawthings") or {}
        active_model_raw = dt_status.get("activeModel")
        engine_label = _friendly_engine_label(active_model_raw, overrides)
        if engine_label and engine_label.lower() != model_label.lower():
            banner_bits.append(f"🖥️ Engine: {engine_label}")

        family_label = _draw_family_label(session.get("selected_model_group"))
        banner_bits.append(f"🎛️ Family: {family_label}")

        preset_label = self._draw_choice_label(session, "preset", session.get("selected_preset"), default="Auto")
        banner_bits.append(f"🎚️ Default preset: {preset_label}")

        style_label = self._draw_choice_label(session, "style", session.get("selected_style"))
        banner_bits.append(f"🎨 Style: {style_label}")

        mode_labels = {"local": "Local", "cloud": "Cloud", "none": "Off"}
        enhance_mode = session.get("enhance_mode") or "local"
        banner_bits.append(f"🖌️ Enhance: {mode_labels.get(enhance_mode, enhance_mode.title())}")

        size_key = session.get("selected_size") or "small"
        size_info = self.DRAW_SIZE_PRESETS.get(size_key, {})
        size_label = size_info.get("label") or size_key.title()
        banner_bits.append(f"🖼️ Size: {size_label}")

        seed_mode = session.get("seed_mode") or "auto"
        seed_value = session.get("seed_value")
        if seed_mode == "reuse":
            if isinstance(seed_value, int):
                banner_bits.append(f"🎲 Seed: Reuse ({seed_value})")
            else:
                banner_bits.append("🎲 Seed: Reuse (not captured)")
        else:
            banner_bits.append("🎲 Seed: Auto")

        health = session.get("drawthings_health") or {}
        probe = health.get("probe") or {}
        elapsed = probe.get("elapsedMs")
        if isinstance(elapsed, (int, float)):
            banner_bits.append(f"🧪 {int(elapsed)} ms probe")
        gpu_hint = health.get("gpuLikely")
        if gpu_hint is True:
            banner_bits.append("🖥️ GPU: likely")
        elif gpu_hint is False:
            banner_bits.append("🖥️ GPU: uncertain")

        switching = "Switchable" if session.get("model_switch_enabled", False) else "Fixed"
        banner_bits.append(f"🔁 {switching}")

        session["status_banner"] = " • ".join(banner_bits)

    def _build_draw_picker_keyboard(self, session: Dict[str, Any], target: str) -> InlineKeyboardMarkup:
        rows: List[List[InlineKeyboardButton]] = []
        if target == "preset":
            mapping = self._draw_mapping(session, "preset")
            group = session.get("selected_model_group")
            selected = session.get("selected_preset")
            auto_label_text = "🤖 Auto (model-based)"
            if selected is None:
                auto_label_text = f"✅ {auto_label_text}"
            rows.append([InlineKeyboardButton(auto_label_text, callback_data="draw:preset_auto")])

            group_names = {
                "flux": "Flux presets",
                "hidream": "HiDream presets",
                "general": "General presets",
            }
            preset_info = session.get("preset_info") or {}
            order = (preset_info.get("orders") or {}).get("presets") or []
            show_all = session.get("preset_picker_show_all", False)

            def iter_entries(group_filter: Optional[str]):
                seen: set = set()
                if order:
                    for key in order:
                        entry = mapping.get(key)
                        if not isinstance(entry, dict):
                            continue
                        if group_filter and entry.get("group") != group_filter:
                            continue
                        seen.add(key)
                        yield entry
                for entry in (preset_info.get("presets") or []):
                    key = entry.get("key")
                    if not isinstance(entry, dict):
                        continue
                    if key in seen:
                        continue
                    if group_filter and entry.get("group") != group_filter:
                        continue
                    seen.add(key)
                    yield entry

            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for entry in iter_entries(None):
                if not isinstance(entry, dict):
                    continue
                entry_group = entry.get("group") or "general"
                grouped.setdefault(entry_group, []).append(entry)

            families_in_order: List[str] = []
            if group and group in grouped:
                families_in_order.append(group)
            if show_all or not group or group not in grouped:
                for fam in grouped.keys():
                    if fam not in families_in_order:
                        families_in_order.append(fam)

            has_any = False
            for fam in families_in_order:
                entries = grouped.get(fam) or []
                header = group_names.get(fam, f"{fam.title()} presets")
                if fam == group:
                    header = f"{header} • Active"
                rows.append([InlineKeyboardButton(header, callback_data="draw:nop")])
                for entry in entries:
                    label = _preset_with_hint(entry)
                    if entry.get("key") == selected:
                        label = f"✅ {label}"
                    rows.append([InlineKeyboardButton(label, callback_data=f"draw:preset_select:{entry.get('key')}")])
                    has_any = True

            if not has_any:
                rows.append([InlineKeyboardButton("No presets available", callback_data="draw:nop")])
            else:
                families_total = len(grouped)
                if families_total > 1:
                    if show_all:
                        rows.append([InlineKeyboardButton("Hide other presets", callback_data="draw:preset_less")])
                    else:
                        rows.append([InlineKeyboardButton("More presets…", callback_data="draw:preset_more")])
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data="draw:picker_back")])
        elif target == "model":
            if not session.get("model_switch_enabled", False):
                label = session.get("selected_model") or "Draw Things"
                rows.append([InlineKeyboardButton(f"🗂️ Active • {label}", callback_data="draw:nop")])
                rows.append([InlineKeyboardButton("⬅️ Back", callback_data="draw:picker_back")])
                return InlineKeyboardMarkup(rows)
            options = session.get("model_options") or []
            selected_name = session.get("selected_model")
            if not options:
                rows.append([InlineKeyboardButton("No models configured", callback_data="draw:nop")])
            else:
                for idx, option in enumerate(options):
                    name = option.get("name") or "Current"
                    label = name
                    if name == selected_name:
                        label = f"✅ {label}"
                    rows.append([InlineKeyboardButton(label, callback_data=f"draw:model_select:{idx}")])
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data="draw:picker_back")])
        elif target == "style":
            mapping = self._draw_mapping(session, "style")
            selected = session.get("selected_style")
            order = ((session.get("preset_info") or {}).get("orders") or {}).get("stylePresets") or []
            ordered_keys: List[str] = []
            for key in order:
                if key in mapping:
                    ordered_keys.append(key)
            for key in mapping.keys():
                if key not in ordered_keys:
                    ordered_keys.append(key)
            for key in ordered_keys:
                entry = mapping.get(key)
                if not isinstance(entry, dict):
                    continue
                base_label = entry.get("label") or _clean_label(key)
                desc = entry.get("desc")
                display = f"{base_label} — {desc}" if desc else base_label
                if key == selected:
                    display = f"✅ {display}"
                rows.append([InlineKeyboardButton(display, callback_data=f"draw:style_select:{key}")])
            rows.append([InlineKeyboardButton("🚫 None", callback_data="draw:style_clear")])
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data="draw:picker_back")])
        elif target == "negative":
            mapping = self._draw_mapping(session, "negative")
            selected = session.get("selected_negative")
            order = ((session.get("preset_info") or {}).get("orders") or {}).get("negativePresets") or []
            ordered_keys: List[str] = []
            for key in order:
                if key in mapping:
                    ordered_keys.append(key)
            for key in mapping.keys():
                if key not in ordered_keys:
                    ordered_keys.append(key)
            for key in ordered_keys:
                entry = mapping.get(key)
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label") or _clean_label(key)
                if key == selected:
                    label = f"✅ {label}"
                rows.append([InlineKeyboardButton(label, callback_data=f"draw:negative_select:{key}")])
            rows.append([InlineKeyboardButton("🚫 None", callback_data="draw:negative_clear")])
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data="draw:picker_back")])
        else:
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data="draw:picker_back")])
        return InlineKeyboardMarkup(rows)

    def _build_draw_keyboard(self, session: Dict[str, Any]) -> InlineKeyboardMarkup:
        picker = session.get("picker")
        if picker:
            return self._build_draw_picker_keyboard(session, picker)

        if session.get("buttons_disabled"):
            label = session.get("status_button_label") or "⏳ Working…"
            rows = [
                [InlineKeyboardButton(label, callback_data="draw:nop")],
                [InlineKeyboardButton("❌ Cancel", callback_data="draw:cancel")],
            ]
            return InlineKeyboardMarkup(rows)

        if session.get("enhance_mode") not in ("local", "cloud", "none"):
            session["enhance_mode"] = "local"
        if session.get("selected_size") not in self.DRAW_SIZE_PRESETS:
            session["selected_size"] = "small"
        if session.get("seed_mode") not in ("auto", "reuse"):
            session["seed_mode"] = "auto"

        rows: List[List[InlineKeyboardButton]] = []
        model_options = session.get("model_options") or []
        selected_model = session.get("selected_model")
        if session.get("model_switch_enabled", False) and model_options:
            model_label = selected_model or "Current"
            rows.append([InlineKeyboardButton(f"🗂️ Model • {model_label}", callback_data="draw:model")])
        elif selected_model:
            rows.append([InlineKeyboardButton(f"🗂️ Model • {selected_model} (set in Draw Things)", callback_data="draw:nop")])

        enhance_mode = session.get("enhance_mode") or "local"
        rows.append([
            InlineKeyboardButton(("✅ " if enhance_mode == "local" else "") + "✨ Local", callback_data="draw:mode_local"),
            InlineKeyboardButton(("✅ " if enhance_mode == "cloud" else "") + "🌐 Cloud", callback_data="draw:mode_cloud"),
            InlineKeyboardButton(("✅ " if enhance_mode == "none" else "") + "🚫 No Enhance", callback_data="draw:mode_none"),
        ])

        group = session.get("selected_model_group")
        mapping = self._draw_mapping(session, "preset")
        current_preset = session.get("selected_preset")
        if current_preset and group and mapping.get(current_preset, {}).get("group") != group:
            session["selected_preset"] = None

        preset_label = self._draw_choice_label(session, "preset", session.get("selected_preset"), default="Auto")
        style_label = self._draw_choice_label(session, "style", session.get("selected_style"))
        negative_label = self._draw_choice_label(session, "negative", session.get("selected_negative"))

        rows.append([
            InlineKeyboardButton(f"🎛️ Preset • {preset_label}", callback_data="draw:preset"),
            InlineKeyboardButton(f"🎨 Style • {style_label}", callback_data="draw:style"),
            InlineKeyboardButton(f"🚫 Negative • {negative_label}", callback_data="draw:negative"),
        ])

        available_presets = self._draw_presets_for_group(session, group)
        logging.info(
            "draw: keyboard model=%s group=%s presets=%s",
            selected_model,
            group,
            [entry.get("key") for entry in available_presets] if isinstance(available_presets, list) else available_presets,
        )

        size_key = session.get("selected_size") or "small"
        seed_mode = session.get("seed_mode") or "auto"
        seed_value = session.get("seed_value")

        seed_auto_label = "🎲 Seed • Auto"
        if seed_mode == "auto":
            seed_auto_label = f"✅ {seed_auto_label}"
        seed_reuse_label = "♻️ Seed • Reuse"
        if isinstance(seed_value, int):
            seed_reuse_label = f"{seed_reuse_label} ({seed_value})"
        else:
            seed_reuse_label = f"{seed_reuse_label} (after generate)"
        if seed_mode == "reuse":
            seed_reuse_label = f"✅ {seed_reuse_label}"
        rows.append([
            InlineKeyboardButton(seed_auto_label, callback_data="draw:seed_auto"),
            InlineKeyboardButton(seed_reuse_label, callback_data="draw:seed_reuse"),
        ])

        size_row: List[InlineKeyboardButton] = []
        for key in ("small", "medium", "large"):
            size_info = self.DRAW_SIZE_PRESETS.get(key) or {}
            label = size_info.get("label") or key.title()
            if key == size_key:
                label = f"✅ {label}"
            size_row.append(InlineKeyboardButton(label, callback_data=f"draw:size_{key}"))
        rows.append(size_row)

        rows.append([InlineKeyboardButton("🎨 Generate Image", callback_data="draw:generate")])
        rows.append([InlineKeyboardButton("❌ Close", callback_data="draw:cancel")])
        return InlineKeyboardMarkup(rows)

    async def draw_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_user_allowed(user_id):
            message = update.effective_message
            if message:
                await message.reply_text("❌ You are not authorized to use this bot.")
            return

        message = update.effective_message
        if not message:
            return

        raw_text = message.text or ""
        parts = raw_text.split(" ", 1)
        prompt = parts[1].strip() if len(parts) > 1 else ""
        if not prompt:
            await message.reply_text(
                "🎨 Usage: /draw <prompt>\nTry `/draw cozy living room with sunlight`.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        logging.info("draw: command from %s prompt=%r", user_id, prompt[:160])

        client = self._resolve_tts_client()
        base = None
        if client and client.base_api_url:
            self.tts_client = client
            base = client.base_api_url
        else:
            base = (os.getenv("TTSHUB_API_BASE") or "").strip()

        logging.info("draw: resolving presets base=%s", base or "<empty>")

        preset_info = None
        dt_info = {}
        if base:
            try:
                preset_info = await draw_service.fetch_presets(base, force_refresh=True)
                dt_info = (preset_info.get("drawthings") or {}) if preset_info else {}
                logging.info("draw: fetched drawthings info %s", dt_info)
            except Exception as exc:
                logging.warning("draw: preset fetch failed: %s", exc)
                preset_info = None
                dt_info = {}
            if not dt_info or not (dt_info.get("activeModel") or dt_info.get("activeFamily")):
                try:
                    await asyncio.sleep(0.5)
                    preset_info = await draw_service.fetch_presets(base, force_refresh=True)
                    dt_info = (preset_info.get("drawthings") or {}) if preset_info else {}
                    logging.info("draw: retry fetched drawthings info %s", dt_info)
                except Exception as exc:
                    logging.warning("draw: preset retry failed: %s", exc)

        model_options = self.draw_models_overrides
        default_model = model_options[0] if model_options else {"name": None, "group": None}

        hub_supports_switch = dt_info.get("supportsModelSwitch")
        model_switch_enabled = self.draw_model_switch_enabled and bool(hub_supports_switch)
        if hub_supports_switch is False:
            model_switch_enabled = False

        active_model_raw = dt_info.get("activeModel")
        active_model_name = _friendly_model_name(active_model_raw, model_options)
        active_family = dt_info.get("activeFamily") or _family_from_name(active_model_raw)
        if not active_family and active_model_raw and "flux" in active_model_raw.lower():
            active_family = "flux"
        if not active_model_name and default_model.get("name"):
            active_model_name = default_model.get("name")
        if not active_family:
            active_family = default_model.get("group") or "general"

        preset_maps = (preset_info.get("maps") or {}) if isinstance(preset_info, dict) else {}
        style_map = preset_maps.get("style") or {}
        orders = (preset_info.get("orders") or {}) if isinstance(preset_info, dict) else {}
        default_style = _default_style_key(style_map, orders, active_family)

        session = {
            "original_prompt": prompt,
            "active_prompt": prompt,
            "active_source": "base",
            "negative_prompt": None,
            "status_message": None,
            "buttons_disabled": False,
            "status_button_label": None,
            "picker": None,
            "preset_info": preset_info,
            "selected_preset": None,
            "selected_style": default_style,
            "selected_negative": None,
            "tts_base": base,
            "model_options": model_options,
            "selected_model": active_model_name,
            "selected_model_group": active_family or (default_model.get("group") or "general"),
            "model_switch_enabled": model_switch_enabled,
            "drawthings": dt_info,
            "preset_picker_show_all": False,
            "enhance_mode": "local",
            "selected_size": "small",
            "drawthings_health": {},
            "seed_mode": "auto",
            "seed_value": None,
        }

        logging.info(
            "draw: session init model=%s group=%s switch=%s (hub model=%s family=%s)",
            session["selected_model"],
            session["selected_model_group"],
            session["model_switch_enabled"],
            active_model_raw,
            active_family,
        )

        health_warning = None
        if base:
            try:
                health_info = await draw_service.fetch_drawthings_health(base, force_refresh=True)
                session["drawthings_health"] = health_info or {}
            except Exception as exc:
                logging.warning("draw: health fetch failed: %s", exc)
                # retain any cached info so we can at least show stale data
                session.setdefault("drawthings_health", {})
            health_warning = _draw_health_warning(session.get("drawthings_health") or {})
        else:
            session["drawthings_health"] = {}

        self._update_draw_status_banner(session)
        session["status_message"] = health_warning

        text = self._format_draw_prompt(session)
        keyboard = self._build_draw_keyboard(session)
        prompt_message = await message.reply_text(
            text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard,
        )
        self._store_draw_session(prompt_message.chat_id, prompt_message.message_id, session)

    async def _refresh_draw_prompt(self, query, session: Dict[str, Any]) -> None:
        text = self._format_draw_prompt(session)
        try:
            await query.edit_message_text(
                text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=self._build_draw_keyboard(session),
            )
        except Exception as exc:
            logging.debug("draw: failed to refresh prompt: %s", exc)

    async def _draw_execute_generation(
        self,
        query,
        session: Dict[str, Any],
        *,
        size_label: str,
        width: Optional[int],
        height: Optional[int],
    ) -> None:
        prompt_text = session.get("active_prompt") or session.get("original_prompt") or ""
        if not prompt_text:
            try:
                await query.answer("Prompt is empty.", show_alert=True)
            except Exception:
                pass
            return

        session["status_message"] = f"🖼️ Generating {size_label}…"
        session["status_button_label"] = "🖼️ Generating…"
        session["buttons_disabled"] = True
        session["picker"] = None
        self._store_draw_session(query.message.chat.id, query.message.message_id, session)
        await self._refresh_draw_prompt(query, session)

        base = session.get("tts_base")
        if not base:
            client = self._resolve_tts_client()
            if client and client.base_api_url:
                self.tts_client = client
                base = client.base_api_url
            else:
                base = (os.getenv("TTSHUB_API_BASE") or "").strip()
            session["tts_base"] = base
        if not base:
            session["status_message"] = "⚠️ TTS hub URL not configured."
            session["buttons_disabled"] = False
            session["status_button_label"] = None
            self._store_draw_session(query.message.chat.id, query.message.message_id, session)
            await self._refresh_draw_prompt(query, session)
            try:
                await query.answer("Set TTSHUB_API_BASE to use Draw Things.", show_alert=True)
            except Exception:
                pass
            return

        if not reach_hub_ok(base):
            session["status_message"] = "⚠️ Hub unreachable. Check the Mac."
            session["buttons_disabled"] = False
            session["status_button_label"] = None
            self._store_draw_session(query.message.chat.id, query.message.message_id, session)
            await self._refresh_draw_prompt(query, session)
            try:
                await query.answer("Draw hub unreachable. Is the Mac online?", show_alert=True)
            except Exception:
                pass
            return

        selected_preset = session.get("selected_preset")
        group = session.get("selected_model_group")
        effective_preset = selected_preset or self._draw_default_preset_key(session, group=group)
        preset_entry = self._draw_mapping(session, "preset").get(effective_preset)
        if not effective_preset or not preset_entry:
            session["status_message"] = "⚠️ No preset found for this model family."
            session["buttons_disabled"] = False
            session["status_button_label"] = None
            self._store_draw_session(query.message.chat.id, query.message.message_id, session)
            await self._refresh_draw_prompt(query, session)
            try:
                await query.answer("Preset unavailable for this family.", show_alert=True)
            except Exception:
                pass
            return
        steps = None
        caption_width = width
        caption_height = height
        if isinstance(preset_entry, dict):
            if caption_width is None:
                caption_width = preset_entry.get("default_width")
            if caption_height is None:
                caption_height = preset_entry.get("default_height")
            preset_steps = preset_entry.get("steps")
            if isinstance(preset_steps, (int, float)):
                steps = int(preset_steps)

        logging.info(
            "draw: generating image preset=%s width=%s height=%s prompt_len=%d",
            effective_preset,
            width,
            height,
            len(prompt_text),
        )
        model_switch_enabled = session.get("model_switch_enabled", False)
        model_name = session.get("selected_model") if model_switch_enabled else None
        current_width = int(width) if width is not None else None
        current_height = int(height) if height is not None else None
        current_steps = steps
        current_sampler: Optional[str] = None
        current_cfg_scale: Optional[float] = None
        current_preset = effective_preset
        current_size_label = size_label
        fallback_used = False
        result: Optional[Dict[str, Any]] = None
        attempt = 0
        seed_mode = session.get("seed_mode") or "auto"
        seed_value = session.get("seed_value")
        current_seed: Optional[int] = None
        if seed_mode == "reuse" and isinstance(seed_value, int):
            current_seed = int(seed_value)
        elif seed_mode == "reuse":
            logging.info("draw: seed reuse requested but no seed available; falling back to auto.")
            session["seed_mode"] = "auto"
            self._update_draw_status_banner(session)

        while attempt < 2:
            attempt += 1
            try:
                result = await draw_service.generate_image(
                    base,
                    prompt_text,
                    width=current_width,
                    height=current_height,
                    steps=current_steps,
                    preset=current_preset,
                    style_preset=session.get("selected_style"),
                    negative_preset=session.get("selected_negative"),
                    model=model_name,
                    sampler=current_sampler,
                    cfg_scale=current_cfg_scale,
                    seed=current_seed,
                )
                break
            except draw_service.DrawGenerationError as exc:
                message = str(exc)
                lowered = message.lower()
                logging.warning("draw: hub generation error (attempt %d, %s): %s", attempt, current_size_label, message)
                offline_tokens = (
                    "sdapi/v1",
                    "service unavailable",
                    "connection refused",
                    "read timed out",
                    "request to hub failed",
                    "client error",
                )
                is_offline = any(token in lowered for token in offline_tokens)
                fallback_plan = None
                if not is_offline and not fallback_used:
                    fallback_plan = self._draw_fallback_generation(session)
                if fallback_plan:
                    fallback_used = True
                    new_width = fallback_plan.get("width")
                    new_height = fallback_plan.get("height")
                    if new_width is not None:
                        current_width = int(new_width)
                        caption_width = current_width
                    if new_height is not None:
                        current_height = int(new_height)
                        caption_height = current_height
                    new_steps = fallback_plan.get("steps")
                    if new_steps is not None:
                        current_steps = int(new_steps)
                    new_sampler = fallback_plan.get("sampler")
                    if new_sampler:
                        current_sampler = new_sampler
                    new_cfg = fallback_plan.get("cfg_scale")
                    if new_cfg is not None:
                        current_cfg_scale = float(new_cfg)
                    new_preset = fallback_plan.get("preset")
                    if new_preset:
                        current_preset = new_preset
                    current_size_label = fallback_plan.get("label") or current_size_label
                    fallback_message = fallback_plan.get("message") or "⚠️ Generation failed; retrying with fallback settings."
                    session["status_message"] = fallback_message
                    session["status_button_label"] = "🖼️ Retrying…"
                    session["buttons_disabled"] = True
                    self._store_draw_session(query.message.chat.id, query.message.message_id, session)
                    await self._refresh_draw_prompt(query, session)
                    continue

                friendly = (
                    "⚠️ Draw Things is not responding (likely offline or still loading the model). "
                    "Open the Draw Things app on the Mac and ensure the target model is ready, then retry."
                    if is_offline
                    else f"⚠️ Generation failed: {message[:160]}"
                )
                session["status_message"] = friendly
                session["buttons_disabled"] = False
                session["status_button_label"] = None
                self._store_draw_session(query.message.chat.id, query.message.message_id, session)
                await self._refresh_draw_prompt(query, session)
                return
            except Exception as exc:
                logging.error("draw: generation failed (%s): %s", current_size_label, exc)
                session["status_message"] = f"⚠️ Generation failed: {str(exc)[:120]}"
                session["buttons_disabled"] = False
                session["status_button_label"] = None
                self._store_draw_session(query.message.chat.id, query.message.message_id, session)
                await self._refresh_draw_prompt(query, session)
                try:
                    await query.answer(f"Generation failed: {str(exc)[:120]}", show_alert=True)
                except Exception:
                    pass
                return

        if result is None:
            session["status_message"] = "⚠️ Generation failed: no result returned."
            session["buttons_disabled"] = False
            session["status_button_label"] = None
            self._store_draw_session(query.message.chat.id, query.message.message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        effective_preset = current_preset
        size_label = current_size_label
        width = current_width if current_width is not None else width
        height = current_height if current_height is not None else height
        steps = current_steps

        image_url = result.get("absolute_url")
        if not image_url:
            session["status_message"] = "⚠️ Hub response missing image URL."
            session["buttons_disabled"] = False
            session["status_button_label"] = None
            self._store_draw_session(query.message.chat.id, query.message.message_id, session)
            await self._refresh_draw_prompt(query, session)
            try:
                await query.answer("Draw Things did not return an image URL.", show_alert=True)
            except Exception:
                pass
            return

        caption_bits = [size_label]
        if caption_width is not None and caption_height is not None:
            caption_bits.append(f"{int(caption_width)}×{int(caption_height)}")
        steps_val = result.get("steps") if steps is None else steps
        if isinstance(steps_val, int) and steps_val > 0:
            caption_bits.append(f"{steps_val} steps")
        raw_seed = result.get("seed")
        seed: Optional[int] = None
        parsed_seed: Optional[int] = None
        if isinstance(raw_seed, (int, float)):
            parsed_seed = int(raw_seed)
        elif isinstance(raw_seed, str):
            try:
                parsed_seed = int(raw_seed.strip())
            except ValueError:
                parsed_seed = None
        if parsed_seed is not None:
            seed = parsed_seed
            caption_bits.append(f"seed {seed}")
        elif raw_seed is not None:
            caption_bits.append(f"seed {raw_seed}")
        else:
            seed = current_seed
            if seed is not None:
                caption_bits.append(f"seed {seed}")
        if selected_preset is None:
            preset_label = self._draw_choice_label(session, "preset", None, default="Auto")
            caption_bits.append(f"Preset: {preset_label}")
        elif effective_preset:
            preset_label = self._draw_choice_label(session, "preset", effective_preset, default=_clean_label(effective_preset))
            caption_bits.append(f"Preset: {preset_label}")
        style_key = session.get("selected_style")
        if style_key:
            caption_bits.append(f"Style: {self._draw_choice_label(session, 'style', style_key)}")
        negative_key = session.get("selected_negative")
        if negative_key:
            caption_bits.append(f"Negative: {self._draw_choice_label(session, 'negative', negative_key)}")
        if model_name:
            caption_bits.append(f"Model: {model_name}")

        caption_header = " • ".join(caption_bits)
        caption_lines = [caption_header, prompt_text]
        if selected_preset is None and effective_preset:
            applied_label = self._draw_choice_label(session, "preset", effective_preset, default=_clean_label(effective_preset))
            caption_lines.append(f"Applied preset: {applied_label}")
        caption = "\n".join(line for line in caption_lines if line)[:1024]

        try:
            photo_payload = None
            if requests is not None:
                try:
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()
                    photo_payload = io.BytesIO(response.content)
                    parsed = urllib.parse.urlparse(image_url)
                    photo_payload.name = Path(parsed.path).name or "draw.png"
                except Exception as download_exc:
                    logging.warning("draw: failed to download image for upload: %s", download_exc)
                    photo_payload = None
            if photo_payload is not None:
                await query.message.reply_photo(photo=photo_payload, caption=caption)
            else:
                await query.message.reply_photo(photo=image_url, caption=caption)
        except Exception as exc:
            logging.error("draw: failed to send image: %s", exc)
            session["status_message"] = f"⚠️ Image ready but sending failed: {str(exc)[:120]}"
            session["buttons_disabled"] = False
            session["status_button_label"] = None
            self._store_draw_session(query.message.chat.id, query.message.message_id, session)
            await self._refresh_draw_prompt(query, session)
            try:
                await query.answer("Image generated but sending failed.", show_alert=True)
            except Exception:
                pass
            return

        if seed is not None:
            session["seed_value"] = seed
        elif isinstance(current_seed, int):
            session["seed_value"] = current_seed

        session["last_generation"] = {
            "label": size_label,
            "steps": steps_val,
            "seed": seed if seed is not None else raw_seed,
            "url": image_url,
            "preset": selected_preset or "auto",
            "applied_preset": effective_preset,
            "model": model_name,
            "fallback_used": fallback_used,
        }
        session["enhance_mode"] = "none"
        if fallback_used:
            session["status_message"] = f"✅ Generated {size_label} (fallback). Enhance reset to Off."
        else:
            session["status_message"] = f"✅ Generated {size_label}. Enhance reset to Off."
        session["buttons_disabled"] = False
        session["status_button_label"] = None
        self._update_draw_status_banner(session)
        logging.info(
            "draw: image generated %s url=%s model=%s (switch=%s) preset=%s applied=%s fallback=%s",
            size_label,
            image_url,
            model_name or "<default>",
            "on" if model_switch_enabled else "off",
            selected_preset or "auto",
            effective_preset or "<auto-inferred>",
            "yes" if fallback_used else "no",
        )
        self._store_draw_session(query.message.chat.id, query.message.message_id, session)
        await self._refresh_draw_prompt(query, session)

    async def _handle_draw_callback(self, query, callback_data: str) -> None:
        chat_id = query.message.chat.id
        message_id = query.message.message_id
        session = self._get_draw_session(chat_id, message_id)
        if not session:
            await query.edit_message_text("⚠️ Draw session expired. Send /draw again.")
            return

        action = callback_data.split(":", 1)[1] if ":" in callback_data else callback_data

        if action == "cancel":
            self._remove_draw_session(chat_id, message_id)
            try:
                await query.edit_message_text("❌ Closed Draw Things session.")
            except Exception:
                pass
            logging.info("draw: session cancelled by user")
            return
        if action == "nop":
            try:
                await query.answer("Working…")
            except Exception:
                pass
            return

        if action.startswith("mode_"):
            mode = action.split("_", 1)[1]
            if mode not in ("local", "cloud", "none"):
                try:
                    await query.answer("Unknown enhancement mode.", show_alert=True)
                except Exception:
                    pass
                return
            session["enhance_mode"] = mode
            self._update_draw_status_banner(session)
            mode_labels = {"local": "Local", "cloud": "Cloud", "none": "Off"}
            session["status_message"] = f"🖌️ Enhance set to {mode_labels.get(mode, mode.title())}."
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action.startswith("size_"):
            size_key = action.split("_", 1)[1]
            size_info = self.DRAW_SIZE_PRESETS.get(size_key)
            if not size_info:
                try:
                    await query.answer("Unknown size.", show_alert=True)
                except Exception:
                    pass
                return
            session["selected_size"] = size_key
            self._update_draw_status_banner(session)
            session["status_message"] = f"🖼️ Size set to {size_info.get('label', size_key.title())}."
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action == "seed_auto":
            session["seed_mode"] = "auto"
            session["status_message"] = "🎲 Seed set to auto (random each run)."
            self._update_draw_status_banner(session)
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action == "seed_reuse":
            seed_value = session.get("seed_value")
            created = False
            if not isinstance(seed_value, int):
                seed_value = random.randint(0, 2**31 - 1)
                session["seed_value"] = seed_value
                created = True
            session["seed_mode"] = "reuse"
            if created:
                session["status_message"] = f"♻️ Seed reuse enabled. Using new seed {seed_value}."
            else:
                session["status_message"] = f"♻️ Reusing seed {seed_value} on next generate."
            self._update_draw_status_banner(session)
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action == "generate":
            base_prompt = session.get("original_prompt") or ""
            if not base_prompt.strip():
                try:
                    await query.answer("Prompt is empty.", show_alert=True)
                except Exception:
                    pass
                return

            enhance_mode = session.get("enhance_mode") or "local"
            family = session.get("selected_model_group")
            style_key = session.get("selected_style")
            style_hint = self._draw_choice_label(session, "style", style_key) if style_key else None

            try:
                if enhance_mode == "local":
                    session["status_message"] = "✨ Enhancing prompt via Local model…"
                    session["status_button_label"] = "✨ Enhancing…"
                    session["buttons_disabled"] = True
                    self._store_draw_session(chat_id, message_id, session)
                    await self._refresh_draw_prompt(query, session)
                    enhanced = await draw_service.enhance_prompt_local(base_prompt, family=family, style_hint=style_hint)
                    session["active_prompt"] = (enhanced or base_prompt).strip() or base_prompt
                    session["active_source"] = "local"
                    session["status_message"] = "✅ Prompt enhanced."
                elif enhance_mode == "cloud":
                    session["status_message"] = "🌐 Enhancing prompt via Cloud model…"
                    session["status_button_label"] = "🌐 Enhancing…"
                    session["buttons_disabled"] = True
                    self._store_draw_session(chat_id, message_id, session)
                    await self._refresh_draw_prompt(query, session)
                    enhanced = await draw_service.enhance_prompt_cloud(base_prompt, family=family, style_hint=style_hint)
                    session["active_prompt"] = (enhanced or base_prompt).strip() or base_prompt
                    session["active_source"] = "cloud"
                    session["status_message"] = "✅ Prompt enhanced."
                else:
                    session["active_prompt"] = session.get("active_prompt") or base_prompt
                    session["active_source"] = "base"
                    session["status_message"] = "🎨 Using current prompt."
            except Exception as exc:
                logging.warning("draw: prompt enhancement failed (%s): %s", enhance_mode, exc)
                session["buttons_disabled"] = False
                session["status_button_label"] = None
                session["status_message"] = f"⚠️ Enhancement failed: {str(exc)[:120]}"
                self._store_draw_session(chat_id, message_id, session)
                await self._refresh_draw_prompt(query, session)
                try:
                    await query.answer(f"Enhancement failed: {str(exc)[:120]}", show_alert=True)
                except Exception:
                    pass
                return

            session["buttons_disabled"] = False
            session["status_button_label"] = None
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)

            size_key = session.get("selected_size") or "small"
            size_info = self.DRAW_SIZE_PRESETS.get(size_key) or {}
            size_label = size_info.get("label") or size_key.title()
            width = size_info.get("width")
            height = size_info.get("height")

            await self._draw_execute_generation(
                query,
                session,
                size_label=size_label,
                width=width,
                height=height,
            )
            return

        if action == "model":
            if not session.get("model_switch_enabled", False):
                try:
                    await query.answer("Switch models directly in Draw Things.", show_alert=True)
                except Exception:
                    pass
                return
            options = session.get("model_options") or []
            if not options:
                try:
                    await query.answer("No models configured.", show_alert=True)
                except Exception:
                    pass
                return
            session["picker"] = "model"
            session["status_message"] = "🗂️ Choose a model."
            session["status_button_label"] = None
            session["buttons_disabled"] = False
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action == "preset":
            if not (session.get("preset_info") and session["preset_info"].get("presets")):
                try:
                    await query.answer("Presets unavailable.", show_alert=True)
                except Exception:
                    pass
                return
            session["picker"] = "preset"
            session.setdefault("preset_picker_show_all", False)
            session["status_message"] = "🎛️ Choose a preset."
            session["status_button_label"] = None
            session["buttons_disabled"] = False
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action == "preset_more":
            session["preset_picker_show_all"] = True
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action == "preset_less":
            session["preset_picker_show_all"] = False
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action == "style":
            if not (session.get("preset_info") and session["preset_info"].get("style_presets")):
                try:
                    await query.answer("No style presets available.", show_alert=True)
                except Exception:
                    pass
                return
            session["picker"] = "style"
            session["status_message"] = "🎨 Choose a style preset. Style presets layer on top of your prompt."
            session["status_button_label"] = None
            session["buttons_disabled"] = False
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action == "negative":
            if not (session.get("preset_info") and session["preset_info"].get("negative_presets")):
                try:
                    await query.answer("No negative presets available.", show_alert=True)
                except Exception:
                    pass
                return
            session["picker"] = "negative"
            session["status_message"] = "🚫 Choose a negative preset."
            session["status_button_label"] = None
            session["buttons_disabled"] = False
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action == "picker_back":
            session["picker"] = None
            session["status_message"] = None
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            return

        if action == "style_clear":
            session["selected_style"] = None
            session["picker"] = None
            self._update_draw_status_banner(session)
            session["status_message"] = "🎨 Style cleared."
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            logging.info("draw: style preset cleared")
            return

        if action == "negative_clear":
            session["selected_negative"] = None
            session["picker"] = None
            session["status_message"] = "🚫 Negative preset cleared."
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            logging.info("draw: negative preset cleared")
            return

        if action.startswith("model_select:"):
            try:
                idx = int(action.split(":", 1)[1])
            except Exception:
                idx = -1
            options = session.get("model_options") or []
            if not (0 <= idx < len(options)):
                try:
                    await query.answer("Unknown model.", show_alert=True)
                except Exception:
                    pass
                return
            choice = options[idx]
            session["selected_model"] = choice.get("name")
            session["selected_model_group"] = choice.get("group") or "general"
            session["picker"] = None
            session["selected_preset"] = None
            style_map = self._draw_mapping(session, "style")
            orders = ((session.get("preset_info") or {}).get("orders") or {}) if isinstance(session.get("preset_info"), dict) else {}
            session["selected_style"] = _default_style_key(style_map, orders, session.get("selected_model_group"))
            session["selected_negative"] = None
            session["preset_picker_show_all"] = False
            self._update_draw_status_banner(session)
            model_label = session.get("selected_model") or "current"
            session["status_message"] = f"🗂️ Model changed to {model_label}."
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            logging.info("draw: model selected %s group=%s", session["selected_model"], session["selected_model_group"])
            return

        if action == "preset_auto":
            session["selected_preset"] = None
            session["picker"] = None
            session["preset_picker_show_all"] = False
            label = self._draw_choice_label(session, "preset", None, default="Auto")
            self._update_draw_status_banner(session)
            session["status_message"] = f"🎛️ Preset set to {label}."
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            logging.info("draw: preset set to auto")
            return

        if action.startswith("preset_select:"):
            key = action.split(":", 1)[1]
            session["selected_preset"] = key
            session["picker"] = None
            label = self._draw_choice_label(session, "preset", key, default=_clean_label(key))
            self._update_draw_status_banner(session)
            session["status_message"] = f"🎛️ Preset set to {label}."
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            logging.info("draw: preset selected %s", key)
            return

        if action.startswith("style_select:"):
            key = action.split(":", 1)[1]
            session["selected_style"] = key
            session["picker"] = None
            label = self._draw_choice_label(session, "style", key, default=_clean_label(key))
            self._update_draw_status_banner(session)
            session["status_message"] = f"🎨 Style preset set to {label}."
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            logging.info("draw: style preset selected %s", key)
            return

        if action.startswith("negative_select:"):
            key = action.split(":", 1)[1]
            session["selected_negative"] = key
            session["picker"] = None
            label = self._draw_choice_label(session, "negative", key, default=_clean_label(key))
            session["status_message"] = f"🚫 Negative preset set to {label}."
            self._store_draw_session(chat_id, message_id, session)
            await self._refresh_draw_prompt(query, session)
            logging.info("draw: negative preset selected %s", key)
            return

        try:
            await query.answer("Unknown draw action.", show_alert=True)
        except Exception:
            pass

    # ------------------------- TTS quick pick memory -------------------------
    def _remember_last_tts_voice(self, user_id: int, alias_slug: Optional[str]) -> None:
        if not alias_slug:
            return
        uid = str(user_id)
        prefs = self.user_prefs.get(uid) or {}
        # Maintain recent list (most-recent first, unique, capped)
        recent: List[str] = list(prefs.get("recent_tts_favorites") or [])
        alias_slug = str(alias_slug).strip()
        recent = [s for s in recent if s != alias_slug]
        recent.insert(0, alias_slug)
        prefs["recent_tts_favorites"] = recent[:5]
        # Keep single last for compatibility
        prefs["last_tts_favorite"] = alias_slug
        self.user_prefs[uid] = prefs
        self._save_user_prefs()

    def _last_tts_voice(self, user_id: int) -> Optional[str]:
        prefs = self.user_prefs.get(str(user_id)) or {}
        slug = (prefs.get("last_tts_favorite") or "").strip()
        return slug or None

    def _last_tts_voices(self, user_id: int, n: int = 2) -> List[str]:
        prefs = self.user_prefs.get(str(user_id)) or {}
        recent = prefs.get("recent_tts_favorites") or []
        # Fallback to single last if list empty
        if not recent:
            last = (prefs.get("last_tts_favorite") or "").strip()
            return [last] if last else []
        return [str(s).strip() for s in recent if str(s).strip()][:n]

    def _build_provider_with_quick_keyboard(
        self,
        cloud_label: str,
        local_label: Optional[str],
        quick_cloud_slug: Optional[str],
        quick_local_slug: Optional[str],
    ) -> InlineKeyboardMarkup:
        rows: List[List[InlineKeyboardButton]] = []
        if quick_cloud_slug:
            rows.append([
                InlineKeyboardButton(
                    f"API • {self._short_label(self._short_model_name(quick_cloud_slug), 24)}",
                    callback_data=f"summary_quick:cloud:{quick_cloud_slug}",
                )
            ])
        if quick_local_slug:
            rows.append([
                InlineKeyboardButton(
                    f"Local • {self._short_label(self._short_model_name(quick_local_slug), 24)}",
                    callback_data=f"summary_quick:ollama:{quick_local_slug}",
                )
            ])
        rows.append([InlineKeyboardButton(cloud_label, callback_data="summary_provider:cloud")])
        if local_label:
            rows.append([InlineKeyboardButton(local_label, callback_data="summary_provider:ollama")])
        rows.append([InlineKeyboardButton("⬅️ Back", callback_data="summarize_back_to_main")])
        return InlineKeyboardMarkup(rows)

    def _build_provider_with_combos_keyboard(
        self,
        cloud_label: str,
        local_label: Optional[str],
        quick_cloud_slug: Optional[str],
        quick_local_slug: Optional[str],
    ) -> InlineKeyboardMarkup:
        rows: List[List[InlineKeyboardButton]] = []
        # Determine availability of combos from env
        cloud_model = get_quick_cloud_env_model()
        local_model = (os.getenv("QUICK_LOCAL_MODEL") or "").strip()
        cloud_voice = (os.getenv("TTS_CLOUD_VOICE") or "fable").strip()
        fav_env = (os.getenv("TTS_QUICK_FAVORITE") or "").strip()
        has_local_voice = any(s.strip() for s in fav_env.split(",")) if fav_env else False

        combo_buttons: List[InlineKeyboardButton] = []
        if local_model and has_local_voice and local_label:
            combo_buttons.append(InlineKeyboardButton("Local Combo", callback_data="summary_combo:local"))
        if cloud_model and cloud_voice:
            combo_buttons.append(InlineKeyboardButton("Cloud Combo", callback_data="summary_combo:cloud"))
        if combo_buttons:
            # Put both on one row when available
            rows.append(combo_buttons)

        # Quick picks (same as non-combo keyboard)
        if quick_cloud_slug:
            rows.append([
                InlineKeyboardButton(
                    f"API • {self._short_label(self._short_model_name(quick_cloud_slug), 24)}",
                    callback_data=f"summary_quick:cloud:{quick_cloud_slug}",
                )
            ])
        if quick_local_slug:
            rows.append([
                InlineKeyboardButton(
                    f"Local • {self._short_label(self._short_model_name(quick_local_slug), 24)}",
                    callback_data=f"summary_quick:ollama:{quick_local_slug}",
                )
            ])

        # Provider choices (labeled as “Other …” to distinguish from combos)
        rows.append([InlineKeyboardButton("Other LLM from Cloud", callback_data="summary_provider:cloud")])
        if local_label:
            rows.append([InlineKeyboardButton("Other LLM (Local)", callback_data="summary_provider:ollama")])
        rows.append([InlineKeyboardButton("⬅️ Back", callback_data="summarize_back_to_main")])
        return InlineKeyboardMarkup(rows)

    async def _start_tts_preselect_flow(self, query, summary_session: Dict[str, Any], provider_key: str, model_option: Dict[str, Any]) -> None:
        """Prompt for TTS selection first, then run summary with preselected TTS.

        Stores a preselect-only TTS session with a pending summary payload. After the
        user selects provider/voice, the TTS session handler will kick off the summary.
        """
        title = "Audio Summary"
        anchor = (query.message.chat.id, query.message.message_id)
        pending = {
            'provider_key': provider_key,
            'model_option': model_option,
            'session': summary_session,
            'origin': anchor,
        }
        tts_payload = {
            'mode': 'summary_audio',
            'summary_type': summary_session.get('summary_type') or 'audio',
            'title': title,
            'video_info': (self.current_item or {}),
            'preselect_only': True,
            'pending_summary': pending,
        }
        await self._prompt_tts_provider(query, tts_payload, title)

    def _friendly_llm_provider(self, provider: Optional[str]) -> str:
        mapping = {
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "openrouter": "OpenRouter",
            "ollama": "Ollama",
        }
        if not provider:
            return "LLM"
        return mapping.get(provider.lower(), provider.title())

    @staticmethod
    def _short_label(text: str, limit: int = 48) -> str:
        if not isinstance(text, str):
            return ""
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def _cloud_model_options(self, base_provider: Optional[str], base_model: Optional[str]) -> List[Dict[str, Any]]:
        options: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        def add_option(provider: Optional[str], model: Optional[str]) -> None:
            try:
                resolved_provider, resolved_model, api_key = llm_config.get_model_config(provider, model)
            except ValueError:
                return
            if resolved_provider == "ollama":
                return
            if resolved_provider in ("openai", "anthropic", "openrouter") and not api_key:
                return
            key = self._summarizer_cache_key(resolved_provider, resolved_model)
            if key in seen:
                return
            seen.add(key)
            provider_name = self._friendly_llm_provider(resolved_provider)
            # Prefer an abbreviated model label in buttons (e.g., show just `gpt-5-mini` for `openai/gpt-5-mini`)
            short_model = None
            if isinstance(resolved_model, str) and "/" in resolved_model:
                short_model = resolved_model.split("/", 1)[1]
            elif isinstance(resolved_model, str):
                short_model = resolved_model

            label_model = short_model or resolved_model
            label_core = provider_name if not label_model else f"{provider_name} • {label_model}"
            display_provider = self._friendly_llm_provider(resolved_provider)
            button_text = f"{display_provider}"
            if label_model:
                button_text = f"{button_text} • {self._short_label(label_model, 24)}"
            options.append(
                {
                    "provider": resolved_provider,
                    "model": resolved_model,
                    "label": label_core,
                    "button_label": button_text,
                }
            )

        add_option(base_provider, base_model)

        explicit_model = getattr(llm_config, "llm_model", None)
        if explicit_model:
            try:
                resolved_provider, resolved_model, _ = llm_config.get_model_config(None, explicit_model)
                add_option(resolved_provider, resolved_model)
            except ValueError:
                pass

        shortlist_name = getattr(llm_config, "llm_shortlist", None)
        shortlist = llm_config.SHORTLISTS.get(shortlist_name, {}) if shortlist_name else {}
        for provider, model in shortlist.get("primary", []):
            add_option(provider, model)
        for provider, model in shortlist.get("fallback", []):
            add_option(provider, model)

        return options[:6]

    def _ollama_model_options(self) -> List[Dict[str, Any]]:
        models: List[str] = []
        try:
            raw = ollama_get_models()
            models = self._ollama_models_list(raw)
        except Exception as exc:
            logging.debug(f"Ollama model list unavailable: {exc}")

        preferred = os.getenv("OLLAMA_SUMMARY_MODEL")
        default_env = os.getenv("OLLAMA_DEFAULT_MODEL")
        quick_env = os.getenv("QUICK_LOCAL_MODEL")
        cached = self._cached_ollama_summarizer()
        cached_model = getattr(cached, "model", None) if cached else None

        ordered: List[str] = []
        for candidate in (cached_model, preferred, default_env, quick_env):
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        for model in models:
            if model and model not in ordered:
                ordered.append(model)

        options: List[Dict[str, Any]] = []
        for model in ordered[:8]:
            if not model:
                continue
            options.append(
                {
                    "provider": "ollama",
                    "model": model,
                    "label": f"Ollama • {model}",
                    "button_label": f"{self._short_label(model, 32)}",
                }
            )
        return options


    def _summary_provider_options(self) -> Dict[str, Dict[str, Any]]:
        options: Dict[str, Dict[str, Any]] = {}
        base_summarizer = self.summarizer
        if not base_summarizer:
            try:
                base_summarizer = YouTubeSummarizer()
                self.summarizer = base_summarizer
                self._cache_summarizer_instance(base_summarizer)
            except Exception as exc:
                logging.error(f"Failed to initialize default summarizer: {exc}")
                return options
        provider = getattr(base_summarizer, "llm_provider", "")
        model = getattr(base_summarizer, "model", None)

        cloud_models = self._cloud_model_options(provider, model)
        if cloud_models:
            default_option = cloud_models[0]
            default_label = "Cloud"
            button_text = "Cloud"
            options["cloud"] = {
                "label": default_label,
                "button_label": button_text,
                "models": cloud_models,
            }

        local_models = self._ollama_model_options()
        if local_models:
            default_option = local_models[0]
            default_label = "Local"
            button_text = "Local"
            options["ollama"] = {
                "label": default_label,
                "button_label": button_text,
                "models": local_models,
            }
        return options

    def _get_summary_summarizer(self, provider: str, model: Optional[str] = None) -> YouTubeSummarizer:
        provider_key = (provider or "").strip().lower()
        model_key = (model or "").strip()
        if (
            self.summarizer
            and getattr(self.summarizer, "llm_provider", "").lower() == provider_key
            and (not model_key or getattr(self.summarizer, "model", None) == model_key)
        ):
            return self.summarizer
        cache_key = self._summarizer_cache_key(provider_key, model_key)
        cached = self.summarizer_cache.get(cache_key)
        if cached:
            return cached
        kwargs = {"llm_provider": provider_key}
        if model_key:
            kwargs["model"] = model_key
        summarizer = YouTubeSummarizer(**kwargs)
        if not self.summarizer and provider_key != "ollama":
            self.summarizer = summarizer
        self._cache_summarizer_instance(summarizer)
        return summarizer

    async def _handle_summary_provider_callback(self, query, provider_key: str) -> None:
        chat_id = query.message.chat.id
        message_id = query.message.message_id
        session = self._get_summary_session(chat_id, message_id)
        if not session:
            await query.answer("Session expired. Please pick a summary again.", show_alert=True)
            await self._show_main_summary_options(query)
            return

        provider_options: Dict[str, Dict[str, Any]] = session.get("provider_options") or {}
        option = provider_options.get(provider_key)
        if not option:
            self._remove_summary_session(chat_id, message_id)
            await query.answer("That option is no longer available.", show_alert=True)
            await self._show_main_summary_options(query)
            return

        model_options = option.get("models") or []
        if not model_options:
            self._remove_summary_session(chat_id, message_id)
            await query.edit_message_text("❌ No models available for this provider.")
            return

        if len(model_options) > 1:
            session["selected_provider"] = provider_key
            session["model_options"] = model_options
            self._store_summary_session(chat_id, message_id, session)
            provider_label = option.get("label") or provider_key.title()
            per_row = 1 if provider_key == "cloud" else 2
            keyboard = ui_build_summary_model_keyboard(provider_key, model_options, per_row=per_row)
            await query.edit_message_text(
                f"⚙️ Choose a model for {provider_label}",
                reply_markup=keyboard,
            )
            return

        selected_model = model_options[0]
        await self._execute_summary_with_model(query, session, provider_key, selected_model)

    async def _execute_summary_with_model(
        self,
        query,
        session: Dict[str, Any],
        provider_key: str,
        model_option: Dict[str, Any],
    ) -> None:
        chat_id = query.message.chat.id
        message_id = query.message.message_id
        try:
            summarizer = self._get_summary_summarizer(model_option.get("provider"), model_option.get("model"))
            # Persist last-used choice for quick picks
            try:
                user_id = query.from_user.id
                self._remember_last_model(user_id, provider_key, model_option.get("model"))
            except Exception:
                pass
        except Exception as exc:
            logging.error(f"Summary provider init failed: {exc}")
            self._remove_summary_session(chat_id, message_id)
            await query.edit_message_text("❌ Failed to initialize the selected summarizer. Please try again later.")
            return

        summary_type = session.get("summary_type")
        user_name = session.get("user_name") or "Unknown"
        proficiency_level = session.get("proficiency_level")
        provider_label = model_option.get("label")

        self._remove_summary_session(chat_id, message_id)

        if not summary_type:
            await self._show_main_summary_options(query)
            return

        # UX: reflect the chosen LLM/model immediately so the menu "updates"
        try:
            chosen_label = provider_label or f"{model_option.get('provider','').title()}"
            await query.edit_message_text(f"🧠 Using {self._escape_markdown(chosen_label)}… Starting summary…", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass

        await self._process_content_summary(
            query,
            summary_type,
            user_name,
            proficiency_level,
            provider_key=provider_key,
            summarizer=summarizer,
            provider_label=provider_label,
        )

    async def _handle_summary_model_callback(self, query, provider_key: str, index: int) -> None:
        chat_id = query.message.chat.id
        message_id = query.message.message_id
        session = self._get_summary_session(chat_id, message_id)
        if not session:
            await query.answer("Session expired. Please pick a summary again.", show_alert=True)
            await self._show_main_summary_options(query)
            return

        if session.get("selected_provider") != provider_key:
            await query.answer("Please choose a provider first.", show_alert=True)
            return

        model_options = session.get("model_options") or []
        if not (0 <= index < len(model_options)):
            await query.answer("Invalid model selection.", show_alert=True)
            return

        selected_model = model_options[index]
        # Linear flow: run summary first; TTS selection happens after summary
        await self._execute_summary_with_model(query, session, provider_key, selected_model)

    async def _handle_summary_model_back(self, query) -> None:
        chat_id = query.message.chat.id
        message_id = query.message.message_id
        session = self._get_summary_session(chat_id, message_id)
        if not session:
            await self._show_main_summary_options(query)
            return

        provider_options: Dict[str, Dict[str, Any]] = session.get("provider_options") or {}
        if len(provider_options) <= 1:
            self._remove_summary_session(chat_id, message_id)
            await self._show_main_summary_options(query)
            return

        session.pop("selected_provider", None)
        session.pop("model_options", None)
        self._store_summary_session(chat_id, message_id, session)

        cloud_button = (provider_options.get("cloud") or {}).get("button_label") or "Cloud"
        local_button = (provider_options.get("ollama") or {}).get("button_label")
        summary_type = session.get("summary_type") or "comprehensive"
        summary_label = self._friendly_variant_label(summary_type)
        keyboard = ui_build_summary_provider_keyboard(cloud_button, local_label=local_button)
        await query.edit_message_text(
            f"⚙️ Choose summarization engine for {summary_label}",
            reply_markup=keyboard,
        )

    # ------------------------- Ollama chat integration -------------------------
    def _ollama_models_list(self, raw: Dict[str, Any]) -> List[str]:
        models = []
        if isinstance(raw, dict):
            items = raw.get("models") or raw.get("data") or raw.get("items") or []
            for m in items:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model") or m.get("id")
                    if name:
                        models.append(name)
        return models

    def _build_ollama_models_keyboard(self, models: List[str], page: int = 0, page_size: int = 12, session: Optional[Dict[str, Any]] = None) -> InlineKeyboardMarkup:
        allow_same = os.getenv('OLLAMA_AI2AI_ALLOW_SAME', '0').lower() in ('1', 'true', 'yes')
        categories = self._ollama_persona_categories()
        sess = session if session is not None else {}
        kb = ui_build_models_keyboard(
            models=models,
            page=page,
            page_size=page_size,
            session=sess,
            categories=categories,
            persona_parse=self._persona_parse,
            ai2ai_default_models=self._ollama_ai2ai_default_models,
            allow_same_models=allow_same,
        )
        # Insert provider toggle rows (Local vs Cloud) to enable Cloud chat in /o
        try:
            rows = list(kb.inline_keyboard or [])
            mode = sess.get('mode') or ('ai-ai' if (sess.get('ai2ai_model_a') and sess.get('ai2ai_model_b')) else 'ai-human')
            if mode == 'ai-human':
                prov = (sess.get('provider') or 'ollama')
                mark_local = '✅' if prov != 'cloud' else '⬜'
                mark_cloud = '✅' if prov == 'cloud' else '⬜'
                rows.insert(1, [
                    InlineKeyboardButton(f"{mark_local} Local", callback_data="ollama_provider:single:local"),
                    InlineKeyboardButton(f"{mark_cloud} Cloud", callback_data="ollama_provider:single:cloud"),
                ])
            else:
                prov_a = (sess.get('ai2ai_provider_a') or 'ollama')
                prov_b = (sess.get('ai2ai_provider_b') or 'ollama')
                mark_la = '✅' if prov_a != 'cloud' else '⬜'
                mark_ca = '✅' if prov_a == 'cloud' else '⬜'
                mark_lb = '✅' if prov_b != 'cloud' else '⬜'
                mark_cb = '✅' if prov_b == 'cloud' else '⬜'
                rows.insert(1, [
                    InlineKeyboardButton(f"A: {mark_la} Local", callback_data="ollama_provider:A:local"),
                    InlineKeyboardButton(f"A: {mark_ca} Cloud", callback_data="ollama_provider:A:cloud"),
                ])
                rows.insert(2, [
                    InlineKeyboardButton(f"B: {mark_lb} Local", callback_data="ollama_provider:B:local"),
                    InlineKeyboardButton(f"B: {mark_cb} Cloud", callback_data="ollama_provider:B:cloud"),
                ])
            kb = InlineKeyboardMarkup(rows)
        except Exception:
            pass
        return kb

    def _ollama_stream_default(self) -> bool:
        val = os.getenv('OLLAMA_STREAM_DEFAULT', '1').lower()
        return val not in ('0', 'false', 'no')

    def _ollama_persona_categories(self) -> Dict[str, Dict[str, Any]]:
        categories: Dict[str, Dict[str, Any]] = {}

        keys: List[str] = []
        base_key = "OLLAMA_PERSONA"
        if base_key in os.environ:
            keys.append(base_key)
        keys.extend(
            sorted(
                k for k in os.environ.keys()
                if k.startswith("OLLAMA_PERSONA_")
            )
        )

        def _label_from_key(key: str) -> str:
            if key == base_key:
                return "Default"
            suffix = key[len("OLLAMA_PERSONA"):].lstrip("_")
            if not suffix:
                return "Custom"
            human = suffix.replace("_", " ").strip()
            return human.title() if human else "Custom"

        for key in keys:
            raw = os.getenv(key, "")
            if not raw:
                continue
            names = [segment.strip() for segment in raw.split(",") if segment.strip()]
            if not names:
                continue
            categories[key] = {
                "label": _label_from_key(key),
                "names": names,
            }

        if not categories:
            categories["DEFAULT"] = {
                "label": "Classic",
                "names": ["Albert Einstein", "Isaac Newton"],
            }

        return categories

    # ------------------------- Persona helpers (gender/display) -------------------------
    def _persona_parse(self, name: Optional[str]) -> Tuple[str, Optional[str]]:
        """Parse a persona name with optional (M)/(F) suffix.

        Returns (display_name, gender) where gender is 'male'/'female' or None.
        """
        raw = (name or "").strip()
        if not raw:
            return "", None
        m = re.search(r"\s*\(([MF])\)\s*$", raw, flags=re.IGNORECASE)
        if not m:
            return raw, None
        gender = m.group(1).upper()
        display = re.sub(r"\s*\([MFmf]\)\s*$", "", raw).strip()
        return display, ("male" if gender == "M" else "female")

    def _update_persona_session_fields(self, session: Dict[str, Any], slot: str, name: Optional[str]) -> None:
        """Update session with display and gender for a given slot ('a' or 'b' or 'single')."""
        slot = (slot or "").lower()
        display, gender = self._persona_parse(name)
        if slot in ("a", "b"):
            session[f"persona_{slot}_display"] = display or name or ""
            session[f"persona_{slot}_gender"] = gender
        elif slot == "single":
            session["persona_single_display"] = display or name or ""
            session["persona_single_gender"] = gender

    def _ollama_persona_pool(self) -> List[str]:
        categories = self._ollama_persona_categories()
        pool: List[str] = []
        for info in categories.values():
            names = info.get("names") or []
            for name in names:
                if isinstance(name, str) and name.strip():
                    pool.append(name.strip())
        if not pool:
            pool = ["Albert Einstein", "Isaac Newton"]
        return pool

    def _ollama_persona_defaults(self) -> List[str]:
        pool = self._ollama_persona_pool()
        if len(pool) >= 2:
            return pool[:2]
        if len(pool) == 1:
            return [pool[0], "Speaker B"]
        return ["Albert Einstein", "Isaac Newton"]

    def _ollama_persona_random_pair(self) -> Tuple[str, str]:
        pool = self._ollama_persona_pool()
        if len(pool) >= 2:
            return tuple(random.sample(pool, 2))  # type: ignore[return-value]
        if len(pool) == 1:
            return pool[0], pool[0]
        defaults = self._ollama_persona_defaults()
        return defaults[0], defaults[1]

    def _ollama_start_ai2ai_task(self, chat_id: int, coro: Awaitable[Any]) -> bool:
        return ollama_service.start_ai2ai_task(self, chat_id, coro)

    def _ollama_ai2ai_default_models(self, models: List[str], allow_same: bool) -> Tuple[Optional[str], Optional[str]]:
        return ai2ai_handler.default_models(self, models, allow_same)

    def _ollama_single_default_model(self, models: List[str]) -> Optional[str]:
        available = list(models or [])
        if not available:
            return None
        preferred = os.getenv('OLLAMA_DEFAULT_MODEL', '').strip()
        if preferred and preferred in available:
            return preferred
        ai2ai_a, _ = self._ollama_ai2ai_default_models(available, allow_same=True)
        if ai2ai_a:
            return ai2ai_a
        return available[0]

    def _ollama_persona_system_prompt(self, persona: str, intro_target: str, intro_pending: bool) -> str:
        # Strip gender suffix for natural prompts
        display, _ = self._persona_parse(persona or "the assistant")
        persona_clean = display or (persona or "the assistant")
        # Historical, context-bound persona behavior + curiosity
        content = (
            f"You are {persona_clean}, a historical figure. "
            "Stay completely in character, using only the knowledge, language, and worldview available in your lifetime. "
            "You are unaware of events, inventions, or people after your death. "
            "When faced with unfamiliar modern concepts, express natural curiosity or skepticism and ask brief clarifying questions. "
            "Debate respectfully and keep replies concise. Do not break character or mention being an AI."
        )
        if intro_pending:
            if intro_target == "user":
                content += (
                    " This is your first reply to the user. Introduce yourself in character—state who you are, your era, and what principles guide your thinking. "
                    "You have not heard of the user before; be curious about who they are and what world they come from. "
                    "Finish by inviting the user to introduce themselves."
                )
            else:
                content += (
                    " This is your first reply in this exchange. Introduce yourself in character—state who you are, your era, and what principles guide your thinking. "
                    "You have never heard of your opponent before; be curious about who they are and what world they come from. "
                    "Finish by inviting your opponent to explain themselves."
                )
        return content

    def _ollama_single_view_toggle_row(self, view: str) -> List[InlineKeyboardButton]:
        from modules.telegram.ui.keyboards import single_view_toggle_row
        return single_view_toggle_row(view)

    def _ollama_single_persona_categories_rows(
        self,
        session: Dict[str, Any],
        page_size: int,
        categories: Dict[str, Dict[str, Any]],
    ) -> List[List[InlineKeyboardButton]]:
        from modules.telegram.ui.keyboards import single_persona_categories_rows
        return single_persona_categories_rows(session or {}, page_size, categories)

    def _ollama_single_persona_list_rows(
        self,
        session: Dict[str, Any],
        page_size: int,
        categories: Dict[str, Dict[str, Any]],
    ) -> List[List[InlineKeyboardButton]]:
        from modules.telegram.ui.keyboards import single_persona_list_rows
        return single_persona_list_rows(session or {}, page_size, categories, self._persona_parse)

    def _ollama_ai2ai_view_toggle_row(self, slot: str, view: str) -> List[InlineKeyboardButton]:
        from modules.telegram.ui.keyboards import ai2ai_view_toggle_row
        return ai2ai_view_toggle_row(slot, view)

    def _ollama_ai2ai_persona_categories_rows(
        self,
        slot: str,
        session: Dict[str, Any],
        page_size: int,
        categories: Dict[str, Dict[str, Any]],
    ) -> List[List[InlineKeyboardButton]]:
        from modules.telegram.ui.keyboards import ai2ai_persona_categories_rows
        return ai2ai_persona_categories_rows(slot, session or {}, page_size, categories)

    def _ollama_ai2ai_persona_list_rows(
        self,
        slot: str,
        session: Dict[str, Any],
        page_size: int,
        categories: Dict[str, Dict[str, Any]],
    ) -> List[List[InlineKeyboardButton]]:
        from modules.telegram.ui.keyboards import ai2ai_persona_list_rows
        return ai2ai_persona_list_rows(slot, session or {}, page_size, categories, self._persona_parse)

    def _ollama_status_text(self, session: Dict[str, Any]) -> str:
        # Determine mode
        a = session.get('ai2ai_model_a')
        b = session.get('ai2ai_model_b')
        mode_key = session.get('mode') or ('ai-ai' if (a and b) else 'ai-human')
        mode_label = 'AI↔AI' if mode_key == 'ai-ai' else 'Single'
        # Streaming indicator only applies to local (Ollama)
        show_stream = False
        if mode_key == 'ai-human':
            prov_single = (session.get('provider') or 'ollama')
            if prov_single != 'cloud' and self._ollama_stream_default():
                show_stream = True
        else:
            prov_a = (session.get('ai2ai_provider_a') or 'ollama')
            prov_b = (session.get('ai2ai_provider_b') or 'ollama')
            if prov_a != 'cloud' and prov_b != 'cloud' and self._ollama_stream_default():
                show_stream = True
        header = f"🤖 Chat · Mode: {mode_label}"
        if show_stream:
            header += " · Streaming: On"
        parts = [header]
        if mode_key == 'ai-ai':
            # Ensure personas exist for display
            if not (session.get('persona_a') and session.get('persona_b')):
                rand_a, rand_b = self._ollama_persona_random_pair()
                session.setdefault('persona_a', rand_a)
                session.setdefault('persona_b', rand_b)
                self._update_persona_session_fields(session, 'a', session.get('persona_a'))
                self._update_persona_session_fields(session, 'b', session.get('persona_b'))
            defaults = self._ollama_persona_defaults()
            pa = session.get('persona_a') or defaults[0]
            pb = session.get('persona_b') or defaults[1]
            pa_disp, _ = self._persona_parse(pa)
            pb_disp, _ = self._persona_parse(pb)
            categories = self._ollama_persona_categories()
            cat_a = session.get("persona_category_a")
            if not cat_a:
                cat_key_a = session.get("ai2ai_persona_category_a")
                if cat_key_a:
                    cat_a = categories.get(cat_key_a, {}).get("label")
            cat_b = session.get("persona_category_b")
            if not cat_b:
                cat_key_b = session.get("ai2ai_persona_category_b")
                if cat_key_b:
                    cat_b = categories.get(cat_key_b, {}).get("label")
            prov_a = (session.get('ai2ai_provider_a') or 'ollama')
            prov_b = (session.get('ai2ai_provider_b') or 'ollama')
            cloud_opt_a = session.get('ai2ai_cloud_option_a') or {}
            cloud_opt_b = session.get('ai2ai_cloud_option_b') or {}

            def _slot_line(slot: str, local_model: Optional[str], persona_disp: str, prov_key: str, cloud_opt: Dict[str, Any]) -> str:
                if prov_key == 'cloud':
                    cp = (cloud_opt.get('provider') or getattr(llm_config, 'llm_provider', 'openrouter'))
                    cm = (cloud_opt.get('model') or getattr(llm_config, 'llm_model', None) or 'Select…')
                    src = f"Cloud/{self._friendly_llm_provider(cp)}"
                    model_label = cm
                else:
                    src = "Local"
                    model_label = local_model or '—'
                return f"{slot}: {model_label} · {persona_disp} ({src})"

            line_a = _slot_line("A", a, pa_disp, prov_a, cloud_opt_a)
            line_b = _slot_line("B", b, pb_disp, prov_b, cloud_opt_b)
            if cat_a:
                line_a = f"{line_a} ({cat_a})"
            if cat_b:
                line_b = f"{line_b} ({cat_b})"
            parts.append(line_a)
            parts.append(line_b)
            turns = session.get('ai2ai_turns_left')
            if isinstance(turns, int):
                parts.append(f"Turns remaining: {turns}")
            topic = (session.get('topic') or '').strip()
            if topic:
                parts.append(f"Topic: {topic}")
        elif mode_key == 'ai-human':
            prov = (session.get('provider') or 'ollama')
            if prov == 'cloud':
                sel = session.get('cloud_single_option') or {}
                # Use selection if present; otherwise fall back to llm_config defaults and mark as default
                provider = sel.get('provider') or getattr(llm_config, 'llm_provider', None)
                model = sel.get('model') or getattr(llm_config, 'llm_model', None)
                prov_name = self._friendly_llm_provider(provider) if provider else 'Cloud'
                model_name = model or 'Select…'
                default_hint = ''
                if not sel and model:
                    default_hint = ' (default)'
                parts.append(f"Model: {prov_name} • {model_name}{default_hint} (Cloud)")
            else:
                model = session.get('model') or '—'
                parts.append(f"Model: {model} (Local)")
            persona_single = session.get("persona_single")
            if persona_single:
                cat_single = session.get("persona_single_category")
                single_disp, _ = self._persona_parse(persona_single)
                persona_line = f"Persona: {single_disp}"
                if cat_single:
                    persona_line = f"{persona_line} ({cat_single})"
                parts.append(persona_line)
        if mode_key == 'ai-ai':
            if a and b:
                parts.append("Type a prompt to start. Options adjusts turns.")
            else:
                parts.append("Select models A and B below to enable AI↔AI chat.")
        else:
            parts.append("Pick a model to chat. Type a prompt to start.")
        return "\n".join(parts)

    async def ollama_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_user_allowed(user_id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return
        # Preflight: detect hub/offline fast; still allow Cloud chat if hub/down
        base = os.getenv('TTSHUB_API_BASE')
        hub_up = bool(base and reach_hub_ok(base))
        ollama_up = hub_up and reach_hub_ollama_ok(base) if hub_up else False
        try:
            models = []
            if ollama_up:
                raw = ollama_get_models()
                models = self._ollama_models_list(raw)
            raw_text = update.message.text or ""
            prompt = ""
            if raw_text:
                parts = raw_text.split(None, 1)
                if len(parts) > 1:
                    prompt = parts[1].strip()
            if not prompt:
                args = getattr(context, "args", None)
                if args:
                    prompt = " ".join(args).strip()
            # Initialize session with defaults (streaming on)
            sess = {
                "active": False,
                "models": models,
                "page": 0,
                "stream": True if self._ollama_stream_default() else False,
                "history": [],
                "mode": "ai-human",
                "single_view": "models",
            }
            default_model = self._ollama_single_default_model(models)
            if default_model:
                sess["model"] = default_model
                sess["active"] = True
            if not ollama_up:
                # Switch to Cloud provider by default when hub is unavailable
                sess["provider"] = "cloud"
                sess["active"] = False
            self.ollama_sessions[update.effective_chat.id] = sess
            kb = self._build_ollama_models_keyboard(models, 0, session=sess)
            # Render dynamic status above the picker
            text = self._ollama_status_text(sess)
            await update.message.reply_text(text, reply_markup=kb)
            if not hub_up:
                await update.message.reply_text("Hub offline. Switched to Cloud provider. Open Options → Pick Model to start.")
            if prompt and sess.get("model"):
                await self._ollama_handle_user_text(update, sess, prompt)
        except Exception as exc:
            await update.message.reply_text(f"❌ Ollama hub error: {exc}")

    async def ollama_stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        session = self.ollama_sessions.get(chat_id)
        if session and session.get("ai2ai_active"):
            session["ai2ai_cancel"] = True
            session["ai2ai_active"] = False
            self.ollama_sessions[chat_id] = session
            task = session.get("ai2ai_task")
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()
            await update.message.reply_text("🛑 Stopping AI↔AI exchange after the current response…")
        else:
            self.ollama_sessions.pop(chat_id, None)
            await update.message.reply_text("🛑 Closed Ollama chat session.")

    async def ollama_ai2ai_chat_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_user_allowed(user_id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return
        chat_id = update.effective_chat.id
        session = self.ollama_sessions.get(chat_id)
        if not session or session.get("mode") != "ai-ai":
            await update.message.reply_text("⚠️ Switch to AI↔AI mode before using /chat.")
            return
        if not (session.get("ai2ai_model_a") and session.get("ai2ai_model_b")):
            await update.message.reply_text("⚠️ Select models A and B first, then try /chat again.")
            return
        raw_text = update.message.text or ""
        prompt = raw_text.partition(" ")[2].strip()
        if not prompt:
            await update.message.reply_text("ℹ️ Usage: /chat <new topic or instruction>")
            return
        session["topic"] = prompt
        self.ollama_sessions[chat_id] = session
        turn_total = session.get("ai2ai_turns_total")
        turn_idx = (session.get("ai2ai_round") or 0) + 1
        coro = self._ollama_ai2ai_continue(chat_id, turn_number=turn_idx, total_turns=turn_total if isinstance(turn_total, int) and turn_total > 0 else None)
        if not self._ollama_start_ai2ai_task(chat_id, coro):
            await update.message.reply_text("⚠️ AI↔AI already running. Wait for this turn or use /stop.")
            return
        await update.message.reply_text("💬 New AI↔AI turn coming up…")

    async def _ollama_handle_user_text(self, update: Update, session: Dict[str, Any], text: str):
        provider_mode = (session.get('provider') or 'ollama').lower()
        if provider_mode == 'cloud':
            await self._cloud_handle_user_text(update, session, text)
        else:
            await ollama_service.handle_user_text(self, update, session, text)

    async def _cloud_handle_user_text(self, update: Update, session: Dict[str, Any], text: str) -> None:
        chat_id = update.effective_chat.id
        # Resolve provider/model from selection or shortlist
        opt = session.get('cloud_single_option') or {}
        provider = opt.get('provider') or getattr(llm_config, 'llm_provider', None) or 'openrouter'
        model = opt.get('model') or getattr(llm_config, 'llm_model', None)
        if not model:
            shortlist = llm_config.SHORTLISTS.get(getattr(llm_config, 'llm_shortlist', 'openrouter_defaults'), {})
            for p, m in (shortlist.get('primary') or []):
                if p != 'ollama':
                    provider, model = p, m
                    break
        if not model:
            await update.message.reply_text("⚠️ No cloud model available. Configure LLM_SHORTLIST/LLM_MODEL.")
            return
        messages: List[Dict[str, str]] = []
        persona_single = session.get("persona_single")
        if persona_single and session.get("persona_single_custom"):
            intro_pending = bool(session.get("persona_single_intro_pending"))
            messages.append({
                "role": "system",
                "content": self._ollama_persona_system_prompt(persona_single, "user", intro_pending),
            })
        history = list(session.get("history") or [])
        messages.extend(history)
        messages.append({"role": "user", "content": text})
        try:
            from telegram.constants import ChatAction
            app = getattr(self, 'application', None)
            if app and getattr(app, 'bot', None):
                await app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        try:
            resp_text = cloud_service.chat(messages, provider=provider, model=model)
        except Exception as exc:
            await update.message.reply_text(f"❌ Cloud chat error: {str(exc)[:200]}")
            return
        display_text = resp_text
        if session.get('mode') == 'ai-human':
            persona_single_name = session.get("persona_single")
            if persona_single_name:
                display_text = f"{persona_single_name} ({provider}/{model})\n\n{resp_text}"
            else:
                display_text = f"{provider}/{model}\n\n{resp_text}"
        await self._send_long_text_reply(update, display_text)
        trimmed_history = (history + [{"role": "user", "content": text}])
        session["history"] = (trimmed_history + [{"role": "assistant", "content": resp_text}])[-16:]
        if session.get("persona_single_custom"):
            session["persona_single_intro_pending"] = False
        self.ollama_sessions[chat_id] = session

    async def _ollama_stream_chat(
        self,
        update: Update,
        model: str,
        messages: List[Dict[str, str]],
        label: Optional[str] = None,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> str:
        app = getattr(self, "application", None)
        bot = getattr(app, "bot", None)
        if bot is None:
            raise RuntimeError("Telegram bot instance is not available")
        from modules.services.ollama_stream import stream_chat as service_stream_chat

        return await service_stream_chat(
            update,
            bot,
            model,
            messages,
            label=label,
            cancel_checker=cancel_checker,
        )

    async def _ollama_ai2ai_continue(
        self,
        chat_id: int,
        turn_number: Optional[int] = None,
        total_turns: Optional[int] = None,
    ) -> None:
        await ai2ai_handler.continue_exchange(self, chat_id, turn_number=turn_number, total_turns=total_turns)

    async def _ollama_ai2ai_run(self, chat_id: int, turns: int):
        await ai2ai_handler.run(self, chat_id, turns)

    async def _send_long_text_reply(self, update: Update, text: str, parse_mode: Optional[str] = None):
        text = text or ""
        # Keep a safety margin for formatting
        safety = 100
        limit = max(1000, self.MAX_MESSAGE_LENGTH - safety)
        if len(text) <= limit:
            await update.message.reply_text(text, parse_mode=parse_mode)
            return
        chunks = self._split_text_into_chunks(text, limit)
        for i, chunk in enumerate(chunks):
            # No parse mode on continuation to avoid formatting issues across chunks
            await update.message.reply_text(chunk)

    async def _handle_ollama_callback(self, query, callback_data: str):
        chat_id = query.message.chat_id
        session = self.ollama_sessions.get(chat_id) or {}
        # Small helper to re-render options with current state
        async def _render_options():
            mode = session.get("mode") or "ai-human"
            stream = bool(session.get("stream"))
            mark_ai = "✅" if mode == "ai-human" else "⬜"
            mark_ai2ai = "✅" if mode == "ai-ai" else "⬜"
            mark_stream = "✅" if stream else "⬜"
            ai2ai_active = bool(session.get("ai2ai_active"))
            ai2ai_row = [InlineKeyboardButton("▶️ Start AI↔AI", callback_data="ollama_ai2ai:start")] if (mode == "ai-ai" and not ai2ai_active) else []
            if mode == "ai-ai" and ai2ai_active:
                ai2ai_row = [InlineKeyboardButton("⏭️ Continue", callback_data="ollama_ai2ai:continue")]
            # Providers for AI↔AI (for streaming toggle visibility/hints)
            prov_a = (session.get('ai2ai_provider_a') or 'ollama')
            prov_b = (session.get('ai2ai_provider_b') or 'ollama')
            rows = [
                [InlineKeyboardButton(f"{mark_ai} Single", callback_data="ollama_mode:ai-human"), InlineKeyboardButton(f"{mark_ai2ai} AI↔AI", callback_data="ollama_mode:ai-ai")],
                [InlineKeyboardButton(f"{mark_stream} Streaming", callback_data="ollama_toggle:stream")],
            ]
            # Hide streaming toggle for Single+Cloud; for AI↔AI hide if both Cloud, hint if mixed
            try:
                if mode == "ai-human":
                    prov = (session.get('provider') or 'ollama')
                    if prov == 'cloud' and len(rows) >= 2 and any(btn.callback_data == 'ollama_toggle:stream' for btn in rows[1]):
                        rows.pop(1)
                else:
                    if prov_a == 'cloud' and prov_b == 'cloud':
                        if len(rows) >= 2 and any(btn.callback_data == 'ollama_toggle:stream' for btn in rows[1]):
                            rows.pop(1)
                    elif (prov_a == 'cloud') != (prov_b == 'cloud'):
                        rows.insert(2, [InlineKeyboardButton("ℹ️ Streaming applies to Local only", callback_data="ollama_nop")])
            except Exception:
                pass
            if ai2ai_row:
                rows.append(ai2ai_row)
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more:{session.get('page', 0)}")])
            kb = InlineKeyboardMarkup(rows)
            try:
                await query.edit_message_text("⚙️ Chat options", reply_markup=kb)
            except Exception:
                pass
        if callback_data.startswith("ollama_more:"):
            try:
                page = int(callback_data.split(":", 1)[1])
            except Exception:
                page = 0
            session["page"] = max(0, page)
            models = session.get("models") or []
            kb = self._build_ollama_models_keyboard(models, session["page"], session=session)
            text = self._ollama_status_text(session)
            await query.edit_message_text(text, reply_markup=kb)
            self.ollama_sessions[chat_id] = session
            await query.answer("Page updated")
            return
        handled_single = await single_handler.handle_callback(self, query, callback_data, session)
        if handled_single:
            return
        if callback_data.startswith("ollama_set_mode:"):
            which = callback_data.split(":", 1)[1]
            if which == "single":
                session["mode"] = "ai-human"
                session["active"] = bool(session.get("model"))
                for key in (
                    "ai2ai_view_a",
                    "ai2ai_view_b",
                    "ai2ai_persona_category_a",
                    "ai2ai_persona_category_b",
                    "persona_category_a",
                    "persona_category_b",
                    "ai2ai_persona_page_a",
                    "ai2ai_persona_page_b",
                    "ai2ai_persona_cat_page_a",
                    "ai2ai_persona_cat_page_b",
                    "ai2ai_round",
                    "ai2ai_turns_total",
                ):
                    session.pop(key, None)
                session["single_view"] = "models"
            else:
                session["mode"] = "ai-ai"
                session.setdefault("ai2ai_page_a", 0)
                session.setdefault("ai2ai_page_b", 0)
                session.setdefault("ai2ai_view_a", "models")
                session.setdefault("ai2ai_view_b", "models")
                session["active"] = bool(session.get("ai2ai_model_a") and session.get("ai2ai_model_b"))
            logging.info(f"Ollama UI: set mode -> {session['mode']}")
            models = session.get("models") or []
            kb = self._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
            await query.edit_message_text(self._ollama_status_text(session), reply_markup=kb)
            self.ollama_sessions[chat_id] = session
            await query.answer("Mode updated")
            return
        if callback_data.startswith("ollama_model:"):
            model = callback_data.split(":", 1)[1]
            # Always select single model on main picker; AI↔AI uses dedicated flow
            if session.get("model") == model:
                session.pop("model", None)
                session["active"] = False
                logging.info("Ollama UI: cleared single model")
            else:
                session["model"] = model
                session["active"] = True
                logging.info(f"Ollama UI: set single model -> {model}")
            self.ollama_sessions[chat_id] = session
            await query.edit_message_text(self._ollama_status_text(session), reply_markup=self._build_ollama_models_keyboard(session.get("models") or [], session.get("page", 0), session=session))
            await query.answer("Model updated")
            return
        if callback_data == "ollama_options":
            # Scaffold options UI (mode/stream toggles)
            mode = session.get("mode") or "ai-human"
            stream = bool(session.get("stream"))
            mark_ai = "✅" if mode == "ai-human" else "⬜"
            mark_ai2ai = "✅" if mode == "ai-ai" else "⬜"
            mark_stream = "✅" if stream else "⬜"
            # AI↔AI scaffolding controls
            ai2ai_active = bool(session.get("ai2ai_active"))
            prov_a = (session.get('ai2ai_provider_a') or 'ollama')
            prov_b = (session.get('ai2ai_provider_b') or 'ollama')
            cloud_opt_a = session.get('ai2ai_cloud_option_a') or {}
            cloud_opt_b = session.get('ai2ai_cloud_option_b') or {}
            def _slot_label(local_model: Optional[str], prov_key: str, cloud_opt: Dict[str, Any]) -> str:
                if prov_key == 'cloud':
                    cm = cloud_opt.get('model') or 'Select…'
                    cp = cloud_opt.get('provider') or getattr(llm_config, 'llm_provider', 'openrouter')
                    return f"{cm} (Cloud/{self._friendly_llm_provider(cp)})"
                return local_model or '—'
            ai2ai_model_a = _slot_label(session.get("ai2ai_model_a"), prov_a, cloud_opt_a)
            ai2ai_model_b = _slot_label(session.get("ai2ai_model_b"), prov_b, cloud_opt_b)
            ai2ai_row = [InlineKeyboardButton("▶️ Start AI↔AI", callback_data="ollama_ai2ai:start")] if (mode == "ai-ai" and not ai2ai_active) else []
            if mode == "ai-ai" and ai2ai_active:
                ai2ai_row = [InlineKeyboardButton("⏭️ Continue", callback_data="ollama_ai2ai:continue")]
            rows = [
                [InlineKeyboardButton(f"{mark_ai} Single", callback_data="ollama_mode:ai-human"), InlineKeyboardButton(f"{mark_ai2ai} AI↔AI", callback_data="ollama_mode:ai-ai")],
                [InlineKeyboardButton(f"{mark_stream} Streaming", callback_data="ollama_toggle:stream")],
            ]
            # Hide streaming in Single when Cloud; hide for AI↔AI when both Cloud; hint when mixed
            try:
                if mode == "ai-human":
                    prov = (session.get('provider') or 'ollama')
                    if prov == 'cloud' and len(rows) >= 2 and any(btn.callback_data == 'ollama_toggle:stream' for btn in rows[1]):
                        rows.pop(1)
                else:
                    if prov_a == 'cloud' and prov_b == 'cloud':
                        if len(rows) >= 2 and any(btn.callback_data == 'ollama_toggle:stream' for btn in rows[1]):
                            rows.pop(1)
                    elif (prov_a == 'cloud') != (prov_b == 'cloud'):
                        rows.insert(2, [InlineKeyboardButton("ℹ️ Streaming applies to Local only", callback_data="ollama_nop")])
            except Exception:
                pass
            # Provider toggles and cloud pickers
            if mode == "ai-human":
                prov = (session.get('provider') or 'ollama')
                rows.append([
                    InlineKeyboardButton("Provider:", callback_data="ollama_nop"),
                    InlineKeyboardButton(("✅ Local" if prov != 'cloud' else "⬜ Local"), callback_data="ollama_provider:single:local"),
                    InlineKeyboardButton(("✅ Cloud" if prov == 'cloud' else "⬜ Cloud"), callback_data="ollama_provider:single:cloud"),
                ])
                if prov == 'cloud':
                    rows.append([InlineKeyboardButton("Pick Model", callback_data="ollama_cloud_pick:single")])
            else:
                pa = (session.get('ai2ai_provider_a') or 'ollama')
                pb = (session.get('ai2ai_provider_b') or 'ollama')
                rows.append([
                    InlineKeyboardButton("A Provider:", callback_data="ollama_nop"),
                    InlineKeyboardButton(("✅ Local" if pa != 'cloud' else "⬜ Local"), callback_data="ollama_provider:A:local"),
                    InlineKeyboardButton(("✅ Cloud" if pa == 'cloud' else "⬜ Cloud"), callback_data="ollama_provider:A:cloud"),
                ])
                rows.append([
                    InlineKeyboardButton("B Provider:", callback_data="ollama_nop"),
                    InlineKeyboardButton(("✅ Local" if pb != 'cloud' else "⬜ Local"), callback_data="ollama_provider:B:local"),
                    InlineKeyboardButton(("✅ Cloud" if pb == 'cloud' else "⬜ Cloud"), callback_data="ollama_provider:B:cloud"),
                ])
                if pa == 'cloud':
                    rows.append([InlineKeyboardButton("Pick Model A", callback_data="ollama_cloud_pick:A")])
                if pb == 'cloud':
                    rows.append([InlineKeyboardButton("Pick Model B", callback_data="ollama_cloud_pick:B")])
            if mode == "ai-ai":
                rows.append([
                    InlineKeyboardButton(f"A: {ai2ai_model_a}", callback_data="ollama_ai2ai:pick_a"),
                    InlineKeyboardButton(f"B: {ai2ai_model_b}", callback_data="ollama_ai2ai:pick_b"),
                ])
            if ai2ai_row:
                rows.append(ai2ai_row)
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more:{session.get('page', 0)}")])
            kb = InlineKeyboardMarkup(rows)
            await query.edit_message_text("⚙️ Chat options", reply_markup=kb)
            await query.answer("Options")
            return
        if callback_data.startswith("ollama_provider:"):
            parts = callback_data.split(":")
            scope = parts[1]
            target = parts[2]
            # Update provider selection in session
            if scope == 'single':
                session['provider'] = 'cloud' if target == 'cloud' else 'ollama'
            else:
                key = f"ai2ai_provider_{scope.lower()}"
                session[key] = 'cloud' if target == 'cloud' else 'ollama'
            # Compute per-slot cloud options if switching to cloud
            if scope in ('A','B') and target == 'cloud':
                base_provider = getattr(llm_config, 'llm_provider', None)
                base_model = getattr(llm_config, 'llm_model', None)
                opts = self._cloud_model_options(base_provider, base_model)
                if opts:
                    session[f'ai2ai_cloud_options_{scope.lower()}'] = opts
                else:
                    await query.answer("No cloud models available", show_alert=True)
            # Clear selection when switching provider type to avoid stale mismatches
            if scope in ('A','B'):
                if target == 'cloud':
                    session.pop(f'ai2ai_model_{scope.lower()}', None)
                else:
                    session.pop(f'ai2ai_cloud_option_{scope.lower()}', None)
            self.ollama_sessions[chat_id] = session
            # Re-render full keyboard (markup only) with per-slot filtering
            kb = self._build_ollama_models_keyboard(session.get('models') or [], session.get('page', 0), session=session)
            try:
                await query.edit_message_reply_markup(reply_markup=kb)
            except Exception:
                await query.edit_message_text(self._ollama_status_text(session), reply_markup=kb)
            await query.answer("Provider updated")
            return
        if callback_data.startswith("ollama_cloud_pick:"):
            which = callback_data.split(":", 1)[1]
            base_provider = getattr(llm_config, 'llm_provider', None)
            base_model = getattr(llm_config, 'llm_model', None)
            opts = self._cloud_model_options(base_provider, base_model)
            if not opts:
                await query.answer("No cloud models available", show_alert=True)
                return
            # Store per-slot options and refresh
            session[f'ai2ai_cloud_options_{which.lower()}'] = opts
            self.ollama_sessions[chat_id] = session
            kb = self._build_ollama_models_keyboard(session.get('models') or [], session.get('page') or 0, session=session)
            try:
                await query.edit_message_reply_markup(reply_markup=kb)
            except Exception:
                await query.edit_message_text(self._ollama_status_text(session), reply_markup=kb)
            return
        if callback_data.startswith("ollama_cloud_model:"):
            _, which, idx_str = callback_data.split(":", 2)
            try:
                idx = int(idx_str)
            except Exception:
                idx = 0
            opts = session.get('cloud_model_options') or []
            if not (0 <= idx < len(opts)):
                await query.answer("Invalid selection", show_alert=True)
                return
            sel = opts[idx]
            if which == 'single':
                session['provider'] = 'cloud'
                session['cloud_single_option'] = sel
            elif which in ('A', 'B'):
                session[f'ai2ai_provider_{which.lower()}'] = 'cloud'
                session[f'ai2ai_cloud_option_{which.lower()}'] = sel
            self.ollama_sessions[chat_id] = session
            await query.answer("Cloud model set")
            # Update only the markup to reflect provider toggles and selections
            kb = self._build_ollama_models_keyboard(session.get('models') or [], session.get('page', 0), session=session)
            try:
                await query.edit_message_reply_markup(reply_markup=kb)
            except Exception:
                # Fallback to full edit if reply_markup fails
                await query.edit_message_text(self._ollama_status_text(session), reply_markup=kb)
            return
        if callback_data.startswith("ollama_pull:"):
            model = callback_data.split(":", 1)[1]
            await query.answer("Pulling…")
            # Show status message
            try:
                status = await query.message.reply_text(f"⏬ Pulling `{self._escape_markdown(model)}`…", parse_mode=ParseMode.MARKDOWN)
            except Exception:
                status = None
            loop = asyncio.get_running_loop()
            def _pull_call():
                try:
                    # Non-stream pull for now (fast path); streaming can be added next
                    from modules.ollama_client import pull as ollama_pull
                    return {"ok": True, "resp": ollama_pull(model, stream=False)}
                except Exception as e:
                    return {"ok": False, "error": str(e)}
            result = await loop.run_in_executor(None, _pull_call)
            if result.get("ok"):
                try:
                    if status:
                        await status.edit_text(f"✅ Pulled `{self._escape_markdown(model)}`.", parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    pass
                # Auto-retry last user prompt if present
                last_user = session.get("last_user")
                if last_user and session.get("model") == model:
                    await query.message.reply_text("🔁 Retrying your last prompt…")
                    # Reuse session and route to chat handler
                    # Construct a pseudo-update for the same chat
                    try:
                        # Send via normal path
                        class _Dummy:
                            pass
                        d = _Dummy()
                        d.effective_chat = query.message.chat
                        d.message = query.message
                        # Use send_message path instead of fake update to avoid confusion
                        await self._ollama_handle_user_text(type("U", (), {"effective_chat": query.message.chat, "message": query.message})(), session, last_user)
                    except Exception:
                        pass
                else:
                    await query.message.reply_text("✅ Model is ready. Send a message to chat.")
            else:
                msg = result.get("error") or "pull failed"
                try:
                    if status:
                        await status.edit_text(f"❌ Pull failed: {msg[:140]}")
                except Exception:
                    pass
            return
        if callback_data.startswith("ollama_mode:"):
            value = callback_data.split(":", 1)[1]
            if value in ("ai-human", "ai-ai"):
                session["mode"] = value
                self.ollama_sessions[chat_id] = session
                await query.answer(f"Mode: {value}")
                await _render_options()
            return
        if callback_data.startswith("ollama_toggle:"):
            flag = callback_data.split(":", 1)[1]
            if flag == "stream":
                session["stream"] = not bool(session.get("stream"))
                self.ollama_sessions[chat_id] = session
                await query.answer(f"Streaming: {'on' if session['stream'] else 'off'}")
                await _render_options()
            return
        if callback_data == "ollama_cancel":
            self.ollama_sessions.pop(chat_id, None)
            await query.answer("Closed")
            try:
                await query.edit_message_text("❌ Closed Ollama picker")
            except Exception:
                pass
            return
        handled_ai2ai = await ai2ai_handler.handle_callback(self, query, callback_data, session, _render_options)
        if handled_ai2ai:
            return
        if callback_data == "ollama_nop":
            await query.answer("Select an option")
            return
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards."""
        query = update.callback_query
        # Acknowledge quickly to stop Telegram's loading spinner
        try:
            await query.answer()
        except Exception as e:
            logging.debug(f"callback answer() failed: {e}")
        # Cancel any pending auto-process for this message
        try:
            key = (query.message.chat.id, query.message.message_id)
            t = self._auto_tasks.pop(key, None)
            if t and not t.done():
                t.cancel()
        except Exception:
            pass
        
        user_id = query.from_user.id
        user_name = query.from_user.first_name or "Unknown"
        
        if not self._is_user_allowed(user_id):
            await query.edit_message_text("❌ You are not authorized to use this bot.")
            return
        
        callback_data = query.data
        logging.info(f"🔔 Callback received: user={user_id} data={callback_data}")
        
        # Handle summary requests
        if callback_data.startswith("summarize_"):
            raw = callback_data.replace("summarize_", "")  # e.g. "audio-fr" or "audio-fr:beginner"

            # Handle back button
            if raw == "back_to_main":
                self._remove_summary_session(query.message.chat.id, query.message.message_id)
                await self._show_main_summary_options(query)
                return

            parts = raw.split(":", 1)
            summary_type = parts[0]  # "audio-fr" / "audio-es" / "audio"
            proficiency_level = parts[1] if len(parts) == 2 else None

            # If French/Spanish audio without level specified, show level picker
            if summary_type in ("audio-fr", "audio-es") and proficiency_level is None:
                self._remove_summary_session(query.message.chat.id, query.message.message_id)
                await self._show_proficiency_selector(query, summary_type)
                return

            provider_options = self._summary_provider_options()
            if not provider_options:
                await query.edit_message_text("❌ Summarizer not configured. Please try /status for details.")
                return

            # If only one provider is available, skip provider selection
            if len(provider_options) == 1:
                provider_key, option = next(iter(provider_options.items()))
                model_options = option.get("models") or []
                session_payload = {
                    "summary_type": summary_type,
                    "proficiency_level": proficiency_level,
                    "user_name": user_name,
                    "provider_options": provider_options,
                }
                if len(model_options) <= 1:
                    selected_model = model_options[0] if model_options else {}
                    await self._execute_summary_with_model(query, session_payload, provider_key, selected_model)
                else:
                    session_payload["selected_provider"] = provider_key
                    session_payload["model_options"] = model_options
                    self._store_summary_session(query.message.chat.id, query.message.message_id, session_payload)
                    provider_label = option.get("label") or provider_key.title()
                    per_row = 1 if provider_key == "cloud" else 2
                    keyboard = ui_build_summary_model_keyboard(provider_key, model_options, per_row=per_row)
                    await query.edit_message_text(
                        f"⚙️ Choose a model for {provider_label}",
                        reply_markup=keyboard,
                    )
                return

            summary_label = self._friendly_variant_label(summary_type)
            session_payload = {
                "summary_type": summary_type,
                "proficiency_level": proficiency_level,
                "user_name": user_name,
                "provider_options": provider_options,
            }
            self._store_summary_session(query.message.chat.id, query.message.message_id, session_payload)
            cloud_option = provider_options.get("cloud") or next(iter(provider_options.values()))
            cloud_label = cloud_option.get("button_label") or "Cloud"
            local_label = (provider_options.get("ollama") or {}).get("button_label")
            prompt_text = f"⚙️ Choose summarization engine for {summary_label}"
            picks = self._quick_pick_candidates(provider_options, user_id)

            # Auto-select LLM for audio summaries after a short delay: prefer local QUICK_LOCAL_MODEL
            if summary_type.startswith("audio"):
                try:
                    default_local = (os.getenv("QUICK_LOCAL_MODEL") or "").strip()
                    llm_delay = int(os.getenv("LLM_AUTO_DEFAULT_SECONDS", "0") or "0")
                except Exception:
                    default_local = ""
                    llm_delay = 0
                if default_local and llm_delay > 0:
                    try:
                        import asyncio as _asyncio
                        chat_id = query.message.chat.id
                        message_id = query.message.message_id
                        async def _auto_llm_default():
                            try:
                                await _asyncio.sleep(llm_delay)
                            except Exception:
                                return
                            sess = self._get_summary_session(chat_id, message_id)
                            if not isinstance(sess, dict):
                                return
                            # If user already picked a provider/model, skip auto-run
                            if sess.get("selected_provider"):
                                return
                            model_option = {
                                "provider": "ollama",
                                "model": default_local,
                                "label": f"Ollama • {self._short_model_name(default_local)}",
                                "button_label": f"{self._short_label(self._short_model_name(default_local), 24)}",
                            }
                            try:
                                # Update the selection message to indicate summary is starting and clear the keyboard
                                from telegram.constants import ParseMode as _PM
                                await self.application.bot.edit_message_text(
                                    chat_id=chat_id,
                                    message_id=message_id,
                                    text=f"🧠 Starting summary • Ollama • {self._short_model_name(default_local)}",
                                    parse_mode=_PM.MARKDOWN,
                                    reply_markup=None,
                                )
                            except Exception:
                                pass
                            await self._execute_summary_with_model(query, sess or session_payload, "ollama", model_option)
                        _asyncio.create_task(_auto_llm_default())
                    except Exception:
                        pass
            # Fallback: show provider keyboard
            if summary_type.startswith("audio"):
                keyboard = self._build_provider_with_combos_keyboard(
                    cloud_label,
                    local_label,
                    picks.get("cloud"),
                    picks.get("ollama"),
                )
            else:
                keyboard = self._build_provider_with_quick_keyboard(
                    cloud_label,
                    local_label,
                    picks.get("cloud"),
                    picks.get("ollama"),
                )
            await query.edit_message_text(prompt_text, reply_markup=keyboard)
            return

        elif callback_data.startswith("tts_"):
            await self._handle_tts_callback(query, callback_data)
        elif callback_data.startswith("draw:"):
            await self._handle_draw_callback(query, callback_data)
        elif callback_data.startswith("ollama_"):
            await self._handle_ollama_callback(query, callback_data)
        elif callback_data.startswith("status:"):
            # Admin shortcut buttons invoked from /status
            parts = callback_data.split(":")
            action = parts[1] if len(parts) > 1 else ""
            try:
                if action == "diag":
                    class _Ctx: args = []
                    await self.diag_command(update, _Ctx)
                elif action == "logs":
                    n = parts[2] if len(parts) > 2 else "120"
                    class _Ctx: args = [n]
                    await self.logs_command(update, _Ctx)
                elif action == "restart":
                    class _Ctx: args = []
                    await self.restart_command(update, _Ctx)
                else:
                    await query.answer("Unknown action", show_alert=True)
                return
            except Exception as exc:
                logging.error("status action failed: %s", exc)
                try:
                    await query.answer("Action failed", show_alert=True)
                except Exception:
                    pass
                return
        elif callback_data.startswith("summary_model:"):
            suffix = callback_data.split(":", 1)[1]
            if suffix == "back":
                await self._handle_summary_model_back(query)
            else:
                parts = callback_data.split(":")
                if len(parts) == 3:
                    provider_key = parts[1]
                    try:
                        index = int(parts[2])
                    except ValueError:
                        await query.answer("Invalid selection", show_alert=True)
                        return
                    await self._handle_summary_model_callback(query, provider_key, index)
                else:
                    await query.answer("Invalid selection", show_alert=True)
            return
        elif callback_data.startswith("summary_provider:"):
            provider_key = callback_data.split(":", 1)[1]
            await self._handle_summary_provider_callback(query, provider_key)
            return
        elif callback_data.startswith("summary_quick:"):
            parts = callback_data.split(":", 2)
            if len(parts) != 3:
                await query.answer("Invalid selection", show_alert=True)
                return
            _, provider_key, model_slug = parts
            chat_id = query.message.chat.id
            message_id = query.message.message_id
            session = self._get_summary_session(chat_id, message_id)
            if not session:
                await query.answer("Session expired. Please pick a summary again.", show_alert=True)
                await self._show_main_summary_options(query)
                return
            # Build a model option similar to list entries
            try:
                if provider_key == "cloud":
                    from llm_config import llm_config as _lc
                    resolved_provider, resolved_model, _ = _lc.get_model_config(None, model_slug)
                    model_option = {
                        "provider": resolved_provider,
                        "model": resolved_model,
                        "label": f"{self._friendly_llm_provider(resolved_provider)} • {self._short_model_name(resolved_model)}",
                        "button_label": f"{self._friendly_llm_provider(resolved_provider)} • {self._short_label(self._short_model_name(resolved_model), 24)}",
                    }
                else:
                    model_option = {
                        "provider": "ollama",
                        "model": model_slug,
                        "label": f"Ollama • {self._short_model_name(model_slug)}",
                        "button_label": f"{self._short_label(self._short_model_name(model_slug), 24)}",
                    }
            except Exception:
                await query.answer("Model unavailable. Choose a provider.")
                await self._handle_summary_provider_callback(query, provider_key if provider_key in ("cloud", "ollama") else "cloud")
                return
            # Remember last used
            try:
                self._remember_last_model(user_id, provider_key, model_option.get("model"))
            except Exception:
                pass
            # Linear flow: always run summary first; TTS selection occurs after summary
            await self._execute_summary_with_model(query, session, provider_key, model_option)
            return
        elif callback_data.startswith("summary_combo:"):
            # One-tap combo: derive model + TTS from env quicks and auto-run end-to-end
            parts = callback_data.split(":", 1)
            combo_kind = parts[1] if len(parts) > 1 else ""
            chat_id = query.message.chat.id
            message_id = query.message.message_id
            session = self._get_summary_session(chat_id, message_id)
            if not session:
                await query.answer("Session expired. Please pick a summary again.", show_alert=True)
                await self._show_main_summary_options(query)
                return
            try:
                if combo_kind == "cloud":
                    # Resolve cloud model via llm_config
                    from llm_config import llm_config as _lc
                    cloud_model_slug = get_quick_cloud_env_model()
                    if not cloud_model_slug:
                        await query.answer("No QUICK_CLOUD_MODEL set", show_alert=True)
                        await self._handle_summary_provider_callback(query, "cloud")
                        return
                    resolved_provider, resolved_model, _ = _lc.get_model_config(None, cloud_model_slug)
                    model_option = {
                        "provider": resolved_provider,
                        "model": resolved_model,
                        "label": f"{self._friendly_llm_provider(resolved_provider)} • {self._short_model_name(resolved_model)}",
                        "button_label": f"{self._friendly_llm_provider(resolved_provider)} • {self._short_label(self._short_model_name(resolved_model), 24)}",
                    }
                    # Preselect OpenAI TTS voice from env (default: fable)
                    cloud_voice = (os.getenv("TTS_CLOUD_VOICE") or "fable").strip()
                    # Do not preselect TTS here; keep linear flow
                    await self._execute_summary_with_model(query, session, "cloud", model_option)
                    return
                elif combo_kind == "local":
                    local_model_slug = (os.getenv("QUICK_LOCAL_MODEL") or "").strip()
                    if not local_model_slug:
                        await query.answer("No QUICK_LOCAL_MODEL set", show_alert=True)
                        await self._handle_summary_provider_callback(query, "ollama")
                        return
                    model_option = {
                        "provider": "ollama",
                        "model": local_model_slug,
                        "label": f"Ollama • {self._short_model_name(local_model_slug)}",
                        "button_label": f"{self._short_label(self._short_model_name(local_model_slug), 24)}",
                    }
                    # Pick first favorite from TTS_QUICK_FAVORITE for local combos
                    fav_env = (os.getenv("TTS_QUICK_FAVORITE") or "").strip()
                    selected_voice = None
                    if fav_env:
                        first = [s.strip() for s in fav_env.split(",") if s.strip()]
                        if first:
                            token = first[0]
                            if "|" in token:
                                eng, slug = token.split("|", 1)
                                selected_voice = {'favorite_slug': slug.strip(), 'voice_id': None, 'engine': eng.strip()}
                    # Do not preselect TTS here; keep linear flow
                    await self._execute_summary_with_model(query, session, "ollama", model_option)
                    return
                else:
                    await query.answer("Unknown combo", show_alert=True)
                    await self._show_main_summary_options(query)
                    return
            except Exception as exc:
                logging.error("Combo start failed: %s", exc)
                await self._show_main_summary_options(query)
                return
        
        elif callback_data.startswith('listen_this'):
            # One-off TTS for the exact summary on this message
            try:
                parts = callback_data.split(':')
                video_id = parts[1] if len(parts) >= 3 else (self._current_normalized_id() or '')
                variant = parts[2] if len(parts) >= 3 else ''

                chat_id = query.message.chat.id
                message_id = query.message.message_id

                payload = self._get_oneoff_tts(chat_id, message_id)
                text = None
                if isinstance(payload, dict):
                    text = payload.get('text')
                    video_id = payload.get('video_id') or video_id
                    variant = payload.get('variant') or variant

                if not text:
                    # Fallback to report lookup
                    if not video_id or not variant:
                        await query.answer("No summary text found", show_alert=True)
                        return
                    text = summary_service.resolve_summary_text(video_id, variant)

                if not text:
                    await query.answer("No summary text available for TTS", show_alert=True)
                    return

                # Minimal transform only; do not LLM-rewrite
                base_variant = (variant or '').split(':', 1)[0]
                clean_text = text if base_variant.startswith('audio') else self._format_for_tts_minimal(text)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"listen_{video_id}_{timestamp}.mp3"
                placeholder_json = f"yt_{video_id}_listen.json"

                await query.answer("Generating audio…")
                # Visible status line below the summary
                status_msg = None
                try:
                    status_msg = await query.message.reply_text("⏳ Generating audio…")
                except Exception:
                    status_msg = None
                audio_path = await self.summarizer.generate_tts_audio(clean_text, filename, placeholder_json)
                if not audio_path or not Path(audio_path).exists():
                    await query.answer("TTS failed", show_alert=True)
                    try:
                        if status_msg:
                            await status_msg.edit_text("❌ Audio generation failed")
                    except Exception:
                        pass
                    return

                with open(audio_path, 'rb') as f:
                    await query.message.reply_voice(voice=f, caption="▶️ One‑off playback", parse_mode=ParseMode.MARKDOWN)
                try:
                    if status_msg:
                        await status_msg.edit_text("✅ Audio sent")
                except Exception:
                    pass
            except Exception as e:
                logging.error(f"listen_this error: {e}")
                await query.answer("Error generating audio", show_alert=True)

        elif callback_data.startswith('gen_quiz'):
            # One-tap quiz generation from Key Points
            try:
                parts = callback_data.split(':')
                video_id = parts[1] if len(parts) >= 2 else (self._current_normalized_id() or '')
                if not video_id:
                    await query.answer("Missing video id", show_alert=True)
                    return
                # Keep the original summary visible; show a toast instead of replacing text
                await query.answer("Generating quiz…")
                # Visible status line below the summary
                status_msg = None
                try:
                    status_msg = await query.message.reply_text("⏳ Generating quiz…")
                except Exception:
                    status_msg = None

                # Find Key Points text, synthesize if needed
                kp_text = summary_service.resolve_summary_text(video_id, 'bullet-points')
                if not kp_text:
                    # synthesize ephemeral Key Points using available URL
                    url = self._resolve_video_url(video_id, self._current_url())
                    if not url:
                        await query.edit_message_text("❌ No source available for quiz generation.")
                        return
                    tmp = await self.summarizer.process_video(url, summary_type='bullet-points')
                    if isinstance(tmp, dict):
                        sv = tmp.get('summary') or {}
                        if isinstance(sv, dict):
                            kp_text = sv.get('bullet_points') or sv.get('summary')
                    if not kp_text and isinstance(tmp, dict):
                        s = tmp.get('summary')
                        if isinstance(s, str):
                            kp_text = s
                if not kp_text:
                    await query.edit_message_text("❌ Could not obtain Key Points for quiz.")
                    return

                # Get metadata for prompt
                report = summary_service.load_local_report(video_id) or {}
                title = (report.get('title') or report.get('video', {}).get('title') or 'Untitled').strip()
                language = (report.get('summary', {}) or {}).get('language') or report.get('summary_language') or 'en'
                difficulty = 'beginner'
                prompt = summary_service.build_quiz_prompt(
                    title=title,
                    keypoints=kp_text,
                    count=10,
                    types=["multiplechoice", "truefalse"],
                    difficulty=difficulty,
                    language=language,
                    explanations=True,
                )

                # Generate quiz via Dashboard
                gen = summary_service.post_dashboard_json('/api/generate-quiz', {
                    'prompt': prompt,
                    'model': 'google/gemini-2.5-flash-lite',
                    'fallback_model': 'deepseek/deepseek-v3.1-terminus',
                    'max_tokens': 1800,
                    'temperature': 0.7,
                })
                if not gen or (gen.get('success') is False):
                    await query.edit_message_text("❌ Quiz generation failed. Please try again.")
                    return
                raw = gen.get('content')
                try:
                    quiz = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    await query.edit_message_text("❌ Invalid quiz JSON returned by generator.")
                    return
                if not summary_service.validate_quiz_payload(quiz, explanations=True):
                    await query.edit_message_text("❌ Generated quiz did not pass validation.")
                    return

                # Optional categorization
                cat = summary_service.post_dashboard_json('/api/categorize-quiz', {
                    'topic': title,
                    'quiz_content': '\n'.join([i.get('question','') for i in quiz.get('items', [])[:3]])
                })
                meta = quiz.setdefault('meta', {})
                meta['topic'] = meta.get('topic') or title
                meta['difficulty'] = meta.get('difficulty') or difficulty
                meta['language'] = language
                # Include count in meta for convenience (frontend can show without computing)
                try:
                    meta['count'] = int(quiz.get('count') or len(quiz.get('items') or []))
                except Exception:
                    meta['count'] = len(quiz.get('items') or [])
                if cat and cat.get('success'):
                    meta['category'] = cat.get('category')
                    meta['subcategory'] = cat.get('subcategory')
                    meta['auto_categorized'] = True
                    if 'confidence' in cat:
                        meta['categorization_confidence'] = cat['confidence']

                # Save
                slug = summary_service.slugify(title)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                clean_vid = video_id.replace('yt:', '')
                filename = f"{slug}_{clean_vid}_{ts}.json"
                saved = summary_service.post_dashboard_json('/api/save-quiz', {'filename': filename, 'quiz': quiz})
                final_name = (saved or {}).get('filename') or filename

                # Reply with links
                dash = summary_service.get_dashboard_base() or ''
                qz = f"https://quizzernator.onrender.com/?quiz=api:{final_name}&autoplay=1"
                kb = InlineKeyboardMarkup([
                    [InlineKeyboardButton("▶️ Play in Quizzernator", url=qz)],
                    [InlineKeyboardButton("📂 See in Dashboard", url=f"{dash.rstrip('/')}/api/quiz/{final_name}")],
                    [InlineKeyboardButton("🧩 Generate Again", callback_data=f"gen_quiz:{video_id}"),
                     InlineKeyboardButton("➕ Add Variant", callback_data="summarize_back_to_main")]
                ])
                await query.message.reply_text(
                    f"✅ Saved quiz: {final_name}\n\nUse the buttons below to play or view details.",
                    reply_markup=kb
                )
                try:
                    if status_msg:
                        await status_msg.edit_text("✅ Quiz saved")
                except Exception:
                    pass
            except Exception as e:
                logging.error(f"gen_quiz error: {e}")
                try:
                    await query.message.reply_text("❌ Error generating quiz.")
                finally:
                    try:
                        if 'status_msg' in locals() and status_msg:
                            await status_msg.edit_text("❌ Quiz generation failed")
                    except Exception:
                        pass

        else:
            await query.edit_message_text("❌ Unknown option selected.")
    
    async def _show_main_summary_options(self, query):
        """Show the main summary type selection buttons"""
        content_id = (self.current_item or {}).get("content_id")
        source = (self.current_item or {}).get("source", "youtube")
        variants = self._discover_summary_types(content_id) if content_id else []
        reply_markup = self._build_summary_keyboard(variants, content_id)
        message = self._existing_variants_message(content_id or '', variants, source=source)
        await query.edit_message_text(
            message,
            reply_markup=reply_markup
        )

    # ------------------------- One-off TTS utilities -------------------------
    def _tts_cache_path(self, chat_id: int, message_id: int) -> Path:
        base = Path("./data/tts_cache") / str(chat_id)
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{message_id}.json"

    def _resolve_tts_client(self, base_hint: Optional[str] = None) -> Optional[TTSHubClient]:
        client = self.tts_client
        if client and client.base_api_url:
            return client
        base = base_hint or os.getenv('TTSHUB_API_BASE')
        if not base:
            return None
        client = TTSHubClient(base)
        if client.base_api_url:
            self.tts_client = client
            return client
        return None

    def _cache_oneoff_tts(self, chat_id: int, message_id: int, payload: Dict[str, Any]) -> None:
        try:
            self.tts_cache[(chat_id, message_id)] = dict(payload)
            path = self._tts_cache_path(chat_id, message_id)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"⚠️ Failed to persist tts_cache for {chat_id}:{message_id}: {e}")

    def _get_oneoff_tts(self, chat_id: int, message_id: int) -> Optional[Dict[str, Any]]:
        payload = self.tts_cache.get((chat_id, message_id))
        if payload:
            return payload
        try:
            path = self._tts_cache_path(chat_id, message_id)
            if path.exists():
                return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            pass
        return None

    def _format_for_tts_minimal(self, text: str) -> str:
        """Minimal cleanup for TTS: remove markdown/HTML and normalize bullet lines."""
        if not isinstance(text, str):
            return ""
        import re as _re
        s = text.replace('\r', '\n')
        # Remove simple markdown markers
        s = s.replace('**', '').replace('__', '').replace('`', '')
        # Strip HTML
        s = _re.sub(r'<[^>]+>', '', s)
        # Normalize bullets to sentences
        lines = [ln.strip() for ln in s.split('\n')]
        out = []
        for ln in lines:
            if not ln:
                out.append('')
                continue
            ln = _re.sub(r'^[\-*•]\s+', '', ln)
            if ln and ln[-1] not in '.!?':
                ln += '.'
            out.append(ln)
        s = '\n'.join(out)
        s = _re.sub(r'\n{3,}', '\n\n', s)
        return s.strip()

    # ------------------------- External TTS hub helpers -------------------------
    def _gender_label(self, gender: Optional[str]) -> str:
        return ui_gender_label(gender)

    def _build_tts_catalog_keyboard(self, session: Dict[str, Any]) -> InlineKeyboardMarkup:
        return ui_build_tts_catalog_keyboard(session)

    async def _refresh_tts_catalog(self, query, session: Dict[str, Any]) -> None:
        catalogs = session.get('catalogs') or {}
        active_engine = session.get('active_engine')
        if active_engine and active_engine in catalogs:
            catalog = catalogs.get(active_engine) or {}
            session['catalog'] = catalog
        else:
            catalog = session.get('catalog') or {}
        prompt = self._tts_prompt_text(
            session.get('text', ''),
            last_voice=session.get('last_voice'),
            gender=session.get('selected_gender'),
            family=session.get('selected_family'),
            catalog=catalog,
        )
        keyboard = self._build_tts_catalog_keyboard(session)
        await query.edit_message_text(prompt, reply_markup=keyboard)

    def _build_provider_keyboard(self, include_local: bool = True) -> InlineKeyboardMarkup:
        row = []
        if include_local:
            row.append(InlineKeyboardButton("Local TTS hub", callback_data="tts_provider:local"))
        else:
            row.append(InlineKeyboardButton("Local TTS hub", callback_data="tts_provider:local"))
        row.append(InlineKeyboardButton("OpenAI TTS", callback_data="tts_provider:openai"))
        buttons = [row, [InlineKeyboardButton("❌ Cancel", callback_data="tts_cancel")]]
        return InlineKeyboardMarkup(buttons)

    def _build_local_failure_keyboard(self) -> InlineKeyboardMarkup:
        return ui_build_local_failure_keyboard()

    async def _execute_tts_job(self, query, session: Dict[str, Any], provider: str) -> None:
        await tts_service.execute_job(self, query, session, provider)

    async def _handle_local_unavailable(self, query, session: Dict[str, Any], message: str = "") -> None:
        await tts_service.handle_local_unavailable(self, query, session, message=message)

    async def _enqueue_tts_job(self, query, session: Dict[str, Any]) -> None:
        await tts_service.enqueue_job(self, query, session)

    async def _finalize_tts_delivery(self, query, session: Dict[str, Any], audio_path: Path, provider: str) -> None:
        await tts_service.finalize_delivery(self, query, session, audio_path, provider)

    def _tts_session_key(self, chat_id: int, message_id: int) -> tuple:
        return (chat_id, message_id)

    def _store_tts_session(self, chat_id: int, message_id: int, payload: Dict[str, Any]) -> None:
        self.tts_sessions[self._tts_session_key(chat_id, message_id)] = payload

    def _get_tts_session(self, chat_id: int, message_id: int) -> Optional[Dict[str, Any]]:
        return self.tts_sessions.get(self._tts_session_key(chat_id, message_id))

    def _remove_tts_session(self, chat_id: int, message_id: int) -> None:
        self.tts_sessions.pop(self._tts_session_key(chat_id, message_id), None)

    # --- Content-anchored TTS preselect helpers ---
    def _store_content_tts_preselect(self, normalized_video_id: str, payload: Dict[str, Any]) -> None:
        if not normalized_video_id:
            return
        self.tts_preselect_by_content[normalized_video_id] = payload

    def _get_content_tts_preselect(self, normalized_video_id: str) -> Optional[Dict[str, Any]]:
        if not normalized_video_id:
            return None
        return self.tts_preselect_by_content.get(normalized_video_id)

    def _remove_content_tts_preselect(self, normalized_video_id: str) -> None:
        if not normalized_video_id:
            return
        self.tts_preselect_by_content.pop(normalized_video_id, None)

    def _build_tts_keyboard(self, favorites: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
        return ui_build_tts_keyboard(favorites)

    def _tts_prompt_text(
        self,
        text: str,
        last_voice: Optional[str] = None,
        gender: Optional[str] = None,
        family: Optional[str] = None,
        catalog: Optional[Dict[str, Any]] = None,
    ) -> str:
        return ui_tts_prompt_text(
            text,
            last_voice=last_voice,
            gender=gender,
            family=family,
            catalog=catalog,
        )

    def _tts_voice_label(self, session: Dict[str, Any], slug: str) -> str:
        return ui_tts_voice_label(session, slug)

    async def tts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_user_allowed(user_id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return

        client = self._resolve_tts_client()
        if not client or not client.base_api_url:
            await update.message.reply_text("⚠️ TTS hub is not configured. Set TTSHUB_API_BASE and try again.")
            return
        # Fast preflight to avoid waiting on multiple long hub calls when offline
        try:
            if not reach_hub_ok(client.base_api_url):
                await update.message.reply_text(
                    "⚠️ Local TTS hub is unreachable. The /tts preview uses the hub.\n"
                    "Try again when your Mac is online, or generate audio via the summary flow using OpenAI."
                )
                return
        except Exception:
            pass
        self.tts_client = client

        full_text = (update.message.text or "").split(" ", 1)
        speak_text = full_text[1].strip() if len(full_text) > 1 else ""
        if not speak_text:
            await update.message.reply_text("🗣️ Usage: /tts Your message here")
            return

        catalog = None
        try:
            catalog = await client.fetch_catalog()
        except Exception as e:
            logging.warning(f"TTS catalog fetch failed: {e}")
            catalog = None

        default_engine = DEFAULT_ENGINE
        catalogs: Dict[str, Dict[str, Any]] = {}
        if isinstance(catalog, dict):
            detected_engine = (
                (catalog.get('meta') or {}).get('engine')
                or catalog.get('engine')
            )
            if isinstance(detected_engine, str) and detected_engine.strip():
                default_engine = detected_engine.strip()
            catalogs[default_engine] = catalog
        else:
            catalogs[default_engine] = {}

        favorites_list: List[Dict[str, Any]] = []
        try:
            favorites_list = await client.fetch_favorites()
        except Exception as e:
            logging.warning(f"TTS favorites fetch failed: {e}")
            favorites_list = []

        def resolve_engine(value: Optional[str]) -> str:
            if isinstance(value, str) and value.strip():
                return value.strip()
            return default_engine

        favorite_engines: Set[str] = set()
        for fav in favorites_list:
            if not isinstance(fav, dict):
                continue
            if not fav.get('voiceId'):
                continue
            favorite_engines.add(resolve_engine(fav.get('engine')))

        for engine in sorted(favorite_engines):
            if engine in catalogs and catalogs[engine]:
                continue
            if engine == default_engine and catalogs.get(engine):
                continue
            try:
                extra_catalog = await client.fetch_catalog(engine=engine)
                catalogs[engine] = extra_catalog or {}
            except Exception as e:
                logging.warning(f"TTS catalog fetch failed for engine {engine}: {e}")
                catalogs.setdefault(engine, {})

        def engine_has_matches(engine: str) -> bool:
            fav_ids = {
                fav.get('voiceId')
                for fav in favorites_list
                if isinstance(fav, dict)
                and fav.get('voiceId')
                and resolve_engine(fav.get('engine')) == engine
            }
            if not fav_ids:
                return False
            cat = catalogs.get(engine) or {}
            catalog_ids = {
                voice.get('id')
                for voice in (cat.get('voices') or [])
                if isinstance(voice, dict) and voice.get('id')
            }
            return bool(catalog_ids & fav_ids)

        active_engine = '__all__' if favorites_list else default_engine
        if active_engine == '__all__' and favorites_list:
            if not engine_has_matches(default_engine):
                for engine in sorted(favorite_engines):
                    if engine_has_matches(engine):
                        default_engine = engine
                        break

        active_catalog = catalogs.get(default_engine if active_engine == '__all__' else active_engine) or {}
        has_catalog_voices = any(
            isinstance(cat, dict) and (cat.get('voices') or [])
            for cat in catalogs.values()
        )

        if has_catalog_voices:
            session_payload = {
                "text": speak_text,
                "catalog": active_catalog,
                "catalogs": catalogs,
                "active_engine": active_engine,
                "default_engine": default_engine,
                "selected_gender": None,
                "selected_family": None,
                "last_voice": None,
                "favorites": favorites_list,
                "voice_mode": 'favorites' if favorites_list else 'all',
                "tts_base": client.base_api_url,
                "mode": "oneoff_tts",
            }
            # Determine up to 2 quick favorites based on mode: env | last | auto
            mode = (os.getenv("TTS_QUICK_MODE") or "auto").strip().lower()
            env_val = (os.getenv("TTS_QUICK_FAVORITE") or "").strip()
            env_list = [s.strip() for s in env_val.split(",") if s.strip()] if env_val else []

            def map_env_to_alias(items: List[str]) -> List[str]:
                aliases: List[str] = []
                for item in items:
                    if "|" in item:
                        eng, slug = item.split("|", 1)
                        eng = eng.strip(); slug = slug.strip()
                        if eng and slug:
                            aliases.append(f"fav|{eng}|{slug}")
                    else:
                        # Resolve engine from favorites list
                        for fav in favorites_list:
                            if not isinstance(fav, dict):
                                continue
                            slug = (fav.get('slug') or fav.get('voiceId') or '').strip()
                            if slug and slug == item:
                                eng = (fav.get('engine') or '').strip()
                                if eng:
                                    aliases.append(f"fav|{eng}|{slug}")
                                    break
                # Deduplicate preserving order
                seen = set()
                result = []
                for a in aliases:
                    if a in seen:
                        continue
                    seen.add(a)
                    result.append(a)
                return result

            quick_slugs: List[str] = []
            if mode == "env" and env_list:
                quick_slugs = map_env_to_alias(env_list)
            elif mode == "last":
                quick_slugs = self._last_tts_voices(user_id, n=2)
            else:  # auto
                if env_list:
                    quick_slugs = map_env_to_alias(env_list)
                if not quick_slugs:
                    quick_slugs = self._last_tts_voices(user_id, n=2)
                if not quick_slugs and favorites_list:
                    # Take first two favorites
                    acc = []
                    for fav in favorites_list:
                        if not isinstance(fav, dict):
                            continue
                        slug = (fav.get('slug') or fav.get('voiceId') or '').strip()
                        eng = (fav.get('engine') or '').strip()
                        if slug and eng:
                            acc.append(f"fav|{eng}|{slug}")
                        if len(acc) >= 2:
                            break
                    quick_slugs = acc
            if quick_slugs:
                session_payload["quick_favorite_slugs"] = quick_slugs[:2]
            prompt = self._tts_prompt_text(speak_text, catalog=active_catalog)
            keyboard = self._build_tts_catalog_keyboard(session_payload)
            prompt_message = await update.message.reply_text(prompt, reply_markup=keyboard)
            self._store_tts_session(prompt_message.chat_id, prompt_message.message_id, session_payload)
            return

        # Fallback to favorites when catalog unavailable
        favorites = favorites_list
        if not favorites:
            try:
                favorites = await client.fetch_favorites()
            except Exception as e:
                logging.error(f"Failed to fetch TTS favorites: {e}")
                await update.message.reply_text("❌ Could not reach the TTS hub. Please try again later.")
                return

        if not favorites:
            await update.message.reply_text("⚠️ No favorite voices available on the TTS hub.")
            return

        prompt = self._tts_prompt_text(speak_text)
        # Build favorites keyboard and a compact lookup for selections
        keyboard = self._build_tts_keyboard(favorites)
        prompt_message = await update.message.reply_text(prompt, reply_markup=keyboard)

        # Build a compact lookup: v0, v1, ... -> favorite voice/meta
        voice_lookup: Dict[str, Dict[str, Any]] = {}
        for i, fav in enumerate(favorites):
            slug = fav.get('slug') or fav.get('voiceId')
            if not slug:
                continue
            engine = fav.get('engine')
            base_label = strip_favorite_label(fav.get('label')) or fav.get('voiceId') or slug
            display_label = f"{short_engine_label(engine)} {base_label}".strip()
            short_key = f"v{i}"
            entry = {
                'label': base_label,
                'display_label': display_label,
                'button_label': display_label if len(display_label) <= 32 else f"{display_label[:29]}…",
                'voice': None,
                'voiceId': fav.get('voiceId'),
                'engine': engine,
                'favoriteSlug': slug,
            }
            voice_lookup[short_key] = entry
            voice_lookup[f"fav|{engine or ''}|{slug}"] = entry

        session_payload = {
            "text": speak_text,
            "favorites": favorites,
            "tts_base": client.base_api_url,
            "last_voice": None,
            "selected_family": None,
            "voice_lookup": voice_lookup,
            "voice_alias_map": {},
            "mode": "oneoff_tts",
        }
        self._store_tts_session(prompt_message.chat_id, prompt_message.message_id, session_payload)

    async def _handle_tts_callback(self, query, callback_data: str):
        await tts_service.handle_callback(self, query, callback_data)

    async def _show_proficiency_selector(self, query, summary_type: str):
        """Show proficiency level selector for language learning"""
        if summary_type == "audio-fr":
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🟢 Débutant", callback_data="summarize_audio-fr:beginner"),
                    InlineKeyboardButton("🟡 Intermédiaire", callback_data="summarize_audio-fr:intermediate"),
                    InlineKeyboardButton("🔵 Avancé", callback_data="summarize_audio-fr:advanced"),
                ],
                [
                    InlineKeyboardButton("⬅️ Retour", callback_data="summarize_back_to_main")
                ]
            ])
            await query.edit_message_text("🇫🇷 **Choisissez votre niveau de français :**", 
                                        parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
        else:  # audio-es
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🟢 Principiante", callback_data="summarize_audio-es:beginner"),
                    InlineKeyboardButton("🟡 Intermedio", callback_data="summarize_audio-es:intermediate"),
                    InlineKeyboardButton("🔵 Avanzado", callback_data="summarize_audio-es:advanced"),
                ],
                [
                    InlineKeyboardButton("⬅️ Volver", callback_data="summarize_back_to_main")
                ]
            ])
            await query.edit_message_text("🇪🇸 **Elige tu nivel de español:**", 
                                        parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
    
    async def _process_content_summary(
        self,
        query,
        summary_type: str,
        user_name: str,
        proficiency_level: str = None,
        *,
        provider_key: str = "cloud",
        summarizer: Optional[YouTubeSummarizer] = None,
        provider_label: Optional[str] = None,
    ):
        await summary_service.process_content_summary(
            self,
            query,
            summary_type,
            user_name,
            proficiency_level,
            provider_key=provider_key,
            summarizer=summarizer,
            provider_label=provider_label,
        )

    async def _send_formatted_response(self, query, result: Dict[str, Any], summary_type: str, export_info: Dict = None):
        await summary_service.send_formatted_response(self, query, result, summary_type, export_info)

    async def _prepare_tts_generation(self, query, result: Dict[str, Any], summary_text: str, summary_type: str):
        await summary_service.prepare_tts_generation(self, query, result, summary_text, summary_type)

    async def _prompt_tts_provider(self, query, session_payload: Dict[str, Any], title: str) -> None:
        await tts_service.prompt_provider(self, query, session_payload, title)

    def _resolve_redirects(self, url: str, timeout: int = 8) -> str:
        """Resolve short links/redirects (e.g., flip.it) to their final destination.

        Returns the final URL on success; on errors or if requests is unavailable,
        returns the original URL.
        """
        try:
            if not url or not isinstance(url, str):
                return url
            if requests is None:  # type: ignore[name-defined]
                return url
            resp = requests.get(url, allow_redirects=True, timeout=timeout)
            try:
                final = getattr(resp, 'url', None) or url
            finally:
                try:
                    resp.close()
                except Exception:
                    pass
            return final
        except Exception:
            return url
    def _extract_youtube_url(self, text: str) -> Optional[str]:
        """Extract YouTube URL from text."""
        match = self.youtube_url_pattern.search(text)
        if match:
            video_id = match.group(1)
            return f"https://www.youtube.com/watch?v={video_id}"
        return None
    
    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to use the bot."""
        return user_id in self.allowed_user_ids
    
    async def _handle_audio_summary(self, query, result: Dict[str, Any], summary_type: str):
        await summary_service.handle_audio_summary(self, query, result, summary_type)

    def _build_audio_inline_keyboard(self, video_id: str, base_variant: str, report_id: str):
        if not report_id:
            return None

        dashboard_url = (
            os.getenv('DASHBOARD_URL')
            or os.getenv('POSTGRES_DASHBOARD_URL')
            or 'https://ytv2-dashboard-postgres.onrender.com'
        )

        if not dashboard_url:
            logging.warning("⚠️ No DASHBOARD_URL set - skipping audio link buttons")
            return None

        report_id_encoded = urllib.parse.quote(report_id, safe='')

        listen_cb = f"listen_this:{video_id}:{base_variant}"
        gen_cb = f"gen_quiz:{video_id}"

        keyboard = [
            [
                InlineKeyboardButton("📊 Dashboard", url=dashboard_url),
                InlineKeyboardButton("📄 Open Summary", url=f"{dashboard_url}#report={report_id_encoded}")
            ],
            [
                InlineKeyboardButton("▶️ Listen", callback_data=listen_cb) if len(listen_cb.encode('utf-8')) <= 64 else None,
                InlineKeyboardButton("🧩 Generate Quiz", callback_data=gen_cb) if len(gen_cb.encode('utf-8')) <= 64 else None,
            ],
            [
                InlineKeyboardButton("➕ Add Variant", callback_data="summarize_back_to_main")
            ]
        ]

        # Filter out None entries in second row
        keyboard[1] = [btn for btn in keyboard[1] if btn is not None]
        keyboard = [row for row in keyboard if row]

        return InlineKeyboardMarkup(keyboard) if keyboard else None

    async def _send_long_message(self, query, header_text: str, summary_text: str, reply_markup=None):
        """Send long messages by splitting into multiple Telegram messages if needed."""
        try:
            summary_text = summary_text or ""
            safe_summary = self._escape_markdown(summary_text)
            safe_header = header_text or ""

            async def _edit_with_retry(text: str, markup=None):
                while True:
                    try:
                        return await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=markup)
                    except RetryAfter as exc:
                        await asyncio.sleep(exc.retry_after)
                    except BadRequest as exc:
                        message = (getattr(exc, 'message', None) or str(exc) or "").lower()
                        if "query is too old" in message or "message to edit not found" in message:
                            return await query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=markup)
                        raise


            # Calculate available space for summary content
            # Reserve space for header, formatting, and safety margin
            header_length = len(safe_header)
            safety_margin = 100  # Buffer for formatting and other text
            available_space = self.MAX_MESSAGE_LENGTH - header_length - safety_margin
            
            # If summary fits in one message, send normally
            if len(safe_summary) <= available_space:
                full_message = f"{safe_header}\n{safe_summary}"
                msg = await _edit_with_retry(full_message, markup=reply_markup)
                return msg
            
            # Summary is too long - split into multiple messages
            print(f"📱 Long summary detected ({len(summary_text):,} chars) - splitting into multiple messages")
            
            # Split summary into chunks that fit within message limits
            chunks = self._split_text_into_chunks(safe_summary, available_space)
            
            # Send first message with header + first chunk
            first_message = f"{safe_header}\n{chunks[0]}"
            if len(chunks) > 1:
                first_message += f"\n\n📄 *Continued in next message... ({len(chunks)} parts total)*"
            
            await _edit_with_retry(first_message)
            
            # Send remaining chunks as follow-up messages
            last_msg = None
            for i, chunk in enumerate(chunks[1:], 2):
                chunk_message = f"📄 **Summary (Part {i}/{len(chunks)}):**\n\n{chunk}"
                
                # Determine if this is the last chunk
                is_last_chunk = (i == len(chunks))
                
                # Add continuation indicator if not the last chunk
                if not is_last_chunk:
                    chunk_message += f"\n\n*Continued in next message...*"
                    chunk_reply_markup = None  # No buttons on continuation messages
                else:
                    chunk_message += f"\n\n✅ *Summary complete ({len(chunks)} parts)*"
                    chunk_reply_markup = reply_markup  # Add buttons to final message
                
                # Send as new message (not edit)
                last_msg = await query.message.reply_text(chunk_message, parse_mode=ParseMode.MARKDOWN, reply_markup=chunk_reply_markup)
            return last_msg
                
        except Exception as e:
            logging.error(f"Error sending long message: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            # Fallback to truncated message with buttons
            truncated_summary = safe_summary[:1000] + "..." if len(safe_summary) > 1000 else safe_summary
            fallback_message = f"{safe_header}\n{truncated_summary}\n\n⚠️ *Summary was truncated due to length. View full summary on dashboard.*"
            try:
                msg = await _edit_with_retry(fallback_message, markup=reply_markup)
                return msg
            except Exception as fallback_e:
                logging.error(f"Even fallback message failed: {fallback_e}")
                # Try without buttons as last resort
                msg = await query.edit_message_text(fallback_message, parse_mode=ParseMode.MARKDOWN)
                return msg
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int) -> List[str]:
        # Delegate to UI helper for chunking
        return ui_split_chunks(text, max_chunk_size)
    
    def _escape_markdown(self, text: str) -> str:
        # Delegate to UI helper for escaping
        return ui_escape_markdown(text)

    # ------------------------- Auto-process helpers -------------------------
    async def _maybe_schedule_auto_process(
        self,
        reply_message,
        *,
        source: str,
        url: str,
        content_id: str,
        user_name: str,
    ) -> None:
        """Schedule auto-processing of a default summary after an idle delay.

        Controlled by env:
          - AUTO_PROCESS_DELAY_SECONDS (int > 0 to enable)
          - AUTO_PROCESS_SUMMARY (e.g., 'bullet-points')
          - AUTO_PROCESS_PROVIDER ('cloud' or 'ollama')
        """
        try:
            raw = str(os.getenv('AUTO_PROCESS_DELAY_SECONDS', '0')).strip()
            delay = int(raw)
        except Exception:
            delay = 0
        if delay <= 0:
            return

        # Parse summary preference list (comma-separated); choose first known type
        types_raw = (os.getenv('AUTO_PROCESS_SUMMARY', 'bullet-points') or 'bullet-points').strip()
        preferred_types = [t.strip() for t in types_raw.split(',') if t.strip()]
        allowed_types = {
            'bullet-points', 'comprehensive', 'key-insights',
            'audio', 'audio-fr', 'audio-es'
        }
        summary_type = None
        for t in preferred_types:
            if t in allowed_types:
                summary_type = t
                break
        if not summary_type:
            summary_type = 'bullet-points'
        providers_raw = (os.getenv('AUTO_PROCESS_PROVIDER', 'cloud') or 'cloud').strip().lower()
        raw_candidates = [p.strip() for p in providers_raw.split(',') if p.strip()] or ['cloud']
        # Normalize common aliases to canonical provider keys
        alias_map = {
            'local': 'ollama',
            'hub': 'ollama',
            'wireguard': 'ollama',
            'ollama': 'ollama',
            'api': 'cloud',
            'cloud': 'cloud',
        }
        candidates = [alias_map.get(p, p) for p in raw_candidates]

        # Choose first available provider from the preference list
        chosen = None
        # Prefer hub proxy for provider checks (we only support hub-based Ollama here)
        via_hub = bool(os.getenv('TTSHUB_API_BASE'))
        for p in candidates:
            if p == 'ollama':
                try:
                    if via_hub and reach_hub_ollama_ok(os.getenv('TTSHUB_API_BASE')):
                        chosen = 'ollama'
                        logging.info("AUTO_PROCESS: picked ollama (hub proxy reachable)")
                        break
                    if chosen != 'ollama':
                        logging.info("AUTO_PROCESS: Ollama unreachable; skipping to next provider")
                except Exception:
                    logging.info("AUTO_PROCESS: Ollama probe failed; skipping to next provider")
                    continue
            elif p == 'cloud':
                chosen = 'cloud'
                logging.info("AUTO_PROCESS: picked cloud (preference/candidates=%s)", candidates)
                break
            else:
                # Unknown entry, skip
                continue
        provider_key = chosen or 'cloud'
        if not chosen:
            logging.info("AUTO_PROCESS: no providers available from %s; defaulting to cloud", candidates)

        # Avoid duplicates: if variant already exists, do not schedule
        current = set(self._discover_summary_types(content_id) or [])
        if any((summary_type == v.split(':', 1)[0]) for v in current):
            return

        chat_id = reply_message.chat.id
        msg_id = reply_message.message_id

        async def _runner():
            try:
                await asyncio.sleep(delay)
                # Check again before running
                again = set(self._discover_summary_types(content_id) or [])
                if any((summary_type == v.split(':', 1)[0]) for v in again):
                    return

                # Provide a minimal Query-like wrapper so downstream uses edit_message_text
                class _Q:
                    def __init__(self, m):
                        self.message = m
                    async def edit_message_text(self, text, parse_mode=None, reply_markup=None):
                        return await reply_message.edit_text(text, parse_mode=parse_mode, reply_markup=reply_markup)

                fake_query = _Q(reply_message)

                # Resolve summarizer for target provider/model
                model_slug = None
                if provider_key == 'cloud':
                    model_slug = get_quick_cloud_env_model() or None
                else:
                    model_slug = (os.getenv('QUICK_LOCAL_MODEL') or '').strip() or None
                summarizer = self._get_summary_summarizer(provider_key, model_slug)

                # Ensure current item context is set
                self.current_item = {
                    'source': source,
                    'url': url,
                    'content_id': content_id if source != 'youtube' else f"yt:{content_id}",
                    'raw_id': content_id,
                    'normalized_id': content_id,
                }

                await summary_service.process_content_summary(
                    self,
                    fake_query,
                    summary_type,
                    user_name,
                    None,
                    provider_key=provider_key,
                    summarizer=summarizer,
                    provider_label=None,
                )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                try:
                    await reply_message.reply_text(f"⚠️ Auto-process failed: {str(exc)[:120]}")
                except Exception:
                    pass
            finally:
                self._auto_tasks.pop((chat_id, msg_id), None)

        # Cancel any existing task for this message and schedule a new one
        key = (chat_id, msg_id)
        old = self._auto_tasks.pop(key, None)
        if old and not old.done():
            old.cancel()
        self._auto_tasks[key] = asyncio.create_task(_runner())

    def _ai2ai_audio_caption(self, session: Dict[str, Any]) -> str:
        defaults = self._ollama_persona_defaults()
        a_raw = session.get("persona_a") or defaults[0]
        b_raw = session.get("persona_b") or defaults[1]
        a_disp, _ = self._persona_parse(a_raw)
        b_disp, _ = self._persona_parse(b_raw)
        model_a = session.get("ai2ai_model_a") or session.get("model") or "?"
        model_b = session.get("ai2ai_model_b") or session.get("model") or "?"
        tts_a = (session.get('ai2ai_tts_a_label') or '').strip()
        tts_b = (session.get('ai2ai_tts_b_label') or '').strip()
        provider = (session.get('ai2ai_tts_provider') or '').strip()
        return build_ai2ai_audio_caption(
            a_display=a_disp,
            b_display=b_disp,
            model_a=str(model_a),
            model_b=str(model_b),
            tts_a_label=tts_a,
            tts_b_label=tts_b,
            provider=provider,
            escape_md=self._escape_markdown,
        )

    async def _ollama_ai2ai_generate_audio(self, chat_id: int, session: Dict[str, Any]) -> Optional[str]:
        # Collect utterances
        transcript = session.get("ai2ai_transcript") or []
        if not isinstance(transcript, list) or not transcript:
            raise RuntimeError("no transcript")
        # Config
        def truthy(val: Optional[str], default: bool = True) -> bool:
            if val is None:
                return default
            v = str(val).strip().lower()
            return v not in ("0", "false", "no", "off")

        intro_enabled = truthy(os.getenv("OLLAMA_AI2AI_TTS_INTRO", "1"), True)
        try:
            intro_pause_ms = max(0, int(os.getenv("OLLAMA_AI2AI_TTS_INTRO_PAUSE_MS", "650")))
        except Exception:
            intro_pause_ms = 650
        try:
            turn_pause_ms = max(0, int(os.getenv("OLLAMA_AI2AI_TTS_PAUSE_MS", "650")))
        except Exception:
            turn_pause_ms = 650

        provider = (os.getenv("OLLAMA_AI2AI_TTS_PROVIDER", "local").strip().lower())
        use_local = provider in ("1", "true", "yes", "local", "hub")

        from modules.tts_hub import TTSHubClient, LocalTTSUnavailable
        client = None
        if use_local:
            client = TTSHubClient.from_env()
            if not client or not client.base_api_url:
                use_local = False

        # Resolve A/B voices – robustly normalize favorites/ids against hub
        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", (s or "").strip().lower())

        async def _resolve_local(client: TTSHubClient, raw: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
            raw = (raw or "").strip()
            if not raw:
                return (None, None, None)
            wanted = _norm(raw.replace("favorite--", ""))
            # Try favorites first (tagged, then all)
            favs: List[Dict[str, Any]] = []
            try:
                favs = await client.fetch_favorites(tag="ai2ai")
            except Exception:
                favs = []
            if not favs:
                try:
                    favs = await client.fetch_favorites()
                except Exception:
                    favs = []
            for fav in favs:
                slug = (fav.get("slug") or fav.get("voiceId") or "").strip()
                label = (fav.get("label") or "").strip()
                if _norm(slug) == wanted or (_norm(label) == wanted and label):
                    return (fav.get("slug") or slug, fav.get("voiceId"), fav.get("engine"))
            # Try catalog voice ids as fallback
            cat: Optional[Dict[str, Any]] = None
            try:
                cat = await client.fetch_catalog()
            except Exception:
                cat = None
            for voice in (cat or {}).get("voices") or []:
                vid = (voice.get("id") or "").strip()
                label = (voice.get("label") or "").strip()
                if _norm(vid) == wanted or (_norm(label) == wanted and label):
                    eng = voice.get("engine") or voice.get("provider")
                    return (None, vid, eng)
            # Last resort: pass through as given, assuming it's a favorite slug
            return (raw, None, None)

        # Gender-aware voice selection for local hub
        fav_a, vid_a, eng_a = (None, None, None)
        fav_b, vid_b, eng_b = (None, None, None)
        selection_log: List[str] = []
        if use_local and client:
            try:
                # Fetch favorites and catalog once
                favs: List[Dict[str, Any]] = []
                try:
                    favs = await client.fetch_favorites(tag="ai2ai")
                except Exception:
                    favs = []
                if not favs:
                    try:
                        favs = await client.fetch_favorites()
                    except Exception:
                        favs = []
                catalog: Optional[Dict[str, Any]] = None
                try:
                    catalog = await client.fetch_catalog()
                except Exception:
                    catalog = None
                id_to_voice: Dict[str, Dict[str, Any]] = {v.get("id"): v for v in (catalog or {}).get("voices") or [] if v.get("id")}

                def _voice_key(engine: Optional[str], vid: Optional[str]) -> Tuple[str, str]:
                    return ((engine or "").strip().lower(), (vid or "").strip().lower())

                # Pre-resolve env overrides where present
                male_env_raw = os.getenv("OLLAMA_AI2AI_TTS_VOICE_MALE", "").strip()
                female_env_raw = os.getenv("OLLAMA_AI2AI_TTS_VOICE_FEMALE", "").strip()
                env_a_raw = os.getenv("OLLAMA_AI2AI_TTS_VOICE_A", "").strip()
                env_b_raw = os.getenv("OLLAMA_AI2AI_TTS_VOICE_B", "").strip()
                male_env_res = await _resolve_local(client, male_env_raw) if male_env_raw else (None, None, None)
                female_env_res = await _resolve_local(client, female_env_raw) if female_env_raw else (None, None, None)
                env_a_res = await _resolve_local(client, env_a_raw) if env_a_raw else (None, None, None)
                env_b_res = await _resolve_local(client, env_b_raw) if env_b_raw else (None, None, None)

                def _candidates_env_gender(gender: Optional[str]) -> List[Tuple[Optional[str], Optional[str], Optional[str], str]]:
                    if gender == "male" and (male_env_res[0] or male_env_res[1]):
                        return [(male_env_res[0], male_env_res[1], male_env_res[2], "male via env")]
                    if gender == "female" and (female_env_res[0] or female_env_res[1]):
                        return [(female_env_res[0], female_env_res[1], female_env_res[2], "female via env")]
                    return []

                def _candidates_favorites_by_gender(gender: Optional[str]) -> List[Tuple[Optional[str], Optional[str], Optional[str], str]]:
                    if not gender:
                        return []
                    results: List[Tuple[Optional[str], Optional[str], Optional[str], str]] = []
                    for fav in favs:
                        vid = (fav.get("voiceId") or "").strip()
                        if not vid:
                            continue
                        voice_meta = id_to_voice.get(vid) or {}
                        if (voice_meta.get("gender") or "").lower() != gender:
                            continue
                        slug = (fav.get("slug") or vid)
                        eng = fav.get("engine") or voice_meta.get("engine") or voice_meta.get("provider")
                        results.append((slug, vid, eng, f"{gender} via favorites"))
                    # Randomize among matching favorites
                    if results:
                        random.shuffle(results)
                    return results

                def _candidates_catalog_by_gender(gender: Optional[str]) -> List[Tuple[Optional[str], Optional[str], Optional[str], str]]:
                    if not gender or not catalog:
                        return []
                    items = filter_catalog_voices(catalog, gender=gender)
                    results: List[Tuple[Optional[str], Optional[str], Optional[str], str]] = []
                    for v in items:
                        vid = (v.get("id") or "").strip()
                        if not vid:
                            continue
                        eng = v.get("engine") or v.get("provider")
                        results.append((None, vid, eng, f"{gender} via catalog"))
                    return results

                def _candidates_env_slot(slot: str) -> List[Tuple[Optional[str], Optional[str], Optional[str], str]]:
                    if slot.upper() == 'A' and (env_a_res[0] or env_a_res[1]):
                        return [(env_a_res[0], env_a_res[1], env_a_res[2], "slot A via env")]
                    if slot.upper() == 'B' and (env_b_res[0] or env_b_res[1]):
                        return [(env_b_res[0], env_b_res[1], env_b_res[2], "slot B via env")]
                    return []

                def _candidates_generic() -> List[Tuple[Optional[str], Optional[str], Optional[str], str]]:
                    results: List[Tuple[Optional[str], Optional[str], Optional[str], str]] = []
                    # any favorites first (randomized)
                    fav_results: List[Tuple[Optional[str], Optional[str], Optional[str], str]] = []
                    for fav in favs:
                        vid = (fav.get("voiceId") or "").strip()
                        slug = (fav.get("slug") or vid)
                        eng = fav.get("engine") or (id_to_voice.get(vid) or {}).get("engine") or (id_to_voice.get(vid) or {}).get("provider")
                        if vid or slug:
                            fav_results.append((slug, vid, eng, "fallback via favorites"))
                    if fav_results:
                        random.shuffle(fav_results)
                        results.extend(fav_results)
                    # then any catalog (keep deterministic label sort already applied upstream)
                    for v in (catalog or {}).get("voices") or []:
                        vid = (v.get("id") or "").strip()
                        if not vid:
                            continue
                        eng = v.get("engine") or v.get("provider")
                        results.append((None, vid, eng, "fallback via catalog"))
                    return results

                gender_toggle = True
                try:
                    gender_toggle = truthy(os.getenv("OLLAMA_AI2AI_TTS_GENDER_FROM_PERSONA", "1"), True)
                except Exception:
                    gender_toggle = True

                defaults = self._ollama_persona_defaults()
                persona_a_raw = session.get("persona_a") or defaults[0]
                persona_b_raw = session.get("persona_b") or defaults[1]
                a_display, a_gender = self._persona_parse(persona_a_raw)
                b_display, b_gender = self._persona_parse(persona_b_raw)
                if not gender_toggle:
                    a_gender = None
                    b_gender = None

                def _pick_for(label: str, gender: Optional[str], avoid: Optional[Tuple[str, str]] = None) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
                    # Build candidate chain per spec
                    chain: List[Tuple[Optional[str], Optional[str], Optional[str], str]] = []
                    if gender:
                        chain.extend(_candidates_env_gender(gender))
                        chain.extend(_candidates_favorites_by_gender(gender))
                        chain.extend(_candidates_catalog_by_gender(gender))
                        chain.extend(_candidates_generic())
                    else:
                        # Gender unknown: use A/B env first, then generic
                        chain.extend(_candidates_env_slot(label))
                        chain.extend(_candidates_generic())
                    # Pick first non-colliding
                    for fa, vi, en, src in chain:
                        key = _voice_key(en, vi)
                        if avoid and key == avoid:
                            selection_log.append(f"{label} skip collision with other speaker -> voiceId={vi} engine={en}")
                            continue
                        selection_log.append(f"{label} resolved: {src} -> fav={fa} voiceId={vi} engine={en}")
                        return fa, vi, en, src
                    return None, None, None, "unresolved"

                # Choose A then B with collision avoidance
                fa, vi, en, src = _pick_for('A', a_gender)
                fav_a, vid_a, eng_a = (fa, vi, en)
                avoid_key = _voice_key(eng_a, vid_a) if (eng_a or vid_a) else None
                fb, vb, eb, srcb = _pick_for('B', b_gender, avoid=avoid_key)
                fav_b, vid_b, eng_b = (fb, vb, eb)

                for line in selection_log:
                    logging.info(f"AI↔AI TTS voice: {line}")

                if not (fav_a or vid_a) or not (fav_b or vid_b):
                    logging.warning("AI↔AI TTS voice resolution fell back to generic or unresolved; switching to OpenAI fallback")
                    # As last resort, let OpenAI path run
                    use_local = False
                else:
                    # Store provider + human labels for caption
                    def _label_for(fav: Optional[str], vid: Optional[str], eng: Optional[str]) -> str:
                        eng_s = (eng or "").strip()
                        if vid and vid in id_to_voice:
                            v = id_to_voice[vid]
                            name = (v.get('label') or vid)
                        else:
                            name = fav or vid or ""
                        if eng_s and name:
                            return f"{eng_s}:{name}"
                        return name or eng_s
                    session['ai2ai_tts_provider'] = 'local'
                    session['ai2ai_tts_a_label'] = _label_for(fav_a, vid_a, eng_a)
                    session['ai2ai_tts_b_label'] = _label_for(fav_b, vid_b, eng_b)
            except Exception as exc:
                logging.exception(f"Gendered TTS voice resolution failed: {exc}")
                use_local = False

        # Prepare utterance order with optional intros
        defaults = self._ollama_persona_defaults()
        persona_a = session.get("persona_a") or defaults[0]
        persona_b = session.get("persona_b") or defaults[1]
        a_display, _ = self._persona_parse(persona_a)
        b_display, _ = self._persona_parse(persona_b)
        sequence: List[Tuple[str, str]] = []  # (speaker, text)
        if intro_enabled:
            sequence.append(("A", f"Hello, my name is {a_display}."))
            sequence.append(("__pause__", str(intro_pause_ms)))
            sequence.append(("B", f"And my name is {b_display}."))
            sequence.append(("__pause__", str(intro_pause_ms)))
        for entry in transcript:
            sp = (entry or {}).get("speaker") or "A"
            tx = (entry or {}).get("text") or ""
            if isinstance(tx, str) and tx.strip():
                sequence.append((sp, tx))
                sequence.append(("__pause__", str(turn_pause_ms)))
        # Remove trailing pause
        if sequence and sequence[-1][0] == "__pause__":
            sequence.pop()
        if not sequence:
            raise RuntimeError("empty sequence")

        # Build combined AudioSegment
        combined = AudioSegment.silent(duration=0)
        tmp_segments: List[AudioSegment] = []
        for sp, content in sequence:
            if sp == "__pause__":
                try:
                    dur = max(0, int(content))
                except Exception:
                    dur = turn_pause_ms
                tmp_segments.append(AudioSegment.silent(duration=dur))
                continue
            if use_local and client:
                # Call hub synth
                try:
                    if sp == "A":
                        if fav_a:
                            data = await client.synthesise(content, favorite_slug=fav_a)
                        else:
                            data = await client.synthesise(content, voice_id=vid_a, engine=eng_a)
                    else:
                        if fav_b:
                            data = await client.synthesise(content, favorite_slug=fav_b)
                        else:
                            data = await client.synthesise(content, voice_id=vid_b, engine=eng_b)
                except Exception as e:
                    raise RuntimeError(f"hub synth failed: {e}")
                audio_bytes = data.get("audio_bytes")
                if not audio_bytes:
                    raise RuntimeError("hub returned no audio")
                try:
                    seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
                except Exception:
                    # fallback assume wav
                    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
                tmp_segments.append(seg)
            else:
                # OpenAI fallback via summarizer; single-voice limitation
                from youtube_summarizer import YouTubeSummarizer
                if not self.summarizer:
                    self.summarizer = YouTubeSummarizer()
                # Use different voice names if desired; here one voice for both as fallback
                out_dir = Path("exports"); out_dir.mkdir(exist_ok=True)
                out_path = out_dir / f"ai2ai_tmp_{int(time.time()*1000)}.mp3"
                res = await self.summarizer._generate_single_tts(content, str(out_path), provider="openai")
                if not res or not out_path.exists():
                    raise RuntimeError("openai tts failed")
                tmp_segments.append(AudioSegment.from_file(str(out_path)))
                try:
                    out_path.unlink()
                except Exception:
                    pass
                # Store info for caption (once)
                try:
                    if not session.get('ai2ai_tts_provider'):
                        session['ai2ai_tts_provider'] = 'openai'
                        # Default OpenAI voice in summarizer is 'fable' on 'tts-1'
                        session['ai2ai_tts_a_label'] = 'openai:tts-1:fable'
                        session['ai2ai_tts_b_label'] = 'openai:tts-1:fable'
                except Exception:
                    pass

        for seg in tmp_segments:
            combined += seg

        # Export final mp3
        out_dir = Path("exports"); out_dir.mkdir(exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        final_path = out_dir / f"ai2ai_chat_{ts}.mp3"
        combined.export(str(final_path), format="mp3", bitrate="192k")
        return str(final_path)
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in the bot."""
        logging.error(f"Exception while handling an update: {context.error}")
        
        # Try to send error message to user if possible
        try:
            if isinstance(update, Update) and update.effective_message:
                await update.effective_message.reply_text(
                    "❌ An error occurred while processing your request. Please try again."
                )
        except Exception:
            pass  # Don't let error handling cause more errors
    
    async def run(self):
        """Start the bot."""
        try:
            self.loop = asyncio.get_running_loop()
            self.application = Application.builder().token(self.token).build()
            self.setup_handlers()
            
            logging.info("🚀 Starting Telegram bot...")
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()

            # Post-startup restart confirmation (if requested before exit)
            try:
                from pathlib import Path as _P
                import json as _J
                p = _P('data')/ 'restart_notify.json'
                if p.exists():
                    obj = _J.loads(p.read_text(encoding='utf-8') or '{}')
                    chat_id = int(obj.get('chat_id') or 0)
                    if chat_id:
                        await self.application.bot.send_message(chat_id=chat_id, text="✅ Bot restarted and is back online.")
                    try:
                        p.unlink()
                    except Exception:
                        pass
            except Exception:
                pass

            stop_event = asyncio.Event()
            await stop_event.wait()
            
            logging.info("✅ Telegram bot is running and listening for messages")
            
            # Keep the bot running
            try:
                import signal
                stop_event = asyncio.Event()
                
                def signal_handler(signum, frame):
                    stop_event.set()
                
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
                
                await stop_event.wait()
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            logging.error(f"Error running bot: {e}")
            raise
        finally:
            if self.application:
                await self.application.stop()
    
    async def stop(self):
        """Stop the bot."""
        if self.application:
            logging.info("🛑 Stopping Telegram bot...")
            await self.application.stop()
            logging.info("✅ Telegram bot stopped")
