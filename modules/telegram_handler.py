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
from telegram.error import RetryAfter
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.constants import ParseMode

from export_utils import SummaryExporter
from modules.report_generator import JSONReportGenerator, create_report_from_youtube_summarizer
from modules import ledger
from modules.metrics import metrics
from modules.event_stream import emit_report_event
from modules.summary_variants import merge_summary_variants, normalize_variant_id

from youtube_summarizer import YouTubeSummarizer
from llm_config import llm_config
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
)
from modules.services import ollama_service, summary_service, tts_service
from modules.services import cloud_service
from modules.services.reachability import hub_ok as reach_hub_ok, hub_ollama_ok as reach_hub_ollama_ok
import hashlib
from pydub import AudioSegment


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
        # Interactive TTS sessions keyed by (chat_id, message_id)
        self.tts_sessions: Dict[tuple, Dict[str, Any]] = {}
        # Interactive summary sessions keyed by (chat_id, message_id)
        self.summary_sessions: Dict[tuple, Dict[str, Any]] = {}
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
        self.application.add_handler(CommandHandler("tts", self.tts_command))
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
            "/status - Check bot and API status\n\n"
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
        
        status_message = (
            "📊 Bot Status:\n\n"
            f"🤖 Telegram Bot: ✅ Running\n"
            f"🔍 Summarizer: {summarizer_status}\n"
            f"🧠 LLM: {llm_status}\n"
            f"👥 Authorized Users: {len(self.allowed_user_ids)}"
        )
        
        await update.message.reply_text(status_message)
    
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
        env_cloud = (os.getenv("QUICK_CLOUD_MODEL") or "").strip()
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
        cloud_model = (os.getenv("QUICK_CLOUD_MODEL") or "").strip()
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
        cached = self._cached_ollama_summarizer()
        cached_model = getattr(cached, "model", None) if cached else None

        ordered: List[str] = []
        for candidate in (cached_model, preferred):
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
        # For audio variants, prompt TTS preselection before starting summary
        summary_type = session.get("summary_type") or ""
        if isinstance(summary_type, str) and summary_type.startswith("audio"):
            await self._start_tts_preselect_flow(query, session, provider_key, selected_model)
        else:
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
            # For audio variants, surface combo buttons that will auto-run end-to-end
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
        elif callback_data.startswith("ollama_"):
            await self._handle_ollama_callback(query, callback_data)
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
            # Early TTS only for audio variants; otherwise run summary directly
            summary_type = (session.get("summary_type") or "").strip().lower()
            if summary_type.startswith("audio"):
                await self._start_tts_preselect_flow(query, session, provider_key, model_option)
            else:
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
                    cloud_model_slug = (os.getenv("QUICK_CLOUD_MODEL") or "").strip()
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
                    preselected = {
                        'auto_run': True,
                        'provider': 'openai',
                        'selected_voice': {'favorite_slug': None, 'voice_id': cloud_voice, 'engine': None},
                    }
                    self._store_tts_session(chat_id, message_id, preselected)
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
                    preselected = {
                        'auto_run': True,
                        'provider': 'local',
                        'selected_voice': selected_voice or {},
                    }
                    self._store_tts_session(chat_id, message_id, preselected)
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
        candidates = [p.strip() for p in providers_raw.split(',') if p.strip()] or ['cloud']

        # Choose first available provider from the preference list
        chosen = None
        # Prefer hub proxy for provider checks (we only support hub-based Ollama here)
        via_hub = bool(os.getenv('TTSHUB_API_BASE'))
        for p in candidates:
            if p == 'ollama':
                try:
                    if via_hub and reach_hub_ollama_ok():
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
                    model_slug = (os.getenv('QUICK_CLOUD_MODEL') or '').strip() or None
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
