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
import secrets
from typing import List, Dict, Any, Optional, Callable, Tuple, Awaitable
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
from modules import ledger, render_probe
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
)
from modules.telegram.ui.tts import (
    build_local_failure_keyboard as ui_build_local_failure_keyboard,
    build_tts_catalog_keyboard as ui_build_tts_catalog_keyboard,
    build_tts_keyboard as ui_build_tts_keyboard,
    gender_label as ui_gender_label,
    tts_prompt_text as ui_tts_prompt_text,
    tts_voice_label as ui_tts_voice_label,
)
from modules.ollama_client import (
    OllamaClientError,
    get_models as ollama_get_models,
    pull as ollama_pull,
)
from modules.services import ollama_service, summary_service, tts_service
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
        # Ollama chat sessions keyed by chat_id
        self.ollama_sessions: Dict[int, Dict[str, Any]] = {}
        # Pending delete tokens mapping short callback tokens to full report IDs
        self._pending_delete_tokens: Dict[str, str] = {}
        
        # YouTube URL regex pattern
        self.youtube_url_pattern = re.compile(
            r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})'
        )
        self.reddit_url_pattern = re.compile(
            r'(https?://(?:www\.)?(?:reddit\.com|redd\.it)/[^\s]+)',
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

            await update.message.reply_text(
                message_text,
                reply_markup=reply_markup
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

            await update.message.reply_text(
                message_text,
                reply_markup=reply_markup
            )
            return

        # Check for generic web URLs
        web_url = self._extract_web_url(message_text)
        if web_url:
            web_url = web_url.strip()
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

            await update.message.reply_text(
                message_text,
                reply_markup=reply_markup
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
        return ui_build_models_keyboard(
            models=models,
            page=page,
            page_size=page_size,
            session=sess,
            categories=categories,
            persona_parse=self._persona_parse,
            ai2ai_default_models=self._ollama_ai2ai_default_models,
            allow_same_models=allow_same,
        )

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
        line = "--------------------------------------------------------------------------"
        # Determine mode
        a = session.get('ai2ai_model_a')
        b = session.get('ai2ai_model_b')
        stream_on = True  # default on
        mode_key = session.get('mode') or ('ai-ai' if (a and b) else 'ai-human')
        mode_label = 'AI↔AI' if mode_key == 'ai-ai' else 'AI→Human'
        parts = [
            line,
            f"🤖 Ollama Chat · Mode: {mode_label} · Streaming: ON",
        ]
        if a and b:
            if not (session.get('persona_a') and session.get('persona_b')):
                rand_a, rand_b = self._ollama_persona_random_pair()
                session.setdefault('persona_a', rand_a)
                session.setdefault('persona_b', rand_b)
                # Populate derived fields
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
            line_a = f"A: {a} · {pa_disp}"
            line_b = f"B: {b} · {pb_disp}"
            if cat_a:
                line_a = f"{line_a} ({cat_a})"
            if cat_b:
                line_b = f"{line_b} ({cat_b})"
            parts.append(line_a)
            parts.append(line_b)
            turns = session.get('ai2ai_turns_left')
            if isinstance(turns, int):
                parts.append(f"Turns remaining: {turns}")
        else:
            model = session.get('model') or '—'
            parts.append(f"Model: {model}")
            persona_single = session.get("persona_single")
            if persona_single:
                cat_single = session.get("persona_single_category")
                single_disp, _ = self._persona_parse(persona_single)
                persona_line = f"Persona: {single_disp}"
                if cat_single:
                    persona_line = f"{persona_line} ({cat_single})"
                parts.append(persona_line)
        parts.append(line)
        if mode_key == 'ai-ai':
            if a and b:
                parts.append("Type a prompt to begin the AI↔AI exchange. Use Options to adjust turns.")
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
        try:
            raw = ollama_get_models()
            models = self._ollama_models_list(raw)
            if not models:
                await update.message.reply_text("⚠️ No models available on the hub.")
                return
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
            self.ollama_sessions[update.effective_chat.id] = sess
            kb = self._build_ollama_models_keyboard(models, 0, session=sess)
            # Render dynamic status above the picker
            text = self._ollama_status_text(sess)
            await update.message.reply_text(text, reply_markup=kb)
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
            await update.message.reply_text("⚠️ AI↔AI exchange already running. Try again after the current turn or use /stop.")
            return
        await update.message.reply_text("💬 New AI↔AI turn coming up…")

    async def _ollama_handle_user_text(self, update: Update, session: Dict[str, Any], text: str):
        await ollama_service.handle_user_text(self, update, session, text)

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
                ai2ai_row = [InlineKeyboardButton("⏭️ Continue exchange", callback_data="ollama_ai2ai:continue")]
            rows = [
                [InlineKeyboardButton(f"{mark_ai} AI→Human", callback_data="ollama_mode:ai-human"), InlineKeyboardButton(f"{mark_ai2ai} AI↔AI", callback_data="ollama_mode:ai-ai")],
                [InlineKeyboardButton(f"{mark_stream} Streaming", callback_data="ollama_toggle:stream")],
            ]
            if ai2ai_row:
                rows.append(ai2ai_row)
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more:{session.get('page', 0)}")])
            kb = InlineKeyboardMarkup(rows)
            try:
                await query.edit_message_text("⚙️ Ollama options", reply_markup=kb)
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
            ai2ai_model_a = session.get("ai2ai_model_a") or session.get("model") or "—"
            ai2ai_model_b = session.get("ai2ai_model_b") or session.get("model") or "—"
            ai2ai_row = [InlineKeyboardButton("▶️ Start AI↔AI", callback_data="ollama_ai2ai:start")] if (mode == "ai-ai" and not ai2ai_active) else []
            if mode == "ai-ai" and ai2ai_active:
                ai2ai_row = [InlineKeyboardButton("⏭️ Continue exchange", callback_data="ollama_ai2ai:continue")]
            rows = [
                [InlineKeyboardButton(f"{mark_ai} AI→Human", callback_data="ollama_mode:ai-human"), InlineKeyboardButton(f"{mark_ai2ai} AI↔AI", callback_data="ollama_mode:ai-ai")],
                [InlineKeyboardButton(f"{mark_stream} Streaming", callback_data="ollama_toggle:stream")],
            ]
            if mode == "ai-ai":
                rows.append([
                    InlineKeyboardButton(f"A: {ai2ai_model_a}", callback_data="ollama_ai2ai:pick_a"),
                    InlineKeyboardButton(f"B: {ai2ai_model_b}", callback_data="ollama_ai2ai:pick_b"),
                ])
            if ai2ai_row:
                rows.append(ai2ai_row)
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more:{session.get('page', 0)}")])
            kb = InlineKeyboardMarkup(rows)
            await query.edit_message_text("⚙️ Ollama options", reply_markup=kb)
            await query.answer("Options")
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
                await self._show_main_summary_options(query)
                return
            
            parts = raw.split(":", 1)
            summary_type = parts[0]  # "audio-fr" / "audio-es" / "audio"
            proficiency_level = parts[1] if len(parts) == 2 else None
            
            # If French/Spanish audio without level specified, show level picker
            if summary_type in ("audio-fr", "audio-es") and proficiency_level is None:
                await self._show_proficiency_selector(query, summary_type)
                return
            
            # Process with proficiency level (None for regular summaries)
            await self._process_content_summary(query, summary_type, user_name, proficiency_level)
        
        elif callback_data.startswith("tts_"):
            await self._handle_tts_callback(query, callback_data)
        elif callback_data.startswith("ollama_"):
            await self._handle_ollama_callback(query, callback_data)
        
        # Handle delete requests
        elif callback_data.startswith('delete_'):
            token = callback_data.replace('delete_', '', 1)
            report_id = self._get_delete_target(token) or token
            display_id = self._escape_markdown(self._shorten_report_id(report_id))

            keyboard = InlineKeyboardMarkup([[ 
                InlineKeyboardButton("✅ Yes, delete", callback_data=f"confirm_del_{token}"),
                InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_del_{token}")
            ]])
            await query.edit_message_text(
                f"⚠️ Delete this summary?\n\nID: `{display_id}`\nThis removes it from both Telegram and the Dashboard.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )

        elif callback_data.startswith('cancel_del_'):
            token = callback_data.replace('cancel_del_', '', 1)
            report_id = self._consume_delete_token(token)
            display_id = self._shorten_report_id(report_id) if report_id else None
            message = "🚫 Delete cancelled."
            if display_id:
                message = f"🚫 Delete cancelled for `{self._escape_markdown(display_id)}`."
            await query.edit_message_text(message, parse_mode=ParseMode.MARKDOWN)

        elif callback_data == 'cancel_del':
            await query.edit_message_text("🚫 Delete cancelled.")

        elif callback_data.startswith('confirm_del_'):
            token = callback_data.replace('confirm_del_', '', 1)
            report_id = self._consume_delete_token(token) or token

            # Helper function to delete from Render
            def delete_from_render(rid):
                if requests is None:
                    return False, "Dashboard (requests library not available)"
                    
                dashboard_url = os.getenv('DASHBOARD_URL', 'https://ytv2-dashboard-postgres.onrender.com')
                dashboard_token = os.getenv('DASHBOARD_TOKEN', '')
                url = f"{dashboard_url}/api/delete/{urllib.parse.quote(rid, safe='')}"
                headers = {}
                if dashboard_token:
                    headers["Authorization"] = f"Bearer {dashboard_token}"
                
                # Try twice with small backoff
                for attempt in range(2):
                    try:
                        r = requests.delete(url, headers=headers, timeout=8)
                        if r.status_code in (200, 404):  # 404 = already gone = success
                            return True, "Dashboard"
                        return False, f"Dashboard (HTTP {r.status_code})"
                    except Exception as e:
                        if attempt == 0:
                            time.sleep(0.6)
                        else:
                            return False, f"Dashboard (network error)"
                return False, "Dashboard (timeout)"
            
            # Delete from both systems
            render_ok, render_msg = delete_from_render(report_id)
            
            # Delete from NAS
            nas_ok = False
            nas_path = Path(f'/app/data/reports/{report_id}.json')
            if nas_path.exists():
                try:
                    nas_path.unlink()
                    nas_ok = True
                    nas_msg = "NAS"
                except Exception as e:
                    nas_msg = f"NAS (error: {e})"
            else:
                nas_ok = True  # Not found = already gone = success
                nas_msg = "NAS (already gone)"
            
            # Build result message
            if render_ok and nas_ok:
                result = "✅ Summary deleted successfully"
            elif render_ok:
                result = f"✅ Deleted from Dashboard\n⚠️ {nas_msg}"
            elif nas_ok:
                result = f"⚠️ Failed: {render_msg}\n✅ Deleted from NAS"
            else:
                result = f"❌ Delete failed:\n• {render_msg}\n• {nas_msg}"
            
            # Update message and remove buttons
            await query.edit_message_text(result)
            await query.answer("Deleted" if (render_ok or nas_ok) else "Failed")

        elif callback_data == 'cancel_del':
            # Just remove the confirmation buttons
            await query.edit_message_reply_markup(reply_markup=None)
            await query.answer("Cancelled")
        
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
            row.append(InlineKeyboardButton("🏠 Local TTS hub", callback_data="tts_provider:local"))
        else:
            row.append(InlineKeyboardButton("🏠 Local TTS hub", callback_data="tts_provider:local"))
        row.append(InlineKeyboardButton("☁️ OpenAI TTS", callback_data="tts_provider:openai"))
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

        favorites_list: List[Dict[str, Any]] = []
        try:
            favorites_list = await client.fetch_favorites(tag="telegram")
            if not favorites_list:
                favorites_list = await client.fetch_favorites()
        except Exception as e:
            logging.warning(f"TTS favorites fetch failed: {e}")
            favorites_list = []

        if catalog and catalog.get('voices'):
            session_payload = {
                "text": speak_text,
                "catalog": catalog,
                "selected_gender": None,
                "selected_family": None,
                "last_voice": None,
                "favorites": favorites_list,
                "voice_mode": 'favorites' if favorites_list else 'all',
                "tts_base": client.base_api_url,
                "mode": "oneoff_tts",
            }
            prompt = self._tts_prompt_text(speak_text, catalog=catalog)
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
            short_key = f"v{i}"
            entry = {
                'label': (fav.get('label') or slug),
                'voice': None,
                'voiceId': fav.get('voiceId'),
                'engine': fav.get('engine'),
                'favoriteSlug': slug,
            }
            voice_lookup[short_key] = entry
            voice_lookup[f"fav|{slug}"] = entry

        session_payload = {
            "text": speak_text,
            "favorites": favorites,
            "tts_base": client.base_api_url,
            "last_voice": None,
            "selected_family": None,
            "voice_lookup": voice_lookup,
            "mode": "oneoff_tts",
        }
        self._store_tts_session(prompt_message.chat_id, prompt_message.message_id, session_payload)

    async def _handle_tts_callback(self, query, callback_data: str):
        await tts_service.handle_callback(self, query, callback_data)

    # Quiz helper utilities live in modules.services.summary_service

    def _register_delete_token(self, report_id: str) -> str:
        token = secrets.token_hex(4)
        while token in self._pending_delete_tokens:
            token = secrets.token_hex(4)
        self._pending_delete_tokens[token] = report_id
        return token

    def _get_delete_target(self, token: str) -> Optional[str]:
        return self._pending_delete_tokens.get(token)

    def _consume_delete_token(self, token: str) -> Optional[str]:
        return self._pending_delete_tokens.pop(token, None)

    @staticmethod
    def _shorten_report_id(report_id: str) -> str:
        if not report_id:
            return "(unknown)"
        if len(report_id) <= 40:
            return report_id
        return f"{report_id[:24]}…{report_id[-8:]}"

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
    
    async def _process_content_summary(self, query, summary_type: str, user_name: str, proficiency_level: str = None):
        await summary_service.process_content_summary(self, query, summary_type, user_name, proficiency_level)

    async def _send_formatted_response(self, query, result: Dict[str, Any], summary_type: str, export_info: Dict = None):
        await summary_service.send_formatted_response(self, query, result, summary_type, export_info)

    async def _prepare_tts_generation(self, query, result: Dict[str, Any], summary_text: str, summary_type: str):
        await summary_service.prepare_tts_generation(self, query, result, summary_text, summary_type)

    async def _prompt_tts_provider(self, query, session_payload: Dict[str, Any], title: str) -> None:
        await tts_service.prompt_provider(self, query, session_payload, title)
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
        delete_token = self._register_delete_token(report_id)

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
                InlineKeyboardButton("➕ Add Variant", callback_data="summarize_back_to_main"),
                InlineKeyboardButton("🗑️ Delete…", callback_data=f"delete_{delete_token}")
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
