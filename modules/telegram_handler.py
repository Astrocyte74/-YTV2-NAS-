"""
Telegram Bot Handler Module

This module contains the YouTubeTelegramBot class extracted from the monolithic file.
It handles all Telegram bot interactions without embedded HTML generation.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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
from nas_sync import upload_to_render, dual_sync_upload
from modules.render_api_client import create_client_from_env as create_render_client
from modules.tts_hub import (
    TTSHubClient,
    LocalTTSUnavailable,
    accent_family_label,
    available_accent_families,
    filter_catalog_voices,
)
from modules.tts_queue import enqueue as enqueue_tts_job
from modules import render_probe
from modules.ollama_client import (
    OllamaClientError,
    get_models as ollama_get_models,
    chat as ollama_chat,
    chat_stream as ollama_chat_stream,
    pull as ollama_pull,
)
import hashlib
import unicodedata


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
        self._render_client = None
        
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
        base, _, suffix = variant.partition(':')
        base_label = self.VARIANT_LABELS.get(base, base.replace('-', ' ').title())
        if suffix:
            suffix_clean = suffix.replace('-', ' ').replace('_', ' ').title()
            return f"{base_label} ({suffix_clean})"
        return base_label

    def _build_summary_keyboard(self, existing_variants: Optional[List[str]] = None, video_id: Optional[str] = None):
        existing_variants = existing_variants or []
        existing_bases = {variant.split(':', 1)[0] for variant in existing_variants}

        def label_for(variant_key: str) -> str:
            label = self.VARIANT_LABELS.get(variant_key, variant_key.replace('-', ' ').title())
            return f"{label} ✅" if variant_key in existing_bases else label

        keyboard = [
            [
                InlineKeyboardButton(label_for('comprehensive'), callback_data="summarize_comprehensive"),
                InlineKeyboardButton(label_for('bullet-points'), callback_data="summarize_bullet-points")
            ],
            [
                InlineKeyboardButton(label_for('key-insights'), callback_data="summarize_key-insights"),
                InlineKeyboardButton(label_for('audio'), callback_data="summarize_audio")
            ],
            [
                InlineKeyboardButton(label_for('audio-fr'), callback_data="summarize_audio-fr"),
                InlineKeyboardButton(label_for('audio-es'), callback_data="summarize_audio-es")
            ]
        ]

        if existing_bases and video_id:
            dashboard_url = (
                os.getenv('DASHBOARD_URL')
                or os.getenv('POSTGRES_DASHBOARD_URL')
                or 'https://ytv2-dashboard-postgres.onrender.com'
            )
            if dashboard_url:
                report_id_encoded = urllib.parse.quote(video_id, safe='')
                keyboard.append([
                    InlineKeyboardButton("📄 Open summary", url=f"{dashboard_url}#report={report_id_encoded}")
                ])

        return InlineKeyboardMarkup(keyboard)

    def _existing_variants_message(self, content_id: str, variants: List[str], source: str = "youtube") -> str:
        if not variants:
            prompts = {
                "youtube": "🎬 Processing YouTube video...\n\nChoose your summary type:",
                "reddit": "🧵 Processing Reddit thread...\n\nChoose your summary type:",
                "web": "📰 Processing web article...\n\nChoose your summary type:",
            }
            return prompts.get(source, prompts["youtube"])

        variants_sorted = sorted(variants)
        noun = {
            "youtube": "video",
            "reddit": "thread",
            "web": "article",
        }.get(source, "item")
        lines = [f"✅ Existing summaries for this {noun}:"]
        lines.extend(f"• {self._friendly_variant_label(variant)}" for variant in variants_sorted)
        lines.append("\nRe-run a variant below or open the summary card.")
        return "\n".join(lines)

    async def _send_existing_summary_notice(self, query, video_id: str, summary_type: str):
        source = self._current_source()
        variants = self._discover_summary_types(video_id)
        message_lines = [
            f"✅ {self._friendly_variant_label(summary_type)} is already on the dashboard."
        ]
        if variants:
            message_lines.append("\nAvailable variants:")
            message_lines.extend(f"• {self._friendly_variant_label(variant)}" for variant in sorted(variants))
        message_lines.append("\nOpen the summary or re-run a variant below.")

        normalized_id = self._normalize_content_id(video_id)
        reply_markup = self._build_summary_keyboard(variants, normalized_id)
        await query.edit_message_text(
            "\n".join(message_lines),
            reply_markup=reply_markup
        )

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
        """Generate TTS audio in a background thread to avoid blocking."""
        if not summary_text:
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_filename = f"audio_{video_id}_{timestamp}.mp3"

        loop = asyncio.get_running_loop()

        def _run() -> Optional[str]:
            return asyncio.run(self.summarizer.generate_tts_audio(summary_text, audio_filename, str(json_path)))

        audio_filepath = await loop.run_in_executor(None, _run)
        if audio_filepath and Path(audio_filepath).exists():
            metrics.record_tts(True)
        else:
            metrics.record_tts(False)
        return audio_filepath if audio_filepath else None

    async def _reprocess_single_summary(
        self,
        video_id: str,
        video_url: str,
        summary_type: str,
        ledger_entry: Optional[Dict[str, Any]] = None,
        force: bool = False,
        regenerate_audio: bool = True,
    ) -> Dict[str, Any]:
        """Reprocess a single summary variant headlessly."""

        ledger_entry = dict(ledger_entry or {})
        proficiency = ledger_entry.get('proficiency')
        job_result: Dict[str, Any] = {
            'summary_type': summary_type,
            'status': 'pending',
        }

        try:
            result = await self.summarizer.process_video(
                video_url,
                summary_type=summary_type,
                proficiency_level=proficiency,
            )

            if not result or result.get('error'):
                job_result.update({'status': 'error', 'error': result.get('error') if isinstance(result, dict) else 'unknown'})
                metrics.record_reprocess_result(False)
                return job_result

            report_dict = create_report_from_youtube_summarizer(result)

            # Merge with existing variants (if any) before persisting
            target_basename = self.json_exporter._generate_filename(report_dict)
            candidate_name = target_basename if target_basename.endswith('.json') else f"{target_basename}.json"
            candidate_path = Path(self.json_exporter.reports_dir) / candidate_name

            existing_report = None
            if candidate_path.exists():
                try:
                    with open(candidate_path, 'r', encoding='utf-8') as existing_file:
                        existing_report = json.load(existing_file)
                except Exception as load_error:
                    logger.warning(
                        "⚠️  Failed to load existing report for variant merge (%s): %s",
                        candidate_path,
                        load_error,
                    )

            report_dict = merge_summary_variants(
                new_report=report_dict,
                requested_variant=summary_type,
                existing_report=existing_report,
            )

            json_path = Path(
                self.json_exporter.save_report(
                    report_dict,
                    filename=target_basename,
                    overwrite=True,
                )
            )
            job_result['report_path'] = str(json_path)

            summary_meta = report_dict.get('summary') or {}
            summary_text = summary_meta.get('summary') or ''

            audio_path = None
            is_audio = summary_type.startswith('audio')

            if is_audio:
                if regenerate_audio:
                    audio_path = await self._generate_tts_audio_file(summary_text, video_id, json_path)
                else:
                    existing_mp3 = ledger_entry.get('mp3')
                    if existing_mp3 and Path(existing_mp3).exists():
                        audio_path = existing_mp3
                if audio_path:
                    job_result['audio_path'] = audio_path

            # Sync with dashboard (include audio if available)
            try:
                audio_path_obj = Path(audio_path) if audio_path else None
                sync_results = dual_sync_upload(json_path, audio_path_obj)
            except Exception as sync_error:
                job_result.update({'status': 'error', 'error': str(sync_error)})
                metrics.record_reprocess_result(False)
                return job_result

            sqlite_ok = bool(sync_results.get('sqlite', {}).get('report')) if isinstance(sync_results, dict) else False
            postgres_ok = bool(sync_results.get('postgres', {}).get('report')) if isinstance(sync_results, dict) else False
            targets = []
            if sqlite_ok:
                targets.append('sqlite')
            if postgres_ok:
                targets.append('postgres')
            job_result['sync_targets'] = targets

            success = bool(targets)
            job_result['status'] = 'ok' if success else 'error'
            metrics.record_reprocess_result(success)

            ledger_entry.update(
                {
                    'stem': json_path.stem,
                    'json': str(json_path),
                    'synced': success,
                    'last_synced': datetime.now().isoformat(),
                    'reprocessed_at': datetime.now().isoformat(),
                }
            )
            if is_audio and audio_path:
                ledger_entry['mp3'] = audio_path
            if targets:
                ledger_entry['sync_targets'] = targets
            ledger.upsert(ledger_id, summary_type, ledger_entry)

            emit_report_event(
                'reprocess-complete',
                {
                    'video_id': video_id,
                    'summary_type': summary_type,
                    'status': job_result['status'],
                    'targets': targets,
                },
            )

            return job_result

        except Exception as e:
            logging.exception(f"Reprocess failure for {video_id}:{summary_type}")
            job_result.update({'status': 'error', 'error': str(e)})
            metrics.record_reprocess_result(False)
            emit_report_event(
                'reprocess-error',
                {
                    'video_id': video_id,
                    'summary_type': summary_type,
                    'error': str(e),
                },
            )
            return job_result

    async def reprocess_video(
        self,
        video_id: str,
        summary_types: Optional[List[str]] = None,
        force: bool = False,
        regenerate_audio: bool = True,
        video_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Public entry point to reprocess summaries for an existing video."""

        normalized_id = self._normalize_content_id(video_id)
        summary_types = summary_types or self._discover_summary_types(normalized_id)

        if not summary_types:
            metrics.record_reprocess_result(False)
            emit_report_event(
                'reprocess-error',
                {
                    'video_id': normalized_id,
                    'error': 'no-summary-types',
                },
            )
            raise ValueError(f"No summary types found for video {normalized_id}")

        resolved_url = self._resolve_video_url(normalized_id, provided_url=video_url)
        if not resolved_url:
            metrics.record_reprocess_result(False)
            emit_report_event(
                'reprocess-error',
                {
                    'video_id': normalized_id,
                    'error': 'missing-url',
                },
            )
            raise ValueError(f"Could not resolve URL for video {normalized_id}")

        metrics.record_reprocess_request(len(summary_types))
        emit_report_event(
            'reprocess-requested',
            {
                'video_id': normalized_id,
                'summary_types': summary_types,
            },
        )

        ledger_data = ledger.list_all()
        results = []
        for summary_type in summary_types:
            ledger_entry = ledger_data.get(f"{normalized_id}:{summary_type}")
            job_result = await self._reprocess_single_summary(
                normalized_id,
                resolved_url,
                summary_type,
                ledger_entry=ledger_entry,
                force=force,
                regenerate_audio=regenerate_audio,
            )
            results.append(job_result)
            if job_result.get('status') == 'ok':
                ledger_data[f"{normalized_id}:{summary_type}"] = ledger.get(normalized_id, summary_type)

        failures = sum(1 for r in results if r.get('status') != 'ok')

        return {
            'video_id': normalized_id,
            'summary_types': summary_types,
            'results': results,
            'failures': failures,
        }
    
    def setup_handlers(self):
        """Set up bot command and message handlers."""
        if not self.application:
            return
            
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("tts", self.tts_command))
        # Ollama chat commands (alias: /o)
        self.application.add_handler(CommandHandler("ollama", self.ollama_command))
        self.application.add_handler(CommandHandler("o", self.ollama_command))
        self.application.add_handler(CommandHandler("ollama_stop", self.ollama_stop_command))
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
            if session and session.get("active"):
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

    def _build_ollama_models_keyboard(self, models: List[str], page: int = 0, page_size: int = 9) -> InlineKeyboardMarkup:
        start = page * page_size
        end = start + page_size
        subset = models[start:end]
        rows: List[List[InlineKeyboardButton]] = []
        row: List[InlineKeyboardButton] = []
        for name in subset:
            label = name
            if len(label) > 28:
                label = f"{label[:25]}…"
            row.append(InlineKeyboardButton(label, callback_data=f"ollama_model:{name}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        nav: List[InlineKeyboardButton] = []
        if end < len(models):
            nav.append(InlineKeyboardButton("➡️ More", callback_data=f"ollama_more:{page+1}"))
        if page > 0:
            nav.insert(0, InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more:{page-1}"))
        if nav:
            rows.append(nav)
        # Options only for AI↔AI advanced settings (appears when both models selected)
        rows.append([InlineKeyboardButton("❌ Close", callback_data="ollama_cancel")])
        return InlineKeyboardMarkup(rows)

    def _ollama_stream_default(self) -> bool:
        val = os.getenv('OLLAMA_STREAM_DEFAULT', '1').lower()
        return val not in ('0', 'false', 'no')

    def _ollama_status_text(self, session: Dict[str, Any]) -> str:
        line = "-" * 54
        # Determine mode
        a = session.get('ai2ai_model_a')
        b = session.get('ai2ai_model_b')
        stream_on = True  # default on
        mode = 'AI↔AI' if (a and b) else 'AI→Human'
        parts = [
            line,
            f"🤖 Ollama Chat · Mode: {mode} · Streaming: ON",
        ]
        if a and b:
            pa = session.get('persona_a') or 'Persona A'
            pb = session.get('persona_b') or 'Persona B'
            parts.append(f"A: {a} · {pa}")
            parts.append(f"B: {b} · {pb}")
            turns = session.get('ai2ai_turns_left')
            if isinstance(turns, int):
                parts.append(f"Turns remaining: {turns}")
        else:
            model = session.get('model') or '—'
            parts.append(f"Model: {model}")
        parts.append(line)
        parts.append("Select a model below or type to chat.")
        return "\n".join(parts)

    def _build_ollama_models_keyboard_ai2ai(self, models: List[str], slot: str, page: int = 0, page_size: int = 9) -> InlineKeyboardMarkup:
        start = page * page_size
        end = start + page_size
        subset = models[start:end]
        rows: List[List[InlineKeyboardButton]] = []
        row: List[InlineKeyboardButton] = []
        for name in subset:
            label = name
            if len(label) > 28:
                label = f"{label[:25]}…"
            row.append(InlineKeyboardButton(label, callback_data=f"ollama_ai2ai_set:{slot}:{name}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        nav: List[InlineKeyboardButton] = []
        if end < len(models):
            nav.append(InlineKeyboardButton("➡️ More", callback_data=f"ollama_more_ai2ai:{slot}:{page+1}"))
        if page > 0:
            nav.insert(0, InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more_ai2ai:{slot}:{page-1}"))
        if nav:
            rows.append(nav)
        rows.append([InlineKeyboardButton("❌ Cancel", callback_data="ollama_cancel")])
        return InlineKeyboardMarkup(rows)

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
            kb = self._build_ollama_models_keyboard(models, 0)
            # Initialize session with defaults (streaming on)
            self.ollama_sessions[update.effective_chat.id] = {
                "active": False,
                "models": models,
                "page": 0,
                "stream": True if self._ollama_stream_default() else False,
                "history": [],
            }
            # Render dynamic status above the picker
            text = self._ollama_status_text(self.ollama_sessions[update.effective_chat.id])
            await update.message.reply_text(text, reply_markup=kb)
        except Exception as exc:
            await update.message.reply_text(f"❌ Ollama hub error: {exc}")

    async def ollama_stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        self.ollama_sessions.pop(chat_id, None)
        await update.message.reply_text("🛑 Closed Ollama chat session.")

    async def _ollama_handle_user_text(self, update: Update, session: Dict[str, Any], text: str):
        chat_id = update.effective_chat.id
        model = session.get("model")
        if not model:
            await update.message.reply_text("⚠️ No model selected. Send /ollama to choose one.")
            return
        # If AI↔AI is active (two models picked), treat user text as topic and run a turn
        if session.get("ai2ai_model_a") and session.get("ai2ai_model_b"):
            session["ai2ai_active"] = True
            session["topic"] = text
            # Initialize turns if not present
            if not isinstance(session.get("ai2ai_turns_left"), int):
                try:
                    session["ai2ai_turns_left"] = int(os.getenv('OLLAMA_AI2AI_TURNS', '10'))
                except Exception:
                    session["ai2ai_turns_left"] = 10
            await update.message.reply_text("🤝 Starting AI↔AI exchange…")
            await self._ollama_ai2ai_continue(update.effective_chat.id)
            # Decrement one turn (A+B considered one combined turn)
            turns = max(0, int(session.get("ai2ai_turns_left") or 0) - 1)
            session["ai2ai_turns_left"] = turns
            self.ollama_sessions[update.effective_chat.id] = session
            if turns > 0:
                await update.message.reply_text(f"⏭️ {turns} turns remaining. Use Options → Continue exchange to proceed.")
            else:
                await update.message.reply_text("✅ AI↔AI session complete.")
            return
        # Build chat payload
        history = session.get("history") or []
        messages = history + [{"role": "user", "content": text}]
        if bool(session.get("stream")):
            # Streaming reply
            try:
                final_text = await self._ollama_stream_chat(update, model, messages)
            except Exception as exc:
                await update.message.reply_text(f"❌ Stream error: {str(exc)[:200]}")
                return
            # Update history
            session["history"] = (messages + [{"role": "assistant", "content": final_text}])[-16:]
            self.ollama_sessions[chat_id] = session
            return

        loop = asyncio.get_running_loop()

        def _call():
            try:
                return ollama_chat(messages, model, stream=False)  # returns dict
            except Exception as e:
                return {"error": str(e)}
        # Try to indicate typing; ignore if unsupported
        try:
            from telegram.constants import ChatAction
            app = getattr(self, 'application', None)
            if app and getattr(app, 'bot', None):
                await app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        logging.info(f"Ollama chat: model={model} text_len={len(text)}")
        resp = await loop.run_in_executor(None, _call)
        try:
            import json as _json
            if isinstance(resp, dict):
                keys = list(resp.keys())
                logging.info(f"Ollama resp keys: {keys[:8]}")
                msg = resp.get("message")
                if isinstance(msg, dict):
                    logging.info(f"Ollama message keys: {list(msg.keys())}")
            else:
                logging.info(f"Ollama resp type: {type(resp)}")
        except Exception:
            pass
        if isinstance(resp, dict) and resp.get("error"):
            err = str(resp["error"]).lower()
            # Offer pull if model appears missing
            if ("404" in err or "not found" in err or "no such model" in err) and model:
                # Persist last user text to retry after pull
                session["last_user"] = text
                self.ollama_sessions[chat_id] = session
                kb = InlineKeyboardMarkup([
                    [InlineKeyboardButton(f"📥 Pull {model}", callback_data=f"ollama_pull:{model}")],
                    [InlineKeyboardButton("❌ Cancel", callback_data="ollama_cancel")],
                ])
                await update.message.reply_text(
                    f"⚠️ Model `{self._escape_markdown(model)}` is not available on the hub. Pull it now?",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=kb,
                )
                return
            await update.message.reply_text(f"❌ Ollama error: {resp['error'][:200]}")
            return
        # Extract reply text robustly
        reply_text = None
        if isinstance(resp, dict):
            # 1) Common shapes
            val = resp.get("response")
            if isinstance(val, str) and val.strip():
                reply_text = val
            if reply_text is None:
                msg = resp.get("message")
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str) and c.strip():
                        reply_text = c
                    elif isinstance(c, list):
                        # Some implementations surface list of segments
                        parts = []
                        for seg in c:
                            if isinstance(seg, dict):
                                t = seg.get("text") or seg.get("content")
                                if isinstance(t, str) and t.strip():
                                    parts.append(t)
                        if parts:
                            reply_text = "\n".join(parts)
            # 2) Fallback: messages array
            if reply_text is None:
                msgs = resp.get("messages")
                if isinstance(msgs, list):
                    for m in reversed(msgs):
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            c = m.get("content")
                            if isinstance(c, str) and c.strip():
                                reply_text = c
                                break
            # 3) Last resort: compact JSON snippet for debugging
            if reply_text is None:
                import json as _json
                snippet = _json.dumps({k: resp[k] for k in list(resp.keys())[:8]})[:380]
                reply_text = f"(No response)\n<pre>{snippet}</pre>"
                try:
                    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
                    return
                except Exception:
                    # Fall through to plain text
                    reply_text = f"(No response)\n{snippet}"
        if not reply_text:
            reply_text = "(No response)"
        # Send safely within Telegram length limits
        await self._send_long_text_reply(update, reply_text)
        # Update conversation history (keep it short)
        session["history"] = (messages + [{"role": "assistant", "content": reply_text}])[-16:]
        self.ollama_sessions[chat_id] = session

    async def _ollama_stream_chat(self, update: Update, model: str, messages: List[Dict[str, str]]) -> str:
        """Stream tokens from the hub and live-edit a single message. Returns final text."""
        from html import escape as _esc
        chat_id = update.effective_chat.id
        # Seed message
        msg = await update.message.reply_text("⏳ …")
        message_id = msg.message_id
        app = getattr(self, 'application', None)
        bot = getattr(app, 'bot', None)
        loop = asyncio.get_running_loop()

        final_text = {"buf": ""}

        def _run_stream():
            import time, logging
            logging.info(f"Ollama streaming start: model={model} msgs={len(messages)}")
            last = 0.0
            try:
                for data in ollama_chat_stream(messages, model):
                    if not isinstance(data, dict):
                        continue
                    # First event is usually {"status":"starting"}
                    if data.get("status") == "starting":
                        continue
                    chunk = data.get("response")
                    if isinstance(chunk, str) and chunk:
                        final_text["buf"] += chunk
                    else:
                        # Some hubs send tokens in message.content
                        msg = data.get("message")
                        if isinstance(msg, dict):
                            c = msg.get("content")
                            if isinstance(c, str) and c:
                                final_text["buf"] += c
                    # Throttle edits
                    now = time.time()
                    if (now - last > 0.4) and final_text["buf"]:
                        last = now
                        txt = final_text["buf"]
                        async def _upd():
                            try:
                                await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=txt[-4000:])
                            except Exception:
                                pass
                        asyncio.run_coroutine_threadsafe(_upd(), loop)
                    if data.get("done"):
                        # Final update (guard empty)
                        txt = final_text["buf"] or "✅ Done"
                        async def _final():
                            try:
                                await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=txt[-4000:])
                            except Exception:
                                pass
                        asyncio.run_coroutine_threadsafe(_final(), loop)
                        break
            except Exception as e:
                async def _err():
                    try:
                        await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=f"❌ Stream error: {e}")
                    except Exception:
                        pass
                asyncio.run_coroutine_threadsafe(_err(), loop)

        await loop.run_in_executor(None, _run_stream)
        return final_text["buf"]

    async def _ollama_ai2ai_continue(self, chat_id: int):
        session = self.ollama_sessions.get(chat_id) or {}
        model_a = session.get("ai2ai_model_a") or session.get("model")
        model_b = session.get("ai2ai_model_b") or session.get("model")
        if not model_a or not model_b:
            try:
                await self.application.bot.send_message(chat_id=chat_id, text="⚠️ Pick models for A and B in Options")
            except Exception:
                pass
            return
        persona_a = session.get("persona_a", "Albert Einstein")
        persona_b = session.get("persona_b", "Isaac Newton")
        topic = session.get("topic", "The nature of space and time")
        # Turn A
        a_messages = [
            {"role": "system", "content": f"You are {persona_a}. Respond concisely."},
            {"role": "user", "content": f"Debate topic: {topic}. Present your view."},
        ]
        # Create a tiny wrapper update-like object for streaming helper
        from types import SimpleNamespace
        class U:
            def __init__(self, app, chat_id):
                self.effective_chat = SimpleNamespace(id=chat_id)
                self._app = app
            @property
            def message(self):
                class M:
                    def __init__(self, app, chat_id):
                        self._app = app; self._chat = chat_id
                    async def reply_text(self, text):
                        return await self._app.bot.send_message(chat_id=self._chat, text=text)
                return M(self._app, chat_id)
        u = U(self.application, chat_id)
        a_text = await self._ollama_stream_chat(u, model_a, a_messages)
        session["ai2ai_last_a"] = a_text
        # Turn B
        b_messages = [
            {"role": "system", "content": f"You are {persona_b}. Respond concisely."},
            {"role": "user", "content": f"Respond to {persona_a}'s statement: {a_text[:800]}"},
        ]
        b_text = await self._ollama_stream_chat(u, model_b, b_messages)
        session["ai2ai_last_b"] = b_text
        self.ollama_sessions[chat_id] = session

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
            kb = self._build_ollama_models_keyboard(models, session["page"])
            text = self._ollama_status_text(session)
            await query.edit_message_text(text, reply_markup=kb)
            self.ollama_sessions[chat_id] = session
            await query.answer("Page updated")
            return
        if callback_data.startswith("ollama_model:"):
            model = callback_data.split(":", 1)[1]
            mode = session.get("mode") or "ai-human"
            if (session.get("ai2ai_model_a") and session.get("ai2ai_model_b")):
                session["ai2ai_active"] = True
            else:
                session["ai2ai_active"] = False
            mode = "ai-ai" if session.get("ai2ai_active") else "ai-human"
            if mode == "ai-ai":
                # Select A then B on the main picker flow
                if not session.get("ai2ai_model_a"):
                    session["ai2ai_model_a"] = model
                    self.ollama_sessions[chat_id] = session
                    kb = self._build_ollama_models_keyboard_ai2ai(session.get("models") or [], "B", session.get("page", 0))
                    await query.edit_message_text(self._ollama_status_text(session), reply_markup=kb)
                    await query.answer("Model A selected")
                    return
                if not session.get("ai2ai_model_b"):
                    session["ai2ai_model_b"] = model
                    self.ollama_sessions[chat_id] = session
                    # After both selected, show updated picker with status
                    kb = self._build_ollama_models_keyboard(session.get("models") or [], session.get("page", 0))
                    await query.edit_message_text(self._ollama_status_text(session), reply_markup=kb)
                    await query.answer("Model B selected")
                    return
                # If both already set, toggle A
                session["ai2ai_model_a"] = model
                self.ollama_sessions[chat_id] = session
                await query.answer("Updated A")
                kb = self._build_ollama_models_keyboard(session.get("models") or [], session.get("page", 0))
                await query.edit_message_text(self._ollama_status_text(session), reply_markup=kb)
                return
            # Default AI→Human
            session["model"] = model
            session["active"] = True
            self.ollama_sessions[chat_id] = session
            await query.edit_message_text(self._ollama_status_text(session), reply_markup=self._build_ollama_models_keyboard(session.get("models") or [], session.get("page", 0)))
            await query.answer("Model selected")
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
        if callback_data.startswith("ollama_ai2ai:"):
            action = callback_data.split(":", 1)[1]
            if action == "start":
                # Initialize AI↔AI session with default personas
                session["ai2ai_active"] = True
                session.setdefault("persona_a", "Albert Einstein")
                session.setdefault("persona_b", "Isaac Newton")
                session.setdefault("topic", session.get("last_user") or "The nature of space and time")
                # Default turns
                try:
                    default_turns = int(os.getenv('OLLAMA_AI2AI_TURNS', '10'))
                except Exception:
                    default_turns = 10
                session.setdefault("ai2ai_turns_left", default_turns)
                # Default models
                if not session.get("ai2ai_model_a"):
                    session["ai2ai_model_a"] = session.get("model")
                if not session.get("ai2ai_model_b"):
                    session["ai2ai_model_b"] = session.get("model")
                self.ollama_sessions[chat_id] = session
                await query.answer("AI↔AI started")
                try:
                    await query.edit_message_text("🤖 AI↔AI mode active. Use Options → Continue exchange to generate turns.")
                except Exception:
                    pass
                await _render_options()
                return
            if action == "continue":
                await query.answer("Continuing…")
                await self._ollama_ai2ai_continue(query.message.chat_id)
                return
            if action == "pick_a":
                models = session.get("models") or []
                if not models:
                    raw = ollama_get_models()
                    models = self._ollama_models_list(raw)
                    session["models"] = models
                kb = self._build_ollama_models_keyboard_ai2ai(models, "A", session.get("page", 0))
                await query.edit_message_text("🤖 Pick model for A:", reply_markup=kb)
                await query.answer("Pick A")
                return
            if action == "pick_b":
                models = session.get("models") or []
                if not models:
                    raw = ollama_get_models()
                    models = self._ollama_models_list(raw)
                    session["models"] = models
                kb = self._build_ollama_models_keyboard_ai2ai(models, "B", session.get("page", 0))
                await query.edit_message_text("🤖 Pick model for B:", reply_markup=kb)
                await query.answer("Pick B")
                return
        if callback_data.startswith("ollama_more_ai2ai:"):
            _, slot, page_str = callback_data.split(":", 2)
            try:
                page = int(page_str)
            except Exception:
                page = 0
            models = session.get("models") or []
            kb = self._build_ollama_models_keyboard_ai2ai(models, slot, page)
            await query.edit_message_text(f"🤖 Pick model for {slot}:", reply_markup=kb)
            session["page"] = page
            self.ollama_sessions[chat_id] = session
            await query.answer("Page updated")
            return
        if callback_data.startswith("ollama_ai2ai_set:"):
            _, slot, name = callback_data.split(":", 2)
            key = "ai2ai_model_a" if slot == "A" else "ai2ai_model_b"
            session[key] = name
            self.ollama_sessions[chat_id] = session
            await query.answer(f"Set {slot} -> {name}")
            # Return to options view
            await _render_options()
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
            report_id = callback_data.replace('delete_', '')
            # Show confirmation with two-button layout
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Yes, delete", callback_data=f"confirm_del_{report_id}"),
                InlineKeyboardButton("❌ Cancel", callback_data="cancel_del")
            ]])
            await query.edit_message_text(
                "⚠️ Delete this summary?\n\nThis will remove it from both Telegram and the Dashboard.",
                reply_markup=keyboard
            )

        elif callback_data.startswith('confirm_del_'):
            report_id = callback_data.replace('confirm_del_', '')
            
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
                    text = self._resolve_summary_text(video_id, variant)

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
                kp_text = self._resolve_summary_text(video_id, 'bullet-points')
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
                report = self._load_local_report(video_id) or {}
                title = (report.get('title') or report.get('video', {}).get('title') or 'Untitled').strip()
                language = (report.get('summary', {}) or {}).get('language') or report.get('summary_language') or 'en'
                difficulty = 'beginner'
                prompt = self._build_quiz_prompt(title=title, keypoints=kp_text, count=10, types=["multiplechoice","truefalse"], difficulty=difficulty, language=language, explanations=True)

                # Generate quiz via Dashboard
                gen = self._post_dashboard_json('/api/generate-quiz', {
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
                if not self._validate_quiz_payload(quiz, explanations=True):
                    await query.edit_message_text("❌ Generated quiz did not pass validation.")
                    return

                # Optional categorization
                cat = self._post_dashboard_json('/api/categorize-quiz', {
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
                slug = self._slugify(title)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                clean_vid = video_id.replace('yt:', '')
                filename = f"{slug}_{clean_vid}_{ts}.json"
                saved = self._post_dashboard_json('/api/save-quiz', {'filename': filename, 'quiz': quiz})
                final_name = (saved or {}).get('filename') or filename

                # Reply with links
                dash = self._get_dashboard_base() or ''
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
        if not gender:
            return "All genders"
        mapping = {
            "female": "Female",
            "male": "Male",
            "unknown": "Unknown",
        }
        return mapping.get(gender, gender.title())

    def _build_tts_catalog_keyboard(self, session: Dict[str, Any]) -> InlineKeyboardMarkup:
        catalog = session.get('catalog') or {}
        filters = (catalog.get('filters') or {}) if catalog else {}
        favorites = session.get('favorites') or []
        selected_gender = session.get('selected_gender')
        selected_family = session.get('selected_family')

        mode = session.get('voice_mode')
        if mode not in ('favorites', 'all'):
            mode = 'favorites' if favorites else 'all'
        if mode == 'favorites' and not favorites:
            mode = 'all'
        session['voice_mode'] = mode

        favorite_by_voice = {}
        allowed_ids = None
        if favorites:
            for fav in favorites:
                voice_id = fav.get('voiceId')
                if voice_id:
                    favorite_by_voice[voice_id] = fav
            if mode == 'favorites':
                allowed_ids = {vid for vid in favorite_by_voice.keys() if vid}

        rows: List[List[InlineKeyboardButton]] = []

        # Mode row
        mark_fav = "✅" if mode == 'favorites' else "⬜"
        mark_all = "✅" if mode == 'all' else "⬜"
        rows.append([
            InlineKeyboardButton(f"{mark_fav} Favorites", callback_data="tts_mode:favorites"),
            InlineKeyboardButton(f"{mark_all} All voices", callback_data="tts_mode:all"),
        ])

        # Gender header and options
        rows.append([InlineKeyboardButton("Gender", callback_data="tts_nop")])
        genders = filters.get('genders') or []
        gender_buttons: List[InlineKeyboardButton] = []
        mark_all_gender = "✅" if not selected_gender else "⬜"
        gender_buttons.append(InlineKeyboardButton(f"{mark_all_gender} All", callback_data="tts_gender:all"))
        for entry in genders:
            gid = entry.get('id')
            if not gid:
                continue
            label = entry.get('label') or gid.title()
            mark = "✅" if selected_gender == gid else "⬜"
            gender_buttons.append(InlineKeyboardButton(f"{mark} {label}", callback_data=f"tts_gender:{gid}"))
        rows.append(gender_buttons)

        # Accent family header and options
        rows.append([InlineKeyboardButton("Accent", callback_data="tts_nop")])
        family_options = available_accent_families(catalog, gender=selected_gender, allowed_ids=allowed_ids)
        accent_rows: List[List[InlineKeyboardButton]] = []
        mark_all_family = "✅" if not selected_family else "⬜"
        accent_rows.append([InlineKeyboardButton(f"{mark_all_family} All", callback_data="tts_accent:all")])

        row: List[InlineKeyboardButton] = []
        for entry in family_options:
            family_id = entry.get('id')
            if not family_id:
                continue
            label = entry.get('label') or family_id.title()
            flag = entry.get('flag') or ''
            mark = "✅" if selected_family == family_id else "⬜"
            button_label = f"{mark} {flag} {label}".strip()
            row.append(InlineKeyboardButton(button_label, callback_data=f"tts_accent:{family_id}"))
            if len(row) == 3:
                accent_rows.append(row)
                row = []
        if row:
            accent_rows.append(row)
        rows.extend(accent_rows)

        # Voices header
        rows.append([InlineKeyboardButton("Voices", callback_data="tts_nop")])
        voice_lookup: Dict[str, Dict[str, Any]] = {}

        # Determine voices to display
        display_voices: List[Dict[str, Any]] = []
        if mode == 'favorites' and allowed_ids:
            filtered = filter_catalog_voices(
                catalog,
                gender=selected_gender,
                family=selected_family,
                allowed_ids=allowed_ids,
            )
            id_to_voice = {voice.get('id'): voice for voice in filtered if voice.get('id')}
            for voice_id, fav in favorite_by_voice.items():
                voice_meta = id_to_voice.get(voice_id)
                if not voice_meta:
                    continue
                entry = dict(voice_meta)
                entry['_favorite'] = fav
                display_voices.append(entry)
        else:
            display_voices = filter_catalog_voices(
                catalog,
                gender=selected_gender,
                family=selected_family,
            )

        if display_voices:
            row: List[InlineKeyboardButton] = []
            idx = 0
            for voice in display_voices:
                voice_id = voice.get('id')
                if not voice_id:
                    continue
                fav_meta = voice.get('_favorite') if mode == 'favorites' else None
                if mode == 'favorites' and not fav_meta:
                    fav_meta = favorite_by_voice.get(voice_id)
                if mode == 'favorites' and not fav_meta:
                    continue
                # Build a compact, index-based key to avoid Telegram's 64-byte limit
                short_key = f"v{idx}"
                idx += 1

                accent = voice.get('accent') or {}
                flag = accent.get('flag') or ''
                label = voice.get('label') or voice_id
                if flag:
                    label = f"{flag} {label}"
                if len(label) > 28:
                    label = f"{label[:25]}…"

                entry_lookup = {
                    'label': label,
                    'voice': voice,
                    'voiceId': voice_id,
                    'engine': voice.get('engine'),
                    'favoriteSlug': (fav_meta.get('slug') if fav_meta else None) or (fav_meta.get('id') if fav_meta else None),
                }
                # Store both short and legacy keys for robustness
                voice_lookup[short_key] = entry_lookup
                if mode == 'favorites' and fav_meta:
                    legacy_key = f"fav|{(fav_meta.get('slug') or fav_meta.get('id') or voice_id)}"
                    voice_lookup[legacy_key] = entry_lookup
                else:
                    legacy_key = f"cat|{voice_id}"
                    voice_lookup[legacy_key] = entry_lookup

                row.append(InlineKeyboardButton(label, callback_data=f"tts_voice:{short_key}"))
                if len(row) == 3:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
        else:
            rows.append([InlineKeyboardButton("(No voices for this filter)", callback_data="tts_nop")])

        session['voice_lookup'] = voice_lookup
        session['current_voices'] = list(voice_lookup.keys())

        rows.append([InlineKeyboardButton("❌ Cancel", callback_data="tts_cancel")])
        return InlineKeyboardMarkup(rows)

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
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📥 Queue for later", callback_data="tts_queue:local"),
             InlineKeyboardButton("☁️ Use OpenAI", callback_data="tts_provider:openai")],
            [InlineKeyboardButton("❌ Cancel", callback_data="tts_cancel")]
        ])

    async def _prompt_tts_provider(self, query, session_payload: Dict[str, Any], title: str) -> None:
        base_hint = session_payload.get('tts_base')
        client = self.tts_client or self._resolve_tts_client(base_hint)
        if client and client.base_api_url:
            self.tts_client = client
            session_payload['tts_base'] = client.base_api_url
        else:
            session_payload['tts_base'] = None
        prompt_text = (
            f"🎙️ Choose how to generate audio for **{self._escape_markdown(title)}**"
        )
        prompt_message = await query.message.reply_text(
            prompt_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self._build_provider_keyboard(include_local=True)
        )
        self._store_tts_session(prompt_message.chat_id, prompt_message.message_id, session_payload)

    async def _execute_tts_job(self, query, session: Dict[str, Any], provider: str) -> None:
        provider = (provider or 'openai').lower()
        summary_text = session.get('summary_text') or session.get('text') or ''
        if not summary_text:
            logging.warning("TTS: session missing text; aborting")
            await query.answer("Missing summary text", show_alert=True)
            return

        placeholders = session.get('placeholders') or {}
        audio_filename = placeholders.get('audio_filename') or f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        json_placeholder = placeholders.get('json_placeholder') or f"tts_{int(time.time())}.json"

        selected_voice = session.get('selected_voice') or {}
        favorite_slug = selected_voice.get('favorite_slug')
        voice_id = selected_voice.get('voice_id')
        engine_id = selected_voice.get('engine')

        await query.answer(f"Generating audio via {provider.title()}…")

        provider_label = "Local TTS hub" if provider == 'local' else "OpenAI TTS"
        voice_label = session.get('last_voice') or ''
        # As a fallback, try to infer a label from the selected_voice payload
        if not voice_label:
            try:
                sv = session.get('selected_voice') or {}
                slug = None
                if sv.get('favorite_slug'):
                    slug = f"fav|{sv.get('favorite_slug')}"
                elif sv.get('voice_id'):
                    slug = f"cat|{sv.get('voice_id')}"
                if slug:
                    voice_label = self._tts_voice_label(session, slug)
            except Exception:
                voice_label = ''

        # Post a visible status bubble so the user knows the click registered
        status_msg = None
        try:
            status_text = (
                f"⏳ Generating TTS"
                + (f" • {self._escape_markdown(voice_label)}" if voice_label else "")
                + f" • {provider_label}"
            )
            status_msg = await query.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            status_msg = None

        try:
            logging.info(
                f"🧩 Starting TTS generation via {provider} | title={session.get('title')}"
            )
            audio_filepath = await self.summarizer.generate_tts_audio(
                summary_text,
                audio_filename,
                json_placeholder,
                provider=provider,
                voice=voice_id,
                engine=engine_id,
                favorite_slug=favorite_slug,
            )
        except LocalTTSUnavailable as exc:
            logging.warning(f"Local TTS unavailable during execution: {exc}")
            await self._handle_local_unavailable(query, session, message=str(exc))
            return
        except Exception as exc:
            logging.error(f"TTS synthesis error: {exc}")
            await query.answer("TTS failed", show_alert=True)
            return

        if not audio_filepath or not Path(audio_filepath).exists():
            logging.warning("TTS generation returned no audio")
            await query.answer("TTS generation failed", show_alert=True)
            return

        logging.info(f"📦 TTS file ready: {audio_filepath}")
        await self._finalize_tts_delivery(query, session, Path(audio_filepath), provider)
        # Update the status bubble to confirm
        try:
            if status_msg:
                done_text = (
                    f"✅ Generated"
                    + (f" • {self._escape_markdown(voice_label)}" if voice_label else "")
                    + f" • {provider_label}"
                )
                await status_msg.edit_text(done_text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass
        # Keep session open so user can test more voices without restarting

    async def _handle_local_unavailable(self, query, session: Dict[str, Any], message: str = "") -> None:
        logging.warning(f"Local TTS unavailable: {message}")
        notice = "⚠️ Local TTS hub unavailable. Queue the job for later or use OpenAI now?"
        await query.edit_message_text(notice, reply_markup=self._build_local_failure_keyboard())
        # keep session so user can choose fallback

    async def _enqueue_tts_job(self, query, session: Dict[str, Any]) -> None:
        job = {
            "created_at": datetime.utcnow().isoformat(),
            "summary_type": session.get('summary_type'),
            "summary_text": session.get('summary_text'),
            "title": session.get('title'),
            "video_info": session.get('video_info'),
            "placeholders": session.get('placeholders'),
            "preferred_provider": "local",
            "selected_voice": session.get('selected_voice'),
        }
        path = enqueue_tts_job(job)
        await query.answer("Queued for local TTS")
        await query.edit_message_text(
            f"📥 Queued TTS job for later processing.\n🗂️ {path.name}",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Close", callback_data="tts_cancel")]])
        )
        logging.info(f"Queued TTS job at {path}")
        self._remove_tts_session(query.message.chat_id, query.message.message_id)

    async def _finalize_tts_delivery(self, query, session: Dict[str, Any], audio_path: Path, provider: str) -> None:
        provider_label = "Local TTS hub" if provider == 'local' else "OpenAI TTS"
        metrics.record_tts(True)
        video_info = session.get('video_info') or {}
        mode = session.get('mode') or 'oneoff_tts'
        title = session.get('title') or video_info.get('title', 'Unknown Title')
        ledger_id = session.get('ledger_id')
        normalized_video_id = session.get('normalized_video_id') or video_info.get('video_id')
        summary_type = session.get('summary_type') or 'audio'
        base_variant = session.get('base_variant') or 'audio'

        # Resolve voice label to display in captions
        voice_label = session.get('last_voice') or ''
        if not voice_label:
            try:
                sv = session.get('selected_voice') or {}
                slug = None
                if sv.get('favorite_slug'):
                    slug = f"fav|{sv.get('favorite_slug')}"
                elif sv.get('voice_id'):
                    slug = f"cat|{sv.get('voice_id')}"
                if slug:
                    voice_label = self._tts_voice_label(session, slug)
            except Exception:
                voice_label = ''

        audio_reply_markup = None
        if mode == 'summary_audio':
            normalized_id = normalized_video_id or video_info.get('video_id') or 'unknown'
            if ledger_id and ':' not in ledger_id:
                ledger_id = f"yt:{ledger_id}"

            content_identifier = (
                session.get('result_id')
                or video_info.get('content_id')
                or ledger_id
                or normalized_id
            )
            if content_identifier and ':' not in content_identifier:
                content_identifier = f"yt:{content_identifier}"

            self._sync_audio_to_targets(normalized_id, audio_path, ledger_id, summary_type)
            if content_identifier:
                self._upload_audio_to_render(content_identifier, audio_path)

            audio_reply_markup = self._build_audio_inline_keyboard(
                normalized_id,
                base_variant,
                video_info.get('video_id', '')
            )

        try:
            with audio_path.open('rb') as audio_file:
                if mode == 'summary_audio':
                    base = f"🎧 **Audio Summary**: {self._escape_markdown(title)}"
                    voice_bit = f" • {self._escape_markdown(voice_label)}" if voice_label else ""
                    caption = f"{base}{voice_bit}\n🎵 {provider_label}"
                else:
                    caption = (
                        f"🎧 **TTS Preview**"
                        + (f" • {self._escape_markdown(voice_label)}" if voice_label else "")
                        + f" • {provider_label}"
                    )
                await query.message.reply_voice(
                    voice=audio_file,
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=audio_reply_markup
                )
            logging.info(f"✅ Successfully sent audio summary for {title} using {provider_label}")
        except Exception as exc:
            logging.error(f"Failed to send voice message: {exc}")
        # Do not close the TTS picker; refresh it (when catalog flow is active)
        try:
            if session.get('catalog'):
                await self._refresh_tts_catalog(query, session)
            else:
                # Favorites-only flow: keep message as-is; show a brief toast
                await query.answer("Audio sent — select another voice")
        except Exception:
            pass

    def _tts_session_key(self, chat_id: int, message_id: int) -> tuple:
        return (chat_id, message_id)

    def _store_tts_session(self, chat_id: int, message_id: int, payload: Dict[str, Any]) -> None:
        self.tts_sessions[self._tts_session_key(chat_id, message_id)] = payload

    def _get_tts_session(self, chat_id: int, message_id: int) -> Optional[Dict[str, Any]]:
        return self.tts_sessions.get(self._tts_session_key(chat_id, message_id))

    def _remove_tts_session(self, chat_id: int, message_id: int) -> None:
        self.tts_sessions.pop(self._tts_session_key(chat_id, message_id), None)

    def _build_tts_keyboard(self, favorites: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
        buttons: List[List[InlineKeyboardButton]] = []
        row: List[InlineKeyboardButton] = []
        max_per_row = 3

        for i, profile in enumerate(favorites):
            slug = profile.get('slug') or profile.get('voiceId')
            if not slug:
                continue
            label = profile.get('label') or profile.get('voiceId') or slug
            if label.startswith("Favorite ·"):
                label = label.split("·", 1)[1].strip() or label
            if len(label) > 28:
                label = f"{label[:25]}…"
            short_key = f"v{i}"
            row.append(InlineKeyboardButton(f"🎤 {label}", callback_data=f"tts_voice:{short_key}"))
            if len(row) == max_per_row:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        buttons.append([InlineKeyboardButton("❌ Cancel", callback_data="tts_cancel")])
        return InlineKeyboardMarkup(buttons)

    def _tts_prompt_text(
        self,
        text: str,
        last_voice: Optional[str] = None,
        gender: Optional[str] = None,
        family: Optional[str] = None,
        catalog: Optional[Dict[str, Any]] = None,
    ) -> str:
        snippet = (text or "").strip()
        if len(snippet) > 280:
            snippet = snippet[:277].rstrip() + "…"
        lines = []
        # Width-keeper line to force Telegram to expand the bubble
        lines.append("──────────────────────────────")
        if last_voice:
            lines.append(f"✅ Last voice: {last_voice}")
        lines.append("🗣️ Ready to synthesize speech for:")
        lines.append(f"“{snippet or '…'}”")
        gender_label = self._gender_label(gender)
        family_label = accent_family_label(catalog or {}, family)
        lines.append(f"Filters: {gender_label} · {family_label}")
        lines.append("Select a voice below or cancel.")
        lines.append("──────────────────────────────")
        return "\n".join(lines)

    def _tts_voice_label(self, session: Dict[str, Any], slug: str) -> str:
        lookup = session.get('voice_lookup') or {}
        entry = lookup.get(slug)
        if entry:
            label = entry.get('label')
            if label:
                return label
            voice = entry.get('voice') or {}
            accent = voice.get('accent') or {}
            base_label = voice.get('label') or entry.get('voiceId') or slug
            flag = accent.get('flag')
            if flag:
                return f"{flag} {base_label}"
            return base_label
        base_slug = slug.split('|', 1)[-1]
        catalog = session.get('catalog') or {}
        for voice in catalog.get('voices') or []:
            if voice.get('id') == base_slug:
                label = voice.get('label') or slug
                accent = voice.get('accent') or {}
                flag = accent.get('flag')
                if flag:
                    return f"{flag} {label}"
                return label
        favorites = session.get('favorites') or []
        for profile in favorites:
            profile_slug = profile.get('slug') or profile.get('voiceId')
            if profile_slug == base_slug:
                raw_label = profile.get('label') or profile.get('voiceId') or slug
                if raw_label.startswith("Favorite ·"):
                    return raw_label.split("·", 1)[1].strip() or slug
                return raw_label
        return slug

    # ------------------------- Telegram command: /tts -------------------------
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
        logging.info(f"🎛️ TTS callback: {callback_data}")
        message = query.message
        if not message:
            return
        chat_id = message.chat_id
        message_id = message.message_id
        session = self._get_tts_session(chat_id, message_id)
        if session:
            logging.debug(f"TTS session found for {chat_id}:{message_id} keys={list(session.keys())}")
        else:
            logging.warning(f"TTS session missing for {chat_id}:{message_id}")

        if not session:
            await query.answer("Session expired", show_alert=True)
            try:
                await query.edit_message_text("⚠️ This TTS session has expired. Send /tts again.")
            except Exception:
                pass
            return

        client = self._resolve_tts_client(session.get('tts_base'))
        if not client or not client.base_api_url:
            logging.warning("TTS hub unavailable during TTS callback")
            await query.answer("TTS hub unavailable", show_alert=True)
            return

        if callback_data == "tts_cancel":
            self._remove_tts_session(chat_id, message_id)
            await query.answer("Cancelled")
            try:
                await query.edit_message_text("❌ TTS session cancelled.")
            except Exception:
                pass
            return

        catalog = session.get("catalog")

        if catalog:
            if callback_data.startswith("tts_mode:"):
                mode_value = callback_data.split(":", 1)[1]
                if mode_value == "favorites":
                    if session.get('favorites'):
                        session['voice_mode'] = 'favorites'
                        session['selected_gender'] = None
                        session['selected_family'] = None
                        await self._refresh_tts_catalog(query, session)
                        await query.answer("Favorites selected")
                    else:
                        await query.answer("No favorites available", show_alert=True)
                    return
                elif mode_value == "all":
                    session['voice_mode'] = 'all'
                    session['selected_gender'] = None
                    session['selected_family'] = None
                    await self._refresh_tts_catalog(query, session)
                    await query.answer("Showing all voices")
                    return
            if callback_data == "tts_nop":
                await query.answer("Select an option below")
                return
            if callback_data.startswith("tts_gender:"):
                value = callback_data.split(":", 1)[1]
                session["selected_gender"] = None if value in ("all", "") else value
                session["selected_family"] = None
                await self._refresh_tts_catalog(query, session)
                await query.answer("Gender updated")
                return
            if callback_data.startswith("tts_accent:"):
                value = callback_data.split(":", 1)[1]
                session["selected_family"] = None if value in ("all", "") else value
                await self._refresh_tts_catalog(query, session)
                await query.answer("Accent updated")
                return

        if callback_data.startswith("tts_provider:"):
            provider = callback_data.split(":", 1)[1]
            session['provider'] = provider
            if provider == 'local':
                client = self._resolve_tts_client(session.get('tts_base'))
                session['tts_base'] = client.base_api_url if client else None
                catalog = session.get('catalog')
                favorites = session.get('favorites')
                if client and not catalog:
                    try:
                        catalog = await client.fetch_catalog()
                        session['catalog'] = catalog
                        # Populate favorites: prefer tag=telegram, fall back to global
                        if not favorites:
                            try:
                                favorites = await client.fetch_favorites(tag="telegram")
                            except Exception:
                                favorites = []
                            if not favorites:
                                try:
                                    favorites = await client.fetch_favorites()
                                except Exception:
                                    favorites = []
                            session['favorites'] = favorites or []
                    except Exception as exc:
                        await self._handle_local_unavailable(query, session, message=str(exc))
                        return
                if not catalog or not catalog.get('voices'):
                    await self._handle_local_unavailable(query, session, message="No voices available")
                    return
                if favorites is None and session.get('favorites') is None:
                    session['favorites'] = []
                # Default to favorites when available
                session['voice_mode'] = session.get('voice_mode') or ('favorites' if session.get('favorites') else 'all')
                self._store_tts_session(query.message.chat_id, query.message.message_id, session)
                await self._refresh_tts_catalog(query, session)
                await query.answer("Select a voice")
                return
            else:
                await self._execute_tts_job(query, session, provider)
            return

        if callback_data.startswith("tts_queue:"):
            await self._enqueue_tts_job(query, session)
            return

        if not callback_data.startswith("tts_voice:"):
            return

        payload = callback_data.split(":", 1)[1]
        logging.info(f"🔊 Voice selected payload={payload}")
        kind, _, identifier = payload.partition("|")
        if not identifier:
            identifier = kind
            kind = 'cat'

        voice_lookup = session.get('voice_lookup') or {}
        entry = voice_lookup.get(payload) or {}

        # Prefer values from the lookup entry (works for both catalog and favorites)
        favorite_slug = entry.get('favoriteSlug') if entry else None
        voice_id = entry.get('voiceId') if entry else None
        engine_id = entry.get('engine') if entry else None

        # Legacy/fallback: if no entry was found, infer from the payload
        if not favorite_slug and not voice_id:
            if kind == 'fav':
                favorite_slug = identifier
            else:
                voice_id = identifier

        session['selected_voice'] = {
            'favorite_slug': favorite_slug,
            'voice_id': voice_id,
            'engine': engine_id,
        }

        provider_choice = session.get('provider') or 'local'
        # Remember last selection for prompt header
        try:
            session['last_voice'] = self._tts_voice_label(session, payload)
        except Exception:
            pass
        self._store_tts_session(chat_id, message_id, session)

        logging.info(
            f"🚀 Executing TTS job provider={provider_choice} fav={favorite_slug} voice_id={voice_id} engine={engine_id}"
        )

        try:
            await self._execute_tts_job(query, session, provider_choice)
        except LocalTTSUnavailable as exc:
            await self._handle_local_unavailable(query, session, message=str(exc))
            return

    def _get_dashboard_base(self) -> Optional[str]:
        return (
            os.getenv('DASHBOARD_URL')
            or os.getenv('POSTGRES_DASHBOARD_URL')
            or os.getenv('RENDER_DASHBOARD_URL')
        )

    # ------------------------- Quiz helpers -------------------------
    def _slugify(self, text: str) -> str:
        text = (text or '').strip().lower()
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'[^a-z0-9\-\_\s]+', '', text)
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'_+', '_', text).strip('_')
        return text or 'quiz'

    def _build_quiz_prompt(self, *, title: str, keypoints: str, count: int = 10,
                           types: Optional[List[str]] = None, difficulty: str = 'beginner', language: str = 'en', explanations: bool = True) -> str:
        allowed = types or ["multiplechoice", "truefalse"]
        type_list = ', '.join(allowed)
        explain_rule = 'Provide a brief explanation (1–2 sentences) for each item.' if explanations else 'Do not include explanations.'
        return (
            f"Create {count} quiz questions in {language}.\n"
            f"Topic: {title}\n\n"
            f"Use ONLY these key points (no outside facts):\n{keypoints}\n\n"
            f"Rules:\n"
            f"- Allowed types: {type_list}\n"
            f"- Only one unambiguous correct answer per question\n"
            f"- {explain_rule}\n"
            f"- Respond ONLY as a JSON object with this shape (no text outside JSON):\n"
            '{"count": number, "meta": {"topic": string, "difficulty": "beginner|intermediate|advanced", "language": ' + f'"{language}"' + '}, '
            '"items": [{"question": string, "type": "multiplechoice|truefalse|yesno|shortanswer", '
            '"options": [string, ...] (omit for shortanswer), "correct": number (omit for shortanswer), '
            '"answer": string (only for shortanswer), "explanation": string}]}'
        )

    def _validate_quiz_payload(self, data: dict, explanations: bool = True) -> bool:
        """Validate and lightly normalize quiz JSON for storage.
        Accepts common type aliases and fixes trivial omissions for TF/YesNo.
        """
        try:
            if not isinstance(data, dict):
                return False
            items = data.get('items')
            if not isinstance(items, list) or not items:
                return False
            # Normalize count
            try:
                data['count'] = int(data.get('count') or len(items))
            except Exception:
                data['count'] = len(items)

            alias_map = {
                'multiplechoice': 'multiplechoice', 'multiple-choice': 'multiplechoice', 'multiple choice': 'multiplechoice', 'mcq': 'multiplechoice',
                'truefalse': 'truefalse', 'true/false': 'truefalse', 'true false': 'truefalse', 'boolean': 'truefalse',
                'yesno': 'yesno', 'yes/no': 'yesno', 'yes no': 'yesno',
                'shortanswer': 'shortanswer', 'short answer': 'shortanswer'
            }

            def norm_type(t: str) -> str:
                key = re.sub(r'[^a-z/ ]+', '', (t or '').strip().lower())
                return alias_map.get(key, key)

            seen = set()
            normalized_items = []
            for q in items:
                if not isinstance(q, dict):
                    return False
                # Question text unique check
                qtext = re.sub(r'\s+', ' ', (q.get('question') or '').strip())
                if not qtext:
                    return False
                qnorm = qtext.lower()
                if qnorm in seen:
                    continue  # drop duplicates silently
                seen.add(qnorm)

                qtype = norm_type(q.get('type'))
                if qtype in ('multiplechoice', 'truefalse', 'yesno'):
                    opts = q.get('options')
                    # Provide defaults for TF/YesNo if missing
                    if qtype == 'truefalse' and not isinstance(opts, list):
                        opts = ["True", "False"]
                        q['options'] = opts
                    if qtype == 'yesno' and not isinstance(opts, list):
                        opts = ["Yes", "No"]
                        q['options'] = opts
                    if not isinstance(opts, list):
                        return False
                    min_opts = 3 if qtype == 'multiplechoice' else 2
                    if len(opts) < min_opts:
                        return False
                    ci = q.get('correct')
                    if not isinstance(ci, int) or ci < 0 or ci >= len(opts):
                        return False
                elif qtype == 'shortanswer':
                    ans = q.get('answer')
                    if not isinstance(ans, str) or not ans.strip():
                        return False
                else:
                    return False

                if explanations and not isinstance(q.get('explanation'), str):
                    q['explanation'] = q.get('explanation') or ""

                q['type'] = qtype  # write back normalized type
                q['question'] = qtext
                normalized_items.append(q)

            if not normalized_items:
                return False
            data['items'] = normalized_items
            data['count'] = len(normalized_items)
            return True
        except Exception as e:
            logging.warning(f"Quiz validation error: {e}")
            return False

    def _post_dashboard_json(self, endpoint: str, payload: dict, timeout: int = 30) -> Optional[dict]:
        base = self._get_dashboard_base()
        if not base or requests is None:
            return None
        try:
            url = f"{base.rstrip('/')}{endpoint}"
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code >= 200 and r.status_code < 300:
                return r.json()
        except Exception:
            return None
        return None

    def _load_local_report(self, video_id: str) -> Optional[Dict[str, Any]]:
        reports_dir = Path('./data/reports')
        for path in sorted(reports_dir.glob(f'*{video_id}*.json')):
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                continue
        return None

    def _extract_variant_text(self, report: Dict[str, Any], variant: str) -> Optional[str]:
        variant = normalize_variant_id(variant)
        if not report:
            return None
        s = report.get('summary') or {}
        # Check explicit variants list
        vs = s.get('variants')
        if isinstance(vs, list):
            for entry in vs:
                if isinstance(entry, dict):
                    v = normalize_variant_id(entry.get('variant') or entry.get('summary_type') or entry.get('type'))
                    if v == variant:
                        txt = entry.get('text') or entry.get('summary') or entry.get('content')
                        if isinstance(txt, str) and txt.strip():
                            return txt
        # Check top-level summary if types match
        st = normalize_variant_id(s.get('summary_type') or s.get('type'))
        if st == variant:
            txt = s.get('summary') or s.get('text')
            if isinstance(txt, str) and txt.strip():
                return txt
        # Try top-level summary_variants (ingest payloads)
        sv = report.get('summary_variants')
        if isinstance(sv, list):
            for entry in sv:
                if isinstance(entry, dict):
                    v = normalize_variant_id(entry.get('variant') or entry.get('summary_type') or entry.get('type'))
                    if v == variant:
                        txt = entry.get('text')
                        if isinstance(txt, str) and txt.strip():
                            return txt
        return None

    @staticmethod
    def _extract_summary_text(summary_payload: Any) -> str:
        """Normalize various summary payload shapes into plain text."""
        if isinstance(summary_payload, dict):
            for key in ("summary", "text", "comprehensive", "audio", "content"):
                value = summary_payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            # Check nested variants list
            variants = summary_payload.get("variants")
            if isinstance(variants, list):
                for entry in variants:
                    if isinstance(entry, dict):
                        text = entry.get("text") or entry.get("summary") or entry.get("content")
                        if isinstance(text, str) and text.strip():
                            return text.strip()
        elif isinstance(summary_payload, str):
            return summary_payload.strip()
        return ""

    def _fetch_report_from_dashboard(self, video_id: str) -> Optional[Dict[str, Any]]:
        base = self._get_dashboard_base()
        if not base or not requests:
            return None
        try:
            url = f"{base.rstrip('/')}/api/reports/{video_id}"
            headers = {}
            token = os.getenv('DASHBOARD_TOKEN')
            if token:
                headers['Authorization'] = f"Bearer {token}"
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        return None

    def _resolve_summary_text(self, video_id: str, variant: str) -> Optional[str]:
        # 1) Local report
        report = self._load_local_report(video_id)
        txt = self._extract_variant_text(report or {}, variant)
        if isinstance(txt, str) and txt.strip():
            return txt
        # 2) Dashboard
        report = self._fetch_report_from_dashboard(video_id)
        txt = self._extract_variant_text(report or {}, variant)
        if isinstance(txt, str) and txt.strip():
            return txt
        return None
    
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
        """Process summarization for the currently selected content item."""
        item = self.current_item or {}
        content_id = item.get("content_id")
        source = item.get("source", "youtube")
        url = item.get("url")

        if not content_id:
            await query.edit_message_text("❌ No content in context. Please send a link first.")
            return

        if not url:
            url = self._resolve_video_url(content_id)

        if not url:
            await query.edit_message_text("❌ Could not resolve the source URL. Please resend the link.")
            return
        
        if not self.summarizer:
            await query.edit_message_text("❌ Summarizer not available. Please try /status for more info.")
            return
        
        # Update message to show processing
        # Create user-friendly processing messages
        noun_map = {
            "youtube": "video",
            "reddit": "thread",
            "web": "article",
        }
        noun = noun_map.get(source, "item")
        processing_messages = {
            "comprehensive": f"📝 Analyzing {noun} and creating comprehensive summary...",
            "bullet-points": f"🎯 Extracting key points from the {noun}...",
            "key-insights": f"💡 Identifying key insights and takeaways from the {noun}...",
            "audio": "🎙️ Creating audio summary with text-to-speech...",
            "audio-fr": "🇫🇷 Translating to French and preparing audio narration...",
            "audio-es": "🇪🇸 Translating to Spanish and preparing audio narration..."
        }
        
        # Handle proficiency-specific messages
        base_type = summary_type.split(':')[0]  # Extract base type from "audio-fr:beginner"
        if base_type.startswith("audio-fr"):
            level_suffix = " (with vocabulary help)" if proficiency_level in ["beginner", "intermediate"] else ""
            message = f"🇫🇷 Creating French audio summary{level_suffix}... This may take a moment."
        elif base_type.startswith("audio-es"):
            level_suffix = " (with vocabulary help)" if proficiency_level in ["beginner", "intermediate"] else ""
            message = f"🇪🇸 Creating Spanish audio summary{level_suffix}... This may take a moment."
        else:
            prefix_map = {
                "youtube": "🔄",
                "reddit": "🧵",
                "web": "📰",
            }
            default_prefix = prefix_map.get(source, "🔄")
            message = processing_messages.get(base_type, f"{default_prefix} Processing {summary_type}... This may take a moment.")

        await query.edit_message_text(message)
        
        try:
            # Determine ledger/content identifiers
            normalized_id = self._normalize_content_id(content_id)
            video_id = normalized_id
            ledger_id = content_id
            display_id = f"{source}:{normalized_id}" if source != "youtube" else normalized_id

            # Check ledger before processing
            entry = ledger.get(ledger_id, summary_type)
            if entry:
                logging.info(f"📄 Found existing entry for {display_id}:{summary_type}")
                if render_probe.render_has(entry.get("stem")):
                    await self._send_existing_summary_notice(query, ledger_id, summary_type)
                    logging.info(f"♻️ SKIPPED: {display_id} already on dashboard")
                    return
                else:
                    # Content exists in DB but not Dashboard - proceed with fresh processing
                    logging.info(f"🔄 Content exists in database but missing from Dashboard - processing fresh")
                    # Continue to fresh processing below instead of trying to re-sync files
            
            # Process the video (new processing)
            logging.info(
                f"🎬 PROCESSING: {display_id} | {summary_type} | user: {user_name} | URL: {url}"
            )
            logging.info(
                f"🧠 LLM: {self.summarizer.llm_provider}/{self.summarizer.model}"
            )
            
            if source == "reddit":
                result = await self.summarizer.process_reddit_thread(
                    url,
                    summary_type=summary_type,
                    proficiency_level=proficiency_level
                )
            elif source == "web":
                result = await self.summarizer.process_web_page(
                    url,
                    summary_type=summary_type,
                    proficiency_level=proficiency_level
                )
            else:
                result = await self.summarizer.process_video(
                    url,
                    summary_type=summary_type,
                    proficiency_level=proficiency_level
                )
            
            if result:
                result_content_id = result.get('id')
                if result_content_id and result_content_id != ledger_id:
                    ledger_id = result_content_id
                    normalized_id = self._normalize_content_id(result_content_id)
                    video_id = normalized_id
                    display_id = f"{source}:{normalized_id}" if source != "youtube" else normalized_id
                    canonical_url = result.get('canonical_url') or url
                    url = canonical_url or url
                    self.current_item = {
                        "source": source,
                        "url": url,
                        "content_id": ledger_id,
                        "raw_id": result_content_id.split(':', 1)[-1],
                        "normalized_id": normalized_id,
                    }
            if not result:
                await query.edit_message_text("❌ Failed to process content. Please check the URL and try again.")
                return

            error_message = result.get('error') if isinstance(result, dict) else None
            if error_message:
                if 'No transcript available' in error_message:
                    await query.edit_message_text(
                        f"⚠️ {error_message}\n\nSkipping this item to prevent empty dashboard entries."
                    )
                    logging.info(f"❌ ABORTED: {error_message}")
                else:
                    await query.edit_message_text(f"❌ {error_message}")
                    logging.info(f"❌ Processing error: {error_message}")
                return
            
            # Export to JSON for dashboard (skip HTML to avoid duplicates)
            export_info = {"html_path": None, "json_path": None}
            try:
                # Export to JSON (primary format for dashboard)
                # Use the proper helper function to transform data structure
                report_dict = create_report_from_youtube_summarizer(result)
                json_path = self.json_exporter.save_report(report_dict)
                export_info["json_path"] = Path(json_path).name

                # Verify the JSON file was actually created
                json_path_obj = Path(json_path)
                if json_path_obj.exists():
                    logging.info(f"✅ Exported JSON report: {json_path}")
                else:
                    logging.warning(f"⚠️ JSON export returned path but file not created: {json_path}")
                    logging.warning(f"   This will cause dual-sync to fail!")
                
                # Add to ledger immediately after saving
                stem = Path(json_path).stem
                ledger_entry = {
                    "stem": stem,
                    "json": str(json_path),
                    "mp3": None,  # Will update after audio is found
                    "synced": False,
                    "created_at": datetime.now().isoformat()
                }
                
                # Add language learning metadata for multilingual requests
                if proficiency_level:
                    ledger_entry["proficiency"] = proficiency_level
                    if summary_type.startswith("audio-"):
                        lang_code = "fr" if summary_type.startswith("audio-fr") else "es"
                        ledger_entry["target_language"] = lang_code
                        ledger_entry["language_flag"] = "🇫🇷" if lang_code == "fr" else "🇪🇸"
                        ledger_entry["learning_mode"] = True
                        
                        # Add proficiency badge for dashboard
                        proficiency_badges = {
                            "beginner": {"fr": "🟢 Débutant", "es": "🟢 Principiante"},
                            "intermediate": {"fr": "🟡 Intermédiaire", "es": "🟡 Intermedio"}, 
                            "advanced": {"fr": "🔵 Avancé", "es": "🔵 Avanzado"}
                        }
                        if proficiency_level in proficiency_badges:
                            ledger_entry["proficiency_badge"] = proficiency_badges[proficiency_level][lang_code]
                
                ledger.upsert(ledger_id, summary_type, ledger_entry)
                logging.info(f"📊 Added to ledger: {display_id}:{summary_type}")
                
                # Sync to Render dashboard (hybrid architecture)
                # For audio summaries, delay sync until after TTS generation to include MP3 metadata
                is_audio_summary = summary_type == "audio" or summary_type.startswith("audio-fr") or summary_type.startswith("audio-es")
                
                if not is_audio_summary:
                    # Immediate sync for non-audio summaries
                    try:
                        json_path_obj = Path(json_path)
                        stem = json_path_obj.stem
                        
                        # Dual-sync to SQLite + PostgreSQL (T-Y020A)
                        logging.info(f"📡 DUAL-SYNC START: Uploading to configured targets...")

                        # Extract video ID for logging
                        video_metadata = result.get('metadata', {})
                        result_content_id = result.get('id') or (ledger_id if ledger_id else stem)

                        # Use dual-sync with JSON report path
                        report_path = Path(f"/app/data/reports/{stem}.json")
                        sync_results = dual_sync_upload(report_path)

                        # Determine success (at least one target succeeded)
                        sqlite_ok = bool(sync_results.get('sqlite', {}).get('report')) if isinstance(sync_results, dict) else False
                        postgres_ok = bool(sync_results.get('postgres', {}).get('report')) if isinstance(sync_results, dict) else False
                        sync_success = sqlite_ok or postgres_ok

                        if sync_success:
                            targets = []
                            if sqlite_ok: targets.append("SQLite")
                            if postgres_ok: targets.append("PostgreSQL")
                            logging.info(f"✅ DUAL-SYNC SUCCESS: 📊 → {result_content_id} (targets: {', '.join(targets)})")

                            # Update ledger and mark as synced
                            entry = ledger.get(ledger_id, summary_type)
                            if entry:
                                entry["synced"] = True
                                entry["last_synced"] = datetime.now().isoformat()
                                entry["sync_targets"] = targets  # Track which targets succeeded
                                ledger.upsert(ledger_id, summary_type, entry)
                                logging.info(f"📊 Updated ledger: synced=True, targets={targets}")
                        else:
                            logging.error(f"❌ DUAL-SYNC FAILED: All targets failed for {stem}")

                    except Exception as sync_e:
                        logging.warning(f"⚠️ Dual-sync error: {sync_e}")
                else:
                    # For audio summaries, still sync the content metadata now
                    # Audio file will be synced later when TTS is complete
                    try:
                        json_path_obj = Path(json_path)
                        stem = json_path_obj.stem

                        # Extract video ID for logging
                        video_metadata = result.get('metadata', {})
                        result_content_id = result.get('id') or (ledger_id if ledger_id else stem)

                        logging.info(f"📡 DUAL-SYNC (content-only): Audio summary - syncing metadata for {result_content_id}")

                        # Use dual-sync with JSON report path (content only, audio comes later)
                        report_path = json_path_obj  # Use the actual JSON file path instead of reconstructing

                        # Handle timing issue - file might not be fully written yet
                        import time
                        max_retries = 3
                        for attempt in range(max_retries):
                            if report_path.exists():
                                break
                            logging.debug(f"📄 Waiting for file to be written (attempt {attempt + 1}/{max_retries}): {report_path}")
                            time.sleep(0.1)  # Wait 100ms between attempts

                        if report_path.exists():
                            sync_results = dual_sync_upload(report_path)

                            # Determine success (at least one target succeeded)
                            sqlite_ok = bool(sync_results.get('sqlite', {}).get('report')) if isinstance(sync_results, dict) else False
                            postgres_ok = bool(sync_results.get('postgres', {}).get('report')) if isinstance(sync_results, dict) else False
                            sync_success = sqlite_ok or postgres_ok

                            if sync_success:
                                targets = []
                                if sqlite_ok: targets.append("SQLite")
                                if postgres_ok: targets.append("PostgreSQL")
                                logging.info(f"✅ DUAL-SYNC CONTENT: 📊 → {result_content_id} (targets: {', '.join(targets)})")
                                logging.info(f"⏳ Audio sync will happen after TTS generation")

                                # Update ledger and mark as synced
                                entry = ledger.get(ledger_id, summary_type)
                                if entry:
                                    entry["synced"] = True
                                    entry["last_synced"] = datetime.now().isoformat()
                                    entry["sync_targets"] = targets  # Track which targets succeeded
                                    ledger.upsert(ledger_id, summary_type, entry)
                                    logging.info(f"📊 Updated ledger: synced=True, targets={targets}")
                            else:
                                logging.error(f"❌ DUAL-SYNC CONTENT FAILED: All targets failed for {stem}")
                        else:
                            logging.warning(f"⚠️ JSON report not found for content sync: {report_path}")

                    except Exception as sync_e:
                        logging.warning(f"⚠️ Dual-sync content error: {sync_e}")
                        logging.info(f"⏳ Will retry full sync after TTS generation")
                
                # TODO: Generate HTML on-demand when "Full Report" is clicked
                # For now, skip HTML to prevent duplicate dashboard cards
                
            except Exception as e:
                logging.warning(f"⚠️ Export failed: {e}")
            
            # Format and send the response
            await self._send_formatted_response(query, result, summary_type, export_info)
            
        except Exception as e:
            logging.error(f"Error processing content {url}: {e}")
            await query.edit_message_text(f"❌ Error processing content: {str(e)[:100]}...")
    
    async def _send_formatted_response(self, query, result: Dict[str, Any], summary_type: str, export_info: Dict = None):
        """Send formatted summary response."""
        try:
            # Get base metadata
            video_info = result.get('metadata', {})
            source = result.get('content_source') or self._current_source()
            title = result.get('title') or video_info.get('title') or 'Untitled content'
            channel = (video_info.get('uploader') or video_info.get('channel') or video_info.get('author')
                      or video_info.get('subreddit') or 'Unknown source')
            duration_info = self._format_duration_and_savings(video_info)

            # Base identifiers used by action buttons and caching
            universal_id = result.get('id') or video_info.get('video_id') or self._current_content_id() or ''
            video_id = self._normalize_content_id(universal_id)
            base_type = (summary_type or '').split(':', 1)[0]
            
            # Get summary content - handle both old and new summary structures
            summary_data = result.get('summary', {})
            summary = 'No summary available'
            
            if isinstance(summary_data, dict):
                # Handle both direct chunked structure and wrapped JSON structure
                if summary_type == "audio":
                    # Try direct chunked structure first (summary.audio), then wrapped (summary.content.audio)
                    summary = (summary_data.get('audio') or 
                              summary_data.get('content', {}).get('audio') or
                              summary_data.get('content', {}).get('comprehensive') or
                              summary_data.get('comprehensive') or
                              summary_data.get('summary') or
                              'No audio summary available')
                elif summary_type == "bullet-points":
                    summary = (summary_data.get('bullet_points') or 
                              summary_data.get('content', {}).get('bullet_points') or
                              summary_data.get('content', {}).get('comprehensive') or
                              summary_data.get('comprehensive') or
                              summary_data.get('summary') or
                              'No bullet points available')
                elif summary_type == "key-insights":
                    summary = (summary_data.get('key_insights') or 
                              summary_data.get('content', {}).get('key_insights') or
                              summary_data.get('content', {}).get('comprehensive') or
                              summary_data.get('comprehensive') or
                              summary_data.get('summary') or
                              'No key insights available')
                else:
                    # Default to comprehensive
                    summary = (summary_data.get('comprehensive') or 
                              summary_data.get('content', {}).get('comprehensive') or
                              summary_data.get('content', {}).get('audio') or
                              summary_data.get('audio') or
                              summary_data.get('summary') or
                              'No comprehensive summary available')
            elif isinstance(summary_data, str):
                summary = summary_data
            
            # Always send text summary first for better UX
            # (For audio summaries, TTS will be generated separately below)
            
            # Format response header (without summary content)
            source_icon = {
                'youtube': '🎬',
                'reddit': '🧵',
                'web': '📰',
            }.get(source, '🎬')
            channel_icon = '👤' if source == 'reddit' else '📺'
            header_parts = [
                f"{source_icon} **{self._escape_markdown(title)}**",
                f"{channel_icon} {self._escape_markdown(channel)}",
                duration_info,
                "",
                f"📝 **{summary_type.replace('-', ' ').title()} Summary:**"
            ]
            
            header_text = "\n".join(part for part in header_parts if part)
            
            # Create inline keyboard with link buttons if exports were successful
            reply_markup = None
            if export_info and (export_info.get('html_path') or export_info.get('json_path')):
                dashboard_url = (
                    os.getenv('DASHBOARD_URL')
                    or os.getenv('POSTGRES_DASHBOARD_URL')
                    or 'https://ytv2-dashboard-postgres.onrender.com'
                )
                
                # Extract report ID for the deep link and delete functionality
                report_id = None
                if export_info.get('json_path'):
                    json_path = Path(export_info['json_path'])
                    report_id = json_path.stem
                elif export_info.get('html_path'):
                    html_path = Path(export_info['html_path'])
                    report_id = html_path.stem
                
                # Only add buttons if we have a public URL (Telegram can't access localhost)
                if dashboard_url:
                    keyboard = []

                    # Encode report ID for URL safety
                    report_id_encoded = urllib.parse.quote(report_id, safe='') if report_id else ''

                    # Row 1: Dashboard | Open Summary
                    row1 = [
                        InlineKeyboardButton("📊 Dashboard", url=dashboard_url)
                    ]
                    if report_id_encoded:
                        row1.append(InlineKeyboardButton("📄 Open Summary", url=f"{dashboard_url}#report={report_id_encoded}"))

                    keyboard.append(row1)

                    # Row 2: Listen | Generate Quiz
                    if report_id_encoded:
                        base_variant = base_type
                        listen_cb = f"listen_this:{video_id}:{base_variant}"
                        gen_cb = f"gen_quiz:{video_id}"
                        row2 = []
                        if len(listen_cb.encode('utf-8')) <= 64:
                            row2.append(InlineKeyboardButton("▶️ Listen", callback_data=listen_cb))
                        if len(gen_cb.encode('utf-8')) <= 64:
                            row2.append(InlineKeyboardButton("🧩 Generate Quiz", callback_data=gen_cb))
                        if row2:
                            keyboard.append(row2)

                        # Row 3: Add Variant | Delete…
                        del_cb = f"delete_{report_id}"
                        if len(del_cb.encode('utf-8')) > 64:
                            max_id_len = 64 - len("delete_")
                            truncated_id = report_id[:max_id_len]
                            del_cb = f"delete_{truncated_id}"
                        keyboard.append([
                            InlineKeyboardButton("➕ Add Variant", callback_data="summarize_back_to_main"),
                            InlineKeyboardButton("🗑️ Delete…", callback_data=del_cb)
                        ])

                    reply_markup = InlineKeyboardMarkup(keyboard)
                else:
                    reply_markup = None
                    logging.warning("⚠️ No DASHBOARD_URL set - skipping link buttons")
            
            # Always send the full text (it will auto-split via _send_long_message)
            # The 'summary' variable already contains the correct text for the chosen summary type
            sent_msg = await self._send_long_message(query, header_text, summary, reply_markup)

            # Cache exact text for one-off TTS replay (message-scoped)
            try:
                if sent_msg is not None:
                    chat_id = sent_msg.chat_id if hasattr(sent_msg, 'chat_id') else sent_msg.chat.id
                    message_id = sent_msg.message_id
                    payload = {
                        "video_id": video_id,
                        "variant": base_type,
                        "language": (result.get('summary', {}) or {}).get('language') or result.get('summary_language') or '',
                        "title": title,
                        "text": summary,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self._cache_oneoff_tts(chat_id, message_id, payload)
            except Exception as e:
                logging.warning(f"⚠️ Failed to cache one-off TTS payload: {e}")
            
            logging.info(f"Successfully sent {summary_type} summary for {title}")
            
            # Generate TTS audio for audio summaries (after text is sent)
            if summary_type == "audio" or summary_type.startswith("audio-fr") or summary_type.startswith("audio-es"):
                await self._prepare_tts_generation(query, result, summary, summary_type)
            
        except Exception as e:
            logging.error(f"Error sending formatted response: {e}")
            await query.edit_message_text("❌ Error formatting response. The summary was generated but couldn't be displayed properly.")
    
    async def _prepare_tts_generation(self, query, result: Dict[str, Any], summary_text: str, summary_type: str):
        """Store context and prompt the user to choose a TTS provider."""
        video_info = result.get('metadata', {})
        title = video_info.get('title', 'Unknown Title')

        ledger_id = (
            result.get('id')
            or video_info.get('content_id')
            or (self.current_item or {}).get('content_id')
        )

        normalized_video_id = video_info.get('video_id')
        if not normalized_video_id and ledger_id:
            normalized_video_id = self._normalize_content_id(ledger_id)

        if not ledger_id and normalized_video_id:
            ledger_id = f"yt:{normalized_video_id}"

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_id = video_info.get('video_id', 'unknown')
        base_variant = (summary_type or "").split(":", 1)[0]
        placeholders = {
            "audio_filename": f"audio_{video_id}_{timestamp}.mp3",
            "json_placeholder": f"yt_{video_id}_placeholder.json",
        }

        session_payload = {
            "mode": "summary_audio",
            "summary_text": summary_text,
            "summary_type": summary_type,
            "title": title,
            "video_info": video_info,
            "ledger_id": ledger_id,
            "normalized_video_id": normalized_video_id,
            "placeholders": placeholders,
            "base_variant": base_variant,
            "result_id": result.get('id'),
        }

        await self._prompt_tts_provider(query, session_payload, title)
    
    def _format_duration_and_savings(self, metadata: Dict) -> str:
        """Format video duration and calculate time savings from summary."""
        duration = metadata.get('duration', 0)
        
        if duration:
            # Format original duration
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            seconds = duration % 60
            
            if hours > 0:
                duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                duration_str = f"{minutes:02d}:{seconds:02d}"
            
            # Calculate time savings (typical summary reading time is 2-3 minutes)
            reading_time_seconds = 180  # 3 minutes average
            if duration > reading_time_seconds:
                time_saved = duration - reading_time_seconds
                saved_hours = time_saved // 3600
                saved_minutes = (time_saved % 3600) // 60
                
                if saved_hours > 0:
                    savings_str = f"{saved_hours:02d}:{saved_minutes:02d}:00"
                else:
                    savings_str = f"{saved_minutes:02d}:{time_saved % 60:02d}"
                
                return f"⏱️ **Duration**: {duration_str} → ~3 min read (⏰ Saves {savings_str})"
            else:
                return f"⏱️ **Duration**: {duration_str}"
        else:
            return f"⏱️ **Duration**: Unknown"
    
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
        """Handle audio summary generation with TTS."""
        try:
            # Get video metadata
            video_info = result.get('metadata', {})
            title = video_info.get('title', 'Unknown Title')
            channel = video_info.get('uploader') or video_info.get('channel') or 'Unknown Channel'
            
            # Get summary content - handle both old and new summary structures (for audio)
            summary_data = result.get('summary', {})
            summary = 'No summary available'
            
            if isinstance(summary_data, dict):
                # Handle both direct chunked structure and wrapped JSON structure for TTS
                # For TTS, prefer audio-optimized version, fallback to comprehensive
                summary = (summary_data.get('audio') or 
                          summary_data.get('content', {}).get('audio') or
                          summary_data.get('comprehensive') or
                          summary_data.get('content', {}).get('comprehensive') or
                          summary_data.get('summary') or
                          'No audio summary available')
            elif isinstance(summary_data, str):
                summary = summary_data
            
            # Update status to show TTS generation
            await query.edit_message_text(f"🎙️ Generating audio summary... Creating TTS audio file.")
            
            # Generate TTS audio
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_id = video_info.get('video_id', 'unknown')
            audio_filename = f"audio_{video_id}_{timestamp}.mp3"
            
            # Generate the audio file
            audio_filepath = await self.summarizer.generate_tts_audio(summary, audio_filename)
            
            if audio_filepath and Path(audio_filepath).exists():
                # Send the audio as a voice message
                try:
                    with open(audio_filepath, 'rb') as audio_file:
                        await query.message.reply_voice(
                            voice=audio_file,
                            caption=f"🎧 **Audio Summary**: {self._escape_markdown(title)}\n"
                                   f"📺 **Channel**: {self._escape_markdown(channel)}\n\n"
                                   f"🎵 Generated with OpenAI TTS",
                            parse_mode=ParseMode.MARKDOWN
                        )
                    
                    # Also send the text summary
                    text_summary = summary
                    if len(text_summary) > 1000:
                        text_summary = text_summary[:1000] + "..."
                    
                    response_text = f"🎙️ **Audio Summary Generated**\n\n" \
                                  f"📝 **Text Version:**\n{text_summary}\n\n" \
                                  f"✅ Voice message sent above!"
                    
                    await query.edit_message_text(
                        response_text,
                        parse_mode=ParseMode.MARKDOWN
                    )
                    
                    logging.info(f"✅ Successfully sent audio summary for: {title}")
                    
                except Exception as e:
                    logging.error(f"❌ Failed to send voice message: {e}")
                    # Ensure summary is a string for slicing
                    summary_text = str(summary) if summary else "No summary available"
                    await query.edit_message_text(
                        f"❌ Generated audio but failed to send voice message.\n\n"
                        f"**Text Summary:**\n{summary_text[:1000]}{'...' if len(summary_text) > 1000 else ''}"
                    )
            else:
                # TTS generation failed, send text only
                logging.warning("⚠️ TTS generation failed, sending text only")
                metrics.record_tts(False)
                # Ensure summary is a string for slicing
                summary_text = str(summary) if summary else "No summary available"
                response_text = f"🎙️ **Audio Summary** (TTS failed)\n\n" \
                              f"🎬 **{self._escape_markdown(title)}**\n" \
                              f"📺 **Channel**: {self._escape_markdown(channel)}\n\n" \
                              f"📝 **Summary:**\n{summary_text[:1000]}{'...' if len(summary_text) > 1000 else ''}\n\n" \
                              f"⚠️ Audio generation failed. Check TTS configuration."
                
                await query.edit_message_text(
                    response_text,
                    parse_mode=ParseMode.MARKDOWN
                )
                
        except Exception as e:
            logging.error(f"Error handling audio summary: {e}")
            await query.edit_message_text(f"❌ Error generating audio summary: {str(e)[:100]}...")

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
        callback_data = f"delete_{report_id}"
        if len(callback_data.encode('utf-8')) > 64:
            max_id_len = 64 - len("delete_")
            truncated_id = report_id[:max_id_len]
            callback_data = f"delete_{truncated_id}"

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
                InlineKeyboardButton("🗑️ Delete…", callback_data=callback_data)
            ]
        ]

        # Filter out None entries in second row
        keyboard[1] = [btn for btn in keyboard[1] if btn is not None]
        keyboard = [row for row in keyboard if row]

        return InlineKeyboardMarkup(keyboard) if keyboard else None

    def _get_render_client(self):
        if self._render_client:
            return self._render_client
        try:
            client = create_render_client()
            self._render_client = client
            return client
        except Exception as exc:
            logging.debug(f"Render client not available: {exc}")
            return None

    def _upload_audio_to_render(self, content_id: str, audio_path: Path) -> None:
        client = self._get_render_client()
        if not client:
            return

        if not audio_path.exists():
            logging.warning(f"⚠️ Render upload skipped; audio file missing: {audio_path}")
            return

        try:
            client.upload_audio_file(audio_path, content_id)
            logging.info(f"✅ Uploaded audio to Render for {content_id}")
        except Exception as exc:
            logging.warning(f"⚠️ Render audio upload failed for {content_id}: {exc}")

    def _sync_audio_to_targets(self, video_id: str, audio_path: Path, ledger_id: Optional[str], summary_type: str) -> None:
        try:
            if not video_id:
                logging.warning("⚠️ Missing video_id for audio sync")
                return
            if not audio_path.exists():
                logging.warning(f"⚠️ Audio file missing for sync: {audio_path}")
                return

            logging.info("🗄️ SYNC: Syncing SQLite database (contains new record + audio metadata)...")

            if ledger_id:
                entry = ledger.get(ledger_id, summary_type)
                if entry:
                    entry["mp3"] = str(audio_path)
                    entry["synced"] = True
                    entry["last_synced"] = datetime.now().isoformat()
                    ledger.upsert(ledger_id, summary_type, entry)
                    logging.info(f"📊 Updated ledger: synced=True, mp3={audio_path.name}")
            else:
                logging.warning("⚠️ ledger_id unknown; skipping ledger mp3 update")

            content_id = f"yt:{video_id}"
            reports_dir = Path("/app/data/reports")
            all_reports = list(reports_dir.glob("*.json"))
            logging.info(f"🔍 Looking for pattern '*{video_id}*.json' in {reports_dir}")
            logging.info(f"🔍 Found {len(all_reports)} JSON files in reports dir")

            report_files = []
            for file_path in all_reports:
                fname = file_path.name
                if (
                    fname == f"{video_id}.json"
                    or fname.endswith(f"_{video_id}.json")
                    or f"_{video_id}_" in fname
                    or (video_id in fname and len(video_id) >= 8)
                ):
                    report_files.append(file_path)

            logging.info(f"🔍 Pattern matching complete. Found {len(report_files)} matches.")

            if not report_files:
                logging.error(f"❌ Could not find JSON report for video_id {video_id}")
                logging.error(f"   Expected file patterns: {video_id}.json, *_{video_id}.json, *{video_id}*.json")
                logging.error("   Will skip dual-sync for this video to avoid syncing wrong content")
                return

            report_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            logging.info(f"🔍 Found {len(report_files)} files matching video_id {video_id}")
            for rf in report_files[:3]:
                logging.info(f"🔍   - {rf.name}")

            report_path = report_files[0]
            logging.info(f"✅ Using report: {report_path.name}")

            sync_results = dual_sync_upload(report_path, audio_path)
            sqlite_ok = bool(sync_results.get('sqlite', {}).get('report')) if isinstance(sync_results, dict) else False
            postgres_ok = bool(sync_results.get('postgres', {}).get('report')) if isinstance(sync_results, dict) else False

            if sqlite_ok or postgres_ok:
                targets = []
                if sqlite_ok:
                    targets.append("SQLite")
                if postgres_ok:
                    targets.append("PostgreSQL")
                logging.info(f"✅ CONTENT+AUDIO SYNCED: 📊+🎵 → {content_id} (targets: {', '.join(targets)})")
            else:
                logging.warning(f"⚠️ Content+audio sync failed for {content_id}")

        except Exception as sync_e:
            logging.error(f"❌ Audio sync error: {sync_e}")

    async def _send_long_message(self, query, header_text: str, summary_text: str, reply_markup=None):
        """Send long messages by splitting into multiple Telegram messages if needed."""
        try:
            summary_text = summary_text or ""
            safe_summary = self._escape_markdown(summary_text)
            safe_header = header_text or ""

            # Calculate available space for summary content
            # Reserve space for header, formatting, and safety margin
            header_length = len(safe_header)
            safety_margin = 100  # Buffer for formatting and other text
            available_space = self.MAX_MESSAGE_LENGTH - header_length - safety_margin
            
            # If summary fits in one message, send normally
            if len(safe_summary) <= available_space:
                full_message = f"{safe_header}\n{safe_summary}"
                msg = await query.edit_message_text(full_message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
                return msg
            
            # Summary is too long - split into multiple messages
            print(f"📱 Long summary detected ({len(summary_text):,} chars) - splitting into multiple messages")
            
            # Split summary into chunks that fit within message limits
            chunks = self._split_text_into_chunks(safe_summary, available_space)
            
            # Send first message with header + first chunk
            first_message = f"{safe_header}\n{chunks[0]}"
            if len(chunks) > 1:
                first_message += f"\n\n📄 *Continued in next message... ({len(chunks)} parts total)*"
            
            await query.edit_message_text(first_message, parse_mode=ParseMode.MARKDOWN)
            
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
                msg = await query.edit_message_text(fallback_message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
                return msg
            except Exception as fallback_e:
                logging.error(f"Even fallback message failed: {fallback_e}")
                # Try without buttons as last resort
                msg = await query.edit_message_text(fallback_message, parse_mode=ParseMode.MARKDOWN)
                return msg
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into chunks that fit within Telegram message limits."""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Calculate end position for this chunk
            end_pos = current_pos + max_chunk_size
            
            if end_pos >= len(text):
                # Last chunk - take the rest
                chunks.append(text[current_pos:].strip())
                break
            
            # Find a good break point (prefer paragraph breaks, then sentences)
            break_point = end_pos
            
            # Look for paragraph break (double newline) within last 200 chars
            paragraph_break = text.rfind('\n\n', current_pos, end_pos - 200)
            if paragraph_break > current_pos:
                break_point = paragraph_break
            else:
                # Look for sentence break within last 100 chars
                sentence_break = text.rfind('. ', current_pos, end_pos - 100)
                if sentence_break > current_pos:
                    break_point = sentence_break + 1  # Include the period

            # Avoid ending chunk on trailing backslash which would escape next chunk's first character
            while break_point > current_pos and text[break_point - 1] == '\\':
                break_point -= 1
            
            # Add this chunk
            chunk = text[current_pos:break_point].strip()
            if chunk:
                chunks.append(chunk)
            
            current_pos = break_point
            
            # Skip whitespace at the start of next chunk
            while current_pos < len(text) and text[current_pos].isspace():
                current_pos += 1
        
        return chunks
    
    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram Markdown (minimal escaping)."""
        if not text:
            return ""
        
        # Only escape truly problematic characters for Telegram
        escape_chars = ['_', '*', '[', ']', '`']
        
        escaped_text = text
        for char in escape_chars:
            escaped_text = escaped_text.replace(char, f'\\{char}')
        
        return escaped_text
    
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
