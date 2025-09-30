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

from youtube_summarizer import YouTubeSummarizer
from llm_config import llm_config
from nas_sync import upload_to_render, dual_sync_upload


class YouTubeTelegramBot:
    """Telegram bot for YouTube video summarization."""
    
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
        self.last_video_url = None
        
        # Initialize exporters and ensure directories exist
        Path("./data/reports").mkdir(parents=True, exist_ok=True)
        Path("./exports").mkdir(parents=True, exist_ok=True)
        
        self.html_exporter = SummaryExporter("./exports")
        self.json_exporter = JSONReportGenerator("./data/reports")
        
        # YouTube URL regex pattern
        self.youtube_url_pattern = re.compile(
            r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})'
        )
        
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
    
    def setup_handlers(self):
        """Set up bot command and message handlers."""
        if not self.application:
            return
            
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        
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
        logging.info(f"Received message from {user_name} ({user_id}): {message_text[:100]}...")
        
        # Check if message contains YouTube URL
        youtube_match = self.youtube_url_pattern.search(message_text)
        
        if not youtube_match:
            await update.message.reply_text(
                "🔍 Please send a YouTube URL to get started.\n\n"
                "Supported formats:\n"
                "• https://youtube.com/watch?v=...\n"
                "• https://youtu.be/...\n"
                "• https://m.youtube.com/watch?v=..."
            )
            return
        
        # Extract and clean URL
        video_url = self._extract_youtube_url(message_text)
        if not video_url:
            await update.message.reply_text("❌ Could not extract a valid YouTube URL from your message.")
            return
        
        # Store the URL for potential model switching
        self.last_video_url = video_url
        
        # Send processing message with options
        keyboard = [
            [
                InlineKeyboardButton("📝 Comprehensive", callback_data="summarize_comprehensive"),
                InlineKeyboardButton("🎯 Key Points", callback_data="summarize_bullet-points")
            ],
            [
                InlineKeyboardButton("💡 Insights", callback_data="summarize_key-insights"),
                InlineKeyboardButton("🎙️ Audio Summary", callback_data="summarize_audio")
            ],
            [
                InlineKeyboardButton("🎙️ Audio français 🇫🇷", callback_data="summarize_audio-fr"),
                InlineKeyboardButton("🎙️ Audio español 🇪🇸", callback_data="summarize_audio-es")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"🎬 Processing YouTube video...\n\n"
            f"Choose your summary type:",
            reply_markup=reply_markup
        )
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards."""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        user_name = query.from_user.first_name or "Unknown"
        
        if not self._is_user_allowed(user_id):
            await query.edit_message_text("❌ You are not authorized to use this bot.")
            return
        
        callback_data = query.data
        
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
            await self._process_video_summary(query, summary_type, user_name, proficiency_level)
        
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
                    
                dashboard_url = os.getenv('DASHBOARD_URL', 'https://ytv2-vy9k.onrender.com')
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
        
        else:
            await query.edit_message_text("❌ Unknown option selected.")
    
    async def _show_main_summary_options(self, query):
        """Show the main summary type selection buttons"""
        keyboard = [
            [
                InlineKeyboardButton("📝 Comprehensive", callback_data="summarize_comprehensive"),
                InlineKeyboardButton("🎯 Key Points", callback_data="summarize_bullet-points")
            ],
            [
                InlineKeyboardButton("💡 Insights", callback_data="summarize_key-insights"),
                InlineKeyboardButton("🎙️ Audio Summary", callback_data="summarize_audio")
            ],
            [
                InlineKeyboardButton("🎙️ Audio français 🇫🇷", callback_data="summarize_audio-fr"),
                InlineKeyboardButton("🎙️ Audio español 🇪🇸", callback_data="summarize_audio-es")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            f"🎬 Choose your summary type:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
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
    
    async def _process_video_summary(self, query, summary_type: str, user_name: str, proficiency_level: str = None):
        """Process video summarization request."""
        if not self.last_video_url:
            await query.edit_message_text("❌ No YouTube URL found. Please send a URL first.")
            return
        
        if not self.summarizer:
            await query.edit_message_text("❌ Summarizer not available. Please try /status for more info.")
            return
        
        # Update message to show processing
        # Create user-friendly processing messages
        processing_messages = {
            "comprehensive": "📝 Analyzing video and creating comprehensive summary...",
            "bullet-points": "🎯 Extracting key points from the video...", 
            "key-insights": "💡 Identifying key insights and takeaways...",
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
            message = processing_messages.get(base_type, f"🔄 Processing {summary_type}... This may take a moment.")
        
        await query.edit_message_text(message)
        
        try:
            # Extract video ID
            video_id = self.last_video_url.split('v=')[-1].split('&')[0] if 'v=' in self.last_video_url else self.last_video_url.split('/')[-1]
            
            # Check ledger before processing
            entry = ledger.get(video_id, summary_type)
            if entry:
                logging.info(f"📄 Found existing entry for {video_id}:{summary_type}")
                if render_probe.render_has(entry["stem"]):
                    await query.edit_message_text("✅ Already summarized and on dashboard!")
                    logging.info(f"♻️ SKIPPED: {video_id} already on dashboard")
                    return
                else:
                    # Content exists in DB but not Dashboard - proceed with fresh processing
                    logging.info(f"🔄 Content exists in database but missing from Dashboard - processing fresh")
                    # Continue to fresh processing below instead of trying to re-sync files
            
            # Process the video (new processing)
            logging.info(
                f"🎬 PROCESSING: {video_id} | {summary_type} | user: {user_name} | URL: {self.last_video_url}"
            )
            logging.info(
                f"🧠 LLM: {self.summarizer.llm_provider}/{self.summarizer.model}"
            )
            
            result = await self.summarizer.process_video(
                self.last_video_url,
                summary_type=summary_type,
                proficiency_level=proficiency_level
            )
            
            if not result:
                await query.edit_message_text("❌ Failed to process video. Please check the URL and try again.")
                return
                
            # Check for transcript extraction errors - abort to prevent empty dashboard cards
            if result.get('error') and 'No transcript available' in result.get('error', ''):
                await query.edit_message_text(f"⚠️ {result.get('error')}\n\nSkipping this video to prevent empty dashboard entries.")
                logging.info(f"❌ ABORTED: {result.get('error')}")
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
                
                ledger.upsert(video_id, summary_type, ledger_entry)
                logging.info(f"📊 Added to ledger: {video_id}:{summary_type}")
                
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
                        video_id = video_metadata.get('video_id', '')
                        content_id = f"yt:{video_id}" if video_id else stem

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
                            logging.info(f"✅ DUAL-SYNC SUCCESS: 📊 → {content_id} (targets: {', '.join(targets)})")

                            # Update ledger and mark as synced
                            entry = ledger.get(video_id, summary_type)
                            if entry:
                                entry["synced"] = True
                                entry["last_synced"] = datetime.now().isoformat()
                                entry["sync_targets"] = targets  # Track which targets succeeded
                                ledger.upsert(video_id, summary_type, entry)
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
                        video_id = video_metadata.get('video_id', '')
                        content_id = f"yt:{video_id}" if video_id else stem

                        logging.info(f"📡 DUAL-SYNC (content-only): Audio summary - syncing metadata for {content_id}")

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
                                logging.info(f"✅ DUAL-SYNC CONTENT: 📊 → {content_id} (targets: {', '.join(targets)})")
                                logging.info(f"⏳ Audio sync will happen after TTS generation")

                                # Update ledger and mark as synced
                                entry = ledger.get(video_id, summary_type)
                                if entry:
                                    entry["synced"] = True
                                    entry["last_synced"] = datetime.now().isoformat()
                                    entry["sync_targets"] = targets  # Track which targets succeeded
                                    ledger.upsert(video_id, summary_type, entry)
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
            logging.error(f"Error processing video {self.last_video_url}: {e}")
            await query.edit_message_text(f"❌ Error processing video: {str(e)[:100]}...")
    
    async def _send_formatted_response(self, query, result: Dict[str, Any], summary_type: str, export_info: Dict = None):
        """Send formatted summary response."""
        try:
            # Get video metadata
            video_info = result.get('metadata', {})
            title = video_info.get('title', 'Unknown Title')
            channel = video_info.get('uploader') or video_info.get('channel') or 'Unknown Channel'
            duration_info = self._format_duration_and_savings(video_info)
            
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
            header_parts = [
                f"🎬 **{self._escape_markdown(title)}**",
                f"📺 {self._escape_markdown(channel)}",
                duration_info,
                "",
                f"📝 **{summary_type.replace('-', ' ').title()} Summary:**"
            ]
            
            header_text = "\n".join(part for part in header_parts if part)
            
            # Create inline keyboard with link buttons if exports were successful
            reply_markup = None
            if export_info and (export_info.get('html_path') or export_info.get('json_path')):
                dashboard_url = os.getenv('DASHBOARD_URL', 'https://ytv2-vy9k.onrender.com')
                
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
                    
                    # Create button row with all three buttons
                    button_row = [
                        InlineKeyboardButton("📊 Dashboard", url=dashboard_url)
                    ]
                    
                    if report_id_encoded:
                        button_row.append(
                            InlineKeyboardButton("📄 Open summary", 
                                                url=f"{dashboard_url}#report={report_id_encoded}")
                        )
                        # Limit callback data to avoid Telegram's 64 byte limit
                        callback_data = f"delete_{report_id}"
                        if len(callback_data.encode('utf-8')) > 64:
                            # Truncate report_id if too long
                            max_id_len = 64 - len("delete_")
                            truncated_id = report_id[:max_id_len]
                            callback_data = f"delete_{truncated_id}"
                        
                        button_row.append(
                            InlineKeyboardButton("🗑️ Delete…", callback_data=callback_data)
                        )
                    
                    keyboard.append(button_row)
                    reply_markup = InlineKeyboardMarkup(keyboard)
                else:
                    reply_markup = None
                    logging.warning("⚠️ No DASHBOARD_URL set - skipping link buttons")
            
            # Always send the full text (it will auto-split via _send_long_message)
            # The 'summary' variable already contains the correct text for the chosen summary type
            await self._send_long_message(query, header_text, summary, reply_markup)
            
            logging.info(f"Successfully sent {summary_type} summary for {title}")
            
            # Generate TTS audio for audio summaries (after text is sent)
            if summary_type == "audio" or summary_type.startswith("audio-fr") or summary_type.startswith("audio-es"):
                await self._generate_and_send_tts(query, result, summary, summary_type)
            
        except Exception as e:
            logging.error(f"Error sending formatted response: {e}")
            await query.edit_message_text("❌ Error formatting response. The summary was generated but couldn't be displayed properly.")
    
    async def _generate_and_send_tts(self, query, result: Dict[str, Any], summary_text: str, summary_type: str):
        """Generate TTS audio and send as voice message (separate from text summary)."""
        try:
            # Get video metadata  
            video_info = result.get('metadata', {})
            title = video_info.get('title', 'Unknown Title')
            
            # Generate TTS audio
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_id = video_info.get('video_id', 'unknown')
            audio_filename = f"audio_{video_id}_{timestamp}.mp3"
            
            # Generate placeholder path for SQLite-only workflow (no JSON files needed)
            json_filepath = f"yt_{video_id}_placeholder.json"  # Used for video ID extraction in SQLite updates
            logging.info(f"📊 Using SQLite-only workflow for video_id: {video_id}")

            # Generate the audio file (run in executor to avoid blocking event loop)
            logging.info(f"🎙️ Generating TTS audio for: {title}")
            loop = asyncio.get_running_loop()
            audio_filepath = await loop.run_in_executor(
                None, 
                lambda: asyncio.run(self.summarizer.generate_tts_audio(summary_text, audio_filename, json_filepath))
            )
            
            if audio_filepath and Path(audio_filepath).exists():
                # Send the audio as a voice message
                try:
                    # Create buttons for the audio message too
                    audio_reply_markup = None
                    report_id = video_info.get('video_id', '')
                    if report_id:
                        dashboard_url = os.getenv('DASHBOARD_URL', 'https://ytv2-vy9k.onrender.com')
                        if dashboard_url:
                            # Encode report ID for URL safety
                            report_id_encoded = urllib.parse.quote(report_id, safe='')
                            
                            # Create button row with all three buttons
                            # Limit callback data to avoid Telegram's 64 byte limit
                            callback_data = f"delete_{report_id}"
                            if len(callback_data.encode('utf-8')) > 64:
                                # Truncate report_id if too long
                                max_id_len = 64 - len("delete_")
                                truncated_id = report_id[:max_id_len]
                                callback_data = f"delete_{truncated_id}"
                            
                            button_row = [
                                InlineKeyboardButton("📊 Dashboard", url=dashboard_url),
                                InlineKeyboardButton("📄 Open summary", 
                                                   url=f"{dashboard_url}#report={report_id_encoded}"),
                                InlineKeyboardButton("🗑️ Delete…", callback_data=callback_data)
                            ]
                            audio_reply_markup = InlineKeyboardMarkup([button_row])
                    
                    with open(audio_filepath, 'rb') as audio_file:
                        await query.message.reply_voice(
                            voice=audio_file,
                            caption=f"🎧 **Audio Summary**: {self._escape_markdown(title)}\n"
                                   f"🎵 Generated with OpenAI TTS",
                            parse_mode=ParseMode.MARKDOWN,
                            reply_markup=audio_reply_markup
                        )
                    logging.info(f"✅ Successfully sent audio summary for: {title}")
                    
                    # Sync SQLite database to Render (SQLite-only workflow)
                    try:
                        video_id = video_info.get('video_id', '')
                        if video_id and audio_filepath and Path(audio_filepath).exists():
                            logging.info(f"🗄️ SYNC: Syncing SQLite database (contains new record + audio metadata)...")
                            
                            # Update ledger with audio path 
                            entry = ledger.get(video_id, summary_type)
                            if entry:
                                entry["mp3"] = str(audio_filepath)
                                entry["synced"] = True
                                entry["last_synced"] = datetime.now().isoformat()
                                ledger.upsert(video_id, summary_type, entry)
                                logging.info(f"📊 Updated ledger: synced=True, mp3={Path(audio_filepath).name}")
                            
                            # Dual-sync content + audio to configured targets (T-Y020A)
                            try:
                                content_id = f"yt:{video_id}"
                                logging.info(f"📡 Dual-syncing content+audio: {content_id}")

                                # Find the JSON report for this content
                                reports_dir = Path("/app/data/reports")
                                report_pattern = f"*{video_id}*.json"

                                # Debug: list what files exist
                                all_reports = list(reports_dir.glob("*.json"))
                                logging.info(f"🔍 Looking for pattern '{report_pattern}' in {reports_dir}")
                                logging.info(f"🔍 Found {len(all_reports)} JSON files in reports dir")

                                # Find exact match by video_id (improved matching)
                                # Look for files that end with video_id.json or contain _video_id pattern
                                report_files = []
                                logging.info(f"🔍 Searching for video_id: '{video_id}' (length: {len(video_id)})")

                                for f in all_reports:
                                    fname = f.name
                                    logging.debug(f"🔍 Checking file: {fname}")

                                    # Match patterns: video_id.json, *_video_id.json, or *video_id*.json (but prefer exact endings)
                                    if fname == f"{video_id}.json":
                                        report_files.append(f)
                                        logging.info(f"🔍 Exact match (pattern 1): {fname}")
                                    elif fname.endswith(f"_{video_id}.json"):
                                        report_files.append(f)
                                        logging.info(f"🔍 Exact match (pattern 2): {fname}")
                                    elif f"_{video_id}_" in fname:
                                        report_files.append(f)
                                        logging.info(f"🔍 Exact match (pattern 3): {fname}")
                                    elif video_id in fname and len(video_id) >= 8:  # Only for long video IDs to avoid false matches
                                        report_files.append(f)
                                        logging.info(f"🔍 Substring match: {fname}")

                                logging.info(f"🔍 Pattern matching complete. Found {len(report_files)} matches.")

                                if report_files:
                                    # Sort by modification time, use most recent
                                    report_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                                    logging.info(f"🔍 Found {len(report_files)} files matching video_id {video_id}")
                                    for rf in report_files[:3]:  # Show top 3
                                        logging.info(f"🔍   - {rf.name}")
                                else:
                                    logging.info(f"🔍 No files found containing video_id {video_id}")
                                    if all_reports:
                                        # Show some recent files for debugging
                                        recent_reports = sorted(all_reports, key=lambda p: p.stat().st_mtime, reverse=True)[:3]
                                        logging.info(f"🔍 Recent files: {[f.name for f in recent_reports]}")

                                if report_files:
                                    report_path = report_files[0]  # Use most recent
                                    logging.info(f"✅ Using report: {report_path.name}")

                                    # Dual-sync with both content and audio
                                    audio_path = Path(audio_filepath) if audio_filepath else None
                                    sync_results = dual_sync_upload(report_path, audio_path)

                                    # Check results
                                    sqlite_ok = bool(sync_results.get('sqlite', {}).get('report')) if isinstance(sync_results, dict) else False
                                    postgres_ok = bool(sync_results.get('postgres', {}).get('report')) if isinstance(sync_results, dict) else False

                                    if sqlite_ok or postgres_ok:
                                        targets = []
                                        if sqlite_ok: targets.append("SQLite")
                                        if postgres_ok: targets.append("PostgreSQL")
                                        logging.info(f"✅ CONTENT+AUDIO SYNCED: 📊+🎵 → {content_id} (targets: {', '.join(targets)})")
                                    else:
                                        logging.warning(f"⚠️ Content+audio sync failed for {content_id}")
                                else:
                                    logging.error(f"❌ Could not find JSON report for video_id {video_id}")
                                    logging.error(f"   Expected file patterns: {video_id}.json, *_{video_id}.json, *{video_id}*.json")
                                    logging.error(f"   Will skip dual-sync for this video to avoid syncing wrong content")

                            except Exception as db_sync_error:
                                logging.error(f"❌ Dual-sync error: {db_sync_error}")
                        else:
                            logging.warning("⚠️ Missing video_id or audio file for sync")
                    except Exception as sync_e:
                        logging.error(f"❌ Audio sync error: {sync_e}")
                    
                except Exception as e:
                    logging.error(f"❌ Failed to send voice message: {e}")
            else:
                logging.warning("⚠️ TTS generation failed")
                
        except Exception as e:
            logging.error(f"Error generating TTS audio: {e}")
    
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
                await query.edit_message_text(full_message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
                return
            
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
                await query.message.reply_text(chunk_message, parse_mode=ParseMode.MARKDOWN, reply_markup=chunk_reply_markup)
                
        except Exception as e:
            logging.error(f"Error sending long message: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            # Fallback to truncated message with buttons
            truncated_summary = safe_summary[:1000] + "..." if len(safe_summary) > 1000 else safe_summary
            fallback_message = f"{safe_header}\n{truncated_summary}\n\n⚠️ *Summary was truncated due to length. View full summary on dashboard.*"
            try:
                await query.edit_message_text(fallback_message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
            except Exception as fallback_e:
                logging.error(f"Even fallback message failed: {fallback_e}")
                # Try without buttons as last resort
                await query.edit_message_text(fallback_message, parse_mode=ParseMode.MARKDOWN)
    
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
