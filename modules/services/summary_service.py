from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from modules.metrics import metrics
from modules import ledger, render_probe
from modules.report_generator import create_report_from_youtube_summarizer
from nas_sync import dual_sync_upload


async def send_formatted_response(handler, query, result: Dict[str, Any], summary_type: str, export_info: Optional[Dict] = None) -> None:
    try:
        video_info = result.get('metadata', {})
        source = result.get('content_source') or handler._current_source()
        title = result.get('title') or video_info.get('title') or 'Untitled content'
        channel = (
            video_info.get('uploader')
            or video_info.get('channel')
            or video_info.get('author')
            or video_info.get('subreddit')
            or 'Unknown source'
        )
        duration_info = handler._format_duration_and_savings(video_info)

        universal_id = result.get('id') or video_info.get('video_id') or handler._current_content_id() or ''
        video_id = handler._normalize_content_id(universal_id)
        base_type = (summary_type or '').split(':', 1)[0]

        summary_data = result.get('summary', {})
        summary = 'No summary available'

        if isinstance(summary_data, dict):
            if summary_type == "audio":
                summary = (
                    summary_data.get('audio')
                    or summary_data.get('content', {}).get('audio')
                    or summary_data.get('content', {}).get('comprehensive')
                    or summary_data.get('comprehensive')
                    or summary_data.get('summary')
                    or 'No audio summary available'
                )
            elif summary_type == "bullet-points":
                summary = (
                    summary_data.get('bullet_points')
                    or summary_data.get('content', {}).get('bullet_points')
                    or summary_data.get('content', {}).get('comprehensive')
                    or summary_data.get('comprehensive')
                    or summary_data.get('summary')
                    or 'No bullet points available'
                )
            elif summary_type == "key-insights":
                summary = (
                    summary_data.get('key_insights')
                    or summary_data.get('content', {}).get('key_insights')
                    or summary_data.get('content', {}).get('comprehensive')
                    or summary_data.get('comprehensive')
                    or summary_data.get('summary')
                    or 'No key insights available'
                )
            else:
                summary = (
                    summary_data.get('comprehensive')
                    or summary_data.get('content', {}).get('comprehensive')
                    or summary_data.get('content', {}).get('audio')
                    or summary_data.get('audio')
                    or summary_data.get('summary')
                    or 'No comprehensive summary available'
                )
        elif isinstance(summary_data, str):
            summary = summary_data

        source_icon = {
            'youtube': '🎬',
            'reddit': '🧵',
            'web': '📰',
        }.get(source, '🎬')
        channel_icon = '👤' if source == 'reddit' else '📺'
        header_parts = [
            f"{source_icon} **{handler._escape_markdown(title)}**",
            f"{channel_icon} {handler._escape_markdown(channel)}",
            duration_info,
            "",
            f"📝 **{summary_type.replace('-', ' ').title()} Summary:**"
        ]

        header_text = "\n".join(part for part in header_parts if part)

        reply_markup = None
        if export_info and (export_info.get('html_path') or export_info.get('json_path')):
            dashboard_url = (
                os.getenv('DASHBOARD_URL')
                or os.getenv('POSTGRES_DASHBOARD_URL')
                or 'https://ytv2-dashboard-postgres.onrender.com'
            )

            report_id = None
            if export_info.get('json_path'):
                json_path = Path(export_info['json_path'])
                report_id = json_path.stem
            elif export_info.get('html_path'):
                html_path = Path(export_info['html_path'])
                report_id = html_path.stem

            if dashboard_url:
                keyboard = []

                report_id_encoded = urllib.parse.quote(report_id, safe='') if report_id else ''

                row1 = [InlineKeyboardButton("📊 Dashboard", url=dashboard_url)]
                if report_id_encoded:
                    row1.append(InlineKeyboardButton("📄 Open Summary", url=f"{dashboard_url}#report={report_id_encoded}"))
                keyboard.append(row1)

                if report_id_encoded:
                    listen_cb = f"listen_this:{video_id}:{base_type}"
                    gen_cb = f"gen_quiz:{video_id}"
                    row2 = []
                    if len(listen_cb.encode('utf-8')) <= 64:
                        row2.append(InlineKeyboardButton("▶️ Listen", callback_data=listen_cb))
                    if len(gen_cb.encode('utf-8')) <= 64:
                        row2.append(InlineKeyboardButton("🧩 Generate Quiz", callback_data=gen_cb))
                    if row2:
                        keyboard.append(row2)

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
                logging.warning("⚠️ No DASHBOARD_URL set - skipping link buttons")

        sent_msg = await handler._send_long_message(query, header_text, summary, reply_markup)

        if summary_type.startswith("audio"):
            await handler._prepare_tts_generation(query, result, summary, summary_type)

    except Exception as exc:
        logging.error("Error sending formatted response: %s", exc)
        await query.edit_message_text(f"❌ Error formatting response. The summary was generated but couldn't be displayed properly.")


async def prepare_tts_generation(handler, query, result: Dict[str, Any], summary_text: str, summary_type: str) -> None:
    video_info = result.get('metadata', {})
    title = video_info.get('title', 'Unknown Title')

    ledger_id = (
        result.get('id')
        or video_info.get('content_id')
        or (handler.current_item or {}).get('content_id')
    )

    normalized_video_id = video_info.get('video_id')
    if not normalized_video_id and ledger_id:
        normalized_video_id = handler._normalize_content_id(ledger_id)

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

    await handler._prompt_tts_provider(query, session_payload, title)


async def handle_audio_summary(handler, query, result: Dict[str, Any], summary_type: str) -> None:
    try:
        video_info = result.get('metadata', {})
        title = video_info.get('title', 'Unknown Title')
        channel = video_info.get('uploader') or video_info.get('channel') or 'Unknown Channel'

        summary_data = result.get('summary', {})
        summary = 'No summary available'

        if isinstance(summary_data, dict):
            summary = (
                summary_data.get('audio')
                or summary_data.get('content', {}).get('audio')
                or summary_data.get('comprehensive')
                or summary_data.get('content', {}).get('comprehensive')
                or summary_data.get('summary')
                or 'No audio summary available'
            )
        elif isinstance(summary_data, str):
            summary = summary_data

        await query.edit_message_text("🎙️ Generating audio summary... Creating TTS audio file.")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_id = video_info.get('video_id', 'unknown')
        audio_filename = f"audio_{video_id}_{timestamp}.mp3"

        audio_filepath = await handler.summarizer.generate_tts_audio(summary, audio_filename)

        if audio_filepath and Path(audio_filepath).exists():
            try:
                with open(audio_filepath, 'rb') as audio_file:
                    await query.message.reply_voice(
                        voice=audio_file,
                        caption=(
                            f"🎧 **Audio Summary**: {handler._escape_markdown(title)}\n"
                            f"📺 **Channel**: {handler._escape_markdown(channel)}\n\n"
                            f"🎵 Generated with OpenAI TTS"
                        ),
                        parse_mode=ParseMode.MARKDOWN
                    )

                text_summary = summary
                if len(text_summary) > 1000:
                    text_summary = text_summary[:1000] + "..."

                response_text = (
                    "🎙️ **Audio Summary Generated**\n\n"
                    f"📝 **Text Version:**\n{text_summary}\n\n"
                    "✅ Voice message sent above!"
                )

                await query.edit_message_text(
                    response_text,
                    parse_mode=ParseMode.MARKDOWN
                )

                logging.info("✅ Successfully sent audio summary for: %s", title)

            except Exception as exc:
                logging.error("❌ Failed to send voice message: %s", exc)
                summary_text = str(summary) if summary else "No summary available"
                await query.edit_message_text(
                    "❌ Generated audio but failed to send voice message.\n\n"
                    f"**Text Summary:**\n{summary_text[:1000]}{'...' if len(summary_text) > 1000 else ''}"
                )
        else:
            logging.warning("⚠️ TTS generation failed, sending text only")
            metrics.record_tts(False)
            summary_text = str(summary) if summary else "No summary available"
            response_text = (
                "🎙️ **Audio Summary** (TTS failed)\n\n"
                f"🎬 **{handler._escape_markdown(title)}**\n"
                f"📺 **Channel**: {handler._escape_markdown(channel)}\n\n"
                f"📝 **Summary:**\n{summary_text[:1000]}{'...' if len(summary_text) > 1000 else ''}\n\n"
                "⚠️ Audio generation failed. Check TTS configuration."
            )

            await query.edit_message_text(
                response_text,
                parse_mode=ParseMode.MARKDOWN
            )

    except Exception as exc:
        logging.error("Error handling audio summary: %s", exc)
        await query.edit_message_text(f"❌ Error generating audio summary: {str(exc)[:100]}...")


async def send_existing_summary_notice(handler, query, video_id: str, summary_type: str) -> None:
    variants = handler._discover_summary_types(video_id)
    message_lines = [
        f"✅ {handler._friendly_variant_label(summary_type)} is already on the dashboard."
    ]
    if variants:
        message_lines.append("\nAvailable variants:")
        message_lines.extend(f"• {handler._friendly_variant_label(variant)}" for variant in sorted(variants))
    message_lines.append("\nOpen the summary or re-run a variant below.")

    normalized_id = handler._normalize_content_id(video_id)
    reply_markup = handler._build_summary_keyboard(variants, normalized_id)
    await query.edit_message_text(
        "\n".join(message_lines),
        reply_markup=reply_markup
    )


async def generate_tts_audio_file(handler, summary_text: str, video_id: str, json_path: Path) -> Optional[str]:
    if not summary_text:
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_filename = f"audio_{video_id}_{timestamp}.mp3"

    loop = asyncio.get_running_loop()

    def _run() -> Optional[str]:
        return asyncio.run(handler.summarizer.generate_tts_audio(summary_text, audio_filename, str(json_path)))

    audio_filepath = await loop.run_in_executor(None, _run)
    if audio_filepath and Path(audio_filepath).exists():
        metrics.record_tts(True)
    else:
        metrics.record_tts(False)
    return audio_filepath if audio_filepath else None


async def process_content_summary(
    handler,
    query,
    summary_type: str,
    user_name: str,
    proficiency_level: Optional[str] = None,
) -> None:
    item = handler.current_item or {}
    content_id = item.get("content_id")
    source = item.get("source", "youtube")
    url = item.get("url")

    if not content_id:
        await query.edit_message_text("❌ No content in context. Please send a link first.")
        return

    if not url:
        url = handler._resolve_video_url(content_id)

    if not url:
        await query.edit_message_text("❌ Could not resolve the source URL. Please resend the link.")
        return

    if not handler.summarizer:
        await query.edit_message_text("❌ Summarizer not available. Please try /status for more info.")
        return

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

    base_type = summary_type.split(':')[0]
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
        normalized_id = handler._normalize_content_id(content_id)
        video_id = normalized_id
        ledger_id = content_id
        display_id = f"{source}:{normalized_id}" if source != "youtube" else normalized_id

        entry = ledger.get(ledger_id, summary_type)
        if entry:
            logging.info(f"📄 Found existing entry for {display_id}:{summary_type}")
            if render_probe.render_has(entry.get("stem")):
                await send_existing_summary_notice(handler, query, ledger_id, summary_type)
                logging.info(f"♻️ SKIPPED: {display_id} already on dashboard")
                return
            else:
                logging.info("🔄 Content exists in database but missing from Dashboard - processing fresh")

        logging.info("🎬 PROCESSING: %s | %s | user: %s | URL: %s", display_id, summary_type, user_name, url)
        logging.info("🧠 LLM: %s/%s", handler.summarizer.llm_provider, handler.summarizer.model)

        if source == "reddit":
            result = await handler.summarizer.process_reddit_thread(
                url,
                summary_type=summary_type,
                proficiency_level=proficiency_level
            )
        elif source == "web":
            result = await handler.summarizer.process_web_page(
                url,
                summary_type=summary_type,
                proficiency_level=proficiency_level
            )
        else:
            result = await handler.summarizer.process_video(
                url,
                summary_type=summary_type,
                proficiency_level=proficiency_level
            )

        if not result:
            await query.edit_message_text("❌ Failed to process content. Please check the URL and try again.")
            return

        error_message = result.get('error') if isinstance(result, dict) else None
        if error_message:
            if 'No transcript available' in error_message:
                await query.edit_message_text(
                    f"⚠️ {error_message}\n\nSkipping this item to prevent empty dashboard entries."
                )
                logging.info("❌ ABORTED: %s", error_message)
            else:
                await query.edit_message_text(f"❌ {error_message}")
                logging.info("❌ Processing error: %s", error_message)
            return

        export_info = {"html_path": None, "json_path": None}
        try:
            report_dict = create_report_from_youtube_summarizer(result)
            json_path = handler.json_exporter.save_report(report_dict)
            export_info["json_path"] = Path(json_path).name

            json_path_obj = Path(json_path)
            if json_path_obj.exists():
                logging.info("✅ Exported JSON report: %s", json_path)
            else:
                logging.warning("⚠️ JSON export returned path but file not created: %s", json_path)
                logging.warning("   This will cause dual-sync to fail!")

            stem = Path(json_path).stem
            ledger_entry = {
                "stem": stem,
                "json": str(json_path),
                "mp3": None,
                "synced": False,
                "created_at": datetime.now().isoformat()
            }

            if proficiency_level:
                ledger_entry["proficiency"] = proficiency_level
                if summary_type.startswith("audio-"):
                    lang_code = "fr" if summary_type.startswith("audio-fr") else "es"
                    ledger_entry["target_language"] = lang_code
                    ledger_entry["language_flag"] = "🇫🇷" if lang_code == "fr" else "🇪🇸"
                    ledger_entry["learning_mode"] = True

                    proficiency_badges = {
                        "beginner": {"fr": "🟢 Débutant", "es": "🟢 Principiante"},
                        "intermediate": {"fr": "🟡 Intermédiaire", "es": "🟡 Intermedio"},
                        "advanced": {"fr": "🔵 Avancé", "es": "🔵 Avanzado"}
                    }
                    if proficiency_level in proficiency_badges:
                        ledger_entry["proficiency_badge"] = proficiency_badges[proficiency_level][lang_code]

            ledger.upsert(ledger_id, summary_type, ledger_entry)
            logging.info("📊 Added to ledger: %s:%s", display_id, summary_type)

            is_audio_summary = summary_type == "audio" or summary_type.startswith("audio-fr") or summary_type.startswith("audio-es")

            if not is_audio_summary:
                try:
                    json_path_obj = Path(json_path)
                    stem = json_path_obj.stem

                    logging.info("📡 DUAL-SYNC START: Uploading to configured targets...")

                    video_metadata = result.get('metadata', {})
                    result_content_id = result.get('id') or (ledger_id if ledger_id else stem)

                    report_path = Path(f"/app/data/reports/{stem}.json")
                    sync_results = dual_sync_upload(report_path)

                    sqlite_ok = bool(sync_results.get('sqlite', {}).get('report')) if isinstance(sync_results, dict) else False
                    postgres_ok = bool(sync_results.get('postgres', {}).get('report')) if isinstance(sync_results, dict) else False
                    sync_success = sqlite_ok or postgres_ok

                    if sync_success:
                        targets = []
                        if sqlite_ok: targets.append("SQLite")
                        if postgres_ok: targets.append("PostgreSQL")
                        logging.info("✅ DUAL-SYNC SUCCESS: 📊 → %s (targets: %s)", result_content_id, ', '.join(targets))

                        entry = ledger.get(ledger_id, summary_type)
                        if entry:
                            entry["synced"] = True
                            entry["last_synced"] = datetime.now().isoformat()
                            entry["sync_targets"] = targets
                            ledger.upsert(ledger_id, summary_type, entry)
                            logging.info("📊 Updated ledger: synced=True, targets=%s", targets)
                    else:
                        logging.error("❌ DUAL-SYNC FAILED: All targets failed for %s", stem)

                except Exception as sync_e:
                    logging.warning("⚠️ Dual-sync error: %s", sync_e)
            else:
                try:
                    json_path_obj = Path(json_path)
                    stem = json_path_obj.stem

                    logging.info("📡 DUAL-SYNC (content-only): Audio summary - syncing metadata for %s", result.get('id'))

                    report_path = json_path_obj
                    max_retries = 3
                    for attempt in range(max_retries):
                        if report_path.exists():
                            break
                        logging.debug("📄 Waiting for file to be written (attempt %s/%s): %s", attempt + 1, max_retries, report_path)
                        time.sleep(0.1)

                    if report_path.exists():
                        sync_results = dual_sync_upload(report_path)
                        sqlite_ok = bool(sync_results.get('sqlite', {}).get('report')) if isinstance(sync_results, dict) else False
                        postgres_ok = bool(sync_results.get('postgres', {}).get('report')) if isinstance(sync_results, dict) else False
                        sync_success = sqlite_ok or postgres_ok

                        if sync_success:
                            targets = []
                            if sqlite_ok: targets.append("SQLite")
                            if postgres_ok: targets.append("PostgreSQL")
                            logging.info("✅ DUAL-SYNC CONTENT: 📊 → %s (targets: %s)", result.get('id'), ', '.join(targets))
                            logging.info("⏳ Audio sync will happen after TTS generation")

                            entry = ledger.get(ledger_id, summary_type)
                            if entry:
                                entry["synced"] = True
                                entry["last_synced"] = datetime.now().isoformat()
                                entry["sync_targets"] = targets
                                ledger.upsert(ledger_id, summary_type, entry)
                                logging.info("📊 Updated ledger: synced=True, targets=%s", targets)
                        else:
                            logging.error("❌ DUAL-SYNC CONTENT FAILED: All targets failed for %s", stem)
                    else:
                        logging.warning("⚠️ JSON report not found for content sync: %s", report_path)

                except Exception as sync_e:
                    logging.warning("⚠️ Dual-sync content error: %s", sync_e)
                    logging.info("⏳ Will retry full sync after TTS generation")

        except Exception as e:
            logging.warning("⚠️ Export failed: %s", e)

        await send_formatted_response(handler, query, result, summary_type, export_info)

    except Exception as e:
        logging.error("Error processing content %s: %s", url, e)
        await query.edit_message_text(f"❌ Error processing content: {str(e)[:100]}...")
