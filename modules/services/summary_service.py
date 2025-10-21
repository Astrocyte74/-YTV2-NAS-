from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from modules.metrics import metrics
from modules.telegram.handlers import ai2ai as ai2ai_handler


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
