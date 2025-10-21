from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from modules.metrics import metrics
from modules.services import sync_service
from modules.tts_hub import LocalTTSUnavailable, TTSHubClient
from modules.tts_queue import enqueue as enqueue_tts_job
from youtube_summarizer import YouTubeSummarizer


async def prompt_provider(handler, query, session_payload: Dict[str, Any], title: str) -> None:
    base_hint = session_payload.get('tts_base')
    client = handler.tts_client or handler._resolve_tts_client(base_hint)
    if client and client.base_api_url:
        handler.tts_client = client
        session_payload['tts_base'] = client.base_api_url
    else:
        session_payload['tts_base'] = None

    prompt_text = f"🎙️ Choose how to generate audio for **{handler._escape_markdown(title)}**"
    prompt_message = await query.message.reply_text(
        prompt_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=handler._build_provider_keyboard(include_local=True),
    )
    handler._store_tts_session(prompt_message.chat_id, prompt_message.message_id, session_payload)


async def handle_local_unavailable(handler, query, session: Dict[str, Any], message: str = "") -> None:
    logging.warning("Local TTS unavailable: %s", message)
    notice = "⚠️ Local TTS hub unavailable. Queue the job for later or use OpenAI now?"
    await query.edit_message_text(notice, reply_markup=handler._build_local_failure_keyboard())


async def enqueue_job(handler, query, session: Dict[str, Any]) -> None:
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
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Close", callback_data="tts_cancel")]]),
    )
    logging.info("Queued TTS job at %s", path)
    handler._remove_tts_session(query.message.chat_id, query.message.message_id)


async def execute_job(handler, query, session: Dict[str, Any], provider: str) -> None:
    provider_key = (provider or "openai").lower()
    summary_text = session.get("summary_text") or session.get("text") or ""
    if not summary_text:
        logging.warning("TTS: session missing text; aborting")
        await query.answer("Missing summary text", show_alert=True)
        return
    placeholders = session.get("placeholders") or {}
    audio_filename = placeholders.get("audio_filename") or f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    json_placeholder = placeholders.get("json_placeholder") or f"tts_{int(time.time())}.json"
    selected_voice = session.get("selected_voice") or {}
    favorite_slug = selected_voice.get("favorite_slug")
    voice_id = selected_voice.get("voice_id")
    engine_id = selected_voice.get("engine")

    await query.answer(f"Generating audio via {provider_key.title()}…")
    provider_label = "Local TTS hub" if provider_key == "local" else "OpenAI TTS"

    voice_label = session.get("last_voice") or ""
    if not voice_label:
        try:
            slug_hint = None
            if favorite_slug:
                slug_hint = f"fav|{favorite_slug}"
            elif voice_id:
                slug_hint = f"cat|{voice_id}"
            if slug_hint:
                voice_label = handler._tts_voice_label(session, slug_hint)
        except Exception:
            voice_label = ""

    status_msg = None
    try:
        status_text = (
            f"⏳ Generating TTS"
            + (f" • {handler._escape_markdown(voice_label)}" if voice_label else "")
            + f" • {provider_label}"
        )
        status_msg = await query.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        status_msg = None

    async def _update_status(message: str) -> None:
        if not status_msg:
            return
        try:
            await status_msg.edit_text(message, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass

    if not handler.summarizer:
        handler.summarizer = YouTubeSummarizer()

    try:
        logging.info(
            "🧩 Starting TTS generation via %s | title=%s",
            provider_key,
            session.get("title"),
        )
        audio_filepath = await handler.summarizer.generate_tts_audio(
            summary_text,
            audio_filename,
            json_placeholder,
            provider=provider_key,
            voice=voice_id,
            engine=engine_id,
            favorite_slug=favorite_slug,
        )
    except LocalTTSUnavailable as exc:
        logging.warning("Local TTS unavailable during execution: %s", exc)
        await handle_local_unavailable(handler, query, session, message=str(exc))
        await _update_status(
            f"⚠️ Local TTS unavailable"
            + (f" • {handler._escape_markdown(voice_label)}" if voice_label else "")
            + f" • {provider_label}"
        )
        return
    except Exception as exc:
        logging.error("TTS synthesis error: %s", exc)
        await query.answer("TTS failed", show_alert=True)
        await _update_status(
            f"❌ TTS failed"
            + (f" • {handler._escape_markdown(voice_label)}" if voice_label else "")
            + f" • {provider_label}"
        )
        return

    if not audio_filepath or not Path(audio_filepath).exists():
        logging.warning("TTS generation returned no audio")
        await query.answer("TTS generation failed", show_alert=True)
        await _update_status(
            f"❌ TTS failed"
            + (f" • {handler._escape_markdown(voice_label)}" if voice_label else "")
            + f" • {provider_label}"
        )
        return

    logging.info("📦 TTS file ready: %s", audio_filepath)
    await finalize_delivery(handler, query, session, Path(audio_filepath), provider_key)

    try:
        if status_msg:
            done_text = (
                f"✅ Generated"
                + (f" • {handler._escape_markdown(voice_label)}" if voice_label else "")
                + f" • {provider_label}"
            )
            await status_msg.edit_text(done_text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        pass


async def handle_callback(handler, query, callback_data: str) -> None:
    logging.info("🎛️ TTS callback: %s", callback_data)
    message = query.message
    if not message:
        return
    chat_id = message.chat_id
    message_id = message.message_id
    session = handler._get_tts_session(chat_id, message_id)
    if session:
        logging.debug("TTS session found for %s:%s keys=%s", chat_id, message_id, list(session.keys()))
    else:
        logging.warning("TTS session missing for %s:%s", chat_id, message_id)

    if not session:
        await query.answer("Session expired", show_alert=True)
        try:
            await query.edit_message_text("⚠️ This TTS session has expired. Send /tts again.")
        except Exception:
            pass
        return

    client = handler._resolve_tts_client(session.get('tts_base'))
    if not client or not client.base_api_url:
        logging.warning("TTS hub unavailable during TTS callback")
        await query.answer("TTS hub unavailable", show_alert=True)
        return

    if callback_data == "tts_cancel":
        handler._remove_tts_session(chat_id, message_id)
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
                    await handler._refresh_tts_catalog(query, session)
                    await query.answer("Favorites selected")
                else:
                    await query.answer("No favorites available", show_alert=True)
                return
            elif mode_value == "all":
                session['voice_mode'] = 'all'
                session['selected_gender'] = None
                session['selected_family'] = None
                await handler._refresh_tts_catalog(query, session)
                await query.answer("Showing all voices")
                return
        if callback_data == "tts_nop":
            await query.answer("Select an option below")
            return
        if callback_data.startswith("tts_gender:"):
            value = callback_data.split(":", 1)[1]
            session["selected_gender"] = None if value in ("all", "") else value
            session["selected_family"] = None
            await handler._refresh_tts_catalog(query, session)
            await query.answer("Gender updated")
            return
        if callback_data.startswith("tts_accent:"):
            value = callback_data.split(":", 1)[1]
            session["selected_family"] = None if value in ("all", "") else value
            await handler._refresh_tts_catalog(query, session)
            await query.answer("Accent updated")
            return

    if callback_data.startswith("tts_provider:"):
        provider = callback_data.split(":", 1)[1]
        session['provider'] = provider
        if provider == 'local':
            client = handler._resolve_tts_client(session.get('tts_base'))
            session['tts_base'] = client.base_api_url if client else None
            catalog = session.get('catalog')
            favorites = session.get('favorites')
            if client and not catalog:
                try:
                    catalog = await client.fetch_catalog()
                    session['catalog'] = catalog
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
                    await handle_local_unavailable(handler, query, session, message=str(exc))
                    return
            if not catalog or not catalog.get('voices'):
                await handle_local_unavailable(handler, query, session, message="No voices available")
                return
            if favorites is None and session.get('favorites') is None:
                session['favorites'] = []
            session['voice_mode'] = session.get('voice_mode') or ('favorites' if session.get('favorites') else 'all')
            handler._store_tts_session(query.message.chat_id, query.message.message_id, session)
            await handler._refresh_tts_catalog(query, session)
            await query.answer("Select a voice")
            return
        else:
            await execute_job(handler, query, session, provider)
        return

    if callback_data.startswith("tts_queue:"):
        await enqueue_job(handler, query, session)
        return

    if not callback_data.startswith("tts_voice:"):
        return

    payload = callback_data.split(":", 1)[1]
    logging.info("🔊 Voice selected payload=%s", payload)
    kind, _, identifier = payload.partition("|")
    if not identifier:
        identifier = kind
        kind = 'cat'

    voice_lookup = session.get('voice_lookup') or {}
    entry = voice_lookup.get(payload) or {}

    favorite_slug = entry.get('favoriteSlug') if entry else None
    voice_id = entry.get('voiceId') if entry else None
    engine_id = entry.get('engine') if entry else None

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
    try:
        session['last_voice'] = handler._tts_voice_label(session, payload)
    except Exception:
        pass
    handler._store_tts_session(chat_id, message_id, session)

    logging.info(
        "🚀 Executing TTS job provider=%s fav=%s voice_id=%s engine=%s",
        provider_choice,
        favorite_slug,
        voice_id,
        engine_id,
    )

    try:
        await execute_job(handler, query, session, provider_choice)
    except LocalTTSUnavailable as exc:
        await handle_local_unavailable(handler, query, session, message=str(exc))
        return

async def finalize_delivery(handler, query, session: Dict[str, Any], audio_path: Path, provider: str) -> None:
    provider_label = "Local TTS hub" if provider == 'local' else "OpenAI TTS"
    metrics.record_tts(True)
    video_info = session.get('video_info') or {}
    mode = session.get('mode') or 'oneoff_tts'
    title = session.get('title') or video_info.get('title', 'Unknown Title')
    ledger_id = session.get('ledger_id')
    normalized_video_id = session.get('normalized_video_id') or video_info.get('video_id')
    summary_type = session.get('summary_type') or 'audio'
    base_variant = session.get('base_variant') or 'audio'

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
                voice_label = handler._tts_voice_label(session, slug)
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

        sync_result = sync_service.sync_audio_variant(
            normalized_id,
            summary_type,
            audio_path,
            ledger_id=ledger_id,
        )
        if not sync_result.get("success"):
            logging.warning(
                "⚠️ Audio dual-sync failed for %s:%s (%s)",
                normalized_id,
                summary_type,
                sync_result.get("error"),
            )
        if content_identifier:
            sync_service.upload_audio_to_render(content_identifier, audio_path)

        audio_reply_markup = handler._build_audio_inline_keyboard(
            normalized_id,
            base_variant,
            video_info.get('video_id', '')
        )

    try:
        with audio_path.open('rb') as audio_file:
            if mode == 'summary_audio':
                base = f"🎧 **Audio Summary**: {handler._escape_markdown(title)}"
                voice_bit = f" • {handler._escape_markdown(voice_label)}" if voice_label else ""
                caption = f"{base}{voice_bit}\n🎵 {provider_label}"
            else:
                caption = (
                    f"🎧 **TTS Preview**"
                    + (f" • {handler._escape_markdown(voice_label)}" if voice_label else "")
                    + f" • {provider_label}"
                )
            await query.message.reply_voice(
                voice=audio_file,
                caption=caption,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=audio_reply_markup
            )
        logging.info("✅ Successfully sent audio summary for %s using %s", title, provider_label)
    except Exception as exc:
        logging.error("Failed to send voice message: %s", exc)
    try:
        if session.get('catalog'):
            await handler._refresh_tts_catalog(query, session)
        else:
            await query.answer("Audio sent — select another voice")
    except Exception:
        pass


async def prepare_generation(handler, query, result: Dict[str, Any], summary_text: str, summary_type: str) -> None:
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

    await prompt_provider(handler, query, session_payload, title)
