from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from modules.metrics import metrics
from modules.services import sync_service
from modules.tts_hub import LocalTTSUnavailable, TTSHubClient, DEFAULT_ENGINE
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

    # Build a keyboard that includes quick favorite voices (if any) on the first screen
    rows = []
    quick_favs = []
    try:
        env_val = (os.getenv("TTS_QUICK_FAVORITE") or "").strip()
        env_list = [s.strip() for s in env_val.split(",") if s.strip()]
        for item in env_list[:3]:
            label = item
            if "|" in item:
                eng, slug = item.split("|", 1)
                short = slug if len(slug) <= 24 else (slug[:23] + "…")
                label = f"Quick • {eng}:{short}"
                alias = f"fav|{eng}|{slug}"
            else:
                short = item if len(item) <= 24 else (item[:23] + "…")
                label = f"Quick • {short}"
                alias = f"fav|{item}"
            quick_favs.append((label, alias))
    except Exception:
        quick_favs = []

    if quick_favs:
        for label, alias in quick_favs:
            rows.append([InlineKeyboardButton(label, callback_data=f"tts_voice:{alias}")])

    # Provider choices
    provider_row = []
    provider_row.append(InlineKeyboardButton("Local TTS hub", callback_data="tts_provider:local"))
    provider_row.append(InlineKeyboardButton("OpenAI TTS", callback_data="tts_provider:openai"))
    rows.append(provider_row)
    rows.append([InlineKeyboardButton("❌ Cancel", callback_data="tts_cancel")])

    prompt_text = f"🎙️ Choose how to generate audio for **{handler._escape_markdown(title)}**"
    prompt_message = await query.message.reply_text(
        prompt_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(rows),
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
        # Remember last-used favorite for quick pick
        try:
            alias_slug = None
            if favorite_slug and engine_id:
                alias_slug = f"fav|{engine_id}|{favorite_slug}"
            elif voice_id and engine_id:
                alias_slug = f"cat|{engine_id}|{voice_id}"
            if alias_slug:
                handler._remember_last_tts_voice(query.from_user.id, alias_slug)
        except Exception:
            pass
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

    if callback_data.startswith("tts_engine:"):
        engine = callback_data.split(":", 1)[1]
        if engine == '__all__':
            session['active_engine'] = engine
            handler._store_tts_session(chat_id, message_id, session)
            await handler._refresh_tts_catalog(query, session)
            await query.answer("All engines")
            return
        catalogs = session.get('catalogs') or {}
        if engine not in catalogs or not catalogs.get(engine):
            client = handler._resolve_tts_client(session.get('tts_base'))
            if client:
                try:
                    fetched = await client.fetch_catalog(engine=engine)
                    catalogs[engine] = fetched or {}
                    session['catalogs'] = catalogs
                except Exception as exc:
                    logging.warning(f"⚠️ Failed to load catalog for engine {engine}: {exc}")
                    await query.answer("Engine unavailable", show_alert=True)
                    return
            else:
                await query.answer("Engine unavailable", show_alert=True)
                return
        session['active_engine'] = engine
        session['catalog'] = catalogs.get(engine) or {}
        handler._store_tts_session(chat_id, message_id, session)
        await handler._refresh_tts_catalog(query, session)
        await query.answer(f"{engine.upper()} voices")
        return

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

    if callback_data == "tts_refresh":
        client = handler._resolve_tts_client(session.get('tts_base'))
        if not client:
            await query.answer("TTS hub unavailable", show_alert=True)
            return
        try:
            favorites = await client.fetch_favorites()
        except Exception:
            favorites = []
        session['favorites'] = favorites or []
        default_engine = session.get('default_engine') or DEFAULT_ENGINE
        active_engine = session.get('active_engine') or default_engine
        target_engine = active_engine if active_engine != '__all__' else default_engine
        catalogs = session.get('catalogs') or {}
        try:
            fetched = await client.fetch_catalog(engine=target_engine)
            catalogs[target_engine] = fetched or {}
        except Exception:
            pass
        # Warm catalogs for favorite engines
        try:
            fav_engs: Set[str] = set()
            for fav in favorites or []:
                eng = (fav.get('engine') or '').strip() or default_engine
                fav_engs.add(eng)
            for eng in sorted(fav_engs):
                if eng in catalogs and catalogs.get(eng):
                    continue
                try:
                    extra = await client.fetch_catalog(engine=eng)
                    catalogs[eng] = extra or {}
                except Exception:
                    pass
        except Exception:
            pass
        session['catalogs'] = catalogs
        session['catalog'] = catalogs.get(target_engine) or session.get('catalog') or {}
        handler._store_tts_session(chat_id, message_id, session)
        await handler._refresh_tts_catalog(query, session)
        await query.answer("Refreshed")
        return

    if callback_data.startswith("tts_provider:"):
        provider = callback_data.split(":", 1)[1]
        session['provider'] = provider
        preselect_only = bool(session.get('preselect_only'))
        if provider == 'local':
            client = handler._resolve_tts_client(session.get('tts_base'))
            session['tts_base'] = client.base_api_url if client else None
            default_engine = session.get('default_engine') or DEFAULT_ENGINE
            session.setdefault('default_engine', default_engine)
            active_engine = session.get('active_engine') or default_engine
            catalog = session.get('catalog')
            catalogs = session.get('catalogs') or {}
            favorites = session.get('favorites')
            if client:
                try:
                    target_engine = active_engine if active_engine != '__all__' else default_engine
                    if target_engine not in catalogs or not catalogs.get(target_engine):
                        fetched_catalog = await client.fetch_catalog(engine=target_engine)
                        catalogs[target_engine] = fetched_catalog or {}
                    catalog = catalogs.get(target_engine) or catalog
                    if not favorites:
                        try:
                            favorites = await client.fetch_favorites()
                        except Exception:
                            favorites = []
                        session['favorites'] = favorites or []

                    def resolve_engine(value: Optional[str]) -> str:
                        if isinstance(value, str) and value.strip():
                            return value.strip()
                        return default_engine

                    favorite_engines: Set[str] = set()
                    if favorites:
                        for fav in favorites:
                            if not isinstance(fav, dict):
                                continue
                            if not fav.get('voiceId'):
                                continue
                            favorite_engines.add(resolve_engine(fav.get('engine')))
                    for engine in sorted(favorite_engines):
                        if engine in catalogs and catalogs.get(engine):
                            continue
                        try:
                            extra_catalog = await client.fetch_catalog(engine=engine)
                            catalogs[engine] = extra_catalog or {}
                        except Exception as exc:
                            logging.warning(f"⚠️ Failed to load catalog for engine {engine}: {exc}")
                    session['catalogs'] = catalogs
                    session['catalog'] = catalog
                except Exception as exc:
                    await handle_local_unavailable(handler, query, session, message=str(exc))
                    return
            has_any_catalog = any(
                isinstance(cat, dict) and (cat.get('voices') or [])
                for cat in catalogs.values()
            )
            if not has_any_catalog:
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
            if preselect_only:
                # Preselect OpenAI voice (default from env) and start pending summary
                voice = (os.getenv('TTS_CLOUD_VOICE') or 'fable').strip()
                session['selected_voice'] = {'favorite_slug': None, 'voice_id': voice, 'engine': None}
                session['auto_run'] = True
                handler._store_tts_session(query.message.chat_id, query.message.message_id, session)
                pending = session.get('pending_summary') or {}
                origin = pending.get('origin')  # (chat_id, message_id)
                if origin and isinstance(origin, tuple) and len(origin) == 2:
                    try:
                        handler._store_tts_session(origin[0], origin[1], {
                            'auto_run': True,
                            'provider': 'openai',
                            'selected_voice': session['selected_voice'],
                            'summary_type': session.get('summary_type') or 'audio',
                        })
                    except Exception:
                        pass
                pend_sess = pending.get('session') or {}
                pend_provider = pending.get('provider_key') or 'cloud'
                pend_model = pending.get('model_option') or {}
                try:
                    await handler._execute_summary_with_model(query, pend_sess, pend_provider, pend_model)
                finally:
                    handler._remove_tts_session(query.message.chat_id, query.message.message_id)
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

    alias_map = session.get('voice_alias_map') or {}
    raw_payload = payload
    if payload.startswith("alias|"):
        alias_token = payload.split("|", 1)[1]
        raw_payload = alias_map.get(alias_token, payload)

    voice_lookup = session.get('voice_lookup') or {}
    entry = voice_lookup.get(payload) or voice_lookup.get(raw_payload) or {}

    parts = raw_payload.split("|")
    kind = parts[0] if parts else ''
    engine_hint = parts[1] if len(parts) > 2 else (parts[1] if len(parts) > 1 and kind != 'cat' else '')
    identifier_hint = parts[-1] if len(parts) > 1 else (parts[0] if parts else '')

    favorite_slug = entry.get('favoriteSlug') if entry else None
    voice_id = entry.get('voiceId') if entry else None
    engine_id = entry.get('engine') if entry else None

    if kind == 'fav' and not favorite_slug:
        favorite_slug = identifier_hint
    if kind != 'fav' and not voice_id:
        voice_id = identifier_hint
    if not engine_id and engine_hint:
        engine_id = engine_hint

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

    if session.get('preselect_only'):
        # Store selection and kick off pending summary; do not synthesize yet
        session['auto_run'] = True
        handler._store_tts_session(chat_id, message_id, session)
        pending = session.get('pending_summary') or {}
        origin = pending.get('origin')
        if origin and isinstance(origin, tuple) and len(origin) == 2:
            try:
                handler._store_tts_session(origin[0], origin[1], {
                    'auto_run': True,
                    'provider': 'local',
                    'selected_voice': session.get('selected_voice') or {},
                    'summary_type': session.get('summary_type') or 'audio',
                })
            except Exception:
                pass
        pend_sess = pending.get('session') or {}
        pend_provider = pending.get('provider_key') or 'ollama'
        pend_model = pending.get('model_option') or {}
        try:
            await handler._execute_summary_with_model(query, pend_sess, pend_provider, pend_model)
        finally:
            handler._remove_tts_session(chat_id, message_id)
        return
    else:
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
            engine_hint = sv.get('engine')
            favorite_slug = sv.get('favorite_slug')
            voice_id = sv.get('voice_id')
            if favorite_slug:
                if engine_hint:
                    slug = f"fav|{engine_hint}|{favorite_slug}"
                else:
                    slug = f"fav|{favorite_slug}"
            elif voice_id:
                if engine_hint:
                    slug = f"cat|{engine_hint}|{voice_id}"
                else:
                    slug = f"cat|{voice_id}"
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
            logging.debug("Audio ready for %s (dashboard upload via Postgres)", content_identifier)

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
