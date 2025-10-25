from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from modules.ollama_client import get_models as ollama_get_models


def default_models(handler, models: List[str], allow_same: bool) -> Tuple[Optional[str], Optional[str]]:
    available = list(models or [])
    defaults_raw = os.getenv('OLLAMA_AI2AI_DEFAULT_MODELS', '')
    preferred = [item.strip() for item in defaults_raw.split(',') if item.strip()]
    model_a: Optional[str] = None
    model_b: Optional[str] = None

    for name in preferred:
        if name in available:
            if model_a is None:
                model_a = name
            elif model_b is None and (allow_same or name != model_a):
                model_b = name
        if model_a and model_b:
            break

    if model_a is None and available:
        model_a = available[0]

    if model_b is None:
        for candidate in available:
            if candidate != model_a:
                model_b = candidate
                break
        if model_b is None and allow_same and model_a is not None and len(available) == 1:
            model_b = model_a

    return model_a, model_b


def build_models_keyboard(
    models: List[str],
    slot: str,
    page: int = 0,
    page_size: int = 12,
    session: Optional[Dict[str, Any]] = None,
) -> InlineKeyboardMarkup:
    start = page * page_size
    end = start + page_size
    subset = models[start:end]
    rows: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    chosen_a = (session or {}).get('ai2ai_model_a') if (session and slot == 'B') else None
    for name in subset:
        if chosen_a and name == chosen_a:
            continue
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


async def run(handler, chat_id: int, turns: int) -> None:
    session = handler.ollama_sessions.get(chat_id) or {}
    if turns <= 0:
        return
    session["ai2ai_round"] = 0
    session["ai2ai_turns_total"] = turns
    session["ai2ai_turns_left"] = turns
    handler.ollama_sessions[chat_id] = session
    for remaining in range(turns, 0, -1):
        if session.get("ai2ai_cancel"):
            break
        session["ai2ai_turns_left"] = remaining
        handler.ollama_sessions[chat_id] = session
        current_turn = turns - remaining + 1
        await handler._ollama_ai2ai_continue(chat_id, turn_number=current_turn, total_turns=turns)
        if session.get("ai2ai_cancel"):
            break
        session["ai2ai_turns_left"] = remaining - 1
        handler.ollama_sessions[chat_id] = session
        if session["ai2ai_turns_left"] <= 0:
            break
    session["ai2ai_turns_left"] = 0
    handler.ollama_sessions[chat_id] = session
    if session.get("ai2ai_cancel"):
        session["ai2ai_active"] = False
        session["ai2ai_cancel"] = False
        handler.ollama_sessions[chat_id] = session
        try:
            await handler.application.bot.send_message(chat_id=chat_id, text="⏹️ AI↔AI exchange stopped.")
        except Exception:
            pass
        return
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⏭️ Continue", callback_data="ollama_ai2ai:auto"),
            InlineKeyboardButton("🧠 Options", callback_data="ollama_ai2ai:opts"),
        ],
        [InlineKeyboardButton("🔊 Audio", callback_data="ollama_ai2ai:tts")],
        [InlineKeyboardButton("♻️ Clear", callback_data="ollama_ai2ai:clear")],
    ])
    await handler.application.bot.send_message(
        chat_id=chat_id,
        text="✅ AI↔AI session complete. Choose Continue to keep the exchange going, or Options to adjust turns.",
        reply_markup=kb,
    )


async def continue_exchange(
    handler,
    chat_id: int,
    turn_number: Optional[int] = None,
    total_turns: Optional[int] = None,
) -> None:
    session = handler.ollama_sessions.get(chat_id) or {}
    if session.get("ai2ai_cancel"):
        return
    model_a = session.get("ai2ai_model_a") or session.get("model")
    model_b = session.get("ai2ai_model_b") or session.get("model")
    if not model_a or not model_b:
        try:
            await handler.application.bot.send_message(chat_id=chat_id, text="⚠️ Pick models for A and B in Options")
        except Exception:
            pass
        return
    session["active"] = True
    if not (session.get("persona_a") and session.get("persona_b")):
        rand_a, rand_b = handler._ollama_persona_random_pair()
        session.setdefault("persona_a", rand_a)
        session.setdefault("persona_b", rand_b)
    defaults = handler._ollama_persona_defaults()
    persona_a = session.get("persona_a") or defaults[0]
    persona_b = session.get("persona_b") or defaults[1]
    persona_a_custom = bool(session.get("persona_a_custom"))
    persona_b_custom = bool(session.get("persona_b_custom"))
    intro_a = bool(persona_a_custom and session.get("persona_a_intro_pending"))
    intro_b = bool(persona_b_custom and session.get("persona_b_intro_pending"))
    topic = session.get("topic", "The nature of space and time")
    turn_idx = turn_number or (session.get("ai2ai_round") or 0) + 1
    session["ai2ai_round"] = turn_idx
    total = total_turns or session.get("ai2ai_turns_total")
    if total and total < turn_idx:
        total = turn_idx
        session["ai2ai_turns_total"] = total
    persona_a = session.get("persona_a") or defaults[0]
    persona_b = session.get("persona_b") or defaults[1]
    turn_suffix = f" · Turn {turn_idx}"
    if total:
        turn_suffix += f"/{total}"
    a_messages = [
        {
            "role": "system",
            "content": handler._ollama_persona_system_prompt(persona_a, "opponent", intro_a),
        },
        {
            "role": "user",
            "content": (
                f"Debate topic: {topic}. Present your view from your own time and culture, "
                "using only knowledge that would have been available in your lifetime."
            ),
        },
    ]
    from types import SimpleNamespace

    class U:
        def __init__(self, app, chat_id):
            self.effective_chat = SimpleNamespace(id=chat_id)
            self._app = app

        @property
        def message(self):
            class M:
                def __init__(self, app, chat_id):
                    self._app = app
                    self._chat = chat_id

                async def reply_text(self, text):
                    return await self._app.bot.send_message(chat_id=self._chat, text=text)

            return M(self._app, chat_id)

    u = U(handler.application, chat_id)
    pa_disp, _ = handler._persona_parse(persona_a)
    # Allow per-slot provider: default to local (ollama)
    prov_a = (session.get('ai2ai_provider_a') or 'ollama')
    if prov_a == 'cloud':
        # Resolve cloud model for A
        opt_a = session.get('ai2ai_cloud_option_a') or {}
        cp = opt_a.get('provider') or getattr(handler, 'llm_config', None) or None
        cm = opt_a.get('model') or None
        if cp is None or cm is None:
            # Use handler.llm_config or global llm_config via handler import
            from llm_config import llm_config as _cfg
            cp = getattr(_cfg, 'llm_provider', None)
            cm = getattr(_cfg, 'llm_model', None)
            if not cm:
                sl = _cfg.SHORTLISTS.get(getattr(_cfg, 'llm_shortlist', 'openrouter_defaults'), {})
                for p, m in (sl.get('primary') or []):
                    if p != 'ollama':
                        cp, cm = p, m
                        break
        a_text = ""
        try:
            from modules.services.cloud_service import chat as cloud_chat
            a_text = cloud_chat(a_messages, provider=cp, model=cm)
        except Exception as exc:
            await u.message.reply_text(f"❌ Cloud (A) error: {str(exc)[:200]}")
            a_text = ""
        # Post combined label + text
        label = f"{pa_disp} ({cp}/{cm}){turn_suffix}"
        try:
            await u.message.reply_text(f"{label}\n\n{a_text}")
        except Exception:
            pass
    else:
        a_text = await handler._ollama_stream_chat(
            u,
            model_a,
            a_messages,
            label=f"{pa_disp} ({model_a}){turn_suffix}",
            cancel_checker=lambda: bool((handler.ollama_sessions.get(chat_id) or {}).get("ai2ai_cancel")),
        )
    session["ai2ai_last_a"] = a_text
    try:
        tr = session.get("ai2ai_transcript")
        if not isinstance(tr, list):
            tr = []
        pa_disp, _ = handler._persona_parse(persona_a)
        tr.append({"speaker": "A", "persona": pa_disp, "model": model_a, "text": a_text or ""})
        session["ai2ai_transcript"] = tr[-200:]
    except Exception:
        pass
    if intro_a:
        session["persona_a_intro_pending"] = False
    if session.get("ai2ai_cancel"):
        return
    b_messages = [
        {
            "role": "system",
            "content": handler._ollama_persona_system_prompt(persona_b, "opponent", intro_b),
        },
        {
            "role": "user",
            "content": (
                f"Respond to {persona_a}'s statement: {a_text[:800]}"
            ),
        },
    ]
    pb_disp, _ = handler._persona_parse(persona_b)
    prov_b = (session.get('ai2ai_provider_b') or 'ollama')
    if prov_b == 'cloud':
        opt_b = session.get('ai2ai_cloud_option_b') or {}
        cp = opt_b.get('provider') or getattr(handler, 'llm_config', None) or None
        cm = opt_b.get('model') or None
        if cp is None or cm is None:
            from llm_config import llm_config as _cfg
            cp = getattr(_cfg, 'llm_provider', None)
            cm = getattr(_cfg, 'llm_model', None)
            if not cm:
                sl = _cfg.SHORTLISTS.get(getattr(_cfg, 'llm_shortlist', 'openrouter_defaults'), {})
                for p, m in (sl.get('primary') or []):
                    if p != 'ollama':
                        cp, cm = p, m
                        break
        b_text = ""
        try:
            from modules.services.cloud_service import chat as cloud_chat
            b_text = cloud_chat(b_messages, provider=cp, model=cm)
        except Exception as exc:
            await u.message.reply_text(f"❌ Cloud (B) error: {str(exc)[:200]}")
            b_text = ""
        label = f"{pb_disp} ({cp}/{cm}){turn_suffix}"
        try:
            await u.message.reply_text(f"{label}\n\n{b_text}")
        except Exception:
            pass
    else:
        b_text = await handler._ollama_stream_chat(
            u,
            model_b,
            b_messages,
            label=f"{pb_disp} ({model_b}){turn_suffix}",
            cancel_checker=lambda: bool((handler.ollama_sessions.get(chat_id) or {}).get("ai2ai_cancel")),
        )
    session["ai2ai_last_b"] = b_text
    try:
        tr = session.get("ai2ai_transcript")
        if not isinstance(tr, list):
            tr = []
        pb_disp, _ = handler._persona_parse(persona_b)
        tr.append({"speaker": "B", "persona": pb_disp, "model": model_b, "text": b_text or ""})
        session["ai2ai_transcript"] = tr[-200:]
    except Exception:
        pass
    if intro_b:
        session["persona_b_intro_pending"] = False
    handler.ollama_sessions[chat_id] = session


async def handle_callback(
    handler,
    query,
    callback_data: str,
    session: Dict[str, Any],
    render_options: Optional[Callable[[], Awaitable[None]]] = None,
) -> bool:
    chat_id = query.message.chat_id

    if callback_data == "ollama_ai2ai:tts":
        try:
            await query.answer("Generating audio…")
        except Exception:
            pass
        status = None
        try:
            status = await query.message.reply_text("🎧 Generating AI↔AI audio…")
        except Exception:
            status = None
        try:
            path = await handler._ollama_ai2ai_generate_audio(chat_id, session)
            if not path or not Path(path).exists():
                raise RuntimeError("no audio produced")
            caption = handler._ai2ai_audio_caption(session)
            with open(path, "rb") as f:
                await query.message.reply_voice(voice=f, caption=caption, parse_mode=ParseMode.MARKDOWN)
            try:
                if status:
                    await status.edit_text("✅ AI↔AI audio ready")
            except Exception:
                pass
        except Exception as exc:
            try:
                message = f"❌ AI↔AI audio failed: {exc}"
                if status:
                    await status.edit_text(message)
                else:
                    await query.message.reply_text(message)
            except Exception:
                pass
        return True

    if callback_data.startswith("ollama_ai2ai:"):
        action = callback_data.split(":", 1)[1]
        if action == "enter":
            models = session.get("models") or []
            if not models:
                raw = ollama_get_models()
                models = handler._ollama_models_list(raw)
                session["models"] = models
            kb = build_models_keyboard(models, "A", session.get("page", 0), session=session)
            await query.edit_message_text("🤝 Select model for A:", reply_markup=kb)
            await query.answer("AI↔AI mode")
            handler.ollama_sessions[chat_id] = session
            return True
        if action == "clear":
            for key in (
                "ai2ai_model_a",
                "ai2ai_model_b",
                "ai2ai_active",
                "persona_a",
                "persona_b",
                "persona_a_display",
                "persona_b_display",
                "persona_a_gender",
                "persona_b_gender",
                "persona_category_a",
                "persona_category_b",
                "persona_a_custom",
                "persona_b_custom",
                "persona_a_intro_pending",
                "persona_b_intro_pending",
                "ai2ai_round",
                "ai2ai_turns_total",
                "topic",
                "ai2ai_turns_left",
                "ai2ai_view_a",
                "ai2ai_view_b",
                "ai2ai_persona_category_a",
                "ai2ai_persona_category_b",
                "ai2ai_persona_page_a",
                "ai2ai_persona_page_b",
                "ai2ai_persona_cat_page_a",
                "ai2ai_persona_cat_page_b",
            ):
                session.pop(key, None)
            session.pop("ai2ai_page_a", None)
            session.pop("ai2ai_page_b", None)
            session["mode"] = "ai-human"
            session["active"] = bool(session.get("model"))
            models = session.get("models") or []
            kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
            await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
            await query.answer("Cleared AI↔AI")
            handler.ollama_sessions[chat_id] = session
            return True
        if action == "auto":
            turns = session.get("ai2ai_turns_config") or session.get("ai2ai_turns_total") or os.getenv('OLLAMA_AI2AI_TURNS', '10')
            try:
                turns_int = int(turns)
            except Exception:
                turns_int = 10
            if turns_int <= 0:
                turns_int = 10
            session["ai2ai_cancel"] = False
            session["ai2ai_active"] = True
            handler.ollama_sessions[chat_id] = session
            await query.answer("Continuing AI↔AI")
            try:
                await handler._ollama_ai2ai_run(chat_id, turns_int)
            except Exception as exc:
                await query.message.reply_text(f"❌ Failed to continue AI↔AI: {exc}")
            return True
        if action == "start":
            session["ai2ai_active"] = True
            if not (session.get("persona_a") and session.get("persona_b")):
                rand_a, rand_b = handler._ollama_persona_random_pair()
                session.setdefault("persona_a", rand_a)
                session.setdefault("persona_b", rand_b)
            handler._update_persona_session_fields(session, 'a', session.get('persona_a'))
            handler._update_persona_session_fields(session, 'b', session.get('persona_b'))
            session.setdefault("persona_a_custom", False)
            session.setdefault("persona_b_custom", False)
            session.setdefault("persona_a_intro_pending", False)
            session.setdefault("persona_b_intro_pending", False)
            session.setdefault("topic", session.get("last_user") or "The nature of space and time")
            try:
                default_turns = int(os.getenv('OLLAMA_AI2AI_TURNS', '10'))
            except Exception:
                default_turns = 10
            session.setdefault("ai2ai_turns_left", default_turns)
            if not session.get("ai2ai_model_a"):
                session["ai2ai_model_a"] = session.get("model")
            if not session.get("ai2ai_model_b"):
                session["ai2ai_model_b"] = session.get("model")
            session["active"] = True
            handler.ollama_sessions[chat_id] = session
            await query.answer("AI↔AI started")
            try:
                await query.edit_message_text("🤖 AI↔AI mode active. Use Options → Continue exchange to generate turns.")
            except Exception:
                pass
            if render_options is not None:
                await render_options()
            return True
        if action in ("continue", "auto"):
            turns_cfg = session.get("ai2ai_turns_config") or session.get("ai2ai_turns_left")
            if not isinstance(turns_cfg, int) or turns_cfg <= 0:
                try:
                    turns_cfg = int(os.getenv('OLLAMA_AI2AI_TURNS', '10'))
                except Exception:
                    turns_cfg = 10
                session["ai2ai_turns_config"] = turns_cfg
            await query.answer("Continuing…")
            await handler._ollama_ai2ai_run(query.message.chat_id, turns_cfg)
            return True
        if action == "opts":
            turns = int(session.get('ai2ai_turns_left') or 10)
            kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("➖ Turns", callback_data="ollama_ai2ai_turns:-"),
                    InlineKeyboardButton(f"{turns} turns", callback_data="ollama_nop"),
                    InlineKeyboardButton("➕ Turns", callback_data="ollama_ai2ai_turns:+"),
                ],
                [InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more:{session.get('page', 0)}")],
            ])
            await query.edit_message_text("🧠 AI↔AI Options", reply_markup=kb)
            await query.answer("Options")
            return True

    if callback_data.startswith("ollama_ai2ai_view:"):
        parts = callback_data.split(":")
        if len(parts) >= 3:
            slot = parts[1].upper()
            target = parts[2]
            slot_lower = slot.lower()
            view_key = f"ai2ai_view_{slot_lower}"
            if target == "models":
                session[view_key] = "models"
            else:
                session[view_key] = "persona_categories"
                session.pop(f"ai2ai_persona_category_{slot_lower}", None)
                session[f"ai2ai_persona_cat_page_{slot_lower}"] = 0
                session[f"ai2ai_persona_page_{slot_lower}"] = 0
            models = session.get("models") or []
            kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
            await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
            handler.ollama_sessions[chat_id] = session
            await query.answer("View updated")
            return True

    if callback_data.startswith("ollama_persona_cat:"):
        parts = callback_data.split(":", 2)
        if len(parts) == 3:
            slot = parts[1].upper()
            cat_key = parts[2]
            slot_lower = slot.lower()
            categories = handler._ollama_persona_categories()
            if cat_key not in categories:
                await query.answer("Category unavailable", show_alert=False)
                return True
            session[f"ai2ai_persona_category_{slot_lower}"] = cat_key
            session[f"ai2ai_view_{slot_lower}"] = "persona_list"
            session[f"ai2ai_persona_page_{slot_lower}"] = 0
            models = session.get("models") or []
            kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
            await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
            handler.ollama_sessions[chat_id] = session
            await query.answer("Category selected")
            return True

    if callback_data.startswith("ollama_persona_more:"):
        parts = callback_data.split(":")
        if len(parts) >= 4:
            slot = parts[1].upper()
            kind = parts[2]
            try:
                page = max(0, int(parts[3]))
            except Exception:
                page = 0
            slot_lower = slot.lower()
            if kind == "cat":
                session[f"ai2ai_persona_cat_page_{slot_lower}"] = page
                session[f"ai2ai_view_{slot_lower}"] = "persona_categories"
            else:
                session[f"ai2ai_persona_page_{slot_lower}"] = page
                session.setdefault(f"ai2ai_view_{slot_lower}", "persona_list")
                session[f"ai2ai_view_{slot_lower}"] = "persona_list"
            models = session.get("models") or []
            kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
            await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
            handler.ollama_sessions[chat_id] = session
            await query.answer("Page updated")
            return True

    if callback_data.startswith("ollama_persona_back:"):
        parts = callback_data.split(":")
        if len(parts) >= 2:
            slot = parts[1].upper()
            slot_lower = slot.lower()
            session[f"ai2ai_view_{slot_lower}"] = "persona_categories"
            session[f"ai2ai_persona_page_{slot_lower}"] = 0
            models = session.get("models") or []
            kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
            await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
            handler.ollama_sessions[chat_id] = session
            await query.answer("Select category")
            return True

    if callback_data.startswith("ollama_persona_pick:"):
        parts = callback_data.split(":")
        if len(parts) >= 3:
            slot = parts[1].upper()
            slot_lower = slot.lower()
            try:
                index = int(parts[2])
            except Exception:
                index = -1
            categories = handler._ollama_persona_categories()
            cat_key = session.get(f"ai2ai_persona_category_{slot_lower}")
            names = []
            if cat_key:
                names = categories.get(cat_key, {}).get("names") or []
            response = f"Persona {slot} updated"
            if 0 <= index < len(names):
                chosen = names[index]
                current = session.get(f"persona_{slot_lower}")
                if current == chosen and session.get(f"persona_{slot_lower}_custom"):
                    session.pop(f"persona_{slot_lower}", None)
                    session.pop(f"persona_category_{slot_lower}", None)
                    session.pop(f"persona_{slot_lower}_custom", None)
                    session.pop(f"persona_{slot_lower}_intro_pending", None)
                    session.pop(f"persona_{slot_lower}_display", None)
                    session.pop(f"persona_{slot_lower}_gender", None)
                    response = f"Persona {slot} cleared"
                else:
                    session[f"persona_{slot_lower}"] = chosen
                    session[f"persona_category_{slot_lower}"] = categories.get(cat_key, {}).get("label")
                    if slot in ("A", "B"):
                        session[f"persona_{slot_lower}_custom"] = True
                        session[f"persona_{slot_lower}_intro_pending"] = True
                    handler._update_persona_session_fields(session, slot_lower, chosen)
                session[f"ai2ai_view_{slot_lower}"] = "persona_list"
            models = session.get("models") or []
            kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
            await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
            handler.ollama_sessions[chat_id] = session
            await query.answer(response)
            return True

    if callback_data.startswith("ollama_persona_clear:"):
        parts = callback_data.split(":")
        if len(parts) >= 2:
            slot = parts[1].upper()
            slot_lower = slot.lower()
            session.pop(f"persona_{slot_lower}", None)
            session.pop(f"persona_category_{slot_lower}", None)
            if slot in ("A", "B"):
                session.pop(f"persona_{slot_lower}_custom", None)
                session.pop(f"persona_{slot_lower}_intro_pending", None)
            session.pop(f"persona_{slot_lower}_display", None)
            session.pop(f"persona_{slot_lower}_gender", None)
            models = session.get("models") or []
            kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
            await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
            handler.ollama_sessions[chat_id] = session
            await query.answer("Persona cleared")
            return True

    if callback_data.startswith("ollama_set_a:"):
        name = callback_data.split(":", 1)[1]
        session["mode"] = "ai-ai"
        if session.get("ai2ai_model_a") == name:
            session.pop("ai2ai_model_a", None)
            session["active"] = bool(session.get("ai2ai_model_b"))
            logging.info("Ollama UI: cleared model A")
        else:
            session["ai2ai_model_a"] = name
            session["active"] = bool(session.get("ai2ai_model_a") and session.get("ai2ai_model_b"))
            logging.info("Ollama UI: set model A -> %s", name)
        models = session.get("models") or []
        kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
        await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
        handler.ollama_sessions[chat_id] = session
        await query.answer("Model A updated")
        return True

    if callback_data.startswith("ollama_set_b:"):
        name = callback_data.split(":", 1)[1]
        session["mode"] = "ai-ai"
        if session.get("ai2ai_model_b") == name:
            session.pop("ai2ai_model_b", None)
            session["active"] = bool(session.get("ai2ai_model_a"))
            logging.info("Ollama UI: cleared model B")
        else:
            session["ai2ai_model_b"] = name
            if "ai2ai_turns_left" not in session:
                try:
                    session["ai2ai_turns_left"] = int(os.getenv('OLLAMA_AI2AI_TURNS', '10'))
                except Exception:
                    session["ai2ai_turns_left"] = 10
            session["active"] = True
            logging.info("Ollama UI: set model B -> %s", name)
        models = session.get("models") or []
        kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
        await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
        handler.ollama_sessions[chat_id] = session
        await query.answer("Model B updated")
        return True

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
        logging.info("Ollama UI: set mode -> %s", session["mode"])
        models = session.get("models") or []
        kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
        await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
        handler.ollama_sessions[chat_id] = session
        await query.answer("Mode updated")
        return True

    if callback_data.startswith("ollama_ai2ai_turns:"):
        op = callback_data.split(":", 1)[1]
        turns = int(session.get('ai2ai_turns_left') or 10)
        if op == '+':
            turns = min(50, turns + 1)
        else:
            turns = max(1, turns - 1)
        session['ai2ai_turns_left'] = turns
        session['ai2ai_turns_config'] = turns
        handler.ollama_sessions[chat_id] = session
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("➖ Turns", callback_data="ollama_ai2ai_turns:-"),
                InlineKeyboardButton(f"{turns} turns", callback_data="ollama_nop"),
                InlineKeyboardButton("➕ Turns", callback_data="ollama_ai2ai_turns:+"),
            ],
            [InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more:{session.get('page', 0)}")],
        ])
        await query.edit_message_text("🧠 AI↔AI Options", reply_markup=kb)
        await query.answer("Updated turns")
        return True

    if callback_data.startswith("ollama_more_ai2ai:"):
        _, slot, page_str = callback_data.split(":", 2)
        try:
            page = int(page_str)
        except Exception:
            page = 0
        models = session.get("models") or []
        key = f"ai2ai_page_{slot.lower()}"
        session[key] = page
        kb = build_models_keyboard(models, slot, page, session=session)
        await query.edit_message_text(f"🤖 Pick model for {slot}:", reply_markup=kb)
        handler.ollama_sessions[chat_id] = session
        await query.answer("Page updated")
        return True

    return False


__all__ = [
    'default_models',
    'build_models_keyboard',
    'run',
    'continue_exchange',
    'handle_callback',
]
