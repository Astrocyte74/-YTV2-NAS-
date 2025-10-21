from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


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
            InlineKeyboardButton("⏭️ Continue AI↔AI", callback_data="ollama_ai2ai:auto"),
            InlineKeyboardButton("🧠 Options", callback_data="ollama_ai2ai:opts"),
        ],
        [InlineKeyboardButton("🔊 AI↔AI Audio", callback_data="ollama_ai2ai:tts")],
        [InlineKeyboardButton("♻️ Clear AI↔AI", callback_data="ollama_ai2ai:clear")],
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


__all__ = [
    'default_models',
    'run',
    'continue_exchange',
]
