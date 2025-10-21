from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode

from modules.ollama_client import chat as ollama_chat


def start_ai2ai_task(handler, chat_id: int, coro: Awaitable[Any]) -> bool:
    session = handler.ollama_sessions.get(chat_id) or {}
    existing = session.get("ai2ai_task")
    if isinstance(existing, asyncio.Task) and not existing.done():
        return False

    loop = asyncio.get_running_loop()

    async def runner():
        try:
            await coro
        except asyncio.CancelledError:
            logging.info("AI↔AI task cancelled for chat %s", chat_id)
            raise
        except Exception:
            logging.exception("AI↔AI task failed for chat %s", chat_id)
        finally:
            sess = handler.ollama_sessions.get(chat_id) or {}
            current = asyncio.current_task()
            if sess.get("ai2ai_task") is current:
                sess["ai2ai_task"] = None
            sess["ai2ai_active"] = False
            if not sess.get("ai2ai_cancel"):
                sess.pop("ai2ai_cancel", None)
            handler.ollama_sessions[chat_id] = sess

    task = loop.create_task(runner())
    session["ai2ai_task"] = task
    session["ai2ai_cancel"] = False
    session["ai2ai_active"] = True
    handler.ollama_sessions[chat_id] = session
    return True


async def handle_user_text(handler, update: Update, session: Dict[str, Any], text: str) -> None:
    chat_id = update.effective_chat.id
    mode_key = session.get("mode") or "ai-human"
    model = session.get("model")
    if mode_key == "ai-ai":
        if not (session.get("ai2ai_model_a") and session.get("ai2ai_model_b")):
            await update.message.reply_text("⚠️ Select models A and B in the picker before starting AI↔AI chat.")
            return
    else:
        if not model:
            await update.message.reply_text("⚠️ No model selected. Pick one in the Ollama picker first.")
            return
    if mode_key == "ai-ai" and session.get("ai2ai_model_a") and session.get("ai2ai_model_b"):
        session["topic"] = text
        if not isinstance(session.get("ai2ai_turns_config"), int) or session.get("ai2ai_turns_config") <= 0:
            try:
                session["ai2ai_turns_config"] = int(os.getenv('OLLAMA_AI2AI_TURNS', '10'))
            except Exception:
                session["ai2ai_turns_config"] = 10
        session["ai2ai_round"] = 0
        turns_total = int(session.get("ai2ai_turns_config") or 0)
        session["ai2ai_turns_total"] = turns_total if turns_total > 0 else None
        handler.ollama_sessions[chat_id] = session
        turns = int(session["ai2ai_turns_config"])
        if not start_ai2ai_task(handler, chat_id, handler._ollama_ai2ai_run(chat_id, turns)):
            await update.message.reply_text("⚠️ AI↔AI exchange already running. Use /stop to interrupt.")
            return
        await update.message.reply_text("🤝 Starting AI↔AI exchange…")
        return

    history = list(session.get("history") or [])
    dispatch_messages: List[Dict[str, str]] = []
    persona_intro_consumed = False
    if mode_key == "ai-human":
        persona_single = session.get("persona_single")
        if persona_single and session.get("persona_single_custom"):
            intro_pending = bool(session.get("persona_single_intro_pending"))
            dispatch_messages.append({
                "role": "system",
                "content": handler._ollama_persona_system_prompt(
                    persona_single,
                    "user",
                    intro_pending,
                ),
            })
            if intro_pending:
                persona_intro_consumed = True
    dispatch_messages.extend(history)
    dispatch_messages.append({"role": "user", "content": text})
    trimmed_history = (history + [{"role": "user", "content": text}])
    if bool(session.get("stream")) and mode_key == "ai-human":
        try:
            final_text = await handler._ollama_stream_chat(update, model, dispatch_messages, label=f"🤖 {model}")
        except Exception as exc:
            await update.message.reply_text(f"❌ Stream error: {str(exc)[:200]}")
            return
        session["history"] = (trimmed_history + [{"role": "assistant", "content": final_text}])[-16:]
        if persona_intro_consumed:
            session["persona_single_intro_pending"] = False
        handler.ollama_sessions[chat_id] = session
        return

    loop = asyncio.get_running_loop()

    def _call():
        try:
            return ollama_chat(dispatch_messages, model, stream=False)
        except Exception as exc:
            return {"error": str(exc)}

    try:
        from telegram.constants import ChatAction
        app = getattr(handler, 'application', None)
        if app and getattr(app, 'bot', None):
            await app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except Exception:
        pass
    logging.info("Ollama chat: model=%s text_len=%s", model, len(text))
    resp = await loop.run_in_executor(None, _call)
    try:
        import json as _json
        if isinstance(resp, dict):
            keys = list(resp.keys())
            logging.info("Ollama resp keys: %s", keys[:8])
            msg = resp.get("message")
            if isinstance(msg, dict):
                logging.info("Ollama message keys: %s", list(msg.keys()))
        else:
            logging.info("Ollama resp type: %s", type(resp))
    except Exception:
        pass
    if isinstance(resp, dict) and resp.get("error"):
        err = str(resp["error"]).lower()
        if ("404" in err or "not found" in err or "no such model" in err) and model:
            session["last_user"] = text
            handler.ollama_sessions[chat_id] = session
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton(f"📥 Pull {model}", callback_data=f"ollama_pull:{model}")],
                [InlineKeyboardButton("❌ Cancel", callback_data="ollama_cancel")],
            ])
            await update.message.reply_text(
                f"⚠️ Model `{handler._escape_markdown(model)}` is not available on the hub. Pull it now?",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb,
            )
            return
        await update.message.reply_text(f"❌ Ollama error: {resp['error'][:200]}")
        return

    reply_text = None
    if isinstance(resp, dict):
        val = resp.get("response")
        if isinstance(val, str) and val.strip():
            reply_text = val
        if reply_text is None:
            msg = resp.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    reply_text = content
                elif isinstance(content, list):
                    parts: List[str] = []
                    for seg in content:
                        if isinstance(seg, dict):
                            text_val = seg.get("text") or seg.get("content")
                            if isinstance(text_val, str) and text_val.strip():
                                parts.append(text_val)
                    if parts:
                        reply_text = "\n".join(parts)
        if reply_text is None:
            msgs = resp.get("messages")
            if isinstance(msgs, list):
                for message in reversed(msgs):
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        content = message.get("content")
                        if isinstance(content, str) and content.strip():
                            reply_text = content
                            break
        if reply_text is None:
            import json as _json
            snippet = _json.dumps({k: resp[k] for k in list(resp.keys())[:8]})[:380]
            reply_text = f"(No response)\n<pre>{snippet}</pre>"
            try:
                await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
                return
            except Exception:
                reply_text = f"(No response)\n{snippet}"
    if not reply_text:
        reply_text = "(No response)"
    display_text = reply_text
    if mode_key == "ai-human":
        persona_single = session.get("persona_single")
        if persona_single:
            display_text = f"{persona_single} ({model})\n\n{reply_text}"
        else:
            display_text = f"🤖 {model}\n\n{reply_text}"
    await handler._send_long_text_reply(update, display_text)
    session["history"] = (trimmed_history + [{"role": "assistant", "content": reply_text}])[-16:]
    if persona_intro_consumed:
        session["persona_single_intro_pending"] = False
    handler.ollama_sessions[chat_id] = session
