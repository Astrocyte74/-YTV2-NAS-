from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from modules.ollama_client import chat_stream as ollama_chat_stream


async def stream_chat(
    update,
    bot,
    model: str,
    messages: List[Dict[str, str]],
    *,
    label: Optional[str] = None,
    cancel_checker: Optional[Callable[[], bool]] = None,
) -> str:
    """Stream tokens from the hub and live-edit a single message. Returns final text."""
    if update is None or update.effective_chat is None:
        raise ValueError("update with effective_chat required")

    chat_id = update.effective_chat.id
    prefix = f"{label}\n\n" if label else ""
    msg = await update.message.reply_text(f"{prefix}⏳ …")
    message_id = msg.message_id
    loop = asyncio.get_running_loop()

    final_text: Dict[str, Any] = {"buf": "", "cancelled": False}

    def compose_text() -> str:
        buf = final_text["buf"]
        if label:
            allowance = max(0, 4000 - len(prefix))
            tail = buf[-allowance:] if len(buf) > allowance else buf
            return f"{prefix}{tail}" if tail else prefix.rstrip()
        return buf[-4000:] if len(buf) > 4000 else (buf or "⏳")

    def run_stream() -> None:
        logging.info("Ollama streaming start: model=%s msgs=%s", model, len(messages))
        last = 0.0
        try:
            stream = ollama_chat_stream(messages, model)
            for data in stream:
                if cancel_checker and cancel_checker():
                    final_text["cancelled"] = True
                    text = compose_text()
                    if text.strip():
                        if label:
                            text = text.rstrip() + "\n\n⏹️ Stopped."
                        else:
                            text = f"{text.rstrip()}\n\n⏹️ Stopped."
                    else:
                        text = f"{prefix}⏹️ Stopped." if label else "⏹️ Stopped."

                    async def _cancel_edit() -> None:
                        try:
                            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text)
                        except Exception:
                            pass

                    asyncio.run_coroutine_threadsafe(_cancel_edit(), loop)
                    close = getattr(stream, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
                    break

                if not isinstance(data, dict):
                    continue
                if data.get("status") == "starting":
                    continue
                chunk = data.get("response")
                if isinstance(chunk, str) and chunk:
                    final_text["buf"] += chunk
                else:
                    msg = data.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content:
                            final_text["buf"] += content
                        elif isinstance(content, list):
                            for seg in content:
                                if isinstance(seg, dict):
                                    text_part = seg.get("text") or seg.get("content")
                                    if isinstance(text_part, str) and text_part:
                                        final_text["buf"] += text_part
                now = time.time()
                if (now - last > 0.4) and final_text["buf"]:
                    last = now
                    text = compose_text()

                    async def _update() -> None:
                        try:
                            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text)
                        except Exception:
                            pass

                    asyncio.run_coroutine_threadsafe(_update(), loop)
                if data.get("done"):
                    text = compose_text()

                    async def _final() -> None:
                        try:
                            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text)
                        except Exception:
                            pass

                    asyncio.run_coroutine_threadsafe(_final(), loop)
                    break
        except Exception as exc:
            async def _error() -> None:
                try:
                    msg_text = f"{prefix}❌ Stream error: {exc}" if label else f"❌ Stream error: {exc}"
                    await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=msg_text)
                except Exception:
                    pass

            asyncio.run_coroutine_threadsafe(_error(), loop)

    await loop.run_in_executor(None, run_stream)
    return final_text["buf"]
