from __future__ import annotations

from typing import Any, Dict


async def handle_callback(handler, query, callback_data: str, session: Dict[str, Any]) -> bool:
    chat_id = query.message.chat_id

    if callback_data.startswith("ollama_single_view:"):
        target = callback_data.split(":", 1)[1]
        if target == "models":
            session["single_view"] = "models"
        else:
            session["single_view"] = "persona_categories"
            session["single_persona_cat_page"] = 0
            session["single_persona_page"] = 0
        models = session.get("models") or []
        kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
        await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
        handler.ollama_sessions[chat_id] = session
        await query.answer("View updated")
        return True

    if callback_data.startswith("ollama_single_persona_cat:"):
        cat_key = callback_data.split(":", 1)[1]
        categories = handler._ollama_persona_categories()
        if cat_key not in categories:
            await query.answer("Category unavailable", show_alert=False)
            return True
        session["single_persona_category"] = cat_key
        session["single_persona_page"] = 0
        session["single_view"] = "persona_list"
        session["persona_single_category"] = categories.get(cat_key, {}).get("label")
        models = session.get("models") or []
        kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
        await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
        handler.ollama_sessions[chat_id] = session
        await query.answer("Category selected")
        return True

    if callback_data.startswith("ollama_single_persona_more:"):
        parts = callback_data.split(":")
        if len(parts) >= 3:
            kind = parts[1]
            try:
                page = max(0, int(parts[2]))
            except Exception:
                page = 0
            if kind == "cat":
                session["single_persona_cat_page"] = page
                session["single_view"] = "persona_categories"
            else:
                session["single_persona_page"] = page
                session["single_view"] = "persona_list"
            models = session.get("models") or []
            kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
            await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
            handler.ollama_sessions[chat_id] = session
            await query.answer("Page updated")
        return True

    if callback_data == "ollama_single_persona_back":
        session["single_view"] = "persona_categories"
        session["single_persona_page"] = 0
        models = session.get("models") or []
        kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
        await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
        handler.ollama_sessions[chat_id] = session
        await query.answer("Select category")
        return True

    if callback_data.startswith("ollama_single_persona_pick:"):
        try:
            index = int(callback_data.split(":", 1)[1])
        except Exception:
            index = -1
        categories = handler._ollama_persona_categories()
        cat_key = session.get("single_persona_category")
        names = []
        if cat_key:
            names = categories.get(cat_key, {}).get("names") or []
        response = "Persona selected"
        if 0 <= index < len(names):
            chosen = names[index]
            current = session.get("persona_single")
            if current == chosen and session.get("persona_single_custom"):
                session.pop("persona_single", None)
                session.pop("persona_single_custom", None)
                session.pop("persona_single_intro_pending", None)
                session.pop("persona_single_category", None)
                response = "Persona cleared"
            else:
                session["persona_single"] = chosen
                session["persona_single_custom"] = True
                session["persona_single_intro_pending"] = True
                session["persona_single_category"] = categories.get(cat_key, {}).get("label")
            session["single_view"] = "persona_list"
        models = session.get("models") or []
        kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
        await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
        handler.ollama_sessions[chat_id] = session
        await query.answer(response)
        return True

    if callback_data == "ollama_single_persona_clear":
        session.pop("persona_single", None)
        session.pop("persona_single_category", None)
        session.pop("persona_single_custom", None)
        session.pop("persona_single_intro_pending", None)
        models = session.get("models") or []
        kb = handler._build_ollama_models_keyboard(models, session.get("page", 0), session=session)
        await query.edit_message_text(handler._ollama_status_text(session), reply_markup=kb)
        handler.ollama_sessions[chat_id] = session
        await query.answer("Persona cleared")
        return True

    return False
