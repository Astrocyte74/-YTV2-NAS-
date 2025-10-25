from typing import Any, Callable, Dict, List, Optional, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def single_view_toggle_row(view: str) -> List[InlineKeyboardButton]:
    mark_models = "✅" if view == "models" else "⬜"
    mark_personas = "✅" if view.startswith("persona") else "⬜"
    return [
        InlineKeyboardButton(f"{mark_models} Models", callback_data="ollama_single_view:models"),
        InlineKeyboardButton(f"{mark_personas} Personas", callback_data="ollama_single_view:personas"),
    ]


def single_persona_categories_rows(
    session: Dict[str, Any], page_size: int, categories: Dict[str, Dict[str, Any]]
) -> List[List[InlineKeyboardButton]]:
    rows: List[List[InlineKeyboardButton]] = []
    current = (session or {}).get("single_persona_category")
    page = int((session or {}).get("single_persona_cat_page") or 0)
    items = list(categories.items())
    start = page * page_size
    end = start + page_size
    row: List[InlineKeyboardButton] = []
    for cat_key, info in items[start:end]:
        label = info.get("label") or cat_key
        if len(label) > 28:
            label = f"{label[:25]}…"
        if cat_key == current:
            label = f"✅ {label}"
        row.append(InlineKeyboardButton(label, callback_data=f"ollama_single_persona_cat:{cat_key}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    nav: List[InlineKeyboardButton] = []
    if end < len(items):
        nav.append(InlineKeyboardButton("➡️ More", callback_data=f"ollama_single_persona_more:cat:{page+1}"))
    if page > 0:
        nav.insert(0, InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_single_persona_more:cat:{page-1}"))
    if nav:
        rows.append(nav)
    return rows


def single_persona_list_rows(
    session: Dict[str, Any],
    page_size: int,
    categories: Dict[str, Dict[str, Any]],
    persona_parse: Callable[[Optional[str]], Tuple[str, Optional[str]]],
) -> List[List[InlineKeyboardButton]]:
    rows: List[List[InlineKeyboardButton]] = []
    cat_key = (session or {}).get("single_persona_category")
    info = categories.get(cat_key or "", {})
    names: List[str] = info.get("names") or []
    page = int((session or {}).get("single_persona_page") or 0)
    start = page * page_size
    end = start + page_size
    subset = list(enumerate(names))[start:end]
    selected = (session or {}).get("persona_single")
    row: List[InlineKeyboardButton] = []
    for idx, name in subset:
        label, _gender = persona_parse(name)
        if len(label) > 28:
            label = f"{label[:25]}…"
        if selected == name:
            label = f"✅ {label}"
        row.append(InlineKeyboardButton(label, callback_data=f"ollama_single_persona_pick:{idx}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    nav: List[InlineKeyboardButton] = []
    if end < len(names):
        nav.append(InlineKeyboardButton("➡️ More", callback_data=f"ollama_single_persona_more:list:{page+1}"))
    if page > 0:
        nav.insert(0, InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_single_persona_more:list:{page-1}"))
    control: List[InlineKeyboardButton] = [InlineKeyboardButton("⬅️ Categories", callback_data="ollama_single_persona_back")]
    if selected:
        control.append(InlineKeyboardButton("♻️ Clear", callback_data="ollama_single_persona_clear"))
    if control:
        rows.append(control)
    if nav:
        rows.append(nav)
    if not names:
        rows.append([InlineKeyboardButton("⚠️ No personas in category", callback_data="ollama_nop")])
    return rows


def ai2ai_view_toggle_row(slot: str, view: str) -> List[InlineKeyboardButton]:
    slot = slot.upper()
    mark_models = "✅" if view == "models" else "⬜"
    mark_personas = "✅" if view.startswith("persona") else "⬜"
    return [
        InlineKeyboardButton(f"{mark_models} Models", callback_data=f"ollama_ai2ai_view:{slot}:models"),
        InlineKeyboardButton(f"{mark_personas} Personas", callback_data=f"ollama_ai2ai_view:{slot}:personas"),
    ]


def ai2ai_persona_categories_rows(
    slot: str,
    session: Dict[str, Any],
    page_size: int,
    categories: Dict[str, Dict[str, Any]],
) -> List[List[InlineKeyboardButton]]:
    rows: List[List[InlineKeyboardButton]] = []
    slot_lower = slot.lower()
    current_category = (session or {}).get(f"ai2ai_persona_category_{slot_lower}")
    page_key = f"ai2ai_persona_cat_page_{slot_lower}"
    page = int((session or {}).get(page_key) or 0)
    items = list(categories.items())
    start = page * page_size
    end = start + page_size
    row: List[InlineKeyboardButton] = []
    for cat_key, info in items[start:end]:
        label = info.get("label") or cat_key
        if len(label) > 28:
            label = f"{label[:25]}…"
        if cat_key == current_category:
            label = f"✅ {label}"
        row.append(InlineKeyboardButton(label, callback_data=f"ollama_persona_cat:{slot}:{cat_key}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    nav: List[InlineKeyboardButton] = []
    if end < len(items):
        nav.append(InlineKeyboardButton("➡️ More", callback_data=f"ollama_persona_more:{slot}:cat:{page+1}"))
    if page > 0:
        nav.insert(0, InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_persona_more:{slot}:cat:{page-1}"))
    if nav:
        rows.append(nav)
    return rows


def ai2ai_persona_list_rows(
    slot: str,
    session: Dict[str, Any],
    page_size: int,
    categories: Dict[str, Dict[str, Any]],
    persona_parse: Callable[[Optional[str]], Tuple[str, Optional[str]]],
) -> List[List[InlineKeyboardButton]]:
    rows: List[List[InlineKeyboardButton]] = []
    slot_lower = slot.lower()
    cat_key = (session or {}).get(f"ai2ai_persona_category_{slot_lower}")
    info = categories.get(cat_key or "", {})
    names: List[str] = info.get("names") or []
    page_key = f"ai2ai_persona_page_{slot_lower}"
    page = int((session or {}).get(page_key) or 0)
    start = page * page_size
    end = start + page_size
    subset = list(enumerate(names))[start:end]
    selected = (session or {}).get(f"persona_{slot_lower}")
    row: List[InlineKeyboardButton] = []
    for idx, name in subset:
        label, _gender = persona_parse(name)
        if len(label) > 28:
            label = f"{label[:25]}…"
        if selected == name:
            label = f"✅ {label}"
        row.append(InlineKeyboardButton(label, callback_data=f"ollama_persona_pick:{slot}:{idx}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    nav: List[InlineKeyboardButton] = []
    if end < len(names):
        nav.append(InlineKeyboardButton("➡️ More", callback_data=f"ollama_persona_more:{slot}:list:{page+1}"))
    if page > 0:
        nav.insert(0, InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_persona_more:{slot}:list:{page-1}"))
    control: List[InlineKeyboardButton] = [InlineKeyboardButton("⬅️ Categories", callback_data=f"ollama_persona_back:{slot}")]
    if selected:
        control.append(InlineKeyboardButton("♻️ Clear", callback_data=f"ollama_persona_clear:{slot}"))
    if control:
        rows.append(control)
    if nav:
        rows.append(nav)
    if not names:
        rows.append([InlineKeyboardButton("⚠️ No personas in category", callback_data="ollama_nop")])
    return rows


def build_ollama_models_keyboard(
    *,
    models: List[str],
    page: int = 0,
    page_size: int = 12,
    session: Optional[Dict[str, Any]] = None,
    categories: Optional[Dict[str, Dict[str, Any]]] = None,
    persona_parse: Optional[Callable[[Optional[str]], Tuple[str, Optional[str]]]] = None,
    ai2ai_default_models: Optional[Callable[[List[str], bool], Tuple[Optional[str], Optional[str]]]] = None,
    allow_same_models: bool = False,
) -> InlineKeyboardMarkup:
    sess = session if session is not None else {}
    rows: List[List[InlineKeyboardButton]] = []
    start = page * page_size
    end = start + page_size
    subset = models[start:end]
    sel_a = sess.get('ai2ai_model_a')
    sel_b = sess.get('ai2ai_model_b')
    current = sess.get('model')
    mode = sess.get('mode') or ('ai-ai' if (sel_a and sel_b) else 'ai-human')

    # Top mode toggle
    mark_single = '✅' if mode == 'ai-human' else '⬜'
    mark_ai2ai = '✅' if mode == 'ai-ai' else '⬜'
    rows.append([
        InlineKeyboardButton(f"{mark_single} Single", callback_data="ollama_set_mode:single"),
        InlineKeyboardButton(f"{mark_ai2ai} AI↔AI", callback_data="ollama_set_mode:ai2ai"),
    ])

    if mode == 'ai-ai':
        # Ensure paging slots and defaults
        page_a = int(sess.get('ai2ai_page_a') or 0)
        page_b = int(sess.get('ai2ai_page_b') or 0)
        if session is not None and models and ai2ai_default_models:
            default_a, default_b = ai2ai_default_models(models, allow_same_models)
            if not sess.get('ai2ai_model_a') and default_a:
                session['ai2ai_model_a'] = default_a
            if not sess.get('ai2ai_model_b') and default_b:
                session['ai2ai_model_b'] = default_b
            session['active'] = bool(session.get('ai2ai_model_a') and session.get('ai2ai_model_b'))
            sel_a = session.get('ai2ai_model_a')
            sel_b = session.get('ai2ai_model_b')

        cats = categories or {}
        # Section A
        rows.append([InlineKeyboardButton("Model A:", callback_data="ollama_nop")])
        view_a = sess.get("ai2ai_view_a") or "models"
        if view_a not in ("models", "persona_categories", "persona_list"):
            view_a = "models"
        if session is not None:
            session["ai2ai_view_a"] = view_a
        rows.append(ai2ai_view_toggle_row("A", view_a))
        if view_a == "models":
            prov_a = (sess.get('ai2ai_provider_a') or 'ollama')
            if prov_a == 'cloud':
                # Render cloud options for A
                cloud_opts: List[Dict[str, Any]] = list(sess.get('ai2ai_cloud_options_a') or [])
                sel_cloud = sess.get('ai2ai_cloud_option_a') or {}
                sel_key = (sel_cloud.get('provider'), sel_cloud.get('model')) if isinstance(sel_cloud, dict) else (None, None)
                row: List[InlineKeyboardButton] = []
                for i, opt in enumerate(cloud_opts[:page_size*2]):  # show up to two rows by default
                    base_label = opt.get('button_label') or opt.get('label') or f"{opt.get('provider')}/{opt.get('model')}"
                    if len(base_label) > 24:
                        base_label = f"{base_label[:22]}…"
                    ok = (opt.get('provider'), opt.get('model')) == sel_key
                    label = f"✅ {base_label}" if ok else base_label
                    row.append(InlineKeyboardButton(label, callback_data=f"ollama_cloud_model:A:{i}"))
                    if len(row) == 2:
                        rows.append(row)
                        row = []
                if row:
                    rows.append(row)
            else:
                # Render local (Ollama) models for A
                a_start = page_a * page_size
                a_end = a_start + page_size
                a_subset = models[a_start:a_end]
                row: List[InlineKeyboardButton] = []
                for name in a_subset:
                    base_label = name
                    if len(base_label) > 28:
                        base_label = f"{base_label[:25]}…"
                    label = f"✅ {base_label}" if sel_a == name else base_label
                    row.append(InlineKeyboardButton(label, callback_data=f"ollama_set_a:{name}"))
                    if len(row) == 3:
                        rows.append(row)
                        row = []
                if row:
                    rows.append(row)
                nav_a: List[InlineKeyboardButton] = []
                if a_end < len(models):
                    nav_a.append(InlineKeyboardButton("➡️ More", callback_data=f"ollama_more_ai2ai:A:{page_a+1}"))
                if page_a > 0:
                    nav_a.insert(0, InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more_ai2ai:A:{page_a-1}"))
                if nav_a:
                    rows.append(nav_a)
        elif view_a == "persona_list":
            rows.extend(ai2ai_persona_list_rows("A", session or {}, page_size, cats, persona_parse or (lambda n: (n or "", None))))
        else:
            rows.extend(ai2ai_persona_categories_rows("A", session or {}, page_size, cats))

        # Section B
        rows.append([InlineKeyboardButton("Model B:", callback_data="ollama_nop")])
        view_b = sess.get("ai2ai_view_b") or "models"
        if view_b not in ("models", "persona_categories", "persona_list"):
            view_b = "models"
        if session is not None:
            session["ai2ai_view_b"] = view_b
        rows.append(ai2ai_view_toggle_row("B", view_b))
        if view_b == "models":
            prov_b = (sess.get('ai2ai_provider_b') or 'ollama')
            if prov_b == 'cloud':
                cloud_opts_b: List[Dict[str, Any]] = list(sess.get('ai2ai_cloud_options_b') or [])
                sel_cloud_b = sess.get('ai2ai_cloud_option_b') or {}
                sel_key_b = (sel_cloud_b.get('provider'), sel_cloud_b.get('model')) if isinstance(sel_cloud_b, dict) else (None, None)
                row: List[InlineKeyboardButton] = []
                for i, opt in enumerate(cloud_opts_b[:page_size*2]):
                    base_label = opt.get('button_label') or opt.get('label') or f"{opt.get('provider')}/{opt.get('model')}"
                    if len(base_label) > 24:
                        base_label = f"{base_label[:22]}…"
                    ok = (opt.get('provider'), opt.get('model')) == sel_key_b
                    label = f"✅ {base_label}" if ok else base_label
                    row.append(InlineKeyboardButton(label, callback_data=f"ollama_cloud_model:B:{i}"))
                    if len(row) == 2:
                        rows.append(row)
                        row = []
                if row:
                    rows.append(row)
            else:
                b_start = page_b * page_size
                b_end = b_start + page_size
                b_subset = models[b_start:b_end]
                row = []
                for name in b_subset:
                    base_label = name
                    if len(base_label) > 28:
                        base_label = f"{base_label[:25]}…"
                    label = f"✅ {base_label}" if sel_b == name else base_label
                    row.append(InlineKeyboardButton(label, callback_data=f"ollama_set_b:{name}"))
                    if len(row) == 3:
                        rows.append(row)
                        row = []
                if row:
                    rows.append(row)
                nav_b: List[InlineKeyboardButton] = []
                if b_end < len(models):
                    nav_b.append(InlineKeyboardButton("➡️ More", callback_data=f"ollama_more_ai2ai:B:{page_b+1}"))
                if page_b > 0:
                    nav_b.insert(0, InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more_ai2ai:B:{page_b-1}"))
                if nav_b:
                    rows.append(nav_b)
        elif view_b == "persona_list":
            rows.extend(ai2ai_persona_list_rows("B", session or {}, page_size, cats, persona_parse or (lambda n: (n or "", None))))
        else:
            rows.extend(ai2ai_persona_categories_rows("B", session or {}, page_size, cats))

        # Footer
        foot: List[InlineKeyboardButton] = []
        if sel_a and sel_b:
            foot.append(InlineKeyboardButton("🧠 Options", callback_data="ollama_ai2ai:opts"))
            foot.append(InlineKeyboardButton("♻️ Clear", callback_data="ollama_ai2ai:clear"))
        foot.append(InlineKeyboardButton("❌ Close", callback_data="ollama_cancel"))
        rows.append(foot)
        return InlineKeyboardMarkup(rows)

    # Single-mode picker
    cats = categories or {}
    view_single = sess.get("single_view") or "models"
    if view_single not in ("models", "persona_categories", "persona_list"):
        view_single = "models"
    if view_single == "persona_list" and not (session or {}).get("single_persona_category"):
        view_single = "persona_categories"
    if session is not None:
        session["single_view"] = view_single
        rows.append(single_view_toggle_row(view_single))
    if view_single == "models":
        row2: List[InlineKeyboardButton] = []
        for name in subset:
            label = name
            if len(label) > 28:
                label = f"{label[:25]}…"
            if name == (session or {}).get("ai2ai_model_a") or name == (session or {}).get("ai2ai_model_b") or name == current:
                label = f"✅ {label}"
            row2.append(InlineKeyboardButton(label, callback_data=f"ollama_model:{name}"))
            if len(row2) == 3:
                rows.append(row2)
                row2 = []
        if row2:
            rows.append(row2)
        nav: List[InlineKeyboardButton] = []
        if end < len(models):
            nav.append(InlineKeyboardButton("➡️ More", callback_data=f"ollama_more:{page+1}"))
        if page > 0:
            nav.insert(0, InlineKeyboardButton("⬅️ Back", callback_data=f"ollama_more:{page-1}"))
        if nav:
            rows.append(nav)
    elif view_single == "persona_list":
        rows.extend(single_persona_list_rows(session or {}, page_size, cats, persona_parse or (lambda n: (n or "", None))))
    else:
        rows.extend(single_persona_categories_rows(session or {}, page_size, cats))
    rows.append([InlineKeyboardButton("❌ Close", callback_data="ollama_cancel")])
    return InlineKeyboardMarkup(rows)
