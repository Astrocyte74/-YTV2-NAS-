from __future__ import annotations

from typing import Any, Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from modules.tts_hub import (
    available_accent_families,
    accent_family_label,
    filter_catalog_voices,
    DEFAULT_ENGINE,
)


def gender_label(gender: Optional[str]) -> str:
    if not gender:
        return "All genders"
    mapping = {
        "female": "Female",
        "male": "Male",
        "unknown": "Unknown",
    }
    return mapping.get(gender, gender.title())


def build_tts_catalog_keyboard(session: Dict[str, Any]) -> InlineKeyboardMarkup:
    catalogs = dict(session.get('catalogs') or {})
    default_engine = session.get('default_engine') or DEFAULT_ENGINE
    active_engine = session.get('active_engine') or default_engine

    existing_catalog = session.get('catalog')
    if active_engine not in catalogs:
        if isinstance(existing_catalog, dict):
            catalogs[active_engine] = existing_catalog
        else:
            catalogs[active_engine] = {}
    session['catalogs'] = catalogs

    favorites = session.get('favorites') or []
    selected_gender = session.get('selected_gender')
    selected_family = session.get('selected_family')

    mode = session.get('voice_mode')
    if mode not in ('favorites', 'all'):
        mode = 'favorites' if favorites else 'all'
    if mode == 'favorites' and not favorites:
        mode = 'all'

    def normalized_engine(value: Optional[str]) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return default_engine

    def build_for_engine(engine: str, use_favorites: bool):
        catalog = catalogs.get(engine) or {}
        fav_map: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for fav in favorites:
            if not isinstance(fav, dict):
                continue
            voice_id = fav.get('voiceId')
            if not voice_id:
                continue
            if normalized_engine(fav.get('engine')) != engine:
                continue
            if voice_id in fav_map:
                continue
            fav_map[voice_id] = fav
            order.append(voice_id)
        if use_favorites and not fav_map:
            return [], fav_map, None, catalog

        allowed_ids = set(fav_map.keys()) if use_favorites and fav_map else None
        if use_favorites:
            filtered = filter_catalog_voices(
                catalog,
                gender=selected_gender,
                family=selected_family,
                allowed_ids=allowed_ids,
            )
        else:
            filtered = filter_catalog_voices(
                catalog,
                gender=selected_gender,
                family=selected_family,
            )

        display: List[Dict[str, Any]] = []
        if use_favorites and allowed_ids:
            voice_map = {
                voice.get('id'): voice
                for voice in filtered
                if isinstance(voice, dict) and voice.get('id')
            }
            for voice_id in order:
                voice_meta = voice_map.get(voice_id)
                if not voice_meta:
                    continue
                entry = dict(voice_meta)
                entry['_favorite'] = fav_map[voice_id]
                display.append(entry)
        else:
            for voice_meta in filtered:
                entry = dict(voice_meta)
                voice_id = entry.get('id')
                fav_meta = fav_map.get(voice_id)
                if fav_meta:
                    entry['_favorite'] = fav_meta
                display.append(entry)

        return display, fav_map, allowed_ids, catalog

    use_favorites = mode == 'favorites'
    display_voices, favorite_by_voice, allowed_ids_set, active_catalog = build_for_engine(active_engine, use_favorites)

    # Determine a stable engine order for the keyboard.
    engine_order: List[str] = list(session.get('engine_order') or [])
    seen_engines = set(engine_order)
    def ensure_engine_order(engine: str) -> None:
        if engine not in seen_engines:
            engine_order.append(engine)
            seen_engines.add(engine)

    ensure_engine_order(default_engine)
    for eng in sorted(catalogs.keys()):
        ensure_engine_order(eng)

    favorite_engines: List[str] = []
    for fav in favorites:
        if not isinstance(fav, dict):
            continue
        voice_id = fav.get('voiceId')
        if not voice_id:
            continue
        fav_engine = normalized_engine(fav.get('engine'))
        if fav_engine not in favorite_engines:
            favorite_engines.append(fav_engine)
    for eng in favorite_engines:
        ensure_engine_order(eng)

    session['engine_order'] = engine_order

    engine_switch_hint: Optional[str] = None
    engine_keys = engine_order

    if use_favorites and not display_voices:
        for engine in engine_keys:
            if engine == active_engine:
                continue
            alt_display, alt_map, alt_allowed, alt_catalog = build_for_engine(engine, True)
            if alt_display:
                active_engine = engine
                display_voices = alt_display
                favorite_by_voice = alt_map
                allowed_ids_set = alt_allowed
                active_catalog = alt_catalog
                engine_switch_hint = f"No favorites matched the previous engine; switched to {engine.upper()}."
                break

    fallback_to_all = False
    if not display_voices:
        mode = 'all'
        use_favorites = False
        display_voices, favorite_by_voice, _, active_catalog = build_for_engine(active_engine, False)
        allowed_ids_set = None
        fallback_to_all = True

    session['voice_mode'] = mode
    session['active_engine'] = active_engine
    session['catalog'] = active_catalog

    filters = (active_catalog.get('filters') or {}) if isinstance(active_catalog, dict) else {}

    rows: List[List[InlineKeyboardButton]] = []

    mark_fav = '✅' if mode == 'favorites' else '⬜'
    mark_all = '✅' if mode == 'all' else '⬜'
    rows.append([
        InlineKeyboardButton(f"{mark_fav} Favorites", callback_data="tts_mode:favorites"),
        InlineKeyboardButton(f"{mark_all} All voices", callback_data="tts_mode:all"),
    ])

    if len(engine_keys) > 1:
        engine_buttons: List[InlineKeyboardButton] = []
        for engine in engine_keys:
            mark = '✅' if engine == active_engine else '⬜'
            label = engine.upper()
            engine_buttons.append(InlineKeyboardButton(f"{mark} {label}", callback_data=f"tts_engine:{engine}"))
            if len(engine_buttons) == 3:
                rows.append(engine_buttons)
                engine_buttons = []
        if engine_buttons:
            rows.append(engine_buttons)

    rows.append([InlineKeyboardButton("Gender", callback_data="tts_nop")])
    genders = filters.get('genders') or []
    gender_buttons: List[InlineKeyboardButton] = []
    mark_all_gender = '✅' if not selected_gender else '⬜'
    gender_buttons.append(InlineKeyboardButton(f"{mark_all_gender} All", callback_data="tts_gender:all"))
    for entry in genders:
        gid = entry.get('id')
        if not gid:
            continue
        label = entry.get('label') or gid.title()
        mark = '✅' if selected_gender == gid else '⬜'
        gender_buttons.append(InlineKeyboardButton(f"{mark} {label}", callback_data=f"tts_gender:{gid}"))
    rows.append(gender_buttons)

    rows.append([InlineKeyboardButton("Accent", callback_data="tts_nop")])
    family_options = available_accent_families(
        active_catalog or {},
        gender=selected_gender,
        allowed_ids=allowed_ids_set,
    )
    accent_rows: List[List[InlineKeyboardButton]] = []
    mark_all_family = '✅' if not selected_family else '⬜'
    accent_rows.append([InlineKeyboardButton(f"{mark_all_family} All", callback_data="tts_accent:all")])

    row: List[InlineKeyboardButton] = []
    for entry in family_options:
        family_id = entry.get('id')
        if not family_id:
            continue
        label = entry.get('label') or family_id.title()
        flag = entry.get('flag') or ''
        mark = '✅' if selected_family == family_id else '⬜'
        button_label = f"{mark} {flag} {label}".strip()
        row.append(InlineKeyboardButton(button_label, callback_data=f"tts_accent:{family_id}"))
        if len(row) == 3:
            accent_rows.append(row)
            row = []
    if row:
        accent_rows.append(row)
    rows.extend(accent_rows)

    rows.append([InlineKeyboardButton("Voices", callback_data="tts_nop")])
    if engine_switch_hint:
        rows.append([InlineKeyboardButton(f"ℹ️ {engine_switch_hint}", callback_data="tts_nop")])
    if fallback_to_all:
        rows.append([
            InlineKeyboardButton(
                f"ℹ️ No favorites for {active_engine.upper()}; showing all voices",
                callback_data="tts_nop",
            )
        ])

    voice_lookup: Dict[str, Dict[str, Any]] = {}
    display_keys: List[str] = []

    for voice in display_voices:
        voice_id = voice.get('id')
        if not voice_id:
            continue
        primary_key = f"cat|{voice_id}"
        entry = {
            'voice': voice,
            'voiceId': voice_id,
            'engine': voice.get('engine'),
            'label': voice.get('label'),
        }
        fav_meta = voice.get('_favorite')
        if fav_meta:
            entry.update(
                {
                    'favoriteSlug': fav_meta.get('slug') or fav_meta.get('id') or voice_id,
                    'label': fav_meta.get('label') or entry['label'],
                }
            )
        voice_lookup[primary_key] = entry
        if primary_key not in display_keys:
            display_keys.append(primary_key)
        if fav_meta:
            alias_key = f"fav|{(fav_meta.get('slug') or fav_meta.get('voiceId') or voice_id)}"
            voice_lookup[alias_key] = entry

    session['voice_lookup'] = voice_lookup
    session['voice_display_keys'] = display_keys

    voice_buttons: List[InlineKeyboardButton] = []
    for key in display_keys:
        entry = voice_lookup.get(key)
        if not entry:
            continue
        label = entry.get('label') or entry.get('voiceId') or key
        if len(label) > 32:
            label = f"{label[:29]}…"
        voice_buttons.append(InlineKeyboardButton(label, callback_data=f"tts_voice:{key}"))

    grouped: List[List[InlineKeyboardButton]] = []
    row = []
    for button in voice_buttons:
        row.append(button)
        if len(row) == 3:
            grouped.append(row)
            row = []
    if row:
        grouped.append(row)
    rows.extend(grouped)

    rows.append([InlineKeyboardButton("❌ Cancel", callback_data="tts_cancel")])
    return InlineKeyboardMarkup(rows)


def build_tts_keyboard(favorites: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    buttons: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    max_per_row = 3

    for i, profile in enumerate(favorites):
        slug = profile.get('slug') or profile.get('voiceId')
        if not slug:
            continue
        label = profile.get('label') or profile.get('voiceId') or slug
        if label.startswith('Favorite ·'):
            label = label.split('·', 1)[1].strip() or label
        if len(label) > 28:
            label = f"{label[:25]}…"
        short_key = f"v{i}"
        row.append(InlineKeyboardButton(f"🎤 {label}", callback_data=f"tts_voice:{short_key}"))
        if len(row) == max_per_row:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton('❌ Cancel', callback_data='tts_cancel')])
    return InlineKeyboardMarkup(buttons)


def tts_voice_label(session: Dict[str, Any], slug: str) -> str:
    lookup = session.get('voice_lookup') or {}
    entry = lookup.get(slug)
    if entry:
        label = entry.get('label')
        if label:
            return label
        voice = entry.get('voice') or {}
        accent = voice.get('accent') or {}
        base_label = voice.get('label') or entry.get('voiceId') or slug
        flag = accent.get('flag')
        if flag:
            return f"{flag} {base_label}"
        return base_label
    base_slug = slug.split('|', 1)[-1]
    catalog = session.get('catalog') or {}
    for voice in catalog.get('voices') or []:
        if voice.get('id') == base_slug:
            label = voice.get('label') or slug
            accent = voice.get('accent') or {}
            flag = accent.get('flag')
            if flag:
                return f"{flag} {label}"
            return label
    favorites = session.get('favorites') or []
    for profile in favorites:
        profile_slug = profile.get('slug') or profile.get('voiceId')
        if profile_slug == base_slug:
            raw_label = profile.get('label') or profile.get('voiceId') or slug
            if raw_label.startswith('Favorite ·'):
                return raw_label.split('·', 1)[1].strip() or slug
            return raw_label
    return slug


def tts_prompt_text(
    text: str,
    *,
    last_voice: Optional[str] = None,
    gender: Optional[str] = None,
    family: Optional[str] = None,
    catalog: Optional[Dict[str, Any]] = None,
) -> str:
    snippet = (text or '').strip()
    if len(snippet) > 280:
        snippet = snippet[:277].rstrip() + '…'
    lines: List[str] = []
    lines.append('──────────────────────────────')
    if last_voice:
        lines.append(f'✅ Last voice: {last_voice}')
    lines.append('🗣️ Ready to synthesize speech for:')
    lines.append(f'“{snippet or "…"}”')
    gender_text = gender_label(gender)
    family_label = accent_family_label(catalog or {}, family)
    lines.append(f'Filters: {gender_text} · {family_label}')
    lines.append('Select a voice below or cancel.')
    lines.append('──────────────────────────────')
    return '\n'.join(lines)


def build_local_failure_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton('⏳ Queue for later', callback_data='tts_queue_local'),
                InlineKeyboardButton('☁️ Use OpenAI now', callback_data='tts_switch_provider:openai'),
            ],
            [InlineKeyboardButton('❌ Cancel', callback_data='tts_cancel')],
        ]
    )
