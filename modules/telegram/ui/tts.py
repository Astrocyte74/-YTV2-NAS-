from __future__ import annotations

from typing import Any, Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from modules.tts_hub import available_accent_families, accent_family_label, filter_catalog_voices


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
    catalog = session.get('catalog') or {}
    filters = (catalog.get('filters') or {}) if catalog else {}
    favorites = session.get('favorites') or []
    selected_gender = session.get('selected_gender')
    selected_family = session.get('selected_family')

    mode = session.get('voice_mode')
    if mode not in ('favorites', 'all'):
        mode = 'favorites' if favorites else 'all'
    if mode == 'favorites' and not favorites:
        mode = 'all'
    session['voice_mode'] = mode

    favorite_by_voice: Dict[str, Dict[str, Any]] = {}
    allowed_ids = None
    if favorites:
        for fav in favorites:
            voice_id = fav.get('voiceId')
            if voice_id:
                favorite_by_voice[voice_id] = fav
        if mode == 'favorites':
            allowed_ids = {vid for vid in favorite_by_voice.keys() if vid}

    rows: List[List[InlineKeyboardButton]] = []

    mark_fav = '✅' if mode == 'favorites' else '⬜'
    mark_all = '✅' if mode == 'all' else '⬜'
    rows.append([
        InlineKeyboardButton(f"{mark_fav} Favorites", callback_data="tts_mode:favorites"),
        InlineKeyboardButton(f"{mark_all} All voices", callback_data="tts_mode:all"),
    ])

    rows.append([InlineKeyboardButton("Gender", callback_data="tts_nop")])
    genders = filters.get('genders') or []
    gender_row: List[InlineKeyboardButton] = []
    mark_all_gender = '✅' if not selected_gender else '⬜'
    gender_row.append(InlineKeyboardButton(f"{mark_all_gender} All", callback_data="tts_gender:all"))
    for entry in genders:
        gid = entry.get('id')
        if not gid:
            continue
        label = entry.get('label') or gid.title()
        mark = '✅' if selected_gender == gid else '⬜'
        gender_row.append(InlineKeyboardButton(f"{mark} {label}", callback_data=f"tts_gender:{gid}"))
    rows.append(gender_row)

    rows.append([InlineKeyboardButton("Accent", callback_data="tts_nop")])
    family_options = available_accent_families(catalog, gender=selected_gender, allowed_ids=allowed_ids)
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
    voice_lookup: Dict[str, Dict[str, Any]] = {}

    display_voices: List[Dict[str, Any]] = []
    if mode == 'favorites' and allowed_ids:
        filtered = filter_catalog_voices(
            catalog,
            gender=selected_gender,
            family=selected_family,
            allowed_ids=allowed_ids,
        )
        id_to_voice = {voice.get('id'): voice for voice in filtered if voice.get('id')}
        for voice_id, fav in favorite_by_voice.items():
            voice_meta = id_to_voice.get(voice_id)
            if not voice_meta:
                continue
            entry = dict(voice_meta)
            entry['_favorite'] = fav
            display_voices.append(entry)
    else:
        display_voices = filter_catalog_voices(
            catalog,
            gender=selected_gender,
            family=selected_family,
        )

    id_to_voice = {voice.get('id'): voice for voice in display_voices if voice.get('id')}
    for voice in display_voices:
        voice_id = voice.get('id')
        if not voice_id:
            continue
        short_key = f"cat|{voice_id}"
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
        voice_lookup[short_key] = entry
        if fav_meta:
            legacy_key = f"fav|{(fav_meta.get('slug') or fav_meta.get('voiceId') or voice_id)}"
            voice_lookup[legacy_key] = entry
        legacy_cat = f"cat|{voice_id}"
        voice_lookup[legacy_cat] = entry

    session['voice_lookup'] = voice_lookup

    voice_buttons: List[InlineKeyboardButton] = []
    for key, entry in voice_lookup.items():
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
    buttons.append([InlineKeyboardButton("❌ Cancel", callback_data="tts_cancel")])
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
