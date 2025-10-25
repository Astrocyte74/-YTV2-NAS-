from __future__ import annotations

import hashlib
import itertools
from typing import Any, Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from modules.tts_hub import (
    available_accent_families,
    accent_family_label,
    filter_catalog_voices,
    DEFAULT_ENGINE,
)

ENGINE_PREFIXES = {
    "kokoro": "[K]",
    "xtts": "[X]",
    "openai": "[O]",
}


def short_engine_label(engine: Optional[str]) -> str:
    value = (engine or "").strip().lower()
    if not value:
        return "[?]"
    return ENGINE_PREFIXES.get(value, f"[{value[:1].upper()}]")


def build_combined_filters(voices: List[Dict[str, Any]]) -> Dict[str, Any]:
    genders_meta: List[Dict[str, Any]] = []
    seen_genders = set()
    accent_meta: List[Dict[str, Any]] = []
    seen_families = set()

    for voice in voices:
        gender = (voice.get('gender') or '').strip().lower()
        if gender and gender not in seen_genders:
            genders_meta.append({'id': gender, 'label': gender.title()})
            seen_genders.add(gender)

        accent = voice.get('accent') or {}
        family_id = (accent.get('id') or '').strip()
        if family_id and family_id not in seen_families:
            accent_meta.append({
                'id': family_id,
                'label': accent.get('label') or family_id.title(),
                'flag': accent.get('flag'),
            })
            seen_families.add(family_id)

    return {
        'genders': genders_meta,
        'accentFamilies': {'any': accent_meta},
    }


def strip_favorite_label(label: Optional[str]) -> str:
    if not label:
        return ""
    text = label.strip()
    if text.lower().startswith("favorite ·"):
        text = text.split("·", 1)[1].strip()
    return text


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
    favorites = session.get('favorites') or []
    selected_gender = session.get('selected_gender')
    selected_family = session.get('selected_family')

    mode = session.get('voice_mode')
    if mode not in ('favorites', 'all'):
        mode = 'favorites' if favorites else 'all'
    if mode == 'favorites' and not favorites:
        mode = 'all'
    use_favorites = mode == 'favorites'

    engine_order: List[str] = list(session.get('engine_order') or [])
    seen_engines = set(engine_order)

    def ensure_engine(engine: Optional[str]) -> None:
        if not engine:
            return
        if engine not in seen_engines:
            engine_order.append(engine)
            seen_engines.add(engine)

    ensure_engine(default_engine)
    for engine in sorted(catalogs.keys()):
        ensure_engine(engine)

    def normalized_engine(value: Optional[str]) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return default_engine

    existing_catalog = session.get('catalog')
    if active_engine != '__all__' and active_engine not in catalogs and isinstance(existing_catalog, dict):
        catalogs[active_engine] = existing_catalog
    catalogs.setdefault(default_engine, catalogs.get(default_engine) or {})

    for fav in favorites:
        if not isinstance(fav, dict):
            continue
        ensure_engine(normalized_engine(fav.get('engine')))

    session['engine_order'] = engine_order
    session['catalogs'] = catalogs

    def build_for_engine(engine: str, favorites_only: bool):
        catalog = catalogs.get(engine) or {}
        fav_map: Dict[str, Dict[str, Any]] = {}
        fav_order: List[str] = []
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
            fav_order.append(voice_id)

        if favorites_only and not fav_map:
            return [], fav_map, None, catalog

        allowed_ids = set(fav_map.keys()) if favorites_only and fav_map else None
        if favorites_only and allowed_ids:
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
        if favorites_only and allowed_ids:
            voice_map = {
                voice.get('id'): voice
                for voice in filtered
                if isinstance(voice, dict) and voice.get('id')
            }
            for voice_id in fav_order:
                voice_meta = voice_map.get(voice_id)
                if not voice_meta:
                    continue
                entry = dict(voice_meta)
                entry['engine'] = normalized_engine(entry.get('engine') or engine)
                entry['_favorite'] = fav_map[voice_id]
                display.append(entry)
        else:
            for voice_meta in filtered:
                if not isinstance(voice_meta, dict):
                    continue
                entry = dict(voice_meta)
                entry['engine'] = normalized_engine(entry.get('engine') or engine)
                fav_meta = fav_map.get(entry.get('id'))
                if fav_meta:
                    entry['_favorite'] = fav_meta
                display.append(entry)

        return display, fav_map, allowed_ids, catalog

    def aggregate_engines(engines: List[str], favorites_only: bool) -> List[Dict[str, Any]]:
        aggregated: List[Dict[str, Any]] = []
        for engine in engines:
            if engine == '__all__':
                continue
            engine_display, _, _, cat = build_for_engine(engine, favorites_only)
            if engine_display:
                aggregated.extend(engine_display)
            if cat and cat is not catalogs.get(engine):
                catalogs[engine] = cat
        return aggregated

    display_voices: List[Dict[str, Any]] = []
    favorite_by_voice: Dict[str, Dict[str, Any]] = {}
    allowed_ids_set: Optional[set[str]] = None
    engine_switch_hint: Optional[str] = None
    active_catalog: Dict[str, Any] = catalogs.get(default_engine if active_engine == '__all__' else active_engine) or {}

    if active_engine == '__all__':
        display_voices = aggregate_engines(engine_order, use_favorites)
        if display_voices:
            active_catalog = {
                'voices': display_voices,
                'filters': build_combined_filters(display_voices),
            }
        else:
            for engine in engine_order:
                if engine == '__all__':
                    continue
                engine_display, fav_map, allowed_ids, cat = build_for_engine(engine, use_favorites)
                if engine_display:
                    active_engine = engine
                    display_voices = engine_display
                    favorite_by_voice = fav_map
                    allowed_ids_set = allowed_ids
                    active_catalog = cat
                    engine_switch_hint = f"No favorites matched all engines; showing {engine.upper()}."
                    break
    else:
        display_voices, favorite_by_voice, allowed_ids_set, active_catalog = build_for_engine(active_engine, use_favorites)

    if use_favorites and not display_voices:
        for engine in engine_order:
            if engine in ('__all__', active_engine):
                continue
            engine_display, fav_map, allowed_ids, cat = build_for_engine(engine, True)
            if engine_display:
                active_engine = engine
                display_voices = engine_display
                favorite_by_voice = fav_map
                allowed_ids_set = allowed_ids
                active_catalog = cat
                engine_switch_hint = f"No favorites matched the previous engine; switched to {engine.upper()}."
                break

    fallback_to_all = False
    if not display_voices:
        mode = 'all'
        use_favorites = False
        if active_engine == '__all__':
            display_voices = aggregate_engines(engine_order, False)
            if display_voices:
                active_catalog = {
                    'voices': display_voices,
                    'filters': build_combined_filters(display_voices),
                }
        else:
            display_voices, _, _, active_catalog = build_for_engine(active_engine, False)
            if not display_voices:
                for engine in engine_order:
                    if engine in ('__all__', active_engine):
                        continue
                    display_voices, _, _, active_catalog = build_for_engine(engine, False)
                    if display_voices:
                        active_engine = engine
                        break
        fallback_to_all = bool(display_voices)

    if not use_favorites:
        allowed_ids_set = None

    session['voice_mode'] = mode
    session['active_engine'] = active_engine
    session['catalog'] = active_catalog

    filters = (active_catalog.get('filters') or {}) if isinstance(active_catalog, dict) else {}

    engine_keys = ['__all__'] + [eng for eng in engine_order if eng != '__all__']

    rows: List[List[InlineKeyboardButton]] = []

    mark_fav = '✅' if mode == 'favorites' else '⬜'
    mark_all = '✅' if mode == 'all' else '⬜'
    rows.append([
        InlineKeyboardButton(f"{mark_fav} Favorites", callback_data="tts_mode:favorites"),
        InlineKeyboardButton(f"{mark_all} All voices", callback_data="tts_mode:all"),
    ])

    if len(engine_keys) > 1:
        remaining_engines = [eng for eng in engine_keys if eng != '__all__']
        if '__all__' in engine_keys:
            mark = '✅' if active_engine == '__all__' else '⬜'
            ordered_engines: List[str] = []
            for eng in engine_order:
                if not eng or eng == '__all__' or eng in ordered_engines:
                    continue
                ordered_engines.append(eng)
            for eng in remaining_engines:
                if not eng or eng in ordered_engines:
                    continue
                ordered_engines.append(eng)
            badges_all = [short_engine_label(eng) for eng in ordered_engines if eng]
            badges_display = " ".join(badges_all[:6])
            if len(badges_all) > 6:
                badges_display = f"{badges_display} ..."
            label = "ALL VOICE ENGINES"
            if badges_display:
                label = f"{label} {badges_display}"
            rows.append([
                InlineKeyboardButton(f"{mark} {label}", callback_data="tts_engine:__all__")
            ])
        engine_buttons: List[InlineKeyboardButton] = []
        chunk_size = 2 if len(remaining_engines) >= 4 else 3
        for engine in remaining_engines:
            mark = '✅' if engine == active_engine else '⬜'
            label = f"{short_engine_label(engine)} {engine.upper()}"
            engine_buttons.append(InlineKeyboardButton(f"{mark} {label}", callback_data=f"tts_engine:{engine}"))
            if len(engine_buttons) == chunk_size:
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
    if fallback_to_all and display_voices:
        info = "No favorites matched; showing all voices"
        if active_engine != '__all__':
            info = f"No favorites matched {active_engine.upper()}; showing all voices"
        rows.append([InlineKeyboardButton(f"ℹ️ {info}", callback_data="tts_nop")])

    voice_lookup: Dict[str, Dict[str, Any]] = {}
    alias_map: Dict[str, str] = {}
    alias_counter = itertools.count(1)
    display_keys: List[str] = []

    def register_voice_key(raw_key: str, entry: Dict[str, Any]) -> str:
        """Store entry under raw key and return a callback-safe slug."""
        voice_lookup[raw_key] = entry
        slug = raw_key
        callback = f"tts_voice:{slug}"
        if len(callback.encode('utf-8')) > 64:
            token = format(next(alias_counter), 'x')
            slug = f"alias|{token}"
            callback = f"tts_voice:{slug}"
            if len(callback.encode('utf-8')) > 64:
                digest = hashlib.sha1(raw_key.encode('utf-8')).hexdigest()[:16]
                slug = f"alias|{digest}"
                callback = f"tts_voice:{slug}"
        voice_lookup[slug] = entry
        if slug.startswith('alias|'):
            alias_token = slug.split('|', 1)[1]
            alias_map[alias_token] = raw_key
        return slug

    for voice in display_voices:
        voice_id = voice.get('id')
        if not voice_id:
            continue
        engine = normalized_engine(voice.get('engine'))
        primary_key = f"cat|{engine}|{voice_id}"
        entry = {
            'voice': voice,
            'voiceId': voice_id,
            'engine': engine,
            'label': voice.get('label'),
        }
        fav_meta = voice.get('_favorite')
        if fav_meta:
            fav_label = strip_favorite_label(fav_meta.get('label'))
            entry.update(
                {
                    'favoriteSlug': fav_meta.get('slug') or fav_meta.get('id') or voice_id,
                    'label': fav_label or entry['label'],
                }
            )
        slug_key = register_voice_key(primary_key, entry)
        display_keys.append(slug_key)
        if fav_meta:
            alias_key = f"fav|{engine}|{(fav_meta.get('slug') or fav_meta.get('voiceId') or voice_id)}"
            register_voice_key(alias_key, entry)

    session['voice_lookup'] = voice_lookup
    session['voice_display_keys'] = display_keys
    session['voice_alias_map'] = alias_map

    voice_buttons: List[InlineKeyboardButton] = []
    for key in display_keys:
        entry = voice_lookup.get(key)
        if not entry:
            continue
        base_label = strip_favorite_label(entry.get('label')) or entry.get('voiceId') or key
        prefix = short_engine_label(entry.get('engine'))
        display_label = f"{prefix} {base_label}".strip()
        entry['display_label'] = display_label
        button_label = display_label if len(display_label) <= 32 else f"{display_label[:29]}…"
        entry['button_label'] = button_label
        voice_buttons.append(InlineKeyboardButton(button_label, callback_data=f"tts_voice:{key}"))

    grouped: List[List[InlineKeyboardButton]] = []
    row = []
    for button in voice_buttons:
        row.append(button)
        if len(row) == 3:
            grouped.append(row)
            row = []
    if row:
        grouped.append(row)
    # Optional quick-pick favorites (up to 2) at the top
    quick_list = []
    q_any = session.get('quick_favorite_slugs')
    if isinstance(q_any, list):
        quick_list = [str(x).strip() for x in q_any if str(x).strip()]
    else:
        # Backward compatibility for single slug
        quick_slug = (session.get('quick_favorite_slug') or '').strip()
        if quick_slug:
            quick_list = [quick_slug]

    quick_buttons: List[InlineKeyboardButton] = []
    for slug in quick_list[:2]:
        entry = voice_lookup.get(slug)
        if not entry:
            continue
        label = entry.get('display_label') or entry.get('button_label') or entry.get('label') or 'favorite'
        quick_label = label if len(label) <= 28 else f"{label[:25]}…"
        quick_buttons.append(InlineKeyboardButton(f"🎤 Quick • {quick_label}", callback_data=f"tts_voice:{slug}"))
    if quick_buttons:
        rows.append(quick_buttons)
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
        base_label = strip_favorite_label(profile.get('label')) or profile.get('voiceId') or slug
        prefix = short_engine_label(profile.get('engine'))
        label = f"{prefix} {base_label}".strip()
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
    alias_map = session.get('voice_alias_map') or {}
    raw_slug = slug
    if slug.startswith('alias|'):
        token = slug.split('|', 1)[1]
        raw_slug = alias_map.get(token, raw_slug)
    entry = lookup.get(slug) or lookup.get(raw_slug)
    if entry:
        return (
            entry.get('display_label')
            or entry.get('button_label')
            or entry.get('label')
            or entry.get('voiceId')
            or slug
        )

    parts = raw_slug.split('|')
    kind = parts[0] if parts else ''
    engine_hint = parts[1] if len(parts) > 2 else (parts[1] if len(parts) > 1 and kind != 'cat' else '')
    identifier = parts[-1] if len(parts) > 1 else raw_slug

    catalogs = session.get('catalogs') or {}
    search_engines: List[str] = []
    if engine_hint:
        search_engines.append(engine_hint)
    for eng in catalogs.keys():
        if eng not in search_engines:
            search_engines.append(eng)

    for engine in search_engines:
        catalog = catalogs.get(engine) or {}
        for voice in catalog.get('voices') or []:
            if not isinstance(voice, dict):
                continue
            voice_id = voice.get('id')
            if voice_id == identifier:
                base_label = strip_favorite_label(voice.get('label')) or identifier
                accent = voice.get('accent') or {}
                flag = accent.get('flag')
                prefix = short_engine_label(engine or voice.get('engine'))
                display = f"{prefix} {base_label}".strip()
                if flag:
                    display = f"{flag} {display}"
                return display

    favorites = session.get('favorites') or []
    for fav in favorites:
        if not isinstance(fav, dict):
            continue
        fav_slug = fav.get('slug') or fav.get('voiceId')
        if fav_slug == identifier:
            prefix = short_engine_label(fav.get('engine'))
            label = strip_favorite_label(fav.get('label')) or fav_slug
            return f"{prefix} {label}".strip()

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
