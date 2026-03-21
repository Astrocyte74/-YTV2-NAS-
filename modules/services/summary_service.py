from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
import os
import re
import time
import urllib.parse
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

try:
    import requests
    from requests.exceptions import ConnectionError as RequestsConnectionError
except ImportError:  # pragma: no cover - optional dependency
    requests = None
    RequestsConnectionError = None

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest, RetryAfter

from modules.metrics import metrics
from modules import ledger
from modules.report_generator import create_report_from_youtube_summarizer
from modules.summary_variants import merge_summary_variants, normalize_variant_id
from modules.event_stream import emit_report_event
from modules.services import sync_service
from modules.summary_queue import enqueue as enqueue_summary_job
from modules.ollama_client import OllamaClientError
from modules.cjclds import classify_and_apply_cjclds


def _format_duration_and_savings(metadata: Dict[str, Any]) -> str:
    """Return formatted duration with estimated summary time savings."""
    duration = int(metadata.get('duration') or 0)
    if not duration:
        return "⏱️ **Duration**: Unknown"

    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60

    if hours > 0:
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        duration_str = f"{minutes:02d}:{seconds:02d}"

    reading_time_seconds = 180  # 3 minutes average summary read
    if duration <= reading_time_seconds:
        return f"⏱️ **Duration**: {duration_str}"

    time_saved = duration - reading_time_seconds
    saved_hours = time_saved // 3600
    saved_minutes = (time_saved % 3600) // 60
    saved_seconds = time_saved % 60

    if saved_hours > 0:
        savings_str = f"{saved_hours:02d}:{saved_minutes:02d}:00"
    else:
        savings_str = f"{saved_minutes:02d}:{saved_seconds:02d}"

    return f"⏱️ **Duration**: {duration_str} → ~3 min read (⏰ Saves {savings_str})"


LOCAL_REPORTS_DIR = Path("./data/reports")


def get_dashboard_base() -> Optional[str]:
    return (
        os.getenv('DASHBOARD_URL')
        or os.getenv('POSTGRES_DASHBOARD_URL')
        or os.getenv('RENDER_DASHBOARD_URL')
    )


def _should_show_follow_up_button(handler, summary_type: str) -> bool:
    base_type = normalize_variant_id(summary_type or "")
    if not getattr(handler, "_follow_up_enabled", lambda: False)():
        return False
    return bool(base_type and not base_type.startswith("audio") and base_type != "deep-research")


def _resolve_content_identity(handler, source: str, requested_content_id: str, result: Dict[str, Any]) -> tuple[str, str, str]:
    canonical_content_id = str(
        result.get("group_content_id")
        or (result.get("metadata") or {}).get("group_content_id")
        or result.get("id")
        or (result.get("metadata") or {}).get("content_id")
        or requested_content_id
        or ""
    ).strip()
    normalized_video_id = handler._normalize_content_id(canonical_content_id)
    display_id = f"{source}:{normalized_video_id}" if source != "youtube" else normalized_video_id
    return canonical_content_id, normalized_video_id, display_id


def _build_summary_result_keyboard(
    handler,
    *,
    dashboard_url: str,
    report_id: str,
    video_id: str,
    base_type: str,
    add_variant_content_id: str | None = None,
    follow_up_state: str | None = None,
) -> InlineKeyboardMarkup:
    keyboard: List[List[InlineKeyboardButton]] = []
    report_id_encoded = urllib.parse.quote(report_id, safe='') if report_id else ''

    row1 = [InlineKeyboardButton("📊 Dashboard", url=dashboard_url)]
    if report_id_encoded:
        row1.append(InlineKeyboardButton("📄 Open Summary", url=f"{dashboard_url}#report={report_id_encoded}"))
    keyboard.append(row1)

    if report_id_encoded:
        listen_cb = f"listen_this:{video_id}:{base_type}"
        gen_cb = f"gen_quiz:{video_id}"
        row2 = []
        if len(listen_cb.encode('utf-8')) <= 64:
            row2.append(InlineKeyboardButton("▶️ Listen", callback_data=listen_cb))
        if len(gen_cb.encode('utf-8')) <= 64:
            row2.append(InlineKeyboardButton("🧩 Generate Quiz", callback_data=gen_cb))
        if row2:
            keyboard.append(row2)

        import hashlib

        short_id = hashlib.md5(video_id.encode()).hexdigest()[:8]
        delete_cb = f"summary_delete:{short_id}"
        if hasattr(handler, '_delete_id_map'):
            handler._delete_id_map[short_id] = video_id
        keyboard.append([InlineKeyboardButton("🗑️ Delete", callback_data=delete_cb)])

        if follow_up_state == "pending":
            keyboard.append([InlineKeyboardButton("⏳ Deep Research Preparing", callback_data="followup:pending")])
        elif follow_up_state == "ready":
            keyboard.append([InlineKeyboardButton("🔍 Deep Research", callback_data="followup:open")])

        keyboard.append([
            InlineKeyboardButton(
                "➕ Add Variant",
                callback_data=handler._build_summary_callback("back_to_main", add_variant_content_id or video_id),
            )
        ])

    return InlineKeyboardMarkup(keyboard)


def _find_existing_report_for_content_id(handler, content_id: Optional[str]) -> tuple[Optional[Path], Optional[Dict[str, Any]]]:
    if not content_id:
        return None, None

    reports_dir = Path(getattr(handler.json_exporter, "reports_dir", "data/reports"))
    normalized_id = handler._normalize_content_id(content_id)
    for path in reports_dir.glob(f"*{normalized_id}*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue

        report_id = str(data.get("id") or "").strip()
        if report_id == content_id or handler._normalize_content_id(report_id) == normalized_id:
            return path, data

    return None, None


async def _safe_edit_status(query, text: str, reply_markup=None):
    while True:
        try:
            return await query.edit_message_text(
                text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )
        except RetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
        except BadRequest as exc:
            message = (getattr(exc, "message", None) or str(exc) or "").lower()
            if "query is too old" in message or "message to edit not found" in message:
                return await query.message.reply_text(
                    text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=reply_markup,
                )
            raise


def _image_queue_has_job(content_id: str) -> bool:
    if not content_id:
        return False
    try:
        qdir = Path("data/image_queue")
        if not qdir.exists():
            return False
        for path in qdir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            content = data.get("content") or {}
            existing = str(content.get("id") or content.get("video_id") or "")
            if existing == content_id:
                return True
    except Exception:
        return False
    return False


def post_dashboard_json(endpoint: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
    base = get_dashboard_base()
    if not base or requests is None:
        return None
    try:
        url = f"{base.rstrip('/')}{endpoint}"
        response = requests.post(url, json=payload, timeout=timeout)
        if 200 <= response.status_code < 300:
            return response.json()
    except Exception as exc:  # pragma: no cover - network call
        logging.debug("Dashboard POST failed (%s): %s", endpoint, exc)
    return None


def _is_local_summary_unavailable(exc: Exception) -> bool:
    if isinstance(exc, OllamaClientError):
        return True
    if RequestsConnectionError and isinstance(exc, RequestsConnectionError):
        return True
    if isinstance(exc, (ConnectionError, asyncio.TimeoutError)):
        return True
    message = str(exc).lower()
    keywords = (
        "connection refused",
        "failed to establish",
        "connection reset",
        "unreachable",
        "timed out",
        "timeout",
        "bad gateway",
    )
    return any(keyword in message for keyword in keywords)


_SUMMARY_PLACEHOLDER_PREFIXES = (
    "unable to generate",
    "unable to create",
    "summary generation failed",
)


def _extract_primary_summary_text(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    summary = result.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    if isinstance(summary, dict):
        for key in ("summary", "comprehensive", "bullet_points", "key_insights", "audio"):
            candidate = summary.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    for key in ("comprehensive", "bullet_points", "key_insights", "audio"):
        candidate = result.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def extract_summary_text_for_variant(result: Dict[str, Any], summary_type: str) -> str:
    summary_data = result.get("summary", {})
    if isinstance(summary_data, dict):
        if summary_type == "audio":
            return (
                summary_data.get("audio")
                or summary_data.get("content", {}).get("audio")
                or summary_data.get("content", {}).get("comprehensive")
                or summary_data.get("comprehensive")
                or summary_data.get("summary")
                or ""
            )
        if summary_type == "bullet-points":
            return (
                summary_data.get("bullet_points")
                or summary_data.get("content", {}).get("bullet_points")
                or summary_data.get("content", {}).get("comprehensive")
                or summary_data.get("comprehensive")
                or summary_data.get("summary")
                or ""
            )
        if summary_type == "key-insights":
            return (
                summary_data.get("key_insights")
                or summary_data.get("content", {}).get("key_insights")
                or summary_data.get("content", {}).get("comprehensive")
                or summary_data.get("comprehensive")
                or summary_data.get("summary")
                or ""
            )
        return (
            summary_data.get("comprehensive")
            or summary_data.get("content", {}).get("comprehensive")
            or summary_data.get("content", {}).get("audio")
            or summary_data.get("audio")
            or summary_data.get("summary")
            or ""
        )
    if isinstance(summary_data, str):
        return summary_data
    return ""


def _summary_has_useful_content(result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    text = _extract_primary_summary_text(result)
    if not text:
        return False, "Summary text was empty."
    lowered = text.strip().lower()
    for prefix in _SUMMARY_PLACEHOLDER_PREFIXES:
        if lowered.startswith(prefix):
            return False, text
    return True, None


async def _handle_summary_placeholder(
    handler,
    query,
    summary_type: str,
    proficiency_level: Optional[str],
    user_name: str,
    provider_key: str,
    provider_label: str,
    detail: Optional[str],
) -> None:
    detail_text = ""
    if detail:
        trimmed = detail.strip()
        if len(trimmed) > 200:
            trimmed = trimmed[:197] + "…"
        detail_text = f"\n• Detail: {trimmed}"

    message_lines = [
        f"❌ {provider_label} could not generate a usable summary.",
        "No report or image was saved." + detail_text,
        "Pick another engine below or try again later.",
    ]

    reply_markup = None
    if query.message:
        provider_options = handler._summary_provider_options()
        if provider_options:
            chat_id = query.message.chat.id
            message_id = query.message.message_id
            session_payload = {
                "summary_type": summary_type,
                "proficiency_level": proficiency_level,
                "user_name": user_name,
                "provider_options": provider_options,
            }
            handler._store_summary_session(chat_id, message_id, session_payload)
            buttons: List[List[InlineKeyboardButton]] = []
            for key, option in provider_options.items():
                button_label = option.get("button_label") or option.get("label") or key.title()
                buttons.append(
                    [
                        InlineKeyboardButton(
                            button_label,
                            callback_data=f"summary_provider:{key}",
                        )
                    ]
                )
            buttons.append([InlineKeyboardButton("⬅️ Back", callback_data=handler._build_summary_callback("back_to_main"))])
            reply_markup = InlineKeyboardMarkup(buttons)

    await query.edit_message_text("\n".join(message_lines), reply_markup=reply_markup)


def fetch_report_from_dashboard(video_id: str, timeout: int = 8) -> Optional[Dict[str, Any]]:
    base = get_dashboard_base()
    if not base or requests is None:
        return None
    try:
        url = f"{base.rstrip('/')}/api/reports/{video_id}"
        headers: Dict[str, str] = {}
        token = os.getenv('DASHBOARD_TOKEN')
        if token:
            headers['Authorization'] = f"Bearer {token}"
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except Exception as exc:  # pragma: no cover - network call
        logging.debug("Dashboard fetch failed for %s: %s", video_id, exc)
    return None


def load_local_report(video_id: str, reports_dir: Path = LOCAL_REPORTS_DIR) -> Optional[Dict[str, Any]]:
    if not reports_dir.exists():
        return None
    for path in sorted(reports_dir.glob(f"*{video_id}*.json")):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
    return None


def extract_variant_text(report: Dict[str, Any], variant: str) -> Optional[str]:
    target = normalize_variant_id(variant)
    if not (report and target):
        return None

    summary_block = report.get('summary') or {}

    variants_list = summary_block.get('variants')
    if isinstance(variants_list, list):
        for entry in variants_list:
            if isinstance(entry, dict):
                current = normalize_variant_id(entry.get('variant') or entry.get('summary_type') or entry.get('type'))
                if current == target:
                    text = entry.get('text') or entry.get('summary') or entry.get('content')
                    if isinstance(text, str) and text.strip():
                        return text

    summary_type = normalize_variant_id(summary_block.get('summary_type') or summary_block.get('type'))
    if summary_type == target:
        text = summary_block.get('summary') or summary_block.get('text')
        if isinstance(text, str) and text.strip():
            return text

    ingest_variants = report.get('summary_variants')
    if isinstance(ingest_variants, list):
        for entry in ingest_variants:
            if isinstance(entry, dict):
                current = normalize_variant_id(entry.get('variant') or entry.get('summary_type') or entry.get('type'))
                if current == target:
                    text = entry.get('text')
                    if isinstance(text, str) and text.strip():
                        return text

    return None


def resolve_summary_text(video_id: str, variant: str) -> Optional[str]:
    report = load_local_report(video_id) or {}
    text = extract_variant_text(report, variant)
    if isinstance(text, str) and text.strip():
        return text

    report = fetch_report_from_dashboard(video_id) or {}
    text = extract_variant_text(report, variant)
    if isinstance(text, str) and text.strip():
        return text

    return None


def slugify(text: str) -> str:
    text = (text or '').strip().lower()
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^a-z0-9\-\_\s]+', '', text)
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text or 'quiz'


def build_quiz_prompt(
    *,
    title: str,
    keypoints: str,
    count: int = 10,
    types: Optional[List[str]] = None,
    difficulty: str = 'beginner',
    language: str = 'en',
    explanations: bool = True,
) -> str:
    allowed = types or ["multiplechoice", "truefalse"]
    type_list = ', '.join(allowed)
    explain_rule = 'Provide a brief explanation (1–2 sentences) for each item.' if explanations else 'Do not include explanations.'
    return (
        f"Create {count} quiz questions in {language}.\n"
        f"Topic: {title}\n\n"
        f"Use ONLY these key points (no outside facts):\n{keypoints}\n\n"
        f"Rules:\n"
        f"- Allowed types: {type_list}\n"
        f"- Only one unambiguous correct answer per question\n"
        f"- {explain_rule}\n"
        f"- Respond ONLY as a JSON object with this shape (no text outside JSON):\n"
        '{"count": number, "meta": {"topic": string, "difficulty": "beginner|intermediate|advanced", "language": '
        + f'"{language}"' + '}, '
        '"items": [{"question": string, "type": "multiplechoice|truefalse|yesno|shortanswer", '
        '"options": [string, ...] (omit for shortanswer), "correct": number (omit for shortanswer), '
        '"answer": string (only for shortanswer), "explanation": string}]}'
    )


def validate_quiz_payload(data: dict, explanations: bool = True) -> bool:
    try:
        if not isinstance(data, dict):
            return False
        items = data.get('items')
        if not isinstance(items, list) or not items:
            return False
        try:
            data['count'] = int(data.get('count') or len(items))
        except Exception:
            data['count'] = len(items)

        alias_map = {
            'multiplechoice': 'multiplechoice',
            'multiple-choice': 'multiplechoice',
            'multiple choice': 'multiplechoice',
            'mcq': 'multiplechoice',
            'truefalse': 'truefalse',
            'true/false': 'truefalse',
            'true false': 'truefalse',
            'boolean': 'truefalse',
            'yesno': 'yesno',
            'yes/no': 'yesno',
            'yes no': 'yesno',
            'shortanswer': 'shortanswer',
            'short answer': 'shortanswer',
        }

        def norm_type(raw: str) -> str:
            key = re.sub(r'[^a-z/ ]+', '', (raw or '').strip().lower())
            return alias_map.get(key, key)

        seen = set()
        normalized_items = []
        for q in items:
            if not isinstance(q, dict):
                return False
            qtext = re.sub(r'\s+', ' ', (q.get('question') or '').strip())
            if not qtext:
                return False
            qnorm = qtext.lower()
            if qnorm in seen:
                continue
            seen.add(qnorm)

            qtype = norm_type(q.get('type'))
            if qtype in ('multiplechoice', 'truefalse', 'yesno'):
                opts = q.get('options')
                if qtype == 'truefalse' and not isinstance(opts, list):
                    opts = ["True", "False"]
                    q['options'] = opts
                if qtype == 'yesno' and not isinstance(opts, list):
                    opts = ["Yes", "No"]
                    q['options'] = opts
                if not isinstance(opts, list):
                    return False
                min_opts = 3 if qtype == 'multiplechoice' else 2
                if len(opts) < min_opts:
                    return False
                ci = q.get('correct')
                if not isinstance(ci, int) or ci < 0 or ci >= len(opts):
                    return False
            elif qtype == 'shortanswer':
                ans = q.get('answer')
                if not isinstance(ans, str) or not ans.strip():
                    return False
            else:
                return False

            if explanations and not isinstance(q.get('explanation'), str):
                q['explanation'] = q.get('explanation') or ""

            q['type'] = qtype
            q['question'] = qtext
            normalized_items.append(q)

        if not normalized_items:
            return False
        data['items'] = normalized_items
        data['count'] = len(normalized_items)
        return True
    except Exception as exc:
        logging.warning("Quiz validation error: %s", exc)
        return False


async def send_formatted_response(handler, query, result: Dict[str, Any], summary_type: str, export_info: Optional[Dict] = None, force_new_message: bool = False):
    sent_msg = None
    try:
        video_info = result.get('metadata', {})
        source = result.get('content_source') or handler._current_source()
        title = result.get('title') or video_info.get('title') or 'Untitled content'
        channel = (
            video_info.get('uploader')
            or video_info.get('channel')
            or video_info.get('author')
            or video_info.get('subreddit')
            or 'Unknown source'
        )
        duration_info = _format_duration_and_savings(video_info)

        universal_id = result.get('id') or video_info.get('video_id') or handler._current_content_id() or ''
        video_id = handler._normalize_content_id(universal_id)
        base_type = (summary_type or '').split(':', 1)[0]

        summary = extract_summary_text_for_variant(result, summary_type) or 'No summary available'

        source_icon = {
            'youtube': '🎬',
            'reddit': '🧵',
            'web': '📰',
        }.get(source, '🎬')
        channel_icon = '👤' if source == 'reddit' else '📺'
        # Augment header with generated-at timestamp and model label (when available)
        gen_line = None
        try:
            gen_at = (export_info or {}).get('generated_at') if export_info else None
            prov = (export_info or {}).get('llm_provider') if export_info else None
            model = (export_info or {}).get('llm_model') if export_info else None
            # Fallback to current summarizer if export_info missing
            if not prov or not model:
                try:
                    prov = prov or getattr(handler.summarizer, 'llm_provider', None)
                    model = model or getattr(handler.summarizer, 'model', None)
                except Exception:
                    pass
            if not gen_at:
                # Localize timestamp per SUMMARY_TIMEZONE
                tz_name = os.getenv('SUMMARY_TIMEZONE', 'America/Denver')
                try:
                    if ZoneInfo:
                        gen_local = datetime.now(timezone.utc).astimezone(ZoneInfo(tz_name))
                        gen_at = gen_local.strftime('%Y-%m-%d %H:%M:%S %Z')
                    else:
                        gen_at = datetime.now().isoformat(timespec='seconds')
                except Exception:
                    gen_at = datetime.now().isoformat(timespec='seconds')
            if prov or model:
                prov_label = handler._friendly_llm_provider(prov) if hasattr(handler, '_friendly_llm_provider') else (prov or '')
                short_model = model.split('/',1)[1] if isinstance(model,str) and '/' in model else model
                model_label = f"{prov_label} • {short_model}".strip(" •")
            else:
                model_label = None
            if model_label:
                gen_line = f"🕒 {gen_at} • Model: {model_label}"
            else:
                gen_line = f"🕒 {gen_at}"
        except Exception:
            gen_line = None

        extraction_line = None
        try:
            if source == "web":
                web_meta = (result.get("source_metadata") or {}).get("web") or {}
                notes = web_meta.get("extractor_notes") or {}
                if isinstance(notes, dict):
                    method = str(notes.get("final_method") or "").strip().lower()
                else:
                    method = ""
                show_cost = str(os.getenv("WEB_URL_CONTEXT_SHOW_COST", "0") or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                show_all = str(os.getenv("WEB_EXTRACT_SHOW_METHOD", "0") or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                if show_cost:
                    show_all = True

                if method and (show_all or method in ("url_context", "playwright")):
                    labels = {
                        "url_context": "Gemini URL context",
                        "playwright": "Dynamic render",
                        "readability": "Readability",
                        "trafilatura": "Trafilatura",
                        "static": "Static HTML",
                    }
                    label = labels.get(method, method)
                    extra_bits = []
                    if isinstance(notes, dict) and show_cost:
                        urlctx_model = str(notes.get("url_context_model") or "").strip()
                        urlctx_cost = str(notes.get("url_context_est_cost_usd") or "").strip()
                        urlctx_calls_per_usd = str(notes.get("url_context_est_calls_per_usd") or "").strip()

                        urlctx_bits = []
                        if urlctx_model:
                            urlctx_bits.append(urlctx_model)
                        if urlctx_cost:
                            try:
                                cost_f = float(urlctx_cost)
                                cost_fmt = f"${cost_f:.6f}".rstrip("0").rstrip(".")
                            except Exception:
                                cost_fmt = f"${urlctx_cost}"
                            if urlctx_calls_per_usd.isdigit():
                                urlctx_bits.append(f"{cost_fmt} (~{urlctx_calls_per_usd}/$1)")
                            else:
                                urlctx_bits.append(cost_fmt)

                        if urlctx_bits:
                            prefix = "URLCtx" if method != "url_context" else ""
                            if prefix:
                                extra_bits.append(f"{prefix}: {' • '.join(urlctx_bits)}")
                            else:
                                extra_bits.extend(urlctx_bits)

                    suffix = f" ({' • '.join(extra_bits)})" if extra_bits else ""
                    extraction_line = f"🔎 Extraction: {label}{suffix}"
        except Exception:
            extraction_line = None

        header_parts = [
            f"{source_icon} **{handler._escape_markdown(title)}**",
            f"{channel_icon} {handler._escape_markdown(channel)}",
            duration_info,
            gen_line or "",
            extraction_line or "",
            "",
            f"📝 **{summary_type.replace('-', ' ').title()} Summary:**"
        ]

        header_text = "\n".join(part for part in header_parts if part)

        reply_markup = None
        if export_info and (export_info.get('html_path') or export_info.get('json_path')):
            dashboard_url = (
                os.getenv('DASHBOARD_URL')
                or os.getenv('POSTGRES_DASHBOARD_URL')
                or 'https://ytv2-dashboard-postgres.onrender.com'
            )

            report_id = None
            if export_info.get('json_path'):
                json_path = Path(export_info['json_path'])
                report_id = json_path.stem
            elif export_info.get('html_path'):
                html_path = Path(export_info['html_path'])
                report_id = html_path.stem

            if dashboard_url:
                follow_up_state = "pending" if _should_show_follow_up_button(handler, summary_type) else None
                reply_markup = _build_summary_result_keyboard(
                    handler,
                    dashboard_url=dashboard_url,
                    report_id=report_id or "",
                    video_id=video_id,
                    base_type=base_type,
                    add_variant_content_id=handler._summary_menu_content_id(),
                    follow_up_state=follow_up_state,
                )
            else:
                logging.warning("⚠️ No DASHBOARD_URL set - skipping link buttons")

        sent_msg = await handler._send_long_message(query, header_text, summary, reply_markup, force_new_message=force_new_message)

        image_meta = result.get("summary_image") or {}
        image_path_value = (
            image_meta.get("path")
            or image_meta.get("relative_path")
        )
        image_path: Optional[Path] = None
        if image_path_value:
            candidate = Path(image_path_value)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.exists():
                image_path = candidate

        if image_path:
            try:
                # Status update: illustration phase (use replies so we do not overwrite the summary text)
                status_msg = None
                try:
                    status_msg = await query.message.reply_text("🎨 Generating illustration…")
                except Exception:
                    status_msg = None
                title_text = result.get("metadata", {}).get("title") or "Summary illustration"
                caption = f"🎨 *Summary Illustration*\n{handler._escape_markdown(title_text)}"
                with image_path.open("rb") as photo_file:
                    await query.message.reply_photo(
                        photo=photo_file,
                        caption=caption,
                        parse_mode=ParseMode.MARKDOWN,
                    )
                if status_msg:
                    try:
                        await status_msg.edit_text("🖼️ Illustration attached.")
                    except Exception:
                        pass
            except Exception as exc:
                logging.debug("Failed to send summary image to Telegram: %s", exc)

        # TTS handoff occurs after this try/except to avoid duplicate starts if UI raises

    except Exception as exc:
        logging.error("Error sending formatted response: %s", exc)
        # Fallback UI message: always reply so we don't overwrite any existing summary content
        try:
            await query.message.reply_text(
                "❌ Error formatting response. The summary was generated but couldn't be displayed properly.")
        except Exception as e2:
            logging.debug("Secondary error while reporting formatting error: %s", e2)

        # (No TTS handoff here; see below)

    # Start TTS once, regardless of UI success/failure
    try:
        if isinstance(summary_type, str) and summary_type.startswith("audio"):
            await handler._prepare_tts_generation(query, result, summary, summary_type)
    except Exception as e3:
        logging.debug("TTS handoff error after formatted response: %s", e3)
    return sent_msg


async def prepare_tts_generation(handler, query, result: Dict[str, Any], summary_text: str, summary_type: str) -> None:
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
    # If a preselected TTS choice exists (from combo/early selection), auto-run TTS
    try:
        chat_id = query.message.chat.id if query.message else None
        message_id = query.message.message_id if query.message else None
        preselected = handler._get_tts_session(chat_id, message_id) if (chat_id and message_id) else None
        # If a message-anchored preselect exists, promote it to content-anchored immediately
        try:
            if isinstance(preselected, dict) and preselected.get('auto_run') and normalized_video_id and hasattr(handler, '_store_content_tts_preselect'):
                handler._store_content_tts_preselect(normalized_video_id, preselected)
                logging.info("[TTS-PREP] promoted message preselect to content preselect for %s", normalized_video_id)
        except Exception:
            pass
        # Fallback: find any auto-run TTS session anchored to this chat that matches the summary type
        if not preselected and chat_id is not None:
            try:
                base_target = (summary_type or '').split(':', 1)[0]
                for (c_id, m_id), sess in list(getattr(handler, 'tts_sessions', {}).items()):
                    if c_id != chat_id or not isinstance(sess, dict):
                        continue
                    if not sess.get('auto_run'):
                        continue
                    st = (sess.get('summary_type') or '')
                    st_base = st.split(':', 1)[0] if isinstance(st, str) else ''
                    if not st or st == summary_type or st_base == base_target:
                        preselected = sess
                        # Also promote to content-anchored for robustness
                        try:
                            if normalized_video_id and hasattr(handler, '_store_content_tts_preselect'):
                                handler._store_content_tts_preselect(normalized_video_id, preselected)
                        except Exception:
                            pass
                        break
            except Exception:
                pass
    except Exception:
        preselected = None

    provider = None
    logging.info("[TTS-PREP] video_id=%s ledger_id=%s", normalized_video_id, ledger_id)
    # Attempt message-anchored preselect first
    # Attempt message-anchored preselect first
    if not (isinstance(preselected, dict) and preselected.get('auto_run')):
        # Fallback to content-anchored preselect (normalized video id)
        try:
            if normalized_video_id and hasattr(handler, '_get_content_tts_preselect'):
                preselected = handler._get_content_tts_preselect(normalized_video_id)
                logging.info("[TTS-PREP] content preselect found=%s", bool(preselected))
        except Exception:
            pass
    # One-line diagnostic of preselect content before branching
    try:
        if isinstance(preselected, dict):
            logging.info(
                "[TTS-PREP] preselected keys=%s auto_run=%s provider=%s",
                list(preselected.keys()),
                preselected.get('auto_run'),
                preselected.get('provider'),
            )
        else:
            logging.info("[TTS-PREP] preselected is not a dict (type=%s)", type(preselected).__name__)
    except Exception:
        pass

    if isinstance(preselected, dict) and preselected.get('auto_run'):
        provider = (preselected.get('provider') or '').strip().lower() or None
        sel = preselected.get('selected_voice') or {}
        if sel:
            session_payload['selected_voice'] = sel
        last_voice = preselected.get('last_voice')
        if last_voice:
            session_payload['last_voice'] = last_voice

    # Force prompt flow (match main's reliable callback path)
    try:
        logging.info("[TTS-PREP] Forcing prompt for provider/voice to match main flow")
    except Exception:
        pass
    provider = None

    processor_info = result.get('processor_info') or {}
    proc_provider = processor_info.get('llm_provider') or ''
    proc_model = processor_info.get('model') or ''
    proc_label = proc_provider or 'unknown'
    if proc_model:
        proc_label = f"{proc_provider}/{proc_model}" if proc_provider else proc_model

    if provider:
        provider_label = "Local TTS hub" if provider == "local" else "OpenAI TTS"
        voice_hint = ""
        selected_voice = session_payload.get('selected_voice') or {}
        last_voice = session_payload.get('last_voice')
        if last_voice:
            voice_hint = f" • {handler._escape_markdown(last_voice)}"
        else:
            eng = selected_voice.get('engine')
            fav = selected_voice.get('favorite_slug')
            voice_id = selected_voice.get('voice_id')
            if eng and fav:
                voice_hint_value = handler._escape_markdown(f"{eng}:{fav}")
                voice_hint = f" • {voice_hint_value}"
            elif voice_id:
                voice_hint = f" • {handler._escape_markdown(voice_id)}"

        status_lines = [
            "🎙️ Summary ready.",
            f"LLM: {handler._escape_markdown(proc_label)}",
            f"Starting text-to-speech • {provider_label}{voice_hint}",
        ]
        try:
            # Post as a new message so we never overwrite the summary
            await query.message.reply_text("\n".join(status_lines), parse_mode=ParseMode.MARKDOWN)
        except Exception as exc:
            logging.debug("Unable to post TTS status message: %s", exc)

        try:
            # Minimal diagnostic: log provider + summary text length before TTS
            logging.info("[TTS-START] provider=%s text_len=%d", provider, len((summary_text or '')))            
            await handler._execute_tts_job(query, session_payload, provider)
        finally:
            try:
                # Consume the one-shot message-anchored preselection
                handler._remove_tts_session(chat_id, message_id)
            except Exception:
                pass
            # Do NOT remove content-anchored preselect here; defer until audio delivery succeeds
    else:
        try:
            logging.info("[TTS-PREP] No auto-run preselect found; prompting provider/voice")
        except Exception:
            pass
        await handler._prompt_tts_provider(query, session_payload, title)


async def handle_audio_summary(handler, query, result: Dict[str, Any], summary_type: str) -> None:
    try:
        video_info = result.get('metadata', {})
        title = video_info.get('title', 'Unknown Title')
        channel = video_info.get('uploader') or video_info.get('channel') or 'Unknown Channel'

        summary_data = result.get('summary', {})
        summary = 'No summary available'

        if isinstance(summary_data, dict):
            summary = (
                summary_data.get('audio')
                or summary_data.get('content', {}).get('audio')
                or summary_data.get('comprehensive')
                or summary_data.get('content', {}).get('comprehensive')
                or summary_data.get('summary')
                or 'No audio summary available'
            )
        elif isinstance(summary_data, str):
            summary = summary_data

        await query.edit_message_text("🎙️ Generating audio summary... Creating TTS audio file.")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_id = video_info.get('video_id', 'unknown')
        audio_filename = f"audio_{video_id}_{timestamp}.mp3"

        audio_filepath = await handler.summarizer.generate_tts_audio(summary, audio_filename)

        if audio_filepath and Path(audio_filepath).exists():
            try:
                with open(audio_filepath, 'rb') as audio_file:
                    await query.message.reply_voice(
                        voice=audio_file,
                        caption=(
                            f"🎧 **Audio Summary**: {handler._escape_markdown(title)}\n"
                            f"📺 **Channel**: {handler._escape_markdown(channel)}\n\n"
                            f"🎵 Generated with OpenAI TTS"
                        ),
                        parse_mode=ParseMode.MARKDOWN
                    )

                text_summary = summary
                if len(text_summary) > 1000:
                    text_summary = text_summary[:1000] + "..."

                response_text = (
                    "🎙️ **Audio Summary Generated**\n\n"
                    f"📝 **Text Version:**\n{text_summary}\n\n"
                    "✅ Voice message sent above!"
                )

                await query.edit_message_text(
                    response_text,
                    parse_mode=ParseMode.MARKDOWN
                )

                logging.info("✅ Successfully sent audio summary for: %s", title)

            except Exception as exc:
                logging.error("❌ Failed to send voice message: %s", exc)
                summary_text = str(summary) if summary else "No summary available"
                await query.edit_message_text(
                    "❌ Generated audio but failed to send voice message.\n\n"
                    f"**Text Summary:**\n{summary_text[:1000]}{'...' if len(summary_text) > 1000 else ''}"
                )
        else:
            logging.warning("⚠️ TTS generation failed, sending text only")
            metrics.record_tts(False)
            summary_text = str(summary) if summary else "No summary available"
            response_text = (
                "🎙️ **Audio Summary** (TTS failed)\n\n"
                f"🎬 **{handler._escape_markdown(title)}**\n"
                f"📺 **Channel**: {handler._escape_markdown(channel)}\n\n"
                f"📝 **Summary:**\n{summary_text[:1000]}{'...' if len(summary_text) > 1000 else ''}\n\n"
                "⚠️ Audio generation failed. Check TTS configuration."
            )

            await query.edit_message_text(
                response_text,
                parse_mode=ParseMode.MARKDOWN
            )

    except Exception as exc:
        logging.error("Error handling audio summary: %s", exc)
        await query.edit_message_text(f"❌ Error generating audio summary: {str(exc)[:100]}...")


async def send_existing_summary_notice(handler, query, video_id: str, summary_type: str) -> None:
    variants = handler._summary_existing_variants()
    message_lines = [
        f"✅ {handler._friendly_variant_label(summary_type)} is already on the dashboard."
    ]
    if variants:
        message_lines.append("\nAvailable variants:")
        message_lines.extend(f"• {handler._friendly_variant_label(variant)}" for variant in sorted(variants))
    message_lines.append("\nOpen the summary or re-run a variant below.")

    menu_content_id = handler._summary_menu_content_id() or video_id
    reply_markup = handler._build_summary_keyboard(
        variants,
        menu_content_id,
        is_reddit_link_post=handler._is_reddit_link_post_context(),
    )
    await query.edit_message_text(
        "\n".join(message_lines),
        reply_markup=reply_markup
    )


async def generate_tts_audio_file(handler, summary_text: str, video_id: str, json_path: Path) -> Optional[str]:
    if not summary_text:
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_filename = f"audio_{video_id}_{timestamp}.mp3"

    loop = asyncio.get_running_loop()

    def _run() -> Optional[str]:
        return asyncio.run(handler.summarizer.generate_tts_audio(summary_text, audio_filename, str(json_path)))

    audio_filepath = await loop.run_in_executor(None, _run)
    if audio_filepath and Path(audio_filepath).exists():
        metrics.record_tts(True)
    else:
        metrics.record_tts(False)
    return audio_filepath if audio_filepath else None


async def process_content_summary(
    handler,
    query,
    summary_type: str,
    user_name: str,
    proficiency_level: Optional[str] = None,
    *,
    provider_key: str = "cloud",
    summarizer=None,
    provider_label: Optional[str] = None,
) -> None:
    export_info: Dict[str, Any] = {"html_path": None, "json_path": None}
    item = handler.current_item or {}
    content_id = item.get("content_id")
    source = item.get("source", "youtube")
    url = item.get("url")

    if not content_id:
        await query.edit_message_text("❌ No content in context. Please send a link first.")
        return

    if not url:
        url = handler._resolve_video_url(content_id)

    if not url:
        await query.edit_message_text("❌ Could not resolve the source URL. Please resend the link.")
        return

    # Optional: post resolved preview for shortlinks like flip.it so Telegram shows a rich preview
    try:
        show_preview = os.getenv('TELEGRAM_SHOW_RESOLVED_PREVIEW', '0').lower() in ('1','true','yes')
        if show_preview and isinstance(url, str):
            parsed = urllib.parse.urlparse(url)
            host = (parsed.netloc or '').lower()
            if host.endswith('flip.it'):
                try:
                    resolved = handler._resolve_redirects(url)
                    if isinstance(resolved, str) and resolved and resolved != url:
                        # Send a one-liner with the resolved URL to trigger Telegram's preview
                        try:
                            await query.message.reply_text(resolved, disable_web_page_preview=False)
                        except Exception:
                            pass
                except Exception:
                    pass
    except Exception:
        pass

    summarizer = summarizer or handler.summarizer
    if not summarizer:
        await query.edit_message_text("❌ Summarizer not available. Please try /status for more info.")
        return
    provider_key = (provider_key or "cloud").lower()

    noun_map = {
        "youtube": "video",
        "reddit": "thread",
        "web": "article",
    }
    noun = noun_map.get(source, "item")
    processing_messages = {
        "comprehensive": f"📝 Analyzing {noun} and creating comprehensive summary...",
        "bullet-points": f"🎯 Extracting key points from the {noun}...",
        "key-insights": f"💡 Identifying key insights and takeaways from the {noun}...",
        "reddit-discussion": "💬 Analyzing the Reddit discussion and extracting the strongest reactions...",
        "audio": "🎙️ Creating audio summary with text-to-speech...",
        "audio-fr": "🇫🇷 Translating to French and preparing audio narration...",
        "audio-es": "🇪🇸 Translating to Spanish and preparing audio narration..."
    }

    base_type = summary_type.split(':')[0]
    if base_type.startswith("audio-fr"):
        level_suffix = " (with vocabulary help)" if proficiency_level in ["beginner", "intermediate"] else ""
        message = f"🇫🇷 Creating French audio summary{level_suffix}... This may take a moment."
    elif base_type.startswith("audio-es"):
        level_suffix = " (with vocabulary help)" if proficiency_level in ["beginner", "intermediate"] else ""
        message = f"🇪🇸 Creating Spanish audio summary{level_suffix}... This may take a moment."
    else:
        prefix_map = {
            "youtube": "🔄",
            "reddit": "🧵",
            "web": "📰",
        }
        default_prefix = prefix_map.get(source, "🔄")
        message = processing_messages.get(base_type, f"{default_prefix} Processing {summary_type}... This may take a moment.")

    # Enrich the status with chosen LLM; do not pre-announce TTS here to avoid duplicate lines
    try:
        chat_id = query.message.chat.id if query.message else None
        message_id = query.message.message_id if query.message else None
        preselected = handler._get_tts_session(chat_id, message_id) if (chat_id and message_id) else None
    except Exception:
        preselected = None

    # LLM line
    llm_provider = getattr(summarizer, "llm_provider", "") or provider_key
    llm_model = getattr(summarizer, "model", "") or ''
    short_model = llm_model.split("/", 1)[1] if isinstance(llm_model, str) and "/" in llm_model else llm_model
    llm_label = f"{(llm_provider or '').title()} • {short_model}".strip(" •")
    extra_lines = []
    if llm_label:
        extra_lines.append(f"LLM: {llm_label}")
    # TTS line (audio only) — show only if a preselect exists, but avoid 'Starting…' here
    if base_type.startswith("audio") and isinstance(preselected, dict):
        tts_provider = (preselected.get('provider') or '').lower()
        sel = preselected.get('selected_voice') or {}
        if tts_provider == 'openai':
            voice = (sel.get('voice_id') or os.getenv('TTS_CLOUD_VOICE') or 'fable')
            extra_lines.append(f"TTS: OpenAI • {voice}")
        elif tts_provider == 'local':
            eng = (sel.get('engine') or '').strip()
            fav = (sel.get('favorite_slug') or '').strip()
            extra_lines.append(f"TTS: {('Local • ' + eng + ':' + fav) if (eng and fav) else 'Local TTS'}")
    if extra_lines:
        message = f"{message}\n\n" + " • ".join(extra_lines)

    await _safe_edit_status(query, message)

    # Periodic status updates during summary phase (until export)
    progress_task = None
    try:
        import asyncio as _asyncio
        start_ts = time.monotonic()
        try:
            interval = int(os.getenv('SUMMARY_STATUS_INTERVAL', '10') or '10')
        except Exception:
            interval = 10
        async def _progress_updater():
            idx = 0
            symbols = ['🔄', '⏳', '⌛']
            while True:
                await _asyncio.sleep(interval)
                elapsed = int(time.monotonic() - start_ts)
                sym = symbols[idx % len(symbols)]
                idx += 1
                status_lines = [f"{sym} Analyzing content and drafting summary… ({elapsed}s)"]
                try:
                    if llm_label:
                        status_lines.append(f"LLM: {handler._escape_markdown(llm_label)}")
                except Exception:
                    pass
                try:
                    await _safe_edit_status(query, "\n".join(status_lines))
                except Exception:
                    break
        progress_task = _asyncio.create_task(_progress_updater())
    except Exception:
        progress_task = None

    try:
        normalized_id = handler._normalize_content_id(content_id)
        video_id = normalized_id
        ledger_id = content_id
        display_id = f"{source}:{normalized_id}" if source != "youtube" else normalized_id

        entry = ledger.get(ledger_id, summary_type)
        if entry:
            logging.info(f"📄 Found existing entry for {display_id}:{summary_type}")
            if entry.get("synced"):
                await send_existing_summary_notice(handler, query, ledger_id, summary_type)
                logging.info(f"♻️ SKIPPED: {display_id} already synced")
                return
            else:
                logging.info("🔄 Content exists but not marked synced - processing fresh")

        logging.info("🎬 PROCESSING: %s | %s | user: %s | URL: %s", display_id, summary_type, user_name, url)
        # Hook summarizer status to Telegram status updates
        try:
            import asyncio as _asyncio
            def _cb(msg: str) -> None:
                try:
                    _asyncio.get_running_loop().create_task(_safe_edit_status(query, msg))
                except Exception:
                    pass
            try:
                setattr(summarizer, 'status_callback', _cb)
            except Exception:
                pass
        except Exception:
            pass
        llm_provider = getattr(summarizer, "llm_provider", "unknown")
        llm_model = getattr(summarizer, "model", "unknown")
        llm_label = provider_label or f"{llm_provider}/{llm_model}"
        logging.info("🧠 LLM: %s", llm_label)
        # Pass LLM + timestamp to render layer for header augmentation
        export_info["llm_provider"] = llm_provider
        export_info["llm_model"] = llm_model
        tz_name = os.getenv('SUMMARY_TIMEZONE', 'America/Denver')
        try:
            if ZoneInfo:
                now_local = datetime.now(timezone.utc).astimezone(ZoneInfo(tz_name))
            else:
                now_local = datetime.now()
            export_info["generated_at"] = now_local.strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception:
            export_info["generated_at"] = datetime.now().isoformat(timespec='seconds')

        # For web URLs, show an explicit extraction step up front. This avoids the UX
        # feeling "stuck" during fetch + extraction (some sites are slow/JS-heavy).
        try:
            if source == "web" and url:
                await _safe_edit_status(
                    query,
                    f"📰 Extracting article text…\nLLM: {handler._escape_markdown(llm_label)}",
                )
        except Exception:
            pass

        try:
            if source == "reddit":
                result = await summarizer.process_reddit_thread(
                    url,
                    summary_type=summary_type,
                    proficiency_level=proficiency_level
                )
            elif source == "web":
                result = await summarizer.process_web_page(
                    url,
                    summary_type=summary_type,
                    proficiency_level=proficiency_level
                )
            else:
                result = await summarizer.process_video(
                    url,
                    summary_type=summary_type,
                    proficiency_level=proficiency_level
                )
        except Exception as exc:
            if provider_key == "ollama" and _is_local_summary_unavailable(exc):
                await _handle_local_summary_unavailable(
                    handler,
                    query,
                    summary_type,
                    proficiency_level,
                    user_name,
                    summarizer,
                    content_id,
                    source,
                    url,
                )
                return
            raise

        if not result:
            try:
                if progress_task:
                    progress_task.cancel()
            except Exception:
                pass
            await query.edit_message_text("❌ Failed to process content. Please check the URL and try again.")
            return

        error_message = result.get('error') if isinstance(result, dict) else None
        if error_message:
            try:
                if progress_task:
                    progress_task.cancel()
            except Exception:
                pass
            if 'No transcript available' in error_message:
                await query.edit_message_text(
                    f"⚠️ {error_message}\n\nSkipping this item to prevent empty dashboard entries."
                )
                logging.info("❌ ABORTED: %s", error_message)
            else:
                await query.edit_message_text(f"❌ {error_message}")
                logging.info("❌ Processing error: %s", error_message)
                # Ensure there's a visible message in chat even if the status message is
                # later overwritten by a background spinner or another edit.
                try:
                    lower = str(error_message).lower()
                    if query.message and (
                        "extractable text" in lower
                        or "all extraction methods" in lower
                        or "url-context" in lower
                        or "url context" in lower
                        or "too_short" in lower
                        or "readtimeout" in lower
                    ):
                        timeout_s = (os.getenv("WEB_URL_CONTEXT_TIMEOUT") or "").strip()
                        pdf_timeout_s = (os.getenv("WEB_URL_CONTEXT_PDF_TIMEOUT") or "").strip()
                        mode = (os.getenv("WEB_URL_CONTEXT_MODE") or "").strip()
                        hint_lines = [
                            "❌ Web extraction failed.",
                            f"URL: {url}",
                            "",
                            str(error_message).strip()[:900],
                        ]
                        cfg = []
                        if mode:
                            cfg.append(f"WEB_URL_CONTEXT_MODE={mode}")
                        if timeout_s:
                            cfg.append(f"WEB_URL_CONTEXT_TIMEOUT={timeout_s}")
                        if pdf_timeout_s:
                            cfg.append(f"WEB_URL_CONTEXT_PDF_TIMEOUT={pdf_timeout_s}")
                        if cfg:
                            hint_lines.append("")
                            hint_lines.append("Current settings: " + " • ".join(cfg))
                        hint_lines.append("")
                        hint_lines.append("Try: increase `WEB_URL_CONTEXT_TIMEOUT` (e.g. 60) or set `WEB_URL_CONTEXT_MODE=always`.")
                        await query.message.reply_text("\n".join(hint_lines))
                except Exception:
                    pass
            return

        current_item = handler.current_item or {}
        if (
            isinstance(result, dict)
            and source == "web"
            and current_item.get("origin_source") == "reddit"
            and current_item.get("origin_content_id")
        ):
            source_metadata = result.setdefault("source_metadata", {})
            if isinstance(source_metadata, dict):
                source_metadata.setdefault(
                    "reddit_origin",
                    {
                        "content_id": current_item.get("origin_content_id"),
                        "canonical_url": current_item.get("origin_url"),
                        "external_url": current_item.get("primary_url") or current_item.get("url"),
                    },
                )
            metadata_block = result.setdefault("metadata", {})
            if isinstance(metadata_block, dict):
                metadata_block.setdefault("origin_source", "reddit")
                metadata_block.setdefault("origin_url", current_item.get("origin_url"))
                metadata_block.setdefault("origin_content_id", current_item.get("origin_content_id"))
            result.setdefault("origin_source", "reddit")
            result.setdefault("origin_url", current_item.get("origin_url"))
            result.setdefault("origin_content_id", current_item.get("origin_content_id"))
        if (
            isinstance(result, dict)
            and summary_type == "reddit-discussion"
            and current_item.get("origin_source") == "reddit"
            and current_item.get("primary_content_id")
        ):
            grouped_content_id = current_item.get("primary_content_id")
            result["group_content_id"] = grouped_content_id
            metadata_block = result.setdefault("metadata", {})
            if isinstance(metadata_block, dict):
                metadata_block["group_content_id"] = grouped_content_id
                metadata_block.setdefault("origin_source", "reddit")
                metadata_block.setdefault("origin_url", current_item.get("origin_url"))
                metadata_block.setdefault("origin_content_id", current_item.get("origin_content_id"))
            result.setdefault("origin_source", "reddit")
            result.setdefault("origin_url", current_item.get("origin_url"))
            result.setdefault("origin_content_id", current_item.get("origin_content_id"))

        canonical_ledger_id, canonical_video_id, canonical_display_id = _resolve_content_identity(
            handler,
            source,
            content_id,
            result,
        )
        if canonical_ledger_id and canonical_ledger_id != ledger_id:
            logging.info("🔁 Canonical content ID resolved: %s → %s", ledger_id, canonical_ledger_id)
            ledger_id = canonical_ledger_id
            video_id = canonical_video_id
            display_id = canonical_display_id

        has_summary, placeholder_detail = _summary_has_useful_content(result)
        if not has_summary:
            logging.warning(
                "Summary output from %s lacked usable content (detail=%s); skipping exports.",
                llm_label,
                placeholder_detail,
            )
            await _handle_summary_placeholder(
                handler,
                query,
                summary_type,
                proficiency_level,
                user_name,
                provider_key,
                llm_label,
                placeholder_detail,
            )
            return

        if summary_type.startswith("audio"):
            status_lines = [
                "🎙️ Summary ready.",
                f"LLM: {handler._escape_markdown(llm_label)}",
                "Preparing text-to-speech…",
            ]
            status_text = "\n".join(status_lines)
            try:
                await _safe_edit_status(query, status_text)
            except Exception as exc:
                logging.debug("Unable to refresh audio status message: %s", exc)

        export_info.update({"html_path": None, "json_path": None})
        sent_summary_message = None
        try:
            # Stop periodic spinner once we reach export phase
            try:
                if progress_task:
                    progress_task.cancel()
            except Exception:
                pass
            report_dict = create_report_from_youtube_summarizer(result)
            existing_report_path = None
            existing_report = None
            preserve_existing_identity = False
            current_item = handler.current_item or {}
            grouped_content_id = result.get("group_content_id") or (result.get("metadata") or {}).get("group_content_id")
            if grouped_content_id:
                report_dict["id"] = grouped_content_id
                metadata_block = report_dict.setdefault("metadata", {})
                if isinstance(metadata_block, dict):
                    metadata_block["group_content_id"] = grouped_content_id
                    metadata_block["content_id"] = grouped_content_id
                existing_report_path, existing_report = _find_existing_report_for_content_id(handler, grouped_content_id)
                preserve_existing_identity = bool(
                    summary_type == "reddit-discussion"
                    and current_item.get("origin_source") == "reddit"
                    and existing_report
                )
                if preserve_existing_identity:
                    primary_url = current_item.get("primary_url") or current_item.get("url")
                    if primary_url:
                        report_dict["canonical_url"] = primary_url
                        report_dict["url"] = primary_url
                        if isinstance(metadata_block, dict):
                            metadata_block["url"] = primary_url
                            metadata_block["canonical_url"] = primary_url
            else:
                existing_report_path, existing_report = _find_existing_report_for_content_id(
                    handler,
                    report_dict.get("id"),
                )

            report_dict = merge_summary_variants(
                new_report=report_dict,
                requested_variant=summary_type,
                existing_report=existing_report,
            )

            if preserve_existing_identity and existing_report:
                merged_report = deepcopy(existing_report)
                merged_report["summary"] = report_dict.get("summary", {})
                merged_report["processed_at"] = report_dict.get("processed_at", merged_report.get("processed_at"))
                if report_dict.get("summary_language"):
                    merged_report["summary_language"] = report_dict["summary_language"]
                merged_source_metadata = dict(existing_report.get("source_metadata") or {})
                merged_source_metadata.update(report_dict.get("source_metadata") or {})
                merged_report["source_metadata"] = merged_source_metadata
                merged_metadata = dict(existing_report.get("metadata") or {})
                merged_metadata.update(
                    {
                        key: value
                        for key, value in (report_dict.get("metadata") or {}).items()
                        if value is not None and key in {"origin_source", "origin_url", "origin_content_id", "group_content_id"}
                    }
                )
                merged_report["metadata"] = merged_metadata
                report_dict = merged_report
            # Apply CJCLDS categorization when applicable (General Conference talks)
            try:
                report_dict = classify_and_apply_cjclds(report_dict, url)
            except Exception:
                pass
            save_filename = existing_report_path.name if existing_report_path else None
            json_path = handler.json_exporter.save_report(report_dict, filename=save_filename, overwrite=bool(save_filename))
            export_info["json_path"] = Path(json_path).name

            json_path_obj = Path(json_path)
            if json_path_obj.exists():
                logging.info("✅ Exported JSON report: %s", json_path)

                # Send summary to Telegram FIRST as a NEW message, before sync operations
                # This ensures users see the summary immediately
                sent_summary_message = await send_formatted_response(
                    handler,
                    query,
                    result,
                    summary_type,
                    export_info,
                    force_new_message=True,
                )

                # Update status to indicate processing complete
                # Sync happens in background, confirmation will appear as new message
                try:
                    await _safe_edit_status(query, "✅ Summary complete.")
                except Exception:
                    pass
            else:
                logging.warning("⚠️ JSON export returned path but file not created: %s", json_path)
                logging.warning("   This will cause dual-sync to fail!")

            stem = Path(json_path).stem
            ledger_entry = {
                "stem": stem,
                "json": str(json_path),
                "mp3": None,
                "synced": False,
                "created_at": datetime.now().isoformat()
            }

            if proficiency_level:
                ledger_entry["proficiency"] = proficiency_level
                if summary_type.startswith("audio-"):
                    lang_code = "fr" if summary_type.startswith("audio-fr") else "es"
                    ledger_entry["target_language"] = lang_code
                    ledger_entry["language_flag"] = "🇫🇷" if lang_code == "fr" else "🇪🇸"
                    ledger_entry["learning_mode"] = True

                    proficiency_badges = {
                        "beginner": {"fr": "🟢 Débutant", "es": "🟢 Principiante"},
                        "intermediate": {"fr": "🟡 Intermédiaire", "es": "🟡 Intermedio"},
                        "advanced": {"fr": "🔵 Avancé", "es": "🔵 Avanzado"}
                    }
                    if proficiency_level in proficiency_badges:
                        ledger_entry["proficiency_badge"] = proficiency_badges[proficiency_level][lang_code]

            ledger.upsert(ledger_id, summary_type, ledger_entry)
            logging.info("📊 Added to ledger: %s:%s", display_id, summary_type)

            # Ensure an image job exists: enqueue here if needed and notify
            try:
                tts_base = os.getenv('TTSHUB_API_BASE') or ''
                img_enabled = os.getenv('SUMMARY_IMAGE_ENABLED','false').lower() in ('1','true','yes','on')
                if tts_base and img_enabled:
                    from modules.services import draw_service as _ds
                    # Use meta mode for health check (/api/meta returns JSON, /drawthings/health returns HTML)
                    health = None
                    health_error = None
                    try:
                        health = await _ds.fetch_drawthings_health(tts_base, ttl=0, force_refresh=True)
                    except Exception as exc:
                        health_error = exc
                        logging.debug("summary image health probe failed: %s", exc)

                    reachable = bool((health or {}).get('reachable', False)) if health is not None else False
                    # Detect if report already has an image URL
                    has_image = False
                    try:
                        s = report_dict.get('summary') or {}
                        if isinstance(s, dict) and (s.get('summary_image') or s.get('summary_image_url')):
                            has_image = True
                        if report_dict.get('summary_image') or report_dict.get('summary_image_url'):
                            has_image = True
                    except Exception:
                        pass

                    # Always enqueue if no image exists (queue processor will generate immediately if hub is online)
                    if not has_image:
                        # Build minimal content payload and enqueue if missing
                        try:
                            from modules import image_queue as _iq
                            content_id_for_job = result.get('id') or ledger_id
                            qdir = Path('data/image_queue')
                            qdir.mkdir(parents=True, exist_ok=True)
                            if not _image_queue_has_job(content_id_for_job):
                                payload = {
                                    'id': content_id_for_job,
                                    'title': report_dict.get('title') or (report_dict.get('metadata') or {}).get('title') or '',
                                    'summary': report_dict.get('summary') or {},
                                    'analysis': report_dict.get('analysis') or {},
                                }
                                _iq.enqueue({'mode':'summary_image','content':payload,'reason':'summary_offline_enqueue'})
                                notice = f"🖼️ Queued image: {handler._escape_markdown(str(content_id_for_job))}"
                                if not reachable:
                                    notice += " (hub offline, will retry when online)"
                                try:
                                    await query.message.reply_text(notice, parse_mode=ParseMode.MARKDOWN)
                                except Exception:
                                    await _safe_edit_status(query, notice)
                        except Exception:
                            pass

                    # Additionally, if a pending file exists now (idempotent queue), surface a visible notice
                    try:
                        content_id_for_job = result.get('id') or ledger_id
                        if _image_queue_has_job(content_id_for_job):
                            notice = f"🖼️ Queued image: {handler._escape_markdown(str(content_id_for_job))}"
                            try:
                                await query.message.reply_text(notice, parse_mode=ParseMode.MARKDOWN)
                            except Exception:
                                await _safe_edit_status(query, notice)
                    except Exception:
                        pass
            except Exception as _exc:
                logging.debug("image enqueue/notify skipped: %s", _exc)

            is_audio_summary = summary_type == "audio" or summary_type.startswith("audio-fr") or summary_type.startswith("audio-es")

            # Run dual-sync in BACKGROUND to avoid blocking Telegram message delivery
            # This ensures the summary appears immediately while sync happens asynchronously
            async def _run_dual_sync_background():
                """Background task for dual-sync operations."""
                try:
                    if not is_audio_summary:
                        json_path_obj = Path(json_path)
                        stem = json_path_obj.stem
                        report_path = Path(f"/app/data/reports/{stem}.json")
                        result_content_id = result.get('id') or (ledger_id if ledger_id else stem)
                        metadata = result.get("metadata") or {} if isinstance(result, dict) else {}

                        logging.info("📡 DUAL-SYNC START: Uploading to configured targets...")
                        outcome = sync_service.run_dual_sync(report_path, label=result_content_id)
                        if outcome["success"]:
                            sync_service.update_ledger_after_sync(
                                ledger_id,
                                summary_type,
                                targets=outcome["targets"],
                            )
                            # Send as NEW message after the summary
                            try:
                                await query.message.reply_text("✅ Synced to dashboard.")
                            except Exception:
                                pass
                            try:
                                source_meta = ((result.get("source_metadata") or {}).get(source) or {}) if isinstance(result, dict) else {}
                                source_context = {
                                    "video_id": video_id,
                                    "id": video_id,
                                    "type": source,
                                    "title": result.get("title") or metadata.get("title") or "",
                                    "url": source_meta.get("canonical_url") or source_meta.get("url") or url or "",
                                    "published_at": source_meta.get("published_at") or metadata.get("upload_date"),
                                }
                                logging.info(
                                    "FOLLOW_UP_OFFER trigger video_id=%s summary_type=%s sync_targets=%s",
                                    video_id,
                                    summary_type,
                                    outcome["targets"],
                                )
                                await handler._maybe_offer_follow_up_research(
                                    query.message,
                                    video_id=video_id,
                                    summary_type=summary_type,
                                    summary=extract_summary_text_for_variant(result, summary_type),
                                    source_context=source_context,
                                    sync_targets=outcome["targets"],
                                    anchor_message=sent_summary_message,
                                )
                            except Exception as exc:
                                logging.exception(
                                    "FOLLOW_UP_OFFER failed video_id=%s summary_type=%s: %s",
                                    video_id,
                                    summary_type,
                                    exc,
                                )
                                try:
                                    await handler._mark_follow_up_anchor_unavailable(
                                        sent_summary_message,
                                        reason="offer_exception",
                                    )
                                except Exception:
                                    logging.exception(
                                        "FOLLOW_UP_OFFER cleanup_failed video_id=%s summary_type=%s",
                                        video_id,
                                        summary_type,
                                    )
                    else:
                        json_path_obj = Path(json_path)
                        stem = json_path_obj.stem
                        report_path = json_path_obj

                        logging.info("📡 DUAL-SYNC (content-only): Audio summary - syncing metadata for %s", result.get('id'))

                        max_retries = 3
                        for attempt in range(max_retries):
                            if report_path.exists():
                                break
                            logging.debug("📄 Waiting for file to be written (attempt %s/%s): %s", attempt + 1, max_retries, report_path)
                            time.sleep(0.1)

                        if report_path.exists():
                            outcome = sync_service.run_dual_sync(report_path, label=result.get('id'))
                            if outcome["success"]:
                                sync_service.update_ledger_after_sync(
                                    ledger_id,
                                    summary_type,
                                    targets=outcome["targets"],
                                )
                                logging.info("⏳ Audio sync will happen after TTS generation")
                                # Send as NEW message after the summary
                                try:
                                    await query.message.reply_text("✅ Synced metadata. ⏳ Audio will upload after TTS…")
                                except Exception:
                                    pass
                        else:
                            logging.warning("⚠️ JSON report not found for content sync: %s", report_path)
                except Exception as e:
                    logging.warning("⚠️ Background dual-sync failed: %s", e)

            # Launch background task - don't await it
            import asyncio
            asyncio.create_task(_run_dual_sync_background())

        except Exception as e:
            logging.warning("⚠️ Export failed: %s", e)

        # Send summary to Telegram if not already sent (happens early if JSON export succeeded)
        # Only send here if json_path is not set (meaning early send didn't happen)
        if not export_info.get("json_path"):
            sent_summary_message = await send_formatted_response(handler, query, result, summary_type, export_info)

    except Exception as e:
        logging.error("Error processing content %s: %s", url, e)
        try:
            if progress_task:
                progress_task.cancel()
        except Exception:
            pass

        detail = str(e).strip() if e is not None else "Unknown error"
        if not detail:
            detail = "Unknown error"

        msg_lines = ["❌ Processing failed."]
        if url:
            msg_lines.append(f"URL: {url}")
        msg_lines.append("")
        msg_lines.append(detail[:900])
        message = "\n".join(msg_lines).strip()

        try:
            await _safe_edit_status(query, message)
        except Exception:
            try:
                if query.message:
                    await query.message.reply_text(message)
            except Exception:
                pass
        else:
            # Send a standalone message for extractor-ish failures, which are otherwise easy to miss.
            try:
                lower = detail.lower()
                if query.message and ("extractable text" in lower or "url-context" in lower or "readtimeout" in lower):
                    await query.message.reply_text(message)
            except Exception:
                pass
    finally:
        # Ensure periodic updater is stopped on every exit path.
        try:
            if progress_task:
                progress_task.cancel()
        except Exception:
            pass
        # Detach status callback to avoid cross-talk with future runs
        try:
            setattr(summarizer, 'status_callback', None)
        except Exception:
            pass


async def _handle_local_summary_unavailable(
    handler,
    query,
    summary_type: str,
    proficiency_level: Optional[str],
    user_name: str,
    summarizer,
    content_id: Optional[str],
    source: str,
    url: Optional[str],
) -> None:
    chat_id = query.message.chat.id if query.message else None
    message_id = query.message.message_id if query.message else None
    job_path = None
    job_payload = {
        "created_at": datetime.utcnow().isoformat(),
        "summary_type": summary_type,
        "proficiency_level": proficiency_level,
        "content_id": content_id,
        "source": source,
        "url": url,
        "provider": "ollama",
        "model": getattr(summarizer, "model", None),
        "user": user_name,
    }
    try:
        job_path = enqueue_summary_job(job_payload)
        logging.info("📥 Queued summary job for later processing: %s", job_path.name)
    except Exception as exc:
        logging.error(f"Failed to enqueue summary job: {exc}")

    cloud_option = None
    try:
        cloud_options = handler._summary_provider_options()
        cloud_option = (cloud_options or {}).get("cloud")
    except Exception as exc:
        logging.debug(f"Unable to build cloud provider option: {exc}")

    buttons: List[List[InlineKeyboardButton]] = []
    if cloud_option and chat_id is not None and message_id is not None:
        session_payload = {
            "summary_type": summary_type,
            "proficiency_level": proficiency_level,
            "user_name": user_name,
            "content_id": content_id,
            "provider_options": {"cloud": cloud_option},
        }
        handler._store_summary_session(chat_id, message_id, session_payload)
        buttons.append([InlineKeyboardButton(cloud_option["button_label"], callback_data="summary_provider:cloud")])
    buttons.append([InlineKeyboardButton("⬅️ Back", callback_data=handler._build_summary_callback("back_to_main", content_id))])

    if job_path:
        queue_line = f"🗂️ Queued job ID: {job_path.name}"
    else:
        queue_line = "⚠️ Could not queue this request automatically."

    message = (
        "⚠️ Ollama summarizer is currently offline.\n"
        f"{queue_line}\n"
        "You can switch to the cloud summarizer now or wait for the queue to process when the local models return."
    )
    await query.edit_message_text(message, reply_markup=InlineKeyboardMarkup(buttons))


async def reprocess_single_summary(
    handler,
    video_id: str,
    video_url: str,
    summary_type: str,
    ledger_entry: Optional[Dict[str, Any]] = None,
    force: bool = False,
    regenerate_audio: bool = True,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    ledger_entry = dict(ledger_entry or {})
    proficiency = ledger_entry.get('proficiency')
    job_result: Dict[str, Any] = {
        'summary_type': summary_type,
        'status': 'pending',
    }

    # Temporarily override LLM model if specified
    original_provider = None
    original_model = None
    if llm_model and hasattr(handler, 'summarizer'):
        original_provider = getattr(handler.summarizer, 'llm_provider', None)
        original_model = getattr(handler.summarizer, 'model', None)

        # Parse llm_model (format: "provider/model" or just "model")
        if '/' in llm_model:
            parts = llm_model.split('/', 1)
            new_provider = parts[0]
            new_model = parts[1]
        else:
            # Just model name, keep current provider
            new_provider = original_provider
            new_model = llm_model

        if new_provider and new_model:
            handler.summarizer.llm_provider = new_provider
            handler.summarizer.model = new_model
            logging.info(f"Reprocess using LLM: {new_provider}/{new_model}")

    try:
        result = await handler.summarizer.process_video(
            video_url,
            summary_type=summary_type,
            proficiency_level=proficiency,
        )

        if not result or result.get('error'):
            job_result.update({'status': 'error', 'error': result.get('error') if isinstance(result, dict) else 'unknown'})
            metrics.record_reprocess_result(False)
            return job_result

        report_dict = create_report_from_youtube_summarizer(result)

        target_basename = handler.json_exporter._generate_filename(report_dict)
        candidate_name = target_basename if target_basename.endswith('.json') else f"{target_basename}.json"
        candidate_path = Path(handler.json_exporter.reports_dir) / candidate_name

        existing_report = None
        if candidate_path.exists():
            try:
                with candidate_path.open('r', encoding='utf-8') as existing_file:
                    existing_report = json.load(existing_file)
            except Exception as load_error:
                logging.warning(
                    "⚠️  Failed to load existing report for variant merge (%s): %s",
                    candidate_path,
                    load_error,
                )

        report_dict = merge_summary_variants(
            new_report=report_dict,
            requested_variant=summary_type,
            existing_report=existing_report,
        )

        json_path = Path(
            handler.json_exporter.save_report(
                report_dict,
                filename=target_basename,
                overwrite=True,
            )
        )
        job_result['report_path'] = str(json_path)

        summary_meta = report_dict.get('summary') or {}
        summary_text = summary_meta.get('summary') or ''

        audio_path = None
        is_audio = summary_type.startswith('audio')

        if is_audio:
            if regenerate_audio:
                audio_path = await generate_tts_audio_file(handler, summary_text, video_id, json_path)
            else:
                existing_mp3 = ledger_entry.get('mp3')
                if existing_mp3 and Path(existing_mp3).exists():
                    audio_path = existing_mp3
            if audio_path:
                job_result['audio_path'] = audio_path

        audio_path_obj = Path(audio_path) if audio_path else None
        sync_outcome = sync_service.run_dual_sync(
            json_path,
            audio_path_obj,
            label=f"{video_id}:{summary_type}",
        )

        targets = sync_outcome["targets"]
        job_result['sync_targets'] = targets

        success = sync_outcome["success"]
        job_result['status'] = 'ok' if success else 'error'
        if not success and sync_outcome.get("error"):
            job_result['error'] = sync_outcome["error"]
        metrics.record_reprocess_result(success)

        ledger_entry = sync_service.update_ledger_after_sync(
            video_id,
            summary_type,
            targets=targets,
            audio_path=audio_path if is_audio else None,
            extra_fields={
                'stem': json_path.stem,
                'json': str(json_path),
                'reprocessed_at': datetime.now().isoformat(),
            },
        )

        emit_report_event(
            'reprocess-complete',
            {
                'video_id': video_id,
                'summary_type': summary_type,
                'status': job_result['status'],
                'targets': targets,
            },
        )

        return job_result

    except Exception as e:
        logging.exception("Reprocess failure for %s:%s", video_id, summary_type)
        job_result.update({'status': 'error', 'error': str(e)})
        metrics.record_reprocess_result(False)
        emit_report_event(
            'reprocess-error',
            {
                'video_id': video_id,
                'summary_type': summary_type,
                'error': str(e),
            },
        )
        return job_result

    finally:
        # Restore original LLM model if we overrode it
        if original_provider and original_model and hasattr(handler, 'summarizer'):
            handler.summarizer.llm_provider = original_provider
            handler.summarizer.model = original_model


async def reprocess_video(
    handler,
    video_id: str,
    summary_types: Optional[List[str]] = None,
    force: bool = False,
    regenerate_audio: bool = True,
    video_url: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_id = handler._normalize_content_id(video_id)
    summary_types = summary_types or handler._discover_summary_types(normalized_id)

    if not summary_types:
        metrics.record_reprocess_result(False)
        emit_report_event(
            'reprocess-error',
            {
                'video_id': normalized_id,
                'error': 'no-summary-types',
            },
        )
        raise ValueError(f"No summary types found for video {normalized_id}")

    resolved_url = handler._resolve_video_url(normalized_id, provided_url=video_url)
    if not resolved_url:
        metrics.record_reprocess_result(False)
        emit_report_event(
            'reprocess-error',
            {
                'video_id': normalized_id,
                'error': 'missing-url',
            },
        )
        raise ValueError(f"Could not resolve URL for video {normalized_id}")

    metrics.record_reprocess_request(len(summary_types))
    emit_report_event(
        'reprocess-requested',
        {
            'video_id': normalized_id,
            'summary_types': summary_types,
        },
    )

    ledger_data = ledger.list_all()
    results = []
    for current_type in summary_types:
        ledger_entry = ledger_data.get(f"{normalized_id}:{current_type}")
        job_result = await reprocess_single_summary(
            handler,
            normalized_id,
            resolved_url,
            current_type,
            ledger_entry=ledger_entry,
            force=force,
            regenerate_audio=regenerate_audio,
            llm_model=llm_model,
        )
        results.append(job_result)
        if job_result.get('status') == 'ok':
            ledger_data[f"{normalized_id}:{current_type}"] = ledger.get(normalized_id, current_type)

    failures = sum(1 for r in results if r.get('status') != 'ok')

    return {
        'video_id': normalized_id,
        'summary_types': summary_types,
        'results': results,
        'failures': failures,
    }
