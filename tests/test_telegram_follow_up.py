import unittest
from unittest.mock import AsyncMock, MagicMock
import sys
import types

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

if "pydub" not in sys.modules:
    pydub_stub = types.ModuleType("pydub")
    pydub_stub.AudioSegment = object
    sys.modules["pydub"] = pydub_stub

from modules.services.summary_service import (
    _resolve_content_identity,
    extract_summary_text_for_variant,
)
from modules.telegram_handler import YouTubeTelegramBot
from modules.telegram.ui.summary import (
    build_summary_callback,
    build_summary_keyboard,
    build_summary_provider_keyboard,
)
from ytv2_api.follow_up_store import ResolvedFollowUpContext, _candidate_video_ids


class _DummyChat:
    def __init__(self, chat_id: int):
        self.id = chat_id


class _DummySentMessage:
    def __init__(self, chat_id: int, message_id: int):
        self.chat_id = chat_id
        self.message_id = message_id
        self.chat = _DummyChat(chat_id)
        self.replies = []
        self.reply_markup = None

    async def reply_text(self, text, parse_mode=None, reply_markup=None):
        self.replies.append({
            "text": text,
            "parse_mode": parse_mode,
            "reply_markup": reply_markup,
        })
        next_id = self.message_id + len(self.replies)
        sent = _DummySentMessage(self.chat_id, next_id)
        sent.reply_markup = reply_markup
        return sent

    async def edit_reply_markup(self, reply_markup=None):
        self.reply_markup = reply_markup
        return self


class _DummyQuery:
    def __init__(self, chat_id: int, message_id: int):
        self.message = _DummySentMessage(chat_id, message_id)
        self.edits = []
        self.answers = []

    async def edit_message_text(self, text, parse_mode=None, reply_markup=None):
        self.edits.append({
            "text": text,
            "parse_mode": parse_mode,
            "reply_markup": reply_markup,
        })
        return self.message

    async def answer(self, text=None, show_alert=False):
        self.answers.append({
            "text": text,
            "show_alert": show_alert,
        })


def _build_bot() -> YouTubeTelegramBot:
    bot = object.__new__(YouTubeTelegramBot)
    bot.follow_up_sessions = {}
    bot._follow_up_store = None
    bot._follow_up_service = None
    bot._follow_up_offered = set()
    return bot


class TestSummaryVariantExtraction(unittest.TestCase):
    def test_extracts_requested_variant_text(self):
        result = {
            "summary": {
                "comprehensive": "Comprehensive version",
                "bullet_points": "Bullet version",
                "key_insights": "Insights version",
            }
        }

        self.assertEqual(extract_summary_text_for_variant(result, "comprehensive"), "Comprehensive version")
        self.assertEqual(extract_summary_text_for_variant(result, "bullet-points"), "Bullet version")
        self.assertEqual(extract_summary_text_for_variant(result, "key-insights"), "Insights version")

    def test_candidate_video_ids_include_source_prefixed_alias(self):
        self.assertEqual(
            _candidate_video_ids("Z_MMxvZyOJs", "youtube"),
            ["Z_MMxvZyOJs", "yt:Z_MMxvZyOJs"],
        )

    def test_summary_callback_can_embed_content_id(self):
        self.assertEqual(
            build_summary_callback("back_to_main", "Z_MMxvZyOJs"),
            "summarize_back_to_main|Z_MMxvZyOJs",
        )

    def test_summary_keyboard_uses_contextual_callbacks(self):
        keyboard, _ = build_summary_keyboard(
            {"comprehensive": "Comprehensive"},
            existing_variants=[],
            video_id="Z_MMxvZyOJs",
        )
        self.assertEqual(
            keyboard.inline_keyboard[0][0].callback_data,
            "summarize_comprehensive|Z_MMxvZyOJs",
        )

    def test_summary_provider_keyboard_uses_contextual_back(self):
        keyboard = build_summary_provider_keyboard(
            "Cloud",
            local_label="Local",
            content_id="Z_MMxvZyOJs",
        )
        self.assertEqual(
            keyboard.inline_keyboard[-1][0].callback_data,
            "summarize_back_to_main|Z_MMxvZyOJs",
        )

    def test_resolve_content_identity_promotes_canonical_result_id(self):
        bot = _build_bot()

        ledger_id, video_id, display_id = _resolve_content_identity(
            bot,
            "web",
            "web:a29f3499ee02ccd647d50498",
            {"id": "web:ab53029474d73c67ea7bfd14"},
        )

        self.assertEqual(ledger_id, "web:ab53029474d73c67ea7bfd14")
        self.assertEqual(video_id, "ab53029474d73c67ea7bfd14")
        self.assertEqual(display_id, "web:ab53029474d73c67ea7bfd14")


class TestTelegramFollowUpFlow(unittest.IsolatedAsyncioTestCase):
    def test_restore_current_item_from_content_id(self):
        bot = _build_bot()
        bot._resolve_video_url = MagicMock(return_value="https://www.youtube.com/watch?v=Z_MMxvZyOJs")

        restored = bot._restore_current_item_from_content_id("Z_MMxvZyOJs")

        self.assertTrue(restored)
        self.assertEqual(bot.current_item["source"], "youtube")
        self.assertEqual(bot.current_item["content_id"], "Z_MMxvZyOJs")
        self.assertEqual(bot.current_item["url"], "https://www.youtube.com/watch?v=Z_MMxvZyOJs")

    async def test_offer_follow_up_uses_stored_suggestions(self):
        bot = _build_bot()
        store = MagicMock()
        store.resolve_context.return_value = ResolvedFollowUpContext(
            video_id="abc123",
            summary_id=55,
            summary="Persisted summary",
            source_context={"title": "Cursor review", "video_id": "abc123"},
        )
        store.get_stored_suggestions.return_value = [
            {
                "id": "q1",
                "label": "Latest pricing",
                "question": "What is the latest pricing?",
                "reason": "Pricing often changes.",
                "default_selected": True,
                "provenance": "suggested",
            }
        ]
        bot._get_follow_up_store = MagicMock(return_value=store)
        bot._get_follow_up_service = MagicMock(return_value={"get_follow_up_suggestions": MagicMock()})

        reply_message = _DummySentMessage(chat_id=10, message_id=20)
        await bot._maybe_offer_follow_up_research(
            reply_message,
            video_id="abc123",
            summary_type="comprehensive",
            summary="Current summary",
            source_context={"title": "Cursor review", "url": "https://youtube.com/watch?v=abc123"},
            sync_targets=["PostgreSQL"],
        )

        session = bot._get_follow_up_session(10, 21)
        self.assertIsNotNone(session)
        self.assertEqual(session["summary_id"], 55)
        self.assertEqual(session["selected_ids"], ["q1"])
        self.assertEqual(session["suggestions"][0]["label"], "Latest pricing")

    async def test_offer_follow_up_can_activate_summary_anchor(self):
        bot = _build_bot()
        store = MagicMock()
        store.resolve_context.return_value = ResolvedFollowUpContext(
            video_id="abc123",
            summary_id=55,
            summary="Persisted summary",
            source_context={"title": "Cursor review", "video_id": "abc123"},
        )
        store.get_stored_suggestions.return_value = [
            {
                "id": "q1",
                "label": "Latest pricing",
                "question": "What is the latest pricing?",
                "reason": "Pricing often changes.",
                "default_selected": True,
                "provenance": "suggested",
            }
        ]
        bot._get_follow_up_store = MagicMock(return_value=store)
        bot._get_follow_up_service = MagicMock(return_value={"get_follow_up_suggestions": MagicMock()})

        anchor_message = _DummySentMessage(chat_id=10, message_id=20)
        anchor_message.reply_markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("⏳ Deep Research Preparing", callback_data="followup:pending")]
        ])

        await bot._maybe_offer_follow_up_research(
            anchor_message,
            video_id="abc123",
            summary_type="comprehensive",
            summary="Current summary",
            source_context={"title": "Cursor review", "url": "https://youtube.com/watch?v=abc123"},
            sync_targets=["PostgreSQL"],
            anchor_message=anchor_message,
        )

        session = bot._get_follow_up_session(10, 20)
        self.assertIsNotNone(session)
        self.assertTrue(session["anchor_summary_message"])
        self.assertEqual(
            anchor_message.reply_markup.inline_keyboard[0][0].callback_data,
            "followup:open",
        )

    async def test_toggle_follow_up_selection_updates_session(self):
        bot = _build_bot()
        bot._store_follow_up_session(1, 2, {
            "stage": "selection",
            "suggestions": [
                {
                    "id": "q1",
                    "label": "Latest pricing",
                    "question": "What is the latest pricing?",
                    "reason": "Pricing shifts over time.",
                }
            ],
            "selected_ids": [],
        })
        query = _DummyQuery(chat_id=1, message_id=2)

        await bot._handle_follow_up_callback(query, "followup:toggle:0")

        session = bot._get_follow_up_session(1, 2)
        self.assertEqual(session["selected_ids"], ["q1"])
        self.assertIn("*Selected:* 1/1", query.edits[-1]["text"])

    async def test_follow_up_pending_callback_answers_without_session(self):
        bot = _build_bot()
        query = _DummyQuery(chat_id=1, message_id=2)

        await bot._handle_follow_up_callback(query, "followup:pending")

        self.assertEqual(query.answers[-1]["text"], "Deep Research will be available after sync completes.")

    async def test_follow_up_unavailable_callback_answers_without_session(self):
        bot = _build_bot()
        query = _DummyQuery(chat_id=1, message_id=2)

        await bot._handle_follow_up_callback(query, "followup:unavailable")

        self.assertEqual(query.answers[-1]["text"], "Deep Research is not available for this summary.")

    async def test_follow_up_open_from_summary_anchor_replies_below(self):
        bot = _build_bot()
        bot._store_follow_up_session(1, 2, {
            "stage": "prompt",
            "video_id": "abc123",
            "summary_id": 55,
            "summary": "Persisted summary",
            "source_context": {"title": "Cursor review", "video_id": "abc123"},
            "suggestions": [
                {
                    "id": "q1",
                    "label": "Latest pricing",
                    "question": "What is the latest pricing?",
                    "reason": "Pricing changes.",
                    "default_selected": True,
                    "provenance": "suggested",
                }
            ],
            "selected_ids": ["q1"],
            "title": "Cursor review",
            "provider_mode": "auto",
            "tool_mode": "auto",
            "depth": "balanced",
            "anchor_summary_message": True,
        })
        query = _DummyQuery(chat_id=1, message_id=2)

        await bot._handle_follow_up_callback(query, "followup:open")

        self.assertFalse(query.edits)
        self.assertEqual(query.message.replies[-1]["reply_markup"].inline_keyboard[-1][0].callback_data, "followup:run")
        self.assertEqual(query.answers[-1]["text"], "Deep Research options opened below.")

    async def test_run_follow_up_research_completes_and_clears_session(self):
        bot = _build_bot()
        bot._store_follow_up_session(3, 4, {
            "stage": "selection",
            "video_id": "abc123",
            "summary_id": 55,
            "summary": "Persisted summary",
            "source_context": {"title": "Cursor review", "video_id": "abc123"},
            "suggestions": [
                {
                    "id": "q1",
                    "label": "Latest pricing",
                    "question": "What is the latest pricing?",
                    "reason": "Pricing changes.",
                    "provenance": "suggested",
                }
            ],
            "selected_ids": ["q1"],
            "title": "Cursor review",
            "provider_mode": "auto",
            "tool_mode": "auto",
            "depth": "balanced",
        })
        bot._execute_follow_up_research_sync = MagicMock(return_value={
            "status": "ok",
            "answer": "Pricing now starts at $20/month.",
            "sources": [{"name": "Cursor", "url": "https://cursor.com/pricing"}],
            "meta": {"cache_hit": False},
        })
        bot._send_follow_up_result = AsyncMock()

        query = _DummyQuery(chat_id=3, message_id=4)
        await bot._handle_follow_up_callback(query, "followup:run")

        self.assertIsNone(bot._get_follow_up_session(3, 4))
        bot._execute_follow_up_research_sync.assert_called_once()
        call_args = bot._execute_follow_up_research_sync.call_args.args
        self.assertEqual(call_args[1], ["What is the latest pricing?"])
        self.assertEqual(call_args[2], ["suggested"])
        bot._send_follow_up_result.assert_awaited_once()
        self.assertEqual(query.edits[-1]["text"], "✅ Deep research complete.")

    async def test_offer_follow_up_marks_anchor_unavailable_when_no_suggestions(self):
        bot = _build_bot()
        store = MagicMock()
        store.resolve_context.return_value = ResolvedFollowUpContext(
            video_id="abc123",
            summary_id=55,
            summary="Persisted summary",
            source_context={"title": "Cursor review", "video_id": "abc123"},
        )
        store.get_stored_suggestions.return_value = []
        service = {"get_follow_up_suggestions": MagicMock(return_value=[])}
        bot._get_follow_up_store = MagicMock(return_value=store)
        bot._get_follow_up_service = MagicMock(return_value=service)

        anchor_message = _DummySentMessage(chat_id=10, message_id=20)
        anchor_message.reply_markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("⏳ Deep Research Preparing", callback_data="followup:pending")]
        ])

        await bot._maybe_offer_follow_up_research(
            anchor_message,
            video_id="abc123",
            summary_type="comprehensive",
            summary="Current summary",
            source_context={"title": "Cursor review", "url": "https://youtube.com/watch?v=abc123"},
            sync_targets=["PostgreSQL"],
            anchor_message=anchor_message,
        )

        self.assertIsNone(bot._get_follow_up_session(10, 20))
        self.assertEqual(
            anchor_message.reply_markup.inline_keyboard[0][0].callback_data,
            "followup:unavailable",
        )


if __name__ == "__main__":
    unittest.main()
