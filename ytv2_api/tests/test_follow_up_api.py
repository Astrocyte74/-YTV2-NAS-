"""Tests for follow-up research API endpoint behavior."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ytv2_api.follow_up_store import ResolvedFollowUpContext
from ytv2_api.models import FollowUpChatRequest, FollowUpRunRequest, FollowUpSuggestionsRequest

try:
    from fastapi import HTTPException
    from ytv2_api.main import (
        answer_follow_up_chat_endpoint,
        check_cached_research,
        get_follow_up_suggestions_endpoint,
        get_follow_up_thread_endpoint,
        run_follow_up_research_endpoint,
    )
    FASTAPI_AVAILABLE = True
except ModuleNotFoundError as exc:
    FASTAPI_AVAILABLE = False
    IMPORT_ERROR = exc


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment")
class TestFollowUpSuggestionEndpoint(unittest.TestCase):
    def test_suggestions_are_stored_when_summary_id_is_resolved(self):
        request = FollowUpSuggestionsRequest(
            video_id="abc123",
            summary="Request summary",
            source_context={"title": "Original"},
        )
        store = MagicMock()
        store.resolve_context.return_value = ResolvedFollowUpContext(
            video_id="abc123",
            summary_id=55,
            summary="Resolved summary",
            source_context={"title": "Original", "video_id": "abc123", "id": "abc123"},
        )
        service = {
            "get_follow_up_suggestions": MagicMock(return_value=[
                {
                    "id": "s1",
                    "label": "Current pricing",
                    "question": "What is the current pricing?",
                    "reason": "Pricing changes",
                    "kind": "pricing",
                    "priority": 0.9,
                    "default_selected": True,
                    "provenance": "suggested",
                }
            ]),
        }

        with patch("ytv2_api.main.get_follow_up_store", return_value=store), \
             patch("ytv2_api.main.get_research_service", return_value=service):
            response = asyncio.run(get_follow_up_suggestions_endpoint(request))

        self.assertEqual(response.summary_id, 55)
        self.assertEqual(len(response.suggestions), 1)
        store.store_suggestions.assert_called_once()
        store.mark_follow_up_available.assert_called_once_with(55)
        self.assertIsNone(store.resolve_context.call_args.kwargs.get("preferred_variant"))

    def test_suggestions_forward_preferred_variant(self):
        request = FollowUpSuggestionsRequest(
            video_id="abc123",
            summary="Request summary",
            preferred_variant="key-insights",
            source_context={"title": "Original"},
        )
        store = MagicMock()
        store.resolve_context.return_value = ResolvedFollowUpContext(
            video_id="abc123",
            summary_id=55,
            summary="Resolved summary",
            source_context={"title": "Original", "video_id": "abc123", "id": "abc123"},
        )
        service = {
            "get_follow_up_suggestions": MagicMock(return_value=[]),
        }

        with patch("ytv2_api.main.get_follow_up_store", return_value=store), \
             patch("ytv2_api.main.get_research_service", return_value=service):
            asyncio.run(get_follow_up_suggestions_endpoint(request))

        self.assertEqual(store.resolve_context.call_args.kwargs.get("preferred_variant"), "key-insights")

    def test_suggestions_require_persisted_summary(self):
        request = FollowUpSuggestionsRequest(
            video_id="abc123",
            summary="Request summary",
            source_context={"title": "Original"},
        )
        store = MagicMock()
        store.resolve_context.return_value = ResolvedFollowUpContext(
            video_id="abc123",
            summary_id=None,
            summary="Resolved summary",
            source_context={"title": "Original", "video_id": "abc123", "id": "abc123"},
        )
        service = {"get_follow_up_suggestions": MagicMock()}

        with patch("ytv2_api.main.get_follow_up_store", return_value=store), \
             patch("ytv2_api.main.get_research_service", return_value=service), \
             self.assertRaises(HTTPException) as context:
            asyncio.run(get_follow_up_suggestions_endpoint(request))

        self.assertEqual(context.exception.status_code, 400)
        service["get_follow_up_suggestions"].assert_not_called()


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment")
class TestFollowUpRunEndpoint(unittest.TestCase):
    def _resolved_context(self):
        return ResolvedFollowUpContext(
            video_id="abc123",
            summary_id=55,
            summary="Resolved summary",
            source_context={
                "title": "Original",
                "url": "https://youtube.com/watch?v=abc123",
                "video_id": "abc123",
                "id": "abc123",
            },
        )

    def test_run_endpoint_returns_cached_database_result(self):
        request = FollowUpRunRequest(
            video_id="abc123",
            summary="Request summary",
            source_context={},
            approved_questions=["What changed?"],
        )
        store = MagicMock()
        store.resolve_context.return_value = self._resolved_context()
        store.get_cached_research.return_value = {
            "run_id": 9,
            "video_id": "abc123",
            "summary_id": 55,
            "answer": "Cached answer",
            "meta": {
                "stored_sources": [],
                "summary_variant_id": 77,
                "summary_variant_revision": 3,
            },
            "status": "ok",
            "cache_key": "cache-key",
            "sources": [
                {
                    "name": "Example",
                    "url": "https://example.com",
                    "domain": "example.com",
                    "tier": "reference",
                    "providers": ["tavily"],
                    "tools": ["search"],
                }
            ],
        }
        service = {
            "build_cache_key": MagicMock(return_value="cache-key"),
            "run_follow_up_research": MagicMock(),
        }

        with patch("ytv2_api.main.get_follow_up_store", return_value=store), \
             patch("ytv2_api.main.get_research_service", return_value=service):
            response = asyncio.run(run_follow_up_research_endpoint(request))

        self.assertEqual(response.answer, "Cached answer")
        self.assertEqual(response.cache_key, "cache-key")
        self.assertTrue(response.meta["cache_hit"])
        self.assertEqual(response.meta["follow_up_run_id"], 9)
        self.assertEqual(response.meta["summary_variant_id"], 77)
        self.assertEqual(response.meta["summary_variant_revision"], 3)
        service["run_follow_up_research"].assert_not_called()

    def test_run_endpoint_persists_fresh_result(self):
        request = FollowUpRunRequest(
            video_id="abc123",
            summary="Request summary",
            source_context={},
            approved_questions=["What changed?"],
        )
        store = MagicMock()
        store.resolve_context.return_value = self._resolved_context()
        store.get_cached_research.return_value = None
        store.store_research_run.return_value = 42
        store.create_summary_variant_reference.return_value = {
            "summary_variant_id": 77,
            "summary_variant_revision": 3,
        }
        result = SimpleNamespace(
            status="ok",
            answer="Fresh answer",
            sources=[
                SimpleNamespace(
                    name="Example",
                    url="https://example.com",
                    domain="example.com",
                    tier="reference",
                    providers=["tavily"],
                    tools=["search"],
                )
            ],
            meta={"cache_key": "cache-key"},
        )
        service = {
            "build_cache_key": MagicMock(return_value="cache-key"),
            "run_follow_up_research": MagicMock(return_value=result),
        }

        with patch("ytv2_api.main.get_follow_up_store", return_value=store), \
             patch("ytv2_api.main.get_research_service", return_value=service):
            response = asyncio.run(run_follow_up_research_endpoint(request))

        self.assertEqual(response.answer, "Fresh answer")
        self.assertEqual(response.meta["follow_up_run_id"], 42)
        self.assertEqual(response.meta["summary_variant_revision"], 3)
        store.store_research_run.assert_called_once()
        store.update_research_run_meta.assert_called_once_with(42, result.meta)
        store.mark_follow_up_available.assert_called_once_with(55)

    def test_run_endpoint_persists_lineage_for_deep_research_follow_up(self):
        request = FollowUpRunRequest(
            video_id="abc123",
            summary="Request summary",
            preferred_variant="deep-research",
            parent_follow_up_run_id=9,
            source_context={},
            approved_questions=["What changed since that report?"],
        )
        store = MagicMock()
        store.resolve_context.return_value = ResolvedFollowUpContext(
            video_id="abc123",
            summary_id=77,
            summary="Resolved deep research answer",
            source_context={
                "title": "Original",
                "url": "https://youtube.com/watch?v=abc123",
                "video_id": "abc123",
                "id": "abc123",
                "parent_follow_up_run_id": 9,
            },
            variant="deep-research",
            summary_revision=4,
        )
        store.get_cached_research.return_value = None
        store.store_research_run.return_value = 42
        store.create_summary_variant_reference.return_value = {
            "summary_variant_id": 88,
            "summary_variant_revision": 5,
        }
        result = SimpleNamespace(
            status="ok",
            answer="Fresh answer",
            sources=[],
            meta={"cache_key": "cache-key"},
        )
        service = {
            "build_cache_key": MagicMock(return_value="cache-key"),
            "run_follow_up_research": MagicMock(return_value=result),
        }

        with patch("ytv2_api.main.get_follow_up_store", return_value=store), \
             patch("ytv2_api.main.get_research_service", return_value=service):
            response = asyncio.run(run_follow_up_research_endpoint(request))

        self.assertEqual(response.meta["parent_follow_up_run_id"], 9)
        self.assertEqual(response.meta["source_summary_variant"], "deep-research")
        self.assertEqual(response.meta["source_summary_revision"], 4)
        store.update_research_run_meta.assert_called_once_with(42, result.meta)

    def test_run_endpoint_forwards_preferred_variant(self):
        request = FollowUpRunRequest(
            video_id="abc123",
            summary="Request summary",
            preferred_variant="bullet-points",
            source_context={},
            approved_questions=["What changed?"],
        )
        store = MagicMock()
        store.resolve_context.return_value = self._resolved_context()
        store.get_cached_research.return_value = {
            "run_id": 9,
            "video_id": "abc123",
            "summary_id": 55,
            "answer": "Cached answer",
            "meta": {"stored_sources": []},
            "status": "ok",
            "cache_key": "cache-key",
            "sources": [],
        }
        service = {
            "build_cache_key": MagicMock(return_value="cache-key"),
            "run_follow_up_research": MagicMock(),
        }

        with patch("ytv2_api.main.get_follow_up_store", return_value=store), \
             patch("ytv2_api.main.get_research_service", return_value=service):
            asyncio.run(run_follow_up_research_endpoint(request))

        self.assertEqual(store.resolve_context.call_args.kwargs.get("preferred_variant"), "bullet-points")

    def test_run_endpoint_requires_persisted_summary(self):
        request = FollowUpRunRequest(
            video_id="abc123",
            summary="Request summary",
            source_context={},
            approved_questions=["What changed?"],
        )
        store = MagicMock()
        store.resolve_context.return_value = ResolvedFollowUpContext(
            video_id="abc123",
            summary_id=None,
            summary="Request summary",
            source_context={"video_id": "abc123", "id": "abc123"},
        )
        service = {
            "build_cache_key": MagicMock(),
            "run_follow_up_research": MagicMock(),
        }

        with patch("ytv2_api.main.get_follow_up_store", return_value=store), \
             patch("ytv2_api.main.get_research_service", return_value=service), \
             self.assertRaises(HTTPException) as context:
            asyncio.run(run_follow_up_research_endpoint(request))

        self.assertEqual(context.exception.status_code, 400)
        service["run_follow_up_research"].assert_not_called()

    def test_run_endpoint_maps_lookup_error_to_404(self):
        request = FollowUpRunRequest(
            video_id="abc123",
            summary="Request summary",
            source_context={},
            approved_questions=["What changed?"],
        )
        store = MagicMock()
        store.resolve_context.side_effect = LookupError("Summary 999 does not belong to video_id abc123")
        service = {
            "build_cache_key": MagicMock(),
            "run_follow_up_research": MagicMock(),
        }

        with patch("ytv2_api.main.get_follow_up_store", return_value=store), \
             patch("ytv2_api.main.get_research_service", return_value=service), \
             self.assertRaises(HTTPException) as context:
            asyncio.run(run_follow_up_research_endpoint(request))

        self.assertEqual(context.exception.status_code, 404)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment")
class TestFollowUpCachedEndpoint(unittest.TestCase):
    def test_cached_lookup_returns_structured_result(self):
        store = MagicMock()
        store.get_cached_research.return_value = {
            "run_id": 9,
            "video_id": "abc123",
            "summary_id": 55,
            "answer": "Cached answer",
            "meta": {"stored_sources": []},
            "status": "ok",
            "cache_key": "cache-key",
            "sources": [],
        }

        with patch("ytv2_api.main.get_follow_up_store", return_value=store):
            response = asyncio.run(check_cached_research("cache-key"))

        self.assertTrue(response.cached)
        self.assertEqual(response.result.answer, "Cached answer")
        self.assertEqual(response.result.meta["follow_up_run_id"], 9)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment")
class TestFollowUpThreadAndChatEndpoints(unittest.TestCase):
    def test_thread_endpoint_returns_lineage(self):
        store = MagicMock()
        store.get_research_run.return_value = {
            "run_id": 12,
            "video_id": "abc123",
            "summary_id": 55,
            "approved_questions": ["What changed?"],
            "question_provenance": ["custom"],
            "answer": "Latest answer",
            "sources": [],
            "status": "ok",
            "created_at": "2026-03-17T12:00:00Z",
            "meta": {},
            "parent_follow_up_run_id": 8,
        }
        store.get_research_thread.return_value = [
            {
                "run_id": 8,
                "video_id": "abc123",
                "summary_id": 55,
                "approved_questions": ["What was the first question?"],
                "question_provenance": ["suggested"],
                "answer": "Earlier answer",
                "sources": [],
                "status": "ok",
                "created_at": "2026-03-17T11:00:00Z",
                "meta": {},
                "parent_follow_up_run_id": None,
            },
            {
                "run_id": 12,
                "video_id": "abc123",
                "summary_id": 55,
                "approved_questions": ["What changed?"],
                "question_provenance": ["custom"],
                "answer": "Latest answer",
                "sources": [],
                "status": "ok",
                "created_at": "2026-03-17T12:00:00Z",
                "meta": {},
                "parent_follow_up_run_id": 8,
            },
        ]

        with patch("ytv2_api.main.get_follow_up_store", return_value=store):
            response = asyncio.run(get_follow_up_thread_endpoint(video_id="abc123", follow_up_run_id=12))

        self.assertEqual(response.root_follow_up_run_id, 8)
        self.assertEqual(response.current_follow_up_run_id, 12)
        self.assertEqual(len(response.turns), 2)
        self.assertEqual(response.turns[0].approved_questions, ["What was the first question?"])

    def test_chat_endpoint_uses_existing_run_context(self):
        request = FollowUpChatRequest(
            video_id="abc123",
            follow_up_run_id=12,
            question="What did the report say about pricing?",
            history=[{"role": "user", "content": "Focus on pricing"}],
        )
        store = MagicMock()
        store.get_research_run.return_value = {
            "run_id": 12,
            "video_id": "abc123",
            "summary_id": 55,
            "approved_questions": ["What changed?"],
            "question_provenance": ["custom"],
            "answer": "Latest answer",
            "sources": [
                {
                    "name": "Example",
                    "url": "https://example.com",
                    "domain": "example.com",
                    "tier": "reference",
                    "providers": ["tavily"],
                    "tools": ["search"],
                }
            ],
            "status": "ok",
            "created_at": "2026-03-17T12:00:00Z",
            "meta": {},
            "parent_follow_up_run_id": 8,
        }
        store.get_research_thread.return_value = [store.get_research_run.return_value]
        service = {
            "answer_follow_up_chat": MagicMock(return_value=(
                "Pricing was discussed in the report.",
                {"llm_provider": "openrouter", "llm_model": "google/gemini-3.1-flash-lite-preview"},
                store.get_research_run.return_value["sources"],
            )),
        }

        with patch("ytv2_api.main.get_follow_up_store", return_value=store), \
             patch("ytv2_api.main.get_research_service", return_value=service):
            response = asyncio.run(answer_follow_up_chat_endpoint(request))

        self.assertEqual(response.follow_up_run_id, 12)
        self.assertEqual(response.answer, "Pricing was discussed in the report.")
        self.assertEqual(response.meta["mode"], "report-chat")
        service["answer_follow_up_chat"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
