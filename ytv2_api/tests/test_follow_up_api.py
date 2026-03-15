"""Tests for follow-up research API endpoint behavior."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ytv2_api.follow_up_store import ResolvedFollowUpContext
from ytv2_api.models import FollowUpRunRequest, FollowUpSuggestionsRequest

try:
    from ytv2_api.main import (
        check_cached_research,
        get_follow_up_suggestions_endpoint,
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
            "meta": {"stored_sources": []},
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
        store.mark_follow_up_available.assert_called_once_with(55)


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


if __name__ == "__main__":
    unittest.main()
