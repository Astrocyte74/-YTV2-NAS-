"""Unit tests for follow-up store context resolution."""

import unittest
from unittest.mock import patch

from ytv2_api.follow_up_store import FollowUpStore


class TestFollowUpStoreResolveContext(unittest.TestCase):
    def _store(self):
        store = FollowUpStore.__new__(FollowUpStore)
        store.dsn = "postgresql://example"
        return store

    def test_resolve_context_uses_canonical_deep_research_response(self):
        store = self._store()
        row = {
            "id": 77,
            "video_id": "abc123",
            "text": "Follow-up research available. Canonical result stored in follow_up_research_runs.",
            "variant": "deep-research",
            "revision": 4,
            "title": "Original",
            "canonical_url": "https://youtube.com/watch?v=abc123",
            "published_at": "2026-03-17T00:00:00Z",
        }
        run = {
            "id": 9,
            "research_response": "Canonical deep research answer",
            "research_meta": {"approved_questions": ["What changed?"]},
        }

        with patch.object(store, "_fetch_summary_row", return_value=row), \
             patch.object(store, "_fetch_deep_research_run", return_value=run):
            resolved = store.resolve_context(
                video_id="abc123",
                preferred_variant="deep-research",
                summary="Request summary",
                source_context={},
            )

        self.assertEqual(resolved.summary, "Canonical deep research answer")
        self.assertEqual(resolved.variant, "deep-research")
        self.assertEqual(resolved.summary_revision, 4)
        self.assertEqual(resolved.source_context["parent_follow_up_run_id"], 9)
        self.assertEqual(resolved.source_context["approved_questions"], ["What changed?"])

    def test_resolve_context_falls_back_when_no_canonical_run_exists(self):
        store = self._store()
        row = {
            "id": 78,
            "video_id": "abc123",
            "text": "Stored summary text",
            "variant": "deep-research",
            "revision": 5,
            "title": "Original",
            "canonical_url": "https://youtube.com/watch?v=abc123",
            "published_at": "2026-03-17T00:00:00Z",
        }

        with patch.object(store, "_fetch_summary_row", return_value=row), \
             patch.object(store, "_fetch_deep_research_run", return_value=None):
            resolved = store.resolve_context(
                video_id="abc123",
                preferred_variant="deep-research",
                summary="Request summary",
                source_context={},
            )

        self.assertEqual(resolved.summary, "Stored summary text")
        self.assertEqual(resolved.variant, "deep-research")

    def test_get_research_thread_walks_parent_chain_oldest_first(self):
        store = self._store()
        runs = {
            12: {
                "run_id": 12,
                "parent_follow_up_run_id": 8,
                "video_id": "abc123",
            },
            8: {
                "run_id": 8,
                "parent_follow_up_run_id": None,
                "video_id": "abc123",
            },
        }

        with patch.object(store, "get_research_run", side_effect=lambda run_id, video_id=None: runs.get(run_id)):
            thread = store.get_research_thread(12, video_id="abc123")

        self.assertEqual([turn["run_id"] for turn in thread], [8, 12])


if __name__ == "__main__":
    unittest.main()
