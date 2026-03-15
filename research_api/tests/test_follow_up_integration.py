"""Integration tests for follow-up research flow.

These tests verify the complete pipeline:
1. Generate suggestions from a summary
2. Approve questions (single and multiple)
3. Plan follow-up research (consolidation)
4. Execute research plan
5. Synthesize sectioned report
6. Verify cache key generation
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_service.service import (
    get_follow_up_suggestions,
    run_follow_up_research,
    get_research_capabilities,
    clear_follow_up_research_cache,
)
from research_service.follow_up import (
    plan_follow_up_research,
    build_cache_key,
    MAX_APPROVED_QUESTIONS,
)
from research_service.models import ResearchRunResult


class TestFollowUpSuggestions(unittest.TestCase):
    """Tests for follow-up suggestion generation."""

    def test_get_suggestions_returns_list(self):
        """Test that get_follow_up_suggestions returns a list."""
        source_context = {
            "title": "Cursor AI Review",
            "url": "https://example.com/cursor-review",
            "published_at": "2024-06-01T00:00:00Z",
            "type": "youtube",
        }
        summary = "This video reviews Cursor AI, an AI-powered code editor. It covers the pricing, features, and how it compares to Copilot."

        suggestions = get_follow_up_suggestions(
            source_context=source_context,
            summary=summary,
            entities=["Cursor AI", "Copilot"],
            max_suggestions=3,
        )

        self.assertIsInstance(suggestions, list)

    def test_get_suggestions_structure(self):
        """Test that suggestions have the correct structure."""
        source_context = {
            "title": "Product Review",
            "url": "https://example.com/review",
            "published_at": "2024-01-01T00:00:00Z",
            "type": "article",
        }
        summary = "A review mentioning pricing and subscription costs."

        with patch('research_service.follow_up._extract_entities', return_value=["Product"]):
            suggestions = get_follow_up_suggestions(
                source_context=source_context,
                summary=summary,
                max_suggestions=3,
            )

        # Check structure of returned suggestions
        for suggestion in suggestions:
            self.assertIn("id", suggestion)
            self.assertIn("label", suggestion)
            self.assertIn("question", suggestion)
            self.assertIn("reason", suggestion)
            self.assertIn("kind", suggestion)
            self.assertIn("priority", suggestion)
            self.assertIn("default_selected", suggestion)
            self.assertIn("provenance", suggestion)

    def test_get_suggestions_empty_when_research_disabled(self):
        """Test that no suggestions are returned when research is disabled."""
        with patch('research_service.service.RESEARCH_ENABLED', False):
            source_context = {"title": "Test", "url": "https://test.com"}
            summary = "Test summary"

            suggestions = get_follow_up_suggestions(
                source_context=source_context,
                summary=summary,
            )

            self.assertEqual(suggestions, [])


class TestFollowUpPlanning(unittest.TestCase):
    """Tests for follow-up research planning."""

    def test_plan_consolidates_questions(self):
        """Test that planning consolidates related questions."""
        source_context = {
            "title": "Pricing Comparison",
            "url": "https://example.com/pricing",
            "published_at": "2024-01-01T00:00:00Z",
            "type": "article",
        }
        summary = "An article discussing product pricing and costs."

        approved_questions = [
            "What is the current pricing?",
            "How much does it cost now?",
        ]  # These should be consolidated

        with patch('research_service.follow_up.chat_json_schema') as mock_llm:
            mock_llm.return_value = (
                {
                    "planned_queries": ["current pricing for product"],
                    "coverage_map": [
                        {"approved_question": "What is the current pricing?", "covered_by": ["current pricing for product"]},
                        {"approved_question": "How much does it cost now?", "covered_by": ["current pricing for product"]},
                    ],
                    "dedupe_notes": "Consolidated pricing questions",
                    "provider_mode": "auto",
                    "tool_mode": "auto",
                    "freshness_sensitive": True,
                },
                "openrouter",
                "gemini-2.5-flash",
            )

            plan = plan_follow_up_research(
                source_context=source_context,
                summary=summary,
                approved_questions=approved_questions,
                provider_mode="auto",
                tool_mode="auto",
                depth="balanced",
            )

            # Verify consolidation happened
            self.assertLessEqual(len(plan.planned_queries), len(approved_questions))

    def test_plan_max_questions_enforced(self):
        """Test that MAX_APPROVED_QUESTIONS is enforced."""
        source_context = {"title": "Test", "url": "https://test.com"}
        summary = "Test summary"

        too_many_questions = [f"Question {i}" for i in range(MAX_APPROVED_QUESTIONS + 2)]

        with self.assertRaises(ValueError) as context:
            plan_follow_up_research(
                source_context=source_context,
                summary=summary,
                approved_questions=too_many_questions,
            )

        self.assertIn("Maximum", str(context.exception))

    def test_plan_coverage_map(self):
        """Test that coverage map covers all approved questions."""
        source_context = {"title": "Test", "url": "https://test.com"}
        summary = "Test summary"

        approved_questions = ["Question 1", "Question 2"]

        with patch('research_service.follow_up.chat_json_schema') as mock_llm:
            mock_llm.return_value = (
                {
                    "planned_queries": ["query 1"],
                    "coverage_map": [
                        {"approved_question": "Question 1", "covered_by": ["query 1"]},
                        {"approved_question": "Question 2", "covered_by": ["query 1"]},
                    ],
                    "dedupe_notes": "Test",
                    "provider_mode": "auto",
                    "tool_mode": "auto",
                },
                "openrouter",
                "gemini-2.5-flash",
            )

            plan = plan_follow_up_research(
                source_context=source_context,
                summary=summary,
                approved_questions=approved_questions,
            )

            # Verify coverage map has entries for all approved questions
            covered_questions = {entry["approved_question"] for entry in plan.coverage_map}
            self.assertEqual(covered_questions, set(approved_questions))

    def test_plan_normalizes_invalid_coverage_map(self):
        """Test that invalid coverage entries are repaired deterministically."""
        source_context = {"title": "Test", "url": "https://test.com"}
        summary = "Test summary"
        approved_questions = ["Question 1", "Question 2"]

        with patch('research_service.follow_up.chat_json_schema') as mock_llm:
            mock_llm.return_value = (
                {
                    "planned_queries": ["query 1"],
                    "coverage_map": [
                        {"approved_question": "Question 1", "covered_by": ["missing query"]},
                    ],
                    "dedupe_notes": "Test",
                    "provider_mode": "auto",
                    "tool_mode": "auto",
                },
                "openrouter",
                "gemini-2.5-flash",
            )

            plan = plan_follow_up_research(
                source_context=source_context,
                summary=summary,
                approved_questions=approved_questions,
            )

            self.assertEqual(
                plan.coverage_map,
                [
                    {"approved_question": "Question 1", "covered_by": ["query 1"]},
                    {"approved_question": "Question 2", "covered_by": ["query 1"]},
                ],
            )


class TestCacheKeyGeneration(unittest.TestCase):
    """Tests for cache key generation in the flow."""

    def test_cache_key_consistency(self):
        """Test that same inputs produce same cache key."""
        source_context = {
            "video_id": "abc123",
            "summary_id": 456,
        }
        approved_questions = ["What is the pricing?"]

        key1 = build_cache_key(
            video_id=source_context["video_id"],
            summary_id=source_context["summary_id"],
            approved_questions=approved_questions,
            provider_mode="auto",
            depth="balanced",
        )

        key2 = build_cache_key(
            video_id=source_context["video_id"],
            summary_id=source_context["summary_id"],
            approved_questions=approved_questions,
            provider_mode="auto",
            depth="balanced",
        )

        self.assertEqual(key1, key2)

    def test_cache_key_difference(self):
        """Test that different inputs produce different cache keys."""
        base_context = {"video_id": "abc123", "summary_id": 456}

        key1 = build_cache_key(
            video_id=base_context["video_id"],
            summary_id=base_context["summary_id"],
            approved_questions=["Question 1"],
            provider_mode="auto",
            depth="balanced",
        )

        key2 = build_cache_key(
            video_id=base_context["video_id"],
            summary_id=base_context["summary_id"],
            approved_questions=["Question 2"],
            provider_mode="auto",
            depth="balanced",
        )

        self.assertNotEqual(key1, key2)


class TestRunFollowUpResearch(unittest.TestCase):
    """Tests for the main run_follow_up_research entry point."""

    def setUp(self):
        clear_follow_up_research_cache()

    def tearDown(self):
        clear_follow_up_research_cache()

    def test_returns_research_run_result(self):
        """Test that run_follow_up_research returns ResearchRunResult."""
        source_context = {
            "title": "Test",
            "url": "https://test.com",
            "video_id": "test123",
        }
        summary = "Test summary"
        approved_questions = ["What is the current state?"]

        with patch('research_service.service.plan_follow_up_research') as mock_plan, \
             patch('research_service.service.execute_research_plan') as mock_exec, \
             patch('research_service.service.synthesize_follow_up') as mock_synth:

            # Setup mocks
            mock_plan.return_value = MagicMock(
                planned_queries=["test query"],
                question_kinds=["background"],
                provider_mode="auto",
                tool_mode="auto",
                depth="balanced",
            )
            mock_exec.return_value = ([], [], {})
            mock_synth.return_value = ("Test answer", {"llm_provider": "test", "llm_model": "test"})

            result = run_follow_up_research(
                source_context=source_context,
                summary=summary,
                approved_questions=approved_questions,
            )

            self.assertIsInstance(result, ResearchRunResult)

    def test_error_when_research_disabled(self):
        """Test that error is returned when research is disabled."""
        with patch('research_service.service.RESEARCH_ENABLED', False):
            result = run_follow_up_research(
                source_context={"title": "Test"},
                summary="Test summary",
                approved_questions=["Test question"],
            )

            self.assertEqual(result.status, "error")
            self.assertIn("disabled", result.meta.get("error", ""))

    def test_error_when_no_questions(self):
        """Test that error is returned when no questions provided."""
        with patch('research_service.service.RESEARCH_ENABLED', True):
            result = run_follow_up_research(
                source_context={"title": "Test"},
                summary="Test summary",
                approved_questions=[],
            )

            self.assertEqual(result.status, "error")
            self.assertIn("no_questions", result.meta.get("error", ""))

    def test_error_when_too_many_questions(self):
        """Test that error is returned when too many questions."""
        with patch('research_service.service.RESEARCH_ENABLED', True):
            too_many = [f"Q{i}" for i in range(MAX_APPROVED_QUESTIONS + 1)]
            result = run_follow_up_research(
                source_context={"title": "Test"},
                summary="Test summary",
                approved_questions=too_many,
            )

            self.assertEqual(result.status, "error")
            self.assertIn("too_many_questions", result.meta.get("error", ""))

    def test_includes_cache_key_in_meta(self):
        """Test that cache key is included in result metadata."""
        source_context = {
            "title": "Test",
            "url": "https://test.com",
            "video_id": "vid123",
            "summary_id": 789,
        }
        approved_questions = ["Test question?"]

        with patch('research_service.service.plan_follow_up_research') as mock_plan, \
             patch('research_service.service.execute_research_plan') as mock_exec, \
             patch('research_service.service.synthesize_follow_up') as mock_synth:

            mock_plan.return_value = MagicMock(
                planned_queries=["test"],
                question_kinds=["background"],
                provider_mode="auto",
                tool_mode="auto",
                depth="balanced",
            )
            mock_exec.return_value = ([], [], {})
            mock_synth.return_value = ("Answer", {"llm_provider": "test", "llm_model": "test"})

            result = run_follow_up_research(
                source_context=source_context,
                summary="Summary",
                approved_questions=approved_questions,
                provider_mode="auto",
                depth="balanced",
            )

            # Verify cache key is in metadata
            self.assertIn("cache_key", result.meta)
            self.assertIsInstance(result.meta["cache_key"], str)

    def test_uses_plan_compare_not_caller_compare(self):
        """Test that plan.compare is used instead of caller's compare argument."""
        source_context = {
            "title": "Test",
            "url": "https://test.com",
            "video_id": "vid123",
        }
        approved_questions = ["How does this compare to alternatives?"]

        with patch('research_service.service.plan_follow_up_research') as mock_plan, \
             patch('research_service.service.execute_research_plan') as mock_exec, \
             patch('research_service.service.synthesize_follow_up') as mock_synth:

            # Plan infers compare=True from question content
            mock_plan.return_value = MagicMock(
                planned_queries=["test"],
                question_kinds=["comparison"],
                provider_mode="auto",
                tool_mode="auto",
                depth="balanced",
                compare=True,  # Planner infers comparison intent
            )
            mock_exec.return_value = ([], [], {})
            mock_synth.return_value = ("Answer", {"llm_provider": "test", "llm_model": "test"})

            # Caller passes compare=False (wrong), but plan should override
            result = run_follow_up_research(
                source_context=source_context,
                summary="Summary",
                approved_questions=approved_questions,
                compare=False,  # Caller incorrectly says no comparison
            )

            # Verify execute_research_plan was called with plan.compare=True
            mock_exec.assert_called_once()
            call_kwargs = mock_exec.call_args[1]
            self.assertTrue(call_kwargs["compare"], "Should use plan.compare=True, not caller's compare=False")

            # Verify synthesize_follow_up was called with plan.compare=True
            mock_synth.assert_called_once()
            call_kwargs = mock_synth.call_args[1]
            self.assertTrue(call_kwargs["compare"], "Should use plan.compare=True, not caller's compare=False")

    def test_uses_explicit_summary_id_for_cache(self):
        """Test that explicit summary_id is used for cache key, not source_context."""
        source_context = {
            "title": "Test",
            "url": "https://test.com",
            "video_id": "vid123",
            "summary_id": 999,  # This should be IGNORED
        }
        approved_questions = ["Test question?"]

        with patch('research_service.service.plan_follow_up_research') as mock_plan, \
             patch('research_service.service.execute_research_plan') as mock_exec, \
             patch('research_service.service.synthesize_follow_up') as mock_synth, \
             patch('research_service.service.build_cache_key') as mock_cache:

            mock_plan.return_value = MagicMock(
                planned_queries=["test"],
                question_kinds=["background"],
                provider_mode="auto",
                tool_mode="auto",
                depth="balanced",
            )
            mock_exec.return_value = ([], [], {})
            mock_synth.return_value = ("Answer", {"llm_provider": "test", "llm_model": "test"})
            mock_cache.return_value = "cache_key_123"

            # Pass explicit summary_id=456
            result = run_follow_up_research(
                source_context=source_context,
                summary="Summary",
                approved_questions=approved_questions,
                summary_id=456,  # Explicit parameter
            )

            # Verify build_cache_key was called with summary_id=456, NOT 999
            mock_cache.assert_called_once()
            call_kwargs = mock_cache.call_args[1]
            self.assertEqual(call_kwargs["summary_id"], 456,
                "Should use explicit summary_id=456, not source_context.summary_id=999")

    def test_second_call_uses_cached_result(self):
        """Test that identical follow-up runs reuse cached results."""
        source_context = {
            "title": "Test",
            "url": "https://test.com",
            "video_id": "vid123",
        }
        approved_questions = ["What is the current state?"]

        with patch('research_service.service.plan_follow_up_research') as mock_plan, \
             patch('research_service.service.execute_research_plan') as mock_exec, \
             patch('research_service.service.synthesize_follow_up') as mock_synth:

            mock_plan.return_value = MagicMock(
                planned_queries=["test query"],
                question_kinds=["current_state"],
                provider_mode="auto",
                tool_mode="auto",
                depth="balanced",
                compare=False,
                coverage_map=[{"approved_question": approved_questions[0], "covered_by": ["test query"]}],
                dedupe_notes="Test",
            )
            mock_exec.return_value = ([], [], {})
            mock_synth.return_value = ("Cached answer", {"llm_provider": "test", "llm_model": "test"})

            first = run_follow_up_research(
                source_context=source_context,
                summary="Summary",
                approved_questions=approved_questions,
            )
            second = run_follow_up_research(
                source_context=source_context,
                summary="Summary",
                approved_questions=approved_questions,
            )

            self.assertEqual(first.answer, "Cached answer")
            self.assertEqual(second.answer, "Cached answer")
            self.assertFalse(first.meta["cache_hit"])
            self.assertTrue(second.meta["cache_hit"])
            mock_plan.assert_called_once()
            mock_exec.assert_called_once()
            mock_synth.assert_called_once()


class TestEndToEndFlow(unittest.TestCase):
    """End-to-end tests for the complete follow-up research flow."""

    def test_full_flow_skeleton(self):
        """Skeleton test for the full flow (requires mocking)."""
        # This test verifies the flow structure without actual API calls
        source_context = {
            "title": "Cursor AI Review",
            "url": "https://example.com/cursor",
            "published_at": "2024-06-01T00:00:00Z",
            "type": "youtube",
            "video_id": "cursor123",
        }

        summary = "Review of Cursor AI covering pricing and features."

        # Step 1: Get suggestions
        with patch('research_service.follow_up._extract_entities', return_value=["Cursor AI"]):
            suggestions = get_follow_up_suggestions(
                source_context=source_context,
                summary=summary,
                max_suggestions=3,
            )
            self.assertIsInstance(suggestions, list)

        # Step 2: Plan research with approved questions
        approved_questions = ["What is the current pricing?"]

        with patch('research_service.follow_up.chat_json_schema') as mock_llm:
            mock_llm.return_value = (
                {
                    "planned_queries": ["Cursor AI current pricing 2026"],
                    "coverage_map": [
                        {"approved_question": "What is the current pricing?", "covered_by": ["Cursor AI current pricing 2026"]},
                    ],
                    "dedupe_notes": "Direct pricing query",
                    "provider_mode": "auto",
                    "tool_mode": "auto",
                },
                "openrouter",
                "gemini-2.5-flash",
            )

            plan = plan_follow_up_research(
                source_context=source_context,
                summary=summary,
                approved_questions=approved_questions,
            )

            self.assertEqual(len(plan.approved_questions), 1)
            self.assertEqual(len(plan.planned_queries), 1)

        # Step 3: Verify cache key generation
        cache_key = build_cache_key(
            video_id=source_context["video_id"],
            summary_id=None,
            approved_questions=approved_questions,
            provider_mode=plan.provider_mode,
            depth=plan.depth,
        )

        self.assertIsInstance(cache_key, str)
        self.assertIn("cursor123", cache_key)


class TestCapabilities(unittest.TestCase):
    """Tests for research capabilities reporting."""

    def test_get_capabilities_returns_dict(self):
        """Test that get_research_capabilities returns expected structure."""
        caps = get_research_capabilities()

        self.assertIsInstance(caps, dict)
        self.assertIn("enabled", caps)
        self.assertIn("providers", caps)
        self.assertIn("llm_configured", caps)
        self.assertIn("defaults", caps)


if __name__ == "__main__":
    unittest.main()
