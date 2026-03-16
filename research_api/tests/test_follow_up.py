"""Unit tests for the follow_up module."""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_service.follow_up import (
    FollowUpSuggestion,
    FollowUpResearchPlan,
    MAX_APPROVED_QUESTIONS,
    MAX_PLANNED_QUERIES,
    _build_cache_key,
    _clip_queries,
    _looks_time_sensitive,
    _is_content_old,
    _looks_like_comparison,
    _infer_question_kinds,
    _should_suggest_follow_up,
    _extract_entities,
    build_cache_key,
    suggest_follow_up_questions,
)


class TestCacheKey(unittest.TestCase):
    """Tests for cache key generation."""

    def test_build_cache_key_basic(self):
        """Test basic cache key generation."""
        key = _build_cache_key(
            video_id="abc123",
            summary_id=None,
            approved_questions=["What is the current pricing?"],
            provider_mode="auto",
            depth="balanced",
        )
        self.assertIsInstance(key, str)
        self.assertIn("abc123", key)
        self.assertIn("latest", key)  # summary_id=None becomes "latest"
        self.assertIn("auto", key)
        self.assertIn("balanced", key)

    def test_build_cache_key_with_summary_id(self):
        """Test cache key with specific summary ID."""
        key = _build_cache_key(
            video_id="abc123",
            summary_id=456,
            approved_questions=["What changed?"],
            provider_mode="tavily",
            depth="deep",
        )
        self.assertIn("abc123", key)
        self.assertIn("456", key)  # summary_id
        self.assertIn("tavily", key)
        self.assertIn("deep", key)

    def test_build_cache_key_normalizes_questions(self):
        """Test that cache key normalizes and dedupes questions."""
        questions = ["What is the pricing?", "What is the pricing?", "  pricing?  "]
        key1 = _build_cache_key(
            video_id="vid1",
            summary_id=1,
            approved_questions=questions,
            provider_mode="auto",
            depth="balanced",
        )
        # Same questions in different order should produce same key
        questions_reordered = ["  pricing?  ", "What is the pricing?"]
        key2 = _build_cache_key(
            video_id="vid1",
            summary_id=1,
            approved_questions=questions_reordered,
            provider_mode="auto",
            depth="balanced",
        )
        self.assertEqual(key1, key2)

    def test_build_cache_key_different_questions(self):
        """Test that different questions produce different keys."""
        key1 = _build_cache_key(
            video_id="vid1",
            summary_id=1,
            approved_questions=["pricing?"],
            provider_mode="auto",
            depth="balanced",
        )
        key2 = _build_cache_key(
            video_id="vid1",
            summary_id=1,
            approved_questions=["alternatives?"],
            provider_mode="auto",
            depth="balanced",
        )
        self.assertNotEqual(key1, key2)

    def test_build_cache_key_wrapper(self):
        """Test the public build_cache_key wrapper function."""
        key = build_cache_key(
            video_id="abc123",
            summary_id=789,
            approved_questions=["What changed?"],
            provider_mode="both",
            depth="quick",
        )
        self.assertIsInstance(key, str)
        self.assertIn("abc123", key)
        self.assertIn("789", key)
        self.assertIn("both", key)
        self.assertIn("quick", key)


class TestClipQueries(unittest.TestCase):
    """Tests for query clipping and deduplication."""

    def test_clip_queries_dedupes(self):
        """Test that duplicate queries are removed."""
        queries = ["query 1", "query 1", "query 2", "QUERY 1", "query 2"]
        clipped = _clip_queries(queries)
        # Should dedupe case-insensitively
        self.assertEqual(len(clipped), 2)
        self.assertIn("query 1", clipped)
        self.assertIn("query 2", clipped)

    def test_clip_queries_limits(self):
        """Test that queries are limited to MAX_PLANNED_QUERIES."""
        queries = [f"query {i}" for i in range(MAX_PLANNED_QUERIES + 5)]
        clipped = _clip_queries(queries)
        self.assertEqual(len(clipped), MAX_PLANNED_QUERIES)

    def test_clip_queries_strips_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        queries = ["  query 1  ", "\tquery 2\t", "\nquery 3\n"]
        clipped = _clip_queries(queries)
        for q in clipped:
            self.assertEqual(q, q.strip())

    def test_clip_filters_empty(self):
        """Test that empty strings are filtered out."""
        queries = ["query 1", "", "  ", "query 2", None]
        clipped = _clip_queries(queries)
        self.assertEqual(len(clipped), 2)
        self.assertIn("query 1", clipped)
        self.assertIn("query 2", clipped)


class TestTimeSensitivity(unittest.TestCase):
    """Tests for time-sensitivity detection."""

    def test_looks_time_sensitive_old_content(self):
        """Test that old content is flagged as time-sensitive."""
        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        self.assertTrue(_looks_time_sensitive(old_date, "Some summary", {}))

    def test_looks_time_sensitive_recent_content(self):
        """Test that recent content is not flagged unless keywords present."""
        recent_date = (datetime.now() - timedelta(days=10)).isoformat()
        self.assertFalse(_looks_time_sensitive(recent_date, "A neutral summary", {}))

    def test_looks_time_sensitive_keywords(self):
        """Test that time-sensitive keywords trigger the flag."""
        keywords = ["pricing", "cost", "subscription", "free tier", "latest", "newest", "current"]
        for keyword in keywords:
            summary = f"This mentions {keyword} information"
            self.assertTrue(_looks_time_sensitive("", summary, {}), f"Keyword '{keyword}' should trigger")

    def test_is_content_old(self):
        """Test content age detection."""
        old_date = (datetime.now() - timedelta(days=95)).isoformat()
        self.assertTrue(_is_content_old(old_date))

        recent_date = (datetime.now() - timedelta(days=30)).isoformat()
        self.assertFalse(_is_content_old(recent_date))

        # Invalid date format should return False, not crash
        self.assertFalse(_is_content_old("invalid-date"))


class TestComparisonDetection(unittest.TestCase):
    """Tests for comparison question detection."""

    def test_looks_like_comparison_keywords(self):
        """Test that comparison keywords are detected."""
        comparison_questions = [
            "How does this compare to alternatives?",
            "What is the difference between X and Y?",
            "X vs Y: which is better?",
            "Compare X versus Y",
            "What alternatives exist to X?",
        ]
        for q in comparison_questions:
            self.assertTrue(_looks_like_comparison([q]), f"Question should be comparison: {q}")

    def test_not_comparison(self):
        """Test that non-comparison questions are not flagged."""
        non_comparison = [
            "What is the current pricing?",
            "How does X work?",
            "Tell me more about Y",
            "What are the features?",
        ]
        for q in non_comparison:
            self.assertFalse(_looks_like_comparison([q]), f"Question should not be comparison: {q}")

    def test_multiple_questions_one_comparison(self):
        """Test that comparison in any question triggers the flag."""
        questions = ["What is the pricing?", "How does this compare to alternatives?"]
        self.assertTrue(_looks_like_comparison(questions))


class TestQuestionKindInference(unittest.TestCase):
    """Tests for question kind inference."""

    def test_infer_pricing_kind(self):
        """Test pricing question kind detection."""
        questions = ["What is the current pricing?", "How much does it cost?", "subscription pricing"]
        kinds = _infer_question_kinds(questions, {})
        self.assertTrue(any(k == "pricing" for k in kinds))

    def test_infer_comparison_kind(self):
        """Test comparison question kind detection."""
        questions = ["How does this compare to alternatives?", "X vs Y comparison"]
        kinds = _infer_question_kinds(questions, {})
        self.assertTrue(any(k == "comparison" for k in kinds))

    def test_infer_what_changed_kind(self):
        """Test what-changed question kind detection."""
        questions = ["What changed since publication?", "How has this updated?"]
        kinds = _infer_question_kinds(questions, {})
        self.assertTrue(any(k == "what_changed" for k in kinds))

    def test_infer_current_state_kind(self):
        """Test current-state question kind detection."""
        questions = ["What is the current status today?", "Is this available right now?"]
        kinds = _infer_question_kinds(questions, {})
        self.assertTrue(all(k == "current_state" for k in kinds))

    def test_infer_background_kind_default(self):
        """Test that unmatched questions default to 'background'."""
        questions = ["Tell me more about this", "What are the features?"]
        kinds = _infer_question_kinds(questions, {})
        self.assertTrue(all(k == "background" for k in kinds))


class TestEntityExtraction(unittest.TestCase):
    """Tests for entity extraction."""

    def test_extract_entities_product_names(self):
        """Test extraction of product/company names."""
        # Use text that matches the regex patterns in _extract_entities
        text = "This review compares Cursor Editor and Windsurf AI with GPT-5 model."
        entities = _extract_entities(text)
        self.assertIsInstance(entities, list)
        # The regex looks for patterns like "GPT-5" and multi-word capitalized names
        # Should find at least GPT-5
        self.assertTrue(len(entities) > 0, f"Should find entities like GPT-5, got: {entities}")

    def test_extract_entities_filters_common_words(self):
        """Test that common words are filtered out."""
        text = "The quick brown Fox jumps over the Lazy Dog"
        entities = _extract_entities(text)
        # "The", "This", "That" etc should be filtered
        for e in entities:
            self.assertNotIn(e.lower(), ["the", "this", "that", "these", "those"])

    def test_extract_entities_limits(self):
        """Test that entity extraction has a limit."""
        text = " ".join([f"Product{i} Name{i}" for i in range(20)])
        entities = _extract_entities(text)
        self.assertLessEqual(len(entities), 10)


class TestSuggestionFiltering(unittest.TestCase):
    """Tests for follow-up suggestion filtering."""

    def test_should_suggest_with_entities(self):
        """Test suggestions are generated when entities exist."""
        result = _should_suggest_follow_up(
            source_context={},
            summary="This discusses Cursor AI pricing and features.",
            entities=["Cursor AI", "pricing"]
        )
        self.assertTrue(result)

    def test_should_not_suggest_without_entities(self):
        """Test that no entities means no suggestions."""
        result = _should_suggest_follow_up(
            source_context={},
            summary="A generic summary without named entities.",
            entities=None
        )
        # Should try to extract entities, but if none found, return False
        self.assertFalse(result)

    def test_should_suggest_recent_with_pricing(self):
        """Test that even recent content triggers suggestions for pricing."""
        recent_date = (datetime.now() - timedelta(days=5)).isoformat()
        result = _should_suggest_follow_up(
            source_context={"published_at": recent_date},
            summary="This mentions the subscription pricing and cost structure.",
            entities=["Product"]
        )
        # Should suggest because of pricing keywords
        self.assertTrue(result)

    def test_should_not_suggest_evergreen_recent(self):
        """Test that evergreen recent content doesn't trigger suggestions."""
        recent_date = (datetime.now() - timedelta(days=5)).isoformat()
        result = _should_suggest_follow_up(
            source_context={"published_at": recent_date},
            summary="An educational overview of machine learning concepts.",
            entities=["Machine Learning", "Concepts"]
        )
        # Should not suggest - recent and no time-sensitive keywords
        self.assertFalse(result)


class TestDataclasses(unittest.TestCase):
    """Tests for dataclass definitions."""

    def test_follow_up_suggestion(self):
        """Test FollowUpSuggestion dataclass."""
        suggestion = FollowUpSuggestion(
            id="test-id",
            label="Test Label",
            question="Test question?",
            reason="Test reason",
            kind="pricing",
            priority=0.8,
            default_selected=True,
            provenance="suggested"
        )
        self.assertEqual(suggestion.id, "test-id")
        self.assertEqual(suggestion.kind, "pricing")
        self.assertEqual(suggestion.provenance, "suggested")

    def test_follow_up_research_plan(self):
        """Test FollowUpResearchPlan dataclass."""
        plan = FollowUpResearchPlan(
            approved_questions=["Question 1", "Question 2"],
            question_provenance=["suggested", "preset"],
            question_kinds=["pricing", "comparison"],
            planned_queries=["query 1", "query 2"],
            coverage_map=[
                {"approved_question": "Question 1", "covered_by": ["query 1"]},
                {"approved_question": "Question 2", "covered_by": ["query 2"]},
            ],
            dedupe_notes="Test notes",
            provider_mode="auto",
            tool_mode="auto",
            depth="balanced",
        )
        self.assertEqual(len(plan.approved_questions), 2)
        self.assertEqual(plan.provider_mode, "auto")
        self.assertFalse(plan.compare)  # Default value
        self.assertFalse(plan.freshness_sensitive)  # Default value


class TestSuggestionGeneration(unittest.TestCase):
    """Tests for follow-up suggestion generation behavior."""

    @patch("research_service.follow_up.chat_json_schema")
    def test_suggestions_retry_once_after_parse_failure(self, mock_chat_json_schema):
        mock_chat_json_schema.side_effect = [
            RuntimeError("Unable to parse structured JSON response"),
            (
                {
                    "should_suggest": True,
                    "suggestions": [
                        {
                            "id": "s1",
                            "label": "Has this been independently replicated?",
                            "question": "Has this research been independently replicated or challenged?",
                            "reason": "Replication is important for scientific claims.",
                            "kind": "fact_check",
                            "priority": 0.9,
                            "default_selected": True,
                        }
                    ],
                },
                "openrouter",
                "google/gemini-3.1-flash-lite-preview",
            ),
        ]

        suggestions = suggest_follow_up_questions(
            source_context={"title": "Butterfly memory", "type": "youtube"},
            summary="A video about inherited memory in butterflies.",
            entities=["Joe", "Jonah Guy"],
            max_suggestions=3,
        )

        self.assertEqual(mock_chat_json_schema.call_count, 2)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].kind, "fact_check")

    @patch("research_service.follow_up.chat_json_schema")
    def test_suggestions_return_none_when_llm_keeps_failing(self, mock_chat_json_schema):
        mock_chat_json_schema.side_effect = RuntimeError("Unable to parse structured JSON response")

        suggestions = suggest_follow_up_questions(
            source_context={"title": "Butterfly memory", "type": "youtube"},
            summary="A video about inherited memory in butterflies.",
            entities=["Joe", "Jonah Guy"],
            max_suggestions=3,
        )

        self.assertEqual(mock_chat_json_schema.call_count, 2)
        self.assertEqual(suggestions, [])

    @patch("research_service.follow_up.RESEARCH_PLANNER_PROVIDER", "openrouter")
    @patch("research_service.follow_up.INCEPTION_MODEL", "mercury-2")
    @patch("research_service.follow_up.INCEPTION_API_KEY", "test-key")
    @patch("research_service.follow_up.chat_json_schema")
    def test_suggestions_escalate_to_inception_after_retry(self, mock_chat_json_schema):
        mock_chat_json_schema.side_effect = [
            RuntimeError("Unable to parse structured JSON response"),
            RuntimeError("Unable to parse structured JSON response"),
            (
                {
                    "should_suggest": True,
                    "suggestions": [
                        {
                            "id": "s1",
                            "label": "Has this been independently replicated?",
                            "question": "Has this research been independently replicated or challenged?",
                            "reason": "Replication is important for scientific claims.",
                            "kind": "fact_check",
                            "priority": 0.9,
                            "default_selected": True,
                        }
                    ],
                },
                "inception",
                "mercury-2",
            ),
        ]

        suggestions = suggest_follow_up_questions(
            source_context={"title": "Butterfly memory", "type": "youtube"},
            summary="A video about inherited memory in butterflies.",
            entities=["Joe", "Jonah Guy"],
            max_suggestions=3,
        )

        self.assertEqual(mock_chat_json_schema.call_count, 3)
        self.assertEqual(mock_chat_json_schema.call_args_list[2].kwargs["provider"], "inception")
        self.assertEqual(mock_chat_json_schema.call_args_list[2].kwargs["model_override"], "mercury-2")
        self.assertEqual(len(suggestions), 1)


class TestConstants(unittest.TestCase):
    """Tests for constant values."""

    def test_max_approved_questions(self):
        """Test MAX_APPROVED_QUESTIONS is reasonable."""
        self.assertGreater(MAX_APPROVED_QUESTIONS, 0)
        self.assertLessEqual(MAX_APPROVED_QUESTIONS, 5)
        self.assertEqual(MAX_APPROVED_QUESTIONS, 3)

    def test_max_planned_queries(self):
        """Test MAX_PLANNED_QUERIES is reasonable."""
        self.assertGreater(MAX_PLANNED_QUERIES, 0)
        self.assertLessEqual(MAX_PLANNED_QUERIES, 10)
        self.assertEqual(MAX_PLANNED_QUERIES, 6)


if __name__ == "__main__":
    unittest.main()
