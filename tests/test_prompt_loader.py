"""
Parity and validation tests for the prompt loader refactor.

Run with: cd backend && python3 -m pytest tests/test_prompt_loader.py -v

These tests verify:
1. The prompt loader resolves all expected dot-paths
2. Invalid paths fail immediately
3. Unresolved placeholders fail immediately
4. Shared context blocks resolve correctly
5. Reddit context is used for reddit prompts
6. Representative prompt renders match expected patterns
"""

import os
import sys
import pytest

# Ensure backend/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_loader import (
    render_prompt,
    render_prompt_only,
    get_prompt_entry,
    get_llm_config,
    list_prompts,
    _load_prompts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_VARS = {
    "title": "Test Video Title",
    "uploader": "Test Channel",
    "upload_date": "20260413",
    "duration": "300",
    "url": "https://youtube.com/watch?v=test123",
    "transcript": "This is a test transcript with some content.",
    "lang_instruction": "IMPORTANT: Respond in English.",
}

REDDIT_VARS = {
    "subreddit": "technology",
    "title": "Reddit Thread Title",
    "url": "https://reddit.com/r/technology/comments/abc",
    "transcript": "Thread content about technology.",
    "lang_instruction": "IMPORTANT: Respond in English.",
}


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Strict validation: bad inputs should fail loudly."""

    def test_unknown_prompt_path(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_prompt_entry("nonexistent.prompt")

    def test_unknown_section(self):
        with pytest.raises(KeyError, match="Unknown prompt section"):
            get_prompt_entry("fake_section.some_prompt")

    def test_unknown_entry_in_valid_section(self):
        with pytest.raises(KeyError, match="Unknown prompt"):
            get_prompt_entry("primary_summaries.nonexistent_variant")

    def test_single_part_unknown(self):
        with pytest.raises(KeyError):
            get_prompt_entry("totally_bogus")

    def test_unresolved_placeholder_fails(self):
        """Rendering a prompt without providing all variables should fail."""
        # The comprehensive prompt needs 'transcript', etc.
        with pytest.raises(ValueError, match="Unresolved"):
            render_prompt("primary_summaries.comprehensive", {"title": "only title"})

    def test_allow_unresolved_flag(self):
        """allow_unresolved=True should suppress the ValueError."""
        result = render_prompt(
            "primary_summaries.comprehensive",
            {"title": "only title"},
            allow_unresolved=True,
        )
        assert "only title" in result


# ---------------------------------------------------------------------------
# Shared context tests
# ---------------------------------------------------------------------------


class TestSharedContext:
    """Verify context blocks resolve correctly."""

    def test_base_context_in_comprehensive(self):
        result = render_prompt("primary_summaries.comprehensive", SAMPLE_VARS)
        assert "Title: Test Video Title" in result
        assert "Source: Test Channel" in result
        assert "Source Text:" in result

    def test_reddit_context_in_reddit_discussion(self):
        result = render_prompt("primary_summaries.reddit_discussion", REDDIT_VARS)
        assert "Subreddit: technology" in result
        assert "Thread content:" in result
        # Should NOT contain video-shaped framing
        assert "Source Text:" not in result

    def test_audio_base_context(self):
        result = render_prompt("audio_prompts.audio_from_transcript", SAMPLE_VARS)
        assert "Video Title: Test Video Title" in result
        assert "Full Transcript:" in result

    def test_no_context_for_audio_from_summary(self):
        result = render_prompt_only("audio_prompts.audio_from_summary", {"summary": "Test"})
        assert "Test" in result
        assert "Source Text:" not in result


# ---------------------------------------------------------------------------
# Resolution tests — all expected prompt paths resolve
# ---------------------------------------------------------------------------


class TestPromptResolution:
    """Verify all prompt families resolve correctly."""

    PRIMARY_PROMPTS = [
        "primary_summaries.comprehensive",
        "primary_summaries.bullet_points",
        "primary_summaries.key_insights",
        "primary_summaries.reddit_discussion",
        "primary_summaries.executive",
        "primary_summaries.adaptive",
    ]

    AUDIO_PROMPTS = [
        "audio_prompts.audio_narration",
        "audio_prompts.audio_from_transcript",
        "audio_prompts.audio_from_summary",
        "audio_prompts.condense_for_tts",
    ]

    STRUCTURED_PROMPTS = [
        "structured_pipeline.planner",
        "structured_pipeline.summarize_with_plan",
        "structured_pipeline.chapter_detailed",
        "structured_pipeline.chapter_basic",
        "structured_pipeline.chapter_merge",
        "structured_pipeline.chapter_intro_conclusion",
    ]

    CHUNKED_PROMPTS = [
        "chunked_pipeline.chunk_summary",
        "chunked_pipeline.combine_chunks",
        "chunked_extractors.extract_bullet_points",
        "chunked_extractors.extract_key_insights",
        "chunked_extractors.extract_reddit_discussion",
    ]

    AUDIO_ON_DEMAND = [
        "audio_on_demand.audio_current",
        "audio_on_demand.audio_briefing",
    ]

    WIKI_PROMPTS = [
        "wikipedia.section_bullets",
        "wikipedia.consolidate_insights",
    ]

    @pytest.mark.parametrize("dotpath", PRIMARY_PROMPTS)
    def test_primary_summaries_resolve(self, dotpath):
        entry = get_prompt_entry(dotpath)
        assert "prompt" in entry
        assert len(entry["prompt"]) > 50

    @pytest.mark.parametrize("dotpath", AUDIO_PROMPTS)
    def test_audio_prompts_resolve(self, dotpath):
        entry = get_prompt_entry(dotpath)
        assert "prompt" in entry

    @pytest.mark.parametrize("dotpath", STRUCTURED_PROMPTS)
    def test_structured_pipeline_resolves(self, dotpath):
        entry = get_prompt_entry(dotpath)
        assert "prompt" in entry

    @pytest.mark.parametrize("dotpath", CHUNKED_PROMPTS)
    def test_chunked_prompts_resolve(self, dotpath):
        entry = get_prompt_entry(dotpath)
        assert "prompt" in entry

    @pytest.mark.parametrize("dotpath", AUDIO_ON_DEMAND)
    def test_audio_on_demand_resolves(self, dotpath):
        entry = get_prompt_entry(dotpath)
        assert "prompt" in entry

    @pytest.mark.parametrize("dotpath", WIKI_PROMPTS)
    def test_wiki_prompts_resolve(self, dotpath):
        entry = get_prompt_entry(dotpath)
        assert "prompt" in entry

    def test_headline_resolves(self):
        entry = get_prompt_entry("headline")
        assert "prompt" in entry


# ---------------------------------------------------------------------------
# Render tests — representative prompts render correctly
# ---------------------------------------------------------------------------


class TestRenderParity:
    """Verify representative prompt renders produce expected output."""

    def test_comprehensive_render(self):
        result = render_prompt("primary_summaries.comprehensive", SAMPLE_VARS)
        # Should contain context header
        assert "Title: Test Video Title" in result
        # Should contain prompt instructions
        assert "comprehensive summary" in result.lower()
        assert "Bottom line" in result
        # Should contain language instruction
        assert "Respond in English" in result

    def test_audio_narration_render(self):
        result = render_prompt("audio_prompts.audio_narration", SAMPLE_VARS)
        assert "text-to-speech" in result.lower() or "text‑to‑speech" in result
        assert "Bottom line" in result

    def test_audio_current_render(self):
        result = render_prompt_only(
            "audio_on_demand.audio_current",
            {"source_text": "This is the source text to narrate."},
        )
        assert "source text to narrate" in result
        assert "Bottom line" in result

    def test_audio_briefing_render(self):
        result = render_prompt_only(
            "audio_on_demand.audio_briefing",
            {"source_text": "Summary content. Research findings."},
        )
        assert "news briefing" in result.lower()
        assert "Summary content" in result

    def test_planner_render(self):
        result = render_prompt_only(
            "structured_pipeline.planner",
            {"max_sections": "5", "transcript_excerpt": "Transcript excerpt here."},
        )
        assert "5 sections" in result
        assert "Transcript excerpt" in result

    def test_extract_bullet_points_render(self):
        result = render_prompt_only(
            "chunked_extractors.extract_bullet_points",
            {"summary": "This is a combined summary with facts."},
        )
        assert "skim" in result.lower() or "bullet" in result.lower()

    def test_condense_for_tts_render(self):
        result = render_prompt_only(
            "audio_prompts.condense_for_tts",
            {
                "text_length": "5000",
                "target_words": "550",
                "target_chars": "3900",
                "text": "Original summary text here that is very long.",
            },
        )
        assert "5000" in result
        assert "550" in result
        assert "Original summary text" in result

    def test_headline_render(self):
        result = render_prompt_only(
            "headline",
            {
                "lang_instruction": "",
                "title": "Test Video",
                "summary_excerpt": "Key points from the video.",
            },
        )
        assert "12" in result  # "12-16 words"
        assert "Test Video" in result


# ---------------------------------------------------------------------------
# LLM config tests
# ---------------------------------------------------------------------------


class TestLLMConfig:
    def test_config_returns_dict(self):
        config = get_llm_config("primary_summaries.comprehensive")
        assert "model" in config
        assert "max_tokens" in config
        assert "temperature" in config

    def test_config_unknown_prompt_returns_defaults(self):
        config = get_llm_config("nonexistent.prompt")
        assert config["model"] is None

    def test_list_prompts(self):
        prompts = list_prompts()
        assert "primary_summaries" in prompts
        assert "comprehensive" in prompts["primary_summaries"]


# ---------------------------------------------------------------------------
# JSON integrity tests
# ---------------------------------------------------------------------------


class TestJsonIntegrity:
    """Verify the JSON file itself is well-formed and complete."""

    def test_json_loads(self):
        prompts = _load_prompts()
        assert isinstance(prompts, dict)

    def test_shared_context_blocks_exist(self):
        prompts = _load_prompts()
        shared = prompts["shared_context"]
        assert "base_context" in shared
        assert "audio_base_context" in shared
        assert "language_instruction" in shared
        assert "reddit_context" in shared

    def test_reddit_context_template(self):
        prompts = _load_prompts()
        template = prompts["shared_context"]["reddit_context"]["template"]
        assert "{subreddit}" in template
        assert "{transcript}" in template

    def test_all_primary_prompts_have_context(self):
        prompts = _load_prompts()
        for name, entry in prompts["primary_summaries"].items():
            if name.startswith("_"):
                continue
            assert "context" in entry, f"primary_summaries.{name} missing 'context'"
            assert "prompt" in entry, f"primary_summaries.{name} missing 'prompt'"

    def test_no_curly_slice_patterns(self):
        """Ensure no {source_text[:4000]} patterns remain in prompts."""
        prompts = _load_prompts()
        for section_name, section in prompts.items():
            if section_name.startswith("_"):
                continue
            if not isinstance(section, dict):
                continue
            for name, entry in section.items():
                if not isinstance(entry, dict) or "prompt" not in entry:
                    continue
                assert "[:" not in entry["prompt"], \
                    f"{section_name}.{name} still has Python slice syntax in prompt"
