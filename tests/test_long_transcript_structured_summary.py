import sys
import types
import unittest


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _DummyLLM:
    def __init__(self, *args, **kwargs):
        pass


class _DummyMessage:
    def __init__(self, content: str = ""):
        self.content = content


_stub_module("yt_dlp")
_stub_module("langchain_openai", ChatOpenAI=_DummyLLM)
_stub_module("langchain_anthropic", ChatAnthropic=_DummyLLM)
_stub_module("langchain_community")
_stub_module("langchain_community.llms", Ollama=_DummyLLM)
_stub_module("langchain_community.chat_models", ChatOllama=_DummyLLM)
_stub_module("langchain_core")
_stub_module(
    "langchain_core.messages",
    HumanMessage=_DummyMessage,
    SystemMessage=_DummyMessage,
)

from youtube_summarizer import (  # noqa: E402
    STRUCTURED_PLANNER_EXCERPT_CHARS,
    YouTubeSummarizer,
)


class StructuredSummaryLongTranscriptTests(unittest.IsolatedAsyncioTestCase):
    def _make_summarizer(self):
        summarizer = object.__new__(YouTubeSummarizer)
        summarizer.status_callback = None
        return summarizer

    async def test_builds_chapter_slices_from_existing_transcript_segments(self):
        summarizer = self._make_summarizer()
        captured = {}

        async def fake_summarize_by_chapters(slices, metadata, summary_type, transcript_language):
            captured["slices"] = slices
            return {"summary": "ok"}

        async def fail_plan(*args, **kwargs):
            raise AssertionError("planner path should not run when chapter slices can be built")

        summarizer._summarize_by_chapters = fake_summarize_by_chapters
        summarizer._plan_sections_from_transcript = fail_plan
        summarizer._summarize_with_plan = fail_plan

        transcript = "intro section next section"
        metadata = {
            "chapters": [
                {"title": "Intro", "start_time": 0.0, "end_time": 5.0},
                {"title": "Next", "start_time": 5.0, "end_time": 10.0},
            ]
        }
        transcript_segments = [
            {"text": "Intro bit", "start": 0.0, "duration": 2.0},
            {"text": "More intro", "start": 3.0, "duration": 1.0},
            {"text": "Next section", "start": 6.0, "duration": 2.0},
        ]

        result = await summarizer._generate_structured_summary(
            transcript,
            metadata,
            "key-insights",
            chapter_slices=[],
            transcript_segments=transcript_segments,
            transcript_language="en",
        )

        self.assertEqual(result["summary"], "ok")
        self.assertIn("chapter_slices", result)
        self.assertEqual([item["title"] for item in captured["slices"]], ["Intro", "Next"])
        self.assertEqual(captured["slices"][0]["text"], "Intro bit More intro")
        self.assertEqual(captured["slices"][1]["text"], "Next section")

    async def test_long_transcript_without_chapter_slices_falls_back_to_classic_pipeline(self):
        summarizer = self._make_summarizer()

        async def fail_if_called(*args, **kwargs):
            raise AssertionError("planner path should not run for long transcripts without chapter slices")

        async def no_chapter_summary(*args, **kwargs):
            return None

        summarizer._summarize_by_chapters = no_chapter_summary
        summarizer._plan_sections_from_transcript = fail_if_called
        summarizer._summarize_with_plan = fail_if_called

        transcript = "A" * (STRUCTURED_PLANNER_EXCERPT_CHARS + 100)

        result = await summarizer._generate_structured_summary(
            transcript,
            {},
            "key-insights",
            chapter_slices=[],
            transcript_segments=[],
            transcript_language="en",
        )

        self.assertIsNone(result)
