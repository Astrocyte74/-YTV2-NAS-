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

    async def test_long_chapter_is_split_merged_and_lightly_combined(self):
        summarizer = self._make_summarizer()
        operations = []

        async def fake_llm_call(messages, operation_name="LLM call", max_retries=3):
            prompt = messages[0].content
            operations.append((operation_name, prompt))
            if operation_name.startswith("chapter summary part"):
                return f"Overview: {operation_name}\n• detail from {operation_name}"
            if operation_name == "chapter summary merge":
                return (
                    "Overview: merged chapter\n"
                    "• merged fact one\n"
                    "• merged fact two\n"
                    "• merged fact three\n"
                    "• merged fact four\n"
                    "• merged fact five\n"
                    "• merged fact six"
                )
            if operation_name == "chapter summary combine":
                self.assertIn("editing a long-form chapter digest", prompt)
                self.assertIn("Keep every chapter heading exactly as written", prompt)
                self.assertIn("Under each chapter heading, keep 4-8 bullets", prompt)
                return "**Big Chapter (00:00-01:40)**\n- merged fact one\n\nBottom line: done."
            raise AssertionError(f"unexpected operation {operation_name}")

        async def fake_headline(summary, metadata):
            return "headline"

        summarizer._robust_llm_call = fake_llm_call
        summarizer._generate_headline_from_summary = fake_headline

        long_text = ("Alpha sentence. " * 1000) + ("Beta sentence. " * 1000)
        result = await summarizer._summarize_by_chapters(
            [
                {
                    "title": "Big Chapter",
                    "start": 0,
                    "end": 100,
                    "text": long_text,
                }
            ],
            {"title": "Rosenvall Test", "uploader": "Test Channel"},
            "comprehensive",
            "en",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["headline"], "headline")
        self.assertEqual(result["summary_plan"]["combine_style"], "light-edit")
        self.assertEqual(result["summary_plan"]["split_chapter_count"], 1)
        self.assertEqual(result["chapter_summaries"][0]["title"], "Big Chapter")
        self.assertIn("merged chapter", result["chapter_summaries"][0]["summary"])

        op_names = [name for name, _prompt in operations]
        self.assertIn("chapter summary merge", op_names)
        self.assertIn("chapter summary combine", op_names)
        self.assertGreaterEqual(sum(1 for name in op_names if name.startswith("chapter summary part")), 2)

    async def test_key_insights_skips_final_combine_and_stitches_chapters(self):
        summarizer = self._make_summarizer()
        operations = []

        async def fake_llm_call(messages, operation_name="LLM call", max_retries=3):
            operations.append(operation_name)
            if operation_name.startswith("chapter summary"):
                return "Overview: chapter summary\n• retained detail"
            if operation_name == "chapter digest framing":
                return '{"intro":"This video develops a geography theory across scripture, travel, climate, and mapping evidence.","conclusion":"Taken together, the chapter evidence is presented as support for the proposed Baja setting."}'
            raise AssertionError(f"unexpected operation {operation_name}")

        async def fake_headline(summary, metadata):
            return "headline"

        summarizer._robust_llm_call = fake_llm_call
        summarizer._generate_headline_from_summary = fake_headline

        result = await summarizer._summarize_by_chapters(
            [
                {
                    "title": "Intro",
                    "start": 0,
                    "end": 50,
                    "text": "Alpha sentence. " * 40,
                },
                {
                    "title": "Second",
                    "start": 50,
                    "end": 100,
                    "text": "Beta sentence. " * 40,
                },
            ],
            {"title": "Rosenvall Test", "uploader": "Test Channel"},
            "key-insights",
            "en",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["headline"], "headline")
        self.assertEqual(result["summary_plan"]["combine_style"], "chapter-stitch")
        self.assertFalse(result["summary_plan"]["combine_fallback_used"])
        self.assertTrue(result["summary_plan"]["frame_generated"])
        self.assertIn("Overview: This video develops a geography theory", result["summary"])
        self.assertIn("**Intro", result["summary"])
        self.assertIn("Bottom line: Taken together, the chapter evidence is presented as support for the proposed Baja setting.", result["summary"])
        self.assertNotIn("chapter summary combine", operations)
        self.assertIn("chapter digest framing", operations)

    async def test_failed_chapter_parts_fall_back_instead_of_disappearing(self):
        summarizer = self._make_summarizer()

        async def fake_llm_call(messages, operation_name="LLM call", max_retries=3):
            if operation_name.startswith("chapter summary part"):
                return None
            if operation_name.endswith("basic fallback"):
                return None
            if operation_name == "chapter summary merge":
                return None
            if operation_name == "chapter digest framing":
                return None
            if operation_name == "chapter summary combine":
                return None
            raise AssertionError(f"unexpected operation {operation_name}")

        async def fake_headline(summary, metadata):
            return "headline"

        summarizer._robust_llm_call = fake_llm_call
        summarizer._generate_headline_from_summary = fake_headline

        long_text = ("Alpha sentence. " * 1000) + ("Beta sentence. " * 1000)
        result = await summarizer._summarize_by_chapters(
            [
                {
                    "title": "Big Chapter",
                    "start": 0,
                    "end": 100,
                    "text": long_text,
                }
            ],
            {"title": "Rosenvall Test", "uploader": "Test Channel"},
            "key-insights",
            "en",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["headline"], "headline")
        self.assertEqual(result["summary_plan"]["split_chapter_count"], 1)
        self.assertEqual(result["summary_plan"]["chapter_fallback_count"], 1)
        self.assertEqual(result["summary_plan"]["chapter_fallback_titles"], ["Big Chapter"])
        self.assertEqual(result["summary_plan"]["combine_style"], "chapter-stitch")
        self.assertFalse(result["summary_plan"]["frame_generated"])
        self.assertEqual(len(result["chapter_summaries"]), 1)
        self.assertIn("Big Chapter", result["summary"])
        self.assertNotIn("Bottom line:", result["summary"])
        self.assertIn("preserves the main claims, examples, and evidence", result["chapter_summaries"][0]["summary"])

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
