# Chapter-Aware Summaries Integration Plan

## Current State
- `youtube_summarizer.extract_transcript()` now returns chapter metadata (`metadata["chapters"]` and top-level `result["chapters"]`).
- `tools/chapters_debug.py` can be run inside the container (`python -m tools.chapters_debug --url <YouTube URL>`) to inspect chapter alignment and preview text.

## Objectives
1. Use YouTube chapters (when available) to drive per-segment summarization.
2. Maintain the existing transcript chunking path for videos without reliable chapter markers.
3. Support both local (Ollama) and cloud providers within the updated flow.

## High-Level Flow
1. **Chapter Detection**
   - After `extract_transcript`, check `metadata.get("chapters")`.
   - Require a minimum count (e.g., ≥2) and filter out low-quality markers (`"<Untitled>"`, duplicates).
   - If the list is unusable, fall back to the original chunking approach.

2. **Transcript Alignment**
   - Use the transcript snippet list (from `YouTubeTranscriptApi`) to assemble text segments per chapter (start/end window).
   - Cache chapter snippets in the JSON report (e.g., `summary.chapters[] = {title, start, end, text}`) for downstream reuse.

3. **LLM Calls**
   - **Per Chapter:** invoke the existing summary prompt (Key Points, Insights, etc.) for each chapter. Include chapter title + timestamp context.
   - **Global Synthesis:** run one final prompt that ingests the chapter-level outputs and produces overarching insights/headlines.
   - Allow configuration to batch tiny chapters or skip per-chapter calls when the count is excessive.

4. **Prompt Adjustments**
   - Chapter prompt template: “Summarize chapter ‘{title}’ ({start} → {end})… produce X bullets + follow-up question.”
   - Global prompt: “Given these chapter summaries, provide global insights/headline/next steps.”
   - Reuse prompt structure across providers; no additional UX changes required.

5. **Provider Handling**
   - Default per-chapter calls to Ollama (cheap) and use the existing provider picker UI for the final synthesis.
   - Keep the ability to choose cloud providers for any stage if the user prefers (the picker already supports this).
   - Optional: add a configuration flag to force single-pass summarization when needed.

6. **Output & Rendering**
   - Store chapter-level outputs in the JSON report (`summary_variants[].chapters`).
   - Update Telegram/HTML renderers to show chapter headings with their bullet points when available.
   - Fall back to existing summary layout if chapters are absent.

7. **Edge Cases & Fallbacks**
   - Videos without chapters or with unreliable markers → revert to transcript chunking.
   - Duplicate chapter titles → prefix with timestamps or auto-normalize.
   - Transcript snippet gaps/failures → skip the chapter and log (with a final warning for the user if many chapters fail).

8. **Testing**
   - Use `tools/chapters_debug.py` to verify chapter detection and preview text.
   - Add regression coverage (manual or automated) that runs the new flow on a chapter-rich video (e.g., `ogfYd705cRs`) and asserts chapter outputs exist.
   - Confirm fallback behavior on videos without chapters to ensure parity with current results.

## Next Steps
1. Implement chapter-aware summarizer helper (transcript alignment and per-chapter calls).
2. Integrate into `summary_service.process_content_summary` with provider/model selection.
3. Update renderers and JSON outputs with chapter detail.
4. Add tests/documentation and QA on multiple video types before rollout.

