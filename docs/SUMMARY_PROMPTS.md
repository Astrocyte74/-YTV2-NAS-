# Summarization Prompts — Reference and Refinements

This document catalogs the active summarization prompts, where they live, and suggested refinements to improve factuality, structure, and rendering in the dashboard inline reader.

## Locations (source of truth)
- Prompts (YouTube + web transcripts): `youtube_summarizer.py:1367`
- Base context (prepended to every prompt): `youtube_summarizer.py:1355`
- Headline generator: `youtube_summarizer.py:1536`

## Base Context (always prepended)
- Injects: Title, Channel, Upload Date, Duration, URL, then Full Transcript
- File: `youtube_summarizer.py:1355`

## Categories (current behavior)

- Comprehensive — `youtube_summarizer.py:1367`
  - Output: Overview (2–3 sentences) → 2–4 short section headings (2–4 words) each with 3–5 “• ” bullets (≤18 words) → “Bottom line: …”
  - Rules: Transcript language; timestamps only if explicitly present; use “Unknown” when needed; no code fences/emojis/markdown headings; length guidance by duration

- Audio (TTS narration) — `youtube_summarizer.py:1392`
  - Output: Opening (2–3 sentences) → Main (conversational transitions; no headings/bullets) → “Bottom line: …” (~180–380 words)
  - Rules: Transcript language; accurate names/numbers; no headings/bullets/code fences/emojis

- Bullet‑Points — `youtube_summarizer.py:1409`
  - Output: One‑sentence overview → 10–16 “• ” bullets → “Bottom line: …”
  - Rules: Transcript language; ≤18 words/bullet; prefer specifics; deduplicate; timestamps only if present; no code fences/emojis/headings

- Key‑Insights — `youtube_summarizer.py:1428`
  - Output: 3–5 short thematic headings → for each, 3–5 “• ” bullets with concrete facts/results/names/metrics
  - Rules: Transcript language; ≤18 words/bullet; no speculation; “Unknown” when missing; no code fences/emojis; no final “Bottom line”

- Executive — `youtube_summarizer.py:1445`
  - Output: EXECUTIVE SUMMARY → PART 1/2/(3): Overview, Key Points (bullets), Conclusion → STRATEGIC RECOMMENDATIONS
  - Guidelines: 2–4 logical parts; analytical tone; timestamps as helpful; focus on implications; adapt if needed; British/Canadian spelling; no speculation

- Adaptive — `youtube_summarizer.py:1501`
  - Choices: Comprehensive; Key Points; Key Insights; or step‑wise (6–12 steps, ≤12 words/step)
  - Global rules: Transcript language; timestamps only if present; use “Unknown”; no code fences/emojis; Key formats use “• ” bullets; step‑wise uses numbered steps; length guidance

- Headline — `youtube_summarizer.py:1536`
  - Output: One headline (12–16 words; no emojis; no colon); must match content language

## General refinements (apply to all)
Add these lines just before each category’s Rules section:

1) Extraction bias
- “Summarize using only information explicitly stated in the transcript; never infer causes or speculate.”

2) Hierarchical compression
- “Prefer short paraphrases of full ideas rather than skipping them entirely.”

3) Entity handling
- “When names, numbers, or organizations are unclear, use ‘Unknown’ rather than guessing.”

4) Temporal grounding
- “Keep events in the original chronological order unless a thematic grouping is requested.”

5) Clarity pass
- “Rewrite for clarity and natural flow after compressing, without adding new meaning.”

## Category‑specific tweaks

- Comprehensive
  - Add: “Section titles must summarize the key phase or theme (not full sentences). Each bullet must be factual and non‑redundant.”

- Audio (TTS)
  - Add: “Use smooth spoken transitions (‘First…’, ‘Next…’, ‘However…’, ‘Finally…’) and vary sentence length for natural rhythm.”
  - Add: “Avoid list‑like phrasing or enumeration; use implicit transitions instead.”

- Bullet‑Points
  - Replace “prefer specifics” with: “Prefer named entities, figures, and actions over general statements.”
  - Add: “If the transcript contains comparisons, include at least one bullet that explicitly states the contrast.”

- Key‑Insights
  - Add: “Use consistent tone and granularity across categories — each heading should capture a distinct conceptual dimension (e.g., Strategy / Technology / Outcome).”

- Executive
  - Add (before Guidelines): “Write for senior readers scanning quickly. Front‑load major outcomes and implications before elaborating.”
  - Add (in Guidelines): “Ensure each PART ends with a concise synthesis sentence (1–2 clauses).”

- Adaptive
  - Add (in Global rules): “If multiple patterns could apply, select the one maximizing clarity for non‑expert readers.”
  - Add (for procedural): “When choosing step‑wise form, begin with a one‑sentence context statement before numbering.”

- Headline
  - Add constraint: “Lead with the most concrete noun or named entity; avoid starting with vague verbs (‘Exploring…’, ‘Discussing…’).”

## Output conventions to align with NAS HTML formatter
The dashboard injects `summary_html` (produced on NAS). Our current NAS formatter `modules/summary_variants.py:73` converts plain text to minimal HTML. To ensure high‑quality rendering without an extra LLM pass:

- Headings as lines ending with a colon
  - “Main topic:”, “Key points:”, “Takeaway:”, and any short thematic heading ending with “:”
  - Formatter upgrade suggestion: detect `^([A-Z][\w \-/]{2,40}):$` → wrap as `<h3 class="kp-heading">…</h3>`

- Bullets
  - Start bullets with “• ” or “- ”
  - Formatter already maps `•`, `-`, `*` lines to `<ul><li>`

- Bottom line
  - Use a single line starting with “Bottom line: …”
  - Formatter upgrade suggestion: map to `<p class="kp-takeaway">…</p>`

- No code fences, no emojis
  - Keeps markup clean for `.prose` and avoids backtick blocks in HTML

## Proposed prompt insertions (ready to paste)
Below are minimal deltas to add to each prompt.

- General (add before “Rules:” in all categories)
```
Summarize using only information explicitly stated in the transcript; never infer causes or speculate.
Prefer short paraphrases of full ideas rather than skipping them entirely.
When names, numbers, or organizations are unclear, use “Unknown” rather than guessing.
Keep events in the original chronological order unless a thematic grouping is requested.
Rewrite for clarity and natural flow after compressing, without adding new meaning.
```

- Comprehensive (add just before Rules)
```
Section titles must summarize the key phase or theme (not full sentences). Each bullet must be factual and non‑redundant.
```

- Audio (TTS) (add in Structure/Rules)
```
Use smooth spoken transitions (“First…”, “Next…”, “However…”, “Finally…”) and vary sentence length for natural rhythm.
Avoid list‑like phrasing or enumeration; use implicit transitions instead.
```

- Bullet‑Points (modify + add)
```
Prefer named entities, figures, and actions over general statements.
If the transcript contains comparisons, include at least one bullet that explicitly states the contrast.
```

- Key‑Insights (add)
```
Use consistent tone and granularity across categories — each heading should capture a distinct conceptual dimension (e.g., Strategy / Technology / Outcome).
```

- Executive (add)
```
Write for senior readers scanning quickly. Front‑load major outcomes and implications before elaborating.
Ensure each PART ends with a concise synthesis sentence (1–2 clauses).
```

- Adaptive (add)
```
If multiple patterns could apply, select the one maximizing clarity for non‑expert readers.
When choosing step‑wise form, begin with a one‑sentence context statement before numbering.
```

- Headline (add)
```
Lead with the most concrete noun or named entity; avoid starting with vague verbs (“Exploring…”, “Discussing…”).
```

## Implementation notes
- Minimal, low‑risk change: paste the insertions above into each prompt block in `youtube_summarizer.py` (near the listed line numbers). No behavioral code changes required.
- Optional formatter upgrade (NAS): extend `format_summary_html` to:
  - Detect heading‑with‑colon lines → `<h3 class="kp-heading">…</h3>`
  - Map `Bottom line:` paragraph → `<p class="kp-takeaway">…</p>`
  - Continue to escape other content and render `•`/`-`/`*` lists
- If you later prefer Markdown outputs, add a Markdown renderer on the dashboard and allow `##`/`- ` in prompts; keep the above colon‑heading conventions as a safe fallback.

---
Maintainers: Update this file when prompt wording changes so downstream UI/formatter expectations stay in sync.

