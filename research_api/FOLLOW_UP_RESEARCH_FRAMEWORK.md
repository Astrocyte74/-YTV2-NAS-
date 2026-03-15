# Follow-Up Research Framework

This document describes a suggested integration model for adding optional deep research to the Intel app's existing summarization flow.

The main goal is:

- keep normal summaries fast and clean
- offer high-value research follow-ups when useful
- avoid redundant multi-run planning/search overlap
- preserve the strong report-style reader output as a separate artifact

## Product principle

Do not treat deep research as part of the base summary.

Instead:

1. generate the normal summary first
2. generate suggested follow-up research directions
3. let the user approve, edit, or ignore them
4. run one coordinated deep-research job from the approved set
5. display the result in its own tab/variant

This keeps a clear separation between:

- what the source content said
- what current web research says now

## Key difference from the current research flow

Current standalone research flow:

- user asks one research question
- planner generates internal search queries immediately
- system executes them without an approval step

Proposed Intel app integration:

- user asks for a summary
- system proposes follow-up research questions
- user approves one or more of those questions
- only then does the planner generate execution queries

Important distinction:

- user approves the research direction
- user does not need to approve the raw internal Brave/Tavily query strings

## Recommended UX

After a summary is generated, optionally show:

- `Deep Research`
- a small set of suggested follow-up questions
- checkboxes or chips
- optional custom query field

Suggested UI pattern:

```text
[Key Insights] [Comprehensive] [Deep Research]

Suggested follow-up research
[x] What changed since this content was published?
[ ] How does Cursor compare to Windsurf and Copilot today?
[ ] What is the current pricing / version / availability?
[+] Custom query

[Run Research]
```

Telegram pattern:

- send summary first
- then offer suggested follow-up buttons/chips
- then run research only after confirmation

## Do not run each approved suggestion independently

This is the main architectural rule.

Bad approach:

- approved question A -> full research run
- approved question B -> full research run
- approved question C -> full research run

Problems:

- duplicated search intent
- overlapping queries
- wasted provider usage
- repetitive outputs

Recommended approach:

- selected questions become one coordinated follow-up research request
- planner dedupes and consolidates them into one shared query plan
- executor runs the consolidated plan once
- synthesizer answers the approved questions in separate sections

## Suggested pipeline

### Stage 1: Base summary

Existing system behavior:

- summarize YouTube / article / Reddit / website
- store summary result

### Stage 2: Follow-up suggestion generation

Use the planner model as a suggestion engine.

Inputs should include more than just the final summary:

- source title
- source URL/domain
- publish date if known
- content type
- extracted entities/topics
- summary text
- optional short body/transcript excerpt if cheap enough

Output should be structured suggestions, not just strings.

Suggested shape:

```json
{
  "suggestions": [
    {
      "id": "pricing-current",
      "label": "What is Cursor's current pricing and free tier?",
      "query": "What is Cursor's current pricing and free tier in 2026?",
      "reason": "The source discusses Cursor, and pricing changes frequently.",
      "kind": "pricing",
      "priority": 0.93,
      "default_selected": true
    }
  ]
}
```

Suggested `kind` values:

- `current_state`
- `pricing`
- `comparison`
- `alternatives`
- `fact_check`
- `background`
- `what_changed`

## Suggestion-generation rules

Do not suggest research for every summary.

Suggested triggers:

- content appears time-sensitive
- content mentions products, companies, models, pricing, releases, versions
- content is older than 3-6 months
- content is clearly comparative or review-like
- user explicitly wants current context

Suggested non-triggers:

- evergreen educational content with no changing factual surface
- purely personal/opinion content unless fact-check mode is requested
- already-current, already-source-heavy research artifacts

## Selection rules

Recommended defaults:

- generate 3 follow-up suggestions
- preselect only 1 best suggestion
- allow multi-select up to 3
- allow custom query entry

Avoid:

- auto-selecting everything
- generating large suggestion lists

## Coordinated planning model

When the user selects multiple follow-up questions, do one consolidated planning pass.

Input shape:

```json
{
  "mode": "follow_up_research",
  "source_context": {
    "content_type": "youtube",
    "title": "Cursor Review",
    "url": "https://youtube.com/...",
    "published_at": "2025-09-10"
  },
  "summary": "...",
  "approved_questions": [
    "What is Cursor's current pricing and free tier?",
    "How does Cursor compare to Windsurf and Copilot today?"
  ]
}
```

Planner instructions should explicitly say:

- consolidate these approved questions into one minimal non-overlapping research plan
- dedupe overlapping search intents
- preserve coverage of every approved question
- keep the final query set compact

Recommended limits:

- maximum approved questions: 3
- maximum planned search queries after consolidation: 4-6

## Execution model

Recommended:

- one research run per approved batch
- shared deduped plan
- one synthesis output with sectioned answers

Possible backend extension:

- support `queries_override: string[]`
- or add a dedicated `follow_up_research(...)` entry point

Avoid using:

- `"; ".join(selected_queries)` as the primary interface

That is too lossy and encourages muddled planning.

## Synthesis format

The final report should answer the approved questions explicitly.

Recommended structure:

1. title
2. executive summary
3. section per approved question
4. optional comparison table if relevant
5. sources

This works well with the existing research reader renderer.

## Storage model

Store these as separate but linked artifacts.

Suggested records:

### Summary record

- `summary_id`
- `source_type`
- `source_url`
- `source_title`
- `summary_variant`
- `summary_content`

### Follow-up suggestion record

- `summary_id`
- `suggestions[]`
- `generated_at`
- `planner_model`

### Research run record

- `summary_id`
- `selected_suggestion_ids[]`
- `custom_queries[]`
- `approved_questions[]`
- `research_response`
- `research_meta`
- `cache_key`
- `created_at`

## Caching

Caching is important.

Cache key should include:

- normalized approved question set
- source URL or source identity
- provider mode
- depth
- compare flag
- stage override config if relevant

Reasons:

- users may re-run the same follow-up with a different tab selection
- Telegram and web UI may request the same follow-up
- expensive overlap should be avoided

## Suggested MVP

Start with two optional follow-up types:

1. `What changed since this content was published?`
2. `How does this compare to alternatives today?`

Why these first:

- easy for users to understand
- high value for time-sensitive tech/product content
- leverage the current Gemini-planner + Mercury-synth strengths

## Mapping to current research stack

Use the current hybrid research setup as-is:

- planner: Gemini via OpenRouter
- synthesis: Mercury via Inception

This is a good fit because:

- Gemini is better at follow-up/query planning
- Mercury is better at final report/table formatting

## Reader integration

Do not blend deep research into the base summary tab.

Recommended:

- new `Deep Research` tab / variant
- render with the portable React reader bundle
- preserve markdown tables and report formatting

Current portable renderer lives in:

- `backend/research_reader_react` on the Intel app machine

## Suggested implementation phases

### Phase 1

- generate 3 follow-up suggestions after summary
- allow user selection
- support one selected suggestion only
- render result in `Deep Research` tab

### Phase 2

- allow multi-select up to 3
- add consolidated follow-up planning
- cache results

### Phase 3

- Telegram button/chip workflow
- custom query support
- richer suggestion typing and smarter defaults

## Guardrails

- never auto-run deep research by default
- never silently mix research facts into the original summary
- never execute each selected suggestion as a fully independent research run unless explicitly intended
- keep the user approval step at the research-direction level, not raw search-query level

## Short recommendation

Best overall pattern:

- summary first
- suggestions second
- user approval third
- one deduped coordinated research run
- separate research tab with the reader renderer

That gives you the best balance of:

- control
- clarity
- quality
- cost containment
