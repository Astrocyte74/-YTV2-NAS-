# Research Architecture Notes

This document explains the main design choices in the portable research stack.
It is meant to travel with the extracted service so another app can adopt it without reverse-engineering the behavior.

## Planner vs synthesis providers

The stack supports separate provider selection for:

- `RESEARCH_PLANNER_PROVIDER`
- `RESEARCH_SYNTH_PROVIDER`

Why the split exists:

- planning and synthesis turned out to have different strengths
- Gemini was stronger at query planning, especially for side-by-side comparisons
- Mercury-2 was stronger at final formatting and compact report-style output

Observed behavior during testing:

- Mercury sometimes narrowed comparison prompts too aggressively
- Gemini more reliably produced broader, better comparison queries
- Mercury more often produced tables and cleaner final report structure

That is why the preferred hybrid mode became:

```env
RESEARCH_PLANNER_PROVIDER=openrouter
RESEARCH_SYNTH_PROVIDER=inception
```

This means:

- Gemini plans the queries
- Mercury synthesizes the final answer

## Why Brave is throttled aggressively

Brave rate-limited during real testing, especially when prompts fanned out into multiple queries or multiple tool calls.

The current settings are conservative safety rails, not a formal statement of Brave's contract:

- `BRAVE_MIN_INTERVAL_SECONDS=1.10`
- `BRAVE_RETRY_429_DELAY_SECONDS=1.50`
- query/request caps per depth level

The goal is to avoid bursts caused by:

- multiple planned queries
- compare prompts
- `news + web` request combinations

If Brave still rate-limits and yields no useful results, the executor falls back to Tavily.

## Source quality heuristics

Source quality filtering is heuristic and manually curated.

Examples:

- low-quality social domains are deprioritized or filtered when stronger sources exist
- community sources like Reddit and YouTube are allowed for review/sentiment-style prompts
- a small set of expert-review domains is explicitly recognized

This is a practical general-web policy, not a universal truth.

For domain-specific deployments like legal, medical, or finance, these lists should probably become configurable.

## What synthesis continuations do

Continuations are not search retries.
They are a mechanism to recover from truncated LLM output during final synthesis.

Continuation is triggered when:

- the provider returns `finish_reason == "length"`, or
- the output appears truncated heuristically

The heuristic checks for things like:

- unfinished trailing punctuation
- broken table shape
- suspiciously incomplete final lines

When triggered, the system asks the model to continue exactly where it left off and avoid duplicating the final `Sources` section.

Related settings:

- `SYNTH_MAX_CONTINUATIONS`
- `SYNTH_CONTINUATION_MAX_TOKENS`

## Retry and fallback behavior

Synthesis retry happens at the single LLM-call level.
It does not rerun the full research pipeline.

That means:

- search results are reused
- evidence stays the same
- only the synthesis call is retried

Current behavior:

1. try the configured synthesis provider
2. retry that provider up to `RESEARCH_SYNTH_RETRY_ATTEMPTS`
3. if synthesis is pinned to Inception and still fails, try OpenRouter synthesis when fallback is enabled
4. if that also fails, use deterministic synthesis

Planner fallback is separate:

- when planning fails, the system falls back to a simple deterministic plan rather than failing the whole run

## What deterministic synthesis means

Deterministic synthesis is the non-LLM emergency fallback.

It constructs a grounded but plain answer directly from the retrieved batches:

- summary heading
- top findings
- source URLs

It is much less polished than Mercury or Gemini, but it preserves graceful degradation and keeps the result source-grounded.

## Tavily research mode vs regular Tavily search

Regular Tavily search:

- fast
- returns ranked results/snippets
- good default retrieval primitive

Tavily `research` mode:

- async and polled
- slower
- intended for broader or more complex research tasks
- can be higher quality for deep source-backed prompts
- can hit plan/account limits

Because of that, the executor only prefers Tavily `research` in deeper or broader cases.
If Tavily research is unavailable or hits plan limits, the stack falls back to Tavily `search + extract`.

## Practical recommendation

For the current general-web research use case:

- use Gemini planning when prompt quality matters
- use Mercury synthesis when output format matters
- keep Brave throttling conservative
- keep deterministic fallback available even if it is rarely used

If this stack is moved into a very different domain, the first things to revisit are:

- source quality heuristics
- provider/tool routing rules
- default provider split
