# Dashboard16 Deep Research Strategy

This note documents the strategy used to ship Deep Research in the active `dashboard16` web UI.

## Summary

The live dashboard integration does **not** currently mount the portable React reader from this folder.

Instead, it uses a **server-rendered HTML strategy**:

1. canonical deep-research results are stored in PostgreSQL
2. the dashboard Postgres index enriches the `deep-research` summary variant with canonical run data
3. Python renders a dedicated Deep Research HTML view on the server
4. the existing dashboard variant switcher displays that HTML like any other summary variant

This was chosen because `dashboard16` is already built around server-rendered variant HTML and vanilla JS, while this package expects a React render path and markdown tooling that the active dashboard does not currently use.

## Canonical Data Model

Canonical storage lives in `backend/ytv2_api/follow_up_store.py` and the follow-up migration.

Important tables/fields:

- `follow_up_research_runs`
- `research_response` (`TEXT`): full deep-research answer markdown/text
- `research_meta` (`JSONB`): structured metadata
- `coverage_map` (`JSONB`)
- `approved_questions` (`TEXT[]`)
- `planned_queries` (`TEXT[]`)
- `summaries`
- `variant='deep-research'`: lightweight reference row for UI tab discovery

The `summaries` row is not the canonical result. It is only how the dashboard discovers that a Deep Research tab exists.

## Dashboard16 Strategy

Implementation lives primarily in:

- `dashboard16/modules/postgres_content_index.py`
- `dashboard16/static/dashboard_v3.js`
- `dashboard16/static/shared.css`

The flow is:

1. query normal summary variants from `v_latest_summaries`
2. when a variant is `deep-research`, join to `follow_up_research_runs`
3. pull:
   - `research_response`
   - `research_meta`
   - `follow_up_run_id`
4. convert the canonical research answer into structured HTML on the server
5. return that rendered HTML inside `summary_variants[].html`
6. let the existing dashboard variant controls render it with no special frontend fetch

That means report pages, wall mode, and the modal reader all get Deep Research through the same existing variant mechanism.

## Why This Path Was Used

This was the fastest low-risk path because:

- `dashboard16` already expects per-variant HTML
- `dashboard16` is not currently set up to import this React package
- the active dashboard does not already ship a markdown rendering dependency
- we wanted Deep Research live without introducing a frontend build migration first

In short: we reused the dashboard’s current architecture instead of forcing a React integration into a non-React surface.

## What The Server Renderer Does

The Python renderer mirrors the same broad product intent as the React reader:

- extract report title
- extract notice text
- extract a lead summary paragraph
- render headings, lists, quotes, code blocks, and markdown tables
- show research metadata chips
- show approved research questions
- show source links

It also marks the block with:

- `data-research-report="1"`

That marker lets dashboard JS skip the legacy summary post-processing step that would otherwise mutate the research markup.

## Relationship To This React Package

This package remains the cleaner long-term portable renderer.

Current state:

- this package = React-based portable reader
- `dashboard16` = server-rendered HTML implementation using the same canonical research data

So the two renderers currently share:

- the same source-of-truth data
- similar section semantics
- similar visual goals

But they do **not** currently share code.

## Why The React Reader Was Not Wired Directly

Direct use of this package in `dashboard16` would have required one of:

- adding a React mount point and bundle/build step to the active dashboard
- loading React and markdown dependencies into the current dashboard runtime
- restructuring the reader flow around client-side rendering instead of prebuilt variant HTML

That is feasible, but it is a larger architecture change than was needed to launch Deep Research.

## Future Convergence Options

There are three reasonable future paths:

1. Keep the current server-rendered dashboard path for Deep Research only.
2. Make `dashboard16` consume this React package directly once a React mount/build path exists.
3. Gradually standardize regular summaries around the same structured rendering model, then unify on one renderer.

Recommended order:

1. validate Deep Research UX in production
2. refine styling/content structure as needed
3. decide later whether to migrate the dashboard toward the React reader

## Practical Takeaway

The current Deep Research UI is:

- canonical-data-backed
- compatible with existing dashboard variant switching
- independent of a React build pipeline
- a tactical implementation, not a rejection of this package

This folder still represents the likely long-term portable rendering direction. The live dashboard implementation simply took the shortest reliable path to production.
