# YTV2 Deep Research Integration Plan

*Planning document for integrating follow-up research into YTV2 backend*
*Last updated: 2026-03-15*

---

## Current State Analysis

### Existing Architecture

**Database Schema** (PostgreSQL):
```sql
content (video_id, title, channel_name, canonical_url, published_at, ...)
summaries (id, video_id, variant, text, html, created_at, revision)
v_latest_summaries (view for latest per variant)
```

**Summary Variants** (currently supported):
- `key-insights` (default)
- `comprehensive`
- `bullet-points`
- `executive`
- `adaptive`
- `audio`, `audio-fr:beginner`, `audio-es:intermediate`

**Entry Points**:
1. **Telegram Bot** (`telegram_bot.py` + `modules/telegram_handler.py`)
   - Auto-mode enabled: skips prompts, uses default
   - Manual mode: shows variant keyboard
   - Sources: YouTube, Reddit/Web URLs, Audio

2. **YTV2 API** (`ytv2_api/main.py`)
   - `POST /api/process` - Video processing
   - `POST /api/transcript` - Transcript extraction
   - Port 6453

3. **Dashboard** (Tailscale access)
   - Displays summaries with tab navigation
   - Regenerate functionality
   - Variant switching

### Research API Status

**Location**: `research_api/` - Fully operational

**Components**:
- `research_service/` - Core research logic
  - `planner.py` - Query planning (Gemini)
  - `executor.py` - Search execution (Brave/Tavily)
  - `synthesizer.py` - Report generation (Mercury-2)
  - `service.py` - Public API (`run_research()`)
- `research_reader_react/` - React renderer component
- `app.py` - Flask HTTP wrapper (standalone)

**Current Configuration** (`.env.nas`):
```bash
RESEARCH_ENABLED=true
RESEARCH_PLANNER_PROVIDER=openrouter  # Gemini
RESEARCH_SYNTH_PROVIDER=inception      # Mercury-2
INCEPTION_API_KEY=sk_...
OPENROUTER_API_KEY=sk-or-...
BRAVE_API_KEY=BSA...
TAVILY_API_KEY=tvly-...
```

---

## Proposed Integration Architecture

### Phase 1: Backend Foundation (No UI Changes)

#### 1.1 Add Follow-Up Planning Module

**File**: `research_api/research_service/follow_up.py`

```python
# New module for follow-up research orchestration

MAX_APPROVED_QUESTIONS = 3
MAX_PLANNED_QUERIES = 6

@dataclass
class FollowUpSuggestion:
    id: str
    label: str
    question: str
    reason: str
    kind: Literal["current_state", "pricing", "comparison", "alternatives",
                "fact_check", "background", "what_changed"]
    priority: float
    default_selected: bool

@dataclass
class FollowUpResearchPlan:
    approved_questions: list[str]
    question_provenance: list[str]  # ["suggested", "preset", "custom"]
    planned_queries: list[str]
    coverage_map: list[dict]
    dedupe_notes: str

def plan_follow_up_research(
    *,
    source_context: dict,
    summary: str,
    approved_questions: list[str],
) -> FollowUpResearchPlan:
    """
    Generate consolidated research plan from approved user questions.

    Key: Planner sees ALL approved questions together and generates
    ONE minimal non-overlapping query plan.
    """

def suggest_follow_up_questions(
    *,
    source_context: dict,
    summary: str,
    entities: list[str] = None,
) -> list[FollowUpSuggestion]:
    """
    Generate 3 suggested follow-up questions using planner LLM.
    Only called when content meets suggestion criteria.
    """
```

#### 1.2 Extend Research Service

**File**: `research_api/research_service/service.py`

```python
# Add new entry point

def run_follow_up_research(
    *,
    source_context: dict,
    summary: str,
    approved_questions: list[str],
    question_provenance: list[str] = None,
    provider_mode: str = "auto",
    depth: str = "balanced",
    progress: Callable[[dict], None] = None,
) -> ResearchRunResult:
    """
    Execute coordinated follow-up research for multiple approved questions.

    Returns standard ResearchRunResult with enhanced metadata:
    - response: Markdown report with sections per question
    - sources: Deduped shared source set
    - meta: Includes coverage_map, approved_questions, planned_queries
    """
```

#### 1.3 Database Schema Extensions

**New Table**: `follow_up_suggestions`
```sql
CREATE TABLE follow_up_suggestions (
  id BIGSERIAL PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES content(video_id),
  summary_id BIGINT REFERENCES summaries(id),
  suggestions JSONB NOT NULL,  -- Array of FollowUpSuggestion
  generated_at TIMESTAMPTZ DEFAULT now(),
  planner_provider TEXT,
  planner_model TEXT,
  expires_at TIMESTAMPTZ DEFAULT (now() + INTERVAL '7 days')
);
CREATE INDEX idx_follow_up_suggestions_video ON follow_up_suggestions(video_id);
```

**New Table**: `follow_up_research_runs`
```sql
CREATE TABLE follow_up_research_runs (
  id BIGSERIAL PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES content(video_id),
  summary_id BIGINT REFERENCES summaries(id),
  approved_questions TEXT[] NOT NULL,
  question_provenance TEXT[],
  planned_queries TEXT[],
  coverage_map JSONB,
  research_response TEXT,
  research_meta JSONB,
  cache_key TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  completed_at TIMESTAMPTZ,
  status TEXT DEFAULT 'pending'
);
CREATE UNIQUE INDEX idx_follow_up_cache ON follow_up_research_runs(cache_key);
```

**New Summary Variant**: Add to summaries table
```sql
-- No schema change needed, just use variant = 'deep-research'
```

#### 1.4 Telegram Handler Extensions

**File**: `modules/telegram_handler.py`

```python
# After summary completion, check if follow-up suggestions should be generated

async def _maybe_offer_follow_up_research(
    self,
    update,
    video_id: str,
    summary: str,
    source_context: dict,
):
    """
    Generate and offer follow-up research suggestions if appropriate.

    Only suggests when:
    - Content is time-sensitive (older than 3 months OR mentions pricing/products)
    - Content has identifiable entities
    - Content is not already research-heavy
    """

    suggestions = suggest_follow_up_questions(
        source_context=source_context,
        summary=summary,
        entities=self._extract_entities(summary),
    )

    if not suggestions:
        return

    # Store suggestions for later use
    await self._store_follow_up_suggestions(video_id, suggestions)

    # Show suggestion buttons
    keyboard = self._build_follow_up_keyboard(suggestions)
    await update.message.reply_text(
        "🔍 *Deep Research Available*\n\n"
        "This content may have updates since publication. "
        "Select questions to research:",
        reply_markup=keyboard,
    )

async def _execute_follow_up_research(
    self,
    update,
    video_id: str,
    approved_questions: list[str],
):
    """
    Execute follow-up research for selected questions.
    Uses ONE coordinated research run.
    """

    source_context = await self._get_source_context(video_id)
    summary = await self._get_latest_summary(video_id)

    result = run_follow_up_research(
        source_context=source_context,
        summary=summary,
        approved_questions=approved_questions,
        provider_mode="tavily",  # Start with tavily to avoid rate limits
        depth="balanced",
    )

    # Store result as 'deep-research' variant
    await self._store_research_result(video_id, result)

    # Send result
    await self._send_research_result(update, result)
```

---

### Phase 2: API Integration

#### 2.1 YTV2 API Extensions

**File**: `ytv2_api/main.py`

```python
from research_service import suggest_follow_up_questions, run_follow_up_research

@app.post("/api/follow-up-suggestions")
async def get_follow_up_suggestions(request: FollowUpSuggestionsRequest):
    """
    Generate follow-up research suggestions for a summary.

    Only generates if content meets criteria (time-sensitive, entities, etc.)
    Returns empty array if suggestions not appropriate.
    """
    video_id = request.video_id
    source_context = await get_content_context(video_id)
    summary = await get_latest_summary(video_id)

    suggestions = suggest_follow_up_questions(
        source_context=source_context,
        summary=summary,
    )

    # Store for later use
    await store_suggestions(video_id, suggestions)

    return {"suggestions": suggestions}

@app.post("/api/follow-up-research")
async def run_followup_research(request: FollowUpResearchRequest):
    """
    Execute coordinated follow-up research.

    Request:
    {
        "video_id": "...",
        "approved_questions": ["Q1", "Q2", "Q3"],
        "question_provenance": ["suggested", "suggested", "custom"]
    }

    Returns standard ResearchRunResult with sections per question.
    """

    # Check cache first
    cache_key = build_cache_key(request.approved_questions, request.video_id)
    cached = await get_cached_research(cache_key)
    if cached:
        return cached

    source_context = await get_content_context(request.video_id)
    summary = await get_latest_summary(request.video_id)

    result = run_follow_up_research(
        source_context=source_context,
        summary=summary,
        approved_questions=request.approved_questions,
        question_provenance=request.question_provenance,
        provider_mode="auto",
        depth="balanced",
    )

    # Store as 'deep-research' variant
    await store_summary_variant(
        video_id=request.video_id,
        variant="deep-research",
        text=result.answer,
        meta=result.meta,
    )

    # Cache result
    await cache_research_result(cache_key, result)

    return result
```

---

### Phase 3: Dashboard Integration

#### 3.1 Frontend Changes

**New Tab**: "Deep Research"

**Location**: After existing variant tabs

**UI Flow**:
1. User loads content → existing summary tabs shown
2. After summary loads → check for follow-up suggestions
3. If suggestions exist → show "🔍 Deep Research" tab with badge
4. User clicks tab → show suggestion cards:
   ```
   ┌─────────────────────────────────────────┐
   │ 🔍 Deep Research                         │
   ├─────────────────────────────────────────┤
   │ Select questions to research:            │
   │                                          │
   │ ☑ What changed since this was published?│
   │ ☐ How does pricing compare today?       │
   │ ☐ What are the current alternatives?     │
   │                                          │
   │ [+ Add custom question]                  │
   │                                          │
   │ [Run Research] [Cancel]                  │
   └─────────────────────────────────────────┘
   ```
5. After research completes → show results using `ResearchReportView`:
   - Title, summary, sections per question
   - Comparison table if applicable
   - Sources list
   - Research metadata (provider breakdown, etc.)

#### 3.2 React Integration

```tsx
// Import the research reader
import { ResearchReportView } from '../../research_reader_react/src';
import '../../research_reader_react/src/research-reader.css';

function DeepResearchTab({ videoId, summary }) {
  const [suggestions, setSuggestions] = useState([]);
  const [selectedQuestions, setSelectedQuestions] = useState([]);
  const [researchResult, setResearchResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Load suggestions on mount
  useEffect(() => {
    fetchSuggestions(videoId).then(setSuggestions);
  }, [videoId]);

  const runResearch = async () => {
    setLoading(true);
    const result = await api.runFollowUpResearch({
      video_id: videoId,
      approved_questions: selectedQuestions,
      question_provenance: suggestions
        .filter(s => selectedQuestions.includes(s.question))
        .map(s => s.kind)
    });
    setResearchResult(result);
    setLoading(false);
  };

  if (researchResult) {
    return (
      <div className="deep-research-results">
        <ResearchReportView
          message={{
            content: researchResult.response,
            research: researchResult.meta
          }}
          fallbackTitle="Deep Research"
          tone="paper"  // or "dark"
        />
      </div>
    );
  }

  return (
    <div className="deep-research-prompt">
      {suggestions.map(suggestion => (
        <label key={suggestion.id}>
          <input
            type="checkbox"
            checked={selectedQuestions.includes(suggestion.question)}
            onChange={(e) => toggleQuestion(suggestion.question, e.target.checked)}
            defaultChecked={suggestion.default_selected}
          />
          {suggestion.label}
          <small>{suggestion.reason}</small>
        </label>
      ))}
      <button onClick={runResearch} disabled={selectedQuestions.length === 0 || loading}>
        {loading ? 'Researching...' : 'Run Research'}
      </button>
    </div>
  );
}
```

---

### Phase 4: Telegram Integration

#### 4.1 Button Flow

**After summary completion**:

```python
# Existing summary flow
await self._send_summary(update, summary_text, variant)

# NEW: Check for follow-up suggestions
if await self._should_suggest_follow_up(video_id, summary_text):
    suggestions = await self._get_or_generate_suggestions(video_id, summary_text)

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔍 Deep Research", callback_data=f"follow_up_show_{video_id}")]
    ])
    await update.message.reply_text(
        "Deep research available for this content",
        reply_markup=keyboard
    )
```

**User clicks "🔍 Deep Research"**:

```python
async def _follow_up_show_handler(self, update, video_id):
    """Show suggestion checkboxes"""
    suggestions = await self._get_stored_suggestions(video_id)

    # Build keyboard with checkboxes (using callback_data for state)
    keyboard = self._build_suggestion_keyboard(suggestions)

    await query.edit_message_text(
        f"🔍 *Deep Research Questions*\n\n"
        f"Select questions to research (max {MAX_APPROVED_QUESTIONS}):",
        reply_markup=keyboard
    )
```

**User confirms selections**:

```python
async def _follow_up_confirm_handler(self, update, video_id, selected_questions):
    """Execute follow-up research"""

    source_context = await self._get_source_context(video_id)
    summary = await self._get_latest_summary(video_id)

    status_msg = await update.message.reply_text("🔍 Running deep research...")

    result = run_follow_up_research(
        source_context=source_context,
        summary=summary,
        approved_questions=selected_questions,
        provider_mode="tavily",  # Avoid rate limits
        depth="balanced",
    )

    # Send as markdown with sections
    await self._send_research_result(update.message, result, status_msg)
```

---

## Implementation Order

### Sprint 1: Backend Foundation
1. ✅ Create `follow_up.py` module
2. ✅ Add `run_follow_up_research()` to `service.py`
3. ✅ Create database migrations for new tables
4. ✅ Add suggestion generation logic
5. ✅ Add caching layer

### Sprint 2: API Layer
1. ✅ Add `/api/follow-up-suggestions` endpoint
2. ✅ Add `/api/follow-up-research` endpoint
3. ✅ Integrate with existing summary flow
4. ✅ Add cache key generation

### Sprint 3: Telegram Bot
1. ✅ Add suggestion trigger after summary
2. ✅ Add suggestion keyboard handler
3. ✅ Add research execution handler
4. ✅ Add result formatting for Telegram
5. ✅ Add progress indicators

### Sprint 4: Dashboard UI
1. ✅ Add "Deep Research" tab to variant tabs
2. ✅ Implement suggestion selection UI
3. ✅ Integrate ResearchReportView component
4. ✅ Add loading states and error handling
5. ✅ Add result display with sections

---

## Key Design Decisions

### 1. Single Coordinated Research Run
**Decision**: All approved questions → ONE planner call → ONE execution
**Rationale**: Avoid redundant searches, minimize API costs, ensure shared sources

### 2. Separate Artifact Storage
**Decision**: Deep research stored as separate variant (`deep-research`), not mixed into original summary
**Rationale**: Clear separation, preserves original summary, allows independent updates

### 3. Cache-First Strategy
**Decision**: Cache results by (video_id + normalized_questions)
**Rationale**: Users may re-run with different tab combinations, expensive to re-run

### 4. Suggestion Throttling
**Decision**: Only generate suggestions for appropriate content (time-sensitive, entities present, etc.)
**Rationale**: Avoid noise for evergreen content, maintain quality perception

### 5. Progressive Disclosure
**Decision**: Show suggestions only after summary completes, never auto-run
**Rationale**: User control, cost containment, clear mental model

---

## Success Criteria

### Backend
- ✓ `run_follow_up_research()` returns coordinated results
- ✓ Coverage map shows which queries answered which questions
- ✓ Cache hits return immediately
- ✓ Single planner call handles multiple questions

### API
- ✓ `/api/follow-up-suggestions` returns relevant questions or empty
- ✓ `/api/follow-up-research` returns structured markdown with sections
- ✓ Response time < 30 seconds for 3 questions

### Telegram
- ✓ Suggestions appear after appropriate summaries
- ✓ Multi-select workflow works smoothly
- ✓ Research results render cleanly in markdown

### Dashboard
- ✓ "Deep Research" tab appears when suggestions exist
- ✓ Suggestion selection UI is intuitive
- ✓ ResearchReportView renders sections correctly
- ✓ Sources and metadata display properly

---

## Open Questions

1. **Should suggestions expire?** Currently 7 days - adjust based on usage?
2. **Should we limit daily research calls per user?** Prevent abuse?
3. **Should research results be regenerable?** Allow re-running with different questions?
4. **Should we add analytics?** Track which suggestion kinds are most popular?
5. **Should we add preset modes instead of suggestions?** Simpler but less flexible?

---

## Next Steps

**Before implementation begins**:
1. Review this plan with stakeholders
2. Confirm database migration approach
3. Decide on cache invalidation strategy
4. Set up monitoring for research costs
5. Prepare rollback strategy

**Implementation priority**:
1. Backend foundation (Sprint 1)
2. API layer (Sprint 2)
3. Telegram bot (Sprint 3) - can proceed in parallel with dashboard
4. Dashboard UI (Sprint 4) - depends on API completion
