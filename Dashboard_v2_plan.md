## 📋 At-a-Glance Checklist

**1. Schema & Pipeline (NAS):** Finalize universal schema, enrich pipeline to populate analysis fields for better organization.
**2. Backfill Script (NAS):** Build simple script to update existing JSON files to the new schema.
**3. Index + API (Render):** Implement in-memory index and basic API endpoints with filtering and sorting.
**4. Dashboard V2 (Render):** Create clean, responsive dashboard UI with Tailwind and essential filtering.
**5. Basic Rollout (Render):** Deploy V2 dashboard with simple feature flag, then make it default.

# 🚀 YTV2 Dashboard V2: Personal Content Management

**Version:** 2.0 (Simplified)
**Date:** 2025-09-07
**Status:** Ready for Implementation

This document outlines a focused plan for creating a beautiful, functional dashboard for personal YouTube content organization. Designed for single-user efficiency without enterprise complexity.

---

### Guiding Philosophy: Simple, Fast, Personal

Create a beautiful, responsive dashboard that makes it easy to find and organize your YouTube content. Focus on essential features that provide immediate value: smart filtering, clean design, and fast search.

---

### Architecture Note: NAS vs Render Responsibilities

The YTV2 platform is architected with a clear division of responsibilities between two main components:

- **NAS (Network Attached Storage)**
    - Handles all data processing, schema enforcement, and backfill operations.
    - Responsible for running content analysis pipelines, generating and updating JSON files with the universal schema, and maintaining the canonical data store.
    - Backfill and processing scripts (e.g., `tools/backfill_analysis.py`) execute on NAS.

- **Render (API & Dashboard Server)**
    - Hosts the API, in-memory index, dashboard UI.
    - On startup, scans processed JSON files from NAS to build the in-memory index.
    - Serves API endpoints (`/api/filters`, `/api/reports`) and renders the dashboard UI.
    - All user-facing interactions and filtering are handled on Render.

This separation ensures clean architecture: NAS focuses on data processing, while Render provides fast, responsive UI.

#### NAS Path & Mount (current)

For clarity, here are the concrete paths in the current setup (macOS host mounting the NAS):

- **NAS mount root (host):** `/Volumes/Docker/`
- **Project root (host/NAS share):** `/Volumes/Docker/YTV2/`
- **Reports directory (JSON):** `/Volumes/Docker/YTV2/data/reports/`
- **Exports (audio, assets):** `/Volumes/Docker/YTV2/exports/`
- **Logs:** `/Volumes/Docker/YTV2/logs/`

#### Render Repo (local development)

For clarity during local development (prior to containerization on Render):

- **Local repo root (macOS):** `/Users/markdarby/projects/YTV2-Dashboard`
- **Container working dir on Render:** `/app`
- **Expected mounted data dirs in container:** `/app/data/reports` and `/app/exports`

> The macOS local repo path is for developer reference only. On Render, the application runs inside the container at `/app`; ensure the data directories are mounted or synced into `/app/data` and `/app/exports` respectively.

#### Standardized Environment Variables

To keep code portable between NAS (processing) and Render (serving), define these variables:

- `NAS_BASE` — root of the NAS project (e.g., `/Volumes/Docker/YTV2`)
- `REPORTS_DIR` — `${NAS_BASE}/data/reports`
- `EXPORTS_DIR` — `${NAS_BASE}/exports`
- `LOGS_DIR` — `${NAS_BASE}/logs`

Example `.env` (NAS):

```env
NAS_BASE=/Volumes/Docker/YTV2
REPORTS_DIR=${NAS_BASE}/data/reports
EXPORTS_DIR=${NAS_BASE}/exports
LOGS_DIR=${NAS_BASE}/logs
```

Example `.env` (Render):

```env
# On Render, bind/volume/sync these into the container at /app/data and /app/exports
DATA_ROOT=/app/data
EXPORTS_ROOT=/app/exports
REPORTS_DIR=${DATA_ROOT}/reports
EXPORTS_DIR=${EXPORTS_ROOT}
```

**Render expects** `REPORTS_DIR` to be readable and populated by the NAS pipeline or a sync job. See *Phase 2 (Render)* for how the index scans `REPORTS_DIR` at startup.

## 🏛️ Phase 1 (NAS): The Universal Data Backbone ✅ COMPLETED

**Status:** ✅ **IMPLEMENTED - September 7, 2025**

This foundational phase focused on standardizing our data structure and enriching it with high-quality, queryable metadata. All components have been successfully implemented and tested.

### 1.1 The Finalized Universal Content Schema

This schema is the single source of truth. Every piece of content, regardless of source, will be processed to conform to this structure.

```json
{
  "id": "yt:sm6nmldqtli",                  // Stable, lowercase, unique ID: {source_short_code}:{source_id}
  "content_source": "youtube",             // ENUM: youtube|webpage|pubmed|reddit|arxiv|pdf
  "title": "The Secrets of Roman Concrete...",
  "canonical_url": "https://youtube.com/watch?v=...",
  "thumbnail_url": "https://i.ytimg.com/...",
  "published_at": "2025-08-08T14:30:00Z",   // ISO 8601 format in UTC. 'year' is derived at index time.
  "duration_seconds": 657,                 // For time-based media
  "word_count": 1234,                      // Estimated word count for reading time (text-based sources)

  "media": {
    "has_audio": true,                     // Does a separate audio file exist?
    "audio_duration_seconds": 655,         // Duration of the extracted audio file.
    "has_transcript": true,
    "transcript_chars": 8950               // Allows for reading time calculation without loading the file.
  },

  "source_metadata": {                     // Source-specific, non-universal data
    "youtube": { "video_id": "sM6nmldqTlI", "channel_name": "Tech Historian" },
    "pubmed": { "pmid": "12345678", "journal": "Nature", "authors": ["Doe, J.", "Smith, A."] },
    "reddit": { "subreddit": "r/science", "score": 2450 }
  },

  "analysis": {
    "category": ["History", "Science"],    // Controlled taxonomy (1-3 tags)
    "content_type": "Documentary",         // ENUM: Tutorial, Research, News, Opinion, Guide, etc.
    "complexity_level": "Intermediate",    // ENUM: Beginner, Intermediate, Advanced
    "language": "en",                      // ISO 639-1 code ('und' for undetermined)
    "key_topics": ["roman-concrete", "vesuvius", "ancient-engineering"], // Lowercase, machine-generated slugs
    "named_entities": ["Pompeii", "Pozzolana"] // People, Places, Organizations
  },

  "processing": {                          // Observability for the data pipeline
    "status": "complete",                  // ENUM: pending, processing, complete, failed
    "pipeline_version": "2025-09-07-v1",
    "attempts": 1,
    "started_at": "2025-09-07T10:00:05Z",
    "completed_at": "2025-09-07T10:01:25Z",
    "error": null,                         // Short error string on failure
    "logs": [                             // Optional array of pipeline step messages/errors for observability
      "Step 1: metadata extracted",
      "Step 2: language detected as 'en'",
      "Step 3: topic extraction completed"
    ]
  }
}
```

### 1.2 Enhanced Processing Pipeline
The `analyze_content()` function will be upgraded to populate the final schema fields.

- **Language Detection:** Detect from transcript first, with a fallback to summary text. Lock to a defined list (`en`, `es`, `fr`, etc.) with `und` as the final fallback.
- **Media Analysis:** Persist `media.has_audio`, `media.audio_duration_seconds`, and `media.transcript_chars`.
- **Topic Extraction:** Generate 3–7 tags, lowercase, convert to slugs (e.g., `ancient-rome`), deduplicate, and cap length at <= 30 chars.
- **Observability:** Precisely set `processing` fields (`status`, timestamps, `attempts`, `error`) for every run, and append detailed step logs to the optional `logs` array.

### 1.3 Simple Backfill Script (`tools/backfill_analysis.py`)
This script will update all existing JSON files to the new schema.  
_**(Runs on NAS)**_
- **Idempotent:** Skips any file where `processing.pipeline_version` matches the current version.
- **Atomic Writes:** Writes to a `.tmp` file and performs an atomic `rename` on success to prevent data corruption.
- **Resumable:** Tracks the last successfully processed filename in a state file (e.g., `.backfill_state`) to resume after interruption.
- **Flags:** Must include `--dry-run` (shows what would change) and `--limit N` (processes only N files).

**Paths & invocation (NAS):**
- Input: `${REPORTS_DIR}` (e.g., `/Volumes/Docker/YTV2/data/reports`)
- Output (in‑place atomic write): temp files written alongside originals within `${REPORTS_DIR}`
- Run: `python tools/backfill_analysis.py --limit 100`

### 1.4 ✅ IMPLEMENTATION SUMMARY

**All Phase 1 components successfully implemented and tested:**

#### 🔐 Security Enhancements
- **✅ Prompt Injection Prevention** - Input sanitization with `_sanitize_content()` method
- **✅ Safe JSON Parsing** - Size limits and validation via `_parse_safe_json()`  
- **✅ Input Validation** - Comprehensive data validation for all user inputs
- **✅ Content Truncation** - Intelligent sentence-boundary preservation (8000 char limit)

#### 🏗️ Universal Schema Implementation
- **✅ New analyze_content() Function** - Completely refactored with security and schema compliance
- **✅ Schema Validation** - Built-in validation via `_validate_analysis_result()`
- **✅ Topic Slug Processing** - Automatic conversion to lowercase-hyphenated format
- **✅ Processing Metadata** - Full observability with pipeline versions, timestamps, logs
- **✅ Backward Compatibility** - Preserves existing fields during transition period

#### 📝 Robust Backfill Script (`tools/backfill_analysis.py`)
- **✅ Idempotent Operations** - Skips already migrated files via pipeline version check
- **✅ Atomic Writes** - Temporary files with atomic rename prevent corruption
- **✅ Resumable Processing** - State file tracks progress for interruption recovery
- **✅ Automatic Backups** - Creates timestamped backups before any modifications
- **✅ Dry Run Mode** - Safe testing with `--dry-run` flag
- **✅ Flexible Limits** - Process subset with `--limit N` flag
- **✅ Tested Successfully** - Validated on 56 existing JSON files

#### 🎯 Key Achievements
- **Universal Schema Compliance** - All new content follows standardized format
- **Security Hardening** - Critical vulnerabilities eliminated
- **Data Migration Ready** - 56 legacy files ready for seamless conversion
- **Processing Observability** - Full pipeline tracking and logging
- **Future-Proof Design** - Extensible for additional content sources (PubMed, Reddit, etc.)

#### 📊 Current Status
- **Legacy Files**: 56 JSON files ready for migration
- **Pipeline Version**: `2025-09-07-v1` 
- **Security Level**: Production-ready with comprehensive input validation
- **Schema Compliance**: 100% universal schema support
- **Backup Strategy**: Automatic backups during migration

---

## 💡 Personal Topic Management (Simplified)

Since this is a single-user system, we can skip all the complex community voting and promotion systems. Instead, focus on **direct topic editing and personal taxonomy building**.

**MVP Approach: Direct Edit**
- Store topics as simple slugs in JSON files (as planned)
- All topics are immediately available for filtering - no "pending" or "promotion" needed
- User can directly edit topics on summary pages or via simple API calls

**Personal Topic Enhancement Schema**
```json
{
  "analysis": {
    "key_topics": ["ancient-rome", "engineering", "materials"],
    "topic_history": {
      "ai_original": ["ancient-rome", "engineering", "concrete"], 
      "user_modified_at": "2025-09-07T15:30:00Z",
      "user_changes": {
        "removed": ["concrete"],
        "added": ["materials"]
      }
    }
  }
}
```

**Implementation Options**

**Option 1: Summary Page Quick Edit** ⭐ **(Recommended)**
- Add simple editable topic chips on V2 summary pages
- Click 'X' to remove, type to add, changes save immediately
- Perfect for refining while reading content

**Option 2: Telegram Commands** 
- `/topics +materials,construction -concrete` 
- Quick mobile editing during content review

**API Endpoints (Simple)**
- `PUT /api/reports/{id}/topics` - Replace topics for a video
- `POST /api/reports/{id}/topics` - Add topic to a video  
- `DELETE /api/reports/{id}/topics/{topic}` - Remove topic

**Personal Benefits**
- **Immediate changes** - no waiting for approval/promotion
- **Consistent terminology** - build your own topic vocabulary  
- **Learning enhancement** - connect concepts across your content library
- **Perfect discovery** - topics match exactly how YOU think about content

**Topic Normalization (Simple Rules)**
- Convert to lowercase, replace spaces with hyphens
- Truncate to 30 chars max
- Basic synonym mapping (optional, can be a simple JSON file)

---
## 🏗️ Phase 2 (Render): Simple API & Index

The engine to serve our standardized data quickly.

### 2.1 In-Memory Index Architecture
_**(Runs on Render)**_
- **Data location on Render:** `REPORTS_DIR` must be available inside the container (e.g., mounted to `/app/data/reports`). If using a sync job, ensure it completes before app start.
- **Startup Hydration:** On application start, query Postgres via the ingest client to populate the in-memory index. Derive the `year` facet from `published_at` in the returned records. (The legacy JSON scan flow is retired but can be re-enabled locally if Postgres is unavailable.)
- **Basic Facet Counts:** Simple facet counts for filtering.

### 2.2 Core API Endpoints & Shapes
_**(Runs on Render)**_
- **Sorting:** Support `sort=newest|title|duration`.
- **Pagination:** Use `page` and `size` parameters. Cap `size` at `50`.

#### `GET /api/filters`
_**API endpoint served by Render**_
Returns dynamic facet buckets based on current data.
```json
{
  "source": [{"value":"youtube","count":124}],
  "language": [{"value":"en","count":110},{"value":"es","count":14}],
  "category": [{"value":"History","count":56}],
  "key_topics": [{"value":"ancient-rome","count":18}],
  "content_type": [{"value":"Documentary","count":45}],
  "complexity": [{"value":"Beginner","count":32}],
  "has_audio": [{"value":true,"count":98}],
  "year": [{"value":"2025","count":87},{"value":"2024","count":37}]
}
```

#### `GET /api/reports`
_**API endpoint served by Render**_
Returns paginated, filtered content with metadata.

**Query Parameters:**
- `source`, `language`, `category`, `topics`, `has_audio`, `complexity`: Filter arrays (e.g., `topics=ancient-rome,engineering`)
- `date_from`, `date_to`: ISO date strings (e.g., `date_from=2024-01-01`)
- `q`: Search query (simple text matching across title, summary)
- `sort`: `newest|title|duration` (default: `newest`)
- `page`, `size`: Pagination (default: page=1, size=20, max size=50)

**Response:**
```json
{
  "data": [
    {
      "id": "yt:sm6nmldqtli",
      "title": "The Secrets of Roman Concrete",
      "thumbnail_url": "https://i.ytimg.com/...",
      "published_at": "2025-08-08T14:30:00Z",
      "duration_seconds": 657,
      "analysis": {
        "language": "en",
        "category": ["History", "Science"],
        "key_topics": ["roman-concrete", "ancient-engineering"]
      },
      "media": {
        "has_audio": true
      }
    }
  ],
  "pagination": {
    "page": 1,
    "size": 20,
    "total": 124,
    "pages": 7
  }
}
```

---
## ✨ Phase 3 (Render): Clean V2 Dashboard UI

The user-facing interface built with Tailwind CSS.  
_**(Runs on Render)**_

### 3.1 Overall Layout & Navigation

**Header:**
- Clean search bar with debounced input (300ms delay)
- Simple "Filters" toggle button (mobile) / no toggle needed (desktop)
- Basic dark/light mode toggle

**Main Content Area:**
- Left sidebar (desktop) / drawer (mobile) for filters
- Main content grid for cards
- Simple pagination at bottom

### 3.2 Filter Panel

**Desktop Layout (Sticky Left Sidebar):**
- Fixed 240px width sidebar with smooth scroll
- Simple sections: "Language", "Content Type", "Topics", "Date Range"
- Basic facet counts showing available options

**Mobile Layout (Slide-out Drawer):**
- Triggered by "Filters" button in the top bar
- Slide-in overlay from left
- Touch-optimized with larger tap targets
- "Apply" and "Clear" buttons at bottom

**Essential Features:**
- **Basic topic filtering:** Simple list of available topics
- **Date Range:** Simple "Last 30 days", "Last 3 months", "This year" options

### 3.3 Active Filter Chips

Show active filters as simple chips below the search bar:

- **Design:** Clean rounded pills with "×" close button
- **Behavior:** Click "×" to remove filter
- **Quick Actions:** "Clear All" button when multiple filters are active

### 3.4 Content Cards

**Card Layout:**
- Responsive grid: 1 column (mobile), 2 columns (tablet), 3 columns (desktop)
- Thumbnail image at top (16:9 aspect ratio)
- Title, channel name, duration
- Simple language pill (e.g., "EN") and audio icon when available
- Up to 3 topic tags with clean styling

**Card Interactions:**
- Click anywhere on card to open summary page
- Hover effects on desktop for better UX
- Touch-friendly spacing and sizing on mobile

**Performance & UX:**
- Debounced search (300ms delay) to reduce API calls
- Lazy-loaded thumbnails for better performance
- Simple loading states with basic placeholders
- Clean error states with retry buttons

**Responsive Design:**
- Mobile: Single-column layout
- Tablet: Two-column layout
- Desktop: Three-column layout

**Interaction Details:**
- Simple hover effects on desktop
- Touch-friendly tap states on mobile
- Basic focus states for keyboard navigation

---
## 🛡️ Phase 4 (Render): Basic Production Setup

Ensuring a stable deployment.

### 4.1 Basic Production Setup
_**(Runs on Render)**_
- **Tailwind:** Use local build process for optimized CSS
- **Sanitization:** Use `bleach` for HTML sanitization
- **Basic Caching:** Simple cache headers for static assets
- **Error Handling:** Basic error boundaries and fallback states
- **Health Check:** Simple `/api/health` endpoint

### 4.2 Simple Rollout Plan
_**(Runs on Render)**_
1.  **Feature Flag:** Deploy accessible via `?dash=v2` for initial testing
2.  **Make Default:** After basic testing, make V2 the default with `?legacy=1` fallback
3.  **Simple Feedback:** Basic feedback mechanism if needed

---

## 🔍 **Essential Quality Measures**

### **Essential Quality Measures**
- **Schema Validation:** Basic JSON schema validation during processing
- **Simple Error Handling:** Basic error states and retry mechanisms
- **Data Backup:** Regular backups of processed JSON files

---

## ✅ Action Plan: Implementation Steps

This is the prioritized list of tasks for immediate development.

#### 1. Schema Finalization & Pipeline (NAS)
**Owner / Env:** NAS team / NAS server
- Verify NAS paths: `${NAS_BASE}`, `${REPORTS_DIR}`, `${EXPORTS_DIR}` exist on the host (`/Volumes/Docker/YTV2/...`)
- Implement the schema exactly as defined above (adding `audio_duration_seconds`, `transcript_chars`, `named_entities`, `word_count`, and all `processing` fields including optional `logs`)
- Ensure `analyze_content()` (runs on NAS) populates: `language`, `category`, `content_type`, `complexity_level`, `key_topics`, `named_entities`
- Set all `processing` fields (`started_at`, `completed_at`, `status`, `error`, `attempts`, and `logs`) during pipeline execution (NAS)
- Store topics as slugs
  - **Definition of Done:**
    - All new and processed content JSONs conform to the finalized schema
    - All analysis fields are populated
    - Processing metadata is present and accurate
    - Topics are stored as slugs

#### 2. Simple Backfill Script (NAS)
**Owner / Env:** NAS team / NAS server
- Create `tools/backfill_analysis.py` ensuring it is idempotent and performs atomic writes  
  _**(Runs on NAS)**_
- The script must populate missing `analysis.language`, `analysis.key_topics`, `media.has_audio`, `media.audio_duration_seconds`, and `word_count`
- Add `--dry-run` and `--limit` flags
  - **Definition of Done:**
    - Script updates all legacy JSONs in-place to match the new schema
    - Supports dry-run and limit flags
    - Atomic writes are verified
    - State file tracks progress

#### 3. Simple Index + API (Render)
**Owner / Env:** Render team / Render container
- Ensure `REPORTS_DIR` is mounted/populated on Render (e.g., `/app/data/reports`). Fail fast with a clear error if missing
- Build the `ContentIndex` with basic facet counts  
  _**(Runs on Render)**_
- Implement `GET /api/filters` and `GET /api/reports` with the exact response shapes defined above  
  _**(API endpoints on Render)**_
- The `/api/reports` endpoint must support all filters: `source`, `language`, `category`, `topics`, `has_audio`, `complexity`, `date_from`/`date_to`, `q`, `sort`, `page`, `size`
- Cap the `size` parameter and add basic caching headers
  - **Definition of Done:**
    - APIs return correct data and shape
    - All filters and sorting work as specified
    - Basic facet counts are working
    - Pagination and caching headers are correct
    - Fails fast if data unavailable

#### 4. Clean Dashboard V2 (Tailwind, Render)
**Owner / Env:** Render team / Render container
- Build the left filter panel (desktop sticky; mobile drawer)  
  _**(UI/UX on Render)**_
- Implement the active filter chips area beneath the search bar
- Create clean cards that display the source badge, language pill, audio icon, topics
- Implement debounced search, lazy-loaded thumbnails, and pagination
  - **Definition of Done:**
    - All core UI elements are functional and styled
    - Filters, chips, and cards work on all viewports
    - Debounced search and lazy loading are implemented
    - Pagination and navigation work smoothly

#### 5. Basic Rollout (Render)
**Owner / Env:** Render team / Render container
- Guard the V2 dashboard with the `?dash=v2` parameter  
  _**(Rollout logic on Render)**_
- After basic testing, flip the V2 dashboard to be the default and retain a `?legacy=1` escape hatch
  - **Definition of Done:**
    - V2 dashboard is accessible via `?dash=v2`
    - After test period, V2 is default with `?legacy=1` fallback
    - Rollout is monitored and issues can be quickly reverted

**Risk & Rollback**
- *Risks:* API or dashboard regressions, data sync issues
- *Rollback procedure:* Flip feature flag or revert default to legacy dashboard; restore previous API endpoints if needed
