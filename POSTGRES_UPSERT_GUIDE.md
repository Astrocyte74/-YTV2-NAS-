# Postgres Upsert Guide (NAS → Dashboard)

This guide defines the minimal schema, roles, and UPSERT patterns the NAS should use to write directly to the dashboard’s Postgres database. The dashboard also accepts authenticated uploads for audio/images; the read path enriches JSON responses with audio variants joined from the `content` row.

## Connectivity

- Preferred: Direct DB writes from NAS to the Render Postgres instance using a least-privileged role.
- Use SSL (`sslmode=require`) and Render IP allowlist.
- Keep credentials in NAS env (`DATABASE_URL` or `PG*` vars); do not store them in the dashboard.

Example DSN:

```
DATABASE_URL=postgresql://ytv2_ingest:password@host:5432/ytv2?sslmode=require
```

## Tables and View

The dashboard reads only from:
- `content` – one row per video/report
- `v_latest_summaries` – one latest row per `(video_id, variant)`

Source slugs (e.g., youtube/reddit) are inferred in SQL from `canonical_url` and `video_id`. NAS should not set `content_source` explicitly.

### DDL

```sql
-- Core content table: columns used by the dashboard
CREATE TABLE IF NOT EXISTS content (
  id TEXT PRIMARY KEY,                 -- ok to reuse video_id if no separate id
  video_id TEXT NOT NULL UNIQUE,       -- stable key (e.g., 11-char YT id or reddit:<id>)
  title TEXT NOT NULL,
  channel_name TEXT,
  canonical_url TEXT,
  thumbnail_url TEXT,
  duration_seconds INTEGER,
  indexed_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  has_audio BOOLEAN DEFAULT false,
  language TEXT,                       -- used by filters if present
  analysis_json JSONB,
  subcategories_json JSONB,            -- preferred categories payload
  topics_json JSONB
);

-- Summaries backing table (view selects latest per variant)
CREATE TABLE IF NOT EXISTS summaries (
  id BIGSERIAL PRIMARY KEY,
  video_id TEXT NOT NULL,
  variant TEXT NOT NULL,               -- 'comprehensive','bullet-points','key-points','executive','key-insights','audio','audio-fr','audio-es', ...
  revision INTEGER NOT NULL DEFAULT 1,
  text TEXT,
  html TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (video_id, variant, revision)
);

-- View: latest row per (video_id, variant)
CREATE OR REPLACE VIEW v_latest_summaries AS
SELECT DISTINCT ON (video_id, variant)
  video_id, variant, text, html, created_at, revision
FROM summaries
WHERE (text IS NOT NULL OR html IS NOT NULL)
ORDER BY video_id, variant, created_at DESC, revision DESC;

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_content_indexed_at ON content (indexed_at DESC);
CREATE INDEX IF NOT EXISTS idx_content_lang ON content (language);
CREATE INDEX IF NOT EXISTS idx_content_analysis_gin ON content USING GIN (analysis_json);
CREATE INDEX IF NOT EXISTS idx_content_subcats_gin ON content USING GIN (subcategories_json);
CREATE INDEX IF NOT EXISTS idx_summaries_lookup ON summaries (video_id, variant, created_at DESC);
```

Note: If you prefer not to create a view, you may query “latest” with a lateral subquery ordering by `created_at DESC, revision DESC LIMIT 1`. Keep semantics consistent with the view.

## Role and Grants

```sql
CREATE ROLE ytv2_ingest LOGIN PASSWORD '...';
GRANT USAGE ON SCHEMA public TO ytv2_ingest;
GRANT INSERT, UPDATE ON TABLE content, summaries TO ytv2_ingest;
GRANT SELECT ON TABLE content, summaries, v_latest_summaries TO ytv2_ingest;
```

## UPSERT Patterns

Write rows using server-side parameters (shown with psycopg-style named params).

### Content (one row per video)

```sql
INSERT INTO content (
  video_id, id, title, channel_name, canonical_url, thumbnail_url,
  duration_seconds, indexed_at, has_audio, language,
  analysis_json, subcategories_json, topics_json
) VALUES (
  %(video_id)s, %(id)s, %(title)s, %(channel_name)s, %(canonical_url)s, %(thumbnail_url)s,
  %(duration_seconds)s, %(indexed_at)s, %(has_audio)s, %(language)s,
  %(analysis_json)s, %(subcategories_json)s, %(topics_json)s
)
ON CONFLICT (video_id) DO UPDATE SET
  title = EXCLUDED.title,
  channel_name = EXCLUDED.channel_name,
  canonical_url = EXCLUDED.canonical_url,
  thumbnail_url = EXCLUDED.thumbnail_url,
  duration_seconds = EXCLUDED.duration_seconds,
  indexed_at = EXCLUDED.indexed_at,
  has_audio = EXCLUDED.has_audio,
  language = EXCLUDED.language,
  analysis_json = EXCLUDED.analysis_json,
  subcategories_json = EXCLUDED.subcategories_json,
  topics_json = EXCLUDED.topics_json,
  updated_at = now();
```

### Summaries (variants)

```sql
INSERT INTO summaries (video_id, variant, revision, text, html, created_at)
VALUES (%(video_id)s, %(variant)s, %(revision)s, %(text)s, %(html)s, %(created_at)s)
ON CONFLICT (video_id, variant, revision) DO UPDATE SET
  text = EXCLUDED.text,
  html = EXCLUDED.html,
  created_at = EXCLUDED.created_at;
```

### Variant Set the Dashboard Filters To

```
('comprehensive','key-points','bullet-points','executive','key-insights','audio','audio-fr','audio-es')
```

## Category/Subcategory JSON

Preferred column: `subcategories_json` (dashboard also checks `analysis_json->'categories'`). Example shape:

```json
{
  "categories": [
    { "category": "Technology", "subcategories": ["Programming & Software Development"] },
    { "category": "History",    "subcategories": ["World War II (WWII)"] }
  ]
}
```

## Audio Handling

Preferred write pattern:
- Upload MP3 via `POST /api/upload-audio` (auth: `Authorization: Bearer $SYNC_SECRET` or `X-INGEST-TOKEN`). The server returns `public_url` (e.g., `/exports/audio/<filename>.mp3`) and `size`.
- Update `content`:
  - `has_audio = true`
  - `media.audio_url = "/exports/audio/<filename>.mp3"` (root‑relative)
  - `media_metadata.mp3_duration_seconds = <int>`
  - `audio_version = <unix_ts>` (used for `?v=` cache busting)

Read pattern:
- The dashboard’s JSON endpoints (`/<id>.json`, `/api/reports`) enrich `summary_variants` with `{ kind:'audio', audio_url, duration }` when the fields above exist on `content`.
- You do not need to store `<audio>` HTML in `summaries` for playback.

## Gotchas

- Card eligibility: requires at least one summary variant with non-null `html` per video.
- Language filter uses `content.language`; you can also mirror into `analysis_json` for completeness.
- SQL LIKE patterns require escaping percent signs in the dashboard code (NAS doesn’t need to handle this on writes).
- Ensure sufficient disk space on the dashboard’s `/app/data` volume; uploads write atomically but will fail with 500 if the disk is full.
