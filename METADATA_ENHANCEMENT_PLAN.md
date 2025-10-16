# YTV2 Metadata Enhancement & Cleanup Plan
**Created**: 2025-09-08
**Status**: Phase 2 completed (metadata fallbacks live); Phase 3 optional

## 🎯 Objective
Enhance YouTube metadata extraction with additional fields and backfill existing JSON reports with missing data, while cleaning up duplicates and orphaned files.

## 📊 New Metadata Fields to Add

All new fields go under `source_metadata.youtube` to maintain clean universal schema:

```json
"source_metadata": {
  "youtube": {
    // EXISTING FIELDS (keep as-is)
    "video_id": "26-shcAnbZA",
    "channel_name": "Microsoft Research",
    "view_count": 95800,
    "tags": ["ai", "research"],
    
    // NEW ENGAGEMENT METRICS
    "like_count": 3200,
    "comment_count": 415,
    "channel_follower_count": 123456,
    
    // NEW CHANNEL INFO
    "uploader_id": "UCxxxxxxxx",
    "uploader_url": "https://www.youtube.com/@microsoftresearch",
    
    // NEW CONTENT CLASSIFICATION
    "categories": ["Science & Technology"],
    "availability": "public",         // public|private|unlisted|members_only
    "live_status": "not_live",        // is_live|was_live|not_live
    "age_limit": 0,
    
    // NEW TECHNICAL DATA
    "resolution": "1920x1080",
    "fps": 30,
    "aspect_ratio": 1.7778,
    "vcodec": "vp9",
    "acodec": "opus",
    
    // NEW CAPTION LANGUAGES (keys only)
    "automatic_captions": ["en", "es", "fr"],
    "subtitles": ["en"],
    
    // NEW TIMESTAMPS
    "release_timestamp": 1725583200  // unix timestamp
  }
}
```

## 📁 File Locations

### NAS (Primary Data)
- Reports: `/Volumes/Docker/YTV2/data/reports/`
- Audio: `/Volumes/Docker/YTV2/exports/`
- Scripts: `/Volumes/Docker/YTV2/tools/`
- Main extractor: `/Volumes/Docker/YTV2/youtube_summarizer.py`

### Render Dashboard (Display Only)
- Dashboard: `https://ytv2.onrender.com`
- Postgres-backed: reads from database tables only; no JSON sync or refresh endpoint.

## 🔄 Implementation Phases

### ✅ Phase 1: Cleanup Script (CURRENT)
**File**: `/Volumes/Docker/YTV2/tools/cleanup_reports.py`

**Features**:
1. Find duplicate JSONs (same video_id, keep newest)
2. Delete JSONs without corresponding MP3 files
3. Create backup before deletion
4. Detailed logging of all actions
5. Dry-run mode for safety

**Usage**:
```bash
# Preview what would be deleted
python tools/cleanup_reports.py --dry-run

# Run cleanup with backup
python tools/cleanup_reports.py --backup-dir ./backups/

# Limit to specific count for testing
python tools/cleanup_reports.py --limit 10 --dry-run
```

### ✅ Phase 2: Update YouTube Summarizer (COMPLETED)
**File**: `/Volumes/Docker/YTV2/youtube_summarizer.py`

- Added fallbacks to parse `ytInitialPlayerResponse` when yt-dlp returns “requested format not available”.
- Populated `source_metadata.youtube` with engagement + technical fields even when formats are blocked.
- Captured transcript language (`transcript_language`) and propagated summary languages (`summary_language`, `audio_language`).

### ⏳ Phase 3: Backfill Script (OPTIONAL)
**File**: `/Volumes/Docker/YTV2/tools/backfill_metadata.py`

**Features**:
- Idempotent (only adds missing fields)
- Resumable with state tracking
- Rate limiting (50 files, then 15s sleep)
- Atomic writes (temp file → rename)
- Dry-run mode
- Progress logging

**Usage**:
```bash
# Test on 10 files with dry-run
python tools/backfill_metadata.py --dry-run --limit 10

# Run on 10 files for real
python tools/backfill_metadata.py --limit 10

# Full run with resume capability
python tools/backfill_metadata.py --resume

# Force update even recent files
python tools/backfill_metadata.py --force --resume
```

### ⏳ Phase 4: Write to Postgres
1. Run cleanup/backfill on NAS (optional, local JSON)
2. Upsert `content` and `summaries` rows directly to Postgres
3. Ensure at least one HTML-bearing summary variant per video
4. Verify dashboard cards appear via DB query

## 🧪 Testing Checklist

### Before Cleanup:
- [ ] Count total JSON files
- [ ] Count unique video_ids
- [ ] Count orphaned JSONs (no MP3)
- [ ] Create backup directory

### After Cleanup:
- [ ] Verify no duplicate video_ids
- [ ] Verify all JSONs have MP3s
- [ ] Check backup contains deleted files
- [ ] Review cleanup.log

### After Backfill:
- [ ] Verify new fields in sample JSONs
- [ ] Confirm existing data preserved
- [ ] Check schema version updated
- [ ] Test dashboard with updated files

## 🛡️ Safety Measures

1. **Backups**: Always create before destructive operations
2. **Dry-run**: Test every script with --dry-run first
3. **Limits**: Use --limit for initial testing
4. **Atomic writes**: Prevent corruption with temp→rename pattern
5. **Rate limiting**: Be gentle on YouTube API
6. **Resume**: Can restart if interrupted
7. **Logging**: Detailed logs for audit trail

## 📈 Progress Tracking

### Cleanup Status:
- [ ] Script created
- [ ] Dry-run tested (10 files)
- [ ] Full dry-run completed
- [ ] Backup created
- [ ] Cleanup executed
- [ ] Results verified

### Backfill Status:
- [ ] Script created
- [ ] youtube_summarizer.py updated
- [ ] Test on 10 files
- [ ] Test on 100 files
- [ ] Full backfill started
- [ ] Full backfill completed
- [ ] Synced to Render
- [ ] Dashboard verified

## 🔍 Current Video Stats (Sample)
Based on `/Volumes/Docker/YTV2/data/reports/`:
- Total JSON files: ~100+ (needs exact count)
- Duplicates: Unknown (cleanup will determine)
- Orphaned JSONs: Unknown (cleanup will determine)
- Average file size: 5-10KB

## 📝 Notes

- OpenAI suggested keeping YouTube-specific fields under `source_metadata.youtube` ✅
- No need to run scripts on Render - just sync files ✅
- Cleanup duplicates and orphans before backfill ✅
- Start simple, add derived metrics later if needed ✅
- Use existing sync mechanism rather than SSH complexity ✅

## 🚀 Next Steps

1. **IMMEDIATE**: Create and test cleanup_reports.py
2. Test cleanup on 10 files
3. Run full cleanup
4. Update youtube_summarizer.py with new fields
5. Create and test backfill_metadata.py
6. Run backfill
7. Sync to Render
8. Verify dashboard

---
**Last Updated**: 2025-09-08 by Claude
**Context Preservation**: This document contains all planning details for the metadata enhancement project
