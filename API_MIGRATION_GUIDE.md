# YTV2 API Migration Guide

This guide documents the migration from database file sync to API-based synchronization between NAS and Render.

## Overview

**Before**: NAS uploaded SQLite database files to Render
**After**: NAS sends individual content records via Postgres ingest (`/ingest/report`, `/ingest/audio`) using UPSERT logic

## Key Changes

### 1. New API Endpoints (Render Side)

- `POST /ingest/report` - Create or update content (UPSERT)
- `POST /ingest/audio`  - Upload MP3 files and link to content

### 2. NAS Components (2025)

- `modules/postgres_sync_client.py` - Ingest client for Postgres dashboard
- `modules/dual_sync_coordinator.py` - Coordinates report/audio uploads and health checks
- Updated `nas_sync.py` - High-level sync entry points (full sync, per-report dual sync)
- `modules/report_generator.py` - Generates reports; writes to SQLite only when explicitly enabled

### 3. Obsolete Components (already removed)

- `sync_sqlite_db.py`, `tools/bulk_sync_to_render.py`, `test_database_sync.py`
- Legacy dashboard endpoints `/api/upload-database` and `/api/upload-report`

## Migration Benefits

### Performance
- ✅ No large database file transfers
- ✅ Individual record updates (faster sync)
- ✅ UPSERT logic eliminates duplicate checking

### Reliability  
- ✅ Retry logic with exponential backoff
- ✅ Network error handling
- ✅ Transactional updates

### Scalability
- ✅ Handles large content databases efficiently  
- ✅ Selective sync (only changed records)
- ✅ Bandwidth optimization

## Usage

### Environment Variables
```bash
# Required for ingest client
export RENDER_DASHBOARD_URL="https://your-dashboard.onrender.com"
export INGEST_TOKEN="your-ingest-token"
export SYNC_SECRET="legacy-shared-secret"   # still used for delete callbacks
export POSTGRES_ONLY=true
export SQLITE_SYNC_ENABLED=false
```

### New Sync Functions

**Full ingest sync (reports + audio):**
```python
from nas_sync import dual_sync_upload
dual_sync_upload("data/reports/video_id.json", "exports/audio_video_id.mp3")
```

**Bulk backfill / sanity check:**
```python
from nas_sync import sync_content_via_api
sync_content_via_api()  # Respects POSTGRES_ONLY / SQLITE_SYNC_ENABLED
```

### Legacy Compatibility

The migration maintains limited backward compatibility:
- `sync_sqlite_database()` now logs a deprecation warning and proxies to `sync_content_via_api()`
- `upload_to_render()` converts legacy reports to universal schema before invoking the ingest client
- `SYNC_SECRET` is still honored for delete callbacks, but ingest auth relies on `INGEST_TOKEN`

## Testing

Run the comprehensive test suite:
```bash
python tools/test_api_sync.py
```

Tests verify:
- API connectivity
- Database sync via API  
- Legacy upload compatibility

## Cleanup

Legacy cleanup helpers were deprecated; obsolete SQLite scripts have already been removed from `main`.

## Rollback Plan

If issues arise, rollback is possible:

1. **Restore obsolete files** from backup directory
2. **Revert environment variables** to use old endpoints
3. **Use legacy sync functions** until issues are resolved

The API endpoints support both old and new sync methods during transition.

## Monitoring

Monitor the new API sync:

**Success Indicators:**
- HTTP 200 responses from `/ingest/report`
- UPSERT actions logged (created/updated)
- Audio uploads to `/ingest/audio`
- No database file uploads needed

**Error Indicators:**
- HTTP 4xx/5xx responses from `/ingest/report` or `/ingest/audio`
- Network timeout errors
- Authentication failures (`INGEST_TOKEN` mismatch)
- Missing content records on dashboard

## Support

**API Client Issues:**
- Check `RENDER_API_URL` and `SYNC_SECRET` environment variables
- Verify network connectivity to Render
- Review API client logs for detailed errors

**Sync Issues:**
- Confirm `POSTGRES_ONLY=true` and `SQLITE_SYNC_ENABLED=false`
- Check logs for `Dual-sync SUCCESS` vs warning messages
- Monitor the dashboard for new cards (consider SSE/WebSocket refresh)

**Performance Issues:**
- Adjust retry logic in API client
- Monitor API response times
- Consider batch size optimization

---

**Migration completed**: ✅ API-based sync system operational  
**Legacy system**: ⚠️ Available for rollback if needed  
**Cleanup**: 🧹 Run cleanup script when confident in new system
