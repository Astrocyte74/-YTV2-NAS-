# YTV2 API Migration Guide

This guide documents the migration from database file sync to API-based synchronization between NAS and Render.

## Overview

**Before**: NAS uploaded SQLite database files to Render
**After**: NAS sends individual content records via REST API using UPSERT logic

## Key Changes

### 1. New API Endpoints (Render Side)

- `POST /api/content` - Create or update content with UPSERT logic
- `PUT /api/content/{id}` - Update specific content fields  
- Existing `POST /api/upload-audio` - Upload MP3 files

### 2. New NAS Components

- `modules/render_api_client.py` - Comprehensive API client
- Updated `nas_sync.py` - API-based sync functions
- Updated `modules/report_generator.py` - Direct API sync capability

### 3. Obsolete Components

**Files marked for cleanup:**
- `sync_sqlite_db.py` - Database file upload (replaced by API client)
- `tools/bulk_sync_to_render.py` - Bulk JSON upload (replaced by API client)
- `test_database_sync.py` - Old database sync test

**Functions/Endpoints marked for cleanup:**
- `POST /api/upload-database` - Database file upload
- `POST /api/upload-report` - JSON file upload (partially replaced)

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
# Required for API client
export RENDER_API_URL="https://your-dashboard.onrender.com"
export SYNC_SECRET="your-sync-secret"
```

### New Sync Functions

**Full database sync via API:**
```python
from nas_sync import sync_content_via_api
success = sync_content_via_api()
```

**Individual report upload:**
```python  
from nas_sync import upload_to_render
success = upload_to_render("path/to/report.json", "path/to/audio.mp3")
```

**Direct API sync from report generator:**
```python
from modules.report_generator import JSONReportGenerator
generator = JSONReportGenerator()
generator.sync_report_to_api(report_data, audio_path)
```

### Legacy Compatibility

The migration maintains backward compatibility:
- `sync_sqlite_database()` redirects to new API sync
- `upload_to_render()` converts legacy reports to new format
- Environment variables work with both old and new variable names

## Testing

Run the comprehensive test suite:
```bash
python test_api_sync.py
```

Tests verify:
- API connectivity
- Database sync via API  
- Legacy upload compatibility

## Cleanup

Run the cleanup script to identify and remove obsolete code:
```bash
python cleanup_obsolete_sync.py
```

This will:
- Identify obsolete files
- Check for active usage
- Create backups before removal
- Provide migration recommendations

## Rollback Plan

If issues arise, rollback is possible:

1. **Restore obsolete files** from backup directory
2. **Revert environment variables** to use old endpoints
3. **Use legacy sync functions** until issues are resolved

The API endpoints support both old and new sync methods during transition.

## Monitoring

Monitor the new API sync:

**Success Indicators:**
- HTTP 200 responses from `/api/content`
- UPSERT actions logged (created/updated)  
- Audio uploads to `/api/upload-audio`
- No database file uploads needed

**Error Indicators:**
- HTTP 4xx/5xx responses from API
- Network timeout errors
- Authentication failures
- Missing content records on dashboard

## Support

**API Client Issues:**
- Check `RENDER_API_URL` and `SYNC_SECRET` environment variables
- Verify network connectivity to Render
- Review API client logs for detailed errors

**Sync Issues:**
- Verify SQLite database exists at `data/ytv2_content.db`
- Check universal schema format conversion
- Monitor Render dashboard for content appearance

**Performance Issues:**
- Adjust retry logic in API client
- Monitor API response times
- Consider batch size optimization

---

**Migration completed**: ✅ API-based sync system operational  
**Legacy system**: ⚠️ Available for rollback if needed  
**Cleanup**: 🧹 Run cleanup script when confident in new system