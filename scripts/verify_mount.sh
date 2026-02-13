#!/bin/bash
# Startup validation script to ensure container is running from correct directory
# This prevents the "wrong directory" issue where data/reports aren't accessible

set -e

echo "🔍 Verifying container mount points..."

# Check if data directory exists and is writable
if [ ! -d "/app/data" ]; then
    echo "❌ ERROR: /app/data directory not found!"
    echo "   Container may be started from wrong directory"
    echo "   Current mount: $(df -h /app | tail -1)"
    exit 1
fi

# Check if reports directory can be created/accessed
REPORTS_DIR="/app/data/reports"
if [ ! -d "$REPORTS_DIR" ]; then
    echo "📁 Creating reports directory: $REPORTS_DIR"
    mkdir -p "$REPORTS_DIR" || {
        echo "❌ ERROR: Cannot create reports directory"
        echo "   /app/data may not be properly mounted"
        exit 1
    }
fi

# Check if directory is writable
if ! touch "$REPORTS_DIR/.write_test" 2>/dev/null; then
    echo "❌ ERROR: Cannot write to $REPORTS_DIR"
    echo "   Volume may be mounted read-only or from wrong directory"
    exit 1
fi
rm -f "$REPORTS_DIR/.write_test"

# Verify we're in the correct project by checking for expected files
if [ ! -f "/app/requirements.txt" ] && [ ! -f "/app/Dockerfile" ]; then
    echo "⚠️  WARNING: /app may not be the correct project directory"
    echo "   Expected files (requirements.txt, Dockerfile) not found"
fi

echo "✅ Mount verification passed"
echo "   Data dir: /app/data ($(df -h /app/data | tail -1 | awk '{print $6}'))"
echo "   Reports dir: $REPORTS_DIR"

exit 0
