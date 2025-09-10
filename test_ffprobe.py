#!/usr/bin/env python3
"""
Quick test script to verify ffprobe extraction with enhanced logging
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

from report_generator import get_mp3_duration_seconds

def test_ffprobe():
    # Test with an existing MP3 file
    exports_dir = Path("./exports")
    
    # Find any MP3 file to test with
    mp3_files = list(exports_dir.glob("audio_*.mp3"))
    
    if not mp3_files:
        print("❌ No MP3 files found in exports directory")
        return False
        
    test_file = mp3_files[0]
    print(f"🧪 Testing ffprobe extraction with: {test_file}")
    print(f"🧪 File exists: {test_file.exists()}")
    print(f"🧪 File size: {test_file.stat().st_size} bytes")
    
    duration = get_mp3_duration_seconds(str(test_file))
    
    if duration:
        print(f"🎉 SUCCESS: Extracted duration = {duration} seconds")
        return True
    else:
        print(f"❌ FAILED: Could not extract duration")
        return False

if __name__ == "__main__":
    success = test_ffprobe()
    sys.exit(0 if success else 1)