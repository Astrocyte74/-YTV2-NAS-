#!/usr/bin/env python3
"""
Test Audio Upload Script
Quick test for the audio upload fix without full video processing.
"""

import sys
import os
import logging
from pathlib import Path
from modules.render_api_client import create_client_from_env

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_audio_upload(content_id, audio_file_pattern=None):
    """Test uploading audio for a specific content ID."""
    
    print(f"🎵 Testing audio upload for content: {content_id}")
    
    # Find the audio file
    exports_dir = Path("exports")
    if not exports_dir.exists():
        print("❌ exports/ directory not found")
        return False
    
    if audio_file_pattern:
        # Use specific pattern
        audio_files = list(exports_dir.glob(audio_file_pattern))
    else:
        # Auto-detect based on content_id
        video_id = content_id.replace('yt:', '')
        audio_files = list(exports_dir.glob(f"audio_{video_id}_*.mp3"))
    
    if not audio_files:
        print(f"❌ No audio files found for {content_id}")
        print(f"   Searched in: {exports_dir.absolute()}")
        if audio_file_pattern:
            print(f"   Pattern: {audio_file_pattern}")
        else:
            print(f"   Pattern: audio_{content_id.replace('yt:', '')}_*.mp3")
        
        # List available files for reference
        all_audio = list(exports_dir.glob("*.mp3"))
        if all_audio:
            print(f"   Available MP3 files:")
            for f in all_audio[-5:]:  # Show last 5
                print(f"     {f.name}")
        return False
    
    # Use the most recent file if multiple matches
    audio_file = max(audio_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 Using audio file: {audio_file.name}")
    print(f"📊 File size: {audio_file.stat().st_size:,} bytes")
    
    try:
        # Set up environment variables for API client
        render_url = os.getenv('RENDER_DASHBOARD_URL') or os.getenv('RENDER_API_URL')
        if render_url and 'RENDER_API_URL' not in os.environ:
            os.environ['RENDER_API_URL'] = render_url
            print(f"📡 Using Render URL: {render_url}")
        
        # Create API client
        client = create_client_from_env()
        
        # Test connection first
        if not client.test_connection():
            print("❌ Failed to connect to Render API")
            return False
        print("✅ API connection successful")
        
        # Upload the audio
        print(f"📤 Uploading audio...")
        result = client.upload_audio_file(audio_file, content_id)
        
        print(f"✅ Audio upload successful!")
        print(f"📊 Result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Audio upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_audio_upload.py <content_id> [audio_pattern]")
        print()
        print("Examples:")
        print("  python test_audio_upload.py yt:TuEpUrQCOkk")
        print("  python test_audio_upload.py yt:TuEpUrQCOkk 'audio_TuEpUrQCOkk_*.mp3'")
        print("  python test_audio_upload.py yt:TuEpUrQCOkk 'audio_TuEpUrQCOkk_20250912_064452.mp3'")
        sys.exit(1)
    
    content_id = sys.argv[1]
    audio_pattern = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("🧪 Audio Upload Test")
    print("=" * 40)
    
    success = test_audio_upload(content_id, audio_pattern)
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Test completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Check the Dashboard to see if audio appears")
        print("   2. Try playing the audio in the web interface")
        print("   3. Verify the content record shows has_audio: true")
    else:
        print("❌ Test failed - check the error messages above")
        
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()