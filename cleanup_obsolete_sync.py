#!/usr/bin/env python3
"""
Cleanup script for obsolete sync code after API migration.
Identifies and optionally removes files/functions that are no longer needed.
"""

import os
import shutil
from pathlib import Path
from typing import List

def identify_obsolete_files() -> List[Path]:
    """Identify files that are obsolete after API migration."""
    
    obsolete_files = [
        # Database file sync scripts (replaced by API client)
        Path("sync_sqlite_db.py"),
        Path("tools/bulk_sync_to_render.py"),
        
        # Test database sync script
        Path("test_database_sync.py"),
    ]
    
    existing_obsolete = []
    for file_path in obsolete_files:
        if file_path.exists():
            existing_obsolete.append(file_path)
    
    return existing_obsolete

def identify_obsolete_functions() -> List[str]:
    """Identify function names that are obsolete."""
    
    obsolete_functions = [
        # In nas_sync.py - old database upload approach
        "upload-database endpoint usage",
        "upload-report endpoint usage (partially replaced)",
        
        # Dashboard endpoints that are replaced
        "/api/upload-database (replaced by /api/content)",
        "/api/upload-report (partially replaced by /api/content + /api/upload-audio)",
    ]
    
    return obsolete_functions

def check_file_usage(file_path: Path) -> bool:
    """Check if an obsolete file is still being imported or used."""
    
    # Search for imports or references
    import subprocess
    
    try:
        # Search for imports of this file
        result = subprocess.run(
            ['grep', '-r', f'import.*{file_path.stem}', '.', '--include=*.py'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            print(f"⚠️  {file_path} is still imported in:")
            for line in result.stdout.strip().split('\n'):
                print(f"   {line}")
            return True
            
        # Search for direct references to the file
        result = subprocess.run(
            ['grep', '-r', str(file_path), '.', '--include=*.py'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            print(f"⚠️  {file_path} is still referenced in:")
            for line in result.stdout.strip().split('\n')[:5]:  # Show first 5 matches
                print(f"   {line}")
            return True
            
        return False
        
    except Exception as e:
        print(f"❌ Error checking usage of {file_path}: {e}")
        return True  # Conservative - assume it's still used

def create_backup_directory() -> Path:
    """Create backup directory for obsolete files."""
    backup_dir = Path("obsolete_sync_backup")
    backup_dir.mkdir(exist_ok=True)
    return backup_dir

def main():
    """Main cleanup function."""
    print("🧹 YTV2 API Migration Cleanup")
    print("=" * 50)
    
    # Identify obsolete files
    print("\n🔍 Identifying obsolete files...")
    obsolete_files = identify_obsolete_files()
    
    if not obsolete_files:
        print("✅ No obsolete files found")
    else:
        print(f"📋 Found {len(obsolete_files)} potentially obsolete files:")
        for file_path in obsolete_files:
            size = file_path.stat().st_size if file_path.exists() else 0
            print(f"   📄 {file_path} ({size} bytes)")
    
    # Check if files are still being used
    print("\n🔍 Checking for active usage...")
    files_to_keep = []
    files_to_remove = []
    
    for file_path in obsolete_files:
        if check_file_usage(file_path):
            files_to_keep.append(file_path)
            print(f"🔒 Keeping {file_path} - still in use")
        else:
            files_to_remove.append(file_path)
            print(f"🗑️  {file_path} - safe to remove")
    
    # Identify obsolete functions
    print("\n🔍 Identifying obsolete function patterns...")
    obsolete_functions = identify_obsolete_functions()
    
    print("📋 Obsolete patterns after API migration:")
    for func in obsolete_functions:
        print(f"   🔧 {func}")
    
    # Summary and recommendations
    print("\n📊 CLEANUP SUMMARY")
    print("=" * 30)
    print(f"✅ Safe to remove: {len(files_to_remove)} files")
    print(f"⚠️  Keep for now: {len(files_to_keep)} files")
    
    # Offer to create backup and remove safe files
    if files_to_remove:
        print(f"\n🗑️  Files safe to remove:")
        for file_path in files_to_remove:
            print(f"   📄 {file_path}")
        
        response = input("\n❓ Create backup and remove obsolete files? (y/n): ").strip().lower()
        
        if response == 'y':
            # Create backup directory
            backup_dir = create_backup_directory()
            print(f"📁 Created backup directory: {backup_dir}")
            
            # Move files to backup
            for file_path in files_to_remove:
                if file_path.exists():
                    backup_path = backup_dir / file_path.name
                    shutil.move(str(file_path), str(backup_path))
                    print(f"🔄 Moved {file_path} → {backup_path}")
            
            print(f"\n✅ Cleanup complete! {len(files_to_remove)} files backed up and removed.")
            print(f"💾 Backup location: {backup_dir.absolute()}")
            print("🔄 You can restore files from backup if needed.")
            
        else:
            print("🚫 Cleanup cancelled - files kept in place")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 20)
    print("✅ API migration is complete - new system uses:")
    print("   • /api/content (UPSERT) for content creation/updates")
    print("   • /api/upload-audio for MP3 files")
    print("   • RenderAPIClient for all communication")
    print()
    print("🔄 Next steps:")
    print("   1. Test the new API sync with: python test_api_sync.py")
    print("   2. Update Docker containers to use new sync functions")
    print("   3. Monitor sync performance and error rates")
    print("   4. Remove backup files after confirming everything works")

if __name__ == "__main__":
    main()