#!/usr/bin/env python3
"""
Test Z-Image recipe selection and generation with our 3 test videos.
"""

import asyncio
import json
import requests
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Z-Image API
ZIMAGE_BASE = "http://100.101.80.13:8000/api"

# Test content from our imagePromptTests folder
TEST_VIDEOS = {
    "climate": {
        "title": "These climate change charts are wrong. Here are the real versions.",
        "summary": "This video discusses how climate data can be presented in misleading ways, particularly through the use of 'hockey stick' graphs.",
        "file": "These climate change charts are wrong. Here are the real versions..txt"
    },
    "shoes": {
        "title": "Can Saucony Beat the Evo SL? | Endorphin Azura Review",
        "summary": "The Saucony Endorphin Azura is a highly anticipated non-plated training shoe designed to bridge the gap between tempo specialists and long-distance grinders.",
        "file": "Can Saucony Beat the Evo SL.txt"
    },
    "ww2": {
        "title": "State Secrets - Still Classified WWII Subjects",
        "summary": "This summary explores the persistent official secrecy surrounding specific events of World War II nearly 80 years after the conflict ended.",
        "file": "State Secrets Still Classified WWII Subjects.txt"
    }
}


def test_recipe_selection():
    """Test recipe selection for each video."""
    print("=" * 70)
    print("RECIPE SELECTION TEST")
    print("=" * 70)

    from modules.services.summary_image_service import _select_zimage_recipe

    for key, video in TEST_VIDEOS.items():
        content = {"title": video["title"]}
        recipe = _select_zimage_recipe(content, {"summary": video["summary"]}, {})

        print(f"\n{key.upper()}: {video['title'][:50]}...")
        print(f"  Selected Recipe: {recipe}")
        print(f"  Summary: {video['summary'][:80]}...")


async def test_image_generation():
    """Test actual image generation with Z-Image."""
    print("\n" + "=" * 70)
    print("IMAGE GENERATION TEST")
    print("=" * 70)

    from modules.services.summary_image_service import _select_zimage_recipe

    output_dir = Path("/Users/markdarby16/16projects/ytv2/backend/imagePromptTests/zimage_results")
    output_dir.mkdir(exist_ok=True)

    for key, video in TEST_VIDEOS.items():
        content = {"title": video["title"]}
        summary_data = {"summary": video["summary"]}
        analysis = {}

        # Get the selected recipe
        recipe_id = _select_zimage_recipe(content, summary_data, analysis)

        # Build prompt (simplified - Z-Image recipe handles most of it)
        prompt = f"Thumbnail for: {video['title']}. {video['summary'][:200]}"

        print(f"\n{'='*70}")
        print(f"Generating: {key}")
        print(f"Recipe: {recipe_id}")
        print(f"Prompt: {prompt[:100]}...")

        # Generate image using Z-Image API
        try:
            url = f"{ZIMAGE_BASE}/generate"
            payload = {
                "prompt": prompt,
                "recipe_id": recipe_id,
                "advanced": {
                    "width": 768,
                    "height": 768,
                    "steps": 5
                }
            }

            resp = requests.post(url, json=payload, timeout=120)

            if resp.status_code == 200:
                # Save image
                filename = f"{key}_{recipe_id.split('-')[0]}_{Path(recipe_id).stem}.png"
                filepath = output_dir / filename

                # Check if response has image content or JSON with URL
                content_type = resp.headers.get("content-type", "")
                if "image" in content_type:
                    filepath.write_bytes(resp.content)
                    print(f"✅ Saved: {filename}")
                else:
                    # Try to get JSON response
                    try:
                        data = resp.json()
                        print(f"Response: {json.dumps(data, indent=2)[:200]}...")
                    except:
                        print(f"Response: {resp.text[:200]}...")
            else:
                print(f"❌ Failed: HTTP {resp.status_code}")
                print(f"Response: {resp.text[:200]}...")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")


async def main():
    """Run all tests."""
    print("\n🎨 Z-IMAGE THUMBNAIL RECIPE TEST")
    print("=" * 70)

    # Test 1: Recipe selection
    test_recipe_selection()

    # Test 2: Image generation
    print("\n⏳ Starting image generation (this may take a few minutes)...")
    await test_image_generation()

    print("\n✅ Testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
