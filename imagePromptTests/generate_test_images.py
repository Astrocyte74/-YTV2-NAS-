#!/usr/bin/env python3
"""
Generate test images using universal templates across 3 video types.
"""

import asyncio
import json
import requests
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Draw Things API endpoint
DRAW_API_BASE = "http://100.101.80.13:7860/api"

# Output directory
OUTPUT_DIR = Path("/Users/markdarby16/16projects/ytv16/imagePromptTests/test_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Templates
TEMPLATES = {
    "photorealistic": """Photorealistic image capturing the essence of "{title}". The scene conveys: {headline} Use natural lighting, realistic textures, and authentic atmosphere. No text, logos, or writing in the image.""",

    "editorial": """Editorial magazine-style illustration representing "{title}". The image captures: {headline} Use bold composition, expressive lighting, and professional photography style. No text or lettering anywhere in the image.""",

    "cinematic": """Cinematic scene depicting "{title}". The visual conveys: {headline} Use dramatic lighting, film-like composition, and atmospheric depth. Movie still aesthetic. No text or logos.""",

    "minimal": """Clean, minimal composition representing "{title}". The image suggests: {headline} Use simple shapes, negative space, and restrained color palette. Modern aesthetic. No text or symbols.""",

    "symbolic": """Symbolic visualization of "{title}". The image represents the essence of: {headline} Use metaphor, visual storytelling, and conceptual imagery. Abstract but accessible. No text or letters.""",

    "data_viz": """Data visualization or infographic style for "{title}". The visual communicates: {headline} Use charts, graphs, diagrams, or information design. Professional and clear presentation.""",
}

# Video data
VIDEOS = {
    "climate": {
        "title": "These climate change charts are wrong. Here are the real versions.",
        "headline": "This video discusses how climate data can be presented in misleading ways, particularly through the use of 'hockey stick' graphs.",
        "short_name": "climate_charts"
    },
    "shoes": {
        "title": "Can Saucony Beat the Evo SL? | Endorphin Azura Review",
        "headline": "The Saucony Endorphin Azura is a highly anticipated non-plated training shoe designed to bridge the gap between tempo specialists and long-distance grinders.",
        "short_name": "saucony_review"
    },
    "ww2": {
        "title": "State Secrets - Still Classified WWII Subjects",
        "headline": "This summary explores the persistent official secrecy surrounding specific events of World War II nearly 80 years after the conflict ended.",
        "short_name": "ww2_secrets"
    }
}


async def generate_image(prompt: str, template_name: str, video_name: str) -> Dict:
    """Generate an image using Draw Things API."""

    url = f"{DRAW_API_BASE}/telegram/draw"
    payload = {
        "prompt": prompt,
        "steps": 4,
        "width": 512,
        "height": 512
    }

    print(f"Generating: {template_name} - {video_name}")
    print(f"Prompt: {prompt[:100]}...")

    try:
        resp = requests.post(url, json=payload, timeout=120)

        if resp.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}"
            }

        data = resp.json()
        image_url = data.get("url")

        if not image_url:
            return {
                "success": False,
                "error": f"No URL in response: {data}"
            }

        # Get the actual image
        full_image_url = f"{DRAW_API_BASE.rstrip('/api')}{image_url}"
        img_resp = requests.get(full_image_url, timeout=30)

        if img_resp.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to fetch image: HTTP {img_resp.status_code}"
            }

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{video_name}_{template_name}_{timestamp}.png"
        filepath = OUTPUT_DIR / filename

        filepath.write_bytes(img_resp.content)

        result = {
            "success": True,
            "filename": filename,
            "filepath": str(filepath),
            "template": template_name,
            "video": video_name,
            "prompt": prompt,
            "api_response": data,
            "timestamp": timestamp
        }

        print(f"✅ Saved: {filename}")
        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def main():
    """Generate all test images."""

    results = []
    total = len(TEMPLATES) * len(VIDEOS)
    current = 0

    print(f"Generating {total} test images...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)

    for video_key, video_data in VIDEOS.items():
        for template_name, template in TEMPLATES.items():

            current += 1
            print(f"\n[{current}/{total}] Processing: {template_name} × {video_key}")

            # Build the prompt
            prompt = template.format(
                title=video_data["title"],
                headline=video_data["headline"]
            )

            # Generate image
            result = await generate_image(prompt, template_name, video_data["short_name"])
            results.append(result)

            # Small delay between requests
            await asyncio.sleep(2)

    # Save summary
    summary_file = OUTPUT_DIR / "generation_summary.json"
    summary_file.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Summary saved to: {summary_file}")

    # Print statistics
    successful = sum(1 for r in results if r.get("success"))
    failed = total - successful

    print(f"\nStatistics:")
    print(f"  Total: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    if failed > 0:
        print("\nFailed generations:")
        for r in results:
            if not r.get("success"):
                print(f"  - {r.get('template')} × {r.get('video')}: {r.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
