# Z-Image Turbo Integration Guide for YTV2

Complete guide for integrating Z-Image Turbo as an alternative image generator to DrawThings.

## Executive Summary

**Current State:** YTV2 uses DrawThings Hub for summary image generation
**Goal:** Add Z-Image Turbo as an alternative provider with fallback support
**Approach:** Adapter pattern with environment variable toggle

---

## Architecture Overview

### Current (DrawThings Only)
```
YTV2 (NAS) → DrawThings Hub → Summary Image
```

### Proposed (Dual Provider)
```
YTV2 (NAS) ┬─→ Z-Image Turbo (preferred)
            │
            └─→ DrawThings Hub (fallback)
```

---

## Quick Integration (5 Steps)

### Step 1: Add Environment Variables

```bash
# Add to your .env or docker-compose.yml

# Z-Image Configuration
ZIMAGE_BASE_URL=http://10.0.4.x:8000  # WireGuard IP of Mac
ZIMAGE_ENABLED=true                    # Enable Z-Image provider
ZIMAGE_GENERATOR=zimage               # Default: "zimage" or "gemini"
ZIMAGE_STYLE_PRESET=Cinematic         # For summary images
ZIMAGE_TIMEOUT=30                      # Request timeout in seconds
ZIMAGE_HEALTH_CHECK=true              # Enable health monitoring

# Provider Selection
SUMMARY_IMAGE_PROVIDERS=zimage,drawthings  # Priority order
IMAGE_GENERATOR_PROVIDER=zimage              # Or: drawthings
```

### Step 2: Install Dependencies

```bash
# If using Python
pip install httpx  # For async HTTP requests

# Or if you already have requests
# pip install requests  # Already installed likely
```

### Step 3: Create Z-Image Adapter

```python
# modules/services/zimage_adapter.py

import asyncio
from typing import Optional, Dict, Any
import httpx
import requests
from datetime import datetime, timedelta

class ZImageClient:
    """Client for Z-Image Turbo API."""

    def __init__(
        self,
        base_url: str = "http://10.0.4.x:8000",
        default_generator: str = "zimage",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_generator = default_generator
        self.timeout = timeout
        self._health_cache = None
        self._health_cache_time = None

    async def health_check(self) -> Dict[str, Any]:
        """Check if Z-Image is reachable and healthy."""
        # Cache for 30 seconds
        if (
            self._health_cache_time
            and datetime.now() - self._health_cache_time < timedelta(seconds=30)
        ):
            return self._health_cache

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    self._health_cache = {"reachable": True}
                    self._health_cache_time = datetime.now()
                    return self._health_cache
        except Exception as e:
            self._health_cache = {"reachable": False, "error": str(e)}
            self._health_cache_time = datetime.now()

        return self._health_cache

    async def generate(
        self,
        prompt: str,
        width: int = 768,
        height: int = 768,
        style_preset: str = "Cinematic",
        generator: Optional[str] = None,
        ephemeral: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate an image using Z-Image API.

        Args:
            prompt: Text prompt for image generation
            width: Image width (256-1536)
            height: Image height (256-1536)
            style_preset: Style preset (None, Cinematic, Anime, etc.)
            generator: "zimage" (local) or "gemini" (cloud)
            ephemeral: If True, return PNG bytes directly (no storage)

        Returns:
            Dictionary with image_url or image_bytes
        """
        generator = generator or self.default_generator
        endpoint = "/api/generate_ephemeral" if ephemeral else "/api/generate"
        url = f"{self.base_url}{endpoint}"

        payload = {
            "prompt": prompt,
            "generator": generator,
            "style_preset": style_preset,
            "advanced": {
                "width": width,
                "height": height,
                "steps": 7,
                "cfg_scale": 0.0,
                "seed": None,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)

            if response.status_code != 200:
                error_detail = await response.aread()
                raise Exception(
                    f"Z-Image error {response.status_code}: {error_detail.decode()}"
                )

            if ephemeral:
                # Return PNG bytes directly
                return {
                    "image_bytes": await response.aread(),
                    "media_type": response.headers.get("content-type"),
                }
            else:
                # Return JSON response
                result = await response.json()
                return {
                    "id": result["id"],
                    "image_url": f"{self.base_url}{result['image_url']}",
                    "prompt": result["prompt"],
                    "duration_sec": result.get("duration_sec", 0),
                    "generator": result.get("generator"),
                    "width": result.get("width"),
                    "height": result.get("height"),
                }

    def generate_sync(
        self,
        prompt: str,
        width: int = 768,
        height: int = 768,
        style_preset: str = "Cinematic",
        generator: Optional[str] = None,
        ephemeral: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous version of generate()."""
        generator = generator or self.default_generator
        endpoint = "/api/generate_ephemeral" if ephemeral else "/api/generate"
        url = f"{self.base_url}{endpoint}"

        payload = {
            "prompt": prompt,
            "generator": generator,
            "style_preset": style_preset,
            "advanced": {
                "width": width,
                "height": height,
                "steps": 7,
                "cfg_scale": 0.0,
                "seed": None,
            },
        }

        response = requests.post(url, json=payload, timeout=self.timeout)

        if response.status_code != 200:
            error_detail = response.json()
            raise Exception(
                f"Z-Image error {response.status_code}: {error_detail.get('detail')}"
            )

        if ephemeral:
            return {
                "image_bytes": response.content,
                "media_type": response.headers.get("content-type"),
            }
        else:
            result = response.json()
            return {
                "id": result["id"],
                "image_url": f"{self.base_url}{result['image_url']}",
                "prompt": result["prompt"],
                "duration_sec": result.get("duration_sec", 0),
                "generator": result.get("generator"),
                "width": result.get("width"),
                "height": result.get("height"),
            }
```

### Step 4: Create Provider Adapter

```python
# modules/services/image_provider_adapter.py

import os
import asyncio
from typing import Optional, Dict, Any
from zimage_adapter import ZImageClient
# Import your existing DrawThings service
# from modules.services.drawthings_service import DrawThingsClient

class ImageProviderAdapter:
    """
    Unified adapter for image generation providers.
    Supports Z-Image Turbo and DrawThings Hub with automatic fallback.
    """

    def __init__(self):
        # Get configuration from environment
        self.zimage_enabled = os.getenv("ZIMAGE_ENABLED", "true").lower() == "true"
        self.zimage_base_url = os.getenv("ZIMAGE_BASE_URL", "http://localhost:8000")
        self.zimage_generator = os.getenv("ZIMAGE_GENERATOR", "zimage")
        self.zimage_style = os.getenv("ZIMAGE_STYLE_PRESET", "Cinematic")

        # Provider priority
        self.providers = (
            os.getenv("SUMMARY_IMAGE_PROVIDERS", "zimage,drawthings")
            .lower()
            .replace(" ", "")
            .split(",")
        )

        # Initialize clients
        self.zimage_client = ZImageClient(
            base_url=self.zimage_base_url,
            default_generator=self.zimage_generator,
        ) if self.zimage_enabled else None

        # TODO: Initialize DrawThings client if needed
        # self.drawthings_client = DrawThingsClient(...)

    async def generate_summary_image(
        self,
        content: Dict[str, Any],
        image_mode: str = "ai1",
        use_ephemeral: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a summary image using the first available provider.

        Args:
            content: Summary content dict with transcript, title, etc.
            image_mode: Image style mode ("ai1", "ai2", etc.)
            use_ephemeral: Use ephemeral generation (faster, no storage)

        Returns:
            Dictionary with image_url and metadata
        """
        # Transform content to Z-Image prompt
        prompt = self._content_to_prompt(content, image_mode)
        style_preset = self._image_mode_to_style(image_mode)

        # Try providers in priority order
        for provider in self.providers:
            if provider == "zimage" and self.zimage_client:
                try:
                    # Check health first
                    health = await self.zimage_client.health_check()
                    if not health.get("reachable"):
                        print(f"⚠️  Z-Image not reachable, trying next provider")
                        continue

                    # Generate image
                    result = await self.zimage_client.generate(
                        prompt=prompt,
                        width=1024,
                        height=1024,
                        style_preset=style_preset,
                        ephemeral=use_ephemeral,
                    )

                    return {
                        "provider": "zimage",
                        "image_url": result.get("image_url"),
                        "image_bytes": result.get("image_bytes"),
                        "duration_sec": result.get("duration_sec"),
                        "generator": result.get("generator"),
                        "success": True,
                    }

                except Exception as e:
                    print(f"❌ Z-Image generation failed: {e}")
                    continue

            elif provider == "drawthings":
                # TODO: Implement DrawThings fallback
                print("⚠️  DrawThings fallback not yet implemented")
                continue

        # All providers failed
        raise Exception("All image providers failed")

    def _content_to_prompt(self, content: Dict[str, Any], image_mode: str) -> str:
        """
        Transform summary content into an image prompt.

        This is where you customize the prompt based on your content.
        """
        title = content.get("title", "Video Summary")
        description = content.get("description", "")

        # Customize prompt based on image_mode
        if image_mode == "ai1":
            # Cinematic movie poster style
            return f"A professional movie poster for '{title}'. {description}. Cinematic lighting, dramatic composition, high quality, detailed, 4k"
        elif image_mode == "ai2":
            # Digital art style
            return f"Digital artwork representing '{title}'. {description}. Vibrant colors, stylized, artistic, detailed"
        else:
            # Default cinematic style
            return f"A cinematic scene depicting '{title}'. {description}. Professional photography, dramatic lighting, high quality"

    def _image_mode_to_style(self, image_mode: str) -> str:
        """Map image_mode to Z-Image style preset."""
        style_map = {
            "ai1": "Cinematic",
            "ai2": "Digital Art",
            "ai3": "Fantasy Art",
            "ai4": "Photographic",
        }
        return style_map.get(image_mode, "Cinematic")
```

### Step 5: Update Your Service

```python
# In your existing summary_image_service.py or equivalent

from modules.services.image_provider_adapter import ImageProviderAdapter

# Initialize the adapter
provider_adapter = ImageProviderAdapter()

async def generate_summary_image(content: Dict[str, Any], image_mode: str = "ai1"):
    """
    Generate summary image using configured provider.
    """
    try:
        result = await provider_adapter.generate_summary_image(
            content=content,
            image_mode=image_mode,
            use_ephemeral=False,  # Set to True for Telegram bot one-offs
        )

        if result["success"]:
            print(f"✅ Image generated via {result['provider']} in {result.get('duration_sec', 0):.2f}s")

            # If ephemeral, handle image_bytes
            if result.get("image_bytes"):
                # Save to your storage
                image_path = await save_image(result["image_bytes"])
                return {"image_url": image_path, **result}

            return result

    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        raise
```

---

## Complete Feature Comparison

### Z-Image Turbo vs DrawThings

| Feature | Z-Image Turbo | DrawThings | Recommendation |
|---------|---------------|------------|----------------|
| **Deployment** | Mac Silicon container | Linux/Mac/Windows | Z-Image requires Mac (or use Gemini cloud) |
| **Local Generation** | ✅ Free, fast enough | ✅ Depends on hardware | Both good for local |
| **Cloud Generation** | ✅ Gemini Nano Banana | ❌ Unknown | Z-Image has cloud option |
| **Ephemeral Mode** | ✅ No disk save | ❌ Unknown | Z-Image better for bots |
| **Recipe System** | ✅ Built-in | ❌ Unknown | Z-Image for consistency |
| **Prompt Enhancement** | ✅ Ollama/OpenRouter | ❌ Unknown | Z-Image for better prompts |
| **LoRA Support** | ✅ Yes | ❌ Unknown | Z-Image for custom styles |
| **Stealth Mode** | ✅ Encrypted storage | ❌ Unknown | Z-Image for privacy |
| **Health Check** | ✅ `/health` | ❌ Unknown | Both should have it |
| **API Documentation** | ✅ Comprehensive | ❌ Unknown | Z-Image well documented |

---

## Request/Response Mapping

### DrawThings Format (Your Current)

```python
{
    "mode": "summary_image",
    "image_mode": "ai1",
    "content": {
        "title": "Video Title",
        "description": "Summary description",
        # ... other fields
    }
}
```

### Z-Image Format

```python
{
    "prompt": "A professional movie poster for 'Video Title'...",
    "generator": "zimage",
    "style_preset": "Cinematic",
    "advanced": {
        "width": 1024,
        "height": 1024,
        "steps": 7,
        "cfg_scale": 0.0,
        "seed": None,
    }
}
```

### Transformation Function

```python
def drawthings_to_zimage(
    content: Dict[str, Any],
    image_mode: str = "ai1",
    generator: str = "zimage"
) -> Dict[str, Any]:
    """
    Transform DrawThings request to Z-Image format.
    """
    title = content.get("title", "")
    description = content.get("description", "")
    channel = content.get("channel", "")

    # Build prompt from content
    prompt_parts = [f"A cinematic image for '{title}'"]

    if description:
        prompt_parts.append(description)

    if channel:
        prompt_parts.append(f"YouTube channel: {channel}")

    prompt = ". ".join(prompt_parts)

    # Map image_mode to style preset
    style_map = {
        "ai1": "Cinematic",
        "ai2": "Digital Art",
        "ai3": "Fantasy Art",
        "ai4": "Photographic",
    }
    style_preset = style_map.get(image_mode, "Cinematic")

    return {
        "prompt": prompt,
        "generator": generator,
        "style_preset": style_preset,
        "advanced": {
            "width": 1024,
            "height": 1024,
            "steps": 7,
            "cfg_scale": 0.0,
        },
    }
```

---

## Advanced Features

### 1. Using Z-Image Recipes

If you want consistent styling, use Z-Image's recipe system:

```python
async def get_recipes() -> Dict[str, Any]:
    """Fetch available recipes from Z-Image."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{zimage_base_url}/api/recipes")
        return await response.json()

async def use_recipe_for_summary(recipe_id: str, content: Dict[str, Any]):
    """Use a Z-Image recipe for generation."""
    # Fetch recipes
    recipes = await get_recipes()
    recipe = next((r for r in recipes["recipes"] if r["id"] == recipe_id), None)

    if recipe:
        # Replace focus token with video title
        prompt = recipe["prompt"].replace(recipe.get("focus_token", "SUBJECT"), content["title"])

        return await zimage_client.generate(
            prompt=prompt,
            style_preset="None",
            advanced={"width": 1024, "height": 1024, "steps": 7}
        )
```

### 2. Auto-Enhance Prompts

Use Z-Image's AI enhancement to improve prompts:

```python
async def generate_with_enhancement(base_prompt: str):
    """Generate with AI prompt enhancement."""
    # First enhance the prompt
    async with httpx.AsyncClient() as client:
        enhance_response = await client.post(
            f"{zimage_base_url}/api/enhance_prompt",
            json={
                "prompt": base_prompt,
                "enhancement_mode": "simple"
            }
        )
        enhanced = await enhance_response.json()

    # Then generate with enhanced prompt
    return await zimage_client.generate(
        prompt=enhanced["prompt"],
        style_preset="Cinematic"
    )
```

### 3. Generator Selection (Local vs Cloud)

```python
async def smart_generate(prompt: str, prefer_fast: bool = False):
    """
    Automatically choose between local and cloud generation.
    """
    if prefer_fast:
        # Use Gemini Nano Banana (cloud, fast)
        return await zimage_client.generate(
            prompt=prompt,
            generator="gemini",
            generator_model="nano-banana"
        )
    else:
        # Use Z-Image Turbo (local, free)
        return await zimage_client.generate(
            prompt=prompt,
            generator="zimage"
        )
```

---

## Health Check Integration

### Add to Your Monitoring

```python
# modules/services/health_monitor.py

async def check_all_providers():
    """Check health of all image providers."""
    providers = {
        "zimage": os.getenv("ZIMAGE_BASE_URL"),
        # "drawthings": os.getenv("DRAWTINGS_HUB_URL"),
    }

    results = {}

    for name, base_url in providers.items():
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    data = await response.json()
                    results[name] = {
                        "status": "healthy",
                        "uptime": data.get("uptime_seconds"),
                        "timestamp": data.get("timestamp")
                    }
                else:
                    results[name] = {"status": "unhealthy", "code": response.status_code}
        except Exception as e:
            results[name] = {"status": "error", "message": str(e)}

    return results
```

---

## Error Handling & Fallback

### Automatic Fallback Strategy

```python
class ResilientImageGenerator:
    """Image generation with automatic fallback."""

    def __init__(self):
        self.adapter = ImageProviderAdapter()
        self.retry_attempts = 2

    async def generate_with_fallback(
        self,
        content: Dict[str, Any],
        image_mode: str = "ai1"
    ):
        """
        Try primary provider, fallback to alternative.
        """
        for attempt in range(self.retry_attempts):
            try:
                result = await self.adapter.generate_summary_image(
                    content=content,
                    image_mode=image_mode
                )

                if result["success"]:
                    return result

            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")

                if attempt == self.retry_attempts - 1:
                    # Last attempt failed, try with different provider
                    continue

        raise Exception("All image generation attempts failed")
```

---

## Queue Draining Strategy

### For Pending DrawThings Jobs

```python
async def migrate_pending_jobs():
    """
    Re-process pending jobs with Z-Image.
    """
    # Fetch pending jobs
    pending_jobs = await get_pending_summary_jobs()

    migrated = 0
    failed = 0

    for job in pending_jobs:
        try:
            # Transform job to Z-Image format
            content = job.get("content", {})
            image_mode = job.get("image_mode", "ai1")

            # Generate with Z-Image
            result = await provider_adapter.generate_summary_image(
                content=content,
                image_mode=image_mode
            )

            if result["success"]:
                # Update job with new image URL
                await update_job_with_image(job["id"], result["image_url"])
                migrated += 1
            else:
                failed += 1

        except Exception as e:
            print(f"❌ Failed to migrate job {job.get('id')}: {e}")
            failed += 1

    print(f"✅ Migration complete: {migrated} succeeded, {failed} failed")
```

---

## Docker Deployment (Optional)

If you want to run Z-Image in a container on your Mac:

```dockerfile
# Dockerfile.zimage
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run with GPU support if available
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  zimage:
    build:
      context: .
      dockerfile: Dockerfile.zimage
    ports:
      - "8000:8000"
    volumes:
      - ./generated:/app/generated
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: apple
              count: 1
              capabilities: [gpu]
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Refused
```
Error: Connection refused
```
**Solution:** Check if Z-Image is running on Mac:
```bash
curl http://10.0.4.x:8000/health
```

#### 2. CORS Errors
```
Error: CORS policy blocked
```
**Solution:** Add your NAS IP to Z-Image's CORS origins in `backend/main.py`:
```python
origins = [
    "http://localhost:5173",
    "http://10.0.4.x:YOUR_NAS_PORT",  # Add this
]
```

#### 3. Timeout
```
Error: Request timeout after 30s
```
**Solution:** Increase timeout or use Gemini cloud generator:
```python
result = await zimage_client.generate(
    prompt=prompt,
    generator="gemini",  # Faster
    timeout=60
)
```

#### 4. Model Not Loaded
```
Error: Model not loaded
```
**Solution:** Z-Image warms up on first request. Wait for initialization or trigger pre-warm.

---

## Performance Comparison

### Generation Times (Approximate)

| Provider | Resolution | Time | Cost |
|----------|------------|------|------|
| Z-Image Local | 1024×1024 | 3-7s | Free |
| Z-Image Local | 768×768 | 2-4s | Free |
| Gemini Nano Banana | 1024×1024 | 1-2s | ~$0.01 |
| Gemini Nano Pro | 1024×1024 | 1-2s | ~$0.03 |

---

## Recommendations

### For Your Use Case (Summary Images)

1. **Primary Provider:** Z-Image Turbo (local)
   - Free for frequent use
   - Good quality for summary cards
   - No API latency

2. **Fallback:** DrawThings (existing)
   - Keep for redundancy
   - Use when Z-Image is down

3. **Cloud Option:** Gemini Nano Banana
   - Use when you need speed
   - For urgent/time-sensitive requests
   - Cost-conscious for low volume

4. **Features to Enable:**
   - ✅ Recipes for consistent summary card styling
   - ✅ Auto-enhance for better prompt quality
   - ✅ Ephemeral mode for Telegram bot
   - ✅ Stored mode for dashboard images
   - ❌ Stealth mode (not needed for your use case)

---

## Testing

### Test Integration

```python
import asyncio
from modules.services.image_provider_adapter import ImageProviderAdapter

async def test_integration():
    adapter = ImageProviderAdapter()

    # Test data
    content = {
        "title": "Amazing Video Summary",
        "description": "An in-depth look at the latest tech trends",
        "channel": "Tech Channel"
    }

    # Test generation
    try:
        result = await adapter.generate_summary_image(
            content=content,
            image_mode="ai1",
            use_ephemeral=False
        )

        print(f"✅ Success! Image: {result['image_url']}")
        print(f"   Provider: {result['provider']}")
        print(f"   Time: {result['duration_sec']:.2f}s")

    except Exception as e:
        print(f"❌ Failed: {e}")

# Run test
asyncio.run(test_integration())
```

---

## Next Steps

1. **Install dependencies:** `pip install httpx`
2. **Add environment variables** to your `.env`
3. **Copy the adapter code** to your project
4. **Update your service** to use the adapter
5. **Test with pending jobs**
6. **Monitor and iterate** based on results

---

## Questions?

Refer to:
- `API_DOCUMENTATION.md` - Full API reference
- `API_EXAMPLES.md` - Code examples in multiple languages
- Z-Image Health Check: `http://10.0.4.x:8000/health`
