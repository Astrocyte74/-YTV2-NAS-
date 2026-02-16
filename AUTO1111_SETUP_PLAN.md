# Automatic1111 Setup on i9 Mac for YTV2 Image Generation Fallback

## Status: ✅ COMPLETE (February 2026)

The i9 Mac is now configured as a fallback image generation provider when the M4 Mac is unavailable.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     YTV2 Backend Image Queue                     │
│                  (ytv2/backend/data/image_queue/)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────▼──────────────┐
        │  Image Queue Processor    │
        │  (drain_image_queue.py)   │
        └────────────┬──────────────┘
                     │
        ┌────────────▼────────────────────────────────┐
        │         Provider Selection (Priority)        │
        │  1. DrawThings (M4 Mac) - via TTSHUB_API_BASE│
        │  2. Automatic1111 (i9 Mac) - port 7860       │
        │  3. Z-Image (cloud) - ZIMAGE_BASE_URL        │
        └──────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables

Add to your `.env` or `.env.nas`:

```bash
# Provider priority (comma-separated, first available wins)
SUMMARY_IMAGE_PROVIDERS=drawthings,auto1111,zimage

# Automatic1111 on i9 Mac (localhost when running on same machine)
AUTOMATIC1111_BASE_URL=http://127.0.0.1:7860

# Optional: Override default model
AUTO1111_MODEL=sd_xl_base_1.0.safetensors

# Optional: Override generation settings
AUTO1111_STEPS=8
AUTO1111_CFG_SCALE=7.5
```

### Provider Priority

The system tries providers in order:
1. **drawthings** - M4 Mac via hub proxy (fastest)
2. **auto1111** - i9 Mac local (fallback)
3. **zimage** - Cloud service (last resort)

When M4 Mac is offline, the i9 Mac automatically takes over.

---

## Models Installed

| Model | Size | Status | Notes |
|-------|------|--------|-------|
| SDXL Base 1.0 | 6.5GB | ✅ Working | Default for thumbnails |
| SDXL Lightning LoRA | 376MB | ✅ Available | 8-step fast generation |
| Flux.1 Schnell | 22GB | ⚠️ Downloaded | May need extension for full support |

**Recommended:** SDXL Base + Lightning LoRA (8 steps, ~25s on Intel i9)

---

## Files Created/Modified

### Created
- `/Users/markdarby16/16projects/ytv2/backend/modules/services/auto1111_service.py` - Automatic1111 API client
- `~/start-auto1111.sh` - Launch script
- `~/stable-diffusion/stable-diffusion-webui/webui-user.sh` - Configuration

### Modified
- `summary_image_service.py` - Added auto1111 provider support

---

## Usage

### Start Automatic1111
```bash
~/start-auto1111.sh
```

### Check if Running
```bash
lsof -i :7860
curl http://127.0.0.1:7860/sdapi/v1/sd-models
```

### Test Image Generation
```bash
cd /Users/markdarby16/16projects/ytv2/backend
python3 -c "
import asyncio
from modules.services import auto1111_service

async def test():
    health = await auto1111_service.fetch_auto1111_health('http://127.0.0.1:7860')
    print(f'Health: {health}')

    result = await auto1111_service.generate_image(
        'http://127.0.0.1:7860',
        'A beautiful sunset over mountains',
        width=512, height=512, steps=8
    )
    print(f'Generated: {len(result[\"image_bytes\"])} bytes in {result[\"duration_sec\"]}s')

asyncio.run(test())
"
```

### Configure Provider Priority
```bash
# Use auto1111 as primary (for testing)
export SUMMARY_IMAGE_PROVIDERS=auto1111

# Use drawthings first, auto1111 as fallback
export SUMMARY_IMAGE_PROVIDERS=drawthings,auto1111

# Use all three in priority order
export SUMMARY_IMAGE_PROVIDERS=drawthings,auto1111,zimage
```

---

## Performance

| Hardware | Model | Resolution | Steps | Time |
|----------|-------|------------|-------|------|
| M4 Mac (DrawThings) | Flux.1 Schnell | 384x384 | 6 | ~5-10s |
| **i9 Mac (Auto1111)** | SDXL Base | 512x512 | 8 | ~25s |
| i9 Mac (Auto1111) | SDXL Base | 384x384 | 8 | ~15s |

Intel Mac is slower than Apple Silicon, but sufficient for fallback processing.

---

## Troubleshooting

### Automatic1111 not starting
```bash
# Check for port conflicts
lsof -i :7860

# Check logs
tail -100 ~/stable-diffusion/stable-diffusion-webui/log/*
```

### Image generation fails
```bash
# Verify API is accessible
curl http://127.0.0.1:7860/sdapi/v1/sd-models

# Check model is loaded
curl http://127.0.0.1:7860/sdapi/v1/options | python3 -c "import sys,json; print(json.load(sys.stdin).get('sd_model_checkpoint'))"
```

### Provider not being selected
```bash
# Check environment variables
echo $SUMMARY_IMAGE_PROVIDERS
echo $AUTOMATIC1111_BASE_URL

# Check health
python3 -c "
import asyncio
from modules.services import auto1111_service
print(asyncio.run(auto1111_service.fetch_auto1111_health('http://127.0.0.1:7860')))
"
```

---

## Integration Details

### auto1111_service.py

Provides:
- `fetch_auto1111_health()` - Health check with caching
- `fetch_available_models()` - List installed models
- `switch_model()` - Change active model
- `generate_image()` - Generate image via txt2img API

### summary_image_service.py Changes

1. Added `auto1111` to provider list parsing
2. Added health check for auto1111
3. Added generation logic for auto1111 provider

---

## Notes

- Automatic1111 must be running for the fallback to work
- Health checks are cached for 30 seconds
- Model switching happens automatically before generation
- Failed generations are queued for retry
