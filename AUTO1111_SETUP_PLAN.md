# Automatic1111 Setup on i9 Mac for YTV2 Image Generation Fallback

## Overview
Set up Automatic1111 (Stable Diffusion Web UI) on the i9 Mac to process image generation queue when the M4 Mac is unavailable.

**Goal:** Enable 24/7 image generation processing with automatic failover from M4 Mac (DrawThings) → i9 Mac (Automatic1111)

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
        │         Provider Selection                  │
        │  1. Try DrawThings (M4 Mac) - port 7861     │
        │  2. Fallback to Automatic1111 (i9 Mac)       │
        └──────────────────────────────────────────────┘
                     │
        ┌────────────▼────────────────────────────────┐
        │         Auto1111 API (i9 Mac)               │
        │         http://127.0.0.1:7860               │
        └──────────────────────────────────────────────┘
```

---

## Prerequisites

### i9 Mac Requirements
- **OS:** macOS 15.4.1 (already installed)
- **Arch:** x86_64 Intel
- **Python:** 3.10 or 3.11 (will install via conda/mamba)
- **RAM:** 8GB+ minimum, 16GB+ recommended
- **Storage:** 50GB+ free space (for models)

### Software to Install
1. **Miniforge** or **Miniconda** (Python environment)
2. **Git** (to clone Automatic1111)
3. **Automatic1111** (Stable Diffusion Web UI)

---

## Installation Steps

### Step 1: Install Python Environment

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Miniforge (conda for Intel Mac)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
bash Miniforge3-MacOSX-x86_64.sh

# Restart shell or run:
source ~/.zshrc
```

### Step 2: Clone Automatic1111

```bash
# Create directory for Stable Diffusion
mkdir -p ~/stable-diffusion
cd ~/stable-diffusion

# Clone Automatic1111
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
```

### Step 3: Download Models

**IMPORTANT:** We'll use **Flux.1 Schnell** (same as M4 Mac DrawThings) for fair speed comparison.

```bash
# Create models directory
mkdir -p ~/stable-diffusion/stable-diffusion-webui/models/stable-diffusion
mkdir -p ~/stable-diffusion/stable-diffusion-webui/models/CLIP

# Download Flux.1 Schnell model (~12GB)
# This matches what DrawThings on M4 Mac uses
wget https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors \
  -P ~/stable-diffusion/stable-diffusion-webui/models/stable-diffusion/

# Download CLIP model (required for Flux text encoding)
wget https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin \
  -P ~/stable-diffusion/stable-diffusion-webui/models/CLIP/

# Download VAE model (required for Flux decoding)
wget https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors \
  -P ~/stable-diffusion/stable-diffusion-webui/models/vae/

# Download CLIP-L model (alternative CLIP for Flux)
wget https://huggingface.co/comfyanonymous/clip_vision_g/resolve/main/clip_vision_g.safetensors \
  -P ~/stable-diffusion/stable-diffusion-webui/models/CLIP/

# Download T5 XXL text encoder (required for Flux)
wget https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors \
  -P ~/stable-diffusion/stable-diffusion-webui/models/

# Alternative: Use huggingface-cli for easier downloads
# pip install huggingface-hub
# huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir ~/stable-diffusion/stable-diffusion-webui/models/
```

**Alternative: SDXL Lightning** (faster, lower quality, ~2GB)
```bash
wget https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors \
  -P ~/stable-diffusion/stable-diffusion-webui/models/stable-diffusion/
```

**Model Comparison:**
| Model | Size | Speed | Quality | Notes |
|-------|------|-------|--------|-------|
| **Flux.1 Schnell** | ~12GB | Medium | Excellent | **M4 Match (recommended)** |
| SDXL Lightning | ~2GB | Very Fast | Good | Fallback option |
| SDXL Turbo | ~2GB | Very Fast | Good | Quick testing |
| SD 1.5 | ~4GB | Fast | Medium | Older model |

### Step 4: Create Launch Script

```bash
# Create launch script with Flux.1 Schnell settings
cat > ~/start-auto1111.sh << 'EOF'
#!/bin/bash
cd ~/stable-diffusion/stable-diffusion-webui

# Set environment for Intel Mac optimization
export PYTORCH_ENABLE_MPS_FALLBACK=1
export COMMANDLINE_ARGS=(
    --xformers
    --opt-sdp-attention
    --listen
    --port 7860
    --api
    --skip-python-version-check
    --skip-torch-cuda-test
    --enable-inpainting
    --api-server-port 7860

    # Teacache settings (if available via extension)
    # Note: Teacache may require additional extension installation
)

# Set default model to Flux.1 Schnell
export SD_MODEL="flux1-schnell.safetensors"

# Log startup
echo "Starting Automatic1111 for YTV2 Image Generation..."
echo "Model: Flux.1 Schnell (matching M4 Mac DrawThings)"
echo "Settings: 384x384, 6 steps, guidance 7.5"
echo "API: http://127.0.0.1:7860/sdapi/v1/txt2img"
echo ""

./webui.sh "${COMMANDLINE_ARGS[@]}"
EOF

chmod +x ~/start-auto1111.sh
```

### Step 5: Create LaunchDaemon for Auto-start

```bash
# Create plist for auto-start on boot
sudo cat > /Library/LaunchDaemons/com.automatic1111.webui.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.automatic1111.webui</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/markdarby16/start-auto1111.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>/Users/markdarby16/stable-diffusion/stable-diffusion-webui</string>
    <key>StandardOutPath</key>
    <string>/Users/markdarby16/Logs/auto1111.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/markdarby16/Logs/auto1111-error.log</string>
</dict>
</plist>
EOF

# Create log directory
mkdir -p ~/Logs

# Load the service
sudo launchctl load -w /Library/LaunchDaemons/com.automatic1111.webui.plist
```

---

## YTV2 Backend Configuration

### Step 6: Update Image Service Configuration

File: `/Users/markdarby16/16projects/ytv2/backend/.env.nas`

```bash
# Add these environment variables
AUTOMATIC1111_BASE_URL=http://127.0.0.1:7860
AUTOMATIC1111_ENABLED=true
IMAGE_GENERATION_PRIORITY=drawthings,automatic1111
```

### Step 7: Add Fallback Logic to Image Queue

File: `/Users/markdarby16/16projects/ytv2/backend/modules/services/draw_service.py`

Add function to try DrawThings first, then Automatic1111:

```python
async def try_generate_with_fallback(prompt, **kwargs):
    """Try DrawThings first, fall back to Automatic1111."""

    # Try DrawThings (M4 Mac)
    try:
        result = await generate_with_drawthings(prompt, **kwargs)
        return result
    except Exception as e:
        logger.warning(f"DrawThings failed: {e}, trying Automatic1111")

    # Fallback to Automatic1111 (i9 Mac)
    try:
        result = await generate_with_auto1111(prompt, **kwargs)
        return result
    except Exception as e:
        logger.error(f"Automatic1111 also failed: {e}")
        raise
```

### Step 8: Add Automatic1111 Client

File: `/Users/markdarby16/16projects/ytv2/backend/modules/services/auto1111_service.py`

```python
import requests
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Automatic1111Client:
    """Client for Automatic1111 API."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("AUTOMATIC1111_BASE_URL", "http://127.0.0.1:7860")

    async def generate(self, prompt: str, **kwargs) -> bytes:
        """Generate image with Automatic1111."""

        payload = {
            "prompt": prompt,
            "steps": kwargs.get("steps", 20),
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "cfg_scale": kwargs.get("cfg_scale", 7),
            "sampler_name": kwargs.get("sampler", "DPM++ 2M Karras"),
        }

        response = requests.post(
            f"{self.base_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=kwargs.get("timeout", 120)
        )
        response.raise_for_status()

        import base64
        data = response.json()
        image_data = base64.b64decode(data["images"][0])

        return image_data

    async def health_check(self) -> bool:
        """Check if Automatic1111 is running."""
        try:
            response = requests.get(f"{self.base_url}/sdapi/v1/sd-models", timeout=5)
            return response.status_code == 200
        except:
            return False
```

---

## Testing

### Test 1: Verify Automatic1111 Running

```bash
curl http://127.0.0.1:7860/sdapi/v1/sd-models
```

Expected output: JSON with model info

### Test 2: Test Image Generation

```bash
curl -X POST http://127.0.0.1:7860/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat",
    "steps": 10
  }'
```

### Test 3: Test YTV2 Integration

```bash
# Manually trigger image generation
cd /Users/markdarby16/16projects/ytv2/backend
docker exec youtube-summarizer-bot python -c "
from modules.services import auto1111_service
import asyncio

async def test():
    client = auto1111_service.Automatic1111Client()
    is_healthy = await client.health_check()
    print(f'Auto1111 healthy: {is_healthy}')

    if is_healthy:
        # Test generation
        image = await client.generate('a red apple', steps=5)
        print(f'Generated {len(image)} bytes')

asyncio.run(test())
"
```

---

## Teacache Configuration

**What is Teacache?**
Teacache is a caching mechanism for Stable Diffusion that skips redundant computations. It caches intermediate steps and reuses them when similar prompts are processed.

**DrawThings Teacache Settings (from M4 Mac):**
- Enabled: ✓
- Start Step: 5-6
- Threshold: 0.06
- Max Skip Steps: 3

**Automatic1111 Teacache Setup:**
```bash
# Teacache may need to be enabled via extensions or command line args
# Add to COMMANDLINE_ARGS in launch script:
--teacache-enabled
--teacache-start-step 5
--teacache-threshold 0.06
--teacache-max-skip-steps 3
```

**Note:** Teacache support in Automatic1111 may be limited compared to DrawThings. We'll implement what's available.

---

## Troubleshooting

### Issue: Port 7860 already in use
```bash
# Find what's using the port
lsof -i :7860

# Kill the process
kill -9 <PID>
```

### Issue: Out of memory errors
- Use smaller model (SDXL Lightning instead of full SDXL)
- Reduce image size (512x512 instead of 768x768)
- Reduce steps (15-20 instead of 30-50)

### Issue: Very slow generation
- Normal for Intel Mac (expect 30-60 seconds per image)
- Use SDXL Lightning for faster generation
- Consider using a lower resolution

### Issue: API errors
```bash
# Check Auto1111 logs
tail -f ~/Logs/auto1111.log

# Check if service is running
sudo launchctl list | grep automatic1111

# Restart service
sudo launchctl unload /Library/LaunchDaemons/com.automatic1111.webui.plist
sudo launchctl load /Library/LaunchDaemons/com.automatic1111.webui.plist
```

---

## Image Generation Settings (Match M4 Mac DrawThings)

**IMPORTANT:** These settings match DrawThings on M4 Mac for fair speed comparison.

### Flux.1 Schnell Settings (Exact Match)
```python
{
    "prompt": "<enhanced_prompt>",
    "seed": -1,
    "width": 384,
    "height": 384,
    "num_inference_steps": 6,
    "guidance_scale": 7.5,
    "shift": 1,
    "scheduler": "euler",  # Flux.1 default

    # Teacache settings (if supported)
    "teacache_enabled": true,
    "teacache_start_step": 5,
    "teacache_threshold": 0.06,
    "teacache_max_skip_steps": 3,

    # Mask settings
    "mask_blur": 2.5,
    "mask_blur_outset": 0
}
```

### Environment Variables for YTV2 Backend

```bash
# Add to .env.nas
FLUX_MODEL_PATH=~/stable-diffusion/stable-diffusion-webui/models/stable-diffusion/flux1-schnell.safetensors
FLUX_DEFAULT_WIDTH=384
FLUX_DEFAULT_HEIGHT=384
FLUX_DEFAULT_STEPS=6
FLUX_DEFAULT_GUIDANCE=7.5
FLUX_DEFAULT_SHIFT=1
FLUX_TEACACHE_ENABLED=true
FLUX_TEACACHE_START_STEP=5
FLUX_TEACACHE_THRESHOLD=0.06
FLUX_TEACACHE_MAX_SKIP=3
```

### API Payload for Automatic1111

```python
payload = {
    "prompt": prompt,
    "seed": -1,
    "width": 384,
    "height": 384,
    "steps": 6,
    "cfg_scale": 7.5,
    "sampler_name": "Euler",
    "scheduler": "Euler",

    # Override for model
    "override_settings": {
        "sd_model_checkpoint": "flux1-schnell.safetensors"
    }
}
```

---

## Performance Expectations (Flux.1 Schnell @ 384x384)

| Hardware | Model | Resolution | Steps | Teacache | Time per Image (Expected) |
|----------|-------|------------|-------|----------|-------------------------|
| M4 Mac (DrawThings) | Flux.1 Schnell | 384x384 | 6 | ✓ (5-6) | ~5-10 seconds |
| i9 Mac (Auto1111) | Flux.1 Schnell | 384x384 | 6 | ✓ (5-6) | ~30-90 seconds (Intel) |
| i9 Mac (Auto1111) | SDXL Lightning | 512x512 | 8 | - | ~20-40 seconds (faster alt) |

---

## Files Modified/Created

### Created:
- `~/start-auto1111.sh` - Launch script
- `~/stable-diffusion/stable-diffusion-webui/` - Auto1111 installation
- `/Library/LaunchDaemons/com.automatic1111.webui.plist` - Auto-start service
- `~/Logs/auto1111.log` - Service logs

### Modified:
- `/Users/markdarby16/16projects/ytv2/backend/.env.nas` - Add Auto1111 config
- `/Users/markdarby16/16projects/ytv2/backend/modules/services/auto1111_service.py` - Create Auto1111 client
- `/Users/markdarby16/16projects/ytv2/backend/modules/services/draw_service.py` - Add fallback logic
- `/Users/markdarby16/16projects/ytv2/backend/modules/services/summary_image_service.py` - Update provider selection

---

## Rollback Plan

If something goes wrong:

1. Stop Automatic1111:
```bash
sudo launchctl unload /Library/LaunchDaemons/com.automatic1111.webui.plist
```

2. Remove from YTV2 config:
```bash
# Edit .env.nas and remove:
# AUTOMATIC1111_BASE_URL
# AUTOMATIC1111_ENABLED
```

3. Restart backend:
```bash
cd /Users/markdarby16/16projects/ytv2/backend
docker-compose restart
```

---

## Next Steps

1. ✅ Create this plan document
2. ⏳ Install Python environment (Miniforge)
3. ⏳ Clone Automatic1111
4. ⏳ Download models
5. ⏳ Create launch script and service
6. ⏳ Configure YTV2 backend integration
7. ⏳ Test image generation
8. ⏳ Monitor queue processing

---

## Notes

- **Intel Mac performance:** Significantly slower than Apple Silicon - this is expected
- **Model choice:** SDXL Lightning recommended for balance of speed/quality
- **Fallback is automatic:** When M4 Mac unavailable, i9 Mac processes queue
- **Queue status:** Can be checked with `ytv2 stats` or `ytv2.` menu
