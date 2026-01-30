# Z-Image Turbo API Documentation

Complete API reference for integrating with Z-Image Turbo, a fast text-to-image generation system running on Mac Silicon.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Core Endpoints](#core-endpoints)
- [AI Enhancement](#ai-enhancement)
- [Recipes System](#recipes-system)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Overview

**Base URL:** `http://localhost:8000`

**Technology:** FastAPI (Python)

**Key Features:**
- Local generation on Mac Silicon (free, no API costs)
- Optional cloud generation via Google Gemini
- Recipe system for reusable prompts
- AI-powered prompt enhancement
- Dual storage modes (normal & encrypted)
- CORS enabled for local web apps

---

## Quick Start

### Minimal Example

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset",
    "generator": "zimage",
    "style_preset": "Cinematic",
    "advanced": {
      "width": 768,
      "height": 768,
      "steps": 7
    }
  }'
```

**Response:**
```json
{
  "id": "202601101234567890",
  "image_url": "/generated/202601101234567890.png",
  "prompt": "A serene mountain landscape at sunset...",
  "created_at": "2026-01-10T12:34:56.789Z",
  "generator": "zimage",
  "width": 768,
  "height": 768,
  "steps": 7,
  "duration_sec": 3.45
}
```

Access the image at: `http://localhost:8000/generated/202601101234567890.png`

---

## Authentication

**No authentication required** for local development. The API is designed for local use.

To enable access from your web app, add your origin to the CORS configuration in `backend/main.py`:

```python
origins = [
    "http://localhost:5173",  # Z-Image frontend
    "http://localhost:5174",  # Slides Prompter
    "http://your-app:PORT",   # Add your app here
]
```

---

## Core Endpoints

### 1. Generate Image

**Endpoint:** `POST /api/generate`

**Description:** Generate an image from a text prompt and optionally save to history.

**Request Body:**

```typescript
interface GenerateRequest {
  // Required
  prompt: string;                    // Text description of desired image

  // Optional - Image content
  negative_prompt?: string;          // Things to avoid in the image
  style_preset?: string;             // "None" | "Cinematic" | "Anime" | "Photographic" | "3D Model" | "Digital Art" | "Fantasy Art"

  // Optional - Generator selection
  generator?: "zimage" | "gemini";   // Default: "zimage" (local) or "gemini" (cloud)
  generator_model?: string;          // For gemini: "nano-banana" | "nano-banana-pro"

  // Optional - Generation settings
  advanced?: {
    width?: number;                  // 256-1536, default: 768
    height?: number;                 // 256-1536, default: 768
    steps?: number;                  // 1-50, default: 7
    cfg_scale?: number;              // 0.0-10.0, default: 0.0
    seed?: number | null;            // Specific seed or null for random
    use_lora?: boolean;              // Enable LoRA model
    lora_id?: string;                // LoRA model ID
    lora_scale?: number;             // 0.0-2.0, default: 1.0
  };

  // Optional - AI enhancement
  auto_enhance?: boolean;            // Auto-enhance prompt with AI
  enhance_provider?: "ollama" | "openrouter";
  enhance_model?: string;
  enhance_mode?: "simple" | "advanced";  // Enhancement style

  // Optional - Storage & metadata
  stealth?: boolean;                 // Use encrypted storage (default: false)
  recipe_id?: string;                // Track which recipe was used
}
```

**Response:**

```typescript
interface GenerationRecord {
  id: string;                        // Unique image ID (timestamp)
  image_url: string;                 // Relative path to image
  prompt: string;                    // The prompt used (possibly enhanced)
  negative_prompt: string;
  style_preset: string;
  created_at: string;                // ISO 8601 timestamp
  model: string;                     // "zimage" or "gemini"
  generator: string;                 // "zimage" or "gemini"
  generator_model?: string;          // For gemini: actual model name
  recipe_id?: string;

  // Generation metadata
  width?: number;
  height?: number;
  steps?: number;
  cfg_scale?: number;
  seed?: number;
  duration_sec?: number;             // Generation time in seconds
  device?: string;                   // Hardware used

  // LoRA metadata
  use_lora?: boolean;
  lora_scale?: number;
  lora_label?: string;               // Display name
  lora_id?: string;

  // AI-generated metadata
  caption?: string;                  // Auto-generated image caption
  tags?: string[];                   // Auto-generated image tags

  stealth: boolean;
}
```

**Example:**

```javascript
const response = await fetch('http://localhost:8000/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
    generator: "zimage",
    style_preset: "Cinematic",
    advanced: {
      width: 1024,
      height: 1024,
      steps: 7,
      seed: 12345  // Reproducible results
    }
  })
});

const result = await response.json();
console.log(`Image generated in ${result.duration_sec}s`);
console.log(`Image URL: http://localhost:8000${result.image_url}`);
```

---

### 2. Generate Ephemeral Image

**Endpoint:** `POST /api/generate_ephemeral`

**Description:** Generate an image and return PNG bytes directly, without saving to disk or history. Ideal for bots and serverless integrations.

**Request Body:** Same as `/api/generate`

**Response:** Binary PNG data with headers:
- `Content-Type: image/png`
- `Content-Disposition: inline; filename="{id}.png"`
- `X-Zimage-Seed`: The seed used
- `X-Zimage-Duration-Sec`: Generation time
- `X-Zimage-Image-Id`: Image ID

**Example:**

```bash
curl -X POST http://localhost:8000/api/generate_ephemeral \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A red rose", "advanced": {"width": 512, "height": 512}}' \
  --output rose.png
```

---

### 3. Get Generation History

**Endpoint:** `GET /api/history?mode={mode}`

**Description:** Retrieve previous generations, most recent first.

**Query Parameters:**
- `mode`:
  - `"default"` - Normal history (default)
  - `"stealth"` - Encrypted/auxiliary history
  - `"all"` - Combined history

**Response:** `GenerationRecord[]` (most recent first)

**Example:**

```javascript
const response = await fetch('http://localhost:8000/api/history?mode=default');
const history = await response.json();

history.forEach(record => {
  console.log(`${record.id}: ${record.prompt}`);
  console.log(`  → http://localhost:8000${record.image_url}`);
});
```

---

### 4. Health Check

**Endpoint:** `GET /health`

**Description:** Lightweight health check for uptime monitoring.

**Response:**

```typescript
interface HealthResponse {
  status: string;           // "ok"
  timestamp: string;        // ISO 8601 timestamp
  uptime_seconds: number;  // Server uptime
}
```

**Example:**

```javascript
const health = await fetch('http://localhost:8000/health')
  .then(r => r.json());

console.log(`Server uptime: ${health.uptime_seconds}s`);
```

---

### 5. Delete History Entry

**Endpoint:** `DELETE /api/history/{image_id}`

**Description:** Delete a history entry and its associated image file.

**Response:**

```json
{ "ok": true }
```

---

## AI Enhancement

### Enhance Prompt

**Endpoint:** `POST /api/enhance_prompt`

**Description:** Enhance a rough prompt using AI (local Ollama or OpenRouter).

**Request Body:**

```typescript
interface PromptEnhanceRequest {
  prompt: string;
  negative_prompt?: string;
  style_preset?: string;
  lora_id?: string;
  provider?: "ollama" | "openrouter";
  model?: string;
  enhancement_mode?: "simple" | "advanced";
}
```

**Response:**

```typescript
interface PromptEnhanceResponse {
  prompt: string;  // Enhanced prompt
  model: string;   // Model used
}
```

**Example:**

```javascript
const response = await fetch('http://localhost:8000/api/enhance_prompt', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "a cat sitting on a fence",
    enhancement_mode: "simple"
  })
});

const { prompt } = await response.json();
// Returns: "A fluffy tabby cat sits gracefully atop a weathered wooden fence, golden hour sunlight casting long shadows, cinematic depth of field, warm amber and orange tones, photorealistic fur texture, peaceful suburban backyard setting"
```

---

## Recipes System

Recipes are reusable prompt templates with pre-configured settings.

### Get Recipes

**Endpoint:** `GET /api/recipes?group={group}&q={search}&mode={mode}`

**Query Parameters:**
- `group`: Filter by recipe group ID (optional)
- `q`: Substring search across title, subtitle, prompt (optional)
- `mode`: `"default"` or `"stealth"` (default: `"default"`)

**Response:**

```typescript
interface RecipesDocument {
  version: number;
  groups: RecipeGroup[];
  recipes: Recipe[];
}

interface RecipeGroup {
  id: string;
  label: string;
  order: number;
}

interface Recipe {
  id: string;
  group: string;
  title: string;
  subtitle?: string;
  prompt: string;
  negative_prompt?: string;
  focus_token?: string;
  visibility: "all" | "secure";  // "secure" recipes only shown in stealth mode
  lora?: {
    id: string;
    scale: number;
  };
}
```

**Example:**

```javascript
// Get all portrait recipes
const response = await fetch('http://localhost:8000/api/recipes?group=characters');
const { recipes } = await response.json();

recipes.forEach(recipe => {
  console.log(`${recipe.title}: ${recipe.subtitle}`);
  console.log(`  Prompt: ${recipe.prompt}`);
  console.log(`  Focus token: ${recipe.focus_token}`);
});
```

---

### Using Recipes in Generation

When you have a recipe, use its `prompt` as a template and replace `focus_token`:

```javascript
// 1. Get recipe
const recipesRes = await fetch('http://localhost:8000/api/recipes');
const { recipes } = await recipesRes.json();
const recipe = recipes.find(r => r.id === 'cinematic-hugo-style-portrait');

// 2. Replace focus token
const prompt = recipe.prompt.replace(/CHARACTER/gi, 'Astronaut');
const negativePrompt = recipe.negative_prompt || '';

// 3. Generate
const genRes = await fetch('http://localhost:8000/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt,
    negative_prompt: negativePrompt,
    style_preset: recipe.style_preset || "None",
    recipe_id: recipe.id,
    advanced: {
      width: 768,
      height: 768,
      steps: 7
    }
  })
});
```

---

## Style Presets

Available style presets:

| Preset | Description |
|--------|-------------|
| `None` | No style modification |
| `Cinematic` | Movie-like lighting and composition |
| `Anime` | Anime/ manga style |
| `Photographic` | Realistic photography |
| `3D Model` | 3D render style |
| `Digital Art` | Digital painting |
| `Fantasy Art` | Fantasy illustration |

---

## Generator Comparison

### Z-Image Turbo (Local)

```json
{
  "generator": "zimage"
}
```

**Pros:**
- ✅ Free (no API costs)
- ✅ Runs on Mac Silicon GPU
- ✅ Supports LoRAs
- ✅ Up to 1536x1536 resolution

**Cons:**
- ⏱️ Slower (~3-7 seconds per image)
- 💻 Requires local hardware

---

### Gemini Nano Banana (Cloud)

```json
{
  "generator": "gemini",
  "generator_model": "nano-banana"  // or "nano-banana-pro"
}
```

**Pros:**
- ⚡ Faster (~1-2 seconds per image)
- 🌐 Cloud-based
- 🎨 High quality

**Cons:**
- 💰 Requires `GEMINI_API_KEY` (API costs)
- 📦 No LoRA support

**Setup:**
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Error Handling

All errors return HTTP status codes with JSON responses:

```typescript
interface ErrorResponse {
  detail: string;  // Human-readable error message
}
```

**Common Status Codes:**

| Status | Description |
|--------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 404 | Not found (image ID doesn't exist) |
| 422 | Unprocessable entity (validation error) |
| 502 | Bad gateway (upstream service failure) |

**Example:**

```javascript
try {
  const response = await fetch('http://localhost:8000/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: '' })  // Invalid: empty prompt
  });

  if (!response.ok) {
    const error = await response.json();
    console.error(`Error ${response.status}: ${error.detail}`);
  }
} catch (err) {
  console.error('Network error:', err);
}
```

---

## Environment Variables

```bash
# Gemini API (for cloud generation)
GEMINI_API_KEY=your_key_here

# OpenRouter (for prompt enhancement)
OPENROUTER_API_KEY=your_key_here
OR_MODELS=google/gemini-2.5-flash

# Ollama (local LLM for prompt enhancement)
OLLAMA_MODELS=gemma3:12b

# Optional: Encryption key for stealth mode
APP_DATA_KEY=hex_encoded_32_byte_key
```

---

## Image Storage

### Normal Storage
- **Path:** `/generated/{timestamp}.png`
- **URL:** `http://localhost:8000/generated/{timestamp}.png`
- **File location:** `backend/generated/{timestamp}.png`

### Encrypted Storage (Stealth Mode)
- **Path:** `/asset/{id}`
- **URL:** `http://localhost:8000/asset/{id}`
- **Decrypted on-the-fly when accessed**
- **File location:** `backend/generated/.sysdata/{id}.dat`

---

## Rate Limiting

**No rate limiting** for local development.

For production deployments, implement rate limiting at the reverse proxy level (nginx, etc.).

---

## Best Practices

### 1. Use Recipe System

Leverage recipes for consistent, high-quality prompts:

```javascript
// Get a recipe, customize it, generate
const recipe = await getRecipe('cinematic-hugo-style-portrait');
const customPrompt = recipe.prompt.replace(/CHARACTER/g, 'Your Subject');
const result = await generateImage(customPrompt);
```

### 2. Handle Async Operations

Image generation takes 3-7 seconds. Show loading states:

```javascript
async function generateWithLoading(prompt) {
  showLoadingSpinner();

  try {
    const result = await fetch('http://localhost:8000/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    }).then(r => r.json());

    hideLoadingSpinner();
    displayImage(result.image_url);
  } catch (err) {
    hideLoadingSpinner();
    showError(err.message);
  }
}
```

### 3. Use Ephemeral Generation for Bots

For Telegram bots, Discord bots, etc., use `/api/generate_ephemeral` to avoid disk I/O:

```javascript
const response = await fetch('http://localhost:8000/api/generate_ephemeral', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: "A beautiful landscape" })
});

const imageBlob = await response.blob();
// Send imageBlob directly to chat service
```

### 4. Cache Recipe Data

Recipes don't change often. Cache them:

```javascript
let recipesCache = null;

async function getRecipes() {
  if (recipesCache) return recipesCache;

  const response = await fetch('http://localhost:8000/api/recipes');
  recipesCache = await response.json();
  return recipesCache;
}
```

---

## Troubleshooting

### "CORS error"

Add your origin to `backend/main.py`:

```python
origins = [
    "http://localhost:5173",
    "https://your-app.com",  # Add this
]
```

### "GEMINI_API_KEY is not set"

Set the environment variable:

```bash
export GEMINI_API_KEY="your_key_here"
# Restart the backend
```

### "Could not reach Ollama"

Ensure Ollama is running:

```bash
ollama serve
```

### Image appears broken

Check the full URL:

```javascript
// Image URLs are relative, prepend base URL
const fullUrl = `http://localhost:8000${result.image_url}`;
```

---

## Support

For issues or questions:
1. Check the browser console for error messages
2. Review backend logs: `tail -f backend/zimage.log`
3. Verify the backend is running: `curl http://localhost:8000/health`

---

## Changelog

### v1.0 (Current)
- Local Z-Image Turbo generation
- Gemini Nano Banana integration
- Recipe system
- AI prompt enhancement
- Stealth/encrypted storage
- LoRA support
- Vision model integration
