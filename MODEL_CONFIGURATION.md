# YTV2 Model Configuration Guide

Complete reference for configuring LLM models in YTV2.

*Last updated: 2026-03-14*

---

## Overview

YTV2 uses a **cascading priority system** for model selection:

```
1. Explicit LLM_MODEL + LLM_PROVIDER     → Direct use
2. LLM_MODEL only                      → Auto-detect provider
3. LLM_SHORTLIST                      → Use shortlist priority
4. Ultimate fallback                    → Any available provider
```

---

## Configuration Files

### Primary Configuration
**Location**: `/Users/markdarby16/16projects/ytv2/backend/.env.nas`

This is the **main configuration file** that Docker reads.

### Model Definition
**Location**: `/Users/markdarby16/16projects/ytv2/backend/llm_config.py`

Contains:
- Model shortlists (research, budget, fast, flash, creative, etc.)
- Default models per provider
- Provider detection logic

---

## How Model Selection Works

### Priority 1: Explicit Provider + Model

Set both in `.env.nas`:
```bash
LLM_PROVIDER=openrouter
LLM_MODEL=google/gemini-3.1-flash-lite-preview
```

**Result**: Uses that exact model directly. **Skips all shortlists.**

### Priority 2: Model Only (Auto-detect)

```bash
LLM_MODEL=google/gemini-3.1-flash-lite-preview
```

**Result**: Detects provider from model name (google → openrouter, claude → anthropic, etc.)

### Priority 3: Shortlist System

```bash
LLM_SHORTLIST=flash
```

**Result**: Tries primary models in order, falls back to fallback list.

### Priority 4: Ultimate Fallback

If nothing else works, tries any available provider in order:
1. openai
2. anthropic
3. openrouter
4. inception

---

## Current Configuration (2026-03-14)

Based on comprehensive testing of 100 summaries across 5 models.

### Active Settings

```bash
LLM_SHORTLIST=flash
LLM_MODEL=google/gemini-3.1-flash-lite-preview
LLM_PROVIDER=openrouter
```

### Flash Shortlist (Updated)

```python
"flash": {
    "primary": [
        ("openrouter", "google/gemini-3.1-flash-lite-preview")  # Primary: Best quality (7.59)
    ],
    "fallback": [
        ("inception", "mercury-2"),                           # Fallback 1: Most accurate (8.25)
        ("openrouter", "google/gemini-2.5-flash"),           # Fallback 2: Fastest (647ms)
        ("openrouter", "google/gemini-3-flash-preview"),      # Fallback 3: Best structure (8.35)
        ("openai", "gpt-5-nano")                              # Fallback 4: Budget option
    ]
}
```

### Rationale

| Position | Model | Why This Order? |
|----------|-------|----------------|
| **Primary** | Gemini 3.1 Flash Lite | Best overall quality (7.59/10), good speed (980ms) |
| **Fallback 1** | Mercury-2 Standard | Highest accuracy (8.25/10) when quality matters most |
| **Fallback 2** | Gemini 2.5 Flash | Fastest (647ms) when speed is critical |
| **Fallback 3** | Gemini 3 Flash Preview | Best structure (8.35/10) for formatting |
| **Fallback 4** | GPT-5 Nano | Budget option, still decent quality |

---

## How to Change Models

### Option 1: Change Primary Model

Edit `.env.nas`:
```bash
# Use a different primary model
LLM_MODEL=google/gemini-3-flash-preview
LLM_PROVIDER=openrouter
```

### Option 2: Switch Shortlists

Edit `.env.nas`:
```bash
# Available shortlists: research, budget, fast, flash, creative, coding, local
LLM_SHORTLIST=research  # For highest quality
LLM_SHORTLIST=budget    # For lowest cost
LLM_SHORTLIST=fast      # For speed
```

### Option 3: Update Shortlist Definitions

Edit `llm_config.py` SHORTLISTS dictionary:

```python
"flash": {
    "primary": [
        ("openrouter", "your-new-model-here"),
    ],
    "fallback": [
        ("provider", "model-1"),
        ("provider", "model-2"),
    ]
}
```

Then rebuild:
```bash
docker-compose down && docker-compose up -d
```

---

## Adding a New Model

### Step 1: Get API Key

Add to `.env.nas`:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
INCEPTION_API_KEY=sk_...
```

### Step 2: Add to Shortlist

Edit `llm_config.py`:
```python
"your_shortlist_name": {
    "primary": [
        ("provider", "model-name"),
    ],
    "fallback": []
}
```

### Step 3: Rebuild Container

```bash
cd /Users/markdarby16/16projects/ytv2/backend
docker-compose down && docker-compose up -d
```

---

## Available Shortlists

| Shortlist | Primary Models | Use Case |
|----------|---------------|----------|
| `research` | gpt-5, claude-4-opus, qwen-2.5-72b | Highest quality analysis |
| `budget` | gpt-5-nano, glm-4.5, phi3 | Lowest cost |
| `fast` | gpt-5-nano, claude-3-haiku, phi3 | Speed |
| **`flash`** | **gemini-3.1-flash-lite, mercury-2, gemini-2.5** | **Balanced speed/quality (current)** |
| `creative` | gpt-5, claude-4-opus, gpt-5-mini | Writing tasks |
| `coding` | kimi-k2, glm-4.5, qwen3-coder | Code generation |
| `local` | ollama models (on-premise) | Privacy/offline |

---

## Telegram Bot Model Selection

The Telegram bot uses the same configuration via:
- **Model picker menu**: Uses `LLM_SHORTLIST` to populate options
- **Auto-mode**: Uses `AUTO_MODE_MODELS` for automatic processing
- **Quick cloud model**: Uses `QUICK_CLOUD_MODEL` for quick switching

### Telegram-Specific Settings

```bash
# Auto-mode (skip prompts, use default)
TELEGRAM_AUTO_MODE=true
AUTO_MODE_MODELS=google/gemini-3.1-flash-lite-preview,google/gemini-2.5-flash
AUTO_MODE_VARIANT=key-insights

# Quick cloud model for bot commands
QUICK_CLOUD_MODEL=google/gemini-3.1-flash-lite-preview,google/gemini-2.5-flash,google/gemini-3-flash-preview
```

---

## Testing Framework Models

Separate configuration for testing models: `promptfoo-testing/models.json`

See: `/Users/markdarby16/16projects/ytv2/promptfoo-testing/MODELS.md`

---

## Troubleshooting

### Check Current Configuration

```bash
cd /Users/markdarby16/16projects/ytv2/backend
docker exec youtube-summarizer-bot python3 -c "
from llm_config import llm_config
llm_config.print_status()
"
```

### View Available Providers

```bash
docker exec youtube-summarizer-bot python3 -c "
from llm_config import llm_config
import json
print(json.dumps(llm_config.get_available_providers(), indent=2))
"
```

### Test a Specific Model

```bash
docker exec youtube-summarizer-bot python3 -c "
from llm_config import llm_config
provider, model, key = llm_config.get_model_config('openrouter', 'google/gemini-3.1-flash-lite-preview')
print(f'Using: {provider}/{model}')
print(f'Has key: {bool(key)}')
"
```

---

## Model Selection Flow Diagram

```
User requests summary
       ↓
┌─────────────────────────────────────┐
│ Is LLM_MODEL + LLM_PROVIDER set?    │
└─────────────────────────────────────┘
       │ YES            NO
       ↓                ↓
┌──────────────┐   ┌──────────────────┐
│ Use directly  │   │ Detect provider │
│              │   │ from model name  │
└──────────────┘   └──────────────────┘
                          ↓
              ┌─────────────────────┐
              │ Is LLM_SHORTLIST set? │
              └─────────────────────┘
                     │ YES    NO
                     ↓       ↓
              ┌────────┐  ┌──────────────┐
              │Use     │  │Fallback to   │
              │shortlist│  │any available│
              │primary │  │provider      │
              │model   │  │              │
              └────────┘  └──────────────┘
                     ↓
              ┌─────────────────────┐
              │ If provider has key  │
              │ → use that model    │
              │ Else try next in    │
              │    fallback list     │
              └─────────────────────┘
```

---

## Quick Reference

### Change primary model
```bash
LLM_MODEL=google/gemini-3.1-flash-lite-preview
LLM_PROVIDER=openrouter
```

### Use a different shortlist
```bash
LLM_SHORTLIST=research  # Best quality
LLM_SHORTLIST=budget   # Lowest cost
LLM_SHORTLIST=fast     # Speed
```

### Add fallback model
Edit `llm_config.py` fallback array, then rebuild.

### View current config
```bash
docker exec youtube-summarizer-bot python3 -c "from llm_config import llm_config; llm_config.print_status()"
```

### Apply changes
```bash
cd /Users/markdarby16/16projects/ytv2/backend
docker-compose down && docker-compose up -d
```

---

## Model Comparison Summary

From comprehensive testing (100 summaries, 5 models, 5 transcripts):

| Model | Quality | Speed | Structure | Best For |
|-------|--------|-------|-----------|----------|
| **Gemini 3.1 Flash Lite** | 7.59 | 980ms | 8.35 | **Overall best** ✅ |
| Gemini 3 Flash Preview | 7.47 | 1505ms | 8.35 | Structure critical |
| Gemini 2.5 Flash | 7.40 | 647ms | 7.95 | Speed critical |
| Mercury-2 Standard | 7.37 | 1707ms | 8.85 | Accuracy critical |
| Mercury-2 Instant | 7.40 | 1418ms | 9.05 | - |

*Quality scores are 1-10 scale across 5 dimensions (Accuracy, Completeness, Clarity, Structure, Conciseness)*

---

## File Locations Summary

| File | Purpose |
|------|---------|
| `.env.nas` | **Main configuration** - edit this to change models |
| `llm_config.py` | Shortlist definitions and model selection logic |
| `llm_config.py` | Provider detection and fallback logic |
| `telegram_bot.py` | Telegram bot model picker integration |
| `modules/telegram/handlers/ai2ai.py` | AI-to-AI feature model selection |

---

*See also: `promptfoo-testing/RESULTS-COMPREHENSIVE-ANALYSIS-GLM-4.7.md` for full testing methodology*
