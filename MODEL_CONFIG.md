# YTV2 Model Configuration Guide

## Quick Setup

### Option 1: Use the "flash" shortlist (Updated!)

Edit `/Users/markdarby16/16projects/ytv2/backend/.env`:
```bash
LLM_SHORTLIST=flash
```

**Now uses**: `gemini-3.1-flash-lite-preview` (our testing winner!)

### Option 2: Specify exact model

In `.env`:
```bash
LLM_MODEL=google/gemini-3.1-flash-lite-preview
LLM_PROVIDER=openrouter
```

## Available Shortlists

| Shortlist | Primary Model | Best For |
|----------|---------------|----------|
| `flash` | gemini-3.1-flash-lite-preview | Fast, good quality ✅ |
| `research` | gpt-5, claude-4-opus | Best quality |
| `budget` | gpt-5-nano, glm-4.5 | Low cost |
| `fast` | gpt-5-nano, claude-3-haiku | Speed |
| `creative` | gpt-5, claude-4-opus | Writing |
| `coding` | kimi-k2, glm-4.5 | Code |

## Model Selection Priority

1. **Environment variable**: `LLM_MODEL` (overrides everything)
2. **Shortlist primary**: First available model in shortlist
3. **Shortlist fallback**: Backup models if primary unavailable
4. **Default fallback**: Ultimate fallback to any available provider

## Restart Required

After changing `.env`:
```bash
cd /Users/markdarby16/16projects/ytv2/backend
docker-compose restart
```

## Check Current Config

```bash
cd /Users/markdarby16/16projects/ytv2/backend
python3 -c "from llm_config import llm_config; llm_config.print_status()"
```

## Our Testing Recommendation

Based on comprehensive testing of 100 summaries:

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Default for YTV2** | gemini-3.1-flash-lite-preview | Best quality (7.59), good speed |
| **Speed critical** | gemini-2.5-flash | Fastest (647ms) |
| **Structure needed** | gemini-3-flash-preview | Best formatting (8.35) |

The "flash" shortlist is now configured with these models in priority order!
