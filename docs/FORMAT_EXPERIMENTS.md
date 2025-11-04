# LLM Formatting Experiments

Run prompt tests to compare Markdown vs. minimal HTML outputs across models.

## Quick Start

Use a local sample:

```
python3 tools/formatting_experiments.py \
  --text-file tests/samples/summary_sample.txt \
  --formats markdown,html \
  --models "ollama:phi3:latest,ollama:gemma3:12b"
```

Outputs are written to `exports/experiments/formatting/<timestamp>/`.

## Providers

- Local (Ollama): set either `TTSHUB_API_BASE` (hub proxy) or `OLLAMA_URL`/`OLLAMA_HOST`.
- OpenAI: set `OPENAI_API_KEY` and include models like `openai:gpt-4o-mini`.
- OpenRouter: set `OPENROUTER_API_KEY` and include models like `openrouter:google/gemini-2.5-flash-lite`.

Example including cloud:

```
python3 tools/formatting_experiments.py \
  --text-file tests/samples/summary_sample.txt \
  --formats markdown,html \
  --models "ollama:phi3:latest,openrouter:google/gemini-2.5-flash-lite,openai:gpt-4o-mini"
```

## Notes

- The script enforces strict output-only prompts (no preambles) and whitelists tags for HTML.
- If a provider isn’t configured, that run is skipped with an error JSON file for the manifest.
- Use `--dry-run` to print the prompts without calling any model.

