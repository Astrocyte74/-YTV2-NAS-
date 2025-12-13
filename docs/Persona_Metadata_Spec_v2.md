# Persona Metadata Spec v2

Here’s the Unified Spec v2 you can drop into your repo as
docs/Persona_Metadata_Spec_v2.md.
It merges your original behavioural memo and Codex CLI’s engineering plan so you have one authoritative document.

2025-10-22 update: the Telegram bot has been modularized into service layers (`modules/services/...`).
This revision aligns the persona plan with that structure—persona loading lives in a new service
module that the handlers consume, while the CLI helper remains an optional future add-on.
Implementation status: `persona_service.py` and the CLI helpers are still planned/not yet implemented; treat these references as roadmap placeholders.

⸻

🧭 Persona Metadata Integration & Behavioural Framework (Unified Spec v2)

Author: Mark Darby
Branch: feature/ai2ai-prompt-optimization
Primary Files:
    •    /data/persona_data.json  → canonical registry
    •    /modules/services/persona_service.py  → loader + prompt builder (planned)
    •    /modules/telegram/...    → handlers consume persona_service
    •    /cli/persona_loader.py, /cli/persona_cli.py  → optional CLI layer (future)

⸻

1  Purpose

Unify persona storage, prompt generation, and runtime behaviour for AI↔AI historical dialogues.
This replaces multiple OLLAMA_PERSONA_* env lists with a single, schema-rich JSON registry that powers:
    •    historical and asymmetric awareness behaviour
    •    persona metadata driven tone and curiosity
    •    automatic TTS voice selection
    •    category-based loading and CLI inspection

⸻

2  Architecture Overview

persona_data.json  →  persona_service.py  →  telegram handlers/UI
                             ↑
                       persona_cli.py (optional CLI)

    •    Registry (JSON) holds identity and traits.
    •    Persona service loads, caches, and exposes persona metadata to handlers.
    •    Prompt helpers (within the service) use era, beliefs, temperament to craft system prompts.
    •    CLI tooling can wrap the service for inspection/testing when needed.

⸻

3  JSON Schema Highlights

Field Group    Keys    Use
Identity    name, gender, era, birth_year, death_year, nationality    Asymmetric awareness & display
Semantics    domain[], archetype, notable_traits[], beliefs[], writing_style, temperament, curiosity_level    Prompt tone & behaviour
TTS    voice_hint, (opt) tts_preferred    Voice/accent biasing
Meta    category (top level), meta_index for env fallback    Backward compat

Example excerpt:

{
  "category": "leaders",
  "personas": [
    {
      "name": "Winston Churchill",
      "gender": "M",
      "era": "1874–1965",
      "birth_year": 1874,
      "death_year": 1965,
      "nationality": "British",
      "archetype": "defiant orator",
      "beliefs": [
        "courage is the greatest virtue",
        "words can inspire nations"
      ],
      "writing_style": "grand, persuasive, rhythmic",
      "temperament": "determined",
      "voice_hint": "gravelly British accent"
    }
  ]
}


⸻

4  Prompt Behaviour (From AI↔AI Memo)

Terminology: use counterpart instead of opponent.

Base system prompt:

"You are {persona[name]}, a {persona[archetype]} active during {persona[era]}.
Stay in character, using only the worldview of your lifetime.
You are unaware of events after your death unless described.
Your counterpart is {counterpart[name]}, from a different time and culture."

Asymmetry logic:

if persona["birth_year"] < counterpart["birth_year"]:
    content += " You have never heard of your counterpart before."
else:
    content += f" You may know of {counterpart['name']}'s deeds but treat this as a live conversation."

Intro variant:

“This is your first reply in this exchange. Introduce yourself in character—state who you are, your era, and what principles guide your thinking.
Show curiosity about your counterpart’s origins and invite them to explain themselves.”

User prompts:
    •    A: “Debate topic: {topic}. Present your view from your own time and culture.”
    •    B: “Respond to {personaA_display}’s statement. Engage from your own era; acknowledge unfamiliar ideas briefly before replying.”

⸻

5  Codex CLI Commands

Command    Purpose
codex personas list    Show categories from meta_index.
codex personas show [category]    List personas in that category.
codex personas info [name]    Print full metadata for one persona.
codex personas compare [nameA] [nameB]    Display era asymmetry and trait contrast.
codex personas prompt [nameA] [nameB] [topic]    Generate paired system/user prompts.
codex personas test-debate [nameA] [nameB] [topic]    Run 3-turn simulated debate.


⸻

6  Runtime Parameters & Toggles

Env Key    Purpose
OLLAMA_PERSONA_DATA_PATH    Path to JSON file (default /data/persona_data.json).
PERSONA_META_BELIEFS_MAX    Limit belief sentences in prompt ( default 2 ).
PERSONA_META_STYLE    Include writing_style ( 0/1 ).
PERSONA_META_TEMPERAMENT    Include temperament descriptor ( 0/1 ).
PERSONA_META_CURIOUS    Map curiosity_level to allowed clarifying Qs per turn.


⸻

7  TTS Mapping & Voice Bias
    •    Use gender + voice_hint to filter voice catalogues.
    •    Parse accent keywords ( “French”, “British”, “Japanese” ).
    •    Future support: tts_preferred { engine, voiceId } for exact pins.

⸻

8  Analytics Hooks

During sessions log:

Field    Example
persona_slot    A/B
archetype    “strategic orator”
temperament    “determined”
curiosity_level    5
era_gap_years    553

This allows later tuning of which trait mixes yield best debate quality.

⸻

9  Fallback & Compatibility
    •    Keep meta_index to map categories ↔ legacy OLLAMA_PERSONA_*.
    •    If JSON load fails, fall back to env lists to avoid runtime breaks.

⸻

10  Benefits

✅ One canonical data source for all personas
✅ Authentic era-bounded behaviour with curiosity dial
✅ Dynamic voice and tone control
✅ Backwards compatible with existing env setup
✅ CLI visibility for QA and rapid iteration
✅ Expandable for future AI agents or UI pickers

⸻

11  Next Steps
    1.    Implement `modules/services/persona_service.py` (JSON loader + env fallback).
    2.    Refactor Telegram handlers to consume persona_service for selection, prompts, and defaults.
    3.    (Optional) Add CLI wrappers (`persona_loader.py` / `persona_cli.py`) for inspection.
    4.    Enable logging for analytics fields (archetype, temperament, curiosity).
    5.    Document env toggles in README and Portainer template.


⸻

12 – Codex Implementation Defaults
    •    Era display: first intro + captions only (PERSONA_META_ERA_FIRST_ONLY=1).
    •    Counterpart info: include only on first system prompt; continue per-turn reference in user message.
    •    Curiosity mapping: curiosity ≥ 4 → one short clarifying question per unfamiliar idea (PERSONA_META_CURIOUS=1).
    •    Traits/beliefs: inject up to 2 traits + 1–2 beliefs (PERSONA_META_TRAITS_MAX, PERSONA_META_BELIEFS_MAX).
    •    Accent handling: infer from voice_hint using accent map; allow optional tts_accent_family override.
    •    Aliases: add "aliases" array for flexible lookup.
    •    Schema version: top-level "schema_version": 1; enforce required fields + unique names.
    •    Analytics: log accent_family and tts_path.
    •    Source selection: PERSONA_SOURCE=json|env (default = json).
    •    CLI enhancements: search, lint, and voice-bias subcommands.

⸻

⸻

✅ Summary

This unified spec combines:
    •    Your original behavioural framework → how personas think and interact
    •    Codex’s engineering plan → how data is stored, loaded, and controlled

Together they deliver a fully metadata-driven, CLI-testable persona ecosystem powering authentic historical AI↔AI debates.
