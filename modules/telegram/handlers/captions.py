from typing import Callable


def _strip_provider_prefix(label: str, provider: str) -> str:
    if not label:
        return ""
    prov = (provider or "").strip()
    if prov and label.lower().startswith((prov + ":").lower()):
        return label[len(prov) + 1 :]
    return label


def build_ai2ai_audio_caption(
    a_display: str,
    b_display: str,
    model_a: str,
    model_b: str,
    tts_a_label: str,
    tts_b_label: str,
    provider: str,
    escape_md: Callable[[str], str],
) -> str:
    """Compose the AI↔AI audio recap caption with inline TTS info.

    TTS labels have their provider prefix stripped and are inlined after
    the LLM model on the same line.
    """
    a_tts = _strip_provider_prefix(tts_a_label or "", provider)
    b_tts = _strip_provider_prefix(tts_b_label or "", provider)

    a_line = f"A · {escape_md(a_display)} ({escape_md(model_a)}"
    if a_tts:
        a_line += f" · {escape_md(a_tts)}"
    a_line += ")"

    b_line = f"B · {escape_md(b_display)} ({escape_md(model_b)}"
    if b_tts:
        b_line += f" · {escape_md(b_tts)}"
    b_line += ")"

    return "\n".join([
        "🔊 *AI↔AI Audio Recap*",
        a_line,
        b_line,
    ])

