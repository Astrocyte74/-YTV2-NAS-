from typing import List


def split_text_into_chunks(text: str, max_chunk_size: int) -> List[str]:
    """Split text into chunks that fit within Telegram message limits.

    Prefers breaking on paragraph and sentence boundaries when possible.
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks: List[str] = []
    current_pos = 0

    while current_pos < len(text):
        end_pos = current_pos + max_chunk_size
        if end_pos >= len(text):
            chunks.append(text[current_pos:].strip())
            break

        break_point = end_pos
        # Prefer paragraph break (double newline) within last 200 chars
        paragraph_break = text.rfind("\n\n", current_pos, max(current_pos, end_pos - 200))
        if paragraph_break > current_pos:
            break_point = paragraph_break
        else:
            # Prefer sentence break within last 100 chars
            sentence_break = text.rfind(". ", current_pos, max(current_pos, end_pos - 100))
            if sentence_break > current_pos:
                break_point = sentence_break + 1

        # Avoid ending on trailing backslash (would escape next chunk's first char)
        while break_point > current_pos and text[break_point - 1] == "\\":
            break_point -= 1

        chunk = text[current_pos:break_point].strip()
        if chunk:
            chunks.append(chunk)
        current_pos = break_point
        while current_pos < len(text) and text[current_pos].isspace():
            current_pos += 1

    return chunks


def escape_markdown(text: str) -> str:
    """Escape a minimal set of characters for Telegram Markdown."""
    if not text:
        return ""
    escape_chars = ["_", "*", "[", "]", "`"]
    escaped = text
    for ch in escape_chars:
        escaped = escaped.replace(ch, f"\\{ch}")
    return escaped

