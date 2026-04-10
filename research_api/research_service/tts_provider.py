"""
TTS Provider abstraction for Audio On-Demand.

Currently implements OpenAI TTS (tts-1). To switch providers,
create a new class implementing TTSProvider and update get_tts_provider().
"""
import os
import logging
import tempfile
import subprocess

logger = logging.getLogger(__name__)

TTS_CHUNK_CHARS = 4000  # OpenAI limit is ~4096 chars per request


class TTSProvider:
    """Base class for TTS providers."""

    def generate(self, text: str, voice: str = "fable", output_format: str = "mp3") -> bytes:
        """Generate audio bytes from text. Returns raw audio bytes."""
        raise NotImplementedError

    def name(self) -> str:
        return "unknown"


class OpenAITTS(TTSProvider):
    """OpenAI TTS provider (tts-1)."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def name(self) -> str:
        return "openai_tts"

    def generate(self, text: str, voice: str = "fable", output_format: str = "mp3") -> bytes:
        import requests

        if len(text) <= TTS_CHUNK_CHARS:
            return self._call_api(text, voice, output_format)

        # Chunk long text and combine
        chunks = self._split_chunks(text)
        logger.info("TTS: generating %d chunks for long text (%d chars)", len(chunks), len(text))

        audio_parts = []
        for i, chunk in enumerate(chunks):
            logger.info("TTS: chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            audio_parts.append(self._call_api(chunk, voice, output_format))

        return self._combine_audio(audio_parts)

    def _call_api(self, text: str, voice: str, output_format: str) -> bytes:
        import requests

        resp = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": output_format,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.content

    def _split_chunks(self, text: str) -> list:
        """Split text into chunks at paragraph boundaries."""
        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= TTS_CHUNK_CHARS:
                chunks.append(remaining)
                break
            # Find last paragraph break within limit
            cut = remaining[:TTS_CHUNK_CHARS].rfind("\n\n")
            if cut < 100:
                cut = remaining[:TTS_CHUNK_CHARS].rfind("\n")
            if cut < 100:
                cut = TTS_CHUNK_CHARS
            chunks.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        return chunks

    def _combine_audio(self, parts: list) -> bytes:
        """Combine multiple MP3 files using ffmpeg."""
        if len(parts) == 1:
            return parts[0]

        try:
            tmpdir = tempfile.mkdtemp()
            inputs = []
            for i, part in enumerate(parts):
                path = os.path.join(tmpdir, f"chunk_{i:03d}.mp3")
                with open(path, "wb") as f:
                    f.write(part)
                inputs.append(path)

            output_path = os.path.join(tmpdir, "combined.mp3")
            list_path = os.path.join(tmpdir, "list.txt")
            with open(list_path, "w") as f:
                for p in inputs:
                    f.write(f"file '{p}'\n")

            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                 "-c", "copy", output_path],
                capture_output=True, timeout=30,
            )

            with open(output_path, "rb") as f:
                combined = f.read()

            # Cleanup
            for p in inputs + [output_path, list_path]:
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

            return combined

        except Exception as e:
            logger.error("TTS: ffmpeg combine failed: %s, returning first chunk", e)
            return parts[0]


def get_tts_provider() -> TTSProvider:
    """Factory: return the configured TTS provider."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        return OpenAITTS(api_key)
    # Future: check for CARTESIA_API_KEY, GROQ_API_KEY, etc.
    raise RuntimeError("No TTS provider configured. Set OPENAI_API_KEY.")
