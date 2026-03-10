import httpx

from adapters.base import STTAdapter
from config import settings


class STTBackendAdapter(STTAdapter):
    """
    STT via the backend /v1/audio/transcriptions endpoint (OpenAI-compatible).
    In the prototype this points at the local faster-whisper backend server.
    In production this points at the STT service on the 8850 SoC.
    """

    async def transcribe(self, audio_bytes: bytes) -> str:
        async with httpx.AsyncClient(base_url=settings.stt_base_url, timeout=60) as client:
            resp = await client.post(
                "/audio/transcriptions",
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"model": settings.stt_model},
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
