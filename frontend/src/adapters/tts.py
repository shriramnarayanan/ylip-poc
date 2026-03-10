import io

from openai import AsyncOpenAI

from adapters.base import TTSAdapter
from config import settings


class TTSBackendAdapter(TTSAdapter):
    """
    Calls any OpenAI-compatible /v1/audio/speech endpoint.
    In the prototype this points at the local Kokoro backend server.
    In production this points at the TTS service on the 8850 SoC.
    """

    def __init__(self):
        self._client = AsyncOpenAI(
            base_url=settings.tts_base_url,
            api_key="unused",
        )

    async def synthesize(self, text: str) -> bytes:
        response = await self._client.audio.speech.create(
            model=settings.tts_model,
            voice=settings.tts_voice,
            input=text,
            response_format="wav",
        )
        return response.content
