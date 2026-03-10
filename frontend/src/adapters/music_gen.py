import httpx

from config import settings


class MusicGenAdapter:
    """Sends a text prompt to the music generation backend and returns WAV bytes."""

    async def generate(self, prompt: str, max_new_tokens: int = 512) -> bytes:
        async with httpx.AsyncClient(base_url=settings.music_gen_base_url, timeout=120) as client:
            resp = await client.post(
                "/audio/generate",
                json={"prompt": prompt, "max_new_tokens": max_new_tokens},
            )
            if resp.status_code == 400:
                raise RuntimeError(f"music_gen error: {resp.json()}")
            resp.raise_for_status()
            return resp.content
