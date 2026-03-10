import base64

import httpx

from adapters.base import ImageGenAdapter
from config import settings


class ImageGenBackendAdapter(ImageGenAdapter):
    """
    Image generation via the backend /v1/images/generations endpoint (OpenAI-compatible).
    In the prototype this points at the local SDXL-Turbo backend server.
    In production this points at the image gen service on the 8850 SoC.
    """

    async def generate(self, prompt: str) -> bytes:
        async with httpx.AsyncClient(base_url=settings.image_base_url, timeout=120) as client:
            resp = await client.post(
                "/images/generations",
                json={
                    "model": settings.image_model,
                    "prompt": prompt,
                    "response_format": "b64_json",
                },
            )
            resp.raise_for_status()
            b64 = resp.json()["data"][0]["b64_json"]
            return base64.b64decode(b64)
