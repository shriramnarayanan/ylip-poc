import base64

from openai import AsyncOpenAI

from adapters.base import VisionAdapter
from config import settings


class LMStudioVisionAdapter(VisionAdapter):
    """
    Vision via LM Studio's chat completions API using the OpenAI image_url format.
    Confirmed working with moondream2 loaded in LM Studio.
    """

    def __init__(self):
        self._client = AsyncOpenAI(
            base_url=settings.lm_studio_base_url,
            api_key="lm-studio",
        )

    async def describe(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        response = await self._client.chat.completions.create(
            model=settings.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
        )
        return response.choices[0].message.content or ""
