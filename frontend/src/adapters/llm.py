from typing import AsyncIterator

from openai import AsyncOpenAI

from config import settings
from adapters.base import LLMAdapter, Message


class LMStudioAdapter(LLMAdapter):
    def __init__(self):
        self._client = AsyncOpenAI(
            base_url=settings.lm_studio_base_url,
            api_key="lm-studio",  # LM Studio ignores the key
        )

    def _to_openai(self, messages: list[Message]) -> list[dict]:
        """
        Convert to OpenAI format, folding any leading system message into the
        first user message. Gemma 3 (and some other models) require strictly
        alternating user/assistant roles with no system role.
        """
        result = []
        system_prefix = ""

        for msg in messages:
            if msg.role == "system":
                system_prefix = msg.content + "\n\n"
            elif msg.role == "user" and system_prefix:
                result.append({"role": "user", "content": system_prefix + msg.content})
                system_prefix = ""
            else:
                result.append({"role": msg.role, "content": msg.content})

        return result

    async def generate(self, messages: list[Message]) -> str:
        response = await self._client.chat.completions.create(
            model=settings.lm_studio_model,
            messages=self._to_openai(messages),
            max_tokens=settings.lm_studio_max_tokens,
            temperature=settings.lm_studio_temperature,
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=settings.lm_studio_model,
            messages=self._to_openai(messages),
            max_tokens=settings.lm_studio_max_tokens,
            temperature=settings.lm_studio_temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
