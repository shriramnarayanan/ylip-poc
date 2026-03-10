from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class PipelineContext:
    """Flows through a pipeline, accumulating inputs and outputs."""
    # Inputs
    text: str | None = None
    image_bytes: bytes | None = None
    audio_bytes: bytes | None = None
    history: list[Message] = field(default_factory=list)

    # Outputs (populated as pipeline stages run)
    llm_response: str | None = None
    tts_audio: bytes | None = None
    music_audio: bytes | None = None
    generated_image: bytes | None = None
    vision_description: str | None = None


class LLMAdapter(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message]) -> str: ...

    @abstractmethod
    def stream(self, messages: list[Message]) -> AsyncIterator[str]: ...


class TTSAdapter(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Returns WAV bytes."""
        ...


class STTAdapter(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> str: ...


class ImageGenAdapter(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> bytes:
        """Returns PNG bytes."""
        ...


class VisionAdapter(ABC):
    @abstractmethod
    async def describe(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str: ...
