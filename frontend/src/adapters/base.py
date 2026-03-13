from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    tool_calls: list | None = None
    tool_call_id: str | None = None


@dataclass
class PipelineContext:
    """Flows through a pipeline, accumulating inputs and outputs."""
    # Inputs
    text: str | None = None
    image_bytes: bytes | None = None
    audio_bytes: bytes | None = None
    history: list[Message] = field(default_factory=list)
    vision_description: str | None = None
    llm_response: str | None = None
    tts_audio: bytes | None = None
    music_audio: bytes | None = None
    generated_image: bytes | None = None
    pending_image_prompt: str | None = None
    pending_music_prompt: str | None = None
    pending_plot_code: str | None = None
    pending_speak_text: str | None = None


class LLMAdapter(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message], ctx: PipelineContext | None = None) -> str:
        """
        Generate a single completion string given a conversation history.
        The optional ctx allows adapters to deposit metadata (like tool traps).
        """
        pass

    @abstractmethod
    async def stream(self, messages: list[Message], ctx: PipelineContext | None = None) -> AsyncIterator[str]:
        """Stream a completion string token by token."""
        pass


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
