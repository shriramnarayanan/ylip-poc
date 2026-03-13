"""
Orchestrator: builds pipelines based on available adapters and input type,
then either runs them to completion or streams the LLM stage for UI responsiveness.
"""

import asyncio
import io
import re
import wave
from typing import AsyncIterator

from config import settings, SYSTEM_PROMPT, SYSTEM_PROMPT_CONVERSATION
from adapters.base import PipelineContext, Message
from adapters.llm import LMStudioAdapter
from core.pipeline import Step, ParallelStep, run_pipeline
from core.session import Session

# Sentence boundary: punctuation followed by whitespace
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

def _concat_wav(a: bytes, b: bytes) -> bytes:
    """Concatenate two WAV byte-strings (must share sample rate and format)."""
    buf = io.BytesIO()
    with wave.open(io.BytesIO(a)) as wa:
        params = wa.getparams()
        frames_a = wa.readframes(wa.getnframes())
    with wave.open(io.BytesIO(b)) as wb:
        frames_b = wb.readframes(wb.getnframes())
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(frames_a)
        out.writeframes(frames_b)
    return buf.getvalue()


class Orchestrator:
    def __init__(self):
        self.llm = LMStudioAdapter()
        self._tts = None
        self._stt = None
        self._image_gen = None
        self._vision = None
        self._code_exec = None
        self._music_gen = None

    # ------------------------------------------------------------------ #
    # Lazy adapter init                                                    #
    # ------------------------------------------------------------------ #

    @property
    def tts(self):
        if self._tts is None and settings.tts_enabled:
            from adapters.tts import TTSBackendAdapter
            self._tts = TTSBackendAdapter()
        return self._tts

    @property
    def stt(self):
        if self._stt is None and settings.stt_enabled:
            from adapters.stt import STTBackendAdapter
            self._stt = STTBackendAdapter()
        return self._stt

    @property
    def image_gen(self):
        if self._image_gen is None and settings.image_enabled:
            from adapters.image_gen import ImageGenBackendAdapter
            self._image_gen = ImageGenBackendAdapter()
        return self._image_gen

    @property
    def vision(self):
        if self._vision is None and settings.vision_enabled:
            from adapters.vision import LMStudioVisionAdapter
            self._vision = LMStudioVisionAdapter()
        return self._vision

    @property
    def code_exec(self):
        if self._code_exec is None and settings.code_exec_enabled:
            from adapters.code_exec import CodeExecAdapter
            self._code_exec = CodeExecAdapter()
        return self._code_exec

    @property
    def music_gen(self):
        if self._music_gen is None and settings.music_gen_enabled:
            from adapters.music_gen import MusicGenAdapter
            self._music_gen = MusicGenAdapter()
        return self._music_gen

    # ------------------------------------------------------------------ #
    # Pipeline step functions                                              #
    # ------------------------------------------------------------------ #

    async def _vision_step(self, ctx: PipelineContext) -> PipelineContext:
        if self.vision and ctx.image_bytes:
            desc = await self.vision.describe(ctx.image_bytes, ctx.text or "Describe this image.")
            ctx.vision_description = desc
            ctx.text = f"[Image description: {desc}]\n\n{ctx.text or ''}"
        return ctx

    async def _llm_step(self, ctx: PipelineContext) -> PipelineContext:
        messages = ctx.history + [Message(role="user", content=ctx.text or "")]
        ctx.llm_response = await self.llm.generate(messages, ctx=ctx)
        return ctx

    async def _tts_step(self, ctx: PipelineContext) -> PipelineContext:
        if self.tts and ctx.llm_response:
            if self.music_gen and ctx.pending_music_prompt:
                return ctx  # music gen handles audio output
            clean = ctx.llm_response.strip()
            if clean:
                ctx.tts_audio = await self.tts.synthesize(clean)
        return ctx

    async def _music_gen_step(self, ctx: PipelineContext) -> PipelineContext:
        if self.music_gen and ctx.pending_music_prompt:
            ctx.music_audio = await self.music_gen.generate(ctx.pending_music_prompt)
        return ctx

    async def _image_gen_step(self, ctx: PipelineContext) -> PipelineContext:
        if self.image_gen and ctx.pending_image_prompt:
            ctx.generated_image = await self.image_gen.generate(ctx.pending_image_prompt)
        return ctx

    async def _code_exec_step(self, ctx: PipelineContext) -> PipelineContext:
        if self.code_exec and ctx.pending_plot_code:
            ctx.generated_image = await self.code_exec.execute(ctx.pending_plot_code)
        return ctx

    # ------------------------------------------------------------------ #
    # Pipeline builders                                                    #
    # ------------------------------------------------------------------ #

    def _build_pipeline(self, has_image: bool) -> list:
        steps = []

        if has_image and self.vision:
            steps.append(Step("vision", self._vision_step))

        steps.append(Step("llm", self._llm_step))

        post = []
        if settings.tts_enabled:
            post.append(Step("tts", self._tts_step))
        if settings.music_gen_enabled:
            post.append(Step("music_gen", self._music_gen_step))
        if settings.image_enabled:
            post.append(Step("image_gen", self._image_gen_step))
        if settings.code_exec_enabled:
            post.append(Step("code_exec", self._code_exec_step))

        if len(post) == 1:
            steps.extend(post)
        elif len(post) > 1:
            steps.append(ParallelStep(post))

        return steps

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        session: Session,
        text: str,
        image_bytes: bytes | None = None,
        audio_bytes: bytes | None = None,
    ) -> PipelineContext:
        if audio_bytes and self.stt:
            text = await self.stt.transcribe(audio_bytes)

        ctx = PipelineContext(
            text=text,
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            history=session.to_messages(SYSTEM_PROMPT),
        )
        return await run_pipeline(self._build_pipeline(has_image=image_bytes is not None), ctx)

    async def stream_run(
        self,
        session: Session,
        text: str,
        image_bytes: bytes | None = None,
        audio_bytes: bytes | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> AsyncIterator[PipelineContext]:
        if audio_bytes and self.stt:
            text = await self.stt.transcribe(audio_bytes)

        if image_bytes and self.vision:
            desc = await self.vision.describe(image_bytes, text or "Describe this image.")
            text = f"[Image description: {desc}]\n\n{text or ''}"

        messages = session.to_messages(system_prompt) + [Message(role="user", content=text)]
        ctx = PipelineContext(text=text, image_bytes=image_bytes, history=messages, llm_response="")

        # Start TTS tasks sentence-by-sentence while the LLM streams.
        # Each time the earliest pending task is done we set ctx.tts_audio and
        # yield immediately so the caller can stream the audio chunk without
        # waiting for the rest of the response.
        sentence_buf = ""
        tts_tasks: list[asyncio.Task] = []
        do_tts = self.tts is not None

        async for token in self.llm.stream(messages, ctx=ctx):
            ctx.llm_response += token
            if do_tts:
                sentence_buf += token
                m = _SENTENCE_RE.search(sentence_buf)
                if m:
                    complete = sentence_buf[: m.start() + 1]
                    sentence_buf = sentence_buf[m.end():]
                    clean = complete.strip()
                    if clean:
                        tts_tasks.append(asyncio.create_task(self.tts.synthesize(clean)))
            # Ship the earliest completed sentence immediately (maintains order).
            ctx.tts_audio = None
            if do_tts and tts_tasks and tts_tasks[0].done():
                try:
                    ctx.tts_audio = tts_tasks.pop(0).result()
                except Exception:
                    tts_tasks.pop(0)
            yield ctx

        ctx.tts_audio = None  # clear before post-LLM phase

        # Flush any remaining text
        if do_tts and sentence_buf.strip():
            clean = sentence_buf.strip()
            if clean:
                tts_tasks.append(asyncio.create_task(self.tts.synthesize(clean)))

        # Cancel TTS if music is present (music gen takes over audio output)
        if tts_tasks and self.music_gen and ctx.pending_music_prompt:
            for t in tts_tasks:
                t.cancel()
            tts_tasks.clear()

        # Yield each remaining sentence in order as it completes
        for task in tts_tasks:
            try:
                ctx.tts_audio = await task
                yield ctx
                ctx.tts_audio = None
            except Exception:
                pass

        # Post-LLM: music gen, image gen, code exec — TTS handled above
        post = []
        if settings.music_gen_enabled:
            post.append(Step("music_gen", self._music_gen_step))
        if settings.image_enabled:
            post.append(Step("image_gen", self._image_gen_step))
        if settings.code_exec_enabled:
            post.append(Step("code_exec", self._code_exec_step))

        if len(post) == 1:
            ctx = await post[0].fn(ctx)
        elif len(post) > 1:
            results = await asyncio.gather(*(s.fn(ctx) for s in post))
            for r in results:
                if r.music_audio:
                    ctx.music_audio = r.music_audio
                if r.generated_image:
                    ctx.generated_image = r.generated_image

        yield ctx
