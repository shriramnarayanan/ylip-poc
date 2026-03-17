"""
Orchestrator: builds pipelines based on available adapters and input type,
then either runs them to completion or streams the LLM stage for UI responsiveness.
"""

import asyncio
import io
import re
import threading
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
        self._student_state = None

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

    @property
    def student_state(self):
        if self._student_state is None and settings.student_state_enabled:
            from adapters.student_state import StudentStateAdapter
            self._student_state = StudentStateAdapter()
        return self._student_state

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
        streaming_tts: bool = False,
        tts_stop_event: threading.Event | None = None,
    ) -> AsyncIterator[PipelineContext]:
        if audio_bytes and self.stt:
            text = await self.stt.transcribe(audio_bytes)

        if image_bytes and self.vision:
            desc = await self.vision.describe(image_bytes, text or "Describe this image.")
            text = f"[Image description: {desc}]\n\n{text or ''}"

        if self.student_state:
            student_ctx = await self.student_state.get_context()
            if student_ctx:
                system_prompt = f"{system_prompt}\n\n{student_ctx}"

        messages = session.to_messages(system_prompt) + [Message(role="user", content=text)]
        ctx = PipelineContext(text=text, image_bytes=image_bytes, history=messages, llm_response="")

        sentence_buf = ""
        tts_tasks: list[asyncio.Task] = []
        image_task: asyncio.Task | None = None
        music_task: asyncio.Task | None = None
        # Streaming TTS (sentence-by-sentence) only in Conversation mode.
        # Text mode uses speak() tool for explicit word pronunciation only.
        do_tts = self.tts is not None and streaming_tts

        def _tts_stopped() -> bool:
            return tts_stop_event is not None and tts_stop_event.is_set()

        try:
            async for token in self.llm.stream(messages, ctx=ctx):
                ctx.llm_response += token
                if do_tts and not _tts_stopped():
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

            ctx.tts_audio = None

            # Flush any remaining sentence buffer (Conversation mode)
            if do_tts and not _tts_stopped() and sentence_buf.strip():
                clean = sentence_buf.strip()
                if clean:
                    tts_tasks.append(asyncio.create_task(self.tts.synthesize(clean)))

            # Cancel TTS if music takes over audio output
            if tts_tasks and self.music_gen and ctx.pending_music_prompt:
                for t in tts_tasks:
                    t.cancel()
                tts_tasks.clear()

            # Text mode: synthesize only the explicitly requested word/phrase.
            # This runs regardless of streaming_tts but respects the stop flag.
            if self.tts and ctx.pending_speak_text and not _tts_stopped():
                try:
                    ctx.tts_audio = await self.tts.synthesize(ctx.pending_speak_text)
                except Exception:
                    ctx.tts_audio = None

            # Start image/code and music tasks immediately — parallel with TTS drain.
            # Image has higher priority: it is yielded as soon as it arrives, even
            # if TTS sentences are still queued.
            if self.code_exec and ctx.pending_plot_code:
                image_task = asyncio.create_task(
                    self.code_exec.execute(ctx.pending_plot_code)
                )
            elif self.image_gen and ctx.pending_image_prompt:
                image_task = asyncio.create_task(
                    self.image_gen.generate(ctx.pending_image_prompt)
                )

            if self.music_gen and ctx.pending_music_prompt:
                music_task = asyncio.create_task(
                    self.music_gen.generate(ctx.pending_music_prompt)
                )

            # Drain TTS queue, surfacing image/music as soon as they complete.
            for tts_task in tts_tasks:
                pending: set[asyncio.Task] = {tts_task}
                if image_task and not image_task.done():
                    pending.add(image_task)
                if music_task and not music_task.done():
                    pending.add(music_task)

                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                if image_task in done:
                    try:
                        ctx.generated_image = image_task.result()
                    except Exception:
                        pass
                    image_task = None

                if music_task in done:
                    try:
                        ctx.music_audio = music_task.result()
                    except Exception:
                        pass
                    music_task = None

                ctx.tts_audio = None
                if tts_task in done:
                    try:
                        ctx.tts_audio = tts_task.result()
                    except Exception:
                        pass
                yield ctx
                ctx.tts_audio = None

                # If image/music arrived before TTS, wait for TTS now and yield audio.
                if not tts_task.done():
                    try:
                        ctx.tts_audio = await tts_task
                        yield ctx
                        ctx.tts_audio = None
                    except Exception:
                        pass

            # Wait for any tasks that outlasted the TTS queue
            if image_task:
                try:
                    ctx.generated_image = await image_task
                except Exception:
                    pass
                image_task = None

            if music_task:
                try:
                    ctx.music_audio = await music_task
                except Exception:
                    pass
                music_task = None

            # Fire-and-forget: record student interaction asynchronously.
            if ctx.pending_interaction and self.student_state:
                asyncio.create_task(
                    self.student_state.record(
                        session_id=session.session_id,
                        **ctx.pending_interaction,
                    )
                )

            yield ctx

        finally:
            # Cancel all pending background tasks when the generator is closed
            # (normal completion, exception, or user-initiated Stop).
            for t in tts_tasks:
                t.cancel()
            if image_task:
                image_task.cancel()
            if music_task:
                music_task.cancel()
