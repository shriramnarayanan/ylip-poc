"""
Gradio UI for the YLIP prototype.
Wires user inputs to the Orchestrator and renders streamed outputs.

Two modes:
  Text         — type to send, mic optional, TTS optional
  Conversation — mic auto-submits on stop, TTS always plays
"""

import asyncio
import io
from dataclasses import dataclass, field

import gradio as gr
import numpy as np

from config import settings, SYSTEM_PROMPT, SYSTEM_PROMPT_CONVERSATION
from core.orchestrator import Orchestrator, _strip_directives
from core.session import Session

orchestrator = Orchestrator()
session = Session()


# ------------------------------------------------------------------ #
# VAD state + chunk processor                                          #
# ------------------------------------------------------------------ #

_SPEECH_THRESHOLD = 0.008   # RMS of float32 signal
_SILENCE_FRAMES   = 3       # consecutive silent ~500 ms chunks ≈ 1.5 s
_MAX_CHUNKS       = 120     # safety cap: ~60 s before auto-reset


@dataclass
class ConvState:
    chunks: list = field(default_factory=list)
    sample_rate: int = 16000
    silence_frames: int = 0
    has_speech: bool = False


def process_chunk(chunk, state: ConvState, paused: bool, trigger_count: int):
    """Accumulate streaming mic chunks; return a completed utterance on speech-end.

    Returns (new_state, ready_audio_or_None, new_trigger_count).
    trigger_count is incremented only when a completed utterance is ready, so
    conv_trigger.change reliably fires only for real speech — not when audio is
    cleared to None after processing.
    """
    no_change = gr.update()
    if paused or chunk is None:
        return state, no_change, no_change

    sr, data = chunk
    if data is None or data.size == 0:
        return state, no_change, no_change

    # Normalise to float32 for RMS calculation
    if data.dtype.kind == "i":
        data_f = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data_f = data.astype(np.float32)

    state.chunks.append(data)
    state.sample_rate = sr

    rms = float(np.sqrt(np.mean(data_f ** 2)))
    if rms > _SPEECH_THRESHOLD:
        state.has_speech = True
        state.silence_frames = 0
    elif state.has_speech:
        state.silence_frames += 1

    # Safety cap: discard and reset if accumulating too long without resolution
    if len(state.chunks) > _MAX_CHUNKS:
        state.chunks = []
        state.silence_frames = 0
        state.has_speech = False
        return state, no_change, no_change

    # Speech ended: compile and return the utterance
    if state.has_speech and state.silence_frames >= _SILENCE_FRAMES:
        full = np.concatenate(state.chunks)
        ready = (state.sample_rate, full)
        state.chunks = []
        state.silence_frames = 0
        state.has_speech = False
        return state, ready, int(trigger_count) + 1

    return state, no_change, no_change


# ------------------------------------------------------------------ #
# Chat handler                                                         #
# ------------------------------------------------------------------ #

async def chat(message: str, history: list[dict], image, audio_text, audio_conv, mode: str):
    # gr.Textbox returns None when empty; normalise to empty string
    message = message or ""
    audio = audio_conv if mode == "Conversation" else audio_text
    if not message and audio is None and image is None:
        return

    image_bytes = None
    if image is not None:
        from PIL import Image as PILImage
        buf = io.BytesIO()
        if isinstance(image, str):
            PILImage.open(image).save(buf, format="PNG")
        else:
            PILImage.fromarray(image).save(buf, format="PNG")
        image_bytes = buf.getvalue()

    audio_bytes = None
    if audio is not None:
        import soundfile as sf
        sample_rate, data = audio
        buf = io.BytesIO()
        sf.write(buf, data, sample_rate, format="WAV")
        audio_bytes = buf.getvalue()

    partial_response = ""
    final_image = None
    final_music = None
    # user_display is updated to the STT transcription on the first context yield
    user_display = message
    # Tracks when the currently-playing audio sentence will end (monotonic clock).
    # Used to delay the next sentence so it doesn't interrupt the previous one.
    audio_ends_at = 0.0

    prompt = SYSTEM_PROMPT_CONVERSATION if mode == "Conversation" else SYSTEM_PROMPT
    async for ctx in orchestrator.stream_run(
        session=session,
        text=message,
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        system_prompt=prompt,
    ):
        # ctx.text is the final input text (STT transcription if audio was provided)
        if ctx.text:
            user_display = ctx.text
        partial_response = _strip_directives(ctx.llm_response or "")
        final_image = ctx.generated_image
        final_music = ctx.music_audio

        current_history = list(history) + [
            {"role": "user", "content": user_display},
            {"role": "assistant", "content": partial_response},
        ]
        image_out = [_bytes_to_pil(final_image)] if final_image else []

        if ctx.tts_audio:
            sr, arr = _bytes_to_audio(ctx.tts_audio)
            duration = len(arr) / sr
            # Sleep until the previous sentence has finished, then autoplay next.
            # asyncio.sleep yields to the event loop so text tokens still stream.
            wait = audio_ends_at - asyncio.get_event_loop().time() + 0.15
            if wait > 0:
                await asyncio.sleep(wait)
            audio_update = gr.update(value=(sr, arr))
            audio_ends_at = asyncio.get_event_loop().time() + duration
        else:
            audio_update = gr.update()

        music_update = gr.update(value=_bytes_to_audio(final_music), autoplay=True) if final_music else gr.update()

        yield (
            current_history,
            image_out,
            audio_update,
            music_update,
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=None),
        )

    session.add("user", user_display)
    session.add("assistant", partial_response)


def clear_session():
    session.clear()
    return [], [], None, None


def _bytes_to_pil(data: bytes):
    from PIL import Image as PILImage
    return PILImage.open(io.BytesIO(data))


def _bytes_to_audio(data: bytes):
    import soundfile as sf
    import numpy as np
    buf = io.BytesIO(data)
    audio_array, sample_rate = sf.read(buf)
    return sample_rate, audio_array


# ------------------------------------------------------------------ #
# Mode switch                                                          #
# ------------------------------------------------------------------ #

def switch_mode(mode: str):
    is_conv = mode == "Conversation"
    return (
        gr.update(visible=not is_conv),  # text_in
        gr.update(visible=not is_conv),  # submit_btn
        gr.update(visible=not is_conv),  # audio_text
        gr.update(visible=is_conv),      # audio_conv
        gr.update(visible=is_conv),      # pause_btn
        ConvState(),                     # reset VAD state
    )


def toggle_pause(paused: bool):
    new_val = not paused
    label = "Resume mic" if new_val else "Pause mic"
    return new_val, gr.update(value=label)


# ------------------------------------------------------------------ #
# Status helpers                                                       #
# ------------------------------------------------------------------ #

def model_status() -> str:
    lines = [
        f"**LLM:** {settings.lm_studio_model} @ {settings.lm_studio_base_url}",
        f"**TTS:** {'enabled — ' + settings.tts_model if settings.tts_enabled else 'disabled'}",
        f"**STT:** {'enabled — whisper/' + settings.stt_model if settings.stt_enabled else 'disabled'}",
        f"**Image gen:** {'enabled — ' + settings.image_model if settings.image_enabled else 'disabled'}",
        f"**Vision:** {'enabled — ' + settings.vision_model if settings.vision_enabled else 'disabled'}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Layout                                                               #
# ------------------------------------------------------------------ #

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="YLIP — Young Lady's Illustrated Primer") as demo:
        gr.Markdown("# Young Lady's Illustrated Primer")

        with gr.Row():
            gr.Markdown(model_status())
            mode = gr.Radio(
                ["Text", "Conversation"],
                value="Text",
                label="Mode",
                scale=0,
            )

        mode_state   = gr.State("Text")
        conv_state   = gr.State(ConvState())
        mic_paused   = gr.State(False)
        conv_trigger = gr.Number(value=0, visible=False)  # increments on each VAD utterance

        # ---- Main output area ----------------------------------------
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=500)
            with gr.Column(scale=1):
                image_out = gr.Gallery(
                    label="Generated image",
                    columns=1,
                    type="pil",
                    allow_preview=True,
                    preview=True,
                )
                audio_out = gr.Audio(label="Voice response", autoplay=True)
                music_out = gr.Audio(label="Music", autoplay=True)

        # ---- Conversation mode: streaming mic + pause button ---------
        with gr.Row():
            audio_conv = gr.Audio(
                label="Speak",
                sources=["microphone"],
                type="numpy",
                streaming=True,
                visible=False,
            )
            pause_btn = gr.Button("Pause mic", visible=False, size="sm", variant="secondary")

        # Hidden component: receives the completed utterance from VAD,
        # whose .change event triggers the chat function.
        audio_conv_ready = gr.Audio(visible=False, type="numpy")

        # ---- Text mode: text input + buttons -------------------------
        with gr.Row():
            text_in = gr.Textbox(
                placeholder="Ask something...",
                label="",
                scale=5,
                autofocus=True,
                lines=2,
            )
            with gr.Column(scale=1, min_width=120):
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear session")

        # ---- Shared: image upload + optional mic in text mode --------
        with gr.Row():
            image_in = gr.Image(
                label="Upload image",
                type="numpy",
                sources=["upload", "clipboard"],
                height=150,
            )
            audio_text = gr.Audio(
                label="Microphone",
                sources=["microphone"],
                type="numpy",
            )

        # ---- Wire mode toggle ----------------------------------------
        mode.change(
            fn=lambda m: (m, *switch_mode(m)),
            inputs=[mode],
            outputs=[mode_state, text_in, submit_btn, audio_text, audio_conv, pause_btn, conv_state],
        )

        # ---- Pause / resume mic ------------------------------------
        pause_btn.click(fn=toggle_pause, inputs=[mic_paused], outputs=[mic_paused, pause_btn])

        # ---- VAD: stream mic chunks → bump counter when utterance ready ---
        audio_conv.stream(
            fn=process_chunk,
            inputs=[audio_conv, conv_state, mic_paused, conv_trigger],
            outputs=[conv_state, audio_conv_ready, conv_trigger],
        )

        # ---- Common inputs/outputs lists -----------------------------
        # audio_conv_ready carries the completed utterance; conv_trigger signals it
        inputs  = [text_in, chatbot, image_in, audio_text, audio_conv_ready, mode_state]
        outputs = [chatbot, image_out, audio_out, music_out, text_in, audio_text, audio_conv_ready]

        # Text mode: button and Enter
        submit_btn.click(fn=chat, inputs=inputs, outputs=outputs)
        text_in.submit(fn=chat, inputs=inputs, outputs=outputs)

        # Conversation mode: trigger fires only when real audio arrives (not when cleared)
        conv_trigger.change(fn=chat, inputs=inputs, outputs=outputs)

        clear_btn.click(fn=clear_session, outputs=[chatbot, image_out, audio_out, music_out])

    return demo
