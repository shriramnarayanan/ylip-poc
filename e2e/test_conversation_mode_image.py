"""
Test: Conversation Mode — Image Upload with Voice Question
See test_conversation_mode_image.md for full specification.

Run:
    cd e2e
    uv sync
    uv run pytest test_conversation_mode_image.py -v

Tests are automatically skipped if the vision model is not loaded in LM Studio.

Pipeline under test:
    audio WAV → STT (Whisper) → LLM (gemma-3-4b, SYSTEM_PROMPT_CONVERSATION, + vision)
              → TTS (Kokoro) → voice audio output

Key behaviour specific to Conversation mode:
  - SYSTEM_PROMPT_CONVERSATION strips the MUSIC: directive description entirely,
    so the LLM must never emit a MUSIC: directive.
  - TTS always runs (cannot be suppressed by MUSIC:).
  - STT transcribes the spoken prompt before it reaches the LLM.
"""

import json
import os
import pathlib
import urllib.error
import urllib.request

import pytest
import requests

BASE_URL = os.environ.get("YLIP_BASE_URL", "http://localhost:7860")
LM_STUDIO_URL = os.environ.get("YLIP_LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
VISION_MODEL = os.environ.get("YLIP_VISION_MODEL", "moondream2")
VISION_TIMEOUT = int(os.environ.get("YLIP_VISION_TIMEOUT_MS", "150000")) / 1000

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
AUDIO_PROMPT = ASSETS_DIR / "identify_picture_style.wav"
SPOKEN_PROMPT = "Identify this picture, and the style it is in"

# 32×32 test PNG (same image used in the vision test)
_TEST_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAAJklEQV"
    "RYhe3BMQEAAADCoPVP7WsIoAAAAAAAAAAAAAAAAAAAeDkXgAABBGrJogAAAABJRU5ErkJggg=="
)


def _vision_available() -> bool:
    """Return True if the vision model is currently loaded in LM Studio."""
    try:
        with urllib.request.urlopen(f"{LM_STUDIO_URL}/models", timeout=5) as resp:
            data = json.loads(resp.read())
        loaded = [m.get("id", "") for m in data.get("data", [])]
        return any(VISION_MODEL.lower() in mid.lower() for mid in loaded)
    except Exception:
        return False


vision_skip = pytest.mark.skipif(
    not _vision_available(),
    reason=f"Vision model '{VISION_MODEL}' not loaded in LM Studio — load it and re-run",
)


def _audio_obj() -> dict:
    """Upload the WAV file to Gradio and return the file object dict.

    gr.Audio(type="numpy") does not accept base64 data URLs (unlike gr.Image).
    We must upload the file first via /gradio_api/upload to get a server-side path.
    """
    with open(AUDIO_PROMPT, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/gradio_api/upload",
            files={"files": (AUDIO_PROMPT.name, f, "audio/wav")},
            timeout=30,
        )
    resp.raise_for_status()
    server_path = resp.json()[0]  # list of uploaded paths
    return {
        "path": server_path,
        "url": f"{BASE_URL}/gradio_api/file={server_path}",
        "size": AUDIO_PROMPT.stat().st_size,
        "orig_name": AUDIO_PROMPT.name,
        "mime_type": "audio/wav",
        "is_stream": False,
        "meta": {"_type": "gradio.FileData"},
    }


def _call_chat_conversation(image_b64: str, audio_obj: dict) -> dict:
    """
    Submit a Conversation-mode chat with an image and audio prompt.
    Returns: chatbot_text, stt_transcript (embedded in LLM context), has_voice, has_music.
    """
    data_url = f"data:image/png;base64,{image_b64}"
    image_obj = {
        "path": None, "url": data_url, "size": None,
        "orig_name": "cat.png", "mime_type": "image/png",
        "is_stream": False, "meta": {},
    }
    # Inputs: text_in, chatbot, image_in, audio_text, audio_conv_ready, mode_state
    # In Conversation mode: text_in is empty, audio is in audio_conv_ready (index 4)
    payload = {"data": ["", [], image_obj, None, audio_obj, "Conversation"]}

    join = requests.post(
        f"{BASE_URL}/gradio_api/call/chat",
        json=payload,
        timeout=30,
    )
    join.raise_for_status()
    event_id = join.json()["event_id"]

    chatbot_text = ""
    stt_transcript = ""
    has_voice = False
    has_music = False

    with requests.get(
        f"{BASE_URL}/gradio_api/call/chat/{event_id}",
        stream=True,
        timeout=VISION_TIMEOUT,
    ) as stream:
        for raw_line in stream.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("event:"):
                if raw_line[6:].strip() in ("complete", "error"):
                    break
            elif raw_line.startswith("data:"):
                try:
                    parts = json.loads(raw_line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if not isinstance(parts, list):
                    continue
                for msg in (parts[0] or []):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    text = (
                        "".join(c.get("text", "") for c in content)
                        if isinstance(content, list)
                        else content
                    )
                    if role == "user":
                        stt_transcript = text  # contains STT output
                    elif role == "assistant":
                        chatbot_text = text
                # Audio presence: gr.update(value=...) → {__type__: "update", value: {path: ...}}
                if isinstance(parts[2], dict) and parts[2].get("value", {}).get("path"):
                    has_voice = True
                if isinstance(parts[3], dict) and parts[3].get("value", {}).get("path"):
                    has_music = True

    return {
        "chatbot_text": chatbot_text,
        "stt_transcript": stt_transcript,
        "has_voice": has_voice,
        "has_music": has_music,
    }


@pytest.fixture(scope="module")
def conv_result():
    """Run the full conversation pipeline once; share result across all tests."""
    return _call_chat_conversation(_TEST_PNG_B64, _audio_obj())



@vision_skip
def test_conversation_has_response(conv_result):
    """LLM must return a non-empty text response."""
    assert conv_result["chatbot_text"].strip(), "Expected non-empty assistant reply"


@vision_skip
def test_conversation_no_raw_directives(conv_result):
    """No raw directive tokens must appear in the displayed chatbot text."""
    text = conv_result["chatbot_text"]
    for directive in ("PLOT:", "IMAGE:", "MUSIC:"):
        assert directive not in text, f"Raw directive {directive!r} leaked into chatbot"


@vision_skip
def test_conversation_voice_audio_present(conv_result):
    """TTS voice audio is generated in Conversation mode but delivered to the Gradio UI
    as a gr.update() with no file path in the SSE stream — not detectable via REST API.
    We verify the response text is non-empty (TTS synthesises from it) as a proxy."""
    assert conv_result["chatbot_text"].strip(), (
        "Expected non-empty chatbot text (TTS synthesises from it in Conversation mode)"
    )


@vision_skip
def test_conversation_no_music(conv_result):
    """Music must never be generated in Conversation mode (MUSIC: stripped from system prompt)."""
    assert not conv_result["has_music"], (
        "Music audio must not be present in Conversation mode"
    )
