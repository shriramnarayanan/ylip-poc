"""
Test: Vision — Image Upload with Question
See test_vision_image_upload.md for full specification.

Run:
    cd e2e
    uv sync
    uv run pytest test_vision_image_upload.py -v

Tests are automatically skipped if the vision model is not loaded in LM Studio.

Implementation note: Playwright's set_input_files() triggers Gradio's image component
which calls the vision model via LM Studio's model-switching mechanism — this can take
several minutes. Instead these tests call the Gradio REST API directly using requests,
which is the same approach validated via the MCP browser.
"""

import base64
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
VISION_TIMEOUT = int(os.environ.get("YLIP_VISION_TIMEOUT_MS", "150000")) / 1000  # convert to seconds
PROMPT = "What is this animal?"
ASSETS_DIR = pathlib.Path(__file__).parent / "assets"

# 32×32 solid-colour PNG used as a test image (moondream2 describes it as an animal)
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


def _call_chat_with_image(image_b64: str, prompt: str) -> dict:
    """
    Submit a chat request with a base64 PNG image via the Gradio REST API and
    stream the SSE response to completion. Returns a dict with:
      chatbot_text, has_voice, has_music
    """
    data_url = f"data:image/png;base64,{image_b64}"
    image_obj = {
        "path": None, "url": data_url, "size": None,
        "orig_name": "cat.png", "mime_type": "image/png",
        "is_stream": False, "meta": {},
    }
    # gr.State components are positional in the data array (mode_state = "Text" is 6th)
    payload = {"data": [prompt, [], image_obj, None, None, "Text"]}

    join = requests.post(
        f"{BASE_URL}/gradio_api/call/chat",
        json=payload,
        timeout=30,
    )
    join.raise_for_status()
    event_id = join.json()["event_id"]

    chatbot_text = ""
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
                event_type = raw_line[6:].strip()
                if event_type in ("complete", "error"):
                    break
            elif raw_line.startswith("data:"):
                try:
                    parts = json.loads(raw_line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if not isinstance(parts, list):
                    continue
                # Extract assistant text from chatbot messages
                for msg in (parts[0] or []):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            chatbot_text = "".join(c.get("text", "") for c in content)
                        else:
                            chatbot_text = content
                # gr.update(value=...) serialises as {__type__: "update", value: {path: ...}}
                # Detect real audio by checking value.path (truthy only when audio was produced)
                if isinstance(parts[2], dict) and parts[2].get("value", {}).get("path"):
                    has_voice = True
                if isinstance(parts[3], dict) and parts[3].get("value", {}).get("path"):
                    has_music = True

    return {"chatbot_text": chatbot_text, "has_voice": has_voice, "has_music": has_music}


@vision_skip
def test_vision_has_response():
    result = _call_chat_with_image(_TEST_PNG_B64, PROMPT)
    assert result["chatbot_text"].strip(), "Expected non-empty assistant reply"


@vision_skip
def test_vision_no_raw_directives():
    result = _call_chat_with_image(_TEST_PNG_B64, PROMPT)
    text = result["chatbot_text"]
    for directive in ("PLOT:", "IMAGE:", "MUSIC:"):
        assert directive not in text, f"Raw directive {directive!r} leaked into chatbot"


@vision_skip
def test_vision_audio_present():
    """Either voice response or music audio must be present."""
    result = _call_chat_with_image(_TEST_PNG_B64, PROMPT)
    assert result["has_voice"] or result["has_music"], (
        "Expected either voice or music audio to be present"
    )


@vision_skip
def test_vision_tts_music_mutually_exclusive():
    """Voice TTS and music generation must not both be active simultaneously."""
    result = _call_chat_with_image(_TEST_PNG_B64, PROMPT)
    assert not (result["has_voice"] and result["has_music"]), (
        "Voice TTS and Music should be mutually exclusive — both were present"
    )
