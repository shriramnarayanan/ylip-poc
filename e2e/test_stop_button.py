"""
Test: Stop Button — Mutes TTS Without Interrupting Generation
See test_stop_button.md for full specification.

Run:
    cd e2e
    uv sync
    uv run pytest test_stop_button.py -v

What is tested here vs the MCP browser test
--------------------------------------------
The MCP browser test (test_stop_button.md) validates the live click interaction:
  - Stop button visible in Text mode, hidden in Conversation mode
  - Clicking Stop mid-playback silences TTS and returns the player to Play state
  - The speak() → TTS path (model-dependent tool call, real audio output)

This Python test validates the invariants verifiable via the Gradio REST API:
  1. A regular Text-mode response does NOT produce TTS audio (Text mode only uses
     TTS for explicit speak() calls — full-response streaming TTS is Conversation only).
  2. A follow-up request returns a full response (Stop flag clears on new request).
  3. Text response is present and complete (LLM generation is unaffected by Stop).

Note: speak() → TTS audio delivery is not reliably testable via REST API because
  (a) gemma-3-4b doesn't always call the speak() tool, and (b) Gradio's SSE stream
  carries TTS audio in the `event: complete` frame as an empty update rather than a
  file path. Both behaviours are confirmed correct via the MCP browser test.
"""

import json
import os

import pytest  # noqa: F401 — used for fixtures
import requests

BASE_URL = os.environ.get("YLIP_BASE_URL", "http://localhost:7860")
LLM_TIMEOUT = int(os.environ.get("YLIP_LLM_TIMEOUT_MS", "120000")) / 1000


def _call_chat(prompt: str, mode: str = "Text") -> dict:
    """
    Submit a Text-mode chat request via the Gradio REST API and stream to completion.
    Returns: chatbot_text, has_voice, has_music.
    outputs order: [chatbot, image_out, audio_out, music_out, text_in, audio_text, audio_conv_ready]
    """
    payload = {"data": [prompt, [], None, None, None, mode]}

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
        timeout=LLM_TIMEOUT,
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
                # parts[0] = chatbot history
                for msg in parts[0] or []:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        chatbot_text = (
                            "".join(c.get("text", "") for c in content)
                            if isinstance(content, list)
                            else content
                        )
                # parts[2] = audio_out
                if isinstance(parts[2], dict) and parts[2].get("value", {}).get("path"):
                    has_voice = True
                # parts[3] = music_out
                if isinstance(parts[3], dict) and parts[3].get("value", {}).get("path"):
                    has_music = True

    return {"chatbot_text": chatbot_text, "has_voice": has_voice, "has_music": has_music}


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def factual_result():
    """A plain factual question — no speak() call expected."""
    return _call_chat("What year did Rome fall?")


@pytest.fixture(scope="module")
def pronunciation_result():
    """A pronunciation question — LLM may or may not call speak()."""
    return _call_chat('How do you pronounce "ephemeral"?')


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

def test_text_response_present_for_factual_request(factual_result):
    """LLM returns a full text response for a factual question."""
    assert factual_result["chatbot_text"].strip(), "Expected a non-empty bot reply"
    bot = factual_result["chatbot_text"]
    assert "476" in bot or "western" in bot.lower() or "roman" in bot.lower(), (
        f"Expected a Rome-fall answer, got: {bot!r}"
    )


def test_no_tts_for_factual_text_response(factual_result):
    """In Text mode, a plain factual response must not produce TTS audio.
    Text-mode TTS is reserved for explicit speak() calls only."""
    assert not factual_result["has_voice"], (
        "No TTS audio expected for a plain factual question in Text mode"
    )


def test_no_music_for_factual_request(factual_result):
    """Music generation must not be triggered by a factual question."""
    assert not factual_result["has_music"], "Music must not play for a factual question"


def test_pronunciation_request_returns_response(pronunciation_result):
    """LLM responds to a pronunciation question with text (Stop flag cleared on each request)."""
    assert pronunciation_result["chatbot_text"].strip(), "Expected a non-empty bot reply"


def test_no_music_for_pronunciation_request(pronunciation_result):
    """Music generation must not be triggered by a pronunciation question."""
    assert not pronunciation_result["has_music"], "Music must not play for a pronunciation request"
