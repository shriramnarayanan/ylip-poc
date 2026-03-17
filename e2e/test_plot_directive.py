"""
Test: PLOT Directive Renders a Graph
See test_plot_directive.md for full specification.

Run:
    cd e2e
    uv sync
    uv run pytest test_plot_directive.py -v
"""

import json
import os

import pytest
import requests

BASE_URL = os.environ.get("YLIP_BASE_URL", "http://localhost:7860")
CODE_EXEC_URL = os.environ.get("YLIP_CODE_EXEC_URL", "http://localhost:8883")
LLM_TIMEOUT = int(os.environ.get("YLIP_LLM_TIMEOUT_MS", "120000")) / 1000
PROMPT = "What is a Gaussian Distribution? Plot one."


def _code_exec_available() -> bool:
    """Return True if the code execution backend is reachable (any HTTP response)."""
    try:
        requests.get(f"{CODE_EXEC_URL}/", timeout=3)
        return True
    except Exception:
        return False


plot_skip = pytest.mark.skipif(
    not _code_exec_available(),
    reason="Code execution backend not running — start it and re-run",
)


def _call_chat(prompt: str) -> dict:
    """
    Submit a Text-mode chat request via the Gradio REST API and stream to completion.
    Returns: chatbot_text, has_plot, has_voice, has_music.
    """
    payload = {"data": [prompt, [], None, None, None, "Text"]}

    join = requests.post(
        f"{BASE_URL}/gradio_api/call/chat",
        json=payload,
        timeout=30,
    )
    join.raise_for_status()
    event_id = join.json()["event_id"]

    chatbot_text = ""
    has_plot = False
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
                for msg in (parts[0] or []):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        chatbot_text = (
                            "".join(c.get("text", "") for c in content)
                            if isinstance(content, list)
                            else content
                        )
                # parts[1] = image_out gallery — list of {"image": {"path": ...}} dicts
                if isinstance(parts[1], list) and parts[1]:
                    has_plot = True
                if isinstance(parts[2], dict) and parts[2].get("value", {}).get("path"):
                    has_voice = True
                if isinstance(parts[3], dict) and parts[3].get("value", {}).get("path"):
                    has_music = True

    return {
        "chatbot_text": chatbot_text,
        "has_plot": has_plot,
        "has_voice": has_voice,
        "has_music": has_music,
    }


@pytest.fixture(scope="module")
def plot_result():
    """Run the full pipeline once; share result across all tests."""
    return _call_chat(PROMPT)


def test_chatbot_has_response(plot_result):
    assert plot_result["chatbot_text"].strip(), "Expected non-empty assistant reply"


@plot_skip
def test_plot_image_rendered(plot_result):
    assert plot_result["has_plot"], "Expected a plot image in the gallery"


def test_voice_audio_absent(plot_result):
    # Text mode TTS only fires for explicit speak() calls.
    # A math/plot prompt does not call speak(), so no audio is expected.
    assert not plot_result["has_voice"], "No TTS audio expected for a math/plot prompt in Text mode"


def test_music_absent(plot_result):
    assert not plot_result["has_music"], "Music must not be generated for a math topic"


def test_no_raw_directives_in_chat(plot_result):
    text = plot_result["chatbot_text"]
    for directive in ("PLOT:", "IMAGE:", "MUSIC:"):
        assert directive not in text, f"Raw directive {directive!r} leaked into chatbot"
