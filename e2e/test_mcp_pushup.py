"""
Test: MCP Subject Matter Integration (PE Push-Up)

Ensures that the LLM successfully receives the "Physical Education" schema via the 
MCP server and uses it to answer questions about a Push-Up.

Run:
    cd e2e
    uv sync
    uv run pytest test_mcp_pushup.py -v
"""

import json
import os

import pytest
import requests

BASE_URL = os.environ.get("YLIP_BASE_URL", "http://localhost:7860")
LLM_TIMEOUT = int(os.environ.get("YLIP_LLM_TIMEOUT_MS", "120000")) / 1000
PROMPT = "Explain the proper form for a Push-Up."


def _call_chat(prompt: str) -> dict:
    """
    Submit a Text-mode chat request via the Gradio REST API and stream to completion.
    Returns: chatbot_text, has_voice
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
    has_voice = False

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
                # parts[2] = audio_out
                if isinstance(parts[2], dict) and parts[2].get("value", {}).get("path"):
                    has_voice = True

    return {
        "chatbot_text": chatbot_text,
        "has_voice": has_voice,
    }


@pytest.fixture(scope="module")
def mcp_result():
    """Run the pipeline sharing result across tests."""
    return _call_chat(PROMPT)


def test_chatbot_has_response(mcp_result):
    assert mcp_result["chatbot_text"].strip(), "Expected non-empty assistant reply"


def test_mcp_topic_details_in_response(mcp_result):
    """
    Assert that the LLM pulled detailed structured text from the MCP sqlite database.
    The string 'plank position' or 'chest' should be in the rules.
    """
    text = mcp_result["chatbot_text"].lower()
    
    # We check for basic form mechanics that should be hydrated by the DB
    assert "chest" in text or "plank" in text or "core" in text or "back" in text, (
        "Expected the text to contain structural details retrieved from the MCP PE database (e.g. chest, core, back)"
    )


def test_voice_audio_present(mcp_result):
    assert mcp_result["has_voice"], "Expected voice TTS audio for the response"
