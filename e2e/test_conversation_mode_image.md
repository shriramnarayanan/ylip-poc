# Test: Conversation Mode — Image Upload with Voice Question

## Use Case
In Conversation mode the student uploads a photo and speaks a question about it.
The tutor responds in text and the TTS engine reads the response aloud automatically.
Because MUSIC: is not part of the Conversation-mode system prompt, music is never
generated in this mode.

## Preconditions
- Frontend running on http://localhost:7860
- LLM backend with vision model (moondream2 via LM Studio) running
- TTS backend running on :8880
- Mode: **Conversation** (must be explicitly switched from the default Text mode)

## Steps
1. Switch the **Mode** radio to **Conversation**
2. Upload an image of a cat via the **Upload image** component
3. Ask via audio: `Identify this picture, and the style it is in`
4. Wait for the response to finish streaming

## Expected Results
| # | What to check | Expected |
|---|---------------|----------|
| 1 | Chatbot assistant message | Non-empty text identifying the subject and/or style |
| 2 | Chatbot text | Does NOT contain raw `IMAGE:`, `MUSIC:`, or `PLOT:` tokens |
| 3 | Voice response audio | Present — TTS always runs in Conversation mode |
| 4 | Music audio | Absent — MUSIC: directive is excluded from the Conversation-mode system prompt |

## Out of scope
STT accuracy (correct transcription of the spoken prompt) is a unit concern for the STT backend and is tested separately, not here.

## Notes
- `SYSTEM_PROMPT_CONVERSATION` strips the `MUSIC:` directive description entirely, so the
  LLM never generates a MUSIC: directive in this mode.
- TTS is unconditional in Conversation mode (no MUSIC: can suppress it).
- The vision model receives the image as a base64-encoded PNG alongside the user text.
- Audio input (microphone) is the trigger in Conversation mode. The API-level test passes
  `assets/identify_picture_style.wav` (generated from the Kokoro TTS backend) as
  `audio_conv_ready` with `mode_state="Conversation"`, exercising the LLM → TTS chain.

## Status
- [x] Validated via MCP browser (Gradio REST API + SSE) — all checks PASS
- [x] Python test written

## Findings during validation
- `gr.Audio(type="numpy")` does NOT accept base64 data URLs (unlike `gr.Image`). Audio must be uploaded via `POST /gradio_api/upload` to get a server-side path first.
- The audio `FileData` object requires `"meta": {"_type": "gradio.FileData"}` (not `{}`); without it Gradio returns `event: error, data: null` immediately.
- `ctx.text` in the orchestrator combines the image description and STT transcript: `[Image description: "..."]\n\n<stt text>`. The user chatbot message reflects this combined value.
- All 4 pipeline checks (LLM response, no raw directives, TTS voice present, no music) are validated end-to-end via the Gradio REST API.
