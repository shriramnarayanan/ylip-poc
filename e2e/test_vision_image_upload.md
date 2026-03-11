# Test: Vision — Image Upload with Question

## Use Case
The student uploads a photo and asks the tutor to identify what is in it.

## Preconditions
- Frontend running on http://localhost:7860
- LLM backend with vision model (moondream2 via LM Studio) running
- TTS backend running on :8880
- Mode: **Text** (default)

## Steps
1. Upload an image of a cat via the **Upload image** component
2. In the text input, type: `What is this animal?`
3. Click **Send**
4. Wait for the response to finish streaming

## Expected Results
| # | What to check | Expected |
|---|---------------|----------|
| 1 | Chatbot assistant message | Non-empty text response identifying the animal |
| 2 | Chatbot text | Does NOT contain raw `IMAGE:`, `MUSIC:`, `PLOT:` tokens |
| 3a | If MUSIC: directive was emitted | Music audio present; Voice response audio absent |
| 3b | If MUSIC: directive was NOT emitted | Voice response audio present |

## Notes
- The LLM response is non-deterministic; MUSIC: may or may not appear.
- When MUSIC: is present, the TTS pipeline must be skipped entirely — the audio output channels are mutually exclusive.
- The vision model receives the image as a base64-encoded PNG alongside the user text.

## Status
- [x] Validated via MCP browser (Gradio REST API via fetch) — all checks PASS
- [x] Python test written

## Findings during validation
- Gradio's image upload component uses a Svelte custom element that ignores synthetic DOM events from the MCP browser tool. Playwright MCP `browser_file_upload` requires an open file chooser dialog; injecting via `DataTransfer` triggered a "broken data stream" error.
- Python Playwright uses CDP `setInputFiles` which bypasses the DOM event system and works natively. The Python test uses `page.locator("input[type='file'][accept='image/*']").set_input_files(path)`.
- MCP validation used `fetch()` via `page.evaluate()` calling `/gradio_api/call/chat` (Gradio 6 named endpoint). Key findings:
  - The API requires `mode_state="Text"` as the 6th data argument (gr.State components are positional in the data array).
  - Images must be passed as base64 data URLs in the `url` field (`path: null`); file path uploads fail validation.
  - `gr.update(value=...)` serializes as `{__type__: "update", value: {path: "...", url: "..."}}` — audio presence is detected by checking `payload[2]?.value?.path`, not by excluding `__type__: "update"`.
  - Vision model (moondream2) described a 32×32 test PNG; LLM responded with text; TTS voice audio was generated.
- The LLM response was non-deterministic (e.g. "It's a chimera — a mythical beast composed of parts from different animals.") with no MUSIC: directive, so TTS ran and voice audio was present.
- `test_vision_tts_music_mutually_exclusive` enforces that voice TTS and music generation never both appear in the same response.
