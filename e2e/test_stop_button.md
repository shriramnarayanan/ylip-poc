# Test: Stop Button — Mutes TTS Without Interrupting Generation

## Use Case
In Text mode TTS is only used to pronounce specific words the student requests.
If TTS is playing when the student clicks **Stop**, it is silenced immediately.
Any in-progress image, plot, or music generation is unaffected and renders normally
when complete.

## Preconditions
- Frontend running on http://localhost:7860
- LLM backend running
- TTS backend running on :8880
- Mode: **Text** (default)

## Steps — Part A: TTS pronunciation then Stop
1. Type: `How do you pronounce "ephemeral"?`
2. Click **Send**
3. Observe: the LLM responds with a text explanation and calls `speak("ephemeral")`
4. The word "ephemeral" is played back via TTS
5. Click **Stop** during playback
6. Observe: TTS audio stops; the text response remains visible

## Steps — Part B: Stop does not block image/plot rendering
1. Type: `What is a Gaussian Distribution? Plot one.`
2. Click **Send**
3. Wait until at least a few tokens appear in the chatbot
4. Click **Stop**
5. Observe: TTS is silenced (no audio plays for this response)
6. Observe: the plot image still renders in the gallery when code execution completes

## Steps — Part C: Normal request after Stop
1. Type: `What year did Rome fall?`
2. Click **Send**
3. Observe: a full response is returned (Stop flag is cleared at the start of each request)

## Expected Results
| # | What to check | Expected |
|---|---|---|
| 1 | Stop button visibility | Visible in Text mode, hidden in Conversation mode |
| 2 | TTS after Stop (Part A) | No audio output once Stop is clicked |
| 3 | Text response after Stop | Still present and complete — LLM generation is unaffected |
| 4 | Plot rendered after Stop (Part B) | Gallery shows the plot even though TTS was stopped |
| 5 | Follow-up response (Part C) | Full response returned — Stop flag cleared on new request |

## Notes
- In Text mode, TTS is only triggered by the LLM calling `speak(word)` — it does not
  read out the full response. Sentence-by-sentence streaming TTS is Conversation mode only.
- Stop sets a `threading.Event` flag (`_tts_stop`). The orchestrator checks the flag before
  synthesising each TTS call; image / plot / music `asyncio.Task` objects are not affected.
- The flag is cleared at the start of every new `chat()` invocation so subsequent requests
  work normally without requiring a page reload.
- Stop is only wired to Text-mode events. Conversation mode is unaffected.

## Findings (MCP browser validation)
- **Part A**: Stop button visible in Text mode ✓, hidden in Conversation mode ✓.
  TTS played for `speak("ephemeral")`; clicking Stop returned the audio player to
  Play state and silenced playback. Text response remained visible.
- **Part B**: Stop clicked mid-stream after LLM began streaming. No new TTS audio
  was generated for the Gaussian response (Voice response still showed prior ephemeral
  audio). The LLM in this run included code inline in its text response rather than
  calling `plot_function` as a tool call, so the code-exec plot path was not exercised.
  The plot-renders-after-stop invariant is enforced by design (image/code tasks are not
  checked against `_tts_stop`) and is covered by the Python test.
- **Part C**: Follow-up "What year did Rome fall?" returned the full answer
  ("476 CE… Western Roman Empire…") — Stop flag was cleared on the new request ✓.

## Status
- [x] Validated via MCP browser
- [x] Python test written
