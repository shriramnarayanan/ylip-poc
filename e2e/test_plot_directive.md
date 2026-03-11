# Test: PLOT Directive Renders a Graph

## Use Case
The student asks the tutor to explain and plot a mathematical function.

## Preconditions
- Frontend running on http://localhost:7860
- LLM backend (LM Studio) running
- Code execution backend running on :8883
- TTS backend running on :8880
- Mode: **Text** (default)

## Steps
1. In the text input, type: `What is a Gaussian Distribution? Plot one.`
2. Click **Send**
3. Wait for the response to finish streaming

## Expected Results
| # | What to check | Expected |
|---|---------------|----------|
| 1 | Chatbot assistant message | Non-empty text describing Gaussian distribution |
| 2 | Generated image gallery | Contains a rendered plot image |
| 3 | Voice response audio | Present (TTS ran on the text portion) |
| 4 | Music audio | Absent (math topic should not trigger MUSIC:) |
| 5 | Chatbot text | Does NOT contain raw `PLOT:`, `IMAGE:`, `MUSIC:` tokens |
| 6 | Chatbot text | Does NOT contain a fenced python code block |

## Notes
- The PLOT: directive and its code block must be stripped from the display text before rendering in the chatbot.
- TTS must synthesize only the text description, not the Python code.
- If the code execution backend is unreachable, the test fails at step 2 (expected image gallery).

## Status
- [x] Validated via Playwright MCP
- [x] Python test written

## Findings during validation
- LLM (Gemma-3-4b) sometimes emits plain ` ```python ``` ` blocks instead of `PLOT:` directive. Fixed by adding `_CODE_BLOCK_RE` fallback in `orchestrator._code_exec_step`.
- TTS was reading Python code aloud (36s audio). Fixed by stripping all code blocks in `_strip_directives`.
- `scipy` was missing from code_exec container. Fixed by adding `scipy>=1.11` to `backend/code_exec/pyproject.toml` and rebuilding.
