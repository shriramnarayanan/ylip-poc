# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

The scope of this project is to plan and construct a version of "The Young Lady's Illustrated Primer" from "The Diamond Age". The output is a portable device with camera, microphone and typing inputs (keyboard or on-screen). The outputs are svisuals (pictures, images, graphs), audio (speech and music) and text.

The device should be capable of operating oofline, so it needs to host one more more models (vision recognition, STT, TTS, image generation, and LLM).

The device must have a battery with large enough capacity to last 4 hours before re-charging.

The device is personal. It is expected to be used by one student. It will accomany the student through middle and high school. The goal of the device is to keep the student curious, encourage practical experimentation, working with their hands, and to collaborate with others.

For hardware, likely choices are m5stack (https://shop.m5stack.com/) or a raspberry PI with an Qualcomm QCS8550 LLM accelerator. The PI variant is here (https://shop.m5stack.com/products/ai-8850-llm-accelerator-m-2-kit-8gb-version-ax8850). There are various modules that platforms have for cameras (with built-in vision models), and for audio in / out. The device should be able to download content from the internet into local storage to become a subject matter expert. The internet sources are from a white list. This is likely wikipedia, published text books, and peer-reviewed papers.

For software, the front end UI should allow text, camera, and audio inputs. It should be able to render text, images and audio as output. The model will auto-compact periodically (either on a schedule, or depending on usage). However, it should still be able to observe long-term trends in the student, and be able to re-visit areas where it observers the student struggling. The tone of the teacher should be neutral - it should be critical of the student's responses. Encouragement should be reserved for when the student struggles.

Do not use system python. Use python 3.14 in the user's home directory. For all python projects, create a venv and activate it before running any scripts. For projects that use Docker, generate the lockfile this way. Use nvm for Node.JS projects

### RUNNING PYTHON
Always activate the python venv, and run uv from the user's python installation.

### RUNNING THE FRONTEND
In the frontend directory after activating the venv, use uv to `run python .\src\main.py`

### END TO END TESTS
Use the Playwright MCP server to test new features or changes to the UI. Before making code changes, create or update the test cases in the e2e markdown files. Use one use-case per file. These changes should be reviewed and approved by a human. Tests should be comprehensive across models, UI panels, and conversation modes. Examples:
- In Text mode, user specifies text input "What is a Gaussian Distribution? Plot one", LLM responds with a PLOT directive which renders a graph. There is no MUSIC: and the TTS model translates the LLM response, but does not include the PLOT directive contents.
- In Text mode, user uploads an image of a cat, and asks "What is this animal"? LLM generates a text response and maybe has IMAGE: and MUSIC: directives. If there is a MUSIC directive, the TTS model doesn't run.
- In Conversation mode, user uploads an image of a cat and asks "Identify this picture, and the style it is in". The LLM  generates a text response that gets converted via TTS and automatically plays back. There is no MUSIC: directive

Once the e2e markdown changes have been approved, proceed with changes to the codebase. Validate the codebase changes using the approved e2e markdown files. If additional changes to the e2e markdown files are needed, make the changes, but always get them reviewed and approved by a human. Finally, after changes to the codebase are complete (and all the e2e markdown tests are passing), convert the e2e markdown tests into corresponding python tests so they can be executed without Claude.