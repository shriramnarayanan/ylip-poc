# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

The scope of this project is to plan and construct a version of "The Young Lady's Illustrated Primer" from "The Diamond Age". The output is a portable device with camera, microphone and typing inputs (keyboard or on-screen). The outputs are svisuals (pictures, images, graphs), audio (speech and music) and text.

The device should be capable of operating oofline, so it needs to host one more more models (vision recognition, STT, TTS, image generation, and LLM).

The device must have a battery with large enough capacity to last 4 hours before re-charging.

The device is personal. It is expected to be used by one student. It will accomany the student through middle and high school. The goal of the device is to keep the student curious, encourage practical experimentation, working with their hands, and to collaborate with others.

For hardware, likely choices are m5stack (https://shop.m5stack.com/) or a raspberry PI with an Qualcomm QCS8550 LLM accelerator. The PI variant is here (https://shop.m5stack.com/products/ai-8850-llm-accelerator-m-2-kit-8gb-version-ax8850). There are various modules that platforms have for cameras (with built-in vision models), and for audio in / out. The device should be able to download content from the internet into local storage to become a subject matter expert. The internet sources are from a white list. This is likely wikipedia, published text books, and peer-reviewed papers.

For software, the front end UI should allow text, camera, and audio inputs. It should be able to render text, images and audio as output. The model will auto-compact periodically (either on a schedule, or depending on usage). However, it should still be able to observe long-term trends in the student, and be able to re-visit areas where it observers the student struggling. The tone of the teacher should be neutral - it should be critical of the student's responses. Encouragement should be reserved for when the student struggles.

## Subject Matter
Subject matter content should be exposed via MCP servers so the frontend changes are minimal. Consider a curriculum of physical education. Content from approved manuals (PDFs), websites, or specific youtube channels should be exposed to the student. However, the curriculum should cover strength training (using body-weight, free weights and maybe gym equipment if available to the student), but also aerobic exercises (running, biking, swimming), and rules of team sports. Similarly, music has vast areas to cover - vocal, instrumental, and differing styles western classical, mongolian throat singing, indian carnatic music, k-pop.

## Agent Design
It is likely that the best design mirrors a human school, where there are separate instructors tasked with specific subjects. However, due to strict 8GB RAM constraints on edge hardware, the design must utilize a **Single Orchestrator Agent**. Running concurrent LLM/VLM models specialized per subject will exceed memory and thermal budgets.
Instead of passing raw continuous video or audio feeds directly into heavy multi-modal models (which drains battery instantly), utilize lightweight edge sensor models (e.g., MediaPipe for PE posture, specialized audio-pitch detectors for music). These edge sensor models emit structured JSON events to the orchestrator LLM via its respective MCP servers. 

The single orchestrator agent uses appropriate MCP tools depending on the subject matter context to maintain the persona of the specific subject instructor.

### RUNNING PYTHON
Do not use system python. Use python 3.14 in the user's home directory. For all python projects, create a venv and activate it before running any scripts. For projects that use Docker, generate the uv lockfile this way. Run uv from the user's python installation.

### RUNNING NODE
Use nvm for Node.JS projects.

### RUNNING THE FRONTEND
In the frontend directory after activating the venv, use uv to `run python .\src\main.py`

### END TO END TESTS
Use the Playwright MCP server to test new features or changes to the UI. Before making code changes, create or update the test cases in the e2e markdown files. Use one use-case per file. These changes should be reviewed and approved by a human. Tests should be comprehensive across models, UI panels, and conversation modes. 

**Hardware Input Mocking in Playwright:**
Since testing continuous hardware inputs (camera/microphone feeds) directly via virtual Playwright browsers is complex, Playwright tests must rely on a dedicated **Debug Sensor API** in the frontend. This API accepts mocked structured JSON events (e.g. injecting `{"posture": "leaning_left"}` or `"sound_detected": false`) to simulate hardware feeds rather than attempting to pass real video buffers into the E2E test.

Examples:
- In Text mode, user specifies text input "What is a Gaussian Distribution? Plot one", LLM responds with a PLOT directive which renders a graph. There is no MUSIC: and the TTS model translates the LLM response, but does not include the PLOT directive contents.
- In Text mode, user uploads an image of a cat, and asks "What is this animal"? LLM generates a text response and maybe has IMAGE: and MUSIC: directives. If there is a MUSIC directive, the TTS model doesn't run.
- In Conversation mode, user uploads an image of a cat and asks "Identify this picture, and the style it is in". The LLM  generates a text response that gets converted via TTS and automatically plays back. There is no MUSIC: directive
- **Hardware Mode:** Inject JSON `{ "posture_event": "squat_depth_shallow" }` via the Debug Sensor API. The Orchestrator LLM receives the event and generates TTS feedback: "You need to get a bit lower on your squat".

Once the e2e markdown changes have been approved, proceed with changes to the codebase. Validate the codebase changes using the approved e2e markdown files. If additional changes to the e2e markdown files are needed, make the changes, but always get them reviewed and approved by a human. Finally, after changes to the codebase are complete (and all the e2e markdown tests are passing), convert the e2e markdown tests into corresponding python tests so they can be executed without Claude. The python tests can use the gradio API directly bypassing the UI control layer for user inputs.