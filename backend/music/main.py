"""
Music generation backend — generates audio from a text prompt using MusicGen.
Exposes POST /v1/audio/generate. Runs on port 8884.

Run:
    uv run python main.py
"""

import asyncio
import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.io.wavfile
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="YLIP Music Generation Backend")

_pipe = None
_executor = ThreadPoolExecutor(max_workers=1)


def _load_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model = os.environ.get("MUSIC_MODEL", "facebook/musicgen-small")
    print(f"Loading {model} on {device}...")
    _pipe = pipeline("text-to-audio", model=model, device=device)
    print("Model loaded.")
    return _pipe


def _generate_sync(prompt: str, max_new_tokens: int) -> bytes:
    pipe = _load_pipe()
    result = pipe(prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
    sampling_rate = result["sampling_rate"]
    audio = result["audio"]  # (1, channels, samples) or (channels, samples)
    if audio.ndim == 3:
        audio = audio[0]  # (channels, samples)
    audio = np.squeeze(audio)  # (samples,) for mono
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, rate=sampling_rate, data=audio_int16)
    return buf.getvalue()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512  # ~10 seconds at musicgen-small's 32kHz / 50 tokens per second


@app.post("/v1/audio/generate")
async def generate(req: GenerateRequest):
    if not req.prompt.strip():
        return JSONResponse(status_code=400, content={"error": "prompt is empty"})
    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(
        _executor, _generate_sync, req.prompt, req.max_new_tokens
    )
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _pipe is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8884)
