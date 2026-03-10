"""
TTS backend server — exposes /v1/audio/speech (OpenAI-compatible).
Uses Kokoro for synthesis. Runs as a standalone process on port 8880.

Install:
    pip install kokoro soundfile numpy fastapi uvicorn

Run:
    python main.py
"""

import io
import struct
import wave

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="YLIP TTS Backend")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code="a")  # "a" = American English
    return _pipeline


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    response_format: str = "wav"
    speed: float = 1.0


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input is empty")

    pipeline = get_pipeline()

    audio_chunks = []
    sample_rate = 24000  # Kokoro default

    for _, _, audio in pipeline(req.input, voice=req.voice, speed=req.speed):
        if audio is not None:
            audio_chunks.append(audio)

    if not audio_chunks:
        raise HTTPException(status_code=500, detail="TTS produced no audio")

    combined = np.concatenate(audio_chunks)

    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV")
    wav_bytes = buf.getvalue()

    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8880)
