"""
STT backend server — exposes /v1/audio/transcriptions (OpenAI-compatible).
Uses faster-whisper for transcription. Runs as a standalone process on port 8881.

Run:
    uv run python main.py
"""

import io
import tempfile
import os

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

app = FastAPI(title="YLIP STT Backend")

_model = None


def get_model(model_size: str = "base") -> WhisperModel:
    global _model
    if _model is None:
        device = os.environ.get("DEVICE", "auto")
        compute_type = os.environ.get("COMPUTE_TYPE", "auto")
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # faster-whisper needs a file path, not bytes
    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        whisper = get_model()
        kwargs = {"beam_size": 5}
        if language:
            kwargs["language"] = language
        segments, info = whisper.transcribe(tmp_path, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments)
    finally:
        os.unlink(tmp_path)

    return JSONResponse({"text": text})


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8881)
