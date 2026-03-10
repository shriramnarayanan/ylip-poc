"""
Image generation backend — exposes /v1/images/generations (OpenAI-compatible).
Uses SDXL-Turbo by default (1-4 step inference, fast). Runs on port 8882.

Run:
    uv run python main.py
"""

import asyncio
import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from huggingface_hub import login as hf_login

if token := os.environ.get("HF_TOKEN"):
    hf_login(token=token, add_to_git_credential=False)
from diffusers import AutoPipelineForText2Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="YLIP Image Generation Backend")

_pipe = None
_executor = ThreadPoolExecutor(max_workers=1)

MODEL_ID = "stabilityai/sdxl-turbo"


def _load_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL_ID} on {device}...")
    _pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    ).to(device)
    print("Model loaded.")
    return _pipe


def _generate_sync(prompt: str, steps: int) -> bytes:
    pipe = _load_pipe()
    kwargs = {"prompt": prompt, "num_inference_steps": steps}
    if "turbo" in MODEL_ID.lower():
        kwargs["guidance_scale"] = 0.0
    image = pipe(**kwargs).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class ImageRequest(BaseModel):
    model: str = "sdxl-turbo"
    prompt: str
    n: int = 1
    size: str = "512x512"
    response_format: str = "b64_json"
    num_inference_steps: int = 4


@app.post("/v1/images/generations")
async def generate(req: ImageRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is empty")
    loop = asyncio.get_event_loop()
    img_bytes = await loop.run_in_executor(
        _executor, _generate_sync, req.prompt, req.num_inference_steps
    )
    b64 = base64.b64encode(img_bytes).decode()
    return JSONResponse({"created": 0, "data": [{"b64_json": b64}]})


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _pipe is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8882)
