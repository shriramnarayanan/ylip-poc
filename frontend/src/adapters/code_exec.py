import base64

import httpx

from config import settings


class CodeExecAdapter:
    """
    Sends Python plot code to the code execution backend and returns PNG bytes.
    """

    async def execute(self, code: str) -> bytes:
        async with httpx.AsyncClient(base_url=settings.code_exec_base_url, timeout=30) as client:
            resp = await client.post("/execute", json={"code": code})
            if resp.status_code == 400:
                raise RuntimeError(f"code_exec error: {resp.json()}")
            resp.raise_for_status()
            return base64.b64decode(resp.json()["image"])
