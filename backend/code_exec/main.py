"""
Code execution backend for mathematical plots.
Executes Python (matplotlib/numpy) and returns the resulting figure as PNG.
Runs on port 8883.

Run:
    uv run python main.py
"""

import base64
import builtins
import io
import re
import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

matplotlib.use("Agg")  # non-interactive, no display needed

app = FastAPI(title="YLIP Code Execution Backend")

# Allow scipy, math, cmath, statistics — block everything else
_ALLOWED_IMPORTS = ("scipy", "math", "cmath", "statistics")


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == a or name.startswith(a + ".") for a in _ALLOWED_IMPORTS):
        return builtins.__import__(name, globals or {}, locals or {}, fromlist, level)
    raise ImportError(
        f"'{name}' is not available. Use np, plt, scipy, or math."
    )


EXEC_GLOBALS = {
    "np": np,
    "plt": plt,
    "__builtins__": {
        "print": print,
        "range": range, "len": len,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "int": int, "float": float, "str": str, "bool": bool,
        "abs": abs, "min": min, "max": max, "sum": sum, "round": round, "pow": pow,
        "zip": zip, "enumerate": enumerate, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed,
        "__import__": _safe_import,
    },
}


class ExecRequest(BaseModel):
    code: str


# Leave scipy, math, etc. intact so they can be imported normally.
_IMPORT_RE = re.compile(
    r"^\s*(import\s+(numpy|matplotlib)[^\n]*|from\s+(numpy|matplotlib)\S*\s+import[^\n]*)\n?",
    re.MULTILINE,
)

@app.post("/v1/execute")
async def execute(req: ExecRequest):
    code = req.code.strip()
    
    # Robustly strip markdown blocks if the LLM hallucinates them
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    
    code = _IMPORT_RE.sub("", code).strip()
    print("--- EXECUTING PLOT CODE ---")
    print(code)
    print("---------------------------")
    
    plt.close("all")
    try:
        exec(code, dict(EXEC_GLOBALS))
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )

    fig = plt.gcf()
    if not fig.axes:
        return JSONResponse(status_code=400, content={"error": "No figure was produced by the code."})

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close("all")

    return JSONResponse({"image": base64.b64encode(buf.getvalue()).decode()})


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8883)
