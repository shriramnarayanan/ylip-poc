"""
MCP server exposing local subject-matter content to the YLIP tutor agent.

Each subject is a self-describing sqlite-vec .db file in the subjects/ directory,
downloaded by the content manager from S3.

Run:
    uv run python server.py
"""

import pathlib
import numpy as np

from fastmcp import FastMCP
from retrieval import SubjectDB, hybrid_search, _embed, _embedder

SUBJECTS_DIR = pathlib.Path(__file__).parent / "subjects"

mcp = FastMCP("ylip-subject-matter")


def _load(subject: str) -> SubjectDB:
    path = (SUBJECTS_DIR / f"{subject}.db").resolve()
    base_dir = SUBJECTS_DIR.resolve()
    if not path.is_relative_to(base_dir) or path.parent != base_dir:
        raise ValueError(f"Invalid subject name: {subject}")

    if not path.exists():
        available = [p.stem for p in sorted(SUBJECTS_DIR.glob("*.db"))]
        hint = f"Available subjects: {available}" if available else "No subjects are currently installed."
        raise FileNotFoundError(
            f"No subject database '{subject}' found. {hint} "
            f"Call list_subjects() to discover installed subjects."
        )
    return SubjectDB(path)


@mcp.tool()
def list_subjects() -> list[dict]:
    """List available curriculum subject databases.
    Available subjects change over time."""
    results = []
    for db_path in sorted(SUBJECTS_DIR.glob("*.db")):
        db = SubjectDB(db_path)
        manifest = db.manifest()
        if manifest is not None:
            results.append({"subject": db.subject, **manifest})
    return results


@mcp.tool()
def list_topics(subject: str) -> list[str]:
    """List all topic areas available in a subject database."""
    return _load(subject).topics()


MIN_RELEVANCE = 0.45  # conservative threshold — only clearly on-topic results pass


@mcp.tool()
def search(query: str, subject: str | None = None, n: int = 10) -> list[dict]:
    """Search curriculum content using hybrid semantic + keyword search.
    Each result includes a relevance score (0.0–1.0). Only results above a
    minimum similarity threshold are returned. Use your judgment: if the content
    is directly relevant to the student's question, use it; otherwise ignore it
    and answer from your own knowledge.
    If subject is omitted, searches across all installed subjects."""
    dbs = (
        [_load(subject)] if subject
        else [SubjectDB(p) for p in sorted(SUBJECTS_DIR.glob("*.db"))]
    )

    query_vec = _embed(query)

    # Gather more candidates than n so thresholding has a wider pool
    candidates: list[dict] = []
    for db in dbs:
        candidates.extend(hybrid_search(db, query, n * 3, query_vec=query_vec))

    if not candidates:
        return []

    # Compute cosine similarity for every candidate
    texts = [f"{r.get('heading') or ''} {r['content']}" for r in candidates]
    cand_vecs = _embedder().encode(texts, normalize_embeddings=True).astype(np.float32)
    similarities = np.dot(cand_vecs, query_vec)

    results = []
    for i, r in enumerate(candidates):
        sim = float(similarities[i])
        if sim < MIN_RELEVANCE:
            continue
        r["relevance"] = round(sim, 4)
        r.pop("score", None)
        r.pop("global_score", None)
        results.append(r)

    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:n]


@mcp.tool()
def get_structured(subject: str, category: str | None = None) -> list[dict]:
    """Retrieve structured entries from a subject database: exercises, drills, music
    pieces, sport rules, etc. Optionally filter by category.
    The subject must be an exact name from list_subjects() — call that first if unsure."""
    return _load(subject).structured(category)


if __name__ == "__main__":
    import os
    transport = os.getenv("TRANSPORT", "stdio")
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8770"))
    
    if transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run(transport="stdio")
