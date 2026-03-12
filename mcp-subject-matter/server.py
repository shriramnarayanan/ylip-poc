"""
MCP server exposing local subject-matter content to the YLIP tutor agent.

Each subject is a self-describing sqlite-vec .db file in the subjects/ directory,
downloaded by the content manager from S3.

Run:
    uv run python server.py
"""

import pathlib

from fastmcp import FastMCP
from retrieval import SubjectDB, hybrid_search

SUBJECTS_DIR = pathlib.Path(__file__).parent / "subjects"

mcp = FastMCP("ylip-subject-matter")


def _load(subject: str) -> SubjectDB:
    path = SUBJECTS_DIR / f"{subject}.db"
    if not path.exists():
        raise FileNotFoundError(
            f"No subject database '{subject}' found. "
            f"Download it via the content manager or run scripts/create_sample_db.py."
        )
    return SubjectDB(path)


@mcp.tool()
def list_subjects() -> list[dict]:
    """List all subject databases currently installed on this device."""
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


@mcp.tool()
def search(query: str, subject: str | None = None, n: int = 5) -> list[dict]:
    """
    Search curriculum content using hybrid semantic + keyword search
    (sqlite-vec ANN + FTS5, fused with Reciprocal Rank Fusion).

    If subject is omitted, searches across all installed subjects and returns
    the top n results overall.
    """
    dbs = [_load(subject)] if subject else [SubjectDB(p) for p in sorted(SUBJECTS_DIR.glob("*.db"))]

    results: list[dict] = []
    for db in dbs:
        results.extend(hybrid_search(db, query, n))

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n]


@mcp.tool()
def get_structured(subject: str, category: str | None = None) -> list[dict]:
    """
    Retrieve structured entries from a subject database: exercises, drills, music
    pieces, sport rules, etc. Optionally filter by category.
    """
    return _load(subject).structured(category)


if __name__ == "__main__":
    mcp.run()
