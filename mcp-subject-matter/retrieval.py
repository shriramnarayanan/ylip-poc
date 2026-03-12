"""
Hybrid retrieval over a subject .db file:
  - sqlite-vec ANN for semantic search
  - FTS5 for keyword search
  - Reciprocal Rank Fusion to combine rankings
"""

import json
import pathlib
import sqlite3
from functools import lru_cache

import numpy as np
import sqlite_vec

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def _embed(text: str) -> np.ndarray:
    return _embedder().encode(text, normalize_embeddings=True).astype(np.float32)


class SubjectDB:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.subject = path.stem
        self._conn: sqlite3.Connection | None = None

    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            c = sqlite3.connect(self.path)
            c.row_factory = sqlite3.Row
            c.enable_load_extension(True)
            sqlite_vec.load(c)
            c.enable_load_extension(False)
            self._conn = c
        return self._conn

    def manifest(self) -> dict | None:
        try:
            row = self.conn().execute("SELECT * FROM _manifest LIMIT 1").fetchone()
        except sqlite3.OperationalError:
            return None
        if not row:
            return None
        d = dict(row)
        for key in ("prerequisites", "curriculum_tags", "source_urls"):
            if d.get(key):
                d[key] = json.loads(d[key])
        return d

    def topics(self) -> list[str]:
        rows = self.conn().execute(
            "SELECT DISTINCT topic FROM chunks ORDER BY topic"
        ).fetchall()
        return [r["topic"] for r in rows]

    def structured(self, category: str | None = None) -> list[dict]:
        if category:
            rows = self.conn().execute(
                "SELECT * FROM structured WHERE category = ? ORDER BY name",
                [category],
            ).fetchall()
        else:
            rows = self.conn().execute(
                "SELECT * FROM structured ORDER BY category, name"
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get("data"):
                d["data"] = json.loads(d["data"])
            result.append(d)
        return result


def hybrid_search(db: SubjectDB, query: str, n: int = 5) -> list[dict]:
    """
    Hybrid FTS5 + sqlite-vec search, fused with Reciprocal Rank Fusion (k=60).
    Returns up to n chunks with subject, topic, heading, content, score.
    """
    c = db.conn()
    k = 60

    # FTS5 keyword search
    fts_ids: list[int] = []
    fts_rows: dict[int, dict] = {}
    try:
        rows = c.execute(
            """
            SELECT c.id, c.topic, c.heading, c.content
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            [query, n * 3],
        ).fetchall()
        for r in rows:
            fts_ids.append(r["id"])
            fts_rows[r["id"]] = dict(r)
    except sqlite3.OperationalError:
        pass

    # sqlite-vec semantic search
    vec_ids: list[int] = []
    vec_rows: dict[int, dict] = {}
    try:
        query_vec = _embed(query)
        rows = c.execute(
            """
            SELECT ve.rowid, ve.distance, c.topic, c.heading, c.content
            FROM chunk_embeddings ve
            JOIN chunks c ON c.id = ve.rowid
            WHERE ve.embedding MATCH ? AND k = ?
            ORDER BY ve.distance
            """,
            [sqlite_vec.serialize_float32(query_vec), n * 3],
        ).fetchall()
        for r in rows:
            vec_ids.append(r["rowid"])
            vec_rows[r["rowid"]] = dict(r)
    except sqlite3.OperationalError:
        pass

    # RRF fusion
    scores: dict[int, float] = {}
    for rank, cid in enumerate(fts_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(vec_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    all_chunks = {**fts_rows, **vec_rows}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    return [
        {
            "subject": db.subject,
            "topic": all_chunks[cid]["topic"],
            "heading": all_chunks[cid]["heading"],
            "content": all_chunks[cid]["content"],
            "score": round(score, 4),
        }
        for cid, score in ranked
        if cid in all_chunks
    ]
