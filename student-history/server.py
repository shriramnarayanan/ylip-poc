"""
Student State Tracking Service — YLIP
======================================
Runs as a standalone HTTP service on port 8771.

Two-layer memory
----------------
  Episodic:  Raw interaction log (topic, signal, approach, notes per turn).
  Semantic:  Per-concept mastery scores that persist and decay across sessions.

Mastery model (BKT-inspired)
-----------------------------
Each concept has a score in [0, 1].  After each rated turn it is updated via
an exponential moving average toward the normalised mastery signal (0–4 → 0.0–1.0):

    score += lr * (signal/4 - score)          # EMA update, lr = 0.15

Between sessions an Ebbinghaus-inspired forgetting curve decays the score.
Stability scales with current mastery so well-learned concepts are forgotten slowly:

    stability = 7 + 14 * score                # 7–21 day range
    score = score * exp(-days/stability) + 0.5 * (1 - exp(-days/stability))

Run
---
    uv sync
    uv run uvicorn server:app --port 8771 --reload
"""

import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "student.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT    NOT NULL,
    timestamp      TEXT    NOT NULL,
    topic          TEXT,
    mastery_signal INTEGER,        -- 0–4 (understanding level); -1 = student only asked
    approach       TEXT,           -- answered | questioned | struggled | demonstrated
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS mastery (
    concept    TEXT    PRIMARY KEY,
    score      REAL    NOT NULL DEFAULT 0.5,  -- 0 = none, 1 = fully mastered
    attempts   INTEGER NOT NULL DEFAULT 0,
    streak     INTEGER NOT NULL DEFAULT 0,    -- +N correct, -N confused streak
    last_seen  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    start_time  TEXT NOT NULL,
    end_time    TEXT
);

CREATE INDEX IF NOT EXISTS ix_int_session ON interactions(session_id);
CREATE INDEX IF NOT EXISTS ix_int_topic   ON interactions(topic);
CREATE INDEX IF NOT EXISTS ix_int_ts      ON interactions(timestamp);
"""

_LEARNING_RATE = 0.15


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_db() -> None:
    with _db() as conn:
        conn.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# Mastery update
# ---------------------------------------------------------------------------

def _update_mastery(conn: sqlite3.Connection, concept: str, signal: int) -> None:
    """Apply one observation to a concept's mastery score.

    Applies forgetting since last observation, then an EMA update, then
    updates the streak counter.
    """
    now = datetime.now(timezone.utc)
    row = conn.execute(
        "SELECT score, attempts, streak, last_seen FROM mastery WHERE concept = ?",
        (concept,),
    ).fetchone()

    if row is None:
        score, attempts, streak = 0.5, 0, 0
    else:
        score    = row["score"]
        attempts = row["attempts"]
        streak   = row["streak"]
        last     = datetime.fromisoformat(row["last_seen"])
        days_ago = max(0.0, (now - last).total_seconds() / 86_400)

        if days_ago > 0:
            # Ebbinghaus: stability scales with mastery (7–21 day half-life)
            stability = 7.0 + 14.0 * score
            retention = math.exp(-days_ago / stability)
            # Drift toward 0.5 (uncertain) as retention fades
            score = score * retention + 0.5 * (1.0 - retention)

    # EMA toward normalised signal
    score = score + _LEARNING_RATE * (signal / 4.0 - score)
    score = max(0.0, min(1.0, score))

    # Streak: +1 for strong signal (≥3), -1 for weak (≤1), reset otherwise
    if signal >= 3:
        streak = max(0, streak) + 1
    elif signal <= 1:
        streak = min(0, streak) - 1
    else:
        streak = 0

    conn.execute(
        """INSERT INTO mastery (concept, score, attempts, streak, last_seen)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(concept) DO UPDATE SET
               score = excluded.score, attempts = excluded.attempts,
               streak = excluded.streak, last_seen = excluded.last_seen""",
        (concept, score, attempts + 1, streak, now.isoformat()),
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(title="YLIP Student State", version="0.1.0")


class InteractionIn(BaseModel):
    session_id: str
    topic:          str | None = None
    mastery_signal: int | None = Field(None, ge=-1, le=4)
    approach:       str | None = None  # answered|questioned|struggled|demonstrated
    notes:          str | None = None


@app.post("/record", status_code=204)
async def record(body: InteractionIn) -> None:
    """Persist one rated interaction turn and update mastery scores."""
    now   = datetime.now(timezone.utc).isoformat()
    topic = (body.topic or "").strip() or None

    with _db() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, start_time) VALUES (?, ?)"
            " ON CONFLICT(session_id) DO NOTHING",
            (body.session_id, now),
        )
        conn.execute(
            """INSERT INTO interactions
               (session_id, timestamp, topic, mastery_signal, approach, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (body.session_id, now, topic, body.mastery_signal,
             body.approach, (body.notes or "")[:200]),
        )
        # Update session end_time
        conn.execute(
            "UPDATE sessions SET end_time = ? WHERE session_id = ?",
            (now, body.session_id),
        )
        # Update mastery for meaningful signals (0–4, not -1)
        if topic and body.mastery_signal is not None and body.mastery_signal >= 0:
            _update_mastery(conn, topic, body.mastery_signal)


@app.get("/context")
async def get_context() -> dict:
    """Compact student-state string (~150 tokens) for system-prompt injection.

    The tutor injects this before each LLM call so it can personalise
    its teaching approach without an extra tool-call round trip.
    """
    with _db() as conn:
        m_rows = conn.execute(
            "SELECT concept, score, streak, attempts FROM mastery ORDER BY score ASC"
        ).fetchall()
        n_interactions = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE mastery_signal IS NOT NULL"
        ).fetchone()[0]
        recent = conn.execute(
            """SELECT topic, mastery_signal, notes FROM interactions
               WHERE notes != '' AND mastery_signal IS NOT NULL
               ORDER BY timestamp DESC LIMIT 4"""
        ).fetchall()

    if not m_rows and n_interactions == 0:
        return {"context": "", "has_data": False}

    lines = []
    n = len(m_rows)
    overall = sum(r["score"] for r in m_rows) / n if n else 0.0
    lines.append(
        f"STUDENT STATE ({n} concept(s), {n_interactions} rated turn(s), "
        f"overall {overall:.0%}):"
    )

    # Struggling: score < 0.45
    struggling = [(r["concept"], r["score"], r["streak"]) for r in m_rows if r["score"] < 0.45]
    if struggling:
        parts = []
        for concept, score, streak in struggling[:4]:
            suffix = f", {abs(streak)}-loss streak" if streak <= -2 else ""
            parts.append(f"{concept} ({score:.0%}{suffix})")
        lines.append("  Needs attention: " + ", ".join(parts))

    # Strong: score > 0.70
    strong = [(r["concept"], r["score"]) for r in reversed(m_rows) if r["score"] > 0.70]
    if strong:
        lines.append("  Strong: " + ", ".join(f"{c} ({s:.0%})" for c, s in strong[:3]))

    # Recent observations
    _LABELS = {0: "confused", 1: "misconception", 2: "partial", 3: "developing", 4: "mastered"}
    if recent:
        obs = []
        for r in recent[:3]:
            label = _LABELS.get(r["mastery_signal"], "")
            note  = (r["notes"] or "").strip()
            obs.append(f"{note} [{label}]" if note and label else note or label)
        lines.append("  Recent: " + "; ".join(filter(None, obs)))

    return {"context": "\n".join(lines), "has_data": True}


@app.get("/mastery")
async def get_mastery() -> dict:
    """Full mastery table — for admin dashboards and debugging."""
    with _db() as conn:
        rows = conn.execute("SELECT * FROM mastery ORDER BY score DESC").fetchall()
    return {"mastery": [dict(r) for r in rows]}


@app.get("/history")
async def get_history(session_id: str | None = None, limit: int = 20) -> dict:
    """Recent interaction log, optionally filtered by session."""
    with _db() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE session_id = ?"
                " ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
    return {"interactions": [dict(r) for r in rows]}


@app.delete("/reset", status_code=204)
async def reset() -> None:
    """Wipe all student data. Development / testing only."""
    with _db() as conn:
        conn.executescript("DELETE FROM interactions; DELETE FROM mastery; DELETE FROM sessions;")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
_init_db()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8771)
