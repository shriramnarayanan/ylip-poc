"""
Creates a sample mathematics subject database for development and testing.

Usage (from mcp-subject-matter/ directory):
    uv run python scripts/create_math_db.py

Output: subjects/mathematics.db
"""

import json
import pathlib
import sqlite3

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

SUBJECTS_DIR = pathlib.Path(__file__).parent.parent / "subjects"
DB_PATH = SUBJECTS_DIR / "mathematics.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MANIFEST = {
    "subject": "mathematics",
    "topic_area": "algebra_and_calculus",
    "title": "Core High School Mathematics",
    "difficulty": "intermediate",
    "prerequisites": ["basic arithmetic", "pre-algebra"],
    "curriculum_tags": ["algebra", "calculus", "geometry", "trigonometry"],
    "source_urls": [],
    "embedding_model": EMBEDDING_MODEL,
    "version": "0.1.0",
    "ingested_at": "2026-03-11T00:00:00Z",
}

CHUNKS = [
    {
        "topic": "algebra",
        "heading": "Quadratic Formula",
        "content": (
            "The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a. It is used to "
            "find the roots of a quadratic equation in the form ax² + bx + c = 0. "
            "The term (b² - 4ac) is called the discriminant. If it is positive, "
            "there are two real roots. If zero, one real root. If negative, "
            "two complex roots."
        ),
    },
    {
        "topic": "calculus",
        "heading": "The Derivative",
        "content": (
            "A derivative represents the instantaneous rate of change of a function "
            "with respect to one of its variables. Geometrically, it is the slope "
            "of the tangent line to the curve at a specific point. The power rule "
            "states that the derivative of x^n is n*x^(n-1)."
        ),
    },
    {
        "topic": "calculus",
        "heading": "The Integral",
        "content": (
            "Integration is the reverse process of differentiation. The definite "
            "integral of a function represents the signed area between the curve "
            "and the x-axis over a specified interval. The Fundamental Theorem of "
            "Calculus connects differentiation and integration."
        ),
    },
    {
        "topic": "trigonometry",
        "heading": "Pythagorean Identity",
        "content": (
            "The most fundamental trigonometric identity is sin²(θ) + cos²(θ) = 1. "
            "It is derived directly from the Pythagorean theorem applied to the "
            "unit circle, where the hypotenuse is 1, the opposite side is sin(θ), "
            "and the adjacent side is cos(θ)."
        ),
    },
]

STRUCTURED = [
    {
        "category": "rule",
        "name": "Power Rule (Differentiation)",
        "data": {
            "equation": "d/dx [x^n] = n * x^(n-1)",
            "prerequisites": ["limits"],
            "difficulty": "beginner",
            "cues": ["Bring the exponent down", "Subtract one from the exponent"],
        },
    },
    {
        "category": "formula",
        "name": "Quadratic Formula",
        "data": {
            "equation": "x = (-b ± √(b² - 4ac)) / 2a",
            "prerequisites": ["factoring", "completing the square"],
            "difficulty": "beginner",
            "notes": "Used when a quadratic cannot be easily factored.",
        },
    },
]


def create_db() -> None:
    SUBJECTS_DIR.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript("""
        CREATE TABLE _manifest (
            subject TEXT, topic_area TEXT, title TEXT, difficulty TEXT,
            prerequisites TEXT, curriculum_tags TEXT, source_urls TEXT,
            embedding_model TEXT, chunk_count INTEGER, version TEXT, content_hash TEXT, ingested_at TEXT
        );

        CREATE TABLE chunks (
            id      INTEGER PRIMARY KEY,
            topic   TEXT NOT NULL,
            heading TEXT,
            content TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            topic, heading, content,
            content='chunks',
            content_rowid='id'
        );

        CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
            embedding FLOAT[384]
        );

        CREATE TABLE structured (
            id       INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            name     TEXT NOT NULL,
            data     TEXT
        );
    """)

    print(f"Loading {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding and inserting chunks...")
    for chunk in CHUNKS:
        text = f"{chunk['heading']}. {chunk['content']}"
        embedding = model.encode(text, normalize_embeddings=True).astype(np.float32)
        cursor = conn.execute(
            "INSERT INTO chunks (topic, heading, content) VALUES (?, ?, ?)",
            [chunk["topic"], chunk["heading"], chunk["content"]],
        )
        conn.execute(
            "INSERT INTO chunk_embeddings(rowid, embedding) VALUES (?, ?)",
            [cursor.lastrowid, sqlite_vec.serialize_float32(embedding)],
        )

    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")

    for entry in STRUCTURED:
        conn.execute(
            "INSERT INTO structured (category, name, data) VALUES (?, ?, ?)",
            [entry["category"], entry["name"], json.dumps(entry["data"])],
        )

    conn.execute(
        "INSERT INTO _manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            MANIFEST["subject"],
            MANIFEST["topic_area"],
            MANIFEST["title"],
            MANIFEST["difficulty"],
            json.dumps(MANIFEST["prerequisites"]),
            json.dumps(MANIFEST["curriculum_tags"]),
            json.dumps(MANIFEST["source_urls"]),
            MANIFEST["embedding_model"],
            len(CHUNKS),
            MANIFEST["version"],
            "not_hashed_locally",
            MANIFEST["ingested_at"],
        ],
    )

    conn.commit()
    conn.close()
    print(f"Created {DB_PATH} ({len(CHUNKS)} chunks, {len(STRUCTURED)} structured entries)")


if __name__ == "__main__":
    create_db()
