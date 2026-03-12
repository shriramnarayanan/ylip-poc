"""
Creates a sample physical-education subject database for development and testing.

Usage (from mcp-subject-matter/ directory):
    uv run python scripts/create_sample_db.py

Output: subjects/physical_education.db
"""

import json
import pathlib
import sqlite3

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

SUBJECTS_DIR = pathlib.Path(__file__).parent.parent / "subjects"
DB_PATH = SUBJECTS_DIR / "physical_education.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MANIFEST = {
    "subject": "physical_education",
    "topic_area": "strength_training/bodyweight",
    "title": "Beginner Bodyweight Strength",
    "difficulty": "beginner",
    "prerequisites": [],
    "curriculum_tags": ["squat", "push-up", "hip-hinge", "core", "bodyweight"],
    "source_urls": [],
    "embedding_model": EMBEDDING_MODEL,
    "version": "0.1.0",
    "ingested_at": "2026-03-11T00:00:00Z",
}

CHUNKS = [
    {
        "topic": "squat",
        "heading": "Air Squat — Technique",
        "content": (
            "Stand with feet shoulder-width apart, toes slightly turned out. "
            "Push your hips back and bend your knees, keeping your chest up and "
            "knees tracking over your toes. Lower until thighs are parallel to "
            "the ground or below. Drive through your heels to stand. Brace your "
            "core throughout."
        ),
    },
    {
        "topic": "squat",
        "heading": "Air Squat — Common Faults",
        "content": (
            "Knees caving inward (valgus collapse): cue to push knees out. "
            "Heels rising: work ankle mobility or elevate heels temporarily. "
            "Forward lean: strengthen upper back and improve thoracic mobility. "
            "Shallow depth: squat to a box until full range of motion is developed."
        ),
    },
    {
        "topic": "push-up",
        "heading": "Push-Up — Technique",
        "content": (
            "Start in a high plank: hands slightly wider than shoulder-width, "
            "body in a straight line from head to heels. Lower chest to the floor, "
            "elbows at roughly 45 degrees from your torso. Press back to start. "
            "Keep hips level — do not let them sag or pike."
        ),
    },
    {
        "topic": "push-up",
        "heading": "Push-Up — Progressions",
        "content": (
            "Begin with incline push-ups (hands on a raised surface) if full "
            "push-ups are too difficult. Progress to knee push-ups, then full "
            "push-ups. Advance to close-grip, archer, and eventually one-arm "
            "push-ups."
        ),
    },
    {
        "topic": "hip-hinge",
        "heading": "Hip Hinge — Technique",
        "content": (
            "Stand tall, feet hip-width apart. Push hips back as if closing a "
            "car door with your glutes, maintaining a neutral spine. Allow a "
            "slight bend in the knees. Feel a stretch in the hamstrings. Return "
            "by driving hips forward. The hip hinge is the foundation of "
            "deadlifts and Romanian deadlifts."
        ),
    },
    {
        "topic": "core",
        "heading": "Plank — Technique",
        "content": (
            "Support your body on forearms and toes, elbows under shoulders. "
            "Keep your body in a straight line from head to heels. Brace your "
            "core as if bracing for a punch. Breathe steadily. Avoid letting "
            "hips sag or pike. Hold for time."
        ),
    },
]

STRUCTURED = [
    {
        "category": "exercise",
        "name": "Air Squat",
        "data": {
            "muscle_groups": ["quadriceps", "glutes", "hamstrings"],
            "equipment": "none",
            "sets_reps": "3×10–15",
            "difficulty": "beginner",
            "cues": ["chest up", "knees out", "drive through heels"],
        },
    },
    {
        "category": "exercise",
        "name": "Push-Up",
        "data": {
            "muscle_groups": ["pectorals", "triceps", "anterior deltoid"],
            "equipment": "none",
            "sets_reps": "3×8–12",
            "difficulty": "beginner",
            "cues": ["elbows 45 degrees", "body straight", "full range"],
        },
    },
    {
        "category": "exercise",
        "name": "Plank",
        "data": {
            "muscle_groups": ["core", "transverse abdominis", "glutes"],
            "equipment": "none",
            "sets_reps": "3×20–60s",
            "difficulty": "beginner",
            "cues": ["brace core", "hips level", "breathe"],
        },
    },
    {
        "category": "workout",
        "name": "Beginner Full-Body A",
        "data": {
            "exercises": ["Air Squat", "Push-Up", "Plank"],
            "rest_between_sets": "60–90s",
            "frequency": "3x per week",
            "notes": "Focus on technique before adding volume.",
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
            embedding_model TEXT, chunk_count INTEGER, version TEXT, ingested_at TEXT
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
        "INSERT INTO _manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            MANIFEST["ingested_at"],
        ],
    )

    conn.commit()
    conn.close()
    print(f"Created {DB_PATH} ({len(CHUNKS)} chunks, {len(STRUCTURED)} structured entries)")


if __name__ == "__main__":
    create_db()
