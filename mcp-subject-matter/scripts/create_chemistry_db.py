"""
Creates a sample chemistry subject database for development and testing.

Usage (from mcp-subject-matter/ directory):
    uv run python scripts/create_chemistry_db.py

Output: subjects/chemistry.db
"""

import json
import pathlib
import sqlite3

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

SUBJECTS_DIR = pathlib.Path(__file__).parent.parent / "subjects"
DB_PATH = SUBJECTS_DIR / "chemistry.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MANIFEST = {
    "subject": "chemistry",
    "topic_area": "elements_and_reactions",
    "title": "Introductory Chemistry",
    "difficulty": "beginner",
    "prerequisites": ["basic arithmetic"],
    "curriculum_tags": ["periodic table", "reactions", "atoms"],
    "source_urls": [],
    "embedding_model": EMBEDDING_MODEL,
    "version": "0.1.0",
    "ingested_at": "2026-03-11T00:00:00Z",
}

CHUNKS = [
    {
        "topic": "atomic_structure",
        "heading": "Protons, Neutrons, and Electrons",
        "content": (
            "Atoms consist of three basic particles: protons, neutrons, and electrons. "
            "The nucleus contains protons (positive charge) and neutrons (no charge). "
            "Electrons (negative charge) orbit the nucleus in distinct energy levels "
            "or shells."
        ),
    },
    {
        "topic": "periodic_table",
        "heading": "Groups and Periods",
        "content": (
            "The periodic table organizes elements into columns called groups and rows "
            "called periods. Elements in the same group have the same number of valence "
            "electrons and share similar chemical properties. For example, Group 1 "
            "elements are highly reactive alkali metals."
        ),
    },
    {
        "topic": "reactions",
        "heading": "Covalent vs Ionic Bonds",
        "content": (
            "Ionic bonds form when one atom completely transfers electrons to another, "
            "creating oppositely charged ions that attract (e.g., NaCl). Covalent bonds "
            "form when two atoms share electrons to achieve stable outer electron shells "
            "(e.g., H2O). Metals and nonmetals usually form ionic bonds, while nonmetals "
            "bond covalently."
        ),
    },
    {
        "topic": "reactions",
        "heading": "Balancing Equations",
        "content": (
            "The law of conservation of mass dictates that matter cannot be created or "
            "destroyed in a chemical reaction. Therefore, a chemical equation must be "
            "balanced: the number of atoms of each element on the reactant side must "
            "equal the number on the product side."
        ),
    },
]

STRUCTURED = [
    {
        "category": "element",
        "name": "Oxygen (O)",
        "data": {
            "atomic_number": 8,
            "group": "Chalcogen",
            "state_at_stp": "Gas",
            "electronegativity": 3.44,
            "discovery": "1774",
        },
    },
    {
        "category": "rule",
        "name": "Octet Rule",
        "data": {
            "description": "Atoms tend to form bonds until they are surrounded by eight valence electrons.",
            "exceptions": ["Hydrogen", "Helium", "Boron"],
            "difficulty": "beginner",
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
