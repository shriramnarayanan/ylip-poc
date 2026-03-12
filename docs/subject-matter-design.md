# Subject Matter Architecture Design

## Context

YLIP is a personal AI tutor device for middle/high school students. Subject matter content (PE, music, maths, etc.) is delivered to the device as sqlite-vec `.db` files downloaded from S3. A remote MCP server handles content discovery and curriculum-aware recommendations based on student progress.

---

## System Overview

```
OFF-DEVICE
  data-ingest/  scrape → chunk → embed → package
      ↓ publishes
  S3 bucket  (shared, all devices)
    /subjects/pe/bodyweight-beginner-v1.2.db
    /catalogue.json   ← index of all manifests
      ↓ read by
  mcp-content-server/  (hosted, well-known URL)
    list_available(subject?, tags?)
    recommend_next(student_id, progress_summary)
        → returns manifest + pre-signed S3 URLs
      ↕ HTTPS (MCP calls + direct S3 downloads)
DEVICE
  content-manager/   (scheduled + on-demand)
    calls recommend_next, downloads .db files, evicts stale
  student-history/
    local SQLite: topics_covered, struggle_areas, session timestamps
    summarised for recommend_next calls
  mcp-subject-matter/   (local, used during tutoring sessions)
    search(query, subject?)
    list_topics(subject)
    get_structured(subject, category?)
    list_subjects()
  frontend/   (unchanged — calls local MCP during sessions)
```

---

## Design Decisions

- **One device per student.** No multi-device sync.
- **Content shared across all students.** Same `.db` files, same curriculum graph. Only the recommendation of what to do *next* is personalised.
- **Student progress pushed inline.** Device sends a progress summary as part of the `recommend_next` call. Remote server is stateless — no persistent student records.
- **Remote MCP server is stateless per call.** Consults curriculum graph + progress summary, returns recommendations. Simple, privacy-preserving.
- **S3 URLs flow through the MCP server.** `recommend_next` returns pre-signed S3 URLs; device downloads `.db` files directly from S3. MCP server never proxies content.
- **Remote server is MCP (not plain HTTP)** so the on-device LLM agent can call `recommend_next` directly during a tutoring session ("I want to learn something new today") as well as from the scheduled content manager.
- **Embedding at ingest time only.** Vectors are computed off-device during ingest and stored in the `.db` file. On-device, only the query needs embedding (fast, small model).
- **Embedding model:** `all-MiniLM-L6-v2` (80 MB, 384-dim, runs on CPU in ~5 ms per query).
- **Hybrid retrieval:** sqlite-vec ANN search + FTS5 keyword search, fused with Reciprocal Rank Fusion (k=60).

---

## Self-Describing .db Schema

```sql
CREATE TABLE _manifest (
    subject         TEXT,
    topic_area      TEXT,
    title           TEXT,
    difficulty      TEXT,       -- beginner | intermediate | advanced
    prerequisites   TEXT,       -- JSON array of topic_area strings
    curriculum_tags TEXT,       -- JSON array
    source_urls     TEXT,       -- JSON array
    embedding_model TEXT,       -- e.g. all-MiniLM-L6-v2
    chunk_count     INTEGER,
    version         TEXT,       -- semver
    ingested_at     TEXT        -- ISO 8601
);

CREATE TABLE chunks (
    id      INTEGER PRIMARY KEY,
    topic   TEXT NOT NULL,
    heading TEXT,
    content TEXT NOT NULL
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    topic, heading, content,
    content='chunks', content_rowid='id'
);

CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
    embedding FLOAT[384]   -- rowid matches chunks.id
);

CREATE TABLE structured (
    id       INTEGER PRIMARY KEY,
    category TEXT NOT NULL,  -- exercise | drill | rule | piece
    name     TEXT NOT NULL,
    data     TEXT            -- JSON
);
```

---

## recommend_next Call Shape

```json
// device → remote MCP server
{
  "student_id": "uuid-v4",
  "progress_summary": {
    "topics_covered":  ["pe/bodyweight/squat", "pe/bodyweight/push-up"],
    "struggle_areas":  ["pe/bodyweight/hip-hinge"],
    "last_active":     "2026-03-11",
    "level":           "beginner"
  }
}

// remote MCP server → device
[
  {
    "db_file":    "pe/bodyweight-intermediate-v1.2.db",
    "s3_url":     "https://s3.../...",
    "expires_at": "2026-03-11T18:00:00Z",
    "manifest":   { "title": "...", "prerequisites": [...], "tags": [...] }
  }
]
```

---

## Repo Layout

```
ylip-poc/
├── data-ingest/           # off-device: scrape → chunk → embed → publish .db files
├── mcp-content-server/    # remote: catalogue + curriculum graph + recommend_next
├── mcp-subject-matter/    # on-device: local MCP reading downloaded .db files  ← built now
├── content-manager/       # on-device: scheduled download/eviction (future)
├── student-history/       # on-device: progress tracking (future)
└── frontend/              # unchanged
```

---

## Build Order

| Component | Blocks | Status |
|---|---|---|
| `mcp-subject-matter/` | Frontend tutoring sessions | **Building now** |
| `mcp-content-server/` | Content download | Next |
| `data-ingest/` | Content production | Can stub with hand-crafted .db |
| `content-manager/` | Automated downloads | After both MCP servers |
| `student-history/` | Personalised recommendations | Alongside content-manager |
