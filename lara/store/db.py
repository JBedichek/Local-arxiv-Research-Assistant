"""SQLite store. WAL mode, one file, opened per-thread.

Schema note: ``papers`` holds *every* record harvested from the ``cs`` and ``stat`` OAI
sets, not just the in-scope ones. Metadata for ~1.1 M papers is ~2 GB, which is nothing
next to the vector index, and keeping it means widening the corpus (D1 Core -> Extended)
later is a re-flag rather than a re-harvest.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id            TEXT PRIMARY KEY,   -- '1706.03762', version-less
    latest_version      INTEGER,
    title               TEXT,
    abstract            TEXT,
    authors             TEXT,
    categories          TEXT,               -- space-separated, as arXiv gives it
    primary_category    TEXT,               -- first entry in categories
    submitted_utc       TEXT,               -- ISO-8601, from the v1 <date>
    updated_utc         TEXT,               -- ISO-8601, from the latest version <date>
    oai_datestamp       TEXT,               -- OAI last-modified; NOT the submission date
    doi                 TEXT,
    journal_ref         TEXT,
    license             TEXT,
    deleted             INTEGER DEFAULT 0,

    in_scope            INTEGER DEFAULT 0,  -- matches the configured corpus scope (D1)
    fulltext_status     TEXT DEFAULT 'pending',  -- pending|ok|failed|unavailable
    fulltext_source     TEXT,               -- arxiv_html|ar5iv|pdf
    fulltext_version    INTEGER,            -- version actually parsed; anchors are per-version
    fulltext_fetched_utc TEXT,
    fulltext_attempts   INTEGER DEFAULT 0,
    n_chunks            INTEGER DEFAULT 0,

    s2_status           TEXT DEFAULT 'pending',
    cited_by_count      INTEGER,
    reference_count     INTEGER
);

CREATE INDEX IF NOT EXISTS papers_scope     ON papers(in_scope, fulltext_status);
CREATE INDEX IF NOT EXISTS papers_priority  ON papers(in_scope, cited_by_count DESC);
CREATE INDEX IF NOT EXISTS papers_submitted ON papers(submitted_utc);
CREATE INDEX IF NOT EXISTS papers_s2        ON papers(in_scope, s2_status);

-- Resumable-harvest bookkeeping. The resumption token is written in the SAME
-- transaction as the records it follows, so a kill -9 can never lose or double-count
-- a page: on restart we resume from a token whose records are already committed.
CREATE TABLE IF NOT EXISTS harvest_state (
    key          TEXT PRIMARY KEY,
    value        TEXT,
    updated_utc  TEXT
);

CREATE TABLE IF NOT EXISTS harvest_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    set_spec  TEXT,
    request_n INTEGER,
    records   INTEGER,
    ts_utc    TEXT,
    note      TEXT
);

-- Citation edges (D6, Semantic Scholar). Both endpoints are arXiv ids; references to
-- non-arXiv works are dropped, which is fine since the graph UI only navigates arXiv.
CREATE TABLE IF NOT EXISTS citations (
    src TEXT NOT NULL,     -- citing paper
    dst TEXT NOT NULL,     -- cited paper
    PRIMARY KEY (src, dst)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS citations_dst ON citations(dst);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uri = f"file:{path}?mode=ro" if read_only else f"file:{path}"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    if not read_only:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.executescript(SCHEMA)
        conn.commit()
    return conn


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM harvest_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_state(conn: sqlite3.Connection, key: str, value: str | None) -> None:
    """Upsert harvest state. Call inside the caller's transaction, not on its own."""
    conn.execute(
        "INSERT INTO harvest_state(key, value, updated_utc) VALUES (?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_utc=excluded.updated_utc",
        (key, value, utcnow()),
    )


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "papers": q("SELECT COUNT(*) FROM papers"),
        "in_scope": q("SELECT COUNT(*) FROM papers WHERE in_scope=1"),
        "fulltext_ok": q("SELECT COUNT(*) FROM papers WHERE fulltext_status='ok'"),
        "fulltext_pending": q(
            "SELECT COUNT(*) FROM papers WHERE in_scope=1 AND fulltext_status='pending'"
        ),
        "citations": q("SELECT COUNT(*) FROM citations"),
    }
