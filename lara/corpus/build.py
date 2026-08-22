"""Turn accepted sources into a searchable index: blocks, chunks, vectors.

This is the "build knowledge base" step, and it is deliberately thin. Chunking is
``lara.ingest.parse.chunk_blocks`` unchanged, embedding is ``lara.index.embed``, storage is
``lara.index.vectors.VectorStore``. What is new is only the part that differs from arXiv:
turning a PDF or a web page into anchored blocks.

**Anchors are page-based for PDFs.** A citation that says "page 47" is worth far more in a
flight manual or a textbook than the paragraph counter arXiv gets away with, because the
reader has the physical document and wants to turn to it. Pages become sections, so a
chunk never spans two pages and every answer can point at one.

**Building is incremental.** The vector store is append-only and documents are keyed by
content hash, so adding one source to a built corpus embeds one source. That matters more
here than for arXiv: a study corpus grows a document at a time over weeks, and a design
that re-embedded everything would make the second document as expensive as the first.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from lara.ingest.parse import Block, chunk_blocks

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        INTEGER PRIMARY KEY,
    sha256        TEXT NOT NULL UNIQUE,   -- content address; the same file twice is one row
    url           TEXT,
    title         TEXT,
    content_type  TEXT,
    licence       TEXT,
    licence_label TEXT,
    n_chars       INTEGER,
    n_pages       INTEGER,
    n_chunks      INTEGER DEFAULT 0,
    added_utc     TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       INTEGER PRIMARY KEY,
    doc_id         INTEGER NOT NULL,
    -- Named `arxiv_id` because lara.index.search.hydrate joins on that name, and one
    -- retrieval path serving both corpus kinds is worth more than an accurate column
    -- name. Here it holds the document's content hash. See the compatibility note below.
    arxiv_id       TEXT NOT NULL,
    version        INTEGER NOT NULL DEFAULT 0,
    ordinal        INTEGER NOT NULL,
    section_anchor TEXT,                  -- 'pg47' for PDFs, 'S3' for structured HTML
    anchor_start   TEXT NOT NULL,
    char_start     INTEGER NOT NULL,
    anchor_end     TEXT NOT NULL,
    char_end       INTEGER NOT NULL,
    kind           TEXT,
    n_chars        INTEGER,
    vector_row     INTEGER,               -- row in the flat vector files; NULL = unembedded
    text           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc    ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_vector ON chunks(vector_row);
CREATE INDEX IF NOT EXISTS chunks_akey   ON chunks(arxiv_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='chunk_id', tokenize='unicode61');
-- Document frequencies, for the BM25 term selector in search.plan_query.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab USING fts5vocab(chunks_fts, row);

-- ── compatibility with the arXiv retrieval path ────────────────────────────────────
-- `lara.index.search.hydrate` joins `papers` and `sections` to render a citation. Rather
-- than fork it, a corpus exposes those two names as views over what it does have. The
-- retriever, the reranker, RRF fusion and the tier-1/tier-2 split then work here
-- unchanged, which is the whole reason the chunk table looks the way it does.
CREATE VIEW IF NOT EXISTS papers AS
    SELECT sha256 AS arxiv_id, 0 AS version, title, url, licence, added_utc
    FROM documents;
-- A PDF's section title is its page, which is exactly what a reader wants to be told.
CREATE VIEW IF NOT EXISTS sections AS
    SELECT arxiv_id, version, section_anchor AS anchor,
           CASE WHEN section_anchor LIKE 'pg%'
                THEN 'page ' || substr(section_anchor, 3)
                ELSE section_anchor END AS title
    FROM chunks WHERE section_anchor <> '' GROUP BY arxiv_id, version, section_anchor;
"""

#: A paragraph shorter than this is a page number, a running header or a stray caption
#: fragment. Keeping them costs an embedding each and pollutes retrieval with text that
#: matches everything weakly.
MIN_BLOCK_CHARS = 60

_PARA = re.compile(r"\n\s*\n+")


@dataclass
class BuildStats:
    documents: int = 0
    chunks: int = 0
    embedded: int = 0
    skipped: int = 0
    seconds: float = 0.0


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # WAL so a build can be read while it runs — the reader may already have this corpus
    # open while another document is being added to it.
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def blocks_from_pages(pages: list[str]) -> list[Block]:
    """Anchored blocks from per-page text, one section per page."""
    out: list[Block] = []
    ordinal = 0
    for pno, page in enumerate(pages, start=1):
        for i, para in enumerate(p.strip() for p in _PARA.split(page)):
            if len(para) < MIN_BLOCK_CHARS:
                continue
            out.append(Block(anchor=f"pg{pno}.p{i}", section_anchor=f"pg{pno}",
                             kind="body", text=para, ordinal=ordinal))
            ordinal += 1
    return out


def blocks_from_text(text: str) -> list[Block]:
    """Anchored blocks from flat text, when there are no pages to key on."""
    out: list[Block] = []
    for i, para in enumerate(p.strip() for p in _PARA.split(text)):
        if len(para) < MIN_BLOCK_CHARS:
            continue
        out.append(Block(anchor=f"p{i}", section_anchor="", kind="body",
                         text=para, ordinal=len(out)))
    return out


def pdf_pages(raw: bytes, max_pages: int = 3000) -> list[str]:
    """Per-page text, so anchors can name a page the reader can turn to."""
    try:
        import pymupdf
    except ImportError:                               # older wheels expose it as fitz
        try:
            import fitz as pymupdf
        except ImportError:
            return []
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as doc:
            return [doc[i].get_text() for i in range(min(doc.page_count, max_pages))]
    except Exception:                                 # noqa: BLE001
        return []


def blocks_for(doc, raw: bytes | None) -> tuple[list[Block], int]:
    """Blocks and a page count for one fetched document."""
    if raw and ("pdf" in (doc.content_type or "") or raw[:5] == b"%PDF-"):
        pages = pdf_pages(raw)
        if pages:
            return blocks_from_pages(pages), len(pages)
    return blocks_from_text(doc.text), 0


def add_document(conn: sqlite3.Connection, doc, raw: bytes | None, *,
                 target_chars: int = 1000, overlap_frac: float = 0.15) -> int:
    """Insert one document and its chunks. Returns chunks added, 0 if already present.

    Keyed on the content hash rather than the URL: the same textbook found on a publisher
    site and an Internet Archive mirror is one document, and a search that returns both
    should cost one embedding rather than two identical ones.
    """
    if conn.execute("SELECT 1 FROM documents WHERE sha256=?", (doc.sha256,)).fetchone():
        return 0

    blocks, n_pages = blocks_for(doc, raw)
    if not blocks:
        return 0
    chunks = chunk_blocks(blocks, target_chars=target_chars, overlap_frac=overlap_frac,
                          never_cross_sections=True)
    if not chunks:
        return 0

    lic = doc.licence
    cur = conn.execute(
        """INSERT INTO documents (sha256, url, title, content_type, licence,
                                  licence_label, n_chars, n_pages, n_chunks, added_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (doc.sha256, doc.url, doc.title, doc.content_type,
         lic.verdict if lic else "unknown", lic.label if lic else "",
         doc.chars, n_pages, len(chunks),
         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    doc_id = cur.lastrowid

    conn.executemany(
        """INSERT INTO chunks (doc_id, arxiv_id, ordinal, section_anchor, anchor_start,
                               char_start, anchor_end, char_end, kind, n_chars, text)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [(doc_id, doc.sha256, c.ordinal, c.section_anchor, c.anchor_start, c.char_start,
          c.anchor_end, c.char_end, c.kind, len(c.text), c.text) for c in chunks])
    conn.commit()
    return len(chunks)


def rebuild_fts(conn: sqlite3.Connection) -> None:
    """Populate the external-content FTS index from the chunks table."""
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()


def embed_pending(conn: sqlite3.Connection, store, embedder, *,
                  batch: int = 256, progress=None) -> int:
    """Embed every chunk without a vector row. Safe to interrupt and re-run.

    Vectors are appended and fsynced before the chunk rows are stamped, exactly as the
    arXiv path does: a crash between the two leaves unreferenced rows at the tail of the
    file — wasted bytes, never wrong answers — and those chunks are simply re-embedded on
    the next run. The reverse order would hand out row numbers pointing at data that was
    never written.
    """
    from lara.index.embed import document_text

    total = 0
    while True:
        rows = conn.execute(
            """SELECT c.chunk_id, c.text, c.section_anchor, d.title
               FROM chunks c JOIN documents d USING(doc_id)
               WHERE c.vector_row IS NULL ORDER BY c.chunk_id LIMIT ?""", (batch,)
        ).fetchall()
        if not rows:
            break
        texts = [document_text(r["title"], r["section_anchor"] or None, r["text"])
                 for r in rows]
        vecs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False, batch_size=min(batch, 64))
        start = store.append(vecs)
        conn.executemany("UPDATE chunks SET vector_row=? WHERE chunk_id=?",
                         [(start + i, r["chunk_id"]) for i, r in enumerate(rows)])
        conn.commit()
        total += len(rows)
        if progress is not None:
            progress(total)
    return total
