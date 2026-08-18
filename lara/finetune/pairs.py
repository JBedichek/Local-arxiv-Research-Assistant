"""Build training pairs from citation contexts (PLAN.md §5.2).

**Why contexts rather than whole papers.** Citations are recorded between papers, which
suggests you need a paper-level representation to train on them — and encoding a whole
paper is out of reach for a 2048-token encoder. The reframe that dissolves the problem: a
citation is not really a relation between two documents, it is a relation between *one
specific claim in the citing paper* and *the contribution of the cited paper*. The
paragraph containing the citation marker localises that claim exactly.

So each training example is::

    anchor   = the paragraph in A that cites B      (~250 tokens, already a chunk)
    positive = B's title + abstract                 (~250 tokens)

Both sides are short, nothing long-context is needed, and the pair has the same shape as
the task at serving time: given a passage, find the paper content that belongs with it.

LaTeXML gives us everything required: ``<cite>`` elements link to ``#bib.bibN``, the
matching ``li.ltx_bibitem`` carries the reference text, and the enclosing ``div.ltx_para``
is the context together with its stable anchor id.
"""

from __future__ import annotations

import glob
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import zstandard as zstd
from lxml import html as LH

from lara.ingest.parse import _ARXIV_IN_HREF, _ARXIV_IN_TEXT, cls, inline_math_as_latex

SCHEMA = """
CREATE TABLE IF NOT EXISTS citation_contexts (
    id          INTEGER PRIMARY KEY,
    src         TEXT NOT NULL,      -- citing paper
    dst         TEXT NOT NULL,      -- cited paper (arXiv id)
    para_anchor TEXT,               -- where in src the citation sits
    context     TEXT NOT NULL,      -- the citing paragraph
    n_refs      INTEGER NOT NULL,   -- papers cited in this same paragraph
    n_chars     INTEGER,
    relevance   REAL,               -- filled in by the relevance filter; NULL = unjudged
    verdict     TEXT                -- substantive | perfunctory | unjudged
);
CREATE INDEX IF NOT EXISTS cc_src ON citation_contexts(src);
CREATE INDEX IF NOT EXISTS cc_dst ON citation_contexts(dst);
CREATE UNIQUE INDEX IF NOT EXISTS cc_uniq ON citation_contexts(src, dst, para_anchor);
CREATE TABLE IF NOT EXISTS citation_context_files (path TEXT PRIMARY KEY);
"""

_WS = re.compile(r"\s+")
_dctx = zstd.ZstdDecompressor()


@dataclass
class Context:
    src: str
    dst: str
    para_anchor: str
    context: str
    n_refs: int


def bib_arxiv_map(root) -> dict[str, str]:
    """Map ``bib.bibN`` ids to arXiv ids, from hrefs and from plain text alike."""
    out: dict[str, str] = {}
    for item in root.xpath(f"//*[{cls('ltx_bibitem')}]"):
        bid = item.get("id")
        if not bid:
            continue
        for a in item.xpath(".//a[@href]"):
            m = _ARXIV_IN_HREF.search(a.get("href") or "")
            if m:
                out[bid] = m.group(1)
                break
        if bid not in out:
            m = _ARXIV_IN_TEXT.search(item.text_content())
            if m:
                out[bid] = m.group(1)
    return out


def extract(raw: bytes, src: str, min_chars: int = 120,
            max_refs: int = 6) -> list[Context]:
    """Pull (citing paragraph -> cited arXiv id) pairs out of one rendered paper."""
    root = LH.fromstring(raw)
    bibmap = bib_arxiv_map(root)
    if not bibmap:
        return []

    # Equations must become LaTeX before the paragraph is read, for the same reason as in
    # the ingest parser: text_content() otherwise interleaves MathML with its fallback.
    inline_math_as_latex(root)

    out: list[Context] = []
    seen: set[tuple[str, str]] = set()
    for cite in root.xpath("//cite"):
        targets = []
        for a in cite.xpath(".//a[@href]"):
            href = (a.get("href") or "").lstrip("#")
            if href in bibmap:
                targets.append(bibmap[href])
        if not targets:
            continue

        para = cite
        while para is not None and not (
            "ltx_para" in (para.get("class") or "").split()
        ):
            para = para.getparent()
        if para is None or not para.get("id"):
            continue
        text = _WS.sub(" ", para.text_content()).strip()
        if len(text) < min_chars:
            continue

        # A paragraph citing a dozen works ("see [3,7,9,12,...] for surveys") says almost
        # nothing specific about any one of them. n_refs is recorded so training can
        # weight or drop these rather than treating them like a focused citation.
        n_refs = len(set(targets))
        if n_refs > max_refs:
            continue
        for dst in set(targets):
            if dst == src or (dst, para.get("id")) in seen:
                continue
            seen.add((dst, para.get("id")))
            out.append(Context(src, dst, para.get("id"), text[:2000], n_refs))
    return out


def paper_id_from_path(path: str) -> str:
    return Path(path).name.split(".arxiv_html")[0].split(".ar5iv")[0].replace("_", "/")


def build(conn: sqlite3.Connection, raw_root: Path, limit: int = 0,
          progress=None) -> dict[str, int]:
    """Scan the raw-HTML cache and persist every citation context found.

    Resumable: processed files are recorded, so a re-run only picks up new papers.
    """
    conn.executescript(SCHEMA)
    done = {r[0] for r in conn.execute("SELECT path FROM citation_context_files")}
    files = sorted(glob.glob(str(raw_root / "*" / "*.html.zst")))
    stats = {"files": 0, "contexts": 0, "skipped": 0}

    batch: list[Context] = []
    batch_files: list[str] = []
    for path in files:
        if path in done:
            continue
        if limit and stats["files"] >= limit:
            break
        src = paper_id_from_path(path)
        try:
            raw = _dctx.decompress(Path(path).read_bytes())
            found = extract(raw, src)
        except Exception:
            stats["skipped"] += 1
            found = []
        batch.extend(found)
        batch_files.append(path)
        stats["files"] += 1
        stats["contexts"] += len(found)

        if len(batch_files) >= 200:
            _flush(conn, batch, batch_files)
            batch, batch_files = [], []
            if progress is not None:
                progress.send(stats)
    _flush(conn, batch, batch_files)
    return stats


def _flush(conn: sqlite3.Connection, batch: list[Context], files: list[str]) -> None:
    if not files:
        return
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO citation_contexts"
            "(src,dst,para_anchor,context,n_refs,n_chars,verdict) "
            "VALUES (?,?,?,?,?,?,'unjudged')",
            [(c.src, c.dst, c.para_anchor, c.context, c.n_refs, len(c.context))
             for c in batch],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO citation_context_files(path) VALUES (?)",
            [(f,) for f in files],
        )
