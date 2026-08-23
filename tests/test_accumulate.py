"""Promoting research output back into the corpus it came from."""

from __future__ import annotations

import sqlite3

import pytest

from lara.index import accumulate as A

SCHEMA = """
CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, latest_version INT, title TEXT,
  abstract TEXT, authors TEXT, categories TEXT, primary_category TEXT,
  submitted_utc TEXT, in_scope INT, deleted INT, fulltext_status TEXT,
  fulltext_version INT, n_chunks INT);
CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, arxiv_id TEXT NOT NULL,
  version INT NOT NULL, ordinal INT NOT NULL, section_anchor TEXT,
  anchor_start TEXT NOT NULL, char_start INT NOT NULL, anchor_end TEXT NOT NULL,
  char_end INT NOT NULL, kind TEXT, n_chars INT, vector_row INT, text TEXT NOT NULL);
CREATE UNIQUE INDEX chunks_unique ON chunks(arxiv_id, version, ordinal);
CREATE TABLE synthesis_runs (run_id TEXT PRIMARY KEY, question TEXT, tldr TEXT,
  thorough TEXT, created_utc TEXT);
CREATE TABLE synthesis_claims (run_id TEXT, chunk_id INT, arxiv_id TEXT,
  paper_title TEXT, section TEXT, name TEXT, claim TEXT, method TEXT, metric TEXT,
  value TEXT, condition TEXT, round_n INT, score REAL, text_len INT);
"""


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA)
    c.execute("INSERT INTO papers (arxiv_id, title) VALUES ('2501.00001','A Real Paper')")
    c.execute("INSERT INTO chunks VALUES (1,'2501.00001',1,0,'s1','a',0,'b',100,'body',100,7,"
              "'real text')")
    return c


def _run(c, run_id, question="q", claims=(), tldr="the short answer", thorough="the long one"):
    c.execute("INSERT INTO synthesis_runs VALUES (?,?,?,?,'2026-01-01')",
              (run_id, question, tldr, thorough))
    for i, (arxiv, claim) in enumerate(claims):
        c.execute("INSERT INTO synthesis_claims (run_id, chunk_id, arxiv_id, name, claim, "
                  "method, metric, value, condition) VALUES (?,1,?,?,?,?,?,?,?)",
                  (run_id, arxiv, f"n{i}", claim, "ablation", "acc", "0.9", "8 seeds"))
    c.commit()


def test_promotes_claims_and_syntheses(conn):
    _run(conn, "r1", claims=[("2501.00001", "bigger batches help")])
    out = A.promote_run(conn, "r1")
    assert out.runs == 1 and out.claim_chunks == 1 and out.synthesis_chunks == 2
    kinds = dict(conn.execute(
        "SELECT kind, COUNT(*) FROM chunks WHERE vector_row IS NULL GROUP BY kind"))
    assert kinds == {A.CLAIM: 1, A.SYNTHESIS: 2}


def test_claim_keeps_the_numbers_that_make_it_checkable(conn):
    _run(conn, "r1", claims=[("2501.00001", "bigger batches help")])
    A.promote_run(conn, "r1")
    text = conn.execute("SELECT text FROM chunks WHERE kind=?", (A.CLAIM,)).fetchone()[0]
    assert "bigger batches help" in text
    assert "acc = 0.9" in text and "8 seeds" in text


def test_claim_is_attributed_to_its_paper_and_inherits_its_anchor(conn):
    _run(conn, "r1", claims=[("2501.00001", "c")])
    A.promote_run(conn, "r1")
    r = conn.execute("SELECT arxiv_id, anchor_start, char_start, section_anchor "
                     "FROM chunks WHERE kind=?", (A.CLAIM,)).fetchone()
    # citing a claim must land on the passage it compresses
    assert r == ("2501.00001", "a", 0, "s1")


def test_synthesis_gets_a_pseudo_paper_so_hydrate_can_join(conn):
    _run(conn, "r1", question="does depth help?")
    A.promote_run(conn, "r1")
    row = conn.execute("SELECT title FROM papers WHERE arxiv_id=?",
                       (A.run_paper_id("r1"),)).fetchone()
    assert row[0] == "does depth help?"


def test_promotion_is_idempotent(conn):
    _run(conn, "r1", claims=[("2501.00001", "c")])
    A.promote_run(conn, "r1")
    again = A.promote_run(conn, "r1")
    assert again.skipped == 1 and again.chunks == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 4


def test_two_runs_citing_one_paper_do_not_collide_on_the_unique_index(conn):
    # the bug this guards: a per-run ordinal counter gives both runs ordinal 900000
    _run(conn, "r1", claims=[("2501.00001", "first claim")])
    _run(conn, "r2", claims=[("2501.00001", "second claim")])
    A.promote_run(conn, "r1")
    A.promote_run(conn, "r2")
    ords = [r[0] for r in conn.execute(
        "SELECT ordinal FROM chunks WHERE arxiv_id='2501.00001' AND kind=? ORDER BY ordinal",
        (A.CLAIM,))]
    assert ords == [A.SYNTHETIC_ORDINAL_BASE, A.SYNTHETIC_ORDINAL_BASE + 1]


def test_accumulation_is_unique_across_runs(conn):
    # the same claim found twice is one chunk: otherwise retrieval ranks a paper by how
    # often it has been researched
    _run(conn, "r1", claims=[("2501.00001", "identical finding")])
    _run(conn, "r2", claims=[("2501.00001", "identical finding")], tldr="x", thorough="y")
    A.promote_run(conn, "r1")
    out = A.promote_run(conn, "r2")
    assert out.duplicates == 1 and out.claim_chunks == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE kind=?", (A.CLAIM,)).fetchone()[0] == 1


def test_synthetic_ordinals_never_touch_real_ones(conn):
    _run(conn, "r1", claims=[("2501.00001", "c")])
    A.promote_run(conn, "r1")
    real = conn.execute("SELECT ordinal FROM chunks WHERE kind='body'").fetchone()[0]
    synth = conn.execute("SELECT MIN(ordinal) FROM chunks WHERE kind=?", (A.CLAIM,)).fetchone()[0]
    assert real == 0 and synth >= A.SYNTHETIC_ORDINAL_BASE


def test_promote_all_skips_what_is_done(conn):
    _run(conn, "r1", claims=[("2501.00001", "a")])
    _run(conn, "r2", claims=[("2501.00001", "b")])
    A.promote_run(conn, "r1")
    out = A.promote_all(conn)
    assert out.runs == 1 and out.skipped == 1


def test_everything_promoted_is_pending_a_vector(conn):
    _run(conn, "r1", claims=[("2501.00001", "c")])
    A.promote_run(conn, "r1")
    # this is the whole mechanism: lara's embedder finds work by vector_row IS NULL
    assert A.pending_vectors(conn) == 3


def test_empty_bodies_are_not_promoted(conn):
    _run(conn, "r1", claims=[("2501.00001", "   ")], tldr="", thorough="  ")
    out = A.promote_run(conn, "r1")
    assert out.chunks == 0
