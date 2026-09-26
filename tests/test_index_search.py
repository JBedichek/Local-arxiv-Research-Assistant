import pytest

from lara.index import search as S
from lara.store import db


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "corpus.db")
    rows = [
        (1, "2401.1", 1, 0, "", "S1", 0, "S1", 10, "body", 20, None,
         "Zeroshot warmup ramps the learning rate upward before decay begins."),
        (2, "2401.1", 1, 1, "", "S2", 0, "S2", 10, "body", 20, None,
         "Cosine decay lowers the rate smoothly after the ramp completes."),
        (3, "2402.2", 1, 0, "", "S1", 0, "S1", 10, "body", 20, None,
         "Batchsize scaling changes the optimal warmup duration for training."),
    ]
    c.executemany(
        "INSERT INTO chunks (chunk_id, arxiv_id, version, ordinal, section_anchor, anchor_start, "
        "char_start, anchor_end, char_end, kind, n_chars, vector_row, text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    c.commit()
    return c


def test_count_matches_counts_chunks_and_distinct_papers(conn):
    # df_ceiling_frac=1.0: a 3-row corpus makes the default common-word ceiling reject
    # everything (it scales with corpus size); this isolates the counting logic from that.
    out = S.count_matches(conn, "warmup", df_ceiling_frac=1.0)
    assert out["chunks"] == 2 and out["papers"] == 2


def test_count_matches_is_zero_for_terms_absent_entirely(conn):
    assert S.count_matches(conn, "quantum entanglement teleportation", df_ceiling_frac=1.0) == {"chunks": 0, "papers": 0}


def test_count_matches_handles_an_empty_query(conn):
    assert S.count_matches(conn, "") == {"chunks": 0, "papers": 0}
