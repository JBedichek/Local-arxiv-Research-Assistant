"""Training examples, and how they are split.

A triple is one query with a passage the teacher liked and one it did not. Everything
here is pure: it reads the judgements table and shuffles lists, and imports no model.

**Splits are BY QUERY, never by triple.** The same question appears in several triples
with different passages, so splitting rows at random puts near-duplicates of a query on
both sides and the held-out score measures memorisation. Both split functions here group
by canonical query first and divide the groups.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass


from lara.index.prefixes import DOC_PREFIX_UNTITLED as DOC_PREFIX


@dataclass
class Triple:
    query: str
    query_hash: str
    pos_text: str
    neg_text: str
    margin: float          # teacher's positive-minus-negative score




def canon_query(q: str) -> str:
    """A grouping key that survives trailing punctuation and casing.

    `judgements.query_hash` is a SHA of the whitespace-normalised, lowercased question, and
    the unique index on (query_hash, chunk_id, teacher) stops the same judgement being
    stored twice. What it cannot catch is the same question stored under two hashes:
    measured, "What optimizer does scaled dot-product attention use?" and the same string
    without the "?" are distinct rows at cosine 0.998.

    That matters more for MNRL than for storage. Two groups that are really one question
    can land in the same batch, and then one group's positive is a false negative for the
    other — exactly the failure `query_disjoint_batches` exists to prevent. Re-grouping
    here fixes old and new rows alike and needs no migration, whereas changing `qhash`
    itself would split every existing question from its future judgements.
    """
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in q).lower().split())


def _render(item: dict, contextual: bool) -> str:
    """The document string, in the format the reader will actually be searched in.

    ``contextual=True`` reproduces `lara.index.embed.document_text`, which is what the
    29.5 M corpus vectors were built with. False keeps the bare-chunk form the earlier runs
    used, so old results stay reproducible.

    The returned string is already a complete EmbeddingGemma document prompt, so callers
    must not prepend DOC_PREFIX again — `doc_input` below is what enforces that.
    """
    if not contextual:
        return item["text"]
    from lara.index.embed import document_text
    return document_text(item.get("paper_title"), item.get("section_title"), item["text"])


def doc_input(text: str) -> str:
    """Prepend the document prompt unless the text already carries one."""
    return text if text.startswith("title:") else DOC_PREFIX + text



def make_triples(conn: sqlite3.Connection, max_per_query: int = 8,
                 min_margin: float = 0.05, limit: int = 0,
                 hard_frac: float = 0.5, contextual: bool = False) -> list[Triple]:
    """Pair every positive with a negative from the same query.

    ``min_margin`` drops pairs the teacher itself could barely separate: near-zero gaps
    are where the reranker is guessing, and asking the student to reproduce a coin flip
    adds variance without signal.

    It was 0.2, which turned out to be too generous. The overfit check opened at pair_acc
    0.963 — the untrained model already ordered almost every pair correctly, so that
    metric measured nothing and only ``margin_mae`` carried information. A lower floor
    admits genuinely hard pairs.

    ``hard_frac`` reserves half of each query's budget for the negatives the teacher
    scored HIGHEST — the ones retrieval surfaced and the reranker still rejected. Those
    are the discriminations worth learning; a positive against a random chunk is a
    distinction the model already makes.
    """
    from lara.finetune.judgements import training_pairs

    out: list[Triple] = []
    for group in training_pairs(conn):
        pos = sorted(group["positives"], key=lambda x: -(x["score"] or 0))
        neg_all = sorted(group["negatives"], key=lambda x: -(x["score"] or 0))
        if not pos or not neg_all:
            continue
        # Hardest negatives first, then the easiest, so each query contributes both.
        n_hard = max(1, int(len(neg_all) * hard_frac))
        neg = neg_all[:n_hard] + neg_all[n_hard:][::-1]
        from lara.finetune.judgements import qhash
        qh = qhash(group["query"])
        made = 0
        for p in pos:
            for n in neg:
                margin = (p["score"] or 0) - (n["score"] or 0)
                if margin < min_margin:
                    continue
                out.append(Triple(
                    group["query"], canon_query(group["query"]),
                    _render(p, contextual), _render(n, contextual), float(margin)))
                made += 1
                if made >= max_per_query:
                    break
            if made >= max_per_query:
                break
        if limit and len(out) >= limit:
            break

    # Identical (query, positive, negative) can be produced more than once when a chunk is
    # judged under two teachers or a question was harvested twice. 350 of 84,874 measured —
    # small, but a duplicate triple is pure redundant compute and, under an in-batch loss,
    # a guaranteed false negative if both copies land in one batch.
    seen: set[tuple[str, str, str]] = set()
    unique: list[Triple] = []
    for t in out:
        key = (t.query_hash, t.pos_text, t.neg_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique



def split_by_query(triples: list[Triple], k: int, seed: int = 0) -> list[list[int]]:
    """Assign fold indices grouped by query, never by triple."""
    rng = random.Random(seed)
    queries = sorted({t.query_hash for t in triples})
    rng.shuffle(queries)
    fold_of = {q: i % k for i, q in enumerate(queries)}
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, t in enumerate(triples):
        folds[fold_of[t.query_hash]].append(i)
    return folds



def inner_split(triples: list[Triple], frac: float, seed: int = 0
                ) -> tuple[list[Triple], list[Triple]]:
    """Carve a validation slice off the training set, **split by query**.

    Same anti-leak rule as the outer folds: one query contributes many triples sharing its
    phrasing, so splitting by triple would let the model see a paraphrase of every
    validation item during training and make the stopping signal useless.
    """
    rng = random.Random(seed)
    queries = sorted({t.query_hash for t in triples})
    rng.shuffle(queries)
    n_val = max(1, int(round(len(queries) * frac)))
    val_q = set(queries[:n_val])
    train = [t for t in triples if t.query_hash not in val_q]
    val = [t for t in triples if t.query_hash in val_q]
    return (train, val) if train and val else (triples, [])

