"""Passages from the paper corpus -- the only thing a lesson may be grounded in."""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Chunk kinds this system wrote into the index itself (`synthesis`, `claim`). Grounding
#: on them would let a lesson cite its own earlier output as if it were literature.
EXCLUDED_KINDS = frozenset({"claim", "synthesis"})

#: A title containing one of these is a survey/tutorial -- preferred for a course skeleton.
OVERVIEW_WORDS = ("survey", "review", "tutorial", "overview", "primer", "introduction to",
                  "a guide", "lecture notes", "handbook")


@dataclass(frozen=True)
class Passage:
    chunk_id: int
    arxiv_id: str
    title: str
    text: str
    section: str = ""
    kind: str = ""
    date: str = ""
    cited_by: int = 0
    journal_ref: str = ""

    @property
    def key(self) -> str:
        return f"{self.arxiv_id}#{self.chunk_id}"

    @property
    def is_overview(self) -> bool:
        low = self.title.lower()
        return any(w in low for w in OVERVIEW_WORDS)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Passage":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


class CorpusRetriever:
    """The warm lara retriever, narrowed to what learning may cite. Blocking -- callers
    run it in a thread."""

    def __init__(self, state):
        self.retriever = state.retriever
        self._conn = state.conn

    def search(self, query: str, k: int = 8) -> list[Passage]:
        hits = self.retriever.retrieve(query, final_k=k * 2).hits
        hits = [h for h in hits if h.kind not in EXCLUDED_KINDS][:k]
        meta = self._meta({h.arxiv_id for h in hits})
        return [Passage(chunk_id=h.chunk_id, arxiv_id=h.arxiv_id, title=h.paper_title,
                        text=h.text, section=h.section_title, kind=h.kind,
                        **meta.get(h.arxiv_id, {})) for h in hits]

    def relevance(self, pairs: list[tuple[str, str]]) -> list[float] | None:
        """Reranker scores for (claim, passage) pairs -- a relevance screen, not entailment.
        None when no reranker is loaded."""
        ce = getattr(self.retriever, "cross_encoder", None)
        if ce is None or not pairs:
            return None
        return [float(s) for s in ce.predict(pairs)]

    def _meta(self, ids: set[str]) -> dict[str, dict]:
        if not ids:
            return {}
        conn = self._conn()
        marks = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT arxiv_id, submitted_utc, cited_by_count, journal_ref FROM papers "
            f"WHERE arxiv_id IN ({marks})", list(ids)).fetchall()
        return {r["arxiv_id"]: {"date": (r["submitted_utc"] or "")[:10],
                                "cited_by": int(r["cited_by_count"] or 0),
                                "journal_ref": r["journal_ref"] or ""} for r in rows}


def embed_fn(embedder):
    """A plain `str -> list[float]` over the retriever's embedder, so callers need not know its
    interface. Returns [] when there is no embedder or it fails."""
    def embed(text: str) -> list[float]:
        if embedder is None or not text.strip():
            return []
        try:
            return [float(x) for x in embedder.encode([text], convert_to_numpy=True)[0]]
        except Exception:                                      # noqa: BLE001
            return []
    return embed
