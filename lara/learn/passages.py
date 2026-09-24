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
    #: The element id of the exact block this passage is -- `S4.F2` for a figure's caption,
    #: `S3.p4` for an ordinary paragraph (see lara.ingest.parse.extract_blocks/Block.anchor).
    #: NOT the enclosing section's id (`section`, above, is that -- a human-readable title,
    #: not an id). Only a `kind="caption"` passage's own anchor is a figure/table float
    #: itself, which is what makes a claim built from one resolvable back to a real image,
    #: not just a citation (see CorpusRetriever.figure).
    anchor: str = ""
    version: int = 0

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

    def __init__(self, state, figure=None):
        self.retriever = state.retriever
        self._conn = state.conn
        self._neighbours = state.neighbours
        # Injected rather than reached for directly, the same shape `Llm`/`embed_fn` already
        # are here: lara.learn never imports lara.serve, so resolving an anchor to an actual
        # image -- which needs the raw cached HTML lara.serve.papers owns -- is a capability
        # the caller hands in, not one this module goes looking for.
        self._figure = figure

    def search(self, query: str, k: int = 8, *, papers: list[str] | None = None) -> list[Passage]:
        hits = self.retriever.retrieve(query, final_k=k * 2, papers=papers).hits
        hits = [h for h in hits if h.kind not in EXCLUDED_KINDS][:k]
        meta = self._meta({h.arxiv_id for h in hits})
        return [Passage(chunk_id=h.chunk_id, arxiv_id=h.arxiv_id, title=h.paper_title,
                        text=h.text, section=h.section_title, kind=h.kind,
                        anchor=h.anchor_start, version=h.version,
                        **meta.get(h.arxiv_id, {})) for h in hits]

    def coverage(self, query: str) -> dict[str, int]:
        """{"chunks", "papers"}: how much of the corpus touches `query`, from FTS5 alone --
        cheap enough to call before deciding how hard to search, unlike `search` itself."""
        from lara.index.search import count_matches

        try:
            return count_matches(self._conn(), query)
        except Exception:                                      # noqa: BLE001
            return {"chunks": 0, "papers": 0}

    def neighbours(self, arxiv_id: str) -> dict[str, list[str]]:
        """{"cites", "cited_by"} arxiv ids one hop out in the citation graph, or both empty
        on any failure -- a citation walk that cannot resolve degrades to no walk, not a
        broken build."""
        try:
            return self._neighbours(arxiv_id)
        except Exception:                                      # noqa: BLE001
            return {"cites": [], "cited_by": []}

    def figure(self, arxiv_id: str, version: int, anchor: str) -> dict | None:
        """The image and caption of the figure/table float at `anchor`, or None -- no image
        there, the paper's HTML is not cached locally, or no lookup was injected at all
        (a caller that never needs figures, e.g. every test in this package)."""
        if self._figure is None:
            return None
        try:
            return self._figure(arxiv_id, version, anchor)
        except Exception:                                      # noqa: BLE001
            return None

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
