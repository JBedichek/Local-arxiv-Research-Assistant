"""Evidence a claim can rest on, and the citations that point at it.

Synthesis cites literature by chunk id -- `[3352954]`, not `[1]` -- which is what keeps a
citation followable once it has been copied out of the text: a chunk id is a primary key, a
positional marker is not. `bind` resolves every bracket in a text to a `Reference`.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

ARXIV_ABS = "https://arxiv.org/abs/"

#: A citation bracket. Models group citations -- `[12345, 67890]` -- so a pattern matching
#: only the single form reports a sentence as grounded on the one key it could see.
#: Deliberately narrow otherwise: `[TODO]`, `[sic]` and `[Fig. 2]` are not citations.
CITATION = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_KEYS_IN = re.compile(r"\d+")


@dataclass(frozen=True)
class Reference:
    """One passage, addressed well enough to act on without any other state."""

    key: str
    chunk_id: int = 0
    arxiv_id: str = ""
    version: int = 0
    paper_title: str = ""
    section: str = ""
    text: str = ""
    #: The structured claim drawn from the passage, when there is one.
    claim: str = ""
    score: float = 0.0

    @property
    def arxiv_url(self) -> str:
        if not self.arxiv_id:
            return ""
        v = f"v{self.version}" if self.version else ""
        return f"{ARXIV_ABS}{self.arxiv_id}{v}"

    @property
    def source(self) -> str:
        return self.arxiv_url

    @property
    def title(self) -> str:
        return self.paper_title

    def to_dict(self) -> dict:
        return {**asdict(self), "source": self.source, "arxiv_url": self.arxiv_url,
                "title": self.title}

    @classmethod
    def from_dict(cls, d) -> "Reference | None":
        """Back from a record, or None if this is not one. `to_dict` adds derived keys that
        are not fields, and a record from an older writer may carry fields this no longer
        has, so unknown keys are dropped rather than refused."""
        if isinstance(d, cls):
            return d
        if not isinstance(d, dict) or not d.get("key"):
            return None
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CitedText:
    """Prose plus everything needed to follow what it cites."""

    text: str
    references: dict[str, Reference] = field(default_factory=dict)
    #: Citations that resolved to nothing -- a finding for the caller, who decides whether to
    #: retry, drop the sentence, or treat the answer as unsupported.
    unresolved: list[str] = field(default_factory=list)

    @property
    def papers(self) -> list[str]:
        return sorted({r.arxiv_id for r in self.references.values() if r.arxiv_id})

    def to_dict(self) -> dict:
        return {"text": self.text,
                "references": {k: v.to_dict() for k, v in self.references.items()},
                "unresolved": self.unresolved, "papers": self.papers}


def parse_keys(text: str) -> list[str]:
    """Every citation key, in order of first appearance, deduplicated; `[a, b]` is two."""
    keys: list[str] = []
    for m in CITATION.finditer(text or ""):
        keys.extend(_KEYS_IN.findall(m.group(0)))
    return list(dict.fromkeys(keys))


def _ref(chunk_id: int, d: dict) -> Reference:
    return Reference(
        key=str(chunk_id), chunk_id=chunk_id, arxiv_id=d.get("arxiv_id", ""),
        version=int(d.get("version") or 0), paper_title=d.get("paper_title", ""),
        section=d.get("section", ""), text=d.get("text", ""), claim=d.get("claim", ""),
        score=float(d.get("score") or 0.0))


def paper_ref(*, chunk_id: int, **fields) -> Reference:
    """A reference to one passage, keyed by its chunk id."""
    return _ref(chunk_id, fields)


def from_claims(claims) -> dict[str, Reference]:
    """Index a synthesis run's claims by citation key. The claim text is carried through
    because it is what the answer was written from."""
    return {str(c.chunk_id): _ref(c.chunk_id, {
        "arxiv_id": c.arxiv_id, "paper_title": c.paper_title, "section": c.section,
        "claim": c.claim, "score": c.score}) for c in claims}


def bind(text: str, *, known: dict[str, Reference] | None = None, conn=None) -> CitedText:
    """Resolve every citation in `text`.

    `known` is consulted first -- it carries the extracted claims, which the corpus does
    not. Unresolved keys are then looked up in the corpus through `conn`, so a citation to
    a passage outside the evidence table still resolves to a real paper.
    """
    known = dict(known or {})
    keys = parse_keys(text)
    missing = [k for k in keys if k not in known]
    if missing and conn is not None:
        from lara.index.search import hydrate

        for cid, hit in hydrate(conn, [int(k) for k in missing]).items():
            known[str(cid)] = _ref(cid, hit.to_dict())
    return CitedText(text=text, references={k: known[k] for k in keys if k in known},
                     unresolved=[k for k in keys if k not in known])
