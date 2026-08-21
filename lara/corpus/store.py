"""A corpus as a self-contained directory, and the recipe that describes how to rebuild it.

    data/corpora/calculus/
        recipe.json      the goal, the queries, and every accept/reject decision
        meta.sqlite      documents, chunks, FTS
        vectors/         fp16.bin, int8.bin
        raw/             fetched originals, named by content hash

**Why a directory rather than a corpus_id column.** Deleting a corpus is `rm -rf` instead
of a twenty-million-row DELETE; sharing one is copying a folder; and a fifty-page flight
manual never has to live inside a 42 GB file built for arXiv. The cost is that searching
two corpora at once means loading two indexes, which is the right trade for a reader who
searches one subject at a time.

**The recipe is the interesting artefact.** It records the goal, the queries the model
generated, and every source with its hash, licence and verdict — but not the documents.
That makes a corpus reproducible and shareable *without redistributing anything*: a
recipe for a corpus of copyrighted manuals is just a list of URLs and decisions, which is
information about documents rather than the documents themselves. Send someone the recipe
and their machine fetches and builds its own copy, or does not, according to what they are
entitled to.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

RECIPE_NAME = "recipe.json"
SAFE_NAME = re.compile(r"[^a-z0-9._-]+")

#: Total extracted text a build may accumulate before it stops and asks. Not a hard cap on
#: corpus size — the reader sets this, and can raise it mid-build — but a default that
#: notices when "pull some calculus textbooks" has quietly become four gigabytes.
DEFAULT_TEXT_BUDGET = 512 << 20      # 512 MB of text, roughly 100 M words

#: Refuse to start a build that would leave the disk this empty, and warn while it runs.
MIN_FREE_BYTES = 5 << 30


def slugify(name: str) -> str:
    """A directory name that is safe, lowercase and recognisable as what the user typed."""
    slug = SAFE_NAME.sub("-", name.strip().lower()).strip("-.")
    return slug or "corpus"


@dataclass
class Source:
    """One document a corpus is built from, and why it is in or out."""

    url: str
    title: str = ""
    sha256: str = ""
    bytes_downloaded: int = 0
    chars: int = 0
    content_type: str = ""
    licence: str = "unknown"          # a lara.corpus.licence verdict
    licence_label: str = ""
    relevance: float | None = None    # cross-encoder score against the goal
    decided: str = "pending"          # pending | accepted | rejected
    reason: str = ""                  # why it was rejected, or who accepted it
    found_by: str = ""                # the query that surfaced it
    added_utc: str = ""

    @property
    def redistributable(self) -> bool:
        from lara.corpus import licence as LIC
        return self.licence in LIC.REDISTRIBUTABLE


@dataclass
class Recipe:
    name: str
    goal: str = ""                    # what the reader asked for, in their words
    queries: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    text_budget: int = DEFAULT_TEXT_BUDGET
    created_utc: str = ""
    built_utc: str = ""
    chunks: int = 0
    embedded: int = 0

    # ── views the builder and the UI both need ──────────────────────────────────────
    def accepted(self) -> list[Source]:
        return [s for s in self.sources if s.decided == "accepted"]

    def pending(self) -> list[Source]:
        return [s for s in self.sources if s.decided == "pending"]

    def text_bytes(self) -> int:
        return sum(s.chars for s in self.accepted())

    def by_url(self, url: str) -> Source | None:
        return next((s for s in self.sources if s.url == url), None)

    def by_hash(self, sha: str) -> Source | None:
        return next((s for s in self.sources if sha and s.sha256 == sha), None)

    def publishable(self) -> tuple[bool, str]:
        from lara.corpus import licence as LIC
        return LIC.corpus_verdict(
            [LIC.Licence(s.licence, s.licence_label) for s in self.accepted()])


class Corpus:
    """One corpus directory, and the operations that do not need the index loaded."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ── layout ──────────────────────────────────────────────────────────────────────
    @property
    def recipe_path(self) -> Path:
        return self.root / RECIPE_NAME

    @property
    def db_path(self) -> Path:
        return self.root / "meta.sqlite"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def vectors_dir(self) -> Path:
        return self.root / "vectors"

    @property
    def fp16_path(self) -> Path:
        return self.vectors_dir / "fp16.bin"

    @property
    def int8_path(self) -> Path:
        return self.vectors_dir / "int8.bin"

    @property
    def built(self) -> bool:
        return self.db_path.exists() and self.int8_path.exists()

    # ── recipe io ───────────────────────────────────────────────────────────────────
    def load(self) -> Recipe:
        try:
            blob = json.loads(self.recipe_path.read_text())
        except (OSError, json.JSONDecodeError):
            return Recipe(name=self.root.name, created_utc=_now())
        sources = [Source(**s) for s in blob.pop("sources", [])]
        blob.pop("name", None)
        return Recipe(name=self.root.name, sources=sources, **blob)

    def save(self, recipe: Recipe) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        blob = asdict(recipe)
        blob.pop("name", None)
        tmp = self.recipe_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, indent=1, ensure_ascii=False))
        os.replace(tmp, self.recipe_path)

    # ── disk ────────────────────────────────────────────────────────────────────────
    def disk_free(self) -> int:
        probe = self.root if self.root.exists() else self.root.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return shutil.disk_usage(probe).free

    def size_on_disk(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())

    def delete(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Registry:
    """Every corpus under one root, and which one the reader is currently searching."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, name: str) -> Path:
        return self.root / slugify(name)

    def get(self, name: str) -> Corpus:
        return Corpus(self.path_for(name))

    def exists(self, name: str) -> bool:
        return self.path_for(name).is_dir()

    def create(self, name: str, goal: str = "",
               text_budget: int = DEFAULT_TEXT_BUDGET) -> Corpus:
        c = Corpus(self.path_for(name))
        if c.recipe_path.exists():
            return c
        c.root.mkdir(parents=True, exist_ok=True)
        c.save(Recipe(name=c.root.name, goal=goal, text_budget=text_budget,
                      created_utc=_now()))
        return c

    def list(self) -> list[tuple[Corpus, Recipe]]:
        if not self.root.is_dir():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            # A directory without a recipe is not a corpus — it is a half-deleted one, or
            # something a user dropped here. Listing it would offer the reader a corpus
            # that cannot be opened.
            if d.is_dir() and (d / RECIPE_NAME).exists():
                c = Corpus(d)
                out.append((c, c.load()))
        return out
