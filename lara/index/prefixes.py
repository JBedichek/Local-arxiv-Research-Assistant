"""The exact strings the embedder sees, and the only copy of them.

EmbeddingGemma is asymmetric: a query and a document are prefixed differently, and the
prefix is part of what was trained. So these have to be byte-identical everywhere -- a
corpus embedded with one prefix and searched with another retrieves nothing, and that is
not a subtle degradation, it is a silent one.

They have drifted before. A copy in config.yaml and a copy in the fine-tuning code went
their own ways, and training ran against "title: none" while ingest used the real title.
This module exists so there is nowhere for a second copy to live.

Deliberately dependency-free -- it is three functions over strings. Importing it must not
cost torch, because the data-preparation code in lara/finetune reads it and has no other
reason to load a model.
"""

from __future__ import annotations

QUERY_PROMPT = "task: search result | query: "


def document_text(title: str | None, section_title: str | None, text: str) -> str:
    """Build the contextual document string. Mirrors EmbeddingGemma's native format."""
    head = (title or "none").strip()
    section = (section_title or "").strip()
    if section and section.lower() not in head.lower():
        head = f"{head} > {section}"
    return f"title: {head} | text: {text}"


#: The two document prefixes every trainer and evaluator needs, derived from the one
#: function that defines the corpus format rather than retyped. Four hand-written copies
#: had already drifted — kfold used "title: none", train used "title: {title}" — and
#: judgements.py and kfold.py both record what a train/serve format mismatch cost when it
#: last happened: 0.909 -> 0.741 cosine. Deriving them makes that drift impossible.
DOC_PREFIX_UNTITLED = document_text(None, None, "")      # "title: none | text: "


def doc_prefix_for(title: str | None) -> str:
    """The document prefix for a paper-level document with this title."""
    return document_text(title, None, "")
