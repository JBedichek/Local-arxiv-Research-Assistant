"""Retrieval: finding the passages an answer can rest on, out of tens of millions.

Hybrid and staged, because no single method survives this scale. :mod:`lara.index.search`
runs dense search on the GPU alongside BM25, fuses the two, and reranks the survivors with
a cross-encoder — cheap and broad first, expensive and precise last.
:mod:`lara.index.retrieve` is the one call that assembles those stages, and is what
everything outside this package should use.

Underneath: :mod:`lara.index.vectors` is append-only flat files rather than a vector
database, :mod:`lara.index.embed` fills them resumably, and :mod:`lara.index.scope` keeps
the relevant fraction resident in RAM so the rest can stay on disk without the search
noticing. :mod:`lara.index.accumulate` closes the loop by making the machine's own
finished research retrievable alongside the papers.
"""
