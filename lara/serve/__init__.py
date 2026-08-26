"""The reader as a service: the HTTP surface, the warm state behind it, and generation.

Three things share this package because they share a process. The routes in
:mod:`lara.serve.routes` are thin; :mod:`lara.serve.state` holds everything expensive —
the index, the embedder, the reranker — loaded exactly once and kept warm, because loading
them per request would cost more than answering it; and :mod:`lara.serve.generate` and
:mod:`lara.serve.generator` own the model, from prompt assembly to the process that serves
it.

The long-running work lives here too rather than in a worker: :mod:`lara.serve.synthesis`
runs deep research as an iterative retrieve-extract-consolidate loop, and
:mod:`lara.serve.thread` is what lets a follow-up question know what "it" refers to. Both
need the warm state, and moving them out would mean loading a second copy of it.
"""
