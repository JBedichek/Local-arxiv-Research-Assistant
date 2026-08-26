"""Getting arXiv onto this machine: metadata, full text, citations, and the parse.

Four resumable crawls, in the order they depend on each other.
:mod:`lara.ingest.oai` harvests metadata over OAI-PMH, checkpointed every page;
:mod:`lara.ingest.fulltext` fetches the documents themselves, politely and adaptively;
:mod:`lara.ingest.parse` turns LaTeXML HTML into anchored sections, blocks and chunks —
the anchors are what let a citation point at a passage rather than a paper; and
:mod:`lara.ingest.citations` enriches the graph from Semantic Scholar.

Every stage is resumable and checkpointed on purpose. These run for days over millions of
documents, and a crawl that has to start again because the process died is a crawl that
never finishes.
"""
