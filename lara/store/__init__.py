"""The database. One SQLite file, WAL mode, opened per thread.

Deliberately unremarkable, and deliberately one file. Everything the reader knows —
papers, chunks, embeddings' bookkeeping, reading history, corpora — is in a file you can
copy, and a store you can copy is a store you can back up, move to another machine, or
inspect with any tool that speaks SQL.

WAL because readers and the writer must not block each other: a crawl that runs for days
cannot pause every time somebody asks a question. Opened per thread because SQLite
connections are not shared safely across them, and the alternative is a pool that exists
to work around a rule instead of following it.
"""
