"""Building a corpus out of a request, with the decisions a person would have made.

The reader asks for a subject in its own words; this turns that into a searchable index
and shows its working. :mod:`lara.corpus.search` finds candidates on the open web with no
account and no API key, :mod:`lara.corpus.fetch` turns each into text and a hash,
:mod:`lara.corpus.validate` decides whether it belongs, and :mod:`lara.corpus.licence`
says in plain terms what the document permits.

**A corpus is a directory, not a database row.** :mod:`lara.corpus.store` keeps each one
self-contained alongside the recipe that rebuilt it — so what a corpus contains, and why
each thing is in it, survives being copied to another machine.
"""
