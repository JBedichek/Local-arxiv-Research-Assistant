"""Teaching the embedder this corpus, using the citations already in it.

The signal is free and nobody had to label it: a paper citing a passage is a statement
that the two are related, so :mod:`lara.finetune.pairs` and :mod:`lara.finetune.bags`
turn citation edges into training pairs, and :mod:`lara.finetune.mnrl` and
:mod:`lara.finetune.trainers` fit the embedder to them.
:mod:`lara.finetune.judgements` adds the other free signal — what retrieval was actually
asked for and what was actually used.

**Measured before and after, every time.** :mod:`lara.finetune.evaluate` runs on both
sides of any fine-tune, because a retrieval model that has been trained and not measured
is a retrieval model nobody can say is better.
"""
