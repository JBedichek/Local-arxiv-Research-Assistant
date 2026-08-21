# Topic-scoped residency

Keeping the fraction of the corpus you actually care about in RAM, and why that is not the
same as deleting the rest.

Implementation: [`lara/index/scope.py`](../../lara/index/scope.py). Design decision D22 in
[`PLAN.md`](../../PLAN.md). Commands: `lara corpus scope`, `scope-status`, `unscope`.

---

## 1. The problem

The tier-1 index costs `n_chunks × dim_truncated` bytes and is loaded before anything else,
which makes it the binding memory constraint on a small machine — not the generator
everyone worries about. At 28.7 M chunks it is 7.3 GB int8 and 14.7 GB fp16, and no laptop
is going to hold that alongside a model. See
[`search_backends.md`](search_backends.md) for what each backend costs.

So let the reader say what they care about, and keep only that slice resident:

```
score(paper) = max over topics of cos(paper_vector, topic_vector)
keep         = top `fraction` by score, unioned with 1-hop citation expansion
resident     = chunks of kept papers
```

## 2. Nothing is deleted

This is a **load-time decision**. Three properties make it safe rather than lossy.

**BM25 is the safety net, and it is free.** Retrieval already fuses dense with lexical, and
FTS5 covers every chunk on disk regardless of residency. An off-topic question still
matches lexically and reciprocal rank fusion folds it back in. The failure mode is reduced
*semantic* recall outside the topic, not blindness. Tier-2 exact rescoring is also
whole-corpus.

**The knob is cheap.** Re-scoring and re-slicing costs seconds and needs no re-embedding —
unlike `dim_truncated`, which invalidates every vector.

**Promotion on demand.** Tier 0 pins the open paper, so following a citation out of the
kept set still works, and a dropped paper opens normally.

## 3. max over topics, not mean

A paper perfectly on one of three interests must not be penalised for ignoring the other
two. The mean would rank a mediocre generalist above a perfect specialist, which is
backwards for a corpus you are pruning by interest. `score_papers()` reduces with
`.max(axis=1)` over the topic vectors, blocked at 200 k papers so a large corpus never
needs a float32 copy of the whole paper matrix at once.

## 4. Citation expansion

A paper cited by many kept papers is worth keeping even when its abstract does not match.
*"Adam: A Method for Stochastic Optimization"* scores near zero against "data selection for
LLMs" and you obviously want it. **Topical similarity finds what a field talks about; the
citation graph finds what it is built on.** The edge table is already there, so this costs
one query.

```sql
SELECT c.dst, COUNT(*) AS n
FROM citations c JOIN _scope_seed s ON c.src = s.arxiv_id
WHERE c.dst NOT IN (SELECT arxiv_id FROM _scope_seed)
  AND EXISTS (SELECT 1 FROM papers p WHERE p.arxiv_id = c.dst AND p.n_chunks > 0)
GROUP BY c.dst HAVING n >= ?
ORDER BY n DESC
```

Restricted to papers that are themselves in the corpus: an edge to something never indexed
has no vectors to make resident. `expand_min_citations: 0` disables expansion entirely.
Papers arriving this way are tagged `via="citation"` and their "score" field holds the
citation count, not a similarity — `scope-status` reports the two counts separately for
that reason.

## 5. The preview, and why it exists

```bash
lara corpus scope -t "data selection for LLMs" -t "optimizer theory" --keep 0.1 --preview
```

`--preview` is the default when neither `--preview` nor `--apply` is given: the command
defaults to the safe half. It prints three tables — the best matches, the last few papers
kept, and the first few dropped — with titles and scores.

> A knob whose effect you cannot see is a knob you cannot set.

A similarity-ranked cut is not a clean topical boundary. Title+abstract embeddings are
decent but the tail is fuzzy, so showing what is about to be dropped is what makes the
fraction meaningful. `cut_preview(window=8)` is what produces those edge tables.

## 6. Applying it

```bash
lara corpus scope -t "data selection for LLMs" --keep 0.1 --expand 3 --apply
lara corpus scope-status
lara corpus unscope
```

| flag | default | meaning |
|---|---|---|
| `--topic` / `-t` | – | repeatable; **at least one is required** |
| `--keep` | 0.1 | fraction of papers to keep resident, in (0, 1] |
| `--expand` | 3 | also keep papers cited by ≥ N kept papers; 0 disables |
| `--preview` | on unless `--apply` | show the cut without writing |
| `--apply` | – | write the keep-set |
| `--device` | auto | device to embed the topics on |

Applying writes two files under `<disk.root>/scope/`:

```
scope.json   topics, fraction, expand_min_citations, created_utc, corpus counts,
             dim_truncated, and every kept paper with its score and via
rows.npy     the sorted int64 array of resident vector rows
```

`rows.npy` is **sorted ascending** because `DenseIndex` binary-searches it to translate
global rows to local indices. `_prepare_rows` re-sorts defensively if it is handed anything
else.

Neither `scope` nor `unscope` restarts anything. **Restart the reader, or `POST
/api/reload`, to apply it** — the tier-1 tensor is a snapshot taken at load time.

## 7. It builds itself from the config

`lara setup` records the topics and the keep fraction in the config rather than printing a
command and trusting the user to run it:

```yaml
corpus:
  scope:
    topics:
      - data selection for LLMs
      - optimizer theory
    keep: 0.05
    expand_min_citations: 3
```

`scope.ensure()` runs on server start (from `AppState`, logging with a `[scope]` prefix)
and makes those real. It compares the saved keep-set against the config: a keep-set built
from *different* topics, a different fraction or a different expansion threshold is stale
and gets rebuilt; one that matches is reused, because building costs a model load and a
pass over every paper vector.

**It is never fatal.** A corpus that has not been embedded yet, or a missing model, means
no scope — which is the same state as not having configured one, and the whole corpus stays
resident. The reason is logged, not raised.

`corpus.scope` is not present in the tracked `config.yaml`. It appears only when `lara
setup` writes it, or when you set it yourself.

## 8. The two ways to scope, and how they interact

| route | what it writes | when it runs |
|---|---|---|
| `lara setup` | `corpus.scope` in `config.local.yaml` | keep-set built on next server start |
| `lara corpus scope --apply` | `<disk.root>/scope/` directly | immediately; applied on next restart |

They are not independent. If `corpus.scope` is set in the config, `ensure()` will
**overwrite** a hand-built keep-set on the next start whenever the config's topics and
fraction do not match what is on disk. To pin a hand-built scope, either unset
`corpus.scope` or make it match.

`lara corpus unscope` deletes the two files; if `corpus.scope` is still configured, the
next start rebuilds them. Unset the config key as well to make unscoping stick:

```bash
lara config unset corpus.scope
lara corpus unscope
```

## 9. What it costs

`Scope.resident_bytes(dtype_bytes=2)` is `n_rows × dim_truncated × dtype_bytes`, with 2 for
the current fp16 residency. `lara corpus scope --apply` prints it against the unscoped
figure, and `scope-status` reports it for the active keep-set.

`lara setup`'s slider is driven by the same arithmetic through `IndexOption.index_gb()`, so
the number the wizard shows and the number the command prints agree. `lara setup` offers
these steps:

```
0.01  0.02  0.05  0.10  0.15  0.20  0.25  0.33  0.50  0.66  0.75  0.90  1.00
```

At `keep=0.05` the current corpus is ~1.4 M chunks, where torch fp16 needs 0.7 GB and faiss
HNSW 1.8 GB — both single-digit milliseconds.

## 10. Things worth knowing

- **`lara corpus scope` requires paper-level vectors.** It scores against
  `paths.papers_int8`, not chunk vectors. Without them it exits with *"no paper-level
  vectors — run `lara embed-papers` first"*.
- **The CLI and the server score against the same vectors.** Both go through
  `scope.inputs()`, precisely so the two cannot drift into scoring against different
  embeddings.
- **`--keep` is a fraction of *papers*, not of chunks.** The resulting chunk fraction
  differs because paper lengths vary and citation expansion adds papers on top;
  `scope-status` prints the realised chunk count and percentage.
- **The keep-set is a snapshot of paper ids, not a live query.** Papers crawled after it
  was built are not resident until it is rebuilt.
- **The whole-corpus row map does not shrink.** `vector_row → chunk_id` covers every row
  regardless of residency, at 4 bytes each. It is the one resident cost the keep fraction
  cannot touch.
- **`/api/health` reports the active scope** in its `scope` field — `null` when the whole
  corpus is resident, otherwise `{topics, fraction, papers, resident_chunks,
  corpus_chunks, resident_gb}`. A scoped index is not a broken one, but a reader deserves
  to know that semantic recall is narrowed to their topics.
