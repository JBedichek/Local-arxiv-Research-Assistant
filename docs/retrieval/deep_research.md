# Deep Automated Research

An iterative retrieve → extract → decide loop that surveys the literature instead of
answering from one search.

Implementation: [`lara/serve/synthesis.py`](../../lara/serve/synthesis.py). Endpoints:
`POST /api/synthesize`, `GET /api/synthesis/runs`, `GET /api/synthesis/run/{run_id}`. UI:
the **Deep Automated Research** button in the top bar, `deepResearch()` in
[`web/app.js`](../../web/app.js).

---

## 1. The problem

*"What is the most sample-efficient Muon-based optimiser in the literature?"* is not
answerable by one retrieval. It needs a survey: find candidates, read them, notice what is
missing, look again, and only then decide what the literature actually says.

```
seed ─▶ [ retrieve ─▶ label + name + extract ─▶ expand ] xN ─▶ consolidate ─▶ answers
```

## 2. Relevance alone re-finds the same cluster

Fusing confirmed chunks back into the query pulls *toward* what is already held, so round
three looks like round two. Four pressures push outward instead:

| pressure | mechanism |
|---|---|
| already seen | chunks in `seen_chunks` are dropped **before** ranking |
| paper dominance | `cap_per_paper(hits, 3)` — a survey of one paper is not a survey |
| near-duplicates | `mmr(..., lambda_=0.7)`: relevance minus max similarity to what is chosen |
| structural reach | every `expand_every`-th round searches within the citation neighbourhood of the confirmed papers instead of the whole corpus |

MMR matters more than it sounds. Retrieval returns the same result restated in the
abstract, the introduction and the conclusion; sending three copies to the extractor costs
three calls and yields one claim.

The citation round (`via="citations"`) collects `cites` + `cited_by` for up to 20 confirmed
papers, 25 neighbours each, minus anything already seen, and passes that as a paper
restriction to the retriever. It finds work that uses different vocabulary for the same
idea, which similarity alone will not surface.

## 3. Extraction is structured, not a relevance flag

Superlative questions invite invention. "Most sample-efficient" has no answer unless
someone measured it, so each kept excerpt is compressed into a row:

```json
{"n": 1, "name": "...", "claim": "...", "method": "...",
 "metric": "...", "value": "...", "condition": "..."}
```

- `name` — 3–6 words, shown as the graph label (capped at 80 chars)
- `claim` — the single most important thing it establishes, in one sentence and **shorter
  than the excerpt** (600)
- `method` (120), `metric` (120), `value` including units (120), `condition` — the setting
  the result holds in: dataset, scale, budget (200)

The prompt states consequences, the same discipline as the hierarchical-scope tagger:

> RELEVANT: the excerpt's embedding becomes a search key for the next round, its claim
> enters the evidence table the final answer is built from, and its name is shown to the
> user in the retrieval graph. A wrongly kept excerpt steers the next round off course and
> puts a false row in the table.

and *"Shared vocabulary is not relevance. An excerpt that merely mentions the topic without
stating a method, result, measurement, limitation or definition is NOT relevant."*

Excerpts are sent numbered, truncated to 1100 characters each, at `temperature=0.0`,
`max_tokens=1400`. Verdicts referencing an out-of-range `n` are dropped.

**A parse failure is not a verdict.** If no JSON array can be found, `extract` returns
`([], [])` — no claims *and no rejections*. Dropping the batch would silently lose a round's
work, and recording it as rejected would poison the training set with judgements the model
never made.

## 4. Termination is not left to the model alone

Models stop early on hard questions and loop on easy ones. Two independent signals, and
saturation overrides the vote.

**Saturation.** A round with `relevant == 0`, or one where MMR selected nothing at all,
increments a dry counter. `dry_rounds` (default 2) consecutive dry rounds ends the run
regardless of what the model wants. A round finding nothing new is evidence the corpus is
exhausted for this query.

**The stop vote.** `should_continue` returns `{decision, gap, next_query}`. It is
deliberately biased toward continuing:

> **Default to continuing.** A survey that stops early is the common failure, and it is the
> expensive one: the answer then reports a fraction of the literature as though it were all
> of it. Another round costs seconds. An incomplete survey is wrong for as long as anyone
> reads it.

and it is told explicitly that looping is already handled elsewhere, so it need only judge
completeness. Two brakes sit on top of its answer:

- `min_rounds` (4) — the vote is ignored entirely while the run is still shallow
- `stop_votes` (2) — after that, it takes **consecutive** stop votes to end the run. Any
  `continue` resets the count.

A `stop` verdict that cannot be parsed also ends the run, tagged `via="fallback"`.

`next_query` is required to name the gap — *"name the method, metric or comparison you
want, not a restatement of the original question"* — and becomes the next round's query. If
it is empty, the round falls back to the original question.

`stopped_because` records which brake fired: `"model voted stop 2x; last gap: …"`,
`"N rounds found nothing new"`, `"N rounds found nothing relevant"`, `"model stopped"`, or
`"cancelled by user"`.

**There is no round cap.** A run ends on saturation or on the vote, not on a counter.

## 5. TLDR is derived from Thorough

Consolidation is two calls, in order:

1. **Thorough** — `temperature=0.2`, `max_tokens=2600`, from the full evidence table.
2. **TLDR** — `temperature=0.1`, `max_tokens=400`, from **the long answer**, not from the
   evidence.

> Generated independently the two can contradict each other, and a reader who notices that
> stops trusting both. Compressing the long answer guarantees the short one asserts nothing
> the long one does not.

The TLDR prompt forbids introducing anything new, requires the `[chunk_id]` citations to be
carried through, and — importantly — says that if the long answer concluded the evidence
cannot settle the question, *say that first rather than picking a winner anyway*.

The Thorough prompt requires a citation on every factual sentence, numbers only where the
table has them with their conditions attached, an explicit refusal to rank where the
evidence does not support one, and a closing **Disagreements** section (or a statement that
none were found).

**Provenance has to survive two hops.** The final answer is chunk → claim → narrative, so
every claim carries its `chunk_id` forward and the answers cite those ids. Without that the
grounding check has nothing to score and citations cannot be rendered.

If no claims were gathered at all, both answers become the literal string *"No relevant
evidence was found in the corpus for this question."* and no model call is made.

## 6. Feedback vectors

Each round passes up to `max_feedback_vectors` (6) full-precision chunk vectors into
`Retriever.retrieve(feedback=...)`, which fuses them as extra ranked lists — the same
mechanism as [hierarchical scope](hierarchical_scope.md#4-confirmed-passages-become-query-vectors).

They are chosen **best-first, not most-recent**: `sorted(claims, key=lambda c: -c.score)`.
A late round should still be steered by the strongest evidence found, not by whatever
happened to arrive last.

## 7. Persistence

Three tables, created on demand by `SCHEMA` in the same SQLite database as the corpus:

| table | contents |
|---|---|
| `synthesis_runs` | `run_id, question, tldr, thorough, stopped_because, n_rounds, n_claims, n_papers, ms, created_utc` |
| `synthesis_rounds` | one row per round: `n, query, retrieved, fresh, relevant, new_papers, via, gap, ms` |
| `synthesis_claims` | the full `Claim` record, indexed by `run_id` |

`save()` **never raises**: the whole body is wrapped, because losing a finished run to a
write error is worse than losing the write.

Every verdict is also recorded as a judgement, with the **same teacher as hierarchical
scope** — `teacher="llm_scope"`, `source="synthesis"`. Claims are label 1, rejected excerpts
are label 0. The rejects are the valuable half: they were retrieved, so they are topically
close, and then judged unhelpful by a model. Those are hard negatives that ordinary usage
does not produce. See
[`../finetuning/gemma_embedder_finetuning.md`](../finetuning/gemma_embedder_finetuning.md).

## 8. The SSE contract

`POST /api/synthesize` with `{"question": "...", "model": null}` returns
`text/event-stream`. Every intermediate result is emitted the moment it exists — a run
whose progress is invisible is indistinguishable from one that has hung.

| event | payload |
|---|---|
| `start` | `{run_id, question}` |
| `round` | `{n, query, via, phase: "retrieving"}` then `{n, phase: "reading", n_chunks}` |
| `claims` | `{round, claims: [Claim, …]}` |
| `round_done` | the full `Round` record |
| `decision` | `{round, decision, gap, next_query, via, stop_votes, needed, forced_continue}` |
| `consolidating` | `{claims, papers}` |
| `token` | `{target: "thorough" \| "tldr", text}` |
| `done` | `{run_id, ms, claims, papers, rounds, stopped_because, tldr, thorough}` |
| `error` | the exception string, truncated to 400 chars |

**Closing the connection does not discard the run.** `asyncio.CancelledError` sets the
cancel flag, the loop exits at the top of the next round, and the run *still consolidates
whatever it gathered* rather than abandoning minutes of retrieval mid-flight. The Stop
button in the UI aborts the fetch, which is the same path.

`GET /api/synthesis/runs?limit=50` lists past runs newest first with a 240-character TLDR
excerpt. `GET /api/synthesis/run/{run_id}` returns one run in full — rounds, claims and
both answers — or 404.

## 9. The UI

`#deep-btn` in the top bar toggles `body.deep-open`, which is a **full-window** view rather
than a pane: a run takes minutes and the reader is not doing anything else meanwhile.
Escape or `#deep-close` returns to the reader.

- **Left** — the retrieval trace. One collapsible `<details>` per round, open by default,
  tagged `similarity` or `citation graph`, showing the query, the counts, and the claims as
  they arrive. Each claim renders as the model's own name for the passage — a real link to
  `/p/{arxiv_id}#chunk-{chunk_id}`, so it opens the exact excerpt it came from — over the
  paper id, section, one-sentence claim, and `metric = value` and condition where they
  exist.
- **Right** — `TLDR` (*"the minimum complete answer"*) and `Thorough` (*"everything found,
  organised"*), both `<details open>`, streamed token by token into whichever the `token`
  event targets.
- **Top** — a `Stop` button while running, a status line, and a `past runs…` dropdown
  populated from `/api/synthesis/runs` that reloads a saved run into the same view.

Answers, claims and excerpts all pass through the inline LaTeX renderer, so
`$O(\epsilon^{-3})$` renders rather than showing as source. Citation markers accept
comma-separated ids (`[123, 456]`).

## 10. Measured

Two real runs against the live corpus, same question — *"What is the most sample-efficient
Muon-based optimizer currently in the literature?"* — read back from
`/api/synthesis/run/{id}`:

| run | rounds | claims | papers | wall | stopped because |
|---|---|---|---|---|---|
| `b8554490cd16` | 3 | 9 | 4 | 60.5 s | `model stopped` |
| `1bcb915e98cf` | 17 | 33 | 12 | 207.4 s | `model voted stop 2x` |

Per-round cost ranges from 0.5 s to 11.8 s; the spread is the model calls, not retrieval —
every round retrieved exactly 24 excerpts (`per_round 12 × over_fetch 4` capped and
diversified down to 12 read).

The `fresh` column shows saturation arriving gradually rather than suddenly. In the
17-round run: 24, 23, 22, 18, 18, 14, 14, 13, 7, 10, 5, 6, 7, 4, 6, 2, 7. By round 16 only
2 of 24 retrieved chunks had not been seen before.

The two runs also illustrate the two honest outcomes. The short one concluded *"The evidence
cannot settle a single winner because 'sample efficiency' refers to distinct theoretical and
empirical metrics"*; the long one named LiMuon with its complexity bound. Neither was forced
to produce a ranking.

## 11. Configuration

```yaml
retrieval:
  synthesis:
    per_round: 12            # excerpts read per round
    over_fetch: 4            # retrieve this many times per_round, then drop seen + diversify
    cap_per_paper: 3         # so one long on-topic paper cannot fill a round
    dry_rounds: 2            # consecutive rounds with nothing new before stopping
    min_rounds: 4            # the model's stop vote is ignored below this depth
    stop_votes: 2            # consecutive stop votes needed to end a run
    max_feedback_vectors: 6
    expand_every: 2          # every Nth round walks the citation graph instead of dense
```

```bash
lara config set retrieval.synthesis.min_rounds 2      # shorter runs
lara config set retrieval.synthesis.per_round 20      # read more per round
```

## 12. Things worth knowing

- **There is no CLI entry point.** Synthesis runs only through `/api/synthesize`; there is
  no `lara synthesize` command.
- **`min_rounds: 4` guarantees a floor on cost, not on quality.** A question the corpus
  cannot answer still burns four rounds unless the dry counter fires first.
- **Rounds alternate `dense` and `citations` by parity**, not by need: `via = "citations"
  if (n % expand_every == 0) else "dense"`. A citation round with no confirmed papers yet
  silently degrades to a plain corpus search.
- **The evidence table shown to `should_continue` is truncated to the last 25 claims**,
  while consolidation sees all of them. A very long run therefore judges completeness
  against a window.
- **Claim scores are the retriever's, not the model's.** `Claim.score` carries the hit's
  fused score forward, which is what makes best-first feedback selection meaningful.
- **`Claim.compression` exists but nothing reads it.** It is `len(claim) / text_len`, there
  for diagnosing an extractor that paraphrases at length instead of compressing.
