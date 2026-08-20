# Fine-tuning the embedder

How we improve `embeddinggemma-300m` on this corpus, what data the system records while
you use it, and how to reproduce or undo any of it.

---

## 1. What gets recorded while you use the reader

**This is the part that is not visible in the UI, so read it first.**

Every search and every question writes rows to the `judgements` table in
`data/meta.sqlite`. Nothing leaves the machine, nothing is sent to any third party, and no
account is involved — but it is recorded silently, by design, because the whole point is to
harvest training signal from ordinary use.

| What | When | Stored |
|---|---|---|
| Your query text, verbatim | every `/api/retrieve` and `/api/ask` | `judgements.query` |
| Which passages were retrieved | same | `judgements.chunk_id` |
| The reranker's relevance score for each | same | `judgements.score` |
| The rank each passage held | same | `judgements.rank` |
| Which citation you clicked | when you click `[1]` or a chip | `teacher='user_click'` |

**Why the click matters.** Everything else in that table is a model's opinion. A citation
you actually followed is the only signal that can *contradict* the teacher rather than echo
it, which makes it the natural held-out check on whether distillation is learning relevance
or merely learning to imitate the reranker.

Deduplication is on `(query_hash, chunk_id, teacher)`, so re-running the same search adds
nothing. The query is stored in full, not hashed — the hash is only for dedup — because
training needs the text.

### Inspecting, exporting, and deleting

```bash
# what has been collected
sqlite3 data/meta.sqlite "SELECT teacher, COUNT(*) FROM judgements GROUP BY teacher;"

# every query you have ever run
sqlite3 data/meta.sqlite "SELECT DISTINCT query FROM judgements ORDER BY created_utc DESC;"

# delete everything harvested from real use, keeping synthetic data
sqlite3 data/meta.sqlite "DELETE FROM judgements WHERE source IN ('ask','search','click');"

# delete all of it
sqlite3 data/meta.sqlite "DELETE FROM judgements;"
```

To turn collection off entirely, remove the two `_capture_judgements` calls in
`lara/serve/app.py` and the `fetch("/api/click", …)` block in `web/app.js`. There is
deliberately no runtime switch: a setting that silently disables data collection is worse
than an explicit code change you can grep for.

---

## 2. Why fine-tune at all

The retriever is a **bi-encoder**: it turns each chunk into 768 numbers *before* your query
exists, then compares vectors. That is what makes searching 18M chunks take 4 ms — and also
what caps its judgement, because it never sees the query and the passage together.

The **cross-encoder** (`bge-reranker-v2-m3`) does read them together and is much better at
relevance, but it can only score a shortlist; running it over the corpus would take hours
per query.

Fine-tuning closes part of that gap by teaching the fast model to imitate the accurate one.
This is standard distillation, and the teacher is already paid for: we score every retrieved
passage with the reranker anyway and were previously discarding the result.

---

## 3. What we train on

Three sources, in increasing order of how hard they are to obtain.

### 3.1 Citation structure (no usage required)

A citation A → B is a human expert asserting that B is relevant to something in A. We have
**2.1M such edges** where both papers are fully indexed.

The obvious way to use them is wrong, and we measured it: anchoring on the paragraph that
*contains* the citation and pairing it with the cited abstract sounds natural, but across
131k extracted contexts **90% of them sit in Introduction, Related Work or Background**.
That trains "related-works prose → abstract", a register nobody types into a search box.

Instead we treat a citation as a **bag-level** label — somewhere in A relates to somewhere
in B — and pair ordinary body chunks from each side, pooling the m×m candidate similarities
with LogSumExp. Both sides then look like what the system actually handles at query time.

### 3.2 Harvested judgements (from your usage)

As described in §1. Free, real, and distribution-matched to how the tool is actually used.

**But usage alone cannot train this**, which we established by measuring: the first three
real queries produced **24 positives and zero negatives**. Everything retrieval returns
scores above threshold, so usage can only teach the model to reorder what it already finds,
never to find what it misses.

### 3.3 Synthetic exploration (the part that fixes the gap)

`lara explore` has the model drive the whole loop: sample a passage, ask it to write a
question about it, retrieve, and judge everything that comes back.

The valuable step is the last one. A question written *from* chunk C has C as a known
positive. **Measured: retrieval misses C about 45% of the time** (source recall@20 ≈ 55%).
Each miss is a labelled relevance the current embedder cannot see — exactly the example
type ordinary usage can never produce.

Six question styles are cycled round-robin (mechanism, quantitative, comparison, limitation,
definition, and "what do you least understand here") because query *type* diversity matters
more to the trained model than raw volume. Self-referential questions — "what do the authors
mean here", "how does Figure 3 support this" — are rejected, since they have no referent
once the passage is gone.

---

## 4. Reproducing the pipeline

```bash
# 1. harvest citation contexts from the crawled HTML (resumable, ~40 min for 200k papers)
lara pairs

# 2. run synthetic exploration — this is where most training data comes from
#    400 questions takes roughly 40 minutes and needs vLLM running
lara explore --n 400 --k 20 --device cuda:2

# 3. check the setup can fit its own data BEFORE training for real (2 minutes)
lara fit-check --mode overfit --n 320 --epochs 4 --batch-size 32

# 4. k-fold cross-validation, split by query
lara fit-check --mode kfold --k 5

# 5. the full citation fine-tune, with before/after retrieval evaluation
lara finetune --device cuda:0 --batch-size 16 --max-edges 200000
```

`lara finetune` **refuses to save a model** unless citation MRR improves *and* paraphrase
MRR falls by less than 0.02. That guard is not decoration: the first run took citation MRR
from 0.386 to 0.045 and paraphrase from 0.747 to 0.114, and adopting it would have meant
hours of re-embedding to make every search substantially worse.

If a model does pass, adopt it by pointing `embedding.model` in `config.yaml` at the saved
directory and re-embedding:

```bash
lara embed --restart          # ~1.7 hours across three GPUs at current corpus size
curl -X POST http://127.0.0.1:8080/api/reload
```

Keep the old vectors until the new index has won on the evaluation. Re-embedding is the
expensive, irreversible-in-practice step.

---

## 5. Training parameters, for someone who has not tuned one before

The single most important thing: **fine-tuning is not training.** The model already knows
how to represent scientific text; you are nudging it. Almost every failure here is from
nudging too hard.

### Learning rate — the one that matters

How big a step to take each update. Ours is `5e-5` for the weight matrices.

We originally used `2e-3`, a reasonable figure for training a model *from scratch*, and it
destroyed the encoder — both evaluation metrics fell ~85% while the loss barely moved. That
combination is the signature of overwriting pretrained knowledge rather than learning the
task.

Rule of thumb: for fine-tuning, start around `1e-5` to `5e-5`. If results get worse, the
learning rate is the first thing to cut, not the last.

> A note specific to Muon, the optimiser we use: it normalises its update per weight matrix,
> so the step size ignores how large the existing weights are. That makes it excellent for
> pretraining and unusually easy to over-drive when fine-tuning.

### Batch size — 64

How many examples per update. Larger means a steadier, less noisy gradient and usually
tolerates a slightly higher learning rate; smaller uses less memory. If you hit
out-of-memory, halve this before touching anything else.

### Epochs — 1 to 4

How many times to pass over the data. With a big dataset, one pass is often enough. Watch
for the loss falling while the *held-out* metrics stop improving — that is memorisation, and
more epochs will make it worse.

### Temperature / margin scale — 10.0

How sharply the model is asked to separate relevant from irrelevant. Rarely worth touching.

### Sequence length — 320 tokens

How much of each chunk the model reads while training. Our chunks average 219 tokens, so
320 covers most of them; raising it costs memory quadratically for little gain.

### How to tell whether it worked

Run `lara fit-check --mode overfit` **first**, always. It trains and evaluates on the *same*
few hundred examples, which is meaningless as a quality measure and invaluable as a
plumbing check: a healthy setup drives pairwise accuracy toward 1.0. If it cannot fit data
it has already seen, the recipe is broken and more data will not help.

Our current state after the fix:

| | before | after |
|---|---|---|
| pairwise accuracy | 0.963 | **1.000** |
| margin error | 0.574 | **0.169** |
| rank correlation | 0.377 | 0.519 |

Then `--mode kfold` for whether it generalises, and finally `lara finetune`, which is the
only one that reports the metrics you actually care about: retrieval quality against
**human** citation judgements.

### What k-fold actually said (2026-08-18)

5 folds split by query, 3,520 triples from 440 queries, on the full harvested set. Three
runs, and the differences between them are the useful part:

| | base | batch 64, 3 ep | **batch 128, 3 ep** | batch 128, 12 ep + stop |
|---|---|---|---|---|
| pairwise accuracy | 0.8219 | 0.8196 *(flat)* | **0.8526** | 0.8472 |
| rank correlation | 0.7289 | 0.6617 *(5/5 worse)* | **0.7243** | 0.7128 |
| margin error | 0.6068 | 0.5166 | 0.5279 | 0.5144 |

**The first run was a negative result, and its shape said why.** Margin error is what
MarginMSE directly optimises, and it improved. The metrics that were *not* optimised did
not: pairwise accuracy moved within noise, and rank correlation fell **in all five folds** —
too consistent for chance. The model was reproducing the *magnitude* of the teacher's score
gap while getting worse at ordering, which is what retrieval needs. Fitting the loss is not
learning the task.

**Doubling the batch to 128 fixed it.** Pairwise accuracy now improves in **5 of 5 folds**
(+0.031 overall) and rank correlation no longer collapses. Fewer, better-conditioned
updates — 54 steps instead of 132 — drift less far from the pretrained weights.

**Early stopping did not help, and the reason is worth knowing.** With `epochs=3` it never
fires: validation loss is still falling at the cap. Raising the cap to 12 makes it fire
(around step 60–70 of 216) and the result gets *worse* despite training longer — because
the cosine LR schedule is sized to the epoch cap, so stopping mid-schedule leaves the model
at a near-peak learning rate. Measured: the 3-epoch run anneals to 1.3e-6 by its last step;
the 12-epoch run stops with LR still at 4.4e-5. An unannealed checkpoint loses more than
the extra steps gain.

> **Practical advice.** Set `--epochs` to a realistic number so the schedule anneals
> properly, and treat `--patience` as a safety net against over-training rather than the
> primary control. Cosine annealing and early stopping are two mechanisms for the same job;
> mis-sizing one to serve the other costs accuracy.

Early stopping selects on an *inner* split carved from the training folds, never on the
held-out fold — choosing a checkpoint with the data you then report on would leak and make
the numbers optimistic by an unknown amount.

**This still is not a model worth adopting.** Pairwise accuracy improved, but rank
correlation is at best unchanged and `lara finetune` measures the thing that actually
matters — retrieval against human citation judgements — which none of these runs tested.
Three things worth trying next, in order:

1. **A ranking objective.** MarginMSE is a regression on score gaps. InfoNCE over in-batch
   negatives, or a listwise loss, optimises order directly — which is what rank correlation
   measures and what the first run degraded.
2. **More signal.** 2,392 training triples per fold against a 300 M-parameter encoder that
   already scores 0.82 is a thin basis for moving it honestly.
3. **Exposure bias in the labels.** Positives come from what retrieval already returned, so
   the data mostly describes orderings the base model already produces. The `explore` run's
   source-recall misses (~45 %) carry the genuinely new information and are a small
   fraction of the total.

---

### The learning-rate sweep at batch 512 (2026-08-19)

The obvious objection to the k-fold result was that the learning rate was wrong: 5e-5 was
chosen by backing away from a 2e-3 run that destroyed the encoder, not by measurement. So
the whole range was swept at a much larger batch, with early stopping deciding the length
of each run.

`lara lr-sweep` — 14,041 triples from 450 queries (`--max-per-query 32`), batch **512**
reached by gradient accumulation over micro-batches of 64, Muon, cosine schedule, up to 4
epochs, early stopping on an inner query-split. Scored on 25 % of queries held out from
every run. `lr_adam` scales with `lr_muon` at the recipe's 5:1 ratio, so the sweep varies
one thing.

Baseline on the held-out queries: pair_acc **0.9066**, spearman **0.6977**, margin_mae
**0.5670**.

| muon lr | steps | early stop | pair_acc | spearman | margin_mae |
|---|---:|---|---|---|---|
| 1e-5 | 68 | no | 0.9106 (+0.0040) | **0.6994 (+0.0017)** | 0.5334 (−0.0335) |
| 3e-5 | 68 | no | 0.9128 (+0.0062) | 0.6933 (−0.0044) | 0.4832 (−0.0838) |
| 1e-4 | 45 | yes | 0.8930 (−0.0136) | 0.6631 (−0.0346) | **0.4430 (−0.1240)** |
| 3e-4 | 25 | yes | 0.9089 (+0.0023) | 0.6801 (−0.0176) | 0.4474 (−0.1196) |
| 1e-3 | 20 | yes | **0.9143 (+0.0076)** | 0.6859 (−0.0118) | 0.4568 (−0.1102) |
| 3e-3 | 20 | yes | 0.8744 (−0.0323) | 0.6079 (−0.0898) | 0.4646 (−0.1024) |

**No learning rate makes the model better at ranking.** Rank correlation fell at five of
six, and the one that rose did so by +0.0017 — against a fold-to-fold standard deviation of
±0.0287 measured in the k-fold run, so about a sixteenth of the noise. It is also the
lowest rate in the sweep, the one that moves the weights least. Pairwise accuracy's best
showing, +0.0076, sits inside a ±0.0328 spread the same way.

Margin error improved at **every** rate, monotonically with LR up to 1e-4 and then
plateauing. That is the quantity MarginMSE minimises.

So the shape of the k-fold result survives a 4× larger batch and two and a half orders of
magnitude of learning rate: **the optimised quantity improves and the ranking metrics do
not.** That rules out the learning rate as the explanation, which is what this sweep was
for. The remaining candidate is the one listed first above — the objective is a regression
on score gaps, and ranking is not what it optimises.

Two details worth carrying forward:

- **3e-3 damages the encoder** (−0.0323 pair_acc, −0.0898 spearman) even at batch 512. The
  original 2e-3 failure was not purely a small-batch artefact; Muon's per-matrix update
  normalisation makes high rates dangerous on a pretrained encoder regardless of batch.
- **Early stopping fired at four of six rates**, always at the higher ones, and always well
  before the cosine schedule annealed. The `epochs`-as-cap warning above still applies: the
  1e-3 run stopped at step 20 of 68 with the LR still near peak.

These numbers are not directly comparable to the k-fold table above. `--max-per-query 32`
admits more pairs per query than the default 8, which raises the baseline pair_acc from
0.82 to 0.91 by including easier pairs. Compare shapes, not absolute values.

---

### More data, whitening, and a metric that measures the right thing (2026-08-20)

Three things changed at once, each addressing a specific objection to the earlier result:

1. **6.7x more data.** A topic-focused `explore` run over Muon, spectral optimization,
   equilibrium propagation, RAG and embeddings took the judgement set from 11,789 to
   **20,351 across 788 queries**, and the triple set from 3,599 to **24,297 from 772
   queries**. Source-chunk recall on that run was 48 %, so 365 of the new positives are
   passages retrieval could not find at all.
2. **Whitening**, fitted on 400 k corpus vectors — no retraining, one 768x768 matrix.
3. **Within-query rank correlation**, because the pooled figure was measuring the wrong
   comparison (see §5).

5 folds, split by query, batch 512, Muon 3e-5, early stopping. Mean +/- std over folds:

| | pair_acc | pooled rho | within-query rho |
|---|---|---|---|
| base, raw | 0.9124 ± 0.0089 | **0.6697 ± 0.0318** | **0.4544 ± 0.0210** |
| base, whitened | **0.9303 ± 0.0068** | 0.6151 ± 0.0322 | 0.4016 ± 0.0285 |
| tuned, raw | 0.9099 ± 0.0072 | 0.6600 ± 0.0278 | 0.4372 ± 0.0183 |
| tuned, whitened | 0.9190 ± 0.0066 | 0.6235 ± 0.0323 | 0.4092 ± 0.0227 |

Per-fold deltas, which say more than the means:

| intervention | pair_acc | pooled rho | within-query rho |
|---|---|---|---|
| fine-tune | −0.0025 (1/5 folds better) | −0.0097 (1/5) | **−0.0172 (0/5)** |
| whitening | **+0.0179 (5/5)** | −0.0545 (0/5) | −0.0528 (0/5) |
| fine-tune *on top of* whitening | −0.0113 (0/5) | | |

**The fine-tune fails again, and "not enough signal" is now ruled out.** That was the
second of the three candidate causes listed above. Nearly seven times the data, a properly
swept learning rate and a four-times larger batch produce a model that is *worse* on every
metric that matters: within-query rank correlation degraded in **five folds out of five**.
Margin error meanwhile improved from 0.5625 to 0.3823 — the objective is being fitted
harder than ever. Fitting the loss is still not learning the task, and the remaining
explanation is the first one: MarginMSE regresses on score gaps, and ranking is not what it
optimises.

**Whitening is the one thing that worked, and it did so unanimously.** +0.0179 pairwise
accuracy in 5 folds out of 5, roughly four times the fold-to-fold spread, for a transform
that costs no training and no re-embedding. It also lowers both rank correlations in 0/5
folds — consistently — which is the expected consequence of rescaling a geometry whose gaps
the teacher's scale was fitted to.

**The two do not compose.** Applying the fine-tune on top of whitening *costs* 0.0113
pairwise accuracy, in five folds out of five. Whitening alone is the best configuration
measured.

**Do not adopt whitening on these numbers either — not yet.** Every metric on this page is
agreement with the cross-encoder that produced the labels, so "whitening improves pair_acc"
means "whitening agrees more with the reranker", which is exactly the circularity §6 warns
about. The independent test is the citation retrieval eval in `lara/finetune/evaluate.py`,
scored against what human authors actually cited, and it has not been run on a whitened
index. That is the next measurement, and it is cheap: whitening is post-hoc, so it needs no
re-embedding, only re-scoring.

---

## 6. Guarding against fooling yourself

The generator, the judge and the student are all the same family of model, so this pipeline
distils one model's notion of relevance rather than discovering ground truth. That is
tolerable only because the **evaluation is independent**:

- **Train** on synthetic and harvested judgements — a model's opinion.
- **Validate** on citation retrieval — what human authors actually cited.

A student that merely learns to flatter its teacher will not move the citation number. If it
does move, something real was learned.

The second guard is the paraphrase task, which the fine-tune does *not* optimise. It exists
to catch catastrophic forgetting: a model that wins on citations while losing there has
traded away general retrieval for a narrow skill, and would make the product worse while the
headline metric improved.
