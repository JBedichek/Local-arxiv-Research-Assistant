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

5 folds split by query, 3,520 triples from 440 queries, on the full harvested set:

| | before | after | |
|---|---|---|---|
| pairwise accuracy | 0.8219 ± 0.0328 | 0.8196 ± 0.0296 | unchanged |
| **rank correlation** | **0.7289 ± 0.0287** | **0.6617 ± 0.0512** | **worse** |
| margin error | 0.6068 ± 0.0114 | 0.5166 ± 0.0143 | better |

**This is a negative result, and the shape of it is the informative part.** Margin error is
the quantity MarginMSE directly optimises, and it improved. The two metrics that were *not*
optimised did not: pairwise accuracy moved within noise (up in 3 folds, down in 2), and
rank correlation fell **in all five folds** — a consistency that rules out chance.

So the student is learning to reproduce the *magnitude* of the teacher's score gap while
getting slightly worse at the thing retrieval actually needs, which is ordering. Fitting
the loss is not the same as learning the task.

**Do not adopt a model on these numbers.** The overfit check passing (pairwise accuracy
1.000) established only that the training loop works; k-fold is what answers whether
anything transfers to unseen queries, and here it does not.

Three plausible causes, in the order worth testing:

1. **The objective rewards the wrong thing.** MarginMSE is a regression on score gaps. A
   ranking loss — InfoNCE over in-batch negatives, or a listwise objective — optimises
   order directly, which is what rank correlation measures.
2. **Not enough signal.** 2,816 training triples per fold against a 300 M-parameter encoder
   that already scores 0.82 is a thin prior to move honestly.
3. **Exposure bias in the labels.** Positives are drawn from what retrieval already
   returned, so the data mostly describes orderings the base model already produces. The
   `explore` run's source-recall misses (~45 %) are the part that carries genuinely new
   information, and they are a small fraction of the total.

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
