# Hierarchical scope

How a question asked while reading a paper is answered from that paper first, and how the
search widens only when it has to.

Implementation: [`lara/serve/hierarchy.py`](../../lara/serve/hierarchy.py). Feedback
retrieval: `Retriever.retrieve(feedback=...)` in
[`lara/index/retrieve.py`](../../lara/index/retrieve.py).

---

## 1. The problem

Someone reading *Attention Is All You Need* and asking *"why is the attention scaled by
1/√d_k?"* means **in this paper**. Searching all 29.5 M chunks answers a question they did
not ask: the paragraph two sections down gets buried under a hundred topically similar
passages from papers they have never opened.

The reverse is also true. *"What is the best learning rate schedule for LLMs?"*, asked
while the same paper is open, is not about that paper at all. It merely happens to mention
one.

Neither a fixed "search this paper" nor a fixed "search everything" is right. The scope is
a property of the question, so a model decides it.

## 2. The three tiers

```
1. paper           the open paper's own chunks
2. neighbourhood   it, plus everything it cites and everything citing it   (7.2 M edges)
3. corpus          all 29.5 M chunks across 377 k papers
```

The middle tier earns its place. *"How does this compare to prior work?"*, *"is this
novel?"*, *"who else found this?"* are answered by the papers this one cites — and jumping
straight from one paper to the whole corpus skips the tier most likely to hold the answer
while adding the most noise.

Escalation may skip a tier. A general question goes `paper → corpus` directly, and does so
because the model judged the neighbourhood irrelevant, not because of a rule.

## 3. Two decisions

### 3.1 Tagging: which passages actually helped

After the in-paper search, the model marks each excerpt keep or drop. This is not the
cross-encoder's job — that scores *topical relatedness*, and is already used to rank the
candidates. The model is asked something the cross-encoder cannot judge: **does this help
answer the question**, including the case where a passage defines a term the question uses
without answering anything itself.

### 3.2 Escalation: how much wider to look

`stop`, `neighbourhood`, or `corpus`. Short-circuited when the cheap signal is already
conclusive: if reranker scores say coverage is `full`, there is nothing for a wider search
to add and no model call is made.

### 3.3 Prompts state consequences, not just questions

This is the design decision that matters most, and it is deliberate.

A model asked *"is this excerpt relevant?"* optimises for topical similarity — the thing it
is easiest to be confidently wrong about. The tagging prompt instead says what the decision
*causes*:

> **KEEP**: the excerpt is shown to the model that writes the answer, AND its embedding
> becomes an additional search key used to find related passages in other papers. A kept
> excerpt about the wrong subtopic will drag that wider search off course.

Now the model is weighing whether that passage would make a good *search key*, which is the
actual question being asked of it. The escalation prompt likewise names each scope's cost
and its characteristic failure — that `corpus` is "the most likely to return passages that
are topically similar but about a different setting", and that refusing to widen when
evidence is absent "produces a confident answer with nothing behind it".

## 4. Confirmed passages become query vectors

Every kept chunk's full-precision vector is carried into the wider search as an extra
query. This is pseudo-relevance feedback with the assumption removed: classic PRF assumes
the top-k are relevant and is wrong roughly half the time, whereas here a model has checked.

### 4.1 Fused, never averaged

The vectors are **not** averaged into a centroid. Normalised embeddings from different
subtopics average to a point close to neither — the classic failure of centroid query
expansion. Instead each confirmed vector runs its **own** dense search, and the ranked
lists join the existing reciprocal rank fusion alongside the question and BM25:

```
{dense, feedback0, feedback1, …, bm25}  ──RRF(k=60)──▶  candidates
```

A fusion of ranked lists cannot land nowhere. Each search costs ~1 ms against the GPU
index, so the whole mechanism is close to free.

### 4.2 Feedback widens recall only

Tier-2 exact rescoring and the cross-encoder still score against **the question alone**.

A confirmed passage can therefore surface a candidate but can never reorder the final
answer. That asymmetry is the structural defence against query drift: without it, feedback
from a narrow subtopic would gradually pull the ranking toward that subtopic and away from
what was asked.

### 4.3 Capped

Four vectors by default. Past a handful they stop adding recall and start flattening the
ranking, because every list contributes the same reciprocal weight regardless of how good
it was.

## 5. Sizing the in-paper search

```python
paper_k(n_chunks, frac=0.2, min_k=5, max_k=30)
```

A bare proportion misbehaves at both tails. Papers average 78 chunks but run from 15 to
300+, so 0.2 alone yields 3 chunks for a workshop paper — too few to find anything — and 60
for a survey, more than fits comfortably in one tagging call.

| paper size | k |
|---|---|
| 15 chunks | 5 *(floor)* |
| 46 chunks | 9 |
| 78 chunks | 16 |
| 300 chunks | 30 *(ceiling)* |

## 6. Measured

Against the live corpus with `Qwen3.8-27B-FP8` deciding, paper `1706.03762`:

| question | path | model's reason | total |
|---|---|---|---|
| why is attention scaled by 1/√d_k? | `paper` | *"the paper explicitly explains the scaling"* | 2.1 s |
| how does this compare to prior work on recurrence? | `paper` | coverage full *(scores, no model call)* | 1.5 s |
| what is the best LR schedule for LLMs? | `paper → corpus` | *"asks for the best schedule generally, not just this paper"* | 1.9 s |

The second stops on the cheap signal alone. The third skips the neighbourhood deliberately
and returns 2023–24 LLM papers.

## 7. What is recorded, and why the rejects matter

| decision | destination |
|---|---|
| keep / drop per chunk | `judgements`, `teacher='llm_scope'`, `source='hierarchy'` |
| tier choice | `scope_decisions` table |

**The dropped passages are the valuable half.** They were retrieved — so they are
topically close — and then judged unhelpful. Those are hard negatives, and ordinary usage
has never produced them here: the first real queries against this system yielded *24
positives and zero negatives*, because everything retrieval returns scores above threshold.
A model that only ever sees positives can learn to reorder what it already finds, never to
find what it misses.

**Only genuine model verdicts are recorded.** When parsing fails, the fallback keeps the
reranker's top pick — a sensible answer, but not a model judgement, and recording it as one
would poison the training set with exactly the signal distillation is meant to improve on.
Fallback decisions are logged to `scope_decisions` with `via='fallback'` and excluded from
`judgements`.

## 8. Degradation

Every failure returns a plain corpus search rather than an error:

| condition | behaviour |
|---|---|
| no paper open | corpus search |
| paper has no embedded chunks | corpus search, noted |
| `hierarchy.enabled: false` | corpus search |
| model unreachable | reranker's top 3 kept, no escalation |
| unparseable tag verdict | keep the top-ranked excerpt |
| unparseable escalation verdict | **widen one step** |

The last is deliberately asymmetric. A parse failure must not silently become "the paper
was enough" — that turns a model hiccup into a confidently under-evidenced answer.

## 9. Configuration

```yaml
retrieval:
  hierarchy:
    enabled: true
    paper_frac: 0.2          # of the paper's chunks to consider...
    paper_min_k: 5           # ...clamped, since papers run 15 to 300+ chunks
    paper_max_k: 30
    tag_with_model: true     # false = keep the reranker's top 3, no LLM call
    max_feedback_vectors: 4
    allow_neighbourhood: true
    allow_corpus: true
```

Change without editing YAML:

```bash
lara config set retrieval.hierarchy.paper_frac 0.3
lara config set retrieval.hierarchy.tag_with_model false   # cheaper, no tagging call
```

## 10. Things worth knowing

- **`tag_with_model: false` is a real option, not a degraded one.** It removes one LLM
  round trip and keeps the reranker's top 3. On a laptop generator that is the difference
  between 1.5 s and 4 s.
- **The tagging call dominates the latency.** Retrieval at every tier is 140–350 ms; the
  model calls are the rest. Running the wider search speculatively in parallel with the
  tagging call would hide almost all of it — not yet implemented.
- **A highlighted passage is already a confirmed chunk.** Tier 1 could be skipped entirely
  when the user has selected text — also not yet implemented.
- **`walk()` is standalone.** It is not yet wired into `/api/ask`; that endpoint still uses
  the flat agent loop.
