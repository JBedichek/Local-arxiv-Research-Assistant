# UI controls — technical reference

What each control changes in the pipeline, with the measurements behind the defaults.
Companion to `guide_layman.md`, which covers the same surface without the internals.

---

## Retrieval pipeline

Every question runs the same stages. Latencies are warm medians at ~18.5M chunks:

```
query ──embed (20ms)──┬─▶ tier 1: exact GPU matmul over MRL-256      ─▶ 200
                      │      0.8ms, 100% recall, 5.7GB VRAM
                      └─▶ BM25 over FTS5 (IDF-planned)               ─▶ 200
                                    │  70-180ms
                        reciprocal rank fusion (k=60)
                                    │
                    tier 2: exact fp16-768 rescore from mmap  0.2ms  ─▶ 48
                                    │
                    cross-encoder Qwen3-Reranker-0.6B-seq-cls 33ms   ─▶ k
```

Tier 1 is exact, not approximate. Measured against real vectors, a GPU matmul beats CPU
HNSW on every axis that matters — 100% recall at 9.5 ms projected to 16M vectors, versus
96.3% for HNSW ef256 — and needs no index build, which matters because the crawler appends
continuously. `faiss-gpu` was unusable here regardless: no sm_120 kernels.

---

## Top bar

### Input box

Dispatches on a regex: `\d{4}\.\d{4,5}` or the old `hep-th/9901001` form opens a paper via
`/api/paper/{id}`; anything else calls `/api/search`.

### Scope

Sets `papers` on the retrieval call, which restricts **both** halves of the hybrid. This was
a bug once — the filter reached only BM25, so "this paper" searches returned whole-corpus
dense hits.

| value | effect |
|---|---|
| `corpus` | no restriction |
| `paper` | `arxiv_id = current` |
| `neighbourhood` | current ∪ 1-hop citation graph, both directions |

Scoping resolves to vector row ids and filters the tier-1 matmul, so a scoped search is
*cheaper* than a global one, not more expensive.

### Model / quantization

`/api/models` scans the HF cache, admitting a repo only if it has a `config.json` with a
vLLM-supported architecture, resolved weight blobs above a size floor (this is what excludes
`tiny-random` stubs and metadata-only snapshots — 209 cached repos, 8 servable), and is not
GGUF-only.

The endpoint also queries the running vLLM for its loaded model and the UI defaults to it.
Selecting anything else 404s at generation time; switching requires restarting vLLM
(`policy: single_resident`).

Quantization is **not** a free dial. AWQ/GPTQ/compressed-tensors are baked into the
checkpoint. What is genuinely runtime-selectable is weight-only FP8 and bitsandbytes, so the
control offers the checkpoint's native quant plus those.

### ＋ Download new model

`POST /api/model/resolve` does a metadata-only lookup returning parameter count,
architecture, quantization, size and a fit calculation (`size × 1.35` for KV cache and
activations, against VRAM or unified RAM). `POST /api/model/download` runs
`snapshot_download` in a thread; progress is polled from bytes-on-disk rather than a
callback, which cannot drift out of sync with reality.

Two API quirks handled: `usedStorage` sums every quantization variant in a GGUF repo (707 GB
reported for an 8B model), and HF returns 401/403 for nonexistent *and* private repos alike,
so a typo cannot be distinguished from a gated model.

---

## Answer depth

One dial over `lara/serve/agent.py:SPECTRUM`:

| name | rounds | k | candidates | expand | clarify | decompose | budget | measured |
|---|---|---|---|---|---|---|---|---|
| instant | 1 | 5 | 100 | ✗ | ✗ | ✗ | 20s | ~5s |
| fast | 1 | 8 | 200 | ✗ | ✗ | ✗ | 30s | ~5s |
| balanced | 2 | 8 | 200 | ✓ | ✗ | ✓ | 60s | ~6s |
| thorough | 3 | 12 | 300 | ✓ | ✓ | ✓ | 120s | ~7-10s |
| exhaustive | 5 | 20 | 400 | ✓ | ✓ | ✓ | 300s | ~10-18s |

Estimates are end-to-end medians. Generation is a ~3s floor (TTFT ~1.2s + streaming), which
is why the bottom three barely differ — the dial buys search depth, and search was never the
expensive part.

### The agent loop

Decisions are a small JSON verdict, not native tool calls. vLLM's tool-calling requires
`--tool-call-parser` chosen per model at launch, which would pin the agent to one generator
and break the model picker; a JSON verdict works on any instruction-tuned model and degrades
to "answer" when parsing fails.

- **search** — re-query in the paper's own terminology, up to 3 queries.
- **expand** — pull neighbouring chunks by `ordinal` (index lookup, no embedding). Reads 2
  before and 1 after, because the usual failure is a chunk referring backwards.
- **clarify** — non-blocking. Emits suggested refinements *and* still answers from what it
  has.

Rounds deduplicate against chunks already seen and stop early when a round adds nothing,
which prevents spending the budget re-retrieving the same material.

### Query decomposition

Gated by a regex pre-filter (`compare|versus|and which|…`, multiple `?`) so atomic questions
cost nothing — measured 5.0s atomic vs 10.8s compound. Parts are retrieved concurrently and
merged **round-robin**, not concatenated: concatenation lets the part with the highest
absolute scores fill the context and starve the other, reproducing the failure decomposition
exists to fix.

### Coverage gate and grounding

Before answering, a verdict classifies the excerpts `full` / `partial` / `none`, each with
its own system instruction. Reranker scores act as a prior so an obviously empty result set
skips the LLM turn.

After answering, every cited sentence is scored against the excerpt it cites.
**Threshold 0.5**, calibrated on deliberately mis-cited text: genuine support scores
0.98–1.00, a fabricated but on-topic claim ~0.29, unrelated ~0.00. The first threshold tried
was 0.06 and waved through the invented-but-plausible case — the exact failure the check
exists for.

---

## Temperature

Passed to vLLM. Prompts are laid out stable-prefix-first
(`system + few-shot → excerpts → selection → question`) so prefix caching survives follow-up
questions about the same paper; temperature does not affect that.

`enable_thinking: false` is sent because Qwen3.x emits a `<think>` block by default which
dominated latency (41s → 3.9s once suppressed). Reasoning tags are also stripped from the
accumulated stream, not per chunk — the tag arrives split across token boundaries (`<`,
`think`, `>`) so a per-chunk substring test never fires.

---

## Passage heatmap

`POST /api/heatmap` scores every chunk of the open paper against a reference vector, at full
768-d precision from the mmap (a single paper is a few hundred rows, so exactness is free).

| mode | reference vector | surfaces |
|---|---|---|
| `answer` | top retrieved chunk's vector | the argument around the answer — setup, caveats, ablations |
| `query` | the embedded question | restatements of the question elsewhere in the paper |

They differ substantially. On "how does multi-head attention work?" in 1706.03762, `answer`
mode surfaces the scaled dot-product definition and the additive-vs-dot-product comparison,
which `query` mode misses entirely.

Bands are assigned by **rank**, not raw score: within one paper scores sit in a narrow range
(0.49–0.69 in that example) and normalising them would render five indistinguishable shades.

Rendering uses the CSS Custom Highlight API with five named registries, which paints ranges
without touching the DOM — important because the LaTeXML element ids are the citation
anchors and must not be disturbed.

---

## Search results graph

`/api/search` fuses two evidence sources:

- **paper-level vectors** (title+abstract) — exist for all 377k in-scope papers
- **chunk-level**, aggregated per paper as a **top-3 mean**

Top-3 mean rather than max or overall mean: the overall mean dilutes a 100-chunk paper with
one perfect chunk, answering "is this paper *about* the query" when exploration needs "does
it *answer* the query"; max lets a single spurious chunk carry a paper.

Scores are averaged directly rather than rank-fused, since both come from the same embedding
model in the same cosine space — unlike the dense/BM25 hybrid, where RRF is needed precisely
because the scales are incomparable.

The graph is the **induced subgraph** over the results: edges to papers outside the result
set are dropped, so the picture shows a topic's internal structure rather than hundreds of
background references. Layout is chronological by **rank order, evenly spaced** — real topics
bunch hard, and a proportional time axis collapsed fifteen same-month papers into an
unreadable smear.

`Show top N` re-runs the search rather than trimming client-side, because the induced
subgraph must be recomputed over the new set.

---

## Typography

CSS custom properties on `:root`. Themes define tokens only, so nothing downstream branches
on which is active, and an explicit `data-theme` beats the `prefers-color-scheme` media query
in both directions.

Line width is specified in characters and converted at `size × chars × 0.5`, the conventional
approximation for average glyph advance. Justification enables `hyphens: auto` to avoid
rivers; equations, tables and figures opt out so they are never stretched.

---

## Data collection

Every retrieval writes to `judgements`. This costs no extra inference — the cross-encoder
already scored those passages to rank them. Teachers: `cross_encoder` (free, every query),
`user_click` (a followed citation), `llm` (adjudication of the 0.15–0.75 uncertain band),
`synthetic` (exploration runs).

Full detail, including how to inspect or delete it, is in
`docs/finetuning/gemma_embedder_finetuning.md`.

---

## Endpoints

| endpoint | purpose |
|---|---|
| `GET /api/health` | readiness, warmup timings, vector count |
| `GET /api/paper/{id}` | metadata + sanitised LaTeXML body + section list |
| `POST /api/fetch/{id}` | on-demand crawl, parse, embed for one paper |
| `POST /api/retrieve` | retrieval only — used for speculative prefetch |
| `POST /api/ask` | SSE: `step`, `hits`, `coverage`, `clarify`, `token`, `grounding`, `done` |
| `POST /api/search` | paper search + induced citation subgraph |
| `POST /api/heatmap` | per-chunk relevance within one paper |
| `GET /api/graph/{id}` | ego network with similarity shading |
| `GET /api/breadth` | the depth spectrum |
| `GET /api/device` | platform, accelerator, memory budget, recommended backend |
| `POST /api/model/resolve` / `download` | HF lookup and fetch |
| `POST /api/click` | records a followed citation |
| `POST /api/reload` | rebuild the GPU index after an embed run |
| `GET /api/dataset/manifest` / `file/{name}` | LAN corpus distribution |

**No endpoint is authenticated.** Binding to `0.0.0.0` exposes all of it, including model
downloads and the corpus, to anything that can route to the host.
