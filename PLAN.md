# arxiv-rag — Design & Implementation Plan

Interactive reader for arXiv ML papers with local-LLM RAG: open a paper, highlight a
passage, ask a question, get a grounded answer with click-through citations, and explore
the citation neighbourhood as a similarity-shaded graph.

Status: **planning**. Nothing implemented yet. Decisions taken so far are in §12.0;
remaining open questions in §12.1.

---

## 0. Verified facts about this machine (2026-08-17)

Measured, not assumed:

| Thing | Value |
|---|---|
| RAM | 125 GiB total, ~104 GiB available, 8 GiB swap (5 GiB already used) |
| CPU | 48 cores |
| Disk | `/dev/nvme0n1p5` 949 G, **245 G free** (73% used) |
| GPUs | 3 NVIDIA devices on PCI: `10de:2bb4` @41:00, `10de:2bb4` @42:00, `10de:2bb1` @81:00 (Blackwell-class, 128 G prefetchable BAR each) |
| GPU status | ⚠️ **`nvidia-smi` fails**: `Driver/library version mismatch`. NVML lib 580.173 vs kernel module 580.159.03 |
| Python | 3.12.3 |
| Existing venv | `/home/user/Desktop/Learned-Data-Selection/venv` — vllm 0.14.0, torch 2.9.1, transformers 4.57.6, sentence-transformers 5.1.2, **faiss-gpu-cu12 1.13.2** |
| Node | v18.19.1, npm 9.2.0 |
| HF cache | `~/.cache/huggingface`, 101 GB, 390 model repos, 37 dataset repos |
| Network | arxiv.org 200, export.arxiv.org API 200, huggingface.co 200 |
| Not installed | `aws` cli, `s5cmd` (needed for arXiv S3 bulk) |

### 0.1 The HF cache is not a serving cache

Of 390 `models--*` repos, only **5 have >1 GB of resolved weight blobs**:

| Size | Repo | Arch | Quant |
|---|---|---|---|
| 93.4 GB | `mistralai/Mixtral-8x7B-v0.1` | MixtralForCausalLM | none (bf16) |
| 5.5 GB | `ibm-granite/granite-3.0-1b-a400m-base` | GraniteMoeForCausalLM | none |
| 2.8 GB | `Isotonic/TinyMixtral-4x248M-MoE` | MixtralForCausalLM | none |
| 2.5 GB | `meta-llama/Llama-3.2-1B` | LlamaForCausalLM | none |
| 1.2 GB | `google/embeddinggemma-300m` | Gemma3TextModel | none |

The other 385 are metadata-only snapshots, partial downloads, GGUF-only repos, or
`tiny-random` architecture stubs from MoE research. **Three consequences:**

- The only complete generator is `Mixtral-8x7B-v0.1` — a *base* model, not instruct-tuned.
  It will be poor at grounded QA and will not follow a citation format reliably.
- `google/embeddinggemma-300m` is fully present and is an excellent embedder for this job
  (768-d, Matryoshka-truncatable to 512/256/128, 2048-token context, ~600 MB in bf16).
  **Adopt it as the default embedder** — see §5.
- The model picker (R7) needs a *completeness* check, not just a directory listing.
  See §8. Absent a download, the picker's generator list is nearly empty.

---

## 1. Requirements

Traceable IDs; every section below references these.

| ID | Requirement |
|---|---|
| **R1** | Vector DB of arXiv ML papers for RAG |
| **R2** | Every chunk indexed by arXiv ID **and** in-paper location, such that a citation the LLM emits is a clickable link that opens the paper scrolled to and highlighting that passage |
| **R3** | Interactive web UI; enter an arXiv number, the paper renders in-app |
| **R4** | **Core interaction:** select/highlight a region of the rendered paper, ask a question about it, get a fast RAG answer |
| **R5** | Low latency is a first-class design constraint, not an afterthought |
| **R6** | LLM / quantization / temperature / sampling selectable from the UI |
| **R7** | Generator list restricted to models already present in the user's HF cache |
| **R8** | Citation graph of papers in the UI; ask questions about related work |
| **R9** | Graph nodes heat-shaded by vector similarity (to the current query/selection) as a relevance marker, for fast exploration |
| **R10** | Corpus limited to ML for now; keep the hot working set in RAM, bounded |

Non-goals for v1: multi-user auth, non-ML arXiv categories, training/fine-tuning anything,
mobile layout.

---

## 2. Corpus scope and sizing

### 2.1 How big is "arXiv ML"?

Live counts from the arXiv API (`cat:<category>`, includes cross-lists), 2026-08-17:

| Category | Papers |
|---|---|
| cs.LG | 281,434 |
| cs.CV | 202,340 |
| cs.AI | 194,769 |
| cs.CL | 116,387 |
| stat.ML | 79,471 |
| cs.NE | 18,126 |
| **sum (with heavy overlap)** | 892,527 |
| **estimated unique union** | **~600 k** |

Whole arXiv is ~2.9 M papers, so ML is ~20% of it. Full-text-of-everything is off the
table on 245 GB of free disk; a targeted ML subset is not.

Two tiers of scope worth distinguishing:

- **Core** = `cs.LG ∪ cs.CL ∪ stat.ML ∪ cs.NE`, 2015→present ≈ **~330 k papers**.
- **Extended** = add `cs.CV ∪ cs.AI` and pre-2015 ≈ **~600 k papers**.

→ **Decided (D1): Core, 2015+ ≈ 330 k papers.** Sizing tables below give the Extended
figures as an upper bound; halve them for the Core numbers actually in play.

### 2.2 Storage math

Chunking target ~1000 chars (~250 tokens) with ~15% overlap, paragraph-aligned (§4).
Average ML paper full text ≈ 40 k chars → **~50 chunks/paper**.

| Scope | Papers | Chunks |
|---|---|---|
| Core | 330 k | ~16 M |
| Extended | 600 k | ~30 M |

At the Extended figure of 30 M chunks:

| Artifact | Precision | Size | Lives in |
|---|---|---|---|
| Chunk text | zstd in SQLite/Parquet | ~10 GB | disk |
| Chunk vectors, full | 768-d fp16 | 46 GB | disk, mmap |
| Chunk vectors, MRL-256 | int8 | **7.7 GB** | **RAM** |
| HNSW graph (M=32) over MRL-256 | — | ~3.8 GB | RAM |
| Alt: IVF-PQ64 (nlist 65536) | — | 1.9 GB | RAM |
| Paper-level vectors (title+abstract) | 768-d fp16 | 0.9 GB | RAM |
| Citation edges (~30 refs/paper) | 18 M edges | <1 GB | RAM |
| **RAM total (warm tier)** | | **~13 GB** | |
| **Disk total** | | **~60 GB** | of 245 GB free |

Comfortable. Even Extended scope leaves >90 GB of RAM for vLLM's host-side needs, the OS
page cache, and the hot tier.

### 2.3 Embedding cost is not the bottleneck

30 M chunks × ~250 tokens = 7.5 B tokens through a 300 M-param encoder. At a conservative
~5 k chunks/s/GPU in bf16 with batch 512, that's ~1.7 GPU-hours; across 2–3 cards, **~1
wall-clock hour**. Re-embedding the whole corpus is cheap enough to do on a whim, which
means the embedder choice is *not* a one-way door.

**The bottleneck is acquiring full text.** See §3.

---

## 3. Data acquisition

Three separate streams: metadata, full text, citations.

### 3.1 Metadata — solved, cheap

- **Bootstrap**: the Kaggle `Cornell-University/arxiv` snapshot — one JSON line per paper
  for all ~2.9 M, ~4.5 GB, refreshed weekly. Gives id, title, abstract, authors,
  categories, version history, DOI, update date. Filter to ML categories on ingest.
- **Incremental**: arXiv OAI-PMH (`http://export.arxiv.org/oai2`, `arXivRaw` metadata
  prefix) with a resumption token, run nightly with `from=<last run>`.
- Abstract-only RAG over the *entire* ML corpus is therefore available on **day one**,
  independent of full-text progress. This matters — see the phasing in §11.

### 3.2 Full text — this is the hard part

Verified availability of rendered HTML (both return 200 with paragraph-level anchors):

- `https://arxiv.org/html/{id}v{n}` — arXiv's native LaTeXML rendering, covers papers
  submitted from **Dec 2023** onward. Confirmed anchor scheme: `id="S1"`, `id="S1.p2"`,
  `id="S1.p2.1"` — section, paragraph, sub-element. **This is exactly the anchor
  granularity R2 needs.**
- `https://ar5iv.labs.arxiv.org/html/{id}` — LaTeXML rendering of older papers, same
  `S{n}.p{m}` anchor scheme. Coverage is good but imperfect (LaTeX that won't convert).
- Fallback: PDF → text via PyMuPDF, anchoring on `(page, bbox)` instead of DOM id, with
  PDF.js in the viewer. Lower quality, but nothing is unreachable.

**The constraint is politeness.** arXiv asks bulk consumers not to hammer the web
frontend; the API guidance is ~1 request / 3 s. 600 k HTML fetches at that rate is **~20
days**. At an aggressive-but-not-abusive 4 req/s it's ~42 hours. This is the single
biggest schedule risk in the project.

Three routes, not mutually exclusive:

**Route A — abstracts now, full text lazily.** Index all ~600 k abstracts immediately
(hours). Fetch and index a paper's full text on first open, and pre-fetch its 1-hop
citation neighbours in the background. Plus a curated core set (say top 30 k by citation
count) crawled up front over a few days. Ships in days; global retrieval is
abstract-granular at first and densifies with use.

**Route B — arXiv S3 bulk (requester-pays).** `s3://arxiv/src/` holds LaTeX source
tarballs. Full arXiv src is ~2.9 TB; the ML subset is maybe ~800 GB. At $0.09/GB egress
that's roughly **$70**, and it exceeds free disk — so it must be
stream-download → convert → keep text → discard. LaTeXML conversion is the real cost:
~10–60 s/paper of CPU, i.e. 600 k × 20 s / 48 cores ≈ **70 core-hours ≈ 3 days wall
clock**, though it parallelizes perfectly across the 48 cores. Requires an AWS account and
`s5cmd`/`aws` (neither installed). Highest fidelity — we get real LaTeX, so equations,
theorem environments and section structure survive cleanly.

**Route C — polite background HTML crawl.** 2–4 req/s with a descriptive User-Agent,
respecting 429s with exponential backoff, prioritized by citation count then recency,
running for weeks as a daemon. Zero cost, zero setup, slow.

→ **Decided (D2): A now, C continuously in the background.** No S3, no cost. See §12.0 for
the design consequences — chiefly that retrieval and the UI must both handle a corpus where
some papers are abstract-only.

### 3.3 Citation graph

- **Semantic Scholar Academic Graph (S2AG)**: `/graph/v1/paper/batch` accepts 500 IDs per
  request and returns `references` + `citations` with arXiv IDs where known. Free API key,
  generous limits. For 600 k papers that's ~1200 requests — trivial. Bulk dataset dumps
  also exist if the API rate proves annoying.
- **OpenAlex** as an alternative/cross-check — fully open, no key, full snapshots.
- **Fallback**: parse the bibliography out of the LaTeX/HTML we already fetched. Free but
  requires reference-string → arXiv-ID resolution, which is fiddly.

→ Recommend S2AG primary, OpenAlex to fill gaps. **Open question Q6** (do you already have
an S2 key).

---

## 4. Document model — the deep-link contract (R2)

This is the schema everything else hangs off. Getting it right up front is what makes the
click-through citations work.

```
Paper       (arxiv_id, version, title, abstract, authors, categories,
             published, updated, source_kind, fulltext_status)
Section     (paper, anchor_id "S3.SS2", title, ordinal, depth)
Chunk       (chunk_id, paper, version, section_anchor, para_anchor "S3.SS2.p4",
             char_start, char_end, n_tokens, text, kind, ordinal)
```

`kind` ∈ {abstract, body, caption, equation, table, theorem, footnote, reference}.
Keeping equations and captions as typed chunks matters — a lot of ML questions are about
a specific loss or a specific figure.

**Chunking policy**: never cross a section boundary; merge paragraphs up to ~1000 chars;
split oversized paragraphs at sentence boundaries with 15% overlap; keep every chunk's
`para_anchor` pointing at the DOM element that *starts* it, plus a `char_start` offset
into that element for sub-paragraph precision.

**Citation URL shape**, emitted by the LLM and rendered as a link:

```
/p/2401.12345v1#S3.SS2.p4:120-480
       │            │        └─ char range within the anchored element
       │            └─ LaTeXML DOM id
       └─ arXiv id + version (version matters: anchors shift between versions)
```

The viewer resolves the anchor, scrolls it into view, and paints a highlight over the char
range using a `Range` + CSS Custom Highlight API (with a `<mark>`-wrapping fallback for
older engines). For PDF-sourced papers the fragment degrades to `#page=7&rect=x,y,w,h`.

**Versioning is a real hazard.** `S3.SS2.p4` in v1 may be a different paragraph in v2. So:
pin chunks to a specific version, store the version in every link, and serve the exact
version we indexed rather than "latest".

---

## 5. Index architecture and the RAM plan (R1, R5, R10)

Three tiers, deliberately matched to the access pattern of R4 — when you highlight text in
an open paper, the answer overwhelmingly lives in *that paper* or its immediate citation
neighbourhood.

**Tier 0 — hot, exact, in-process (target <1 ms).**
On paper open, pin into RAM: every chunk vector of the open paper at full fp16 precision,
plus those of its 1-hop citation neighbours, plus the last N papers viewed. That's roughly
50 chunks × (1 + ~30 neighbours) ≈ 1.5–5 k vectors — a brute-force fp16 matmul, faster
than any index. LRU-evicted, capped at a configurable ~2 GB.

**Tier 1 — warm, whole corpus, in RAM (target <15 ms).**
MRL-truncated 256-d int8 vectors for all ~30 M chunks (7.7 GB) under an HNSW graph
(~3.8 GB). EmbeddingGemma is Matryoshka-trained, so truncating 768→256 costs little
accuracy — and we recover it in tier 2 anyway. If RAM ever gets tight, swap HNSW for
IVF-PQ64 at 1.9 GB.

**Tier 2 — cold rerank, mmap'd from NVMe (target <25 ms).**
Take tier-1's top ~200, read their full 768-d fp16 vectors from an mmap'd flat file, score
exactly, keep top ~20. Random NVMe reads at this volume are ~µs each and the page cache
will hold the working set anyway.

**Lexical hybrid — non-optional for this corpus.** Dense retrieval is bad at exact model
names, acronyms, dataset names, symbol names, citation keys. Run SQLite FTS5 (or tantivy)
BM25 in parallel with the dense path and fuse with Reciprocal Rank Fusion. Cheap and a
large quality win on arXiv specifically.

**Paper-level index** — one vector per paper from title+abstract, 600 k × 768 fp16 =
0.9 GB, always resident. Drives the graph heatmap (R9) and lets us shortlist papers before
descending to chunks.

Everything lives in **one process** with the vectors in shared memory. No index server, no
IPC hop, no serialization on the hot path.

---

### 5.1 What a cross-encoder reranker is, and whether we want one (Q9)

Two different ways to score "how relevant is this chunk to this question", with very
different cost/quality profiles.

**Bi-encoder** — what §5 uses for retrieval. The model embeds the query and each chunk
*separately* into vectors; relevance is their dot product. The crucial property is that
chunk vectors are computed **offline**, once. At query time you embed only the query and do
a nearest-neighbour search. That's why 16 M chunks can be searched in ~12 ms.

The weakness follows from the same property: the model never sees the query and the chunk
*together*. Each is compressed into 768 numbers in ignorance of the other, so nothing can
attend across them. In practice that means it is unreliable on:

- **negation** — "methods that do *not* require a replay buffer" scores nearly the same as
  "methods that require a replay buffer";
- **exact discriminators** — "the 7B variant" vs "the 70B variant", "Table 3" vs "Table 4";
- **near-duplicate methodology prose** — arXiv is full of paragraphs that are lexically
  and semantically almost identical but describe subtly different setups.

**Cross-encoder** — the reranker. It takes the query and the chunk **concatenated as one
input** and outputs a single relevance score. Because self-attention runs over both
jointly, every query token can attend to every chunk token. That is exactly what resolves
the three failure modes above.

The cost is structural: nothing can be precomputed, because the score depends on the pair.
You must run the model at query time on every candidate you want scored. So it is never a
replacement for retrieval — it's a **second stage on a short list**:

```
16M chunks ──bi-encoder ANN──▶ top 200 ──exact fp16 rescore──▶ top 50
                                                                  │
                                                     cross-encoder │  (50 forward passes)
                                                                  ▼
                                                               top 8 → prompt
```

Fifty pairs of ~300 tokens through a ~300 M-param cross-encoder is one padded batch —
roughly **30–60 ms** on a Blackwell card. Typical quality gain in the retrieval literature
is a **10–20% relative improvement in nDCG@10**, and it is largest precisely on the
technical-discrimination cases arXiv is made of.

**Where it sits in our latency budget** (§7): total time-to-first-token is ~350–600 ms,
dominated by vLLM's prefill. Adding 40 ms is roughly a **10% increase** — real but not
felt. And it partly pays for itself: better top-8 chunks means a *shorter* context can be
sent, which shortens prefill.

**The tension with D3.** D3 said "strict cache-only, no downloads" — but that decision was
about the *generator*, the model that writes the answer. A reranker is retrieval
infrastructure, the same category as the embedder (`embeddinggemma-300m`, which happens to
already be cached). There is no cross-encoder in the cache, so enabling this means one
~600 MB download (`BAAI/bge-reranker-v2-m3` or `mixedbread-ai/mxbai-rerank-base-v2`).
Whether that violates the spirit of D3 is a judgement call, not a technical one.

**Decided (D9): build the slot, leave it off.** `index.rerank.cross_encoder.enabled:
false` in `config.yaml`. The retrieval pipeline gets a well-defined reranker interface from
day one so turning it on later is a config change plus a download — and, more usefully, so
we can A/B it against the bi-encoder baseline on real questions once there is a corpus to
ask them about. Deciding this on measurement beats deciding it now on priors.

---

## 6. Citation graph and similarity heatmap (R8, R9)

- In-memory CSR adjacency over ~18 M edges, both directions. Neighbour lookup is a slice.
- UI renders an ego-network around the focused paper: 1-hop by default, 2-hop on demand,
  capped at ~150 nodes with the rest collapsed behind "N more". Force-directed layout via
  a small WebGL/canvas renderer (sigma.js or cosmograph — decide at build time; avoid
  d3-force at this node count).
- **Heatmap (R9)**: on every query *and* every highlight-selection, embed the query, dot it
  against the paper-level vectors of the visible subgraph, and shade nodes on a
  perceptually-uniform sequential ramp. That's ~150 dot products — sub-millisecond, so it
  can update live as the user drags a selection.
- Node colour = similarity, node size = citation count, edge direction = cites/cited-by.
- "Ask about related work" = a retrieval scoped to the subgraph rather than the global
  index — same pipeline, a node-ID filter on tier 1.

---

## 7. Serving architecture and latency budget (R3, R5)

```
Browser (vanilla TS + Preact, no heavy framework)
   │  static assets, HTTP/2, precompressed
   │  SSE for token streaming, one persistent connection
   ▼
FastAPI / uvicorn  — single process, holds tiers 0/1/2 + graph in RAM
   ├─ embedder: embeddinggemma-300m resident on GPU, warm, batch-of-1 path
   ├─ retriever: dense + BM25 + RRF + tier-2 rerank
   ├─ paper store: SQLite (WAL) for text + anchors
   └─ vLLM manager: supervises OpenAI-compatible vLLM servers
   ▼
vLLM (separate process, own GPU(s)) — prefix caching ON
```

### Latency budget for R4 (highlight → first token)

| Stage | Target |
|---|---|
| Selection → request (client) | 5 ms |
| Embed selection + question | 5 ms |
| Tier 0 exact search | 1 ms |
| Tier 1 HNSW search | 12 ms |
| Tier 2 mmap rerank | 20 ms |
| BM25 + RRF fusion | 10 ms |
| Prompt assembly | 5 ms |
| **vLLM TTFT** | **200–500 ms** |
| **Total to first token** | **~350–600 ms** |

vLLM dominates, so the optimizations that actually matter are:

1. **Prefix caching.** Structure the prompt as `[system][open paper's context][retrieved
   chunks][selection][question]`. The first two are stable across every question about the
   same paper, so vLLM reuses their KV cache and TTFT collapses.
2. **Speculative prefetch.** Fire retrieval on `selectionchange` — *before* the user
   finishes typing the question. By submit time the chunks are already in hand.
3. **Warm the model.** Keep the chosen generator resident; never cold-start on the hot
   path.
4. **Stream everything.** SSE from the first token, and render citation links
   incrementally as they're parsed out of the stream.

---

## 8. Model & quantization selection (R6, R7)

**Discovery.** Walk `~/.cache/huggingface/hub/models--*/snapshots/*`, and admit a repo only
if it passes *all* of:

- has a `config.json` with an `architectures` entry that vLLM supports;
- has resolved weight blobs (following the symlinks into `blobs/`, since a snapshot dir of
  dangling links looks identical to a complete one on a naïve listing);
- total weight bytes ≥ a floor, and consistent with `config.json`'s parameter count —
  this is what excludes the ~200 `tiny-random` stubs;
- is not GGUF-only — **decided (D4)**: GGUF repos are filtered out entirely, vLLM is the
  sole backend.

Cache the scan result and refresh on a filesystem watch, so the picker opens instantly.

**Quantization is mostly not a free-floating dial.** AWQ/GPTQ/compressed-tensors are baked
into a checkpoint; you can't select them at load time. What is genuinely runtime-selectable
in vLLM is `--quantization fp8` (on-the-fly weight-only FP8, well-supported on Blackwell)
and bitsandbytes int8/nf4 (slower). So the UI dial should be: *native checkpoint quant*
(read from `config.json`'s `quantization_config`) **+** a runtime FP8 toggle **+**
KV-cache dtype (`auto` / `fp8`). Presenting a free "pick any quantization" dropdown would
be lying to the user.

**Model switching costs 30–120 s** of weight loading. Mitigations: keep 1–2 models
resident across the 3 GPUs; show an explicit "loading" state; consider vLLM sleep-mode to
park a model in host RAM. Worth deciding whether switching is rare (simple: restart the
server) or common (complex: a resident pool). **Open question Q4.**

**Sampling controls** exposed: temperature, top_p, top_k, max_tokens, repetition_penalty,
seed. Plus RAG knobs — top_k chunks, whether to restrict to the open paper, whether to
include the citation neighbourhood.

**Decided (D3): strict cache-only.** The picker will therefore offer exactly three
generators — `mistralai/Mixtral-8x7B-v0.1` (default, runtime FP8), plus
`ibm-granite/granite-3.0-1b-a400m-base` and `meta-llama/Llama-3.2-1B` as low-latency
options. All are base models, so §7's prompt layer carries the grounding burden via
few-shot exemplars pinned in the cached prefix. The backend interface stays model-agnostic
so an instruct model can be adopted later without a refactor.

---

## 9. UI specification (R3, R4, R6, R8, R9)

Three-pane layout, all panes resizable, state in the URL so any view is shareable:

```
┌────────────┬───────────────────────────────┬──────────────┐
│  Graph     │  Paper (LaTeXML HTML)         │  Chat        │
│  (R8/R9)   │                               │              │
│  ego-net,  │  selectable text, anchors      │  streamed    │
│  heat-     │  live, highlight → floating    │  answers,    │
│  shaded    │  "Ask about this" affordance   │  citation    │
│            │                               │  links       │
└────────────┴───────────────────────────────┴──────────────┘
 top bar: arXiv id input · model · quant · temp · top_k · retrieval scope
```

Interaction details that matter:

- **Selection → ask** (R4): on `mouseup` with a non-empty selection, a small floating
  button appears at the selection's end. Click (or ⌘/Ctrl-K) opens an inline question box
  pre-loaded with the selected text as context. Retrieval has already fired speculatively.
- **Citation links**: answers render `[2401.12345 §3.2]` chips; click scrolls the paper
  pane (or opens a new tab if it's a different paper) and paints the highlight. Hover shows
  the chunk text in a popover — no navigation needed to sanity-check a citation.
- **Highlight rendering**: CSS Custom Highlight API where available, `<mark>`-wrapping
  fallback. Never mutate the DOM structure of the paper, or the anchors break.
- **Graph ↔ paper ↔ chat are linked**: hovering a graph node previews its abstract;
  clicking loads it; asking a question re-shades the graph.
- Keyboard-first: `/` focus chat, `⌘K` ask-about-selection, `g` focus graph, `o` open paper.

---

## 10. Repository layout

```
arxiv-rag/
├── PLAN.md                    ← this file
├── pyproject.toml
├── config.yaml                ← corpus scope, paths, index params, latency knobs
├── arxiv_rag/
│   ├── ingest/
│   │   ├── metadata.py        Kaggle snapshot + OAI-PMH incremental
│   │   ├── fulltext.py        arxiv/html + ar5iv + PDF fallback, rate-limited
│   │   ├── latexml.py         Route B: S3 src → LaTeXML (optional)
│   │   ├── parse.py           HTML → Section/Chunk with anchors
│   │   └── citations.py       S2AG / OpenAlex
│   ├── index/
│   │   ├── embed.py           embeddinggemma-300m, batched, multi-GPU
│   │   ├── tiers.py           tier 0/1/2 + working-set pinning
│   │   ├── lexical.py         FTS5/tantivy BM25
│   │   └── build.py           offline index build
│   ├── graph/                 CSR adjacency, ego-net extraction, heatmap
│   ├── serve/
│   │   ├── app.py             FastAPI, SSE
│   │   ├── retrieve.py        dense + BM25 + RRF + rerank
│   │   ├── models.py          HF-cache scan, vLLM lifecycle
│   │   └── prompts.py         prefix-cache-friendly prompt assembly
│   └── store/                 SQLite schema + accessors
├── web/                       Preact + TS, Vite
├── scripts/                   bootstrap_corpus.sh, build_index.py, start_vllm.sh
└── docs/                      ADRs
```

---

## 11. Phased roadmap

**Phase 0 — unblock (hours).** Fix the NVIDIA driver mismatch; confirm VRAM per card;
decide venv (reuse `Learned-Data-Selection/venv` vs fresh); smoke-test vLLM; download an
instruct generator if Q3 says so.

**Phase 1 — corpus & embeddings (days).** *First order of business, per the brief.*
Metadata for all ML papers from the Kaggle snapshot + OAI-PMH. Full text for the Phase-1
subset per Q2. HTML → anchored chunks. Embed with embeddinggemma-300m. Build tier-1/2
indexes + BM25. Deliverable: a CLI that answers a question with anchored citations.

**Phase 2 — reader UI (days).** FastAPI + paper pane rendering LaTeXML HTML with anchors
intact; arXiv-ID box; deep-link resolution and highlight painting. Deliverable: R2 + R3
demonstrably working end to end.

**Phase 3 — the core interaction (days).** Selection → floating ask → speculative
retrieval → streamed grounded answer with clickable citations. Tier-0 pinning. Prefix-cache
prompt layout. Deliverable: R4 + R5, measured against the §7 budget.

**Phase 4 — model controls (days).** HF-cache scan with the completeness filter, vLLM
lifecycle management, sampling + RAG knobs. Deliverable: R6 + R7.

**Phase 5 — graph (days).** Citation ingest, CSR store, ego-net renderer, live similarity
heatmap, subgraph-scoped retrieval. Deliverable: R8 + R9.

**Phase 6 — densify.** Background crawler expands full-text coverage toward the whole ML
corpus; incremental nightly index updates.

---

## 12. Decisions and open questions

### 12.0 Decided (2026-08-17)

- **D1 — Corpus scope: Core, 2015+.** `cs.LG ∪ cs.CL ∪ stat.ML ∪ cs.NE`, published or
  updated 2015-01-01 onward. ~330 k papers, ~16 M chunks. Revised budget: **~4 GB RAM**
  for the MRL-256 int8 warm tier + ~2 GB HNSW, ~25 GB fp16 vectors on disk, ~5 GB chunk
  text, ~35 GB total on disk. Comfortable headroom for a later expansion to Extended.
- **D2 — Full text: Route A + C.** Abstracts for the whole Core scope indexed immediately;
  full text fetched on paper-open and for the 1-hop citation neighbourhood; a curated
  top-30 k-by-citation-count crawled up front; a polite background crawler (2–4 req/s,
  descriptive User-Agent, 429 backoff, priority queue) densifies coverage continuously.
  **Design consequence:** every chunk carries a `source_kind` of `abstract` or `body`, and
  retrieval must degrade gracefully when a paper is abstract-only. The UI must show
  full-text coverage state so the user knows what the answer was grounded in.
- **D3 — Generator: strict cache-only, base Mixtral-8x7B.** No downloads.
  **Consequences, accepted:** (a) all three complete generators in cache are *base* models,
  so grounding and citation formatting must come from few-shot prompting rather than
  instruction-following — budget real effort for §7's prompt layer, and put the few-shot
  exemplars in the cached prefix so they cost ~nothing at inference; (b) Mixtral-8x7B is
  93 GB in bf16, so it needs runtime FP8 (`--quantization fp8`, well-supported on
  Blackwell → ~47 GB) or tensor-parallel across cards — depends on per-card VRAM, which is
  unknown until the driver is fixed; (c) `granite-3.0-1b-a400m-base` and `Llama-3.2-1B`
  serve as fast low-latency options for the picker. The model layer stays pluggable so
  adding an instruct model later is a config change, not a refactor.
- **D4 — GGUF: ignored.** vLLM is the only backend. The cache scan filters to
  safetensors repos with a vLLM-supported architecture. No llama.cpp path.

- **D5 — Model switching: single resident (Q4).** One generator on the GPU at a time; a
  switch stops the running vLLM server and starts a new one (~30–120 s, shown as an
  explicit loading state in the UI). No multi-model pool. `serving.vllm.policy:
  single_resident`.
- **D6 — Citations: Semantic Scholar, unauthenticated (Q6).** Verified working
  2026-08-17 without a key: `POST /graph/v1/paper/batch` accepts `ARXIV:1706.03762` IDs
  natively and returns `references` with `externalIds`. Since arXiv ID *is* our primary
  key, this needs no ID-resolution step at all. 330 k papers ÷ 500 per batch ≈ **660
  requests**. OpenAlex was tested as the alternative and is worse for us — its arXiv
  linkage goes through DOIs that don't resolve for older papers (the DataCite
  `10.48550/arXiv.*` lookup 404s on 1706.03762), so it would need a fuzzy title-matching
  step. Kept as `ingest.citations.fallback` for gap-filling. A free S2 key (a web form at
  semanticscholar.org/product/api, granted in a few days) is worth requesting **only if we
  start seeing 429s** — `S2_API_KEY` is already wired in config.
- **D7 — Crawler rate: 3 req/s, adaptive (Q11).** Above arXiv's published ~1 req/3 s API
  guidance, so it backs off aggressively and honestly: descriptive User-Agent with a
  contact address, 60 s initial backoff on 429 doubling to 1 h, and
  `adaptive_throttle` halves the standing rate after three consecutive 429s rather than
  retrying at the same pace. At 3 req/s, 330 k papers is ~30 hours.
- **D8 — Packaging: reuse the venv, ship a real project (Q7).** Development uses the
  existing `Learned-Data-Selection/venv` (vllm 0.14.0, torch 2.9.1, faiss-gpu-cu12) via
  `pip install -e . --no-deps`, so nothing in that environment gets version-churned. The
  repo nonetheless carries a proper `pyproject.toml` (hatchling, `lara` package, pinned
  floors, `[cpu]`/`[gpu]`/`[serve]`/`[dev]` extras, `lara` console script) so a new
  contributor gets a working setup from `python -m venv .venv && pip install -e ".[cpu,dev]"`.
  Both paths documented in the README.
- **D9 — Reranker: slot built, disabled (Q9).** See §5.1 for the full rationale. Interface
  exists from day one; `enabled: false` until we can A/B it on real questions.
- **D10 — Disk: root filesystem only, enforced (Q8).** Of three mounted filesystems only
  `/` (nvme0n1p5, 245 GB free) is usable — `/data` is 100% full at 0 bytes free, and
  `/media/user/Extreme SSD` has 31 GB on an external SATA drive. `~/.cache/huggingface` was
  verified to be a real directory on `/`, not a symlink. `lara preflight` resolves every
  configured path through `realpath`, checks the backing device against
  `disk.required_device`, hard-blocks the two bad mounts via `disk.forbid_paths`, and
  refuses to start below 60 GB free. It also catches a stray `HF_HOME` in the environment.
- **D11 — Name (Q10).** `Local-arxiv-Research-Assistant`, package `lara`, remote
  `git@github.com:JBedichek/Local-arxiv-Research-Assistant.git`.
- **D12 — Metadata via OAI-PMH, not Kaggle.** No Kaggle credentials exist on this machine,
  and OAI-PMH needs no account, no API token and no 4.5 GB download — plus it doubles as
  the nightly incremental path. Verified 2026-08-17: ~1300 records/request with a
  resumption token. **Trap found and handled:** OAI's `from=`/`until=` filter on the *OAI
  datestamp* (last-modified), not the submission date — paper 1107.0901, submitted 2011,
  carries datestamp 2026-08-03. So the sets are harvested in **full** and D1's 2015 floor
  is applied to the `v1` version date at ingest. Consequence: `papers` holds every `cs` +
  `stat` record (~1.1 M rows, ~2 GB) with an `in_scope` flag, so widening D1 Core →
  Extended later is a re-flag, not a re-harvest.
- **D13 — Reranker enabled: `Qwen/Qwen3-Reranker-4B`.** Reverses D9. Chosen from current
  HF download rankings rather than priors: 4.02 B params, and since April it ships native
  sentence-transformers support (`config_sentence_transformers.json`, `1_LogitScore`), so
  it drops into a `CrossEncoder` with no seq-cls conversion shim. ~8 GB bf16 on card 1.
  Rejected: `bge-reranker-v2-m3` (most-downloaded but older), `Qwen3-Reranker-8B` (~2×
  the latency for a marginal gain on a 50-candidate list).
- **D14 — GPU allocation.** Verified via torch: 3 × RTX PRO 6000 Blackwell, **102 GB
  each**, sm_120. NVML is broken but `torch.cuda.is_available()` is True and device
  properties read correctly, so vLLM is expected to run — the `nvidia-smi` failure is
  cosmetic. Mixtral-8x7B at bf16 (93 GB) fits on one card without quantization, making
  `fp8` optional rather than required. Allocation: *bulk indexing* — embedder on cards
  0+1, vLLM not running; *serving* — embedder card 0, reranker card 1, vLLM card 2.

### 12.0.1 Checkpointing (per the "save often so we can iterate" requirement)

Every ingest stage is resumable at page granularity, so long jobs can run while the rest of
the system is being built:

- **Metadata.** `write_page()` commits the records *and* the next resumption token in one
  SQLite transaction. A `kill -9` mid-harvest resumes from a token whose page is already
  committed — never a lost page, never a double-counted one. Re-running `lara harvest`
  picks up where it stopped; `--restart` forces a fresh pass.
- **Full text.** Per-paper status column (`pending|ok|failed|unavailable`) plus
  `fulltext_attempts`, so the crawler is a resumable queue rather than a linear scan.
- **Citations.** Per-paper `s2_status`, same pattern, 500 ids per batch.
- **Embeddings.** Vectors append to a fixed-stride flat file with the row offset recorded
  per chunk, so an interrupted embed run resumes from the last committed offset.

`lara status` reports all of it.

### 12.1 Still open

Nothing blocking. Deferred until there is a corpus to measure against:

- Whether to enable the cross-encoder reranker (D9) — decide on measured nDCG, not priors.
- Whether to request a Semantic Scholar API key (D6) — only if 429s appear.
- Whether to expand from Core to Extended scope (D1) — the RAM/disk budget has room.

### 12.2 Immediate blocker

`nvidia-smi` fails with `Driver/library version mismatch` (NVML 580.173 vs loaded kernel
module 580.159.03). No CUDA process — vLLM or the embedder — can start until this is
resolved, and per-card VRAM is unknown until it is, which in turn determines whether
Mixtral-8x7B runs FP8 on one card or needs tensor parallelism. `lara preflight` reports it
with the fix:

```bash
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia
# or reboot if the modules are held by a display server
```
