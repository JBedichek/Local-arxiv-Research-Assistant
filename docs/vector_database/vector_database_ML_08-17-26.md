# The ML vector database: what it is, what is in it, what it costs

A full accounting of the EmbeddingGemma-embedded corpus on the reference machine — the
storage format, the scope of what was scraped, and every byte the program needs on disk to
run end to end.

All figures measured **2026-08-18** against `/home/user/Desktop/Local-arxiv-Research-Assistant/data`,
with the corpus at 377,093 in-scope papers / 28,723,432 chunks, fully embedded. Sizes are
given in decimal GB (10⁹ bytes) unless marked GiB, because that is what `stat` and the
dataset manifest report; `du -h` prints GiB and will look ~7% smaller for the same file.

---

## 1. What kind of vector database this is

**There is no vector database.** No Faiss index, no Chroma, no pgvector, no Qdrant. The
whole thing is two append-only flat files of raw little-endian tensor bytes, plus one
SQLite database holding everything that is not a vector.

| layer | format | how it is read |
|---|---|---|
| tier 1 (shortlist) | `vectors/int8.bin` — `(N, 256)` int8, C-contiguous, no header | read once at startup, uploaded to GPU as fp16, searched by exact matmul |
| tier 2 (rescore) | `vectors/fp16.bin` — `(N, 768)` float16, C-contiguous, no header | `np.memmap`, gather ~200 rows per query |
| tier 0 (pinned) | *(same files)* | a row-subset of tier 1, not a separate structure |
| paper-level | `vectors/papers_{int8,fp16}.bin` — same layout, one row per paper | title+abstract vectors, used for topic scoping and paper search |
| everything else | `meta.sqlite` — papers, sections, chunk text, FTS5, citation graph | SQLite, WAL mode |

There is no index file because **there is no approximate index.** `lara/index/search.py`
does an exact `torch` matmul of the query against the entire resident int8 matrix on the
GPU — 9.5 ms at 16 M chunks at 100% recall — which is faster than the HNSW build was worth.
`config.yaml` still declares `paths.ann_index: vectors/hnsw.faiss`; **that file does not
exist and nothing writes it.** Same story for two other declared-but-unused paths:

| declared in config | intended | actual |
|---|---|---|
| `paths.ann_index` → `vectors/hnsw.faiss` | Faiss HNSW over int8 | never created — exact GPU matmul instead |
| `paths.chunk_text` → `chunks/` | zstd parquet shards | never created — chunk text lives in `meta.sqlite`, `chunks.text` |
| `paths.graph` → `graph/` | CSR citation adjacency | never created — the graph is the `citations` table |

Those three entries account for 0 bytes. They are worth knowing about because a reader of
`config.yaml` will otherwise budget disk for them.

### Row addressing

The files have no header and no per-row metadata. Row *i* of `fp16.bin` and row *i* of
`int8.bin` are the same chunk; the mapping to a chunk is `chunks.vector_row` in SQLite, and
`VectorStore.rows()` derives the row count from `filesize / (dim × itemsize)` and refuses to
proceed if the two files disagree. Vectors are fsynced *before* the owning chunk row is
stamped with its `vector_row`, so a crash leaves orphan rows at the tail — wasted bytes,
never wrong answers.

int8 is Matryoshka truncation, not product quantization: take the leading 256 of the 768
dimensions, renormalize, and scale by 127 into int8. One global scale, no per-vector
codebook, so tier 1 is a plain integer dot product.

---

## 2. Scope of the dataset

### What the config asks for

```yaml
corpus:
  categories: [cs.LG, cs.CL, stat.ML, cs.NE]
  date_floor: "2015-01-01"
  include_cross_lists: true

ingest:
  metadata_source: oai
  oai_endpoint: https://export.arxiv.org/oai2
  oai_sets: [cs, stat]
```

The harvest and the scope are deliberately different sizes. OAI-PMH is harvested over the
**whole `cs` and `stat` sets with no category or date filter** — arXiv's `from=` parameter
filters on last-modified datestamp rather than submission date, so filtering server-side
would silently drop papers whose v1 predates the window but which were revised inside it.
The 2015 floor and the four-category test are applied locally, at ingest, against the v1
submission date.

### What that produced

| | count | note |
|---|---:|---|
| paper metadata rows harvested | 1,011,039 | all of `cs` + `stat`, submissions from 1990-01-01 |
| **in scope** (`in_scope=1`) | **377,093** | 37.3% of the harvest |
| — with full text parsed | 368,477 | `fulltext_status='ok'` |
| — fetch failed | 8,231 | |
| — no source available | 386 | no HTML, no ar5iv, no usable PDF |
| out of scope, metadata only | 633,946 | `fulltext_status='pending'`, never attempted |
| sections | 9,635,173 | over 349,593 distinct papers |
| chunks | 28,723,432 | 78.0 per full-text paper |
| chunks embedded | 28,723,432 | 100% |
| paper-level vectors | 377,090 | title+abstract, one per in-scope paper |
| citation edges | 7,218,834 | Semantic Scholar; 366,527 papers resolved, 10,566 missing |
| extracted citation contexts | 312,664 | citing paragraphs, for fine-tuning |
| teacher judgements | 11,661 | retrieval eval / distillation labels |

In-scope submission dates run **2015-01-01 → 2026-08-14**, growing roughly 16× from 2015 to
2025:

| year | in-scope papers | | year | in-scope papers |
|---|---:|---|---|---:|
| 2015 | 4,118 | | 2021 | 33,843 |
| 2016 | 6,278 | | 2022 | 36,830 |
| 2017 | 9,164 | | 2023 | 44,573 |
| 2018 | 14,884 | | 2024 | 56,479 |
| 2019 | 24,026 | | 2025 | 65,943 |
| 2020 | 32,312 | | 2026 (to 08-14) | 48,643 |

**`include_cross_lists: true` is doing a third of the work.** 126,959 in-scope papers —
33.7% — have a *primary* category outside the four requested; they qualify by cross-list
only. The top primary categories in scope:

| primary | papers | | primary | papers |
|---|---:|---|---|---:|
| cs.LG | 138,851 | | cs.CR | 5,966 |
| cs.CL | 84,387 | | cs.RO | 5,671 |
| cs.CV | 28,664 | | eess.IV | 5,347 |
| stat.ML | 19,587 | | cs.IR | 5,274 |
| cs.AI | 12,066 | | cs.SD | 4,158 |
| cs.NE | 7,309 | | math.OC | 4,035 |

So the corpus is "the ML literature" in a broad sense: an ML-primary core plus the vision,
robotics, security, IR, audio and optimization papers that chose to cross-list into it.

### Full-text sourcing and chunking

Full text was fetched in source order `arxiv_html → ar5iv → pdf`:

| source | papers | share |
|---|---:|---:|
| `arxiv_html` (native arXiv HTML) | 300,417 | 81.5% |
| `ar5iv` (LaTeXML rendering) | 50,539 | 13.7% |
| `pdf` (parsed fallback) | 17,521 | 4.8% |

Chunking is 1,000 chars target with 15% overlap and `never_cross_sections: true`, giving:

| kind | chunks | share |
|---|---:|---:|
| body | 25,386,749 | 88.4% |
| caption | 1,386,250 | 4.8% |
| equation | 897,694 | 3.1% |
| theorem | 627,224 | 2.2% |
| abstract | 425,515 | 1.5% |

Two things to note. `keep_kinds` also lists `table` and `footnote`, and **zero chunks of
either kind exist** — the parsers do not currently emit them. And 368,477 papers have
chunks while only 349,593 have section rows, so ~18.9k papers were chunked without a
recovered section structure (typically PDF-sourced, where headings did not survive).

Total chunk text is 22,173,227,803 characters — 22.2 GB of raw text before SQLite overhead.

### Embedding parameters

| | |
|---|---|
| model | `google/embeddinggemma-300m` (1.27 GB in the HF cache) |
| full dim | 768 → `fp16.bin` |
| truncated dim | 256 (Matryoshka) → `int8.bin` |
| max sequence | 512 tokens |
| batch | 512 chunks, 8,192-chunk durable checkpoints |
| document prefix | `title: {title} > {section} | text: ` |
| query prefix | `task: search result | query: ` (asymmetric — EmbeddingGemma requires it) |
| devices | cards 0/1/2 for bulk indexing; `torch.compile` mode `default`, `dynamic=true` |

Measured throughput ~900–1,030 chunks/s per card compiled (2.19× eager). The 377k
paper-level vectors took 9.4 minutes end to end.

---

## 3. Disk usage, component by component

### 3.1 Everything under `data/`

| path | bytes | GB | GiB | share |
|---|---:|---:|---:|---:|
| `vectors/fp16.bin` | 45,392,286,720 | 45.39 | 42.27 | 31.3% |
| `meta.sqlite` | 42,768,662,528 | 42.77 | 39.83 | 29.5% |
| `raw/` | 39,766,208,911 | 39.77 | 37.04 | 27.4% |
| `meta.sqlite-wal` | 8,817,467,472 | 8.82 | 8.21 | 6.1% |
| `vectors/int8.bin` | 7,565,381,120 | 7.57 | 7.05 | 5.2% |
| `vectors/papers_fp16.bin` | 579,210,240 | 0.58 | 0.54 | 0.4% |
| `vectors/papers_int8.bin` | 96,535,040 | 0.10 | 0.09 | 0.07% |
| `meta.sqlite-shm` | 17,137,664 | 0.02 | 0.02 | 0.01% |
| `logs/` | 10,500,039 | 0.01 | 0.01 | 0.01% |
| `scope/` | ~0 | — | — | — |
| `dataset_manifest.json` | 440 | — | — | — |
| **total** | **145,013,390,232** | **145.01** | **135.05** | |

`du -sh data` reports **136G** because `du` counts allocated blocks in GiB; the 368k small
files under `raw/` round up to 4 KiB each.

### 3.2 The vector files

Both chunk-level files carry **29,552,270 rows**, derived from file size:

| file | row stride | rows | live rows | orphan rows |
|---|---:|---:|---:|---:|
| `fp16.bin` | 768 × 2 = 1,536 B | 29,552,270 | 28,723,432 | 828,838 |
| `int8.bin` | 256 × 1 = 256 B | 29,552,270 | 28,723,432 | 828,838 |
| `papers_fp16.bin` | 1,536 B | 377,090 | 377,090 | 0 |
| `papers_int8.bin` | 256 B | 377,090 | 377,090 | 0 |

**828,838 orphan rows — 1.27 GB in `fp16.bin` and 0.21 GB in `int8.bin`, 1.49 GB total
(2.8%) — are dead weight.** They are the crash-safety design working as intended: rows
written and fsynced during runs that died before stamping `chunks.vector_row`, whose chunks
were then re-embedded at a later row. Nothing points at them, and searching them is
harmless because the fused result only surfaces rows reachable from SQLite. Reclaiming them
means a compaction pass that rewrites both files and renumbers `vector_row`; at 1.5 GB out
of 53.6 GB that has not been worth doing.

Live vector cost per chunk is fixed and exact:

| | bytes/chunk | at 28.72 M chunks |
|---|---:|---:|
| fp16 768-d (tier 2) | 1,536 | 44.12 GB |
| int8 256-d (tier 1) | 256 | 7.35 GB |
| **both** | **1,792** | **51.48 GB** |

Tier 1 is the number that governs memory, not disk: 7.35 GB of int8 must be resident, and
`search.py` uploads it to the GPU as fp16, which doubles it to ~14.7 GB of VRAM. That is
what topic-scoped residency (D22, `lara/index/scope.py`) exists to cut — it keeps a
row-subset resident and deletes nothing, so it changes RAM and VRAM, never disk.

### 3.3 Inside `meta.sqlite`

Page size 4,096 B, 10,441,568 pages, freelist 0 (no reclaimable slack). Per-object usage
from `dbstat`:

| object | GB | share | what it is |
|---|---:|---:|---|
| `chunks` | 28.07 | 65.6% | **chunk text itself** — 22.2 GB of characters plus row overhead |
| `chunks_fts_data` | 8.43 | 19.7% | FTS5 inverted index (BM25 half of hybrid retrieval) |
| `papers` | 1.95 | 4.6% | 1.01 M rows; abstracts alone are 1.21 GB |
| `chunks_unique` | 0.82 | 1.9% | index |
| `chunks_paper` | 0.82 | 1.9% | index |
| `sections` | 0.79 | 1.8% | 9.64 M rows, `WITHOUT ROWID` |
| `chunks_vector` | 0.44 | 1.0% | index on `vector_row` |
| `citation_contexts` | 0.39 | 0.9% | 312,664 citing paragraphs |
| `chunks_fts_docsize` | 0.33 | 0.8% | FTS5 length normalization |
| `citations_dst` | 0.25 | 0.6% | reverse-edge index |
| `citations` | 0.23 | 0.5% | 7.22 M edges, `WITHOUT ROWID` |
| `chunk_df` | 0.07 | 0.2% | 3.88 M term document-frequencies for IDF query planning |
| all remaining objects | 0.19 | 0.4% | `papers_*` indexes, `paper_vectors`, `cc_*`, `judgements`, FTS config |
| **total** | **42.77** | | |

Two thirds of the database is text the reader has to be able to quote and anchor, and a
fifth is the BM25 index over it. The vector-related bookkeeping (`chunks_vector`,
`paper_vectors`, `paper_vectors_row`) is 0.46 GB — 1% of the database.

**The 8.82 GB WAL is not permanent.** SQLite is in WAL mode and the file grows to the
high-water mark of the largest write burst (the embedder stamping millions of
`vector_row`s). A `PRAGMA wal_checkpoint(TRUNCATE)` with no readers attached returns those
8.8 GB. The 17 MB `-shm` is the shared-memory index for the WAL and is recreated on open.

### 3.4 The raw HTML cache

`raw/` is 39.77 GB in **368,479 files** across 140 month-shard directories
(`raw/YYMM/{arxiv_id}.{source}.html.zst`), zstd-compressed — averaging 107.9 KB per paper
compressed. Growth tracks arXiv volume: `raw/1501` is 16 MB, `raw/2510` is 1.2 GB.

This is the one large component that is **prunable**. It exists so the corpus can be
re-parsed and re-chunked after a parser change without re-crawling arXiv — several days of
polite crawling at 40 req/s. Delete it and search, answers and citations all keep working;
only re-parsing breaks. In the dataset manifest it is its own `archive` tier for exactly
this reason.

### 3.5 Produced but not stored

Worth stating explicitly, because they are natural things to budget for and cost nothing:

- **`scope/`** is empty. Topic-scoped keep-sets are small JSON + row-id arrays written on
  demand; no scope is currently resolved.
- **Fine-tuning produces no checkpoint.** `lara/finetune/` builds its training pairs by
  query from SQLite and the existing vectors, trains in memory, and reports metrics — the
  k-fold run (`logs/kfold.log`) writes 6.9 KB of log and nothing else. There are no
  `.safetensors`, `.pt`, or adapter files anywhere in the project tree.
- **`logs/`** is 10.5 MB total, dominated by `crawl.log` at 8.3 MB.

---

## 4. What the program needs outside `data/`

The corpus is a minority of the disk this program actually occupies end to end.

| component | size | note |
|---|---:|---|
| `data/` (corpus) | 145.0 GB | §3 |
| `~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8` | 30.9 GB | the configured generator, `default_model` |
| `.venv-vllm/` | 8.6 GB | torch + vLLM + CUDA wheels |
| `~/.cache/vllm/torch_compile_cache` | 3.3 GB | vLLM's compiled graphs, rebuilt if deleted |
| `~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3` | 2.29 GB | the configured cross-encoder |
| `~/.cache/huggingface/hub/models--google--embeddinggemma-300m` | 1.27 GB | the embedder |
| repo source (`lara/`, `docs/`, `web/`, `scripts/`, `.git/`, `PLAN.md`) | ~4 MB | |
| **required total** | **~191.4 GB** | |

**The HF cache as it stands is 161 GB, not the 34.5 GB this program requires.** The rest is
other work sharing `~/.cache/huggingface`, and the largest single item in it is unrelated to
this project:

| in the cache but not needed here | size |
|---|---:|
| `mistralai/Mixtral-8x7B-v0.1` | 87 GB |
| `AxionML/Qwen3.5-9B-NVFP4` | 8.8 GB |
| `tomaarsen/Qwen3-Reranker-4B-seq-cls` | 7.6 GB |
| `Qwen/Qwen3-Reranker-4B` | 7.6 GB |
| `ibm-granite/granite-3.0-1b-a400m-base` | 5.2 GB |
| rejected reranker candidates (`Qwen3-Reranker-0.6B`, `-0.6B-seq-cls`) | 3.5 GB |
| misc. models + eval datasets (`cnn_dailymail`, `hellaswag`, `mmlu`, …) | ~2.5 GB |

The two 4B Qwen3-Reranker checkpoints and the two 0.6B variants are the
measured-and-rejected alternatives documented in `config.yaml` under `index.rerank` —
18.7 GB that can be deleted. There is also a **38 GB `~/.cache/pip`** that is pure build
cache.

### Free space — currently below the floor

`config.local.yaml` sets `disk.min_free_gb: 60` on `/dev/nvme0n1p5`. That filesystem now has
**43.9 GB free of 1,018 GB (96% used)**, so `lara preflight` fails its free-space check as of
this measurement. Clearing the check needs only ~17 GB; the options, ordered by how little
it costs to lose them:

| action | reclaims | cost |
|---|---:|---|
| `rm -rf ~/.cache/pip` | 38 GB | re-download wheels on next install |
| delete `mistralai/Mixtral-8x7B-v0.1` from the HF cache | 87 GB | unrelated to this project |
| delete the two rejected 4B reranker checkpoints | 15.2 GB | re-download to re-run the probe |
| `PRAGMA wal_checkpoint(TRUNCATE)` on `meta.sqlite` | 8.8 GB | none — it grows back on the next big write |
| compact the vector files | 1.5 GB | a full rewrite of 53 GB; not worth it |
| delete `data/raw/` | 39.8 GB | cannot re-parse without re-crawling |

---

## 5. Cost per unit, for sizing a different corpus

Marginal cost of one more chunk, measured rather than estimated:

| | bytes/chunk |
|---|---:|
| fp16 768-d vector | 1,536 |
| int8 256-d vector | 256 |
| `chunks` table (text + row) | 977 |
| FTS5 index (`_data` + `_docsize` + `_idx`) | 305 |
| B-tree indexes on `chunks` | 72 |
| **total** | **≈ 3,146** |

Per in-scope paper, amortising papers/sections/citations/raw over 368,477 full-text papers:

| | per paper |
|---|---:|
| chunks (78.0 × 3,146 B) | ≈ 245 KB |
| compressed raw HTML | ≈ 108 KB |
| metadata, sections, citations, paper vectors | ≈ 12 KB |
| **total, with raw** | **≈ 365 KB** |
| **total, raw pruned** | **≈ 257 KB** |

Rules of thumb that follow: **1 M chunks ≈ 3.1 GB**, **100 k full-text papers ≈ 37 GB with
the raw archive, ≈ 26 GB without.** Dropping tier 2 (`fp16.bin`) removes 49% of the
per-chunk cost and costs exact rescore; dropping the raw archive removes 30% of the
per-paper cost and costs re-parseability.

---

## 6. What is distributable

`lara/serve/dataset.py` publishes the corpus over the reader's own HTTP server in three
tiers, and the split maps directly onto the components above:

| tier | files | size | what you get |
|---|---|---:|---|
| `core` | `meta.sqlite`, `vectors/int8.bin`, `vectors/papers_{int8,fp16}.bin` | 51.0 GB | search, BM25, citations, answers; tier-2 rescore degrades to int8 |
| `full` | adds `vectors/fp16.bin` | +45.4 GB → 96.4 GB | exact fp16 rescore restored |
| `archive` | adds `raw/` | +39.8 GB → 136.2 GB | re-parse after a parser change |

The current `data/dataset_manifest.json` is a **partial** snapshot: created 2026-08-18T06:20Z
with tier `core` but listing only the two paper-level vector files (675,745,280 bytes total).
`meta.sqlite` and `int8.bin` were still being written at publish time and are absent, so a
client fetching `core` from this manifest gets paper vectors and nothing else. Re-publishing
with the corpus quiesced is what makes `core` a usable download — the snapshot semantics are
deliberate (hashing a live file yields a digest that is wrong by the time it is read), but
the snapshot has to be taken when the writers are stopped.
