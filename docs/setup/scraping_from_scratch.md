# Building the corpus from scratch

How the arXiv pipeline works, the commands to reproduce it, and what to change to build a
different dataset.

Every number here is measured on the reference build: **377,093 in-scope papers, 368,477 with
full text, 28.7M chunks, 6.7M citation edges, 119 GB on disk.**

---

## 1. The pipeline

Four independent stages. Each is resumable and checkpoints as it goes, so all of them can be
interrupted, restarted, or run concurrently.

```
   OAI-PMH ──▶ papers          metadata for every cs/stat paper       ~1 hr
      │
      ▼
   crawl ─────▶ raw HTML       arxiv.org/html, ar5iv, PDF fallback    ~6 hrs
      │
      ▼
   parse ─────▶ chunks         anchored paragraphs, equations, captions
      │
      ▼
   embed ─────▶ vectors        fp16 768-d + int8 MRL-256              ~4 hrs
      
   citations ─▶ graph          Semantic Scholar, runs independently   ~5 hrs
```

### Metadata — OAI-PMH

arXiv's OAI-PMH endpoint returns ~1300 records per request with a resumption token. No
account, no API key, no download quota. The whole `cs` and `stat` sets take about an hour.

**One trap worth knowing.** OAI's `from=` parameter filters on the record's *last-modified*
datestamp, not its submission date — paper 1107.0901 was submitted in 2011 but carries a
2026 datestamp because its metadata was touched. So a 2015 date floor cannot be expressed as
`from=2015-01-01`. We harvest the sets in full and apply the floor to the `v1` version date
at ingest, which also means widening the corpus later is a re-flag rather than a re-harvest.

`from=` is still useful as an *ordering* control: records arrive in datestamp order, so
passing it skips the long pre-2015 runway and starts returning in-scope papers on page one.

### Full text — three sources, in order

| source | share of build | anchors |
|---|---|---|
| `arxiv.org/html/{id}v{n}` | 300,417 (82%) | LaTeXML DOM ids — precise |
| `ar5iv.labs.arxiv.org` | 50,539 (14%) | same scheme |
| PDF via PyMuPDF | 17,521 (5%) | page-level only |

arXiv's native HTML was expected to cover only Dec 2023 onward; in practice they have
backfilled LaTeXML conversions and it serves 2015-era papers too, which is why the PDF
fallback is needed for only 5% of the corpus.

Fetched HTML is kept zstd-compressed under `data/raw/` (~38 GB). That is optional but
strongly recommended: it means a parser bug costs a re-parse rather than another multi-hour
crawl.

### Parsing and chunking

Both HTML sources emit identical LaTeXML conventions, so one parser handles them:

| structure | element | anchor |
|---|---|---|
| section / subsection | `section.ltx_section` | `S3`, `S3.SS2` |
| paragraph (the retrieval unit) | `div.ltx_para` | `S3.p4` |
| numbered equation | `table.ltx_equation` | `S2.E1` |
| figure / table float | `figure` | `S4.F2` |

Anchoring on `div.ltx_para` is what makes a citation clickable: the link `#S3.p4:120-480`
scrolls to that paragraph and highlights those characters.

**Two parsing details that matter more than they look.** Math must be read from each
`<math>` element's `alttext` (the original LaTeX) — `text_content()` interleaves the MathML
rendering with LaTeXML's fallback and produces `NN=8, SNR 10dB, 𝑹\bm{R}` for what should be
`$N$=8, SNR 10dB, $\bm{R}$`. And class matching must be token-based: `contains(@class,
'ltx_authors')` also matches `<article class="ltx_document ltx_authors_1line">`, which is
the entire paper.

Chunks merge consecutive paragraphs up to ~1000 characters, never crossing a section
boundary, with a hard 1500-character ceiling — sentence splitting alone cannot divide
math-heavy prose, where a 3.6k-character appendix paragraph can contain one sentence
boundary. Measured result: **78 chunks per paper**, median 788 characters, mean 219 tokens.

### Embedding

`embeddinggemma-300m`, 768-d, Matryoshka-truncatable. Two files are written per chunk:

- `fp16.bin` — full 768-d, memory-mapped from disk, used for exact rescoring
- `int8.bin` — truncated to 256-d, resident on GPU, used for the first-pass search

Throughput on one RTX PRO 6000, measured:

| configuration | chunks/s |
|---|---|
| fp32, eager | 251 |
| bf16, eager | 469 |
| bf16 + `torch.compile` (default) | 1,027 |
| the above × 3 GPUs | **2,600** |

`max-autotune` is a trap here — 51 chunks/s, nine times *slower* than eager, because
length-sorted batching hands the compiler a new shape almost every batch and it re-tunes
each time.

### Citations

Semantic Scholar's batch endpoint accepts `ARXIV:1706.03762` identifiers directly, which
matters because the arXiv id is already our primary key — no id resolution step. Works
without an API key; 429s are routine on the shared pool and are backed off rather than
treated as errors.

Bibliography scraping from the parsed HTML gives a free head start (~2 edges per paper), but
S2 is what makes the graph useful: **~11 edges per paper**, 6.7M total.

---

## 2. Running it

```bash
# 0. verify disks and GPUs before anything writes
lara preflight

# 1. metadata — the from= is an ordering hint, not a filter (see above)
lara harvest --sets cs,stat --from 2015-01-01

# 2. full text. Resumable: re-run after any interruption and it continues.
lara crawl

# 3. embed chunks, then paper-level title+abstract vectors
lara embed
lara embed-papers

# 4. citation graph (independent of 2 and 3; run it concurrently)
lara citations

# progress at any time
lara status
```

Stages 2, 3 and 4 can run simultaneously — 2 is network-bound, 3 is GPU-bound, 4 is
rate-limited by a remote API. That is how the reference build was produced.

Once vectors exist:

```bash
lara serve --host 0.0.0.0        # reader on :8080
lara serve-llm                   # generator (CUDA only; see below)
```

### Wall-clock on the reference machine

3× RTX PRO 6000, 48-core Threadripper, NVMe:

| stage | time | bound by |
|---|---|---|
| harvest | ~1 hr | arXiv's server (~20s per 1300 records) |
| crawl | ~6 hrs | politeness rate limit |
| embed | ~4 hrs | GPU |
| embed-papers | 9 min | GPU |
| citations | ~5 hrs | S2 rate limit |

The crawl is the long pole and it is *self-imposed*. See rate configuration below.

---

## 3. Configuration

All of it lives in `config.yaml`.

### Scope — which papers

```yaml
corpus:
  categories: [cs.LG, cs.CL, stat.ML, cs.NE]
  date_floor: "2015-01-01"
```

This is the single biggest lever on build time and disk. Approximate arXiv category sizes:

| category | papers |
|---|---|
| cs.LG | 281k |
| cs.CV | 202k |
| cs.AI | 195k |
| cs.CL | 116k |
| stat.ML | 79k |
| cs.NE | 18k |

They overlap heavily via cross-listing, so the union is far smaller than the sum. Adding
`cs.CV` and `cs.AI` roughly doubles the corpus.

Metadata is harvested for **all** of `cs` and `stat` regardless — that is only ~2 GB and it
means changing `categories` later needs a re-flag, not a re-harvest:

```sql
UPDATE papers SET in_scope = (categories LIKE '%cs.CV%' AND submitted_utc >= '2015-01-01');
```

Then re-run `lara crawl`.

To harvest categories outside `cs`/`stat` (e.g. `q-bio`, `eess`), add the OAI set:

```bash
lara harvest --sets cs,stat,eess,q-bio
```

### Crawl rate — the build-time dial

```yaml
ingest:
  fulltext:
    rate_per_sec: 40.0
    max_concurrency: 48
    user_agent: "YourProject/0.1 (+url; mailto:you@example.com)"
```

Measured throughput against arXiv:

| rate limit | papers/s | 429s seen |
|---|---|---|
| 3 | 2.26 | 0 |
| 15 | 7.85 | 0 |
| 40 | 17.11 | 0 |

We never found arXiv's ceiling; at 40 req/s the bottleneck became our own single-threaded
HTML parsing (~72 ms/paper, ~81% of one core).

**Please set a real `user_agent` with a contact address, and consider whether you need to go
this fast.** arXiv is a free public service and asks bulk consumers to use their S3 bulk
access instead. 17 papers/s sustained is real load. The adaptive backoff halves the standing
rate after three consecutive 429s and never raises it again, so an overshoot self-corrects —
but that is a safety net, not permission.

Set `sources: [arxiv_html, ar5iv]` to skip PDFs if page-level anchors are not good enough
for your use.

### Chunking

```yaml
chunking:
  target_chars: 1000
  overlap_frac: 0.15
  never_cross_sections: true
```

Smaller chunks give more precise citations and more vectors; larger give more context per
retrieval and fewer vectors. 1000 characters (~250 tokens) sits comfortably inside
EmbeddingGemma's window while keeping highlights tight enough to be useful. Changing this
requires re-parsing (`lara crawl` after clearing `chunks`) and re-embedding.

### Embedding

```yaml
embedding:
  model: google/embeddinggemma-300m
  dim_full: 768
  dim_truncated: 256
  max_seq_len: 512
  devices: [0, 1, 2]
  compile: default
  batch_size: 512
```

- **`devices`** — list every GPU during a build; drop to `[0, 1]` when serving so vLLM keeps
  a card.
- **`max_seq_len`** — 512 truncates 1.9% of chunks; 384 is 15% faster but truncates 9.2%,
  and a truncated chunk loses its tail, which is where conclusions live.
- **`dim_truncated`** — 256 keeps the resident index at 8 GB per 31M chunks. 128 halves that
  at some recall cost, recovered anyway by the exact fp16 rescore.
- **`compile`** — `default`, never `max-autotune`.

Swapping the model means re-embedding everything. Any sentence-transformers encoder works;
adjust `dim_full` to match.

### Storage

```yaml
disk:
  root: /path/with/space
  min_free_gb: 60
  forbid_paths: [/mnt/full-disk]
```

`lara preflight` resolves every configured path through `realpath` and refuses to start if
one lands on the wrong filesystem — a symlink in the middle of a path is otherwise an easy
way to discover, 30 GB in, that your index is being written to a disk with no room.

Budget for the reference build:

| artefact | size | can you skip it? |
|---|---|---|
| `meta.sqlite` | 40 GB | no — chunk text, BM25, citations |
| `vectors/fp16.bin` | 34 GB | yes, at the cost of exact rescoring |
| `vectors/int8.bin` | 6 GB | no — this is the search index |
| `raw/` | 38 GB | yes, but re-parsing then means re-crawling |
| **total** | **119 GB** | |

---

## 4. Skipping the build

If someone on your network has already built a corpus, copy theirs instead — it is free and
takes an hour rather than a day:

```bash
# on the machine that has the data
lara dataset publish --tiers core,full

# on the new machine
lara dataset fetch http://<their-host>:8080 --tiers core
lara preflight && lara serve
```

Transfers resume if interrupted and are SHA-256 verified. `core` (~46 GB) gives working
search and answers; `full` adds the fp16 vectors for exact rescoring.

---

## 5. Things that will bite you

- **Stop ingest before publishing a dataset.** The corpus is written continuously, so a
  digest taken over a growing file is wrong by the time it is read.
- **`lara embed` drains its queue and exits.** The crawler keeps producing chunks, so re-run
  it periodically until the crawl finishes.
- **Refresh BM25 statistics after large growth.** Term frequencies are materialised for query
  planning; when the corpus went 2.5M → 18.5M chunks the stale table made BM25 45× slower.
  `lara embed` refreshes them at the end of a run.
- **Reload the server after embedding.** The tier-1 index is a GPU snapshot taken at startup;
  new vectors are not searchable until `POST /api/reload`.
- **vLLM has no Metal backend.** Its platforms are cpu/cuda/rocm/tpu/xpu, so on Apple Silicon
  it runs CPU-only. Use llama.cpp, Ollama or LM Studio and point `serving.vllm.base_url` at
  it — the reader only speaks OpenAI-compatible HTTP and never imports vLLM.
- **Retrieval works without a GPU; building the index barely does.** CPU HNSW measured ~7 ms
  at 96% recall, but the embedder is ~50× slower on CPU, turning a 4-hour build into days.
