# Tier-1 search backends

Which engine holds the vectors, what each one costs, and why `auto` never picks faiss.

Implementation: [`lara/index/backends.py`](../../lara/index/backends.py). Residency plumbing:
`DenseIndex` in [`lara/index/search.py`](../../lara/index/search.py). Benchmark:
`lara bench-index`.

---

## 1. The problem

There used to be exactly one implementation: convert the int8 file to fp16, put all of it
on a CUDA device, matmul. That is the fastest option on a machine with a discrete GPU and
impossible on most others — the code called `torch.cuda` unconditionally on the way there,
and 28.7 M chunks at 256 dims is 14.7 GB in fp16, which no laptop will hold.

Two axes are now selectable, and `lara setup` asks about both:

```
backend    torch | faiss          (auto resolves to torch)
precision  fp16  | int8
faiss.kind flat  | sq8 | hnsw     (only consulted when backend is faiss)
```

## 2. Measured

From the module docstring. **Measured on 1–2 M real corpus vectors, recall@200 against
exact fp16.** The full-memory column is the 2 M figure scaled by 14.8x, which is the
current corpus: `/api/health` reports 29,552,270 vectors today.

| backend | p50 | recall@200 | full corpus | verdict |
|---|---|---|---|---|
| `torch` fp16 cuda | 1.1 ms | 1.000 | 15.1 GB | fastest anywhere; the CUDA default |
| `torch` int8 cuda | 8.2 ms | 0.996 | 7.5 GB | half the memory for 0.4 % recall |
| `torch` fp16 cpu | 12.3 ms | 0.995 | 15.1 GB | **best exact CPU option** |
| `torch` int8 cpu | 282 ms | 0.995 | 7.5 GB | dequant is not vectorised on CPU — avoid |
| `faiss` flat | 61.3 ms | 0.996 | 30.3 GB | slower *and* larger than torch fp16 |
| `faiss` sq8 | 372 ms | 0.991 | 7.5 GB | smallest, far too slow |
| `faiss` hnsw | 2.1 ms | 0.979 | 37.9 GB | fastest at scale, but memory-hungry |

Two of these contradicted the assumptions the module was first written with.

**faiss is not automatically the right CPU backend.** A plain torch matmul on CPU beats
faiss flat by 5x and faiss sq8 by 30x, at lower memory than flat. Faiss earns its place
only through HNSW, whose sublinear search is in another class — and which pays for it in
memory and a slow build. `choose_backend("auto")` therefore returns `"torch"` on every
platform, and faiss is opt-in.

**int8 is a GPU optimisation, not a memory trick you can apply anywhere.** On CUDA it
halves memory for 0.4 % recall. On CPU the same code path costs 282 ms, because the
per-block dequantisation a GPU absorbs is the dominant cost without one. Selecting it on
CPU emits a `RuntimeWarning` naming the figure and pointing at `lara corpus scope` instead.

Metal is worse than the CUDA number implies, and `lara/setup.py` carries a separate table
for it (`P50_BY_DEVICE`): measured on 2.76 M rows at dim 256 on an M-series Mac, torch
int8 is **403 ms** against **28.9 ms** for fp16 — 14x slower to save 0.70 GB. The wizard
quotes those figures on a Mac rather than the CUDA ones.

## 3. Precision costs less than it looks

Tier 1 only has to produce a shortlist. Every survivor is re-scored at exact fp16-768 from
the mmap in tier 2, then reranked by a cross-encoder. int8 residency loses a little
ordering fidelity in a stage whose output is re-ordered twice more downstream — 0.996
recall confirms it. The same logic is why HNSW's 0.979 is a defensible choice and not a
broken one.

`FaissConfig.ef_search` defaults to **512**, not faiss's usual 64. Measured recall@200 on
1 M real vectors: ef 64 → 0.750, 128 → 0.872, 256 → 0.938, 512 → 0.979, 1024 → 0.993, at
0.3 / 0.6 / 1.1 / 2.1 / 4.1 ms respectively. The old default of 64 discarded a quarter of
the true top-200 to save 1.8 ms.

## 4. No backend fits a laptop

The smallest usable configuration is 7.5 GB and the fastest is 37.9 GB, against maybe
10 GB usable on a 16 GB Mac. Topic-scoped residency is therefore not a nicety on that
hardware, it is the enabling feature — see
[`corpus_scoping.md`](corpus_scoping.md). At `keep=0.05` the corpus is ~1.4 M chunks, where
torch fp16 needs 0.7 GB and HNSW 1.8 GB, both in single-digit milliseconds.

## 5. Residency: `row_ids`

Both backends inherit `_Resident`, which is the whole of the partial-residency contract:

```python
to_local(global_rows)  -> local indices, dropping rows that are not resident
to_global(local_idx)   -> global vector rows
```

Implemented once, because this is where a silent bug would live: returning a local index
where a caller expects a global vector row points at the wrong passage without raising
anything. `row_ids` must be sorted ascending — `to_local` binary-searches it — and
`_prepare_rows` enforces that with `np.unique` if it is handed anything else.

Every backend method returns **global** rows, so residency is invisible to callers.

## 6. Subset search never goes through faiss

Scoping to a paper or a citation neighbourhood hands the index explicit rows. Brute force
over a gathered handful is exact, fast, and free of the `IDSelector` support differences
between faiss versions and index types. `FaissBackend.search(rows=...)` therefore ignores
the faiss index entirely and does a numpy matmul against the retained int8 source. Faiss
is used only for whole-index search.

## 7. `search_multi` and its asymmetry

Scoring every row against K reference vectors and reducing to one score per row (used by
the taste profile). `reduce` is `max`, `sum`, `mean` or `lse` (log-sum-exp, the default,
`temp=0.05`).

On torch this is exact. Measured on 29.6 M × 256 resident fp16: **~11 ms for one vector
and ~16 ms for fifty**, because streaming the 15.1 GB matrix dominates and the extra
columns ride along almost free. LogSumExp costs more — **~52 ms at fifty** — and that is
the exp/log over the (block × K) block, not the matmul.

On faiss it is **approximate in a way the torch path is not**: faiss has no
reduce-across-queries primitive, so the merge happens on the host, and a row that scores
moderately against every reference but tops none of their individual result lists is never
seen. Each search is widened 4x to limit that.

`sum` and `mean` are not worth calling on either backend: adding cosine similarities is
identical to searching once with the summed vector, so the caller should collapse them and
use `search`. `max` and `lse` are the reductions that do not collapse.

## 8. The Metal correctness limit

`SEARCH_BLOCK_BYTES` is a **correctness limit on Metal, not a tuning knob.**

```python
SEARCH_BLOCK_BYTES = {"mps": 64_000_000, "cuda": 1_000_000_000, "cpu": 512_000_000}
```

Under memory pressure MPS does not raise: it returns wrong answers. Measured on an
M-series Mac, a 2.5 GB block (2.47 M rows at dim 256, which the old fixed 4 M-row default
produced) yielded a normalised tensor reporting 1,082,002,458 non-zero elements out of
128,000,000 — an impossible count — scores that were uniformly 0.0, and NaN at larger
sizes. `topk` over that returns uninitialised indices, which surfaced as
`IndexError: index 4431045198971321921 is out of bounds` from row-id translation: three
frames from the cause, and looking nothing like an allocation failure.

256 MB was still too much in practice, because `normalize(raw.float())` holds the cast and
the normalised copy at once and the peak is twice the figure. 64 MB keeps the peak near
128 MB.

`TorchBackend._degenerate()` checks for exactly those two shapes (non-finite, or a
uniformly zero score vector) and `_scores_for` **retries at a quarter the block size, up to
three times**, before raising. The failure is memory pressure and pressure is transient —
a generator loading 5 GB alongside the index tips it over, and the same query succeeds
once the block is small enough.

Only the int8 path allocates a large float32 transient per search, so only the int8 path
can hit this. That is a second reason the wizard never recommends int8 off CUDA.

## 9. `lara bench-index`

```bash
lara bench-index                                    # 2M vectors, 20 queries, k=200
lara bench-index --n 0                              # whole corpus
lara bench-index --only 'torch fp16,faiss hnsw'
```

| flag | default | meaning |
|---|---|---|
| `--n` | 2000000 | vectors to benchmark over; `0` = whole corpus |
| `--queries` | 20 | queries to time |
| `--k` | 200 | candidates per query, as tier 1 is actually used |
| `--only` | – | comma-separated subset of the row labels |

Candidate labels are exactly `torch fp16`, `torch int8`, `torch fp16 cpu`, `faiss flat`,
`faiss sq8`, `faiss hnsw`. faiss rows are skipped with a notice if faiss is not installed.

**Recall is measured against the first backend that ran, which is exact fp16.** That is
the definition of correct here — not ground truth, which tier 1 was never trying to
produce on its own. Consequence worth knowing: `--only 'faiss hnsw'` reports recall 1.000,
because HNSW became its own reference. Always leave `torch fp16` in the set.

Queries are drawn from the corpus itself (`rng.default_rng(0)`, so the sample is
reproducible), which means every query has a guaranteed exact match. A backend that cannot
build is reported as a row with the exception type rather than aborting the run.

The output ends with the linear scale factor to the full corpus, and a reminder that the
winner goes into `config.local.yaml` — the command does not write it for you.

## 10. Configuration

```yaml
index:
  backend: auto          # auto | torch | faiss   (auto -> torch)
  precision: fp16        # fp16 | int8
  faiss:
    kind: hnsw           # flat | sq8 | hnsw
    m: 32                # hnsw graph degree
    ef_construction: 200
    ef_search: 512
    threads: 0           # 0 leaves faiss's default
```

```bash
lara config set index.backend faiss
lara config set index.faiss.kind hnsw
lara config set index.precision fp16
```

These four keys are enum-validated by `lara config set`, so `fiass` is rejected at write
time rather than silently falling through to the default at load time. See
[`../setup/configuration.md`](../setup/configuration.md).

Index changes take effect on restart; `lara config set` says so for any key under
`index.` or `embedding.`.

## 11. Things worth knowing

- **`make_index` falls back loudly, never fatally.** `backend: faiss` without faiss
  installed emits a `RuntimeWarning` naming `pip install 'lara[cpu]'` and builds the torch
  matmul instead.
- **`precision: int8` with `faiss.kind: flat` is silently upgraded to `sq8`**, because a
  flat index stores float32 regardless and asking for int8 memory from it is incoherent.
- **The memory column is the index only.** The embedder, the cross-encoder reranker and
  the whole-corpus `vector_row → chunk_id` map are resident on top of it. `lara setup`
  spells the addends out; `overhead_gb()` in `lara/setup.py` computes them.
- **The row map does not shrink when the corpus is scoped.** At 4 bytes per row it is
  0.12 GB for the current corpus — no longer material, but it is the one fixed cost the
  keep fraction cannot touch.
- **Retrieval mmaps the int8 file lazily** when the backend is faiss or a keep-set is
  active, since a scoped index gathers only a few percent of the rows and reading 7.3 GB
  to use 5 % of it is pure startup cost.
