# UI controls — technical reference

What each control changes in the pipeline, with the measurements behind the defaults.
Companion to `guide_layman.md`, which covers the same surface without the internals.

---

## Retrieval pipeline

Every question runs the same stages. Widths come from `index.rerank` in the config;
`/api/health` reports 29,552,270 vectors on the machine these figures were taken on:

```
query ──embed (20ms)──┬─▶ tier 1: dense over MRL-256                 ─▶ 200
                      │      torch fp16 cuda: 1.1ms, recall@200 1.000
                      └─▶ BM25 over FTS5 (IDF-planned)               ─▶ 200
                                    │  70-180ms
                        reciprocal rank fusion (k=60)
                                    │
                    tier 2: exact fp16-768 rescore from mmap  0.2ms  ─▶ 48
                                    │                     (rerank_candidates × 2)
                    cross-encoder Qwen3-Reranker-0.6B-seq-cls 33ms   ─▶ 24 ─▶ final_k 20
```

Tier 1 is **selectable**, and the default is exact. Seven backend/precision combinations
were measured against real corpus vectors — a torch fp16 matmul is 1.1 ms at recall 1.000
on CUDA and 12.3 ms at 0.995 on CPU, faiss flat is 61.3 ms and *larger*, faiss HNSW is
2.1 ms at 0.979 and 2.5× the memory. `auto` resolves to torch on every platform and faiss
is opt-in. Full table, the Metal figures, the `ef_search` recall curve and
`lara bench-index` are in
[`../retrieval/search_backends.md`](../retrieval/search_backends.md).

None of them holds the whole corpus on a laptop — 7.5 GB at the smallest, 37.9 GB at the
fastest — which is what topic-scoped residency exists for. When a keep-set is active only
its rows enter tier 1; BM25 and the tier-2 rescore stay whole-corpus, so scoping narrows
semantic recall rather than coverage. See
[`../retrieval/corpus_scoping.md`](../retrieval/corpus_scoping.md).

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

The model list follows the **effective generation backend**, not the platform's advisory
one. `Device.backend` says "llama.cpp" on any Mac; `generator.effective_backend()` takes
installation into account and returns MLX where `mlx-lm` is present. They name different
weight formats, so anything listing servable models has to ask the latter. Five backends
exist (`vllm`, `llamacpp`, `mlx`, `ollama`, `external`) — see
[`../setup/generation_backends.md`](../setup/generation_backends.md).

### Deep Automated Research

`#deep-btn` toggles `body.deep-open`: a full-window view over `POST /api/synthesize`, which
is SSE and runs for minutes. It is a survey loop — retrieve, extract structured claims,
decide whether a gap remains, repeat — with no round cap; it ends on saturation or on
consecutive stop votes.

Two real runs on the live corpus, same question, read back from
`/api/synthesis/run/{id}`: 3 rounds / 9 claims / 4 papers / 60.5 s, and 17 rounds /
33 claims / 12 papers / 207.4 s. Mechanism, prompts, the SSE event contract and the
persistence schema are in [`../retrieval/deep_research.md`](../retrieval/deep_research.md).

Aborting the fetch does **not** discard the run: the cancel flag is checked at the top of
each round and the run still consolidates what it gathered.

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

## Conversation threads

Threads are keyed by paper (`arxiv_id`, or `"corpus"`), so opening another paper starts a
fresh conversation.

### The `rewrite` step

The first `step` event of a follow-up, emitted **before any retrieval**:

```
event: step
data: {"kind":"rewrite","detail":"<the standalone query>","original":"why is that?",
       "why":"only 3 words"}
```

The client renders it as `searched for: "<detail>"` in the progress list. Silently searching
for something other than what the reader typed is the kind of helpfulness that erodes trust
when noticed, so the resolved query is always shown.

Reference resolution has to happen before the search, not in the answer prompt — by the time
the prompt is assembled the wrong excerpts have already been fetched. A cheap regex
pre-filter (deictic pronouns, elliptical openers, ≤ 6 words) gates one 120-token call at
`temperature=0.0`; a rewrite under 8 or over 400 characters is rejected as a malfunction and
the original is used.

**Retrieval gets the rewrite; the answering model gets `req.query` verbatim.** The judgement
written to `judgements` uses the resolved query — that is the honest training pair.

### History placement

`build_prompt` puts history between the system block and the excerpts:

```
SYSTEM · FEWSHOT · --- · history · Excerpts: …  |  selection · parts · Question:
└───────────────── stable prefix ──────────────┘  └────────── tail ────────────┘
```

History is append-only across a thread, so everything before the excerpts stays
byte-identical from turn to turn and the prefix cache still hits. Putting it after would
invalidate the excerpts on every question. Four turns by default, each answer truncated to
700 characters.

Chunk ids cited by recent answers (up to 8, newest first) become feedback vectors on the
next retrieval — recall only, never reordering. Full detail, including compression, in
[`../retrieval/conversation_threads.md`](../retrieval/conversation_threads.md).

---

## Context viewer

`<details class="ctx">` under each answer, driven by the `context` SSE event. **Both numbers
come from the generator, not from a local estimate.**

- `count_tokens()` POSTs each prompt segment to `{base_url}/tokenize` and reads `count`. It
  returns `None` on any non-200 rather than guessing — *a viewer that presents guesses as
  facts is worse than one that says it cannot tell.*
- `context_limit()` reads `max_model_len` from `GET /models`.
- `prompt_tokens` and `completion_tokens` come from vLLM's own usage report, requested with
  `stream_options: {include_usage: true}` and stashed on `stream_answer.last_usage`.

Segments are exactly `build_prompt.last_parts`, in prompt order:
`system`, `few-shot`, `history`, `excerpts`, `question`. The bar is proportional to the real
context window and includes two segments that are not prompt: `reserved` (the request's
`max_tokens`) and `free`. The point of the panel is as much what was left unused as what was
spent.

When `/tokenize` is unavailable the server falls back to `round(len(t) / 4)` per segment,
sets `exact: false`, and the panel labels itself **estimated** and says why. The only
client-side constant is a `32768` fallback for `limit`.

The **Compress conversation** button is rendered only when the `history` segment has a
non-zero token count, and it shows that count. *A button in the toolbar would be a feature
nobody connects to the number that motivates it.*

## Throughput

The `perf` SSE event, rendered as one line under the answer:
`N ms to first token · X tok/s · N out · N in`.

```python
gen_s = done_at - (first_token_at or gen_started)
tok_per_sec = round((out_tokens - 1) / gen_s, 1)
```

**Measured from the first token, not from the request.** Including the prefill in a decode
rate makes a long prompt look like a slow model. The first token is excluded from the count
because its emission time is the clock's origin, and `completion_tokens` is the generator's
own count rather than a client tally. `ttft_ms` *is* measured from generation start.
`tok_per_sec` is `null` — and the field is omitted from the line — when fewer than two
tokens were produced.

`lara bench-generate` uses the same convention, so its numbers and the UI's are comparable.

Distinct from this, the status bar shows a client-side `first token Nms` measured with
`performance.now()` from before retrieval, so it includes retrieval and prefill.

---

## Inline LaTeX

66 % of corpus chunks contain `$math$`. Papers themselves render fine — LaTeXML ships MathML
and browsers handle it natively — but chunk *text* is stored as each element's `alttext`,
i.e. the original LaTeX. So every excerpt, answer and extracted claim showed raw
`$O(\epsilon^{-3})$`.

**Deliberately not KaTeX.** Vendoring it means ~1 MB of third-party code and webfonts inside
a proprietary repository, and the maths that actually appears in these answers is inline
notation — subscripts, Greek, big-O, fractions — not displayed multi-line equations. The
command set was **measured rather than guessed**: `frac`, `mathcal`, `in`, `left`/`right`,
`mathbf`, `leq`, `displaystyle`, `bm`, `sum`, `prime`, `text`, `alpha`, `textbf`, `theta`,
`hat`, `mathbb`, `tilde` account for nearly all of it. KaTeX drops into `texToHtml`
unchanged if full fidelity is ever wanted.

What it covers:

| construct | handling |
|---|---|
| `\frac`, `\dfrac`, `\tfrac` | nested `<span class="mfrac">` with numerator/denominator |
| `\sqrt` | `√` plus an overlined span |
| `^`, `_` | `<sup>` / `<sub>`, recursively |
| ~180 symbol macros | Greek both cases, relations, set/logic, arrows, `\sum \prod \int`, delimiters, spacing |
| `\mathbb` | ℝ ℕ ℤ ℚ ℂ 𝔼 ℙ 𝟙, else a styled span |
| `\mathcal`, `\mathscr` | full A–Z script alphabet |
| `\mathbf`, `\bm`, `\boldsymbol`, `\textbf` | `<b>` |
| `\text`, `\mathrm`, `\textrm`, `\operatorname` | upright span |
| `\hat \tilde \bar \vec \dot \widehat \widetilde` | Unicode combining marks over the glyph |
| 24 upright function names | `log exp min max argmin argmax sin cos tan det dim ker deg gcd lim sup inf ln Pr tr diag softmax sign rank` |
| `\left \right \displaystyle \limits \! \, \; \:` | dropped; sizing hints with no meaning outside a typesetter |
| `\lx@sectionsign` | `§` — a LaTeXML internal, common in `alttext` |
| unknown `\command` | rendered as its own name, upright |

Groups are read by a depth counter, not a regex: `\frac{\frac{a}{b}}{c}` is common and
nesting defeats a regex.

Delimiters recognised: **`$…$`** (inline, cannot cross a newline) and **`$$…$$`** (display).
`\(…\)` and `\[…\]` are **not** recognised. Both inline delimiters carry a `(?<!\\)`
lookbehind so an escaped `\$` is not a delimiter.

### The prose trap

Chunking can split a formula, leaving a chunk with an odd number of `$`. Pairing then runs
from one formula's closing delimiter to the next one's opening one and swallows the sentence
between them — the observed case was `$X$ -QAM scheme, where $Y$` rendering the prose as
maths.

```js
function looksLikeProse(src) {
  if (/[\\^_{}]/.test(src)) return false;
  const words = src.trim().split(/\s+/);
  return words.length >= 4 || src.length > 60;
}
```

Any of `\ ^ _ { }` present means it is definitely maths. Otherwise four or more words, or
over 60 characters, is prose and is returned **with its delimiters intact**. Real inline
maths is short and carries at least one LaTeX marker.

`renderMath` runs on **already HTML-escaped** input — it emits tags, so running it on raw
text would let a passage containing markup inject it — and it therefore runs *before*
citation linking. Anything it cannot express degrades to the source text with the delimiters
removed, which is what was displayed before: never worse, usually much better.

---

## Library graph

`GET /api/library/graph` summarises every thread into a node and connects them. Two model
calls over the whole library, cached against a SHA-1 fingerprint of the question entry ids
— `hashlib`, not `hash()`, because `PYTHONHASHSEED` is randomised per process and a built-in
hash would produce a cache that can never hit. `?refresh=1` forces a rebuild; `#lib-rebuild`
is that button.

Edges always point **forward in time**, which makes the graph a DAG by construction and lets
it be laid out git-log style in lanes by longest-path depth. The direction is enforced, not
trusted: a backward edge is swapped rather than dropped, because the model is reliable about
noticing a relationship and unreliable about which way it runs. The relation vocabulary is a
closed set of seven; anything else becomes `same-topic`.

Clicking a node — or any saved question in list mode — restores the **whole thread**, keyed
by paper exactly as the server keys it. Detail in
[`../retrieval/conversation_threads.md`](../retrieval/conversation_threads.md#part-2--the-library-graph).

---

## Taste profile

`★ Interesting` on a selection records a chunk in the taste marks; the **For you** pane
scores against them.

Reductions are `max` and `lse` (log-sum-exp, `temp=0.05`, the default), **not** `mean` or
`sum`. Adding cosine similarities is identical to searching once with the summed vector, so
the collapsing reductions are pointless here — the whole point is that distinct interests
must stay distinct, and a mean would rank a mediocre generalist above a perfect match for
one of them.

`/api/taste/paper/{id}` scores within the open paper; the **corpus** button switches to
`/api/taste/recommend`, which runs `search_multi` over the resident tier-1 index. Measured
on this corpus: streaming the 15.1 GB matrix dominates, so **50 taste vectors cost ~16 ms
against ~11 ms for one** — the profile size is nearly free, and LogSumExp at 50 marks
(~52 ms) is what carries the cost.

---

## Temperature

Passed to vLLM. Prompts are laid out stable-prefix-first
(`system + few-shot → history → excerpts → selection → question`) so prefix caching survives
follow-up questions about the same paper; temperature does not affect that.

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
`synthetic` (exploration runs), and `llm_scope` — model relevance verdicts, written by
hierarchical scope (`source='hierarchy'`) and by deep research (`source='synthesis'`).

`llm_scope` is the only teacher that produces **negatives**. Passages retrieval surfaced and
a model then judged unhelpful are hard negatives, and ordinary usage never yielded any:
everything retrieval returns scores above threshold, so the first real queries against this
system produced 24 positives and zero negatives. Deep research generates them in bulk, one
rejected excerpt at a time.

Full detail, including how to inspect or delete it, is in
`docs/finetuning/gemma_embedder_finetuning.md`.

---

## Endpoints

| endpoint | purpose |
|---|---|
| `GET /api/health` | readiness, warmup timings, vector count, active scope (or `null`) |
| `GET /healthz` | liveness only — **reachable without a token** |
| `GET /api/paper/{id}` | metadata + sanitised LaTeXML body + section list |
| `POST /api/fetch/{id}` | on-demand crawl, parse, embed for one paper |
| `POST /api/retrieve` | retrieval only — used for speculative prefetch |
| `POST /api/ask` | SSE: `step`, `hits`, `coverage`, `clarify`, `token`, `context`, `perf`, `grounding`, `error`, `done` |
| `POST /api/synthesize` | SSE deep research: `start`, `round`, `claims`, `round_done`, `decision`, `consolidating`, `token`, `error`, `done` |
| `GET /api/synthesis/runs` / `run/{run_id}` | past runs; one run in full |
| `GET /api/library/graph` | conversations as a DAG; `?refresh=1` rebuilds |
| `POST /api/thread/compress` / `uncompress` | fold older turns into a summary, or restore |
| `GET /api/thread/state` | summary, live/compressed/total turns, `history_chars` |
| `POST /api/search` | paper search + induced citation subgraph |
| `POST /api/heatmap` | per-chunk relevance within one paper |
| `GET /api/graph/{id}` | ego network with similarity shading |
| `GET /api/breadth` | the depth spectrum |
| `GET /api/device` / `meminfo` | platform, accelerator, memory budget; resident breakdown |
| `POST /api/model/resolve` / `download`, `GET /api/model/download/{repo}` | HF lookup, fetch, progress |
| `GET`/`POST`/`DELETE /api/taste*` | taste marks, per-paper and corpus recommendations |
| `GET`/`PUT /api/settings/prompt` | the editable answer instructions |
| `GET`/`POST`/`DELETE /api/memory*` | the library: visits, questions, folders |
| `POST /api/click` | records a followed citation |
| `POST /api/reload` | rebuild the GPU index after an embed run |
| `GET /api/dataset/manifest` / `file/{name}` | LAN corpus distribution |

**Authentication is bearer-token and fail-closed.** `lara serve` refuses to bind off
loopback without a token, and accepts it as `Authorization: Bearer`, as a one-shot `?token=`
that is traded for an `httponly` cookie by a 303 redirect, or as that cookie. `/healthz` is
the only reachable path without one.

Two traps: `serving.auth.mode: off` written unquoted becomes the YAML boolean `False`, which
is honoured as "off" rather than silently falling back to `auto`; and the middleware is
installed **at import time from `LARA_TOKEN`**, so running `uvicorn lara.serve.app:app`
directly with no such variable gives no authentication whatever the config says. Detail in
[`../setup/authentication.md`](../setup/authentication.md).
