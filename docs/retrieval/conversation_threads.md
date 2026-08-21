# Conversation threads and the library graph

What the previous questions were, what "it" refers to, and how the whole library becomes a
graph of investigations rather than a list of queries.

Implementation: [`lara/serve/thread.py`](../../lara/serve/thread.py) and
[`lara/serve/library_graph.py`](../../lara/serve/library_graph.py). Endpoints:
`POST /api/thread/compress`, `POST /api/thread/uncompress`, `GET /api/thread/state`,
`GET /api/library/graph`.

---

# Part 1 — Threads

## 1. The problem

Every question used to be asked in isolation: no prior turn reached the request, the
prompt, or the retriever. Asking *"why is that?"* embedded those three words and searched
29.5 M chunks for them, which matches nothing in particular. The answer prompt then saw a
fresh question with unrelated excerpts and no sign an earlier exchange had happened.

The exchanges were already being stored — the library holds every question with its answer,
paper and selection — and simply never read back.

Two distinct problems, fixed in different places, because fixing only one leaves the
feature broken in a way that is hard to see.

## 2. Reference resolution happens BEFORE retrieval

This is the half that history-in-the-prompt cannot fix. By the time the answer prompt is
assembled, the wrong excerpts have already been fetched.

In `/api/ask`, the rewrite is the first thing that happens, before decomposition and before
any search:

```
turns ─▶ looks_like_followup? ─▶ rewrite ─▶ [decompose] ─▶ retrieve
                                    │
                                    └─▶ SSE: step{kind:"rewrite"}
```

### The pre-filter

`looks_like_followup(question) -> (bool, why)` is cheap and **deliberately generous**:

| trigger | `why` |
|---|---|
| ≤ 6 words | `"only N words"` |
| opens with `and\|but\|so\|also\|what about\|how about\|why\|why not\|how so\|and then\|go on\|continue\|more\|explain\|elaborate\|expand\|say more\|really\|compared to\|versus\|vs` | `"opens with a continuation"` |
| contains `it\|its\|it's\|that\|those\|these\|this\|they\|them\|their\|he\|she\|his\|her\|the former/latter/first/second/other/same/above/previous\|there\|then\|such` | `"contains a reference to something unstated"` |

> A rewrite that was not needed costs one short model call and returns the question roughly
> unchanged, while a missed follow-up silently retrieves the wrong passages and produces a
> confident answer about the wrong thing. The asymmetry is not close.

### The rewrite

`temperature=0.0`, `max_tokens=120`, against the **last two** turns only. The prompt states
what the rewrite is for:

> The rewritten question is used to SEARCH a corpus of papers. It is not shown to the reader
> as their question, and it is not answered directly — so it should read like a search query
> stated as a full question, carrying every noun the search needs.

and what a bad rewrite causes: *"If you add a term the reader did not mean, the search goes
somewhere they did not ask about, and the answer will be confidently about the wrong
subject."*

The result is sanity-checked before use. `< 8` or `> 400` characters is rejected as
`"rewrite unusable"` — a blunt check, but it reliably catches the two ways this fails: an
empty answer, and a model that starts explaining itself instead of rewriting. Identity to
the original is reported as `"already standalone"`. Every failure path falls back to the
original question; `rewrite()` never raises.

### The asymmetry that matters

**Retrieval gets the rewrite. The answering model gets what the reader typed.**

```python
# app.py /api/ask
search_query["q"] = rw["query"]      # used to retrieve
...
stream_answer(s.cfg, req.query, hits, ..., history=history)   # req.query, not the rewrite
```

Retrieval needed a self-contained query; the answer must address the question actually
asked. History is in the prompt, so the model can resolve the reference itself. The
judgement written to `judgements` uses the **resolved** query — that is the honest training
pair; the pronoun the reader typed is not.

## 3. History goes in the cached region

The prompt is laid out stable-prefix-first so a prefix cache survives follow-ups about one
paper. History is append-only across a thread, so it belongs immediately after the system
block and **before** the excerpts, which change every turn:

```
SYSTEM · FEWSHOT · --- · history · Excerpts: … | selection · parts · Question:
└──────────────── stable prefix ────────────────┘ └────────── tail ──────────┘
```

Putting history after the excerpts would invalidate the excerpt cache on every question
instead.

`history_block()` truncates each answer to 700 characters at a word boundary, because its
job here is to establish what was discussed, not to be re-read in full — an untruncated
thread would crowd out the excerpts that carry the evidence. It ends with an explicit
instruction: *"The reader can see the above. Do not repeat it; answer the new question, and
use the earlier exchange only to understand what they are referring to."*

Default window: **4 turns** (`DEFAULT_TURNS`). Enough to resolve a reference; few enough
that the prefix stays small.

## 4. Threads are keyed by paper

`thread_id(arxiv_id) -> arxiv_id or "corpus"`. Opening a different paper starts a new
thread, because most reading is several short conversations rather than one long one, and a
global history would drag a question about optimisers into an answer about tokenisation.

## 5. Feedback vectors from cited chunks

`prior_chunk_ids(turns, cap=8)` scrapes `[\d{3,}]` citation markers out of recent answers,
newest first, and `/api/ask` turns them into feedback vectors for the retrieval.

> A follow-up usually concerns a passage already on screen, so the passages the last answers
> actually cited are strong candidates for this one — and they cost nothing to reuse, having
> already been retrieved and reranked.

These fuse as extra ranked lists exactly as in
[hierarchical scope](hierarchical_scope.md#4-confirmed-passages-become-query-vectors):
recall only, never reordering the final answer.

## 6. Compression

A long thread eventually costs more prompt than it is worth. `POST /api/thread/compress`
folds the older turns into a summary and keeps the recent ones verbatim.

```json
{"paper": "1706.03762", "model": null, "keep": 2}
```

`KEEP_VERBATIM = 2`. The most recent turns are left alone deliberately: **they are what
follow-ups point at, and compressing them is what would break reference resolution — the
opposite of the point.**

The prompt is unusually blunt about consequences, because a summary is destructive:

> The summary REPLACES the exchanges it covers. Nothing else from them survives … Anything
> you leave out is gone. If a method, dataset, number or comparison was discussed and you
> omit it, a later question about it will be answered as though it were never mentioned.
> Anything you add that was not discussed will be treated as established fact.

Priorities, in order: named entities (methods, papers, datasets, metrics, authors);
conclusions actually reached with their numbers and conditions; what the reader seemed to
be pursuing, so a vague follow-up still has a subject; anything explicitly ruled out. Under
200 words, compact notes rather than prose. `temperature=0.1`, `max_tokens=500`.

Refusals rather than bad summaries: `{"ok": false, "reason": "only N turns; nothing to
compress"}` when there is nothing older than `keep`, and `"the model returned no usable
summary"` when the result is under 40 characters.

### It reports coverage, not just characters

```json
{"ok": true, "summary": "...", "compressed_turns": 6, "kept_verbatim": 2,
 "chars_before": 3120, "chars_after": 1840, "saves_chars": true,
 "turns_before": 4, "turns_after": 8}
```

Two honesty details are built into those numbers:

- `chars_before` is measured against `history_block(turns[-DEFAULT_TURNS:])` — **what the
  model was actually being sent**, not the whole thread. Measuring against all of it would
  report a saving that never existed.
- What compression actually buys is **coverage**: every turn is represented
  (`turns_after`), instead of a sliding window of the last four (`turns_before`). On a short
  thread that costs more characters than it saves, and `saves_chars: false` says so rather
  than implying a win that is not there.

Once compressed, `turns_for()` skips the covered turns by default — they are represented by
the summary now, and sending both would cost the tokens compression was invoked to reclaim.
In the prompt the summary is introduced as *"Summary of earlier turns in this
conversation"* followed by *"Most recent exchanges"*, because that is what it now is: the
exchanges are gone and only these notes can resolve a reference back to them.

Compression is **stacking**: an existing summary is fed back in as *"Existing summary of
even earlier turns"* when compressing again.

`POST /api/thread/uncompress` with `{"paper": …}` drops the summary and restores the full
history. It returns `{"ok": bool}` — false if there was no summary.

### `GET /api/thread/state`

```
GET /api/thread/state?paper=1706.03762
{"thread":"1706.03762","summary":"","compressed_turns":0,
 "live_turns":1,"total_turns":1,"history_chars":562}
```

Omit `paper` for the corpus thread. `history_chars` is the exact length of the block that
will be prepended to the next prompt, which is what the UI's compress button is sized
against.

## 7. Where compression is offered in the UI

Inside the context viewer under an answer, and only when the history segment has a non-zero
token count:

> Offered where the cost is actually visible. A button in the toolbar would be a feature
> nobody connects to the number that motivates it.

See [`../UI_guide/guide_technical.md`](../UI_guide/guide_technical.md#context-viewer).

## 8. Restoring a thread from the library

Clicking a saved question in the library rebuilds the **whole thread**, not the single
exchange — every question entry with the same `arxiv_id`, oldest first. The clicked turn is
marked, the others are shown as restored context, and the input placeholder changes to
*"Continue this thread (N questions)…"*. Threads are keyed by paper in the client exactly as
they are on the server, so what is restored is what the next question will actually be
answered against.

---

# Part 2 — The library graph

## 9. Why a graph

A flat list answers "what did I ask?" and nothing else. What a reader wants back is *why*
they asked it — which question led to which, what turned out to be the same investigation
under two names, and where a thread from last week already answered the one they are about
to start.

```
node   one conversation (a paper's thread, or the corpus thread), with a label the
       model chose, a one-line summary, and the questions it contains
edge   a logical connection, always pointing FORWARD IN TIME
```

## 10. Forward in time is structural, not decorative

Every edge runs from an older conversation to a newer one, which makes the graph a **DAG by
construction** — no cycle can form. That is what lets it be laid out in columns by depth
and read as the order the reader's thinking actually went. An undirected "these are related"
graph is a hairball that says much less.

The direction is **enforced, not trusted**:

```python
if src.first_utc > dst.first_utc:
    src, dst = dst, src
```

> The model is reliable about noticing a relationship and unreliable about which way it
> runs, and one backward edge turns a readable DAG into a cycle that no layout can order.

Also filtered: edges whose endpoints are out of range, self-edges, and duplicate
`(source, target)` pairs.

`_assign_depth()` computes longest-path depth by relaxation — safe because the edges are
already acyclic; the `len(nodes)` loop bound is belt and braces against a malformed edge
set rather than a real possibility.

## 11. The relation taxonomy

A **closed set**, which keeps the legend readable and stops the graph acquiring fifty
synonyms for "related to". Anything the model returns outside it becomes `same-topic`.

| relation | meaning |
|---|---|
| `follows-up` | the later one continues the earlier investigation |
| `applies` | the later one uses a method or result from the earlier |
| `compares` | the later one weighs the earlier subject against something else |
| `contradicts` | the later one found something that conflicts with the earlier |
| `background-for` | the earlier one is prerequisite understanding for the later |
| `same-topic` | the same subject revisited, without a clear dependency |
| `diverges` | started from the earlier one but went somewhere unrelated |

The edge prompt asks for restraint explicitly: *"Be sparing. Only connect conversations
with a real relationship; an unconnected node is a perfectly good outcome and far better
than a false arrow. Do not connect everything to everything."*

That is not a hypothetical. On the live library today, three conversations produced **zero
edges** — two about optimiser theory and one about embedding losses, correctly judged
unrelated rather than joined by keyword overlap.

## 12. Labels

A separate call, over up to the first 6 exchanges of each thread. It is told what the label
is for:

> the label becomes a node in a graph the reader navigates instead of a list. It is the only
> text they see before clicking, so it must distinguish this conversation from the others —
> "Muon optimizer" is useless if four conversations are about Muon; "Muon sample complexity
> bounds" is not.

Per node: `label` (3–6 words, capped 60 chars), `summary` (one sentence, 240),
`topics` (2–4 lowercase tags, 24 chars each). Labelling failure falls back to the first 48
characters of the thread's first question, rather than showing an unnamed dot.

Real output from the live library:

```
2509.14562  "LiMuon sample complexity and STORM"
            topics: limuon, sample complexity, storm optimizer, 6 questions
2509.20354  "EmbeddingGemma multi-objective training losses"
            topics: embeddinggemma, loss function, training objective, 1 question
```

## 13. Caching

The whole thing costs **two model calls over the entire library**, and the answer only
changes when a new question is asked. It is cached in the library store against a
fingerprint:

```python
digest = hashlib.sha1("\n".join(sorted_question_ids).encode()).hexdigest()[:12]
return f"{len(ids)}:{digest}"
```

`hashlib`, **not** `hash()`: `PYTHONHASHSEED` is randomised per process, so a built-in hash
would produce a different fingerprint in every worker and after every restart — a cache
that can never hit, silently rebuilding the whole graph on each request.

`GET /api/library/graph` returns the cached graph with `"cached": true`;
`?refresh=1` forces a rebuild. An empty graph is never stored.

## 14. The UI

The Library pane has a **graph / list** toggle and, in graph mode, a **↻** rebuild button
(`?refresh=1`). The chosen mode persists in `localStorage`.

Graph mode is laid out **like a git log**: one node per row, oldest at the top, with edges
drawn as lanes down the left (capped at 4 lanes) and coloured by relation. Each node shows
its model-chosen label, its topic tags, and its question count. Clicking one restores that
whole thread into the chat pane.

## 15. Things worth knowing

- **The graph covers question entries only.** Papers you merely opened are in the library
  but are not nodes.
- **The corpus thread is one node**, `id: "corpus"`, however many unrelated questions were
  asked with no paper open.
- **Both model calls are best-effort.** Unparseable JSON yields no labels or no edges, and
  the graph still renders — with fallback labels and no arrows.
- **Compression does not affect the graph.** It reads the raw library entries, so a
  compressed thread still contributes all of its questions as a node.
- **`GET /api/library/graph` blocks for the two model calls on a cold cache.** On a large
  library that is the slowest read endpoint in the app.
