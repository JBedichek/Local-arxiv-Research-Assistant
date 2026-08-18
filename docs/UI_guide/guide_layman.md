# Using the reader

What every control does, in terms of what you will notice. No jargon.

---

## The three panes

**Left** is a citation graph, **middle** is the paper, **right** is where you ask questions.

Drag the thin dividers to resize any of them. Double-click a divider to collapse that pane
entirely and give the space to the paper. Or use the keyboard:

| key | does |
|---|---|
| `[` | hide/show the graph pane |
| `]` | hide/show the question pane |
| `\` | hide/show both — the paper takes the whole window |
| `/` or `Ctrl-K` | jump to the question box |
| `Ctrl-Enter` | send your question |

Sizes are remembered, so the layout you set is the one you get next time.

---

## The box at the top

It does two different things depending on what you type.

- Type **an arXiv number** (`1706.03762`, or paste the URL) and that paper opens.
- Type **anything else** and it searches the whole corpus for papers about it.

There is no mode to switch. It works out which you meant.

### If the paper has not been downloaded yet

Most papers in the index have only their abstract. When you open one, the reader fetches
the full text from arXiv on the spot — you will see "Fetching full text…" for a few seconds
and then the paper appears. It is now permanently available and searchable.

If you open many uncrawled papers quickly, this will feel slow, because the reader is being
polite to arXiv's servers. At reading speed you will not notice.

---

## Search results

Results come back as a **citation graph** rather than a list, because a list tells you what
matched and a graph tells you how a field fits together.

Each row is one paper. Reading it:

- **Left to right** is time — older papers on the left, newer on the right.
- **Arrows point backwards in time**, from a paper to something it cites.
- **Bigger dots** are cited more often *by the other results* — usually the foundational
  work for that topic.
- **Warmer colours** are more relevant to what you asked.

Click anywhere on a row to open that paper. Hover for the abstract.

**"Show top N papers"** controls how many results appear. More results means more
connections visible, but a busier picture. It re-runs the search, so the arrows are always
correct for the set you are looking at.

**List** switches to a plain ranked list, which is better for quickly scanning titles.

---

## Asking questions

Two ways:

1. **Type a question** in the right-hand pane. It searches everything.
2. **Highlight a passage** in the paper first, then click **"Ask about this"**. Your
   question is then answered with that passage in mind.

Highlighting alone does nothing — you have to click the button. This is deliberate: people
highlight text to read it, copy it, or keep their place, and hijacking that would be
annoying. Once staged, the `×` next to the passage removes it.

### What the answer tells you

Above each answer is a coloured badge:

- 🟢 **Sources answer this** — the excerpts found contain the answer.
- 🟠 **Partially answered** — some of it is supported; it will tell you what is missing.
- 🔴 **Not answered by sources** — it could not find the answer and says so, then summarises
  the closest material it did find.

That last one is a feature, not a failure. A confident wrong answer is worse than an honest
"I could not find this."

Below the answer is a line like *"✓ all 3 cited statements matched their source"*. The
system re-checks its own citations and flags any sentence whose cited passage does not
actually support it. If it flags something, click through and judge for yourself.

**The numbered links** — `[1]`, `[2]` — are clickable. They open the paper at exactly the
passage used and highlight it.

---

## Answer depth

A single slider trading speed for thoroughness.

| setting | roughly | what changes |
|---|---|---|
| Instant | ~5s | one search, fewest excerpts |
| Fast | ~5s | one search, more excerpts |
| **Balanced** | ~6s | may search twice, reads around confusing excerpts |
| Thorough | ~7-10s | up to three searches, may ask you to narrow a vague question |
| Exhaustive | ~10-18s | up to five searches, widest net |

The times are measured, not estimated. Note how little the bottom three differ: writing the
answer takes about 3 seconds no matter what, so the slider is buying *search depth*, not
raw speed.

At Balanced and above, a compound question like *"How does X compare to Y, and which is
faster?"* is split into separate searches, because searching for both at once finds neither
well.

While it works, you will see what it is doing — "Searching again…", "Reading around 3
excerpts for context…". A **Stop** button appears if you want to cut it short.

---

## Scope

Where to search when you ask a question:

- **Whole corpus** — everything indexed. The default.
- **This paper** — only the paper you have open. Use it for "what does this paper say
  about…".
- **Citation neighbourhood** — this paper plus everything it cites and everything citing it.
  Good for "how does this compare to related work".

---

## Model and quantization

**Model** is the AI that writes the answers. It shows what is available on this machine and
marks which one is loaded — only the loaded one can actually answer. Choosing another
requires restarting the generator.

**＋ Download new model…** at the bottom of that list lets you fetch one from Hugging Face.
Type its name (like `Qwen/Qwen3-8B`) and press Check: it tells you the size, and whether it
will fit in this machine's memory, *before* downloading anything. That matters — these are
often 20–60 GB.

**Quantization** is how compressed the model's weights are. Smaller uses less memory and
runs faster, at some cost in quality. It is fixed by whichever model file you have, so there
is usually nothing to decide.

---

## Style

Six presets for reading comfort. Each changes the theme, typeface, size, spacing and column
width together, because those settings only work in combination.

- **Default** — matches your system's light/dark setting.
- **Paper** — warm white page, serif type, justified. Closest to a printed journal.
- **Night** — dark with looser spacing, for low light.
- **Sepia** — warm cream page, like an e-reader. Easiest for long sessions.
- **Compact** — smaller and denser, for skimming.
- **High contrast** — black on white, larger, with a typeface designed so similar letters
  cannot be confused. For low vision.

Changing any individual setting switches the preset to "Custom", so the label never claims
something untrue.

---

## The ⚙ button — advanced settings

### Passage heatmap

When you follow a citation into a paper, other relevant passages get shaded so you can see
at a glance where else the answer is supported. Warmer shading is more relevant.

- **Similar to the answer passage** (default) — highlights the surrounding argument: the
  setup, the caveats, the experiment that qualifies it. Usually what you want after
  following a citation.
- **Similar to my question** — highlights other places the paper discusses your question.
- **Off**.

**Highlight top N passages** controls how many get shaded. More shading gives more context
but a busier page.

### Reading controls

Individual settings behind the Style presets: theme, typeface (eight options), text size,
line height, line width, and justification.

**Line width** is worth knowing about. It is measured in characters per line, not pixels.
Very long lines are tiring because your eye loses its place returning to the left margin;
60–70 characters is the usual comfortable range.

### Temperature

How predictable the answer is. Low (0.0–0.3) gives careful, repeatable answers that stay
close to the sources. High (0.8+) gives more varied phrasing and more risk of drifting from
what the sources actually say.

For looking things up in papers, keep it low. The default of 0.2 is a good place to stay.

---

## Things that are not obvious

- **Your searches are recorded.** Every query and every citation you click is stored locally
  in `data/meta.sqlite` and used to improve the search. Nothing leaves the machine. See
  `docs/finetuning/gemma_embedder_finetuning.md` for exactly what is kept and how to
  inspect or delete it.
- **The back and forward buttons work.** Papers and searches are proper history entries, and
  search results are linkable — the URL contains your query.
- **Citation links are real links.** Middle-click or Ctrl-click opens a paper in a new tab;
  right-click gives you "copy link address".
- **The graph pane is empty until you open a paper.** It shows the citation neighbourhood of
  whatever you are reading, shaded by relevance to your last question.
- **There is no login and no authentication.** If the server is reachable on your network,
  anyone on that network can use it.
