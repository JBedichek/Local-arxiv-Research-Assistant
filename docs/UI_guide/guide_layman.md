# Using the reader

What every control does, in terms of what you will notice. No jargon.

---

## The four panes

**Far left** is your library, then the **citation graph**, then the **paper**, and on the
**right** is where you ask questions.

Drag the thin dividers to resize any of them. Double-click a divider to collapse that pane
entirely and give the space to the paper. Or use the keyboard:

| key | does |
|---|---|
| `;` | hide/show the library pane |
| `[` | hide/show the graph pane |
| `]` | hide/show the question pane |
| `\` | hide/show all three — the paper takes the whole window |
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

**Maths reads as maths.** Answers, excerpts and quoted passages show `√`, `ε`, fractions,
subscripts and Greek letters rather than the raw `$O(\epsilon^{-3})$` you would see in a
plain text box. Occasionally something unusual will still appear as its source text; that is
deliberate, and it is never worse than what a plain box would have shown you.

### Asking a follow-up

Just ask. *"Why is that?"*, *"and the second one?"*, *"say more"* — you do not have to
restate the subject.

Behind the scenes the reader rewrites the question into something that can actually be
searched for, and it tells you: among the progress lines above the answer you will see
*searched for: "…"*. Worth glancing at. If the rewrite went somewhere you did not mean, that
explains an answer that feels off-topic, and rephrasing fixes it. Silently searching for
something other than what you typed would be the kind of helpfulness that stops being
helpful the moment you notice it.

Conversations are kept **per paper**. Open a different paper and you start a fresh one, so a
question about optimisers cannot bleed into an answer about tokenisation.

### The line under the answer

A short line reading *"… ms to first token · … tok/s · … out · … in"*. Just how fast the
machine was: how long before the first word appeared, how quickly the rest came, and how
much text went in and came out. The tok/s figure is timed from the first word onwards, so it
describes the writing speed rather than the wait.

### "Context" — what the question actually cost

Click it to expand. The model can only hold so much text at once, and this shows exactly
what was in there: the instructions, the examples, the earlier conversation, the excerpts
from papers, and your question — each with its share, plus the space left over and the room
reserved for the answer.

You mostly do not need it. It matters when a long conversation starts crowding out the
paper excerpts, and the panel makes that visible instead of leaving you to guess. When it
does, there is a **Compress conversation** button right there.

**Compress conversation** replaces the older exchanges with a short set of notes, keeping
the last two word for word. You keep the thread going and stop paying for every earlier
turn. Afterwards it tells you exactly what it did — how many exchanges it folded away, and
whether that came out smaller or merely more complete, because on a short conversation it
does not save anything and says so. You can open **"summary the model will now see"** to
read the notes, and **undo** puts the full history back.

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

## Deep Automated Research

The button at the top of the window, next to the model picker. It is for a different kind
of question from the ones above: not *"what does this paper say about X"* but *"what does
the literature say about X"* — the sort of question you would answer by reading a dozen
papers.

**It takes minutes, not seconds.** It searches, reads what it found, notices what is
missing, searches again for that specifically, and keeps going until new searches stop
turning anything up. Fifteen or twenty rounds is normal for a hard question. There is a
**Stop** button, and stopping is not throwing the work away — it writes up whatever it has
found so far.

It takes over the whole window, because you are not doing anything else while it runs.

- **On the left**, the work as it happens: each round, what it decided to search for next
  and why, and every fact it extracted along the way. Each fact is named by the model, and
  clicking the name opens the exact passage it came from.
- **On the right**, two answers. **TLDR** is the short one — a few sentences. **Thorough**
  is the full write-up, organised, with everything cited.

The short answer is written *from* the long one rather than separately, so the two cannot
contradict each other.

Do not expect it to always pick a winner. When the papers measured different things under
different conditions, it will say so and show you the table instead of inventing a ranking.
That is the correct answer to a lot of "which is best?" questions.

**past runs…** at the top reopens anything you have run before, exactly as it was.

---

## Your library

The pane on the far left. Everything you open and every question you ask ends up here, and
it never leaves the machine.

Two ways to look at it, via the **graph / list** button:

- **List** — folders and entries. Rename, drag into folders, delete.
- **Graph** — your conversations as a map. Each conversation gets a short name the reader
  chose for it and a few topic tags, oldest at the top, with arrows joining one to another
  where one really did lead to the next. Not everything is connected, and that is fine — a
  false arrow is worse than no arrow.

The **↻** button rebuilds the graph after you have asked new questions.

**Clicking a question brings back the whole conversation**, not just that one exchange. The
question you clicked is marked, and the box at the bottom invites you to carry on from
there rather than starting over.

---

## ★ Interesting — and the "For you" list

Highlight a passage and, alongside "Ask about this", there is **★ Interesting**. It saves
that passage as an example of the kind of thing you want to read.

Once you have a few, the **For you** list under the citation graph shows passages from
elsewhere in the corpus that resemble them. **corpus** widens it to the whole collection
rather than the paper you have open.

It is genuinely a *taste* profile, not a summary of your interests: marking three unrelated
things does not average them into one blurry middle, it looks for material close to any one
of them.

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

### Answer instructions

The standing instructions the model is given before every question — how to cite, when to
admit it cannot answer, how to handle numbers. **Save** applies your version; **Restore
default** puts the shipped one back, so nothing you do here is permanent damage.

Worth editing if you consistently want, say, shorter answers or a particular style. Worth
leaving alone otherwise: the default is doing quite a lot of work to keep answers honest.

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
- **Your conversation is per paper.** Switching papers starts a new one; switching back
  picks the old one up again.
- **A deep research run keeps going if you close the window**, and finishes writing up what
  it found. It will be waiting under "past runs…".
- **Whether there is a login depends on how it was started.** Run on this machine only, it
  asks for nothing. Made reachable from other machines, it will refuse to start without an
  access token, and then asks for it once per browser. Someone can deliberately turn that
  off, so if you did not set it up yourself, ask.
