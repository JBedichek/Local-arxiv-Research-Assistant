"""Background reading on what a lesson leans on without teaching.

A lesson can use a term or result in passing -- "Adam's second-moment estimate", "cosine
annealing" -- that was never its own concept in the course. Once a lesson is written, one call
reads it and names the specific outside ideas it assumes; the learner says whether they already
know each one, and a "no" or "partial" answer gets a short document of its own: the same
retrieval and claim-verification discipline as a lesson section, just without a quiz or a place
in the concept graph. It opens beside the lesson, not instead of it, so a reader can go find out
what they are missing without losing their place in the thread they were following."""

from __future__ import annotations

from lara.learn import claims as CL
from lara.learn import lesson as LE
from lara.learn import visuals as VS
from lara.learn.llm import Llm

MAX_TOPICS = 6

TOPICS_SYSTEM = """A learner is about to read this lesson. Find the specific outside ideas it \
leans on without teaching -- terms, methods or results it assumes the reader already knows,
stated or used rather than explained.

Reply with JSON only: [{"title": "...", "note": "..."}]

- title: the term or idea itself, short and specific (e.g. "Adam's second-moment estimate", \
not "optimizers").
- note: one sentence on what about it the lesson leans on -- specific enough to search a paper \
corpus for.
- Only real prerequisites the lesson itself does not explain -- not things it already defines, \
and not the lesson's own subject.
- At most 6, most load-bearing first. If the lesson is self-contained, reply []."""

TOPIC_SYSTEM = """You write a short background note on ONE topic, for a learner who needs it to \
follow a lesson on something else -- using ONLY the numbered claims provided.

Format: one sentence per line, each ending -- before its full stop -- with the keys of the \
claims it rests on in brackets, like [c1] or [c2, c5]. A sentence with no key is not allowed.

- Answer just enough to follow the lesson that sent the reader here -- this is a background \
note, not a full lesson on the topic. Do not add any fact that is not in the claims.
- Convey certainty as marked: one paper, a hypothesis, replaced by later work.
- If READER ALREADY KNOWS is given, do not repeat any of it -- write only what it leaves out."""


def _lesson_text(lesson: dict | None) -> str:
    if not lesson or lesson.get("insufficient"):
        return ""
    return "\n".join(s["text"] for sec in lesson.get("sections", []) for s in sec["sentences"])


async def extract_topics(llm: Llm, concept: dict, lesson: dict | None) -> list[dict]:
    """Topics this lesson's prose leans on, or [] when there is no lesson yet, it is
    insufficient, or the model finds nothing worth a background note (a bad reply degrades to
    [] the same way -- no topics is a safe default, an invented one is not)."""
    text = _lesson_text(lesson)
    if not text.strip():
        return []
    data = await llm.ask_json(TOPICS_SYSTEM, f"CONCEPT: {concept['title']}\n\nLESSON:\n{text}",
                              default=500, cap=1_500, stage="learn_topics")
    out = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if title:
            out.append({"id": f"t{len(out) + 1}", "title": title, "note": str(item.get("note", "")).strip()})
    return out[:MAX_TOPICS]


async def build_doc(llm: Llm, corpus, concept: dict, topic: dict, *, tailor: str = "",
                    embed=None) -> dict:
    """A short grounded document on one topic: the same retrieval, extraction and verification a
    lesson section gets, just for one term rather than the whole concept, and with no quiz.
    `tailor` is what the learner said they already know about it (a "partial" familiarity
    answer), so the note covers only what that leaves out."""
    focus = f"{topic['title']}. {topic.get('note', '')}".strip()
    passages = await CL.gather_passages(corpus, concept, focus=focus)
    found, *_ = await CL.extract(llm, concept, passages, embed=embed, focus=focus)
    claims = [c.to_dict() for c in found]
    if len(claims) < LE.MIN_CLAIMS:
        return {"insufficient": True, "sections": [], "claims": [], "chart": None,
                "message": "The paper corpus holds too little on this to write a background note."}
    by_key = {c["key"]: c for c in claims}
    listing = "\n".join(LE.line(c) for c in claims)
    prompt = (f"TOPIC: {topic['title']} -- {topic.get('note', '')}\n"
              + (f"READER ALREADY KNOWS: {tailor}\n" if tailor else "")
              + f"\nCLAIMS:\n{listing}")
    text = await llm.ask(TOPIC_SYSTEM, prompt, default=700, cap=0, stage="learn_topic_doc")
    sections = LE.parse(text, set(by_key))
    out, stats = await LE.verify(llm, sections, by_key)
    if not out:
        return {"insufficient": True, "sections": [], "claims": [], "chart": None,
                "message": "Nothing here could be verified against its sources."}
    cited = {k for sec in out for s in sec["sentences"] for k in s["claims"]}
    kept = [c for c in claims if c["key"] in cited]
    chart = await VS.chart(llm, {"title": topic["title"]}, kept)
    return {"insufficient": False, "sections": out, "stats": stats, "claims": kept, "chart": chart}
