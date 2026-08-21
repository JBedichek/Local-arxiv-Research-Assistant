"""The library as a directed graph of conversations rather than a list.

A flat list answers "what did I ask?" and nothing else. What a reader actually wants back
is *why* they asked it — which question led to which, what turned out to be the same
investigation under two names, and where a thread from last week already answered the one
they are about to start.

So each thread is summarised into a topic, and the topics are arranged into a graph:

    node   one conversation (a paper's thread, or the corpus thread), with a label the
           model chose, a one-line summary, and the questions it contains
    edge   a logical connection, always pointing FORWARD IN TIME

The time direction is not decoration. It makes the graph a DAG by construction — no cycle
can form when every edge runs from an older conversation to a newer one — which means it
can be laid out in columns by depth and read left to right as the order the reader's
thinking actually went. An undirected "these are related" graph is a hairball that says
much less.

Edges are *proposed* by the model and then filtered here: a claimed edge whose endpoints
do not exist, or which points backward in time, is dropped rather than trusted. The model
is good at noticing relationships and unreliable about identifiers.

The whole thing is cached, because it costs two model calls over the entire library and
the answer only changes when a new question is asked.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

#: Relationship kinds the model may use. A closed set keeps the legend readable and stops
#: the graph acquiring fifty synonyms for "related to".
RELATIONS = ("follows-up", "applies", "compares", "contradicts", "background-for",
             "same-topic", "diverges")

LABEL_SYSTEM = """You are labelling saved conversations in a research library.

For each numbered conversation, give a short topic label and a one-line summary.

What this causes: the label becomes a node in a graph the reader navigates instead of a
list. It is the only text they see before clicking, so it must distinguish this
conversation from the others — "Muon optimizer" is useless if four conversations are about
Muon; "Muon sample complexity bounds" is not.

For each conversation give:
  "label"   3-6 words naming the specific subject
  "summary" one sentence on what was actually established or asked
  "topics"  2-4 lowercase keyword tags

Reply with a JSON array only:
[{"n": 1, "label": "...", "summary": "...", "topics": ["...", "..."]}]"""


EDGE_SYSTEM = """You are connecting saved conversations into a directed graph.

Conversations are listed oldest first with their dates. Propose edges between them.

Every edge points FORWARD IN TIME: from an earlier conversation to a later one. An edge
means the later one genuinely relates to the earlier one — not that they share a word.

What this causes: edges are drawn as arrows the reader follows to retrace their own
thinking. A wrong edge invents a connection they never made; a missing one hides that they
have already investigated something.

Use exactly these relations:
  follows-up      the later one continues the earlier investigation
  applies         the later one uses a method or result from the earlier
  compares        the later one weighs the earlier subject against something else
  contradicts     the later one found something that conflicts with the earlier
  background-for  the earlier one is prerequisite understanding for the later
  same-topic      the same subject revisited, without a clear dependency
  diverges        started from the earlier one but went somewhere unrelated

Be sparing. Only connect conversations with a real relationship; an unconnected node is a
perfectly good outcome and far better than a false arrow. Do not connect everything to
everything.

Reply with a JSON array only:
[{"from": 1, "to": 3, "relation": "follows-up", "why": "one short clause"}]"""


@dataclass
class Node:
    id: str                       # thread id: an arxiv id, or "corpus"
    label: str
    summary: str
    topics: list[str] = field(default_factory=list)
    paper_title: str = ""
    n_questions: int = 0
    first_utc: str = ""
    last_utc: str = ""
    entry_ids: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    depth: int = 0


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    why: str = ""


def threads(root) -> list[dict]:
    """Group question entries into conversations, oldest first."""
    from lara.serve import memory as MEM
    from lara.serve import thread as TH

    data = MEM.load(root)
    groups: dict[str, list[dict]] = {}
    for e in data.get("entries", []):
        if e.get("kind") != "question" or not (e.get("question") or "").strip():
            continue
        groups.setdefault(TH.thread_id(e.get("arxiv_id")), []).append(e)

    out = []
    for tid, items in groups.items():
        items.sort(key=lambda x: x.get("created_utc") or "")
        out.append({
            "id": tid,
            "paper_title": next((i.get("title") for i in items if i.get("title")), ""),
            "entries": items,
            "first": items[0].get("created_utc") or "",
            "last": items[-1].get("created_utc") or "",
        })
    out.sort(key=lambda t: t["first"])
    return out


def _describe(t: dict, limit: int = 900) -> str:
    lines = []
    for e in t["entries"][:6]:
        a = (e.get("answer") or "").strip().replace("\n", " ")
        lines.append(f"  Q: {e.get('question','').strip()[:200]}")
        if a:
            lines.append(f"  A: {a[:limit // max(1, len(t['entries']))]}")
    head = f"{t['paper_title'] or t['id']} ({len(t['entries'])} questions, {t['first'][:10]})"
    return head + "\n" + "\n".join(lines)


async def _label(cfg, ts: list[dict], model, stream_answer) -> dict[int, dict]:
    listing = "\n\n".join(f"[{i}] {_describe(t)}" for i, t in enumerate(ts, 1))
    from lara.serve.generate import complete_json

    rows = await complete_json(
        cfg, f"Conversations:\n{listing}\n\nReply with the JSON array only.",
        system=LABEL_SYSTEM, shape="array", model=model, max_tokens=1600, default=[])
    out: dict[int, dict] = {}
    for r in rows if isinstance(rows, list) else []:
        try:
            n = int(r.get("n", 0))
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(ts):
            out[n] = {
                "label": str(r.get("label") or "")[:60],
                "summary": str(r.get("summary") or "")[:240],
                "topics": [str(x)[:24] for x in (r.get("topics") or [])][:4],
            }
    return out


async def _edges(cfg, nodes: list[Node], model, stream_answer) -> list[Edge]:
    if len(nodes) < 2:
        return []
    listing = "\n".join(
        f"[{i}] {n.first_utc[:10]} · {n.label} — {n.summary}"
        for i, n in enumerate(nodes, 1))
    from lara.serve.generate import complete_json

    rows = await complete_json(
        cfg, f"Conversations, oldest first:\n{listing}\n\nReply with the JSON array only.",
        system=EDGE_SYSTEM, shape="array", model=model, max_tokens=1200, default=[])

    seen: set[tuple[str, str]] = set()
    edges: list[Edge] = []
    for r in rows if isinstance(rows, list) else []:
        try:
            a, b = int(r.get("from", 0)), int(r.get("to", 0))
        except (TypeError, ValueError):
            continue
        if not (1 <= a <= len(nodes) and 1 <= b <= len(nodes)) or a == b:
            continue
        src, dst = nodes[a - 1], nodes[b - 1]
        # Enforce the time direction rather than trusting it. The model is reliable about
        # noticing a relationship and unreliable about which way it runs, and one backward
        # edge turns a readable DAG into a cycle that no layout can order.
        if src.first_utc > dst.first_utc:
            src, dst = dst, src
        key = (src.id, dst.id)
        if key in seen:
            continue
        seen.add(key)
        rel = str(r.get("relation") or "same-topic")
        edges.append(Edge(src.id, dst.id, rel if rel in RELATIONS else "same-topic",
                          str(r.get("why") or "")[:160]))
    return edges


def _assign_depth(nodes: list[Node], edges: list[Edge]) -> None:
    """Longest-path depth, which is what makes the columns read as progression.

    Safe because the edges are already forced forward in time, so the graph is acyclic;
    the loop bound is belt and braces against a malformed edge set rather than a real
    possibility.
    """
    index = {n.id: n for n in nodes}
    incoming: dict[str, list[str]] = {n.id: [] for n in nodes}
    for e in edges:
        if e.target in incoming:
            incoming[e.target].append(e.source)
    for _ in range(len(nodes)):
        changed = False
        for n in nodes:
            d = max((index[s].depth + 1 for s in incoming[n.id] if s in index), default=0)
            if d != n.depth:
                n.depth, changed = d, True
        if not changed:
            break


async def build(cfg, root, model, stream_answer) -> dict:
    """Summarise every thread and connect them. Returns a serialisable graph."""
    ts = threads(root)
    if not ts:
        return {"nodes": [], "edges": [], "built_utc": "", "n_threads": 0}

    labels = await _label(cfg, ts, model, stream_answer)
    nodes: list[Node] = []
    for i, t in enumerate(ts, 1):
        lab = labels.get(i) or {}
        qs = [e.get("question", "") for e in t["entries"]]
        nodes.append(Node(
            id=t["id"],
            # Falling back to the first question keeps a node readable when labelling
            # failed, rather than showing an unnamed dot.
            label=lab.get("label") or (qs[0][:48] if qs else t["id"]),
            summary=lab.get("summary") or "",
            topics=lab.get("topics") or [],
            paper_title=t["paper_title"], n_questions=len(t["entries"]),
            first_utc=t["first"], last_utc=t["last"],
            entry_ids=[e["id"] for e in t["entries"]], questions=qs,
        ))

    edges = await _edges(cfg, nodes, model, stream_answer)
    _assign_depth(nodes, edges)
    return {
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
        "relations": list(RELATIONS),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_threads": len(nodes),
    }


# ── cache ─────────────────────────────────────────────────────────────────────────


def _fingerprint(ids: list[str]) -> str:
    """Changes exactly when the library does, so a stale graph is never served.

    Takes the ids rather than fetching them, which makes it pure — and means the caller
    reads the library once instead of once here and once again around the call.
    """
    import hashlib

    # hashlib, not hash(): PYTHONHASHSEED is randomised per process, so a built-in hash
    # would produce a different fingerprint in every worker and after every restart —
    # a cache that can never hit, silently rebuilding the whole graph on each request.
    digest = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]
    return f"{len(ids)}:{digest}"


def cached(root) -> dict | None:
    from lara.serve import memory as MEM

    ids, entry = MEM.graph_cache(root)
    if entry and entry.get("fingerprint") == _fingerprint(ids):
        return entry.get("graph")
    return None


def store(root, graph: dict) -> None:
    from lara.serve import memory as MEM

    ids, _ = MEM.graph_cache(root)
    MEM.store_graph(root, _fingerprint(ids), graph)
