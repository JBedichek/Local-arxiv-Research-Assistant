"""LaTeXML HTML -> anchored sections, blocks and chunks (requirement R2).

Both full-text sources emit the same LaTeXML conventions, verified against
``arxiv.org/html/2401.12345v1`` and ``ar5iv.labs.arxiv.org/html/1706.03762``:

===========================  ==========================  ==============================
structure                    element                     anchor id
===========================  ==========================  ==============================
section                      ``section.ltx_section``     ``S3``
subsection                   ``section.ltx_subsection``  ``S3.SS2``
subsubsection                ``section.ltx_subsubsection`` ``S3.SS2.SSS1``
named paragraph              ``section.ltx_paragraph``   ``S3.SS1.SSS0.Px1``
appendix                     ``section.ltx_appendix``    ``A1``
paragraph (the RAG unit)     ``div.ltx_para``            ``S3.p4``
  its text child             ``p.ltx_p``                 ``S3.p4.1``
numbered equation            ``table.ltx_equation``      ``S2.E1``
unnumbered display equation  ``table.ltx_equation``      ``S1.Ex1``
figure / table float         ``figure``                  ``S4.F2`` / ``S4.T1``
abstract                     ``div.ltx_abstract``        ``abstract1``
bibliography entry           ``li.ltx_bibitem``          ``bib.bib17``
===========================  ==========================  ==============================

We anchor on ``div.ltx_para`` (``S3.p4``) rather than the inner ``p`` — it is the stable
unit, it is what the citation contract in PLAN.md §4 specifies, and it survives the
``ltx_p``-less cases (lists, floats).

**Math must not go through ``text_content()``.** LaTeXML emits both a MathML rendering and
a LaTeX fallback, so naive text extraction concatenates them and yields corruption like
``NN=8, SNR 10dB, 𝑹\\bm{R}`` for what is really ``$N$=8, SNR 10dB, $\\bm{R}$``. Since
equations are exactly what many questions are about, every ``<math>`` is replaced by its
``alttext`` (the original LaTeX) before any text is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import html as LH
from lxml.html import HtmlElement

# Structural containers, most specific first — the class list is checked in this order so
# a subsubsection is not mistaken for a section.
SECTION_KINDS: list[tuple[str, str, int]] = [
    ("ltx_subsubsection", "subsubsection", 3),
    ("ltx_subsection", "subsection", 2),
    ("ltx_paragraph", "paragraph", 4),
    ("ltx_appendix", "appendix", 1),
    ("ltx_section", "section", 1),
]

def cls(token: str) -> str:
    """XPath predicate matching a whole class *token*, not a substring.

    ``contains(@class, 'ltx_authors')`` is a trap: it matches
    ``<article class="ltx_document ltx_authors_1line">``, which is the entire paper, so a
    drop rule aimed at the author block silently deletes the document. The same trap
    catches ``ltx_para`` inside ``ltx_paragraph``. Pad with spaces and match the token.
    """
    return f"contains(concat(' ', normalize-space(@class), ' '), ' {token} ')"


def has_cls(el: HtmlElement, token: str) -> bool:
    """Python-side equivalent of :func:`cls` — token membership, not substring."""
    return token in (el.get("class") or "").split()


# Dropped wholesale: navigation chrome, scripts, and the bibliography (handled separately
# by the citation pipeline, and useless as RAG context).
DROP_XPATH = " | ".join([
    "//script", "//style", "//nav", "//noscript",
    f"//*[{cls('ltx_page_navbar')}]",
    f"//*[{cls('ltx_page_footer')}]",
    f"//*[{cls('ltx_bibliography')}]",
    f"//*[{cls('ltx_authors')}]",
    f"//*[{cls('ltx_dates')}]",
    f"//*[{cls('ltx_role_affiliation')}]",
    "//*[@id='selectedTextModalDescription']",
])

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


@dataclass
class Section:
    anchor: str
    title: str
    kind: str
    depth: int
    ordinal: int


@dataclass
class Block:
    """One atomic anchored unit — a paragraph, an equation, a caption."""
    anchor: str            # e.g. 'S3.p4'
    section_anchor: str    # e.g. 'S3.SS2'
    kind: str              # body|abstract|equation|caption|theorem
    text: str
    ordinal: int


@dataclass
class Chunk:
    """A retrieval unit spanning one or more blocks, with a precise highlight range."""
    ordinal: int
    section_anchor: str
    anchor_start: str
    char_start: int
    anchor_end: str
    char_end: int
    kind: str
    text: str
    block_anchors: list[str] = field(default_factory=list)

    def fragment(self) -> str:
        """URL fragment per PLAN.md §4: ``#S3.p4:120-480``."""
        if self.anchor_start == self.anchor_end:
            return f"#{self.anchor_start}:{self.char_start}-{self.char_end}"
        return f"#{self.anchor_start}:{self.char_start}-{self.anchor_end}:{self.char_end}"


@dataclass
class ParsedPaper:
    title: str
    sections: list[Section]
    blocks: list[Block]
    chunks: list[Chunk]
    bib_arxiv_ids: list[str]


# ── math handling ──────────────────────────────────────────────────────────────────

_ARXIV_IN_HREF = re.compile(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})")
_ARXIV_IN_TEXT = re.compile(r"arXiv[:\s]\s*([0-9]{4}\.[0-9]{4,5})", re.IGNORECASE)


def inline_math_as_latex(root: HtmlElement) -> None:
    """Replace every ``<math>`` with its ``alttext``, in document order.

    Without this, ``text_content()`` interleaves the MathML rendering with LaTeXML's
    fallback text and produces corrupted output (see module docstring).
    """
    for node in root.xpath("//*[local-name()='math']"):
        alt = node.get("alttext")
        display = node.get("display") == "block"
        if alt:
            latex = _clean(alt)
            repl = f" $${latex}$$ " if display else f" ${latex}$ "
        else:
            repl = " "
        tail = node.tail or ""
        parent = node.getparent()
        if parent is None:
            continue
        prev = node.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + repl + tail
        else:
            parent.text = (parent.text or "") + repl + tail
        parent.remove(node)


def _text_of(el: HtmlElement) -> str:
    return _clean(el.text_content())


def remove_keeping_tail(el: HtmlElement) -> None:
    """Remove an element but preserve its tail text.

    lxml's ``parent.remove(child)`` discards ``child.tail`` as well. Stripping the
    ``<span class="ltx_tag">1 </span>`` from ``<h2>1 Introduction</h2>`` therefore deletes
    "Introduction" too — which is exactly how every ar5iv section title came out empty.
    """
    parent = el.getparent()
    if parent is None:
        return
    tail = el.tail or ""
    if tail:
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(el)


# ── structure walk ─────────────────────────────────────────────────────────────────

def _section_kind(el: HtmlElement) -> tuple[str, int] | None:
    for token, kind, depth in SECTION_KINDS:
        if has_cls(el, token):
            return kind, depth
    return None


def _section_title(el: HtmlElement) -> str:
    for child in el.iterchildren():
        if has_cls(child, "ltx_title") or any(
            c.startswith("ltx_title_") for c in (child.get("class") or "").split()
        ):
            # Strip the leading number tag ("3.2") so titles read cleanly. Must preserve
            # tails, or the title text following the tag disappears with it.
            for tag in child.xpath(f".//*[{cls('ltx_tag')}]"):
                remove_keeping_tail(tag)
            return _text_of(child)
    return ""


def _nearest_section_anchor(el: HtmlElement) -> str:
    node = el.getparent()
    while node is not None:
        if node.tag == "section" and node.get("id") and _section_kind(node):
            return node.get("id")
        if has_cls(node, "ltx_abstract"):
            return node.get("id") or "abstract"
        node = node.getparent()
    return ""


def extract_blocks(root: HtmlElement) -> tuple[list[Section], list[Block]]:
    sections: list[Section] = []
    for i, el in enumerate(root.xpath("//section[@id]")):
        sk = _section_kind(el)
        if sk:
            sections.append(Section(el.get("id"), _section_title(el), sk[0], sk[1], i))

    blocks: list[Block] = []
    seen: set[str] = set()

    # Document order matters — the chunker relies on it to group neighbouring paragraphs.
    query = " | ".join([
        f"//div[{cls('ltx_para')}][@id]",
        f"//div[{cls('ltx_abstract')}][@id]",
        f"//table[{cls('ltx_equation')}][@id]",
        f"//*[{cls('ltx_theorem')}][@id]",
        "//figure[@id]",
    ])
    for el in root.xpath(query):
        anchor = el.get("id")
        if not anchor or anchor in seen:
            continue

        if has_cls(el, "ltx_abstract"):
            kind = "abstract"
            text = _text_of(el)
            text = re.sub(r"^Abstract\s*", "", text)
        elif has_cls(el, "ltx_equation"):
            kind = "equation"
            text = _text_of(el)
        elif has_cls(el, "ltx_theorem"):
            kind = "theorem"
            text = _text_of(el)
        elif el.tag == "figure":
            # Anchor the caption to the float; figcaption itself carries no id.
            caps = el.xpath(".//figcaption | .//caption")
            if not caps:
                continue
            kind = "caption"
            text = " ".join(_text_of(c) for c in caps)
        else:
            kind = "body"
            text = _text_of(el)

        if len(text) < 2:
            continue
        seen.add(anchor)
        blocks.append(Block(
            anchor=anchor,
            section_anchor=_nearest_section_anchor(el) or anchor.split(".")[0],
            kind=kind, text=text, ordinal=len(blocks),
        ))
    return sections, blocks


def extract_bib_arxiv_ids(root: HtmlElement) -> list[str]:
    """arXiv ids cited in the bibliography — a free fallback for the citation graph.

    Two sources, because they disagree: arXiv's own HTML tends to hyperlink references,
    while ar5iv renders them as plain text (``arXiv:1607.06450``). Checking only hrefs
    finds nothing on ar5iv; checking only text misses linked-but-unlabelled entries.
    """
    ids: list[str] = []
    for entry in root.xpath(f"//*[{cls('ltx_bibitem')}] | //*[{cls('ltx_bibliography')}]"):
        for a in entry.xpath(".//a[@href]"):
            m = _ARXIV_IN_HREF.search(a.get("href") or "")
            if m:
                ids.append(m.group(1))
        ids.extend(_ARXIV_IN_TEXT.findall(entry.text_content()))
    return sorted(set(ids))


# ── chunking ───────────────────────────────────────────────────────────────────────

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\\$])")


def _hard_split(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Last-resort split on whitespace, guaranteeing no span exceeds ``limit``.

    Sentence splitting fails outright on math-heavy prose — an appendix paragraph made of
    display equations can run 3.6k chars without a single ``. `` boundary, and would
    otherwise sail through as one chunk far bigger than the embedder's window.
    """
    spans: list[tuple[int, int]] = []
    pos = start
    while end - pos > limit:
        cut = text.rfind(" ", pos + limit // 2, pos + limit)
        if cut <= pos:
            cut = pos + limit          # no whitespace at all; split mid-token
        spans.append((pos, cut))
        pos = cut
    if pos < end:
        spans.append((pos, end))
    return spans


def _split_long(text: str, target: int, overlap: int) -> list[tuple[int, int]]:
    """Split an oversized block at sentence boundaries; return (start, end) char spans."""
    hard_limit = int(target * 1.5)
    sentences = _SENT.split(text)
    spans: list[tuple[int, int]] = []
    pos = 0
    start = 0
    for s in sentences:
        idx = text.find(s, pos)
        if idx < 0:
            idx = pos
        end = idx + len(s)
        if end - start >= target:
            spans.append((start, end))
            start = max(start, end - overlap)
        pos = end
    if start < len(text) and (not spans or spans[-1][1] < len(text)):
        spans.append((start, len(text)))
    if not spans:
        spans = [(0, len(text))]

    # Enforce the ceiling regardless of how the sentence pass went.
    capped: list[tuple[int, int]] = []
    for s, e in spans:
        capped.extend(_hard_split(text, s, e, hard_limit) if e - s > hard_limit else [(s, e)])
    return capped


def chunk_blocks(
    blocks: list[Block],
    *,
    target_chars: int = 1000,
    overlap_frac: float = 0.15,
    never_cross_sections: bool = True,
) -> list[Chunk]:
    """Merge neighbouring blocks up to ``target_chars``; split oversized ones.

    A chunk records where it starts and ends precisely enough to paint a highlight:
    ``anchor_start``/``char_start`` through ``anchor_end``/``char_end``.
    """
    overlap = int(target_chars * overlap_frac)
    chunks: list[Chunk] = []
    pending: list[Block] = []

    def flush() -> None:
        if not pending:
            return
        text = "\n\n".join(b.text for b in pending)
        chunks.append(Chunk(
            ordinal=len(chunks),
            section_anchor=pending[0].section_anchor,
            anchor_start=pending[0].anchor, char_start=0,
            anchor_end=pending[-1].anchor, char_end=len(pending[-1].text),
            kind=pending[0].kind if len({b.kind for b in pending}) == 1 else "body",
            text=text,
            block_anchors=[b.anchor for b in pending],
        ))
        pending.clear()

    for block in blocks:
        if never_cross_sections and pending and block.section_anchor != pending[0].section_anchor:
            flush()

        if len(block.text) > target_chars * 1.5:
            flush()
            for start, end in _split_long(block.text, target_chars, overlap):
                chunks.append(Chunk(
                    ordinal=len(chunks), section_anchor=block.section_anchor,
                    anchor_start=block.anchor, char_start=start,
                    anchor_end=block.anchor, char_end=end,
                    kind=block.kind, text=block.text[start:end],
                    block_anchors=[block.anchor],
                ))
            continue

        current = sum(len(b.text) for b in pending)
        if pending and current + len(block.text) > target_chars:
            flush()
        pending.append(block)

    flush()
    return chunks


# ── entry point ────────────────────────────────────────────────────────────────────

def parse_html(
    content: bytes | str,
    *,
    target_chars: int = 1000,
    overlap_frac: float = 0.15,
    never_cross_sections: bool = True,
) -> ParsedPaper:
    root = LH.fromstring(content)

    bib_ids = extract_bib_arxiv_ids(root)   # before the bibliography is dropped

    title_nodes = root.xpath("//h1[contains(@class,'ltx_title')]")
    title = _text_of(title_nodes[0]) if title_nodes else ""

    for el in root.xpath(DROP_XPATH):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

    inline_math_as_latex(root)
    sections, blocks = extract_blocks(root)
    chunks = chunk_blocks(
        blocks, target_chars=target_chars, overlap_frac=overlap_frac,
        never_cross_sections=never_cross_sections,
    )
    return ParsedPaper(title, sections, blocks, chunks, bib_ids)
