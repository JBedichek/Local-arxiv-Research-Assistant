"""Download a candidate document and turn it into text, a hash and a licence.

Three formats cover almost everything a study corpus pulls in: HTML pages, PDFs, and
plain text. PDFs matter most and are the reason this module exists separately from
``lara.ingest.parse`` — that one understands LaTeXML output from arXiv, which is a
different problem from "a textbook someone put on a university web server".

**Text is extracted before the licence is judged.** The first end-to-end run got this
wrong: it passed HTML to the licence detector and nothing at all for PDFs, so a genuinely
free calculus textbook came back ``unknown`` purely because its licence statement was on
page 2 of a PDF rather than in a meta tag.

**Size is checked while streaming, not after.** A corpus build is allowed to be large —
that is the point — but "large" should be a number the reader chose, and discovering a
1 GB download by finding it on disk afterwards is not a choice.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from lara.corpus import licence as LIC
from lara.corpus.search import UA

#: Refuse anything that is obviously not a document before spending bandwidth on it.
BINARY_HINTS = ("image/", "video/", "audio/", "application/zip", "application/x-tar",
                "application/octet-stream")

#: Read in chunks so an over-large file is abandoned mid-stream rather than after it has
#: already been paid for.
CHUNK = 1 << 16

#: Statuses worth trying again. 429 and 5xx are the server saying "not now" rather than
#: "no"; a 404 is a fact and retrying it just wastes the reader's time.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
RETRIES = 3
RETRY_BACKOFF_SEC = 2.0

#: Below this, an HTML page yielded so little text that it is almost certainly rendered by
#: JavaScript rather than actually empty. Measured: the OpenStax book page returns 61
#: characters and links to no PDF at all. Such a page is reported as `thin` rather than
#: quietly accepted (it would score as noise) or quietly dropped (the reader would never
#: learn that a good source was skipped for a fixable reason).
THIN_TEXT_CHARS = 800

#: A browser-shaped header set. Some university and publisher hosts serve a redirect or an
#: error to clients that send nothing but a User-Agent.
BROWSE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class Document:
    url: str
    title: str = ""
    text: str = ""
    content_type: str = ""
    bytes_downloaded: int = 0
    sha256: str = ""
    licence: LIC.Licence | None = None
    error: str = ""
    thin: bool = False                  # HTML that rendered to almost no text
    alternates: list[str] = field(default_factory=list)   # documents the page links to
    attempts: int = 1
    raw: bytes | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return not self.error and len(self.text.strip()) >= 200

    @property
    def chars(self) -> int:
        return len(self.text)


def _html_to_text(body: str) -> tuple[str, str]:
    """Readable text and a title, with the furniture removed.

    Not a readability implementation — just enough to stop navigation menus, scripts and
    cookie banners from dominating a page whose actual content is three paragraphs. The
    relevance judgement downstream reads this text, and a page that scores well on its own
    navigation is worse than useless.
    """
    from lxml import html as LH

    try:
        doc = LH.fromstring(body)
    except Exception:                                 # noqa: BLE001
        return "", ""
    for bad in doc.xpath("//script|//style|//nav|//footer|//header|//noscript|//form"):
        bad.getparent().remove(bad) if bad.getparent() is not None else None
    title = (doc.xpath("string(//title)") or "").strip()
    main = doc.xpath("//main|//article") or [doc]
    text = main[0].text_content()
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text)).strip(), title


def _pdf_to_text(raw: bytes, max_pages: int = 3000) -> tuple[str, str]:
    """Text and title from a PDF, via PyMuPDF.

    ``max_pages`` is a guard against the occasional scanned monster, not a quality
    setting: a 3,000-page document is either a bound archive or a mistake, and either way
    the reader should be told rather than made to wait.
    """
    try:
        import fitz                                   # PyMuPDF
    except ImportError:
        return "", ""
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            title = (doc.metadata or {}).get("title") or ""
            parts = [doc[i].get_text() for i in range(min(doc.page_count, max_pages))]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(parts)).strip(), title.strip()
    except Exception:                                 # noqa: BLE001
        return "", ""


def _document_links(body: str, base: str) -> list[str]:
    """Links on a page that look like documents rather than navigation.

    Only useful when a page is thin: many university course pages are a thin index that
    links the PDF everyone actually wants. It does not rescue a true single-page app —
    the OpenStax book page links no PDF at all — but it costs one pass over the HTML and
    turns a whole class of dead ends into live ones.
    """
    from urllib.parse import urljoin
    from lxml import html as LH

    try:
        doc = LH.fromstring(body)
    except Exception:                                 # noqa: BLE001
        return []
    out, seen = [], set()
    for href in doc.xpath("//a/@href") + doc.xpath("//link/@href"):
        if not isinstance(href, str):
            continue
        full = urljoin(base, href)
        if full.lower().split("?")[0].endswith((".pdf", ".epub", ".txt", ".md")):
            if full not in seen:
                seen.add(full)
                out.append(full)
    return out[:20]


def fetch(url: str, *, max_bytes: int = 64 << 20, timeout: float = 40.0,
          keep_raw: bool = True, retries: int = RETRIES) -> Document:
    """Download one URL and extract what a corpus needs from it.

    Never raises. A candidate that 404s, times out or turns out to be a video is a
    candidate rejected with a reason, not a build that stops. Transient refusals — 429,
    5xx, a dropped connection — are retried, because a build fetching fifty documents will
    meet one of those, and losing a textbook to a momentary 503 is a bad outcome that
    looks exactly like a bad source.
    """
    doc = Document(url=url)
    raw = None
    for attempt in range(1, max(1, retries) + 1):
        doc.attempts = attempt
        try:
            with httpx.stream("GET", url, headers={"User-Agent": UA, **BROWSE_HEADERS},
                              timeout=timeout, follow_redirects=True) as resp:
                if resp.status_code in RETRY_STATUS and attempt < retries:
                    time.sleep(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
                    continue
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                doc.content_type = ctype
                if any(ctype.startswith(b) for b in BINARY_HINTS):
                    doc.error = f"not a document ({ctype})"
                    return doc

                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    doc.error = (f"too large ({int(declared)/1e6:.0f} MB > "
                                 f"{max_bytes/1e6:.0f} MB)")
                    return doc

                buf = bytearray()
                for part in resp.iter_bytes(CHUNK):
                    buf += part
                    if len(buf) > max_bytes:
                        doc.error = f"exceeded {max_bytes/1e6:.0f} MB while downloading"
                        return doc
                raw = bytes(buf)
                doc.error = ""
                break
        except httpx.HTTPStatusError as exc:
            doc.error = f"HTTP {exc.response.status_code}"
            return doc                                # a 404 is a fact, not a hiccup
        except Exception as exc:                      # noqa: BLE001
            doc.error = f"{type(exc).__name__}: {str(exc)[:120]}"
            if attempt < retries:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
                continue
            return doc
    if raw is None:
        doc.error = doc.error or f"gave up after {doc.attempts} attempts"
        return doc

    doc.bytes_downloaded = len(raw)
    doc.sha256 = hashlib.sha256(raw).hexdigest()
    if keep_raw:
        doc.raw = raw

    body_html = ""
    if "pdf" in doc.content_type or url.lower().endswith(".pdf") or raw[:5] == b"%PDF-":
        doc.text, doc.title = _pdf_to_text(raw)
        doc.content_type = doc.content_type or "application/pdf"
    elif "html" in doc.content_type or raw[:200].lstrip()[:1] == b"<":
        body_html = raw.decode("utf-8", "replace")
        doc.text, doc.title = _html_to_text(body_html)
    else:
        doc.text = raw.decode("utf-8", "replace")

    if body_html and len(doc.text.strip()) < THIN_TEXT_CHARS:
        # Say what happened rather than reporting an empty document. A reader told "this
        # page needs JavaScript, and here is what it links to" can act; one told "no
        # extractable text" concludes the source was bad.
        doc.thin = True
        doc.alternates = _document_links(body_html, doc.url)
        doc.error = (f"page rendered only {len(doc.text.strip())} characters "
                     f"(JavaScript-rendered?)"
                     + (f"; links {len(doc.alternates)} document(s)" if doc.alternates else ""))
    elif not doc.text.strip():
        doc.error = doc.error or "no extractable text"

    # Licence last, on the extracted text, so a statement inside a PDF is seen.
    doc.licence = LIC.detect(url, doc.text, body_html or None)
    if not doc.title:
        doc.title = url.rsplit("/", 1)[-1] or url
    return doc


def save_raw(doc: Document, root: Path) -> Path | None:
    """Store the original bytes under a content-addressed name.

    Named by hash rather than by URL so the same document fetched from three mirrors —
    which is the normal outcome of a search — occupies disk once and is trivially
    recognised as already present.
    """
    if doc.raw is None or not doc.sha256:
        return None
    ext = ".pdf" if "pdf" in doc.content_type else (
        ".html" if "html" in doc.content_type else ".txt")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{doc.sha256[:16]}{ext}"
    if not path.exists():
        path.write_bytes(doc.raw)
    return path
