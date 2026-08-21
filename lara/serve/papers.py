"""Serve the paper pane from the local zstd cache (R3).

Papers are rendered from HTML we already fetched and compressed during the crawl, so a
paper opens in ~5 ms of local decompression instead of a 0.5–2 s round trip to arXiv. The
crawl had to download it anyway; keeping it means the reader never waits on the network.

The one inviolable rule: **do not touch element ids**. Every ``S3.p4`` in the DOM is a
citation target (PLAN.md §4). Sanitisation removes scripts, styles and navigation chrome,
and rewrites relative asset URLs — it never renumbers, wraps or restructures anything that
carries an id.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import zstandard as zstd
from lxml import html as LH

from lara.ingest.parse import cls

# Removed before serving: scripts (a strict CSP would block them anyway), arXiv's own
# navigation chrome, and stylesheet links to origins the page can no longer reach.
STRIP = " | ".join([
    "//script", "//noscript", "//link", "//style",
    f"//*[{cls('ltx_page_navbar')}]",
    f"//*[{cls('ltx_page_footer')}]",
    f"//*[{cls('ltx_TOC')}]",
    "//nav[contains(@class,'html-header-nav')]",
    "//nav[contains(@class,'ds-site-footer-links')]",
    "//*[@id='selectedTextModalDescription']",
])

_dctx = zstd.ZstdDecompressor()


def _resolve(url: str, origin: str, doc_base: str, html_root: str, arxiv_id: str) -> str:
    """Turn one relative reference into an absolute one.

    Three shapes appear in arXiv's HTML and the old single-prefix rule got two of them
    wrong, which is why no figure ever loaded:

    ``2504.11884v1/Fig1.png``  already carries the paper id, because the page is served
                               without a trailing slash and its own relative refs resolve
                               against ``/html/``. Prefixing the id again produced
                               ``/html/2504.11884/2504.11884v1/Fig1.png`` -> 404.
    ``/static/base/...svg``    root-relative, and belongs to the origin, not to this
                               paper. Concatenating gave ``/html/2504.11884//static/...``.
    ``x1.png``                 genuinely relative to the document directory.
    """
    if url.startswith("/"):
        return origin + url
    # An id-qualified path is relative to /html/, not to this paper's directory.
    if re.match(rf"^{re.escape(arxiv_id)}(v\d+)?/", url):
        return html_root + url
    return doc_base + url.lstrip("./")


def _absolutize(root, arxiv_id: str, version: int = 1, source: str = "arxiv_html") -> None:
    """Point relative assets at their origin so figures still render."""
    origin = ("https://ar5iv.labs.arxiv.org" if source == "ar5iv"
              else "https://arxiv.org")
    html_root = f"{origin}/html/"
    # ar5iv serves one document per id with no version segment; arXiv's own HTML is
    # versioned, and an unversioned path only works because it redirects.
    doc_base = (f"{html_root}{arxiv_id}/" if source == "ar5iv"
                else f"{html_root}{arxiv_id}v{version}/")

    for el in root.xpath("//img[@src] | //source[@src]"):
        src = el.get("src") or ""
        if src and not src.startswith(("http://", "https://", "data:")):
            el.set("src", _resolve(src, origin, doc_base, html_root, arxiv_id))
        el.set("loading", "lazy")
    for el in root.xpath("//a[@href]"):
        href = el.get("href") or ""
        if href.startswith("#"):
            continue
        if href and not href.startswith(("http://", "https://", "mailto:")):
            el.set("href", _resolve(href, origin, doc_base, html_root, arxiv_id))
        el.set("target", "_blank")
        el.set("rel", "noopener")


@functools.lru_cache(maxsize=32)
def render(path_str: str, arxiv_id: str, version: int = 1) -> str:
    """Decompress, sanitise and return the paper body. Cached — re-opens are free."""
    raw = _dctx.decompress(Path(path_str).read_bytes())
    root = LH.fromstring(raw)

    for el in root.xpath(STRIP):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)
    # The cache filename carries which source produced this copy, and the two serve
    # their assets from different origins.
    source = "ar5iv" if ".ar5iv." in Path(path_str).name else "arxiv_html"
    _absolutize(root, arxiv_id, version, source)

    body = root.xpath("//article") or root.xpath("//body") or [root]
    out = LH.tostring(body[0], encoding="unicode")
    # Drop inline event handlers; the CSP blocks them but this keeps the DOM clean.
    return re.sub(r'\son\w+="[^"]*"', "", out)


def anchor_exists(html_text: str, anchor: str) -> bool:
    return f'id="{anchor}"' in html_text
