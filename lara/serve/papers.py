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


def _absolutize(root, arxiv_id: str) -> None:
    """Point relative assets at arXiv so figures still render."""
    base = f"https://arxiv.org/html/{arxiv_id}/"
    for el in root.xpath("//img[@src] | //source[@src]"):
        src = el.get("src") or ""
        if src and not src.startswith(("http://", "https://", "data:")):
            el.set("src", base + src.lstrip("./"))
        el.set("loading", "lazy")
    for el in root.xpath("//a[@href]"):
        href = el.get("href") or ""
        if href.startswith("#"):
            continue
        if href and not href.startswith(("http://", "https://", "mailto:")):
            el.set("href", base + href.lstrip("./"))
        el.set("target", "_blank")
        el.set("rel", "noopener")


@functools.lru_cache(maxsize=32)
def render(path_str: str, arxiv_id: str) -> str:
    """Decompress, sanitise and return the paper body. Cached — re-opens are free."""
    raw = _dctx.decompress(Path(path_str).read_bytes())
    root = LH.fromstring(raw)

    for el in root.xpath(STRIP):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)
    _absolutize(root, arxiv_id)

    body = root.xpath("//article") or root.xpath("//body") or [root]
    out = LH.tostring(body[0], encoding="unicode")
    # Drop inline event handlers; the CSP blocks them but this keeps the DOM clean.
    return re.sub(r'\son\w+="[^"]*"', "", out)


def anchor_exists(html_text: str, anchor: str) -> bool:
    return f'id="{anchor}"' in html_text
