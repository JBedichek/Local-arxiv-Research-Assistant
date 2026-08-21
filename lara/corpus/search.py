"""Web search, with no account and no API key.

The model never touches the network. It emits a query string; this module performs the
search and hands results back. That is an ordinary tool-use loop, and nothing about
serving the model locally prevents it.

**Why DuckDuckGo rather than Google.** Not a licensing argument — a measurement. Fetching
``google.com/search`` from here returns HTTP 200 and *zero extractable links*, because the
results are rendered by JavaScript and the HTML that arrives contains no result anchors.
DuckDuckGo's HTML endpoint returns parseable anchors and, on the first query tried
("openstax calculus volume 1 pdf"), put the textbook itself and an Internet Archive mirror
in the top four. Google's sanctioned route is the Custom Search JSON API, which needs a key
and an engine id; the point of this module is that the reader needs neither.

``search()`` is deliberately a one-function interface returning plain dicts, so a keyed
backend can be dropped in later without anything upstream noticing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from lxml import html as LH

DDG_HTML = "https://html.duckduckgo.com/html/"

#: A real browser string. The endpoint serves a reduced page to clients it does not
#: recognise, and the reduced page has no result anchors at all.
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0.0.0 Safari/537.36")

#: Minimum gap between queries.
#:
#: 1.5 s was not enough: eight queries inside a few minutes got this address soft-blocked,
#: after which every search returned an "anomaly" page. A corpus build issues tens of
#: searches over minutes, not thousands per second, so waiting is nearly free here and
#: being blocked halfway through a build is not.
MIN_INTERVAL_SEC = 6.0

#: After a block, wait this long before the next attempt, doubling each time. The block is
#: applied to the address rather than the query, so hammering a different query is not a
#: way around it — only time is.
BACKOFF_START_SEC = 30.0
BACKOFF_MAX_SEC = 480.0

#: The block is served as HTTP 202 with a page that mentions an anomaly, NOT as 429. So
#: `raise_for_status()` sees success and the parser simply finds no results — which is
#: indistinguishable from "this query has no matches" unless it is checked for explicitly.
#: The first version of this module reported an empty list and no error, which is precisely
#: the confusion SearchStats.errors exists to prevent.
BLOCK_MARKERS = ("anomaly", "unusual traffic", "captcha")

#: Where answered queries are kept. Caching matters more here than it usually does: the
#: engine throttles by address, so every query avoided is throttle budget preserved, and a
#: corpus recipe re-runs its queries on every rebuild. With a cache, rebuilding a corpus
#: from its recipe costs no searches at all.
CACHE_DIR = Path(os.environ.get("LARA_SEARCH_CACHE",
                                Path.home() / ".cache" / "lara" / "search"))

#: Search results for textbooks and manuals do not turn over quickly, and a month-old
#: answer to "free calculus textbook pdf" is the same answer. Long by the standards of a
#: news crawler, correct for this.
CACHE_TTL_SEC = 30 * 24 * 3600

#: Hosts that are the search engine talking about itself, never a result worth fetching.
SELF_DOMAINS = ("duckduckgo.com", "duck.com")

_last_query = 0.0
_backoff = 0.0


class Blocked(Exception):
    """The search engine is refusing this address for now."""


def _cache_path(query: str, k: int) -> Path:
    key = hashlib.sha256(f"{query}\x00{k}".encode()).hexdigest()[:24]
    return CACHE_DIR / f"{key}.json"


def _cache_read(query: str, k: int, *, allow_stale: bool = False
                ) -> tuple[list["Result"], bool] | None:
    """Cached results and whether they were stale, or None if nothing is stored."""
    path = _cache_path(query, k)
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    age = time.time() - blob.get("stored", 0)
    stale = age > CACHE_TTL_SEC
    if stale and not allow_stale:
        return None
    return [Result(**r) for r in blob.get("results", [])], stale


def _cache_write(query: str, k: int, results: list["Result"]) -> None:
    # An empty result set is not cached: it is far more often a block or a transient
    # failure than a genuine absence, and caching it would make one bad minute look like a
    # permanent fact about the query.
    if not results:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(query, k).write_text(json.dumps(
            {"query": query, "k": k, "stored": time.time(),
             "results": [asdict(r) for r in results]}))
    except OSError:
        pass


def _looks_blocked(status: int, body: str) -> bool:
    if status == 429:
        return True
    if not body:
        return False
    head = body[:20000].lower()
    return "result__a" not in head and any(m in head for m in BLOCK_MARKERS)


@dataclass
class Result:
    url: str
    title: str = ""
    snippet: str = ""
    query: str = ""
    rank: int = 0

    def domain(self) -> str:
        return urllib.parse.urlsplit(self.url).netloc.lower()


@dataclass
class SearchStats:
    queries: int = 0
    results: int = 0
    blocked: int = 0
    cached: int = 0          # served from cache, no request made
    stale: int = 0           # served from an expired cache because we were blocked
    errors: list[str] = field(default_factory=list)

    @property
    def was_blocked(self) -> bool:
        """True when emptiness means "refused", not "nothing matched"."""
        return self.blocked > 0


def _unwrap(href: str) -> str:
    """DuckDuckGo wraps results in a redirect; the real URL is the ``uddg`` parameter."""
    if href.startswith("//"):
        href = "https:" + href
    m = re.search(r"[?&]uddg=([^&]+)", href)
    return urllib.parse.unquote(m.group(1)) if m else href


def search(query: str, k: int = 10, *, timeout: float = 25.0,
           stats: SearchStats | None = None, use_cache: bool = True) -> list[Result]:
    """Top-k results for one query. Never raises: a failed search returns nothing.

    A corpus build issues many queries and one of them timing out should cost that query's
    results, not the build. Failures are recorded on ``stats`` when given, so the caller
    can tell "found nothing" apart from "could not reach the internet" — which look
    identical from an empty list and mean completely different things to a user.
    """
    global _last_query, _backoff
    if use_cache:
        hit = _cache_read(query, k)
        if hit is not None:
            if stats is not None:
                stats.cached += 1
            return hit[0]

    wait = max(MIN_INTERVAL_SEC - (time.monotonic() - _last_query), _backoff)
    if wait > 0:
        time.sleep(wait)
    _backoff = 0.0
    _last_query = time.monotonic()

    if stats is not None:
        stats.queries += 1
    try:
        resp = httpx.post(
            DDG_HTML, data={"q": query},
            headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
            timeout=timeout, follow_redirects=True,
        )
        if _looks_blocked(resp.status_code, resp.text):
            raise Blocked(f"HTTP {resp.status_code}, no results and an anomaly page")
        resp.raise_for_status()
    except Blocked as exc:
        # Arm the backoff for whoever searches next, and say plainly that this is a block
        # rather than an absence of matches.
        _backoff = min(max(_backoff * 2, BACKOFF_START_SEC), BACKOFF_MAX_SEC)
        if stats is not None:
            stats.blocked += 1
            stats.errors.append(f"{query}: blocked ({exc}); waiting {_backoff:.0f}s")
        # Blocked is exactly when a stale answer earns its keep. Month-old results for
        # "free calculus textbook" are worth incomparably more than nothing, and the
        # alternative is a build that stops because a rate limiter is unhappy.
        if use_cache:
            hit = _cache_read(query, k, allow_stale=True)
            if hit is not None:
                if stats is not None:
                    stats.stale += 1
                return hit[0]
        return []
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        if stats is not None:
            stats.errors.append(f"{query}: {type(exc).__name__}")
        return []

    doc = LH.fromstring(resp.text)
    out: list[Result] = []
    seen: set[str] = set()
    for i, node in enumerate(doc.xpath('//div[contains(@class,"result")]')):
        href = node.xpath('.//a[contains(@class,"result__a")]/@href')
        if not href:
            continue
        url = _unwrap(href[0])
        if not url.startswith("http") or url in seen:
            continue
        # The results page links to itself — settings, feedback, "images" verticals — and
        # an unwrapped ad slot occasionally survives as a plain result. Neither is a
        # document, and both waste a fetch and a relevance judgement.
        if urllib.parse.urlsplit(url).netloc.lower().endswith(SELF_DOMAINS):
            continue
        seen.add(url)
        title = " ".join(node.xpath('.//a[contains(@class,"result__a")]//text()')).strip()
        snip = " ".join(node.xpath('.//a[contains(@class,"result__snippet")]//text()')).strip()
        out.append(Result(url=url, title=title, snippet=snip, query=query, rank=len(out) + 1))
        if len(out) >= k:
            break

    if stats is not None:
        stats.results += len(out)
    if use_cache:
        _cache_write(query, k, out)
    return out


def search_many(queries: list[str], k: int = 10,
                stats: SearchStats | None = None) -> list[Result]:
    """Several queries, deduplicated by URL, keeping the best rank each URL achieved.

    Overlap between queries is the normal case rather than the exception — "calculus
    textbook pdf" and "free calculus book" return many of the same pages — and a URL that
    ranked first for one query should not be demoted because it ranked eighth for another.
    """
    best: dict[str, Result] = {}
    for q in queries:
        for r in search(q, k=k, stats=stats):
            prev = best.get(r.url)
            if prev is None or r.rank < prev.rank:
                best[r.url] = r
    return sorted(best.values(), key=lambda r: r.rank)
