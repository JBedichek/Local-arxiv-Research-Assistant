"""Work out what a fetched document permits, and say so in plain terms.

The reader will happily index a document nobody may redistribute. That is usually fine —
indexing the manual for the aircraft you fly is ordinary personal use — and it is exactly
the same act that becomes a problem the moment the corpus is published. The difference is
not technical, so the software cannot decide it; what it *can* do is find the licence,
report it in words rather than in an SPDX code, and refuse to publish what it cannot
clear.

Detection is deliberately conservative. A page with no licence signal is reported as
UNKNOWN rather than assumed permissive, because the failure modes are asymmetric: calling
a CC-BY textbook "unknown" costs the reader a moment's confirmation, and calling a
copyrighted manual "public domain" costs them a legal problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── verdicts, ordered from most to least freedom ────────────────────────────────────
PUBLIC_DOMAIN = "public-domain"
PERMISSIVE = "permissive"          # CC BY, CC BY-SA, MIT, Apache — redistribution allowed
RESTRICTED = "restricted"          # CC NC / ND — redistribution allowed, with conditions
COPYRIGHTED = "copyrighted"        # all rights reserved
UNKNOWN = "unknown"                # no signal found; assume nothing

REDISTRIBUTABLE = {PUBLIC_DOMAIN, PERMISSIVE}

PLAIN = {
    PUBLIC_DOMAIN: "Public domain — free to use, keep and share.",
    PERMISSIVE: "Openly licensed — you may share a corpus containing this, with attribution.",
    RESTRICTED: "Openly licensed but with conditions (non-commercial and/or no derivatives). "
                "Fine to read; sharing a corpus built from it may breach the licence.",
    COPYRIGHTED: "All rights reserved. Fine to keep and search for yourself; "
                 "publishing a corpus containing it would infringe copyright.",
    UNKNOWN: "No licence found. Treat as all rights reserved: keep it, do not publish it.",
}

#: Sites whose licensing is a property of the site rather than of the page. Cheaper and
#: far more reliable than parsing, and it covers most of what a study corpus pulls in.
BY_DOMAIN: dict[str, tuple[str, str]] = {
    "openstax.org": (PERMISSIVE, "CC BY 4.0 (OpenStax)"),
    "assets.openstax.org": (PERMISSIVE, "CC BY 4.0 (OpenStax)"),
    "gutenberg.org": (PUBLIC_DOMAIN, "Project Gutenberg (public domain in the US)"),
    "www.gutenberg.org": (PUBLIC_DOMAIN, "Project Gutenberg (public domain in the US)"),
    "arxiv.org": (UNKNOWN, "arXiv — licence is per paper, not per site"),
    "ocw.mit.edu": (RESTRICTED, "MIT OpenCourseWare, CC BY-NC-SA"),
    "en.wikipedia.org": (PERMISSIVE, "CC BY-SA 4.0 (Wikipedia)"),
    "en.wikibooks.org": (PERMISSIVE, "CC BY-SA 4.0 (Wikibooks)"),
    "plato.stanford.edu": (RESTRICTED, "Stanford Encyclopedia, no redistribution"),
}

_CC_URL = re.compile(
    r"creativecommons\.org/(?:licenses|publicdomain)/([a-z0-9\-]+)(?:/([0-9.]+))?", re.I)
_SPDX = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+)")
_ARR = re.compile(r"all\s+rights\s+reserved", re.I)
_COPYRIGHT = re.compile(r"(?:©|\(c\)|copyright)\s*(?:19|20)\d{2}", re.I)

_SPDX_PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc",
                    "cc-by-4.0", "cc-by-sa-4.0", "cc0-1.0", "unlicense"}


@dataclass
class Licence:
    verdict: str
    label: str
    evidence: str = ""

    @property
    def redistributable(self) -> bool:
        return self.verdict in REDISTRIBUTABLE

    def explain(self) -> str:
        return PLAIN[self.verdict]


def _from_cc_code(code: str, version: str | None) -> Licence:
    code = code.lower()
    if code in ("zero", "mark"):
        return Licence(PUBLIC_DOMAIN, f"CC0{' ' + version if version else ''}")
    label = f"CC {code.upper()}{' ' + version if version else ''}"
    # NC forbids commercial use and ND forbids derivatives; a chunked, re-embedded corpus
    # is arguably a derivative, so neither is safe to call redistributable.
    if "nc" in code.split("-") or "nd" in code.split("-"):
        return Licence(RESTRICTED, label)
    return Licence(PERMISSIVE, label)


def detect(url: str, text: str, html: str | None = None) -> Licence:
    """Best guess at the licence of one document.

    Checked in order of reliability: the site, then an explicit machine-readable marker,
    then a Creative Commons link anywhere in the page, then a bare copyright notice. The
    first two are assertions by the publisher; the last is only evidence that someone
    claims rights, which is still worth acting on.
    """
    from urllib.parse import urlsplit

    host = urlsplit(url).netloc.lower()
    if host in BY_DOMAIN:
        verdict, label = BY_DOMAIN[host]
        if verdict != UNKNOWN:
            return Licence(verdict, label, evidence=f"known publisher: {host}")

    blob = f"{html or ''}\n{text[:200_000]}"

    m = _SPDX.search(blob)
    if m:
        code = m.group(1)
        verdict = PERMISSIVE if code.lower() in _SPDX_PERMISSIVE else UNKNOWN
        return Licence(verdict, code, evidence="SPDX-License-Identifier")

    m = _CC_URL.search(blob)
    if m:
        lic = _from_cc_code(m.group(1), m.group(2))
        lic.evidence = "creativecommons.org link"
        return lic

    if _ARR.search(blob):
        return Licence(COPYRIGHTED, "All rights reserved", evidence='"all rights reserved"')
    m = _COPYRIGHT.search(blob)
    if m:
        return Licence(COPYRIGHTED, m.group(0).strip(), evidence="copyright notice")

    if host in BY_DOMAIN:                       # arXiv lands here: known site, no page signal
        return Licence(*BY_DOMAIN[host], evidence=f"known publisher: {host}")
    return Licence(UNKNOWN, "unknown")


def corpus_verdict(licences: list[Licence]) -> tuple[bool, str]:
    """Whether a whole corpus may be published, and why not when it may not."""
    blocking = [x for x in licences if not x.redistributable]
    if not blocking:
        return True, f"All {len(licences)} sources are openly licensed or public domain."
    counts: dict[str, int] = {}
    for x in blocking:
        counts[x.verdict] = counts.get(x.verdict, 0) + 1
    parts = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
    return False, (
        f"{len(blocking)} of {len(licences)} sources cannot be redistributed ({parts}). "
        "Keeping and searching them locally is fine; publishing this corpus is not."
    )
