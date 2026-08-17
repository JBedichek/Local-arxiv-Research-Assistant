"""arXiv OAI-PMH metadata harvester — resumable, checkpointed every page.

Why OAI-PMH rather than the Kaggle snapshot: no account, no API token, no 4.5 GB
download, and it doubles as the nightly incremental path. Confirmed working 2026-08-17,
~1300 records per request with a resumption token.

**The `from=` trap.** OAI-PMH's `from`/`until` filter on the OAI *datestamp*, which is the
record's last-modified time, not the submission date. Paper 1107.0901 (submitted 2011)
carries datestamp 2026-08-03 because its metadata was touched. So `from=2015-01-01` would
neither exclude old papers nor include everything we want. We therefore harvest the sets in
full and apply the 2015 submission floor from the ``v1`` version date at ingest.

**Checkpointing.** Records and the next resumption token are written in one transaction.
A kill -9 mid-harvest resumes from a token whose page is already committed — never a lost
page, never a double-counted one. That means the harvest can run for an hour while we
iterate on everything downstream.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Iterator

import httpx
from lxml import etree

from lara.store.db import set_state, utcnow

OAI_NS = "http://www.openarchives.org/OAI/2.0/"
RAW_NS = "http://arxiv.org/OAI/arXivRaw/"
NS = {"o": OAI_NS, "r": RAW_NS}

# arXiv OAI sets that contain our categories: cs.LG/cs.CL/cs.NE live in `cs`,
# stat.ML lives in `stat`.
DEFAULT_SETS = ("cs", "stat")


@dataclass
class Record:
    arxiv_id: str
    latest_version: int
    title: str
    abstract: str
    authors: str
    categories: str
    submitted_utc: str | None
    updated_utc: str | None
    oai_datestamp: str
    doi: str | None
    journal_ref: str | None
    license: str | None
    deleted: bool = False

    @property
    def primary_category(self) -> str:
        return self.categories.split()[0] if self.categories else ""


def _text(node, path: str) -> str | None:
    found = node.find(path, NS)
    if found is None or found.text is None:
        return None
    return " ".join(found.text.split())


def _to_iso(raw: str | None) -> str | None:
    """'Tue, 05 Jul 2011 15:27:34 GMT' -> ISO-8601 UTC."""
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def parse_page(xml: bytes) -> tuple[list[Record], str | None]:
    """Parse one ListRecords response into records plus the next resumption token."""
    root = etree.fromstring(xml)

    err = root.find("o:error", NS)
    if err is not None:
        code = err.get("code", "")
        # noRecordsMatch / badResumptionToken are terminal, not retryable.
        raise OAIError(code, (err.text or "").strip())

    records: list[Record] = []
    for rec in root.iter(f"{{{OAI_NS}}}record"):
        header = rec.find("o:header", NS)
        if header is None:
            continue
        identifier = _text(header, "o:identifier") or ""
        arxiv_id = identifier.rsplit(":", 1)[-1]
        datestamp = _text(header, "o:datestamp") or ""

        if header.get("status") == "deleted":
            records.append(Record(
                arxiv_id=arxiv_id, latest_version=0, title="", abstract="", authors="",
                categories="", submitted_utc=None, updated_utc=None,
                oai_datestamp=datestamp, doi=None, journal_ref=None, license=None,
                deleted=True,
            ))
            continue

        raw = rec.find("o:metadata/r:arXivRaw", NS)
        if raw is None:
            continue

        versions = raw.findall("r:version", NS)
        v_dates = [_to_iso(_text(v, "r:date")) for v in versions]
        records.append(Record(
            arxiv_id=_text(raw, "r:id") or arxiv_id,
            latest_version=len(versions) or 1,
            title=_text(raw, "r:title") or "",
            abstract=_text(raw, "r:abstract") or "",
            authors=_text(raw, "r:authors") or "",
            categories=_text(raw, "r:categories") or "",
            submitted_utc=v_dates[0] if v_dates else None,
            updated_utc=next((d for d in reversed(v_dates) if d), None),
            oai_datestamp=datestamp,
            doi=_text(raw, "r:doi"),
            journal_ref=_text(raw, "r:journal-ref"),
            license=_text(raw, "r:license"),
        ))

    token_node = root.find(".//o:resumptionToken", NS)
    token = token_node.text.strip() if token_node is not None and token_node.text else None
    return records, token


class OAIError(RuntimeError):
    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(f"OAI error [{code}] {message}".strip())


class Harvester:
    """Sequential, polite, resumable.

    arXiv uses HTTP 503 with ``Retry-After`` for OAI flow control rather than as an
    error — it is the documented way the server asks you to slow down, so it is honoured
    exactly rather than treated as a failure.
    """

    def __init__(
        self,
        endpoint: str,
        user_agent: str,
        *,
        min_interval_sec: float = 3.0,
        timeout_sec: float = 120.0,
        max_retries: int = 8,
    ) -> None:
        self.endpoint = endpoint
        self.min_interval = min_interval_sec
        self.max_retries = max_retries
        self._last_request = 0.0
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_sec,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, params: dict[str, str]) -> bytes:
        delay = 30.0
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self._client.get(self.endpoint, params=params)
            except httpx.RequestError as exc:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(min(delay, 600))
                delay *= 2
                _ = exc
                continue

            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 503:
                # Documented flow control, not an error. Obey Retry-After.
                wait = float(resp.headers.get("Retry-After", delay))
                time.sleep(min(wait, 600))
                continue
            if resp.status_code in (429, 500, 502, 504):
                time.sleep(min(delay, 600))
                delay *= 2
                continue
            resp.raise_for_status()
        raise RuntimeError(f"OAI request failed after {self.max_retries} attempts: {params}")

    def pages(self, set_spec: str, resume_token: str | None = None) -> Iterator[tuple[list[Record], str | None]]:
        """Yield (records, next_token) per page. Caller commits, then continues."""
        token = resume_token
        while True:
            params = (
                {"verb": "ListRecords", "resumptionToken": token}
                if token else
                {"verb": "ListRecords", "set": set_spec, "metadataPrefix": "arXivRaw"}
            )
            try:
                records, token = parse_page(self._get(params))
            except OAIError as exc:
                if exc.code in ("noRecordsMatch",):
                    return
                raise
            yield records, token
            if not token:
                return


# ── persistence ────────────────────────────────────────────────────────────────────

UPSERT = """
INSERT INTO papers (
    arxiv_id, latest_version, title, abstract, authors, categories, primary_category,
    submitted_utc, updated_utc, oai_datestamp, doi, journal_ref, license, deleted, in_scope
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(arxiv_id) DO UPDATE SET
    latest_version=excluded.latest_version,
    title=excluded.title,
    abstract=excluded.abstract,
    authors=excluded.authors,
    categories=excluded.categories,
    primary_category=excluded.primary_category,
    submitted_utc=excluded.submitted_utc,
    updated_utc=excluded.updated_utc,
    oai_datestamp=excluded.oai_datestamp,
    doi=excluded.doi,
    journal_ref=excluded.journal_ref,
    license=excluded.license,
    deleted=excluded.deleted,
    in_scope=excluded.in_scope
"""


def make_scope_filter(categories: list[str], date_floor: str) -> Callable[[Record], bool]:
    """D1: in scope iff it carries one of our categories and was submitted on/after the floor.

    The floor is applied to the *v1 submission* date, not the OAI datestamp — see the
    module docstring.
    """
    wanted = set(categories)
    floor = datetime.fromisoformat(date_floor).replace(tzinfo=timezone.utc)

    def predicate(rec: Record) -> bool:
        if rec.deleted or not rec.categories:
            return False
        if not wanted.intersection(rec.categories.split()):
            return False
        if not rec.submitted_utc:
            return False
        return datetime.fromisoformat(rec.submitted_utc) >= floor

    return predicate


def write_page(
    conn: sqlite3.Connection,
    records: list[Record],
    in_scope: Callable[[Record], bool],
    *,
    set_spec: str,
    next_token: str | None,
    request_n: int,
) -> int:
    """Commit one page of records and its resumption token atomically."""
    rows = [
        (
            r.arxiv_id, r.latest_version, r.title, r.abstract, r.authors, r.categories,
            r.primary_category, r.submitted_utc, r.updated_utc, r.oai_datestamp,
            r.doi, r.journal_ref, r.license, int(r.deleted), int(in_scope(r)),
        )
        for r in records
    ]
    with conn:  # single transaction: records + token together
        conn.executemany(UPSERT, rows)
        set_state(conn, f"oai_token:{set_spec}", next_token)
        set_state(conn, f"oai_requests:{set_spec}", str(request_n))
        if next_token is None:
            set_state(conn, f"oai_complete:{set_spec}", utcnow())
        conn.execute(
            "INSERT INTO harvest_log(set_spec, request_n, records, ts_utc, note) "
            "VALUES (?,?,?,?,?)",
            (set_spec, request_n, len(records), utcnow(), "done" if next_token is None else ""),
        )
    return sum(1 for r in records if in_scope(r))
