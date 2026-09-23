"""Tests for lara.serve.papers.figure_image -- the image and caption at one figure/table
float's own anchor, out of the paper's already-cached, already-sanitised HTML."""
from __future__ import annotations

from pathlib import Path

import zstandard as zstd

from lara.serve import papers as P

ARXIV_ID = "2401.00001"


def _cached(tmp_path: Path, html: str, *, source: str = "arxiv_html") -> str:
    path = tmp_path / f"{ARXIV_ID}.{source}.html.zst"
    path.write_bytes(zstd.ZstdCompressor().compress(html.encode()))
    P.render.cache_clear()          # a fresh path string per test still shares the process cache
    return str(path)


HTML = """<html><body><article>
  <section id="S4"><h2>Results</h2>
    <figure id="S4.F2">
      <img src="x1.png">
      <figcaption>Figure 2: loss over training.</figcaption>
    </figure>
    <figure id="S4.F3"><figcaption>Figure 3: no image, just a table description.</figcaption></figure>
    <div id="S4.p1" class="ltx_para"><p>Ordinary paragraph, not a figure.</p></div>
  </section>
</article></body></html>"""


def test_figure_image_finds_the_image_and_caption_by_anchor(tmp_path):
    out = P.figure_image(_cached(tmp_path, HTML), ARXIV_ID, 1, "S4.F2")
    assert out["src"] == "https://arxiv.org/html/2401.00001v1/x1.png"
    assert out["caption"] == "Figure 2: loss over training."


def test_figure_image_is_none_for_a_figure_with_no_image(tmp_path):
    assert P.figure_image(_cached(tmp_path, HTML), ARXIV_ID, 1, "S4.F3") is None


def test_figure_image_is_none_for_an_anchor_that_is_not_a_figure(tmp_path):
    assert P.figure_image(_cached(tmp_path, HTML), ARXIV_ID, 1, "S4.p1") is None
    assert P.figure_image(_cached(tmp_path, HTML), ARXIV_ID, 1, "S4") is None


def test_figure_image_is_none_for_an_anchor_that_does_not_exist(tmp_path):
    assert P.figure_image(_cached(tmp_path, HTML), ARXIV_ID, 1, "S9.F1") is None


def test_figure_image_resolves_against_ar5iv_when_that_is_the_cached_source(tmp_path):
    out = P.figure_image(_cached(tmp_path, HTML, source="ar5iv"), ARXIV_ID, 1, "S4.F2")
    assert out["src"] == "https://ar5iv.labs.arxiv.org/html/2401.00001/x1.png"
