"""Tests for lara.serve.generation_rate -- tokens/sec read from vLLM's own /metrics."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from lara.serve import generation_rate as GR

BASE_URL = "http://127.0.0.1:8000/v1"


def run(c):
    return asyncio.run(c)


def _client(body: str = None, *, status: int = 200, raise_on=None):
    def handler(request):
        if raise_on is not None:
            raise raise_on
        return httpx.Response(status, text=body or "")
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _reset():
    GR._last.clear()
    yield
    GR._last.clear()


def test_metrics_url_is_the_replica_root_not_under_v1():
    assert GR._metrics_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/metrics"
    assert GR._metrics_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/metrics"


def test_the_first_read_of_a_replica_has_nothing_to_diff_against():
    body = "vllm:generation_tokens_total 100.0\n"
    out = run(GR.rate(BASE_URL, client=_client(body)))
    assert out == {"tokens_per_sec": 0.0, "reachable": True}


def test_a_second_read_reports_the_delta_over_elapsed_time():
    run(GR.rate(BASE_URL, client=_client("vllm:generation_tokens_total 100.0\n"), now=1000.0))
    out = run(GR.rate(BASE_URL, client=_client("vllm:generation_tokens_total 150.0\n"), now=1002.0))
    assert out == {"tokens_per_sec": 25.0, "reachable": True}


def test_a_counter_that_only_went_backwards_floors_at_zero_not_negative():
    run(GR.rate(BASE_URL, client=_client("vllm:generation_tokens_total 100.0\n"), now=1000.0))
    out = run(GR.rate(BASE_URL, client=_client("vllm:generation_tokens_total 40.0\n"), now=1002.0))
    assert out == {"tokens_per_sec": 0.0, "reachable": True}


def test_labeled_counter_form_is_parsed_the_same_as_the_bare_one():
    body = 'vllm:generation_tokens_total{model_name="q"} 100.0\n'
    assert run(GR.rate(BASE_URL, client=_client(body)))["reachable"] is True


def test_an_endpoint_that_answers_but_has_no_counter_is_unreachable():
    out = run(GR.rate(BASE_URL, client=_client("vllm:num_requests_running 1.0\n")))
    assert out == {"tokens_per_sec": 0.0, "reachable": False}


def test_a_non_200_status_is_unreachable():
    out = run(GR.rate(BASE_URL, client=_client("vllm:generation_tokens_total 1.0\n", status=500)))
    assert out == {"tokens_per_sec": 0.0, "reachable": False}


def test_a_connection_failure_is_unreachable_not_an_exception():
    out = run(GR.rate(BASE_URL, client=_client(raise_on=httpx.ConnectError("refused"))))
    assert out == {"tokens_per_sec": 0.0, "reachable": False}


def test_different_base_urls_are_tracked_independently():
    run(GR.rate("http://127.0.0.1:8000/v1", client=_client("vllm:generation_tokens_total 100.0\n"), now=1000.0))
    run(GR.rate("http://127.0.0.1:8001/v1", client=_client("vllm:generation_tokens_total 500.0\n"), now=1000.0))
    out = run(GR.rate("http://127.0.0.1:8000/v1", client=_client("vllm:generation_tokens_total 120.0\n"), now=1002.0))
    assert out == {"tokens_per_sec": 10.0, "reachable": True}
