import asyncio

from lara.learn.llm import CHARS_PER_TOKEN, TokenMeter, metered
from learn_helpers import llm


def run(c):
    return asyncio.run(c)


def _tokens(text: str) -> int:
    return round(len(text) / CHARS_PER_TOKEN)


def test_metered_accumulates_estimated_tokens_from_one_call():
    m = llm(("system", "a reply"))
    meter = TokenMeter()
    wrapped = metered(m, meter)
    run(wrapped.ask("a system prompt", "a user prompt"))
    assert meter.tokens_in == _tokens("a system prompt") + _tokens("a user prompt")
    assert meter.tokens_out == _tokens("a reply")


def test_metered_adds_across_calls_rather_than_replacing():
    m = llm(("system", "a reply"))
    meter = TokenMeter()
    wrapped = metered(m, meter)
    run(wrapped.ask("system", "first prompt"))
    after_one = meter.tokens_in
    run(wrapped.ask("system", "second prompt, a bit longer than the first"))
    assert meter.tokens_in > after_one
    assert meter.tokens_out == 2 * _tokens("a reply")


def test_metered_returns_exactly_what_the_original_would_have():
    m = llm(("system", "the exact reply"))
    meter = TokenMeter()
    wrapped = metered(m, meter)
    assert run(wrapped.ask("system", "prompt")) == run(m.ask("system", "prompt")) == "the exact reply"


def test_two_independently_metered_wrappers_do_not_share_a_counter():
    m = llm(("system", "a reply"))
    meter_a, meter_b = TokenMeter(), TokenMeter()
    a, b = metered(m, meter_a), metered(m, meter_b)
    run(a.ask("system", "prompt"))
    assert meter_a.tokens_in > 0 and meter_b.tokens_in == 0 and meter_b.tokens_out == 0


def test_metered_shares_the_original_llms_concurrency_semaphore():
    m = llm(("system", "a reply"))
    assert m._sem is None
    wrapped = metered(m, TokenMeter())
    assert m._sem is not None, "created eagerly so the wrapped copy shares it, not its own"
    assert wrapped._sem is m._sem
