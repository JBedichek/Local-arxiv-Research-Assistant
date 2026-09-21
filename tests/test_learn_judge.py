import asyncio

from lara.learn import judge as J
from lara.learn.llm import parse_json
from learn_helpers import llm


def run(coro):
    return asyncio.run(coro)


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('Here you go:\n```json\n{"a": [1, 2]}\n```\nthanks') == {"a": [1, 2]}
    assert parse_json("no json here") is None
    assert parse_json('[{"x": 1}] trailing') == [{"x": 1}]


def test_judge_reads_the_first_word_and_fails_closed():
    for reply, expect in [("supports", J.SUPPORTS), ("Contradicts.", J.CONTRADICTS),
                          ("unrelated", J.UNRELATED), ("The passage supports it", J.UNRELATED),
                          ("", J.UNRELATED), ("maybe", J.UNRELATED)]:
        assert run(J.judge(llm(("CLAIM", reply)), "c", "p")) == expect


def test_judge_of_an_empty_claim_or_passage_never_calls_the_model():
    m = llm(("CLAIM", "supports"))
    assert run(J.judge(m, "", "p")) == J.UNRELATED and run(J.judge(m, "c", " ")) == J.UNRELATED
    assert m.calls == []


def test_judge_prompt_carries_only_the_claim_and_the_passage():
    m = llm(("CLAIM", "supports"))
    run(J.judge(m, "warmup helps", "the passage text"))
    system, prompt = m.calls[0]
    assert "CLAIM: warmup helps" in prompt and "PASSAGE: the passage text" in prompt


def test_compare_returns_relation_and_note_and_rejects_unknown_relations():
    m = llm(("CLAIM A", '{"relation": "scope", "note": "1B vs 70B"}'))
    assert run(J.compare(m, "a", "pa", "b", "pb")) == ("scope", "1B vs 70B")
    m = llm(("CLAIM A", '{"relation": "bogus"}'))
    assert run(J.compare(m, "a", "pa", "b", "pb")) == ("unrelated", "")
    assert run(J.compare(llm(("CLAIM A", "garbage")), "a", "pa", "b", "pb")) == ("unrelated", "")


def test_solve_returns_none_when_the_solver_cannot_tell():
    assert run(J.solve(llm(("QUESTION", "CANNOT DETERMINE")), "q", "p")) is None
    assert run(J.solve(llm(("QUESTION", "B")), "q", "p", ["x", "y"])) == "B"


def test_same_reads_the_first_word():
    assert run(J.same(llm(("CLAIM A", "same")), "a", "b")) is True
    assert run(J.same(llm(("CLAIM A", "different")), "a", "b")) is False
    assert run(J.same(llm(("CLAIM A", "")), "a", "b")) is False


def test_relevance_defaults_to_yes_and_reads_no():
    assert run(J.relevant(llm(), "", "c", "x")) is True
    assert run(J.relevant(llm(("GOAL", "no")), "goal", "c", "x")) is False
    assert run(J.relevant(llm(("GOAL", "Yes")), "goal", "c", "x")) is True
    assert run(J.relevant(llm(("GOAL", "")), "goal", "c", "x")) is True, "an unreadable reply must not silently drop claims"


def test_parse_json_survives_latex_with_single_backslashes():
    got = parse_json('{"sections": [{"heading": "Bias", "focus": "Derive $\\hat{m}_t = m_t/(1-\\beta^t)$ for the first moment"}]}')
    assert got["sections"][0]["focus"] == "Derive $\\hat{m}_t = m_t/(1-\\beta^t)$ for the first moment"


def test_parse_json_leaves_valid_escapes_alone():
    assert parse_json('{"a": "line1\\nline2", "b": "say \\"hi\\"", "c": "back\\\\slash"}') == \
        {"a": "line1\nline2", "b": 'say "hi"', "c": "back\\slash"}


def test_a_broken_first_candidate_is_repaired_before_an_inner_object_is_tried():
    """The outer object holds the LaTeX; without the repair, an inner object without any would
    be returned instead and the caller would get a fragment."""
    text = '{"sections": [{"heading": "A", "focus": "uses $\\beta$"}, {"heading": "B", "focus": "plain"}]}'
    assert len(parse_json(text)["sections"]) == 2


def test_latex_whose_first_letters_look_like_json_escapes_is_not_silently_mangled():
    got = parse_json('{"focus": "the step \\beta_1 and \\frac{a}{b} with \\tau"}')
    assert got["focus"] == "the step \\beta_1 and \\frac{a}{b} with \\tau"
    assert "\x08" not in got["focus"] and "\x0c" not in got["focus"] and "\t" not in got["focus"]
