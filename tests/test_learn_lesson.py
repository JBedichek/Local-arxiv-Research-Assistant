import asyncio
import json

from lara.learn import lesson as LE
from learn_helpers import llm


def run(c):
    return asyncio.run(c)


def claim(key, text, certainty="established", conditions="", date="2024-01-01"):
    return {"key": key, "text": text, "certainty": certainty, "conditions": conditions,
            "passage": {"date": date, "title": "P", "arxiv_id": "2401.1"}}


CLAIMS = [claim("c1", "Warmup avoids early loss spikes."), claim("c2", "Peak LR should scale with batch size.", "single-source")]
CONCEPT = {"id": "c1", "title": "LR warmup", "summary": "s"}


def test_parse_reads_headings_sentences_and_keys_and_flags_unkeyed_lines():
    text = "## Basics\nWarmup avoids spikes [c1].\nAnother point [c2, c1].\nNo key here.\n- Bullet [c9]."
    [(heading, sents)] = LE.parse(text, {"c1", "c2"})
    assert heading == "Basics"
    assert sents[0] == {"text": "Warmup avoids spikes.", "claims": ["c1"]}
    assert sents[1]["claims"] == ["c2", "c1"]
    assert sents[2]["claims"] == [] and sents[3]["claims"] == [], "unknown keys are not grounding"


def test_sentences_are_kept_only_when_the_judge_finds_them_in_their_claims():
    body = "## Basics\nWarmup avoids early loss spikes [c1].\nWarmup doubles accuracy [c1].\nUnkeyed filler sentence."
    m = llm(("SENTENCES", "[]"), ("CLAIM: Warmup avoids", "supports"), ("CLAIM: Warmup doubles", "unrelated"),
            ("CLAIMS:\n[c1]", body))
    out = run(LE.compose(m, CONCEPT, CLAIMS, [], []))
    texts = [s["text"] for sec in out["sections"] for s in sec["sentences"]]
    assert texts == ["Warmup avoids early loss spikes."]
    assert out["stats"] == {"written": 3, "kept_first_pass": 1, "repaired": 0, "dropped": 2, "grounded_pct": 33}


def test_a_failed_sentence_is_rewritten_once_and_kept_if_the_rewrite_holds():
    body = "## Basics\nWarmup doubles accuracy [c1]."
    fixed = json.dumps([{"n": 1, "text": "Warmup avoids early loss spikes [c1]."}])
    m = llm(("CLAIM: Warmup doubles", "unrelated"), ("CLAIM: Warmup avoids", "supports"),
            ("SENTENCES", fixed), ("CLAIMS:\n[c1]", body))
    out = run(LE.compose(m, CONCEPT, CLAIMS, [], []))
    [sent] = out["sections"][0]["sentences"]
    assert sent["text"] == "Warmup avoids early loss spikes." and sent["repaired"] is True
    assert out["stats"]["repaired"] == 1 and out["stats"]["dropped"] == 0 and out["stats"]["grounded_pct"] == 0


def test_a_contradicted_sentence_is_dropped_without_a_repair_attempt():
    body = "## Basics\nWarmup never helps [c1]."
    m = llm(("CLAIM: Warmup never", "contradicts"), ("SENTENCES", '[{"n": 1, "text": "x [c1]."}]'), ("CLAIMS:\n[c1]", body))
    out = run(LE.compose(m, CONCEPT, CLAIMS, [], []))
    assert out["sections"] == [] and not any("SENTENCES" in c[1] for c in m.calls)


def test_a_rewrite_that_is_still_unsupported_is_dropped():
    body = "## B\nWarmup doubles accuracy [c1]."
    m = llm(("CLAIM: Warmup doubles", "unrelated"), ("CLAIM: Still wrong", "unrelated"),
            ("SENTENCES", json.dumps([{"n": 1, "text": "Still wrong [c1]."}])), ("CLAIMS:\n[c1]", body))
    assert run(LE.compose(m, CONCEPT, CLAIMS, [], []))["sections"] == []


def test_too_few_claims_abstains_instead_of_writing():
    m = llm()
    out = run(LE.compose(m, CONCEPT, CLAIMS[:1], [], []))
    assert out["insufficient"] and out["sections"] == [] and m.calls == []


def test_withdrawn_claims_are_not_taught_from():
    withdrawn = {**CLAIMS[0], "withdrawn": True}
    assert [c["key"] for c in LE.usable([withdrawn, CLAIMS[1]])] == ["c2"]


def test_prompt_carries_certainty_conditions_conflicts_and_prereqs():
    cl = [claim("c1", "A holds.", "contested", "1B"), claim("c2", "A fails.", "superseded")]
    conflicts = [{"a": "c1", "b": "c2", "relation": "contradict", "note": "opposite"}]
    m = llm(("CLAIMS:\n[c1]", "## X\nA holds [c1]."), ("CLAIM: A holds", "supports"))
    run(LE.compose(m, CONCEPT, cl, conflicts, ["Gradient descent"]))
    prompt = m.calls[0][1]
    assert "contested; 2024-01-01; conditions: 1B" in prompt and "[c1] vs [c2] (contradict): opposite" in prompt
    assert "Gradient descent" in prompt


def test_a_standard_lesson_caps_at_the_strongest_claims_not_everything_found():
    # A rich retrieval pass can produce far more claims than one flat lesson prompt should try
    # to teach from in one sitting (see MAX_LESSON_CLAIMS) -- more than the cap, all but a
    # handful "single-source" so sorting-strongest-first is what has to keep the established
    # ones in, not just whichever came first.
    many = [claim(f"c{i}", f"Finding number {i}.", "single-source") for i in range(1, 60)]
    strong = [claim("c60", "The one established finding.", "established")]
    m = llm(("CLAIMS:\n[c60]", "## X\nThe established finding [c60]."), ("CLAIM:", "supports"))
    run(LE.compose(m, CONCEPT, many + strong, [], []))
    prompt = m.calls[0][1]
    assert prompt.count("\n[c") <= LE.MAX_LESSON_CLAIMS
    assert "[c60]" in prompt, "the established claim outranks the single-source ones and survives the cap"
