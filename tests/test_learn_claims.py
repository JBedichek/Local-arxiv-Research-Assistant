import asyncio
import json

from lara.learn import claims as CL
from learn_helpers import FakeCorpus, llm, passage

LONG = "x " * 120


def run(c):
    return asyncio.run(c)


def concept():
    return {"id": "c1", "title": "learning rate warmup", "summary": "ramping the rate"}


def extract_reply(items):
    return ("PASSAGES", json.dumps(items))


def test_extraction_keeps_only_claims_the_judge_finds_in_their_passage():
    ps = [passage(1, "warmup avoids early loss spikes " + LONG), passage(2, "batch size scaling " + LONG, arxiv="2401.00002")]
    items = [{"passage": 1, "claim": "Warmup avoids early loss spikes.", "conditions": "1B", "kind": "finding"},
             {"passage": 2, "claim": "Warmup doubles final accuracy.", "conditions": "", "kind": "finding"}]
    m = llm(("CLAIM: Warmup avoids", "supports"), ("CLAIM: Warmup doubles", "unrelated"), extract_reply(items))
    claims, dropped, _, _ = run(CL.extract(m, concept(), ps))
    assert [c.text for c in claims] == ["Warmup avoids early loss spikes."] and dropped == 1
    assert claims[0].key == "c1" and claims[0].conditions == "1B"
    assert claims[0].passage["arxiv_id"] == "2401.00001"


def test_extraction_ignores_bad_passage_indexes_and_near_duplicate_claims():
    ps = [passage(1, LONG)]
    items = [{"passage": 9, "claim": "out of range"}, {"passage": "x", "claim": "bad"},
             {"passage": 1, "claim": "Warmup avoids early loss spikes in training."},
             {"passage": 1, "claim": "Warmup avoids early loss spikes in training runs."}]
    claims, _, _, _ = run(CL.extract(llm(("CLAIM", "supports"), extract_reply(items)), concept(), ps))
    assert len(claims) == 1


def test_a_hypothesis_is_labelled_speculative():
    items = [{"passage": 1, "claim": "Maybe warmup helps.", "kind": "hypothesis"}]
    claims, _, _, _ = run(CL.extract(llm(("CLAIM", "supports"), extract_reply(items)), concept(), [passage(1, LONG)]))
    assert claims[0].certainty == "speculative"


def _two(a_date="2023-01-01", b_date="2024-06-01"):
    a = CL.Claim("c1", "Warmup is unnecessary for large batch training.", passage(1, "pa", arxiv="2301.1", date=a_date).to_dict())
    b = CL.Claim("c2", "Warmup is necessary for large batch training.", passage(2, "pb", arxiv="2406.1", date=b_date).to_dict())
    return a, b


def test_agreement_corroborates_and_makes_a_claim_established():
    a, b = _two()
    n = run(CL.relate(llm(("CLAIM A", '{"relation": "agree", "note": ""}')), [a, b]))
    assert n == 1 and a.corroborated_by == ["c2"] and a.certainty == "established"


def test_opposite_conclusions_conflict_and_the_newer_supersedes():
    a, b = _two()
    run(CL.relate(llm(("CLAIM A", '{"relation": "contradict", "note": "opposite"}')), [a, b]))
    assert a.superseded_by == "c2" and a.certainty == "superseded"
    assert b.certainty == "contested" and b.superseded_by == ""
    [conflict] = CL.conflicts([a, b])
    assert conflict["relation"] == "contradict" and len(conflict["sides"]) == 2


def test_close_dates_conflict_without_superseding():
    a, b = _two("2024-05-01", "2024-06-01")
    run(CL.relate(llm(("CLAIM A", '{"relation": "contradict", "note": ""}')), [a, b]))
    assert a.superseded_by == "" and a.certainty == b.certainty == "contested"


def test_scope_differences_are_recorded_but_do_not_make_a_claim_contested():
    a, b = _two()
    run(CL.relate(llm(("CLAIM A", '{"relation": "scope", "note": "1B vs 70B"}')), [a, b]))
    assert a.certainty == "single-source" and a.conflicts[0]["note"] == "1B vs 70B"
    assert CL.conflicts([a, b])[0]["relation"] == "scope"


def test_claims_from_one_paper_or_unrelated_topics_are_not_compared():
    a, b = _two()
    b.passage["arxiv_id"] = a.passage["arxiv_id"]
    assert run(CL.relate(llm(), [a, b])) == 0
    a, b = _two()
    b.text = "Tokenizer vocabularies affect multilingual fertility."
    assert run(CL.relate(llm(), [a, b])) == 0


def test_an_embedder_can_surface_a_pair_word_overlap_misses():
    a, b = _two()
    b.text = "Ramping the step size early is essential when batches are huge."
    same = lambda t: [1.0, 0.0]
    assert run(CL.relate(llm(("CLAIM A", '{"relation": "agree", "note": ""}')), [a, b], embed=same)) == 1


def test_gather_takes_at_most_two_substantial_passages_per_paper():
    ps = [passage(i, LONG, arxiv="2401.1") for i in range(1, 5)] + [passage(9, "short", arxiv="2401.2")]
    got = run(CL.gather_passages(FakeCorpus(default=ps), concept()))
    assert len(got) == 2 and all(p.arxiv_id == "2401.1" for p in got)


def test_build_returns_serialisable_claims_conflicts_and_stats():
    ps = [passage(1, "a " + LONG, arxiv="2301.1", date="2023-01-01"), passage(2, "b " + LONG, arxiv="2406.1", date="2024-06-01")]
    items = [{"passage": 1, "claim": "Warmup is unnecessary for large batch training."},
             {"passage": 2, "claim": "Warmup is necessary for large batch training."}]
    m = llm(("CLAIM A", '{"relation": "contradict", "note": "n"}'), ("CLAIM:", "supports"), extract_reply(items))
    out = run(CL.build(m, FakeCorpus(default=ps), concept()))
    json.dumps(out)
    assert len(out["claims"]) == 2 and len(out["conflicts"]) == 1
    assert out["claims"][0]["certainty"] == "superseded" and out["stats"]["comparisons"] == 1


def test_the_same_claim_from_two_papers_is_kept_for_corroboration_but_not_twice_from_one():
    ps = [passage(1, LONG, arxiv="2401.1"), passage(2, LONG, arxiv="2402.2")]
    items = [{"passage": 1, "claim": "Warmup avoids early loss spikes in training."},
             {"passage": 1, "claim": "Warmup avoids early loss spikes in training runs."},
             {"passage": 2, "claim": "Warmup avoids early loss spikes in training."}]
    claims, _, _, _ = run(CL.extract(llm(("CLAIM", "supports"), extract_reply(items)), concept(), ps))
    assert [c.paper for c in claims] == ["2401.1", "2402.2"]


def test_a_paper_stating_one_idea_twice_in_different_words_is_merged_by_the_judge():
    ps = [passage(1, LONG, arxiv="2401.1"), passage(2, LONG, arxiv="2401.1")]
    items = [{"passage": 1, "claim": "Tokenizer choice matters little in English but a lot in multilingual settings."},
             {"passage": 2, "claim": "Multilingual settings are far more sensitive to tokenizer choice than English ones."},
             {"passage": 2, "claim": "Tokenizer vocabularies above 250k rarely pay off in English or multilingual settings."}]
    same = lambda system, prompt: "same" if "matters little" in prompt and "far more sensitive" in prompt else "different"
    m = llm(("Do these two claims", same), ("CLAIM:", "supports"), extract_reply(items))
    claims, dropped, merged, _ = run(CL.extract(m, concept(), ps))
    assert [c.key for c in claims] == ["c1", "c2"] and merged == 1 and dropped == 0
    assert "far more sensitive" not in " ".join(c.text for c in claims)


def test_claims_with_nothing_in_common_are_never_put_to_the_judge():
    ps = [passage(1, LONG, arxiv="2401.1")]
    items = [{"passage": 1, "claim": "Warmup avoids loss spikes."}, {"passage": 1, "claim": "Tokenizers segment text into subwords."}]
    m = llm(("CLAIM:", "supports"), extract_reply(items))
    run(CL.extract(m, concept(), ps))
    assert not any("Do these two claims" in s for s, _ in m.calls)


def test_an_embedder_can_surface_a_same_paper_repeat_that_shares_no_words():
    ps = [passage(1, LONG, arxiv="2401.1")]
    items = [{"passage": 1, "claim": "Bigger input vocabularies help at every scale."},
             {"passage": 1, "claim": "Expanding the embedding table always improves results."}]
    m = llm(("Do these two claims", "same"), ("CLAIM:", "supports"), extract_reply(items))
    claims, _, merged, _ = run(CL.extract(m, concept(), ps, embed=lambda t: [1.0, 0.0]))
    assert merged == 1 and len(claims) == 1


def test_extraction_and_retrieval_are_told_the_learners_goal():
    c = dict(concept(), goal="pretrain an LLM from scratch")
    ps = [passage(1, LONG)]
    m = llm(("CLAIM:", "supports"), extract_reply([]))
    run(CL.extract(m, c, ps))
    assert "LEARNER'S GOAL: pretrain an LLM from scratch" in m.calls[0][1]
    assert "unrelated to it" in m.calls[0][0]
    corpus = FakeCorpus(default=[passage(1, LONG)])
    run(CL.gather_passages(corpus, c))
    assert any("pretrain an LLM from scratch" in q for q in corpus.queries)


def test_a_true_claim_that_does_not_serve_the_learners_goal_is_dropped():
    c = dict(concept(), goal="pretrain an LLM from scratch")
    ps = [passage(1, LONG, arxiv="2401.1"), passage(2, LONG, arxiv="2402.2")]
    items = [{"passage": 1, "claim": "Cosine schedules decay the rate smoothly."},
             {"passage": 2, "claim": "Shallow networks reach near-optimal sample complexity."}]
    m = llm(("Would this claim help", lambda s, p: "no" if "Shallow networks" in p else "yes"),
            ("CLAIM:", "supports"), extract_reply(items))
    claims, _, _, off_topic = run(CL.extract(m, c, ps))
    assert [x.text for x in claims] == ["Cosine schedules decay the rate smoothly."] and off_topic == 1


def test_without_a_goal_nothing_is_dropped_as_off_topic():
    ps = [passage(1, LONG)]
    m = llm(("CLAIM:", "supports"), extract_reply([{"passage": 1, "claim": "Anything."}]))
    claims, _, _, off_topic = run(CL.extract(m, concept(), ps))
    assert len(claims) == 1 and off_topic == 0 and not any("Would this claim help" in s for s, _ in m.calls)
