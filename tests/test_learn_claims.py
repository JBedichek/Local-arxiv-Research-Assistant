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


def test_facets_are_parsed_and_capped():
    m = llm(("choosing what a learner needs evidence", json.dumps(["a", "b", "c", "d", "e", "f", "g", "h"])))
    assert run(CL.facets(m, concept())) == ["a", "b", "c", "d", "e", "f"]


def test_facets_falls_back_to_the_concept_title_on_a_bad_reply():
    assert run(CL.facets(llm(("choosing what a learner needs evidence", "not json")), concept())) == [concept()["title"]]


def test_build_researches_each_facet_and_merges_what_they_find():
    fc = FakeCorpus(by_query={"facet a": [passage(1, LONG, arxiv="2401.1")],
                              "facet b": [passage(2, LONG, arxiv="2402.2")]})

    def extract_by_facet(system, prompt):
        if "facet a" in prompt:
            return json.dumps([{"passage": 1, "claim": "Warmup gradually raises the learning rate."},
                               {"passage": 1, "claim": "A linear ramp is the most common warmup shape."}])
        if "facet b" in prompt:
            return json.dumps([{"passage": 1, "claim": "A cosine schedule is a less common warmup shape."},
                               {"passage": 1, "claim": "Warmup length often scales with batch size."}])
        return "[]"

    m = llm(("choosing what a learner needs evidence", json.dumps(["facet a", "facet b"])),
            ("extract atomic claims", extract_by_facet), ("strict fact-checker", "supports"),
            ("compare two claims", '{"relation": "unrelated", "note": ""}'))
    out = run(CL.build(m, fc, concept()))
    assert out["facets"] == ["facet a", "facet b"]
    assert {c["text"] for c in out["claims"]} == {
        "Warmup gradually raises the learning rate.", "A linear ramp is the most common warmup shape.",
        "A cosine schedule is a less common warmup shape.", "Warmup length often scales with batch size."}
    assert [c["key"] for c in out["claims"]] == ["c1", "c2", "c3", "c4"] and out["stats"]["facets_widened"] == 0


def test_a_thin_facet_is_retried_against_a_wider_search():
    """"thin facet" only has 2 passages allowed through in round one (MAX_PER_PAPER, all from
    the same paper); its one claim is below MIN_FACET_CLAIMS, so it is searched again excluding
    what every facet has used -- and the paper's third passage, held back the first time only by
    the per-call MAX_PER_PAPER count, comes through."""
    thin = [passage(1, LONG, arxiv="2401.1"), passage(2, LONG, arxiv="2401.1"), passage(3, LONG, arxiv="2401.1")]
    rich = [passage(9, LONG, arxiv="2402.1"), passage(10, LONG, arxiv="2402.2")]
    fc = FakeCorpus(by_query={"thin facet": thin, "rich facet": rich})

    seen_thin_calls = []

    def extract_by_facet(system, prompt):
        if "thin facet" in prompt:
            seen_thin_calls.append(prompt)
            text = "Thin facet claim one." if len(seen_thin_calls) == 1 else "Thin facet claim two, found widening."
            return json.dumps([{"passage": 1, "claim": text}])
        if "rich facet" in prompt:
            return json.dumps([{"passage": 1, "claim": "Rich facet claim one."},
                               {"passage": 2, "claim": "Rich facet claim two."}])
        return "[]"

    m = llm(("choosing what a learner needs evidence", json.dumps(["thin facet", "rich facet"])),
            ("extract atomic claims", extract_by_facet), ("strict fact-checker", "supports"),
            ("compare two claims", '{"relation": "unrelated", "note": ""}'))
    out = run(CL.build(m, fc, concept()))
    assert len(seen_thin_calls) == 2, "the thin facet was searched again"
    assert out["stats"]["facets_widened"] == 1
    texts = {c["text"] for c in out["claims"]}
    assert {"Thin facet claim one.", "Thin facet claim two, found widening.",
           "Rich facet claim one.", "Rich facet claim two."} == texts


def test_near_duplicate_claims_from_different_papers_across_facets_stay_separate_for_corroboration():
    """A `_merge_facets` regression: the same idea surfacing from two different papers, via two
    different facets, is corroboration -- it must not be merged away like a same-paper repeat."""
    fc = FakeCorpus(by_query={"facet a": [passage(1, LONG, arxiv="2401.1")],
                              "facet b": [passage(2, LONG, arxiv="2402.2")]})
    same_claim = json.dumps([{"passage": 1, "claim": "Warmup avoids early loss spikes."}])
    m = llm(("choosing what a learner needs evidence", json.dumps(["facet a", "facet b"])),
            ("extract atomic claims", same_claim), ("strict fact-checker", "supports"),
            ("compare two claims", '{"relation": "agree", "note": ""}'))
    out = run(CL.build(m, fc, concept()))
    assert len(out["claims"]) == 2 and out["stats"]["cross_facet_merged"] == 0
    assert out["claims"][0]["certainty"] == out["claims"][1]["certainty"] == "established"


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


# ── coverage-probe budget ────────────────────────────────────────────────────────

def test_coverage_tier_buckets_by_papers_count():
    assert CL.coverage_tier({"papers": 0}) == "thin"
    assert CL.coverage_tier({"papers": CL.COVERAGE_THIN - 1}) == "thin"
    assert CL.coverage_tier({"papers": CL.COVERAGE_THIN}) == "typical"
    assert CL.coverage_tier({"papers": CL.COVERAGE_RICH - 1}) == "typical"
    assert CL.coverage_tier({"papers": CL.COVERAGE_RICH}) == "rich"


def test_budget_researches_a_thin_corpus_less_and_a_rich_one_more():
    thin = CL.budget({"papers": 1})
    rich = CL.budget({"papers": 50})
    assert thin["tier"] == "thin" and thin["citation_walk"] is False
    assert rich["tier"] == "rich" and rich["citation_walk"] is True
    assert thin["facets"] < rich["facets"] and thin["per_query"] < rich["per_query"]


def test_facets_asks_for_no_more_than_the_budgeted_count():
    reply = json.dumps([f"facet {i}" for i in range(8)])
    m = llm(("choosing what a learner needs evidence", reply))
    assert run(CL.facets(m, concept(), max_facets=2)) == ["facet 0", "facet 1"]
    assert "FACETS: up to 2" in m.calls[0][1]


# ── citation-graph walk ──────────────────────────────────────────────────────────

def test_gather_citation_passages_restricts_search_to_the_given_papers():
    only_b = passage(2, LONG, arxiv="2402.2")
    fc = FakeCorpus(default=[passage(1, LONG, arxiv="2401.1")], by_paper=[only_b])
    out = run(CL.gather_citation_passages(fc, "warmup", ["2402.2"]))
    assert [p.arxiv_id for p in out] == ["2402.2"]
    assert fc.queries == ["warmup"]


def test_gather_citation_passages_with_no_papers_does_not_search_at_all():
    fc = FakeCorpus(default=[passage(1, LONG)])
    assert run(CL.gather_citation_passages(fc, "warmup", [])) == [] and fc.queries == []


def test_build_walks_citations_from_a_facets_top_result_when_the_budget_allows_it():
    top = passage(1, LONG, arxiv="2401.1")
    neighbour = passage(9, LONG, arxiv="2402.2")
    fc = FakeCorpus(default=[top], by_paper=[neighbour], coverage={"chunks": 50, "papers": 20},
                    neighbours={"2401.1": {"cites": ["2402.2"], "cited_by": []}})
    items = [{"passage": 1, "claim": "Warmup avoids early loss spikes."},
             {"passage": 2, "claim": "The cited paper reports the same effect."}]
    m = llm(("choosing what a learner needs evidence", json.dumps(["warmup basics"])),
            ("extract atomic claims", json.dumps(items)), ("strict fact-checker", "supports"),
            ("compare two claims", '{"relation": "agree", "note": ""}'))
    out = run(CL.build(m, fc, concept()))
    trace = out["trace"]
    assert trace["budget"]["tier"] == "rich" and trace["budget"]["citation_walk"] is True
    [round_] = [r for r in trace["rounds"] if r["round"] == 1]
    assert round_["citation_papers_tried"] == 1 and round_["citation_passages_kept"] == 1
    assert {c["passage"]["arxiv_id"] for c in out["claims"]} == {"2401.1", "2402.2"}


def test_build_skips_the_citation_walk_when_the_budget_is_thin():
    top = passage(1, LONG, arxiv="2401.1")
    calls = []
    fc = FakeCorpus(default=[top], coverage={"chunks": 2, "papers": 1},
                    neighbours={"2401.1": {"cites": ["2402.2"], "cited_by": []}})
    fc.neighbours = lambda a: calls.append(a) or {"cites": ["2402.2"], "cited_by": []}
    m = llm(("choosing what a learner needs evidence", json.dumps(["warmup basics"])),
            ("extract atomic claims", extract_reply([{"passage": 1, "claim": "Warmup avoids early loss spikes."}])[1]),
            ("strict fact-checker", "supports"))
    out = run(CL.build(m, fc, concept()))
    assert out["trace"]["budget"]["tier"] == "thin" and out["trace"]["budget"]["citation_walk"] is False
    assert calls == [], "a thin budget never even asks the corpus for citation neighbours"


def test_build_trace_reports_coverage_budget_and_per_facet_rounds():
    fc = FakeCorpus(default=[passage(1, LONG, arxiv="2401.1")], coverage={"chunks": 10, "papers": 5})
    m = llm(("choosing what a learner needs evidence", json.dumps(["warmup basics"])),
            ("extract atomic claims", extract_reply([{"passage": 1, "claim": "Warmup avoids early loss spikes."}])[1]),
            ("strict fact-checker", "supports"))
    out = run(CL.build(m, fc, concept()))
    trace = out["trace"]
    assert trace["coverage"] == {"chunks": 10, "papers": 5}
    assert trace["budget"]["tier"] == "typical"
    assert trace["rounds"][0]["facet"] == "warmup basics" and trace["rounds"][0]["claims"] == 1
    assert trace["claims"] == 1 and trace["papers"] == 1 and "ms" in trace


# ── decision loop / named gap ────────────────────────────────────────────────────

def test_facet_gap_returns_the_facet_itself_when_claims_are_still_thin():
    claims = [CL.Claim("c1", "Warmup avoids early loss spikes.", passage(1, LONG).to_dict())]
    m = llm(("remains worth one more search", "should not be called"))
    assert run(CL._facet_gap(m, "warmup basics", claims)) == "warmup basics"
    assert m.calls == [], "too few claims to reason about -- no call needed"


def test_facet_gap_names_a_specific_missing_angle():
    claims = [CL.Claim("c1", "Warmup ramps the rate up.", passage(1, LONG).to_dict()),
             CL.Claim("c2", "The ramp is usually linear.", passage(2, LONG).to_dict())]
    m = llm(("remains worth one more search", json.dumps({"gap": "warmup empirical results"})))
    assert run(CL._facet_gap(m, "warmup basics", claims)) == "warmup empirical results"
    assert "FACET: warmup basics" in m.calls[0][1] and "Warmup ramps the rate up." in m.calls[0][1]


def test_facet_gap_is_empty_when_the_model_finds_nothing_missing():
    claims = [CL.Claim("c1", "a", passage(1, LONG).to_dict()), CL.Claim("c2", "b", passage(2, LONG).to_dict())]
    assert run(CL._facet_gap(llm(("remains worth one more search", "null")), "warmup basics", claims)) == ""


def test_build_runs_a_gap_driven_third_round_when_the_richest_budget_names_one():
    p1, p2 = passage(1, LONG, arxiv="2401.1"), passage(2, LONG, arxiv="2402.2")
    p3 = passage(3, LONG, arxiv="2403.3")
    fc = FakeCorpus(by_query={"warmup basics": [p1, p2], "warmup numbers": [p3]},
                    coverage={"chunks": 50, "papers": 20})

    def extract_by_query(system, prompt):
        if "warmup numbers" in prompt:
            return json.dumps([{"passage": 1, "claim": "Third round claim about numbers."}])
        return json.dumps([{"passage": 1, "claim": "Warmup ramps the rate up."},
                           {"passage": 2, "claim": "The ramp is usually linear."}])

    m = llm(("choosing what a learner needs evidence", json.dumps(["warmup basics"])),
            ("extract atomic claims", extract_by_query), ("strict fact-checker", "supports"),
            ("compare two claims", '{"relation": "unrelated", "note": ""}'),
            ("remains worth one more search", json.dumps({"gap": "warmup numbers"})))
    out = run(CL.build(m, fc, concept()))
    trace = out["trace"]
    assert {r["round"] for r in trace["rounds"]} == {1, 3}, "round 2 never fires -- round 1 was not thin"
    third = next(r for r in trace["rounds"] if r["round"] == 3)
    assert third["query"] == "warmup numbers" and third["claims"] == 1
    assert out["stats"]["facets_gap_researched"] == 1
    assert {c["text"] for c in out["claims"]} >= {"Third round claim about numbers."}


def test_build_skips_the_gap_round_when_the_budget_is_not_rich():
    fc = FakeCorpus(default=[passage(1, LONG, arxiv="2401.1"), passage(2, LONG, arxiv="2402.2")],
                    coverage={"chunks": 8, "papers": 6})   # typical tier
    items = [{"passage": 1, "claim": "Warmup ramps the rate up."}, {"passage": 2, "claim": "The ramp is linear."}]
    m = llm(("choosing what a learner needs evidence", json.dumps(["warmup basics"])),
            ("extract atomic claims", json.dumps(items)), ("strict fact-checker", "supports"),
            ("compare two claims", '{"relation": "unrelated", "note": ""}'),
            ("remains worth one more search", json.dumps({"gap": "should not be reached"})))
    out = run(CL.build(m, fc, concept()))
    assert out["trace"]["budget"]["tier"] == "typical" and out["trace"]["budget"]["gap_round"] is False
    assert {r["round"] for r in out["trace"]["rounds"]} == {1}
    assert out["stats"]["facets_gap_researched"] == 0


# ── full-paper read ───────────────────────────────────────────────────────────────

def test_build_reads_a_dominant_papers_full_text_when_one_paper_anchors_a_round():
    dense_chunk = passage(1, LONG, arxiv="2401.1")
    full_chunks = [passage(1, LONG, arxiv="2401.1"), passage(2, LONG, arxiv="2401.1"),
                  passage(3, LONG, arxiv="2401.1")]
    fc = FakeCorpus(default=[dense_chunk], coverage={"chunks": 50, "papers": 20},
                    full_papers={"2401.1": full_chunks})
    items = [{"passage": 1, "claim": "First finding from the paper."},
             {"passage": 2, "claim": "Second finding from the paper."},
             {"passage": 3, "claim": "Third finding from the paper."}]
    m = llm(("choosing what a learner needs evidence", json.dumps(["warmup basics"])),
            ("extract atomic claims", json.dumps(items)), ("strict fact-checker", "supports"),
            ("remains worth one more search", "null"))
    out = run(CL.build(m, fc, concept()))
    [round1] = [r for r in out["trace"]["rounds"] if r["round"] == 1]
    assert round1["full_paper_read"] == "2401.1"
    assert len(out["claims"]) == 3, "all three of the paper's chunks were read, not just the one dense search found"


def test_build_does_not_read_the_full_paper_when_more_than_one_paper_was_found():
    p1, p2 = passage(1, LONG, arxiv="2401.1"), passage(2, LONG, arxiv="2402.2")
    fc = FakeCorpus(default=[p1, p2], coverage={"chunks": 50, "papers": 20},
                    full_papers={"2401.1": [p1, passage(9, LONG, arxiv="2401.1")]})
    items = [{"passage": 1, "claim": "Warmup avoids early loss spikes."},
             {"passage": 2, "claim": "A second, corroborating result."}]
    m = llm(("choosing what a learner needs evidence", json.dumps(["warmup basics"])),
            ("extract atomic claims", json.dumps(items)), ("strict fact-checker", "supports"),
            ("compare two claims", '{"relation": "unrelated", "note": ""}'),
            ("remains worth one more search", "null"))
    out = run(CL.build(m, fc, concept()))
    [round1] = [r for r in out["trace"]["rounds"] if r["round"] == 1]
    assert round1["full_paper_read"] == ""


def test_build_skips_the_full_paper_read_when_the_budget_is_not_rich():
    fc = FakeCorpus(default=[passage(1, LONG, arxiv="2401.1")], coverage={"chunks": 8, "papers": 6},
                    full_papers={"2401.1": [passage(1, LONG, arxiv="2401.1"), passage(2, LONG, arxiv="2401.1")]})
    m = llm(("choosing what a learner needs evidence", json.dumps(["warmup basics"])),
            ("extract atomic claims", extract_reply([{"passage": 1, "claim": "Warmup avoids early loss spikes."}])[1]),
            ("strict fact-checker", "supports"))
    out = run(CL.build(m, fc, concept()))
    assert out["trace"]["budget"]["tier"] == "typical" and out["trace"]["budget"]["full_paper"] is False
    [round1] = [r for r in out["trace"]["rounds"] if r["round"] == 1]
    assert round1["full_paper_read"] == ""


# ── live trace events ─────────────────────────────────────────────────────────────

def test_build_reports_a_start_event_then_one_round_event_per_round():
    fc = FakeCorpus(default=[passage(1, LONG, arxiv="2401.1")], coverage={"chunks": 10, "papers": 5})
    m = llm(("choosing what a learner needs evidence", json.dumps(["warmup basics"])),
            ("extract atomic claims", extract_reply([{"passage": 1, "claim": "Warmup avoids early loss spikes."}])[1]),
            ("strict fact-checker", "supports"))
    events = []

    async def on_event(name, payload):
        events.append((name, payload))

    out = run(CL.build(m, fc, concept(), on_event=on_event))
    assert events[0] == ("start", {"coverage": {"chunks": 10, "papers": 5},
                                   "budget": out["trace"]["budget"], "facets": ["warmup basics"]})
    rounds = [p for n, p in events if n == "round"]
    assert rounds == out["trace"]["rounds"], "every round reported live matches the final trace exactly"


def test_build_works_with_no_on_event_given():
    fc = FakeCorpus(default=[passage(1, LONG, arxiv="2401.1")], coverage={"chunks": 10, "papers": 5})
    m = llm(("choosing what a learner needs evidence", json.dumps(["warmup basics"])),
            ("extract atomic claims", extract_reply([{"passage": 1, "claim": "Warmup avoids early loss spikes."}])[1]),
            ("strict fact-checker", "supports"))
    assert run(CL.build(m, fc, concept()))["trace"]["rounds"]
