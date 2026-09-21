import asyncio
import json

from lara.learn import quiz as Q
from learn_helpers import llm


def run(c):
    return asyncio.run(c)


def claim(key, text, ptext="passage text"):
    return {"key": key, "text": text, "certainty": "established",
            "passage": {"text": ptext, "arxiv_id": "2401.1", "title": "P", "chunk_id": 7}}


CLAIMS = [claim("c1", "Warmup avoids early loss spikes.", "Warmup avoids early loss spikes. It ran 1000 steps."),
          claim("c2", "Peak LR was 3e-4.", "Peak LR was 3e-4 for the 1B model.")]
CONCEPT = {"id": "c1", "title": "warmup"}


def items_reply(items):
    return ("CLAIMS:\n[c1]", json.dumps(items))


MCQ = {"type": "mcq", "question": "What does warmup avoid?", "choices": ["spikes", "a", "b", "c"],
       "answer": "A", "claim": "c1", "explanation": "Early spikes."}
SHORT = {"type": "short", "question": "How many warmup steps ran?", "answer": "1000", "claim": "c1", "explanation": "e"}


def test_generation_rejects_malformed_and_duplicate_items():
    bad = [{"type": "mcq", "question": "q", "choices": ["only", "two"], "answer": "A", "claim": "c1"},
           {"type": "essay", "question": "q", "answer": "a", "claim": "c1"},
           {"type": "short", "question": "q", "answer": "a", "claim": "zz"}, dict(MCQ), dict(MCQ)]
    got = run(Q.generate(llm(items_reply(bad)), CONCEPT, CLAIMS))
    assert len(got) == 1 and got[0]["id"] == "c1-q1" and got[0]["answer"] == "A"
    assert got[0]["source"]["chunk_id"] == 7


def test_the_solver_sees_the_passage_and_question_but_not_the_answer():
    m = llm(("PASSAGE:", "A"), items_reply([MCQ]))
    items = run(Q.generate(m, CONCEPT, CLAIMS))
    kept, dropped = run(Q.validate(m, items, CLAIMS))
    solve_prompt = [p for s, p in m.calls if "ONLY the passage" in s][0]
    assert "Warmup avoids early loss spikes." in solve_prompt and "A. spikes" in solve_prompt
    assert "Early spikes." not in solve_prompt and kept and dropped == 0


def test_an_item_the_solver_gets_wrong_or_cannot_answer_is_dropped():
    items = run(Q.generate(llm(items_reply([MCQ, SHORT])), CONCEPT, CLAIMS))
    m = llm(("Which", "B"), ("QUESTION: What does", "B"), ("QUESTION: How many", "CANNOT DETERMINE"))
    kept, dropped = run(Q.validate(m, items, CLAIMS))
    assert kept == [] and dropped == 2


def test_a_short_answer_survives_when_the_judge_agrees_with_the_solver():
    items = run(Q.generate(llm(items_reply([SHORT])), CONCEPT, CLAIMS))
    m = llm(("QUESTION: How many", "about 1000 steps"), ("The answer is:", "supports"))
    kept, _ = run(Q.validate(m, items, CLAIMS))
    assert kept[0]["validated"] is True


def test_grading_a_multiple_choice_by_letter_or_choice_text():
    item = {"type": "mcq", "question": "q", "choices": ["spikes", "a", "b", "c"], "answer": "A", "explanation": "e"}
    assert run(Q.grade(llm(), item, "a")) ["correct"] and run(Q.grade(llm(), item, "(A)"))["correct"]
    assert run(Q.grade(llm(), item, "spikes"))["correct"]
    assert not run(Q.grade(llm(), item, "B"))["correct"]
    assert run(Q.grade(llm(), item, "B"))["answer"] == "A. spikes"


def test_predict_items_accept_a_number_within_tolerance_without_the_judge():
    item = {"type": "predict", "question": "peak lr?", "choices": [], "answer": "3e-4 ... 300", "explanation": ""}
    m = llm()
    assert run(Q.grade(m, {**item, "answer": "300"}, "about 320"))["correct"] and m.calls == []
    assert not run(Q.grade(llm(("The answer is:", "unrelated")), {**item, "answer": "300"}, "5000"))["correct"]


def test_short_answers_are_graded_by_the_judge_and_blank_is_wrong():
    item = {"type": "short", "question": "q", "choices": [], "answer": "spikes", "explanation": "e"}
    assert run(Q.grade(llm(("The answer is:", "supports")), item, "loss spikes"))["correct"]
    m = llm(("The answer is:", "supports"))
    assert not run(Q.grade(m, item, "  "))["correct"] and m.calls == []


def test_critique_keeps_only_points_the_judge_confirms():
    points = [{"statement": "I will skip warmup", "verdict": "contradicted", "claims": ["c1"], "advice": "Use it."},
              {"statement": "LR 3e-4", "verdict": "supported", "claims": ["c2"], "advice": "ok"},
              {"statement": "made up", "verdict": "supported", "claims": ["c1"], "advice": "x"},
              {"statement": "y", "verdict": "opinion", "claims": ["c1"]}, {"statement": "z", "verdict": "supported", "claims": ["nope"]}]
    m = llm(("STATEMENT", ""), ("CLAIM: I will skip", "contradicts"), ("CLAIM: LR 3e-4", "supports"),
            ("CLAIM: made up", "unrelated"), ("LEARNER:", json.dumps(points)))
    got = run(Q.critique(m, CLAIMS, "my plan"))
    assert [p["statement"] for p in got] == ["I will skip warmup", "LR 3e-4"]
    assert run(Q.critique(llm(), CLAIMS, "  ")) == []


def test_the_generation_prompt_forbids_meta_references_in_questions():
    assert "never mention" in Q.QUIZ_SYSTEM and "claim keys" in Q.QUIZ_SYSTEM


def test_letter_prefixes_the_model_puts_on_choices_are_stripped():
    item = dict(MCQ, choices=["A. spikes", "(B) x", "C) y", "d: z"])
    got = run(Q.generate(llm(items_reply([item])), CONCEPT, CLAIMS))
    assert got[0]["choices"] == ["spikes", "x", "y", "z"]
