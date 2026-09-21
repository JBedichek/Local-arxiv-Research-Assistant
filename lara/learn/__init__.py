"""Grounded, adaptive self-education: a curriculum built from the paper corpus, where the
model composes and the sources decide.

Every lesson sentence traces to a claim, every claim to a passage, and each link is
checked by an independent judge pass -- see `judge`. Stages, in the order a course runs:

    scope    goal -> competencies, clarifying questions only where the answer changes them
    graph    prerequisite graph of concepts, skeleton drawn from survey passages
    claims   per-concept claims with conditions, corroboration, conflicts, supersession
    lesson   prose written only from verified claims, each sentence re-checked
    quiz     items validated by an independent solver that sees only the source
    visuals  charts and diagrams whose every number and edge is checked against a claim
    learner  mastery, spaced repetition, diagnostic pretest, flags that force re-checks
"""
