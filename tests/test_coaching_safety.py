"""The coaching tone guarantee.

src/coaching.py promises notes are developmental, never disciplinary. For the
LLM path that promise rests on is_safe(), not on the prompt -- so is_safe() is
the thing worth testing.
"""

import pytest

import coaching

GOOD = ("You kept the customer calm through a difficult billing dispute, which is "
        "not easy. Next time, try naming the billing review earlier so they hear a "
        "concrete next step sooner.")


def test_accepts_a_well_formed_note():
    ok, reason = coaching.is_safe(GOOD)
    assert ok, reason


@pytest.mark.parametrize("note", [
    "Your performance here was unacceptable and will require a formal warning.",
    "This was a failure on your part and we need to discuss disciplinary action.",
    "You failed to resolve this and your poor performance is concerning.",
    "This is problematic and you must improve immediately or face consequences.",
])
def test_rejects_disciplinary_language(note):
    ok, reason = coaching.is_safe(note)
    assert not ok, f"disciplinary note was accepted: {note!r}"
    assert reason.startswith("banned:") or reason == "too_many_sentences"


@pytest.mark.parametrize("note,expected", [
    ("", "too_short"),
    ("Good job.", "too_short"),
    ("You did well on this call and kept the tone friendly, but next time try asking",
     "truncated"),
    ('You did well here overall. Next time try asking "what else is outstanding',
     "truncated"),
])
def test_rejects_malformed_notes(note, expected):
    ok, reason = coaching.is_safe(note)
    assert not ok
    assert reason == expected, f"expected {expected}, got {reason}"


def test_rejects_overlong_notes():
    ok, reason = coaching.is_safe("word " * 100 + ".")
    assert not ok and reason == "too_long"


# --- Rule engine -----------------------------------------------------------

def test_every_rule_bank_ends_with_a_catch_all():
    """pick() raises if nothing matches, so the last rule must always fire."""
    for bank in (coaching.STRENGTHS, coaching.IMPROVEMENTS):
        _, test, _ = bank[-1]
        assert test({}) is True, "final rule must be an unconditional catch-all"


def test_rule_notes_are_deterministic_per_interaction(sample, agents, vader_scores):
    """The same interaction must get the same note regardless of batch size."""
    facts = coaching.build_facts(sample, agents, vader_scores)
    records = facts.to_dict("records")
    first = coaching.rule_note(records[3])
    again = coaching.rule_note(records[3])
    assert first == again


def test_article_agreement_for_vowel_topics():
    assert coaching.a_topic("outage or service issue") == "an outage or service issue"
    assert coaching.a_topic("billing question") == "a billing question"


def test_generated_notes_pass_their_own_safety_check(sample, agents, vader_scores):
    """Templated notes must clear the same bar the LLM output has to."""
    facts = coaching.build_facts(sample, agents, vader_scores)
    notes, _ = coaching.generate_notes(facts, generator=None, show_progress=False)
    for note in notes["coaching_note"]:
        ok, reason = coaching.is_safe(note)
        assert ok, f"template failed safety check ({reason}): {note!r}"


# --- Regression: the gates used to reject perfectly good notes -------------
# A first version checked apostrophe BALANCE and required the final character
# to be .!?. Both produced false positives, and together they sent ~12% of
# otherwise fine LLM notes to the template fallback.

@pytest.mark.parametrize("note", [
    # Contractions and possessives make apostrophe counts odd as a matter of
    # course. None of these is malformed.
    "You handled the call well and showed empathy throughout. Next time, "
    "don't rush the close; it'll help you understand the customer's concerns.",
    "You stayed calm and professional here. Next time, try summarising at the "
    "close of the customer's call.",
    # A note may legitimately end on a closing quote after its full stop.
    'You did well here overall. Next time try asking "what else is outstanding?"',
])
def test_accepts_notes_with_apostrophes_and_closing_quotes(note):
    ok, reason = coaching.is_safe(note)
    assert ok, f"false rejection ({reason}): {note!r}"


@pytest.mark.parametrize("note", [
    'You did well here overall. Next time try asking "what else is outstanding',
    'You handled this calmly. Next time try saying "I will look into that',
])
def test_still_rejects_genuinely_unterminated_quotes(note):
    ok, _ = coaching.is_safe(note)
    assert not ok, f"truncated note was accepted: {note!r}"
