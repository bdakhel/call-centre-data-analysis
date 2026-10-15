"""Topic classifier label validity."""

import pytest

import topics


def test_all_predictions_are_valid_labels(topic_predictions):
    assert set(topic_predictions) <= set(topics.TOPICS), (
        f"unexpected labels: {set(topic_predictions) - set(topics.TOPICS)}"
    )


def test_one_prediction_per_transcript(topic_predictions, sample):
    assert len(topic_predictions) == len(sample)


def test_no_empty_predictions(topic_predictions):
    assert all(isinstance(p, str) and p.strip() for p in topic_predictions)


def test_unrecognised_text_falls_back_to_a_valid_label():
    """Garbage in must still produce a label in the schema, not None."""
    classifier = topics.RuleTopicClassifier()
    preds = classifier.predict(["", "   ", "zzzz qqqq", "...."], show_progress=False)
    assert all(p in topics.TOPICS for p in preds)
    assert all(p == topics.DEFAULT_TOPIC for p in preds), (
        "text with no topic signal should route to the catch-all"
    )


def test_topic_list_matches_the_generator():
    """The classifier's label set must match what the data actually contains."""
    import pandas as pd

    path = "data/raw/interactions.csv"
    actual = set(pd.read_csv(path, usecols=["topic"])["topic"])
    assert actual == set(topics.TOPICS), (
        f"classifier labels drifted from the data: "
        f"only in data {actual - set(topics.TOPICS)}, "
        f"only in classifier {set(topics.TOPICS) - actual}"
    )


def test_every_topic_has_rules():
    assert set(topics.RULES) == set(topics.TOPICS)
    for topic, patterns in topics.RULES.items():
        assert patterns, f"{topic} has no keyword rules"


def test_clear_cut_examples_classify_correctly():
    """Not an accuracy bar -- just proof the rule table is wired to the labels."""
    classifier = topics.RuleTopicClassifier()
    cases = {
        "My power has been out for three hours and the outage map says nothing.":
            "outage or service issue",
        "I'd like to submit my own meter reading, the dial shows 412 on the gas side.":
            "meter reading",
        "I got a disconnection notice and I can't pay it all at once.":
            "payment arrangement",
        "I'm moving out on the 14th and need the final read scheduled.":
            "move-in/move-out",
    }
    preds = classifier.predict(list(cases), show_progress=False)
    for (text, expected), got in zip(cases.items(), preds):
        assert got == expected, f"{text!r} -> {got}, expected {expected}"
