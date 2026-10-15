"""Scorer output ranges and contracts."""

import numpy as np
import pytest

import sentiment

SCORE_COLUMNS = ["sentiment_score", "sentiment_start", "sentiment_end"]
VALID_LABELS = {"negative", "neutral", "positive"}


# --- Output ranges ---------------------------------------------------------

@pytest.mark.parametrize("column", SCORE_COLUMNS)
def test_scores_within_range(vader_scores, column):
    """Every score is P(pos) - P(neg), so it must land inside [-1, 1]."""
    assert vader_scores[column].between(-1, 1).all(), (
        f"{column} out of range: min {vader_scores[column].min()}, "
        f"max {vader_scores[column].max()}"
    )


@pytest.mark.parametrize("column", SCORE_COLUMNS)
def test_scores_are_finite(vader_scores, column):
    assert np.isfinite(vader_scores[column]).all(), f"{column} contains NaN or inf"


def test_labels_are_valid(vader_scores):
    assert set(vader_scores["sentiment_label"]) <= VALID_LABELS


def test_one_row_per_interaction(vader_scores, sample):
    assert len(vader_scores) == len(sample)
    assert not vader_scores["interaction_id"].duplicated().any()


def test_no_nulls(vader_scores):
    assert not vader_scores.isna().any().any()


# --- Edge cases ------------------------------------------------------------

def test_empty_text_scores_neutral_zero():
    """An empty window must not be guessed at -- it has no evidence either way."""
    scorer = sentiment.VaderScorer(use_cache=False)
    assert scorer.score_texts(["", "   "]) == [("neutral", 0.0), ("neutral", 0.0)]


def test_obvious_polarity_has_the_right_sign():
    """A sanity check that the scorer is wired up at all, not a quality bar."""
    scorer = sentiment.VaderScorer(use_cache=False)
    (_, angry), (_, happy) = scorer.score_texts([
        "This is terrible, awful service and I am furious.",
        "Fantastic, wonderful help today. Thank you so much!",
    ])
    assert angry < 0 < happy


# --- Transcript parsing ----------------------------------------------------

def test_customer_turns_excludes_agent_lines(sample):
    transcript = sample["transcript_text"].iloc[0]
    turns = sentiment.customer_turns(transcript)
    assert turns, "no customer turns parsed"
    assert not any(t.startswith(("Agent:", "Customer:")) for t in turns)
    assert len(turns) < len([l for l in transcript.split("\n") if l.strip()])


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 9, 12])
def test_thirds_partitions_without_overlap(n):
    """start and end must not share a turn, except in the degenerate n=1 case."""
    turns = [f"t{i}" for i in range(n)]
    start, end = sentiment.thirds(turns)
    if n == 0:
        assert start == [] and end == []
        return
    assert start and end
    assert start[0] == turns[0] and end[-1] == turns[-1]
    if n > 1:
        assert not (set(start) & set(end)), f"overlap at n={n}"


def test_windows_cover_the_conversation(sample):
    for transcript in sample["transcript_text"].head(5):
        windows = sentiment.transcript_windows(transcript)
        assert set(windows) == {"full", "start", "end"}
        assert windows["start"] and windows["end"]
        assert windows["start"] in windows["full"]
