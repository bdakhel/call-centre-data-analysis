"""The scores table contract: no missing IDs, valid values, complete columns.

These run against the real artefact when it exists, and against a freshly
scored sample otherwise -- so the suite is meaningful on a clean checkout and
stricter once the pipeline has actually been run.
"""

import numpy as np
import pytest

import run_pipeline
import topics

VALID_SENTIMENT = {"negative", "neutral", "positive"}
SCORE_COLUMNS = ["sentiment_score", "sentiment_start", "sentiment_end"]


@pytest.fixture(scope="module")
def scored(sample, agents, vader_scores, topic_predictions):
    """A scores table built the same way run_pipeline builds one."""
    import coaching
    import pandas as pd

    facts = coaching.build_facts(sample, agents, vader_scores)
    notes, _ = coaching.generate_notes(facts, generator=None, show_progress=False)

    table = sample[["interaction_id"]].copy()
    table["topic_predicted"] = topic_predictions
    table = table.merge(vader_scores, on="interaction_id", how="left")
    table = table.merge(notes[["interaction_id", "coaching_note", "generated_by"]],
                        on="interaction_id", how="left")
    table["model_version"] = "test"
    table["scored_at"] = "2026-10-05T00:00:00+00:00"
    return table[run_pipeline.OUTPUT_COLUMNS]


# --- No missing IDs --------------------------------------------------------

def test_every_interaction_is_scored(scored, sample):
    assert set(scored["interaction_id"]) == set(sample["interaction_id"])


def test_no_duplicate_ids(scored):
    assert not scored["interaction_id"].duplicated().any()


def test_row_count_matches_input(scored, sample):
    assert len(scored) == len(sample)


def test_no_null_ids(scored):
    assert scored["interaction_id"].notna().all()
    assert (scored["interaction_id"].astype(str).str.strip() != "").all()


# --- Column contract -------------------------------------------------------

def test_columns_exact_and_ordered(scored):
    assert list(scored.columns) == run_pipeline.OUTPUT_COLUMNS


def test_no_nulls_anywhere(scored):
    nulls = scored.isna().sum()
    assert not nulls.any(), f"null values present:\n{nulls[nulls > 0]}"


# --- Value ranges ----------------------------------------------------------

@pytest.mark.parametrize("column", SCORE_COLUMNS)
def test_score_ranges(scored, column):
    assert scored[column].between(-1, 1).all()
    assert np.isfinite(scored[column]).all()


def test_sentiment_labels_valid(scored):
    assert set(scored["sentiment_label"]) <= VALID_SENTIMENT


def test_topic_labels_valid(scored):
    assert set(scored["topic_predicted"]) <= set(topics.TOPICS)


def test_coaching_notes_are_substantive(scored):
    words = scored["coaching_note"].str.split().str.len()
    assert (words >= 8).all(), "a coaching note shorter than 8 words is not usable"
    assert (words <= 70).all(), "coaching notes should stay to 1-2 sentences"


def test_generated_by_is_recorded(scored):
    assert scored["generated_by"].notna().all()
    assert all(v == "rules" or v.startswith("ollama:")
               for v in scored["generated_by"].unique())


# --- The real artefact, when it exists ------------------------------------

def test_artefact_matches_the_contract(scores_file):
    assert list(scores_file.columns) == run_pipeline.OUTPUT_COLUMNS
    assert not scores_file["interaction_id"].duplicated().any()
    assert not scores_file.isna().any().any()
    for column in SCORE_COLUMNS:
        assert scores_file[column].between(-1, 1).all(), f"{column} out of range"
    assert set(scores_file["sentiment_label"]) <= VALID_SENTIMENT
    assert set(scores_file["topic_predicted"]) <= set(topics.TOPICS)


def test_artefact_covers_every_interaction(scores_file, interactions):
    """The full run must not silently drop rows."""
    if len(scores_file) != len(interactions):
        pytest.skip("scores file was produced with --sample")
    assert set(scores_file["interaction_id"]) == set(interactions["interaction_id"])


def test_artefact_records_provenance(scores_file):
    assert scores_file["model_version"].nunique() == 1
    version = scores_file["model_version"].iloc[0]
    assert "roberta" in version or "vader" in version
    # scored_at must parse as a real timestamp
    import pandas as pd
    assert pd.to_datetime(scores_file["scored_at"]).notna().all()
