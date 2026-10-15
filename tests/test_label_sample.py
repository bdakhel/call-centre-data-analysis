"""The labelling sample must be blind, complete and representative.

The anti-bias property is the one that actually matters: if an anchoring
column ever leaks into to_label.csv, the human labels silently stop being an
independent yardstick and every agreement number computed from them is
inflated. That failure is invisible by inspection, so it gets a test.
"""

import pandas as pd
import pytest

import make_label_sample as mls


@pytest.fixture(scope="module")
def to_label():
    path = mls.OUTPUT_PATH
    if not path.exists():
        pytest.skip("to_label.csv missing -- run src/make_label_sample.py")
    return pd.read_csv(path, keep_default_na=False)


@pytest.fixture(scope="module")
def manifest():
    if not mls.MANIFEST_PATH.exists():
        pytest.skip("manifest missing -- run src/make_label_sample.py")
    return pd.read_csv(mls.MANIFEST_PATH)


# --- Blindness -------------------------------------------------------------

@pytest.mark.parametrize("column", mls.FORBIDDEN)
def test_anchoring_columns_are_absent(to_label, column):
    assert column not in to_label.columns, (
        f"'{column}' leaked into the labelling file -- labels would be anchored"
    )


def test_columns_are_exactly_the_contract(to_label):
    assert list(to_label.columns) == mls.LABEL_COLUMNS


def test_no_sentiment_words_hidden_in_the_columns(to_label):
    """Guards against a future column like 'predicted_sentiment_hint'."""
    suspicious = [c for c in to_label.columns
                  if any(w in c.lower() for w in ("sentiment", "predict", "label_hint", "score"))
                  and c != "human_label"]
    assert not suspicious, f"suspicious columns: {suspicious}"


# --- Shape -----------------------------------------------------------------

def test_sample_size(to_label):
    assert len(to_label) == mls.DEFAULT_N


def test_human_label_starts_empty(to_label):
    assert (to_label["human_label"].astype(str).str.strip() == "").all(), (
        "human_label must ship empty -- a pre-filled value is an anchor"
    )


def test_ids_are_unique_and_real(to_label, interactions):
    assert not to_label["interaction_id"].duplicated().any()
    assert set(to_label["interaction_id"]) <= set(interactions["interaction_id"])


def test_transcripts_are_present(to_label):
    assert to_label["transcript_text"].str.len().gt(100).all()


# --- Stratification --------------------------------------------------------

def test_every_stratum_is_represented(manifest, interactions):
    population = set(map(tuple, interactions[["topic", "channel"]].drop_duplicates().values))
    sampled = set(map(tuple, manifest[["topic", "channel"]].drop_duplicates().values))
    assert sampled == population, f"missing strata: {population - sampled}"


def test_sample_mirrors_the_population(manifest, interactions):
    """Proportional allocation: no cell should drift far from its real share."""
    pop = interactions.groupby(["topic", "channel"]).size() / len(interactions) * 100
    got = manifest.groupby(["topic", "channel"]).size() / len(manifest) * 100
    drift = (got - pop).abs().max()
    assert drift <= 2.0, f"largest stratum drift {drift:.1f}pp exceeds 2pp"


def test_manifest_joins_back_to_the_sample(manifest, to_label):
    assert set(manifest["interaction_id"]) == set(to_label["interaction_id"])


def test_rows_are_shuffled_not_grouped_by_stratum(manifest):
    """Blocked ordering creates run-length effects in human labelling."""
    topics_in_order = manifest["topic"].tolist()
    runs = sum(1 for a, b in zip(topics_in_order, topics_in_order[1:]) if a != b)
    # Sorted by topic, a 100-row 7-topic file would have only 6 transitions.
    assert runs > 50, f"only {runs} topic transitions -- file looks grouped, not shuffled"


# --- Allocation maths ------------------------------------------------------

def test_allocation_sums_exactly_to_n():
    counts = pd.Series([100, 50, 33, 17, 3], index=list("abcde"))
    for n in (10, 37, 100, 250):
        assert mls.allocate(counts, n).sum() == n


def test_allocation_is_proportional():
    counts = pd.Series([800, 200], index=["big", "small"])
    quota = mls.allocate(counts, 100)
    assert quota["big"] == 80 and quota["small"] == 20
