"""The pandas measure library must agree with powerbi/measures.dax.

The Streamlit report and the Power BI report show the same numbers to the same
people. If these two implementations drift, one of them is lying and there is
no way to tell which from the screen. These tests pin the semantics that are
easy to get wrong — the ones where a plausible-looking shortcut gives a
different answer.
"""

import numpy as np
import pandas as pd
import pytest

import measures as M

PROCESSED = "data/processed"


@pytest.fixture(scope="module")
def df():
    import os
    need = ["fact_interactions", "dim_agent", "dim_team", "dim_topic", "dim_channel"]
    if any(not os.path.exists(f"{PROCESSED}/{t}.csv") for t in need):
        pytest.skip("model tables missing -- run src/build_model_tables.py")
    fact = pd.read_csv(f"{PROCESSED}/fact_interactions.csv", parse_dates=["start_datetime"])
    agent = pd.read_csv(f"{PROCESSED}/dim_agent.csv")
    team = pd.read_csv(f"{PROCESSED}/dim_team.csv")
    topic = pd.read_csv(f"{PROCESSED}/dim_topic.csv")
    channel = pd.read_csv(f"{PROCESSED}/dim_channel.csv")
    out = (fact
           .merge(agent[["agent_key", "agent_name", "team_key", "csat_target",
                         "aht_target_sec", "tenure_band"]], on="agent_key")
           .merge(team[["team_key", "team_name"]], on="team_key")
           .merge(topic[["topic_key", "topic_name"]], on="topic_key")
           .merge(channel[["channel_key", "channel_name"]], on="channel_key"))
    out["month"] = out["start_datetime"].dt.to_period("M").astype(str)
    return out


@pytest.fixture(scope="module")
def sqm():
    import os
    if not os.path.exists(f"{PROCESSED}/fact_sqm_monthly.csv"):
        pytest.skip("sqm missing")
    return pd.read_csv(f"{PROCESSED}/fact_sqm_monthly.csv")


# --- The semantics that are easy to get wrong -----------------------------

def test_aht_vs_target_sums_per_row_targets(df):
    """Averaging dim_agent's targets weights a 10-contact agent like a 170-one."""
    correct = M.aht_vs_target_pct(df)
    naive = df["handle_time_sec"].mean() / df["aht_target_sec"].unique().mean() - 1
    assert not np.isclose(correct, naive, atol=1e-4), \
        "per-row and roster-average targets agree here, so this test proves nothing"
    expected = (df["handle_time_sec"].sum() - df["aht_target_sec"].sum()) / df["aht_target_sec"].sum()
    assert np.isclose(correct, expected)


def test_repeat_rate_denominator_is_cases_not_contacts(df):
    assert np.isclose(M.repeat_contact_rate_pct(df),
                      df["is_repeat_contact_7d"].sum() / df["has_case"].sum())
    assert M.repeat_contact_rate_pct(df) > df["is_repeat_contact_7d"].mean(), \
        "using contacts as the denominator understates the rate"


def test_csat_denominator_is_respondents_not_contacts(df):
    assert np.isclose(M.csat_pct(df),
                      df["is_csat_promoter"].sum() / df["has_survey_response"].sum())
    # Over all contacts it would be ~5%, which would be badly wrong.
    assert M.csat_pct(df) > df["is_csat_promoter"].mean() * 5


def test_csat_target_is_weighted_by_response_volume(df):
    responders = df[df["has_survey_response"]]
    assert np.isclose(M.csat_target_pct(df),
                      responders["csat_target"].sum() / len(responders))


def test_fcr_is_an_unweighted_mean(sqm):
    """Documented as such — the source carries no counts to weight by."""
    assert np.isclose(M.fcr_pct(sqm), sqm["fcr_pct"].mean())


def test_avg_days_to_close_excludes_open_cases(df):
    """Open cases have no closed date; counting them as zero would flatter it."""
    assert np.isclose(M.avg_days_to_close(df), df["days_to_close"].dropna().mean())
    assert df["days_to_close"].isna().sum() > 0, "no open cases, so this proves nothing"


# --- Ranges and blanks ----------------------------------------------------

@pytest.mark.parametrize("measure", [
    M.csat_pct, M.pct_negative, M.pct_positive, M.repeat_contact_rate_pct,
    M.survey_response_rate_pct, M.hold_rate_pct, M.escalation_rate_pct,
    M.pct_within_aht_target, M.pct_negative_calibrated,
])
def test_rates_are_between_zero_and_one(df, measure):
    value = measure(df)
    assert 0 <= value <= 1, f"{measure.__name__} returned {value}"


@pytest.mark.parametrize("measure", [
    M.interactions, M.cases, M.survey_responses, M.aht_minutes, M.csat_pct,
    M.pct_negative, M.net_sentiment, M.repeat_contact_rate_pct, M.sentiment_shift,
    M.avg_sentiment_score, M.aht_vs_target_pct, M.customers,
])
def test_measures_survive_an_empty_frame(df, measure):
    """An empty filter selection must blank, never raise."""
    result = measure(df.iloc[0:0])
    assert result == 0 or pd.isna(result), f"{measure.__name__} returned {result}"


def test_net_sentiment_is_on_a_hundred_point_scale(df):
    assert -100 <= M.net_sentiment(df) <= 100
    assert np.isclose(M.net_sentiment(df), (M.pct_positive(df) - M.pct_negative(df)) * 100)


def test_calibrated_differs_from_uncalibrated(df):
    """If these agreed, the whole validation finding would be moot."""
    assert not np.isclose(M.pct_negative(df), M.pct_negative_calibrated(df), atol=0.02)
    assert M.pct_negative_calibrated(df) < M.pct_negative(df), \
        "the uncalibrated label is the pessimistic one"


# --- Coaching queue -------------------------------------------------------

def test_coaching_priority_is_bounded(df):
    score = M.coaching_priority(df)
    assert score.min() >= 0 and score.max() <= 100


def test_coaching_queue_returns_top_n_ranked(df):
    queue = M.coaching_queue(df, 10)
    assert len(queue) == 10
    assert list(queue["rank"]) == list(range(1, 11))
    assert queue["coaching_priority"].is_monotonic_decreasing


def test_coaching_queue_reranks_within_the_filter(df):
    """ALLSELECTED semantics: a team's queue is its own, not the global top ten."""
    team = df["team_name"].iloc[0]
    scoped = df[df["team_name"] == team]
    assert set(M.coaching_queue(scoped, 10)["interaction_id"]) <= set(scoped["interaction_id"])


def test_coaching_queue_handles_fewer_rows_than_requested(df):
    assert len(M.coaching_queue(df.head(3), 10)) == 3


# --- Report furniture -----------------------------------------------------

def test_data_as_of_reads_the_data_not_the_clock(df):
    """A clock keeps advancing and hides a failed refresh."""
    stamp = M.data_as_of(df)
    assert stamp == df["start_datetime"].max()
    assert stamp < pd.Timestamp.now(), "stamp should be historical, not live"


def test_bands_match_the_validated_thresholds():
    assert (M.NEG_BAND, M.POS_BAND) == (-0.25, 0.90)
