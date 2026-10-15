"""Star schema integrity: keys join, grains hold, types are right.

A broken dimensional model does not error — it quietly returns wrong numbers in
a report, which is the hardest kind of bug to notice. These assert the
properties a Power BI model depends on.
"""

import pandas as pd
import pytest

import build_model_tables as bmt

PROCESSED = bmt.PROCESSED
TABLES = ["dim_date", "dim_team", "dim_agent", "dim_topic", "dim_channel",
          "fact_interactions", "fact_sqm_monthly"]


@pytest.fixture(scope="module")
def model():
    missing = [t for t in TABLES if not (PROCESSED / f"{t}.csv").exists()]
    if missing:
        pytest.skip(f"missing {missing} -- run src/build_model_tables.py")
    return {t: pd.read_csv(PROCESSED / f"{t}.csv") for t in TABLES}


# --- Primary keys ---------------------------------------------------------

@pytest.mark.parametrize("table,key", [
    ("dim_date", "date_key"), ("dim_team", "team_key"), ("dim_agent", "agent_key"),
    ("dim_topic", "topic_key"), ("dim_channel", "channel_key"),
    ("fact_interactions", "interaction_key"), ("fact_sqm_monthly", "sqm_key"),
])
def test_primary_key_unique_and_populated(model, table, key):
    col = model[table][key]
    assert col.is_unique, f"{table}.{key} has duplicates"
    assert col.notna().all(), f"{table}.{key} has nulls"
    assert pd.api.types.is_integer_dtype(col), f"{table}.{key} must be an integer"


# --- Foreign keys ---------------------------------------------------------

@pytest.mark.parametrize("fact,fk,dim,pk", [
    ("fact_interactions", "date_key", "dim_date", "date_key"),
    ("fact_interactions", "agent_key", "dim_agent", "agent_key"),
    ("fact_interactions", "topic_key", "dim_topic", "topic_key"),
    ("fact_interactions", "channel_key", "dim_channel", "channel_key"),
    ("fact_interactions", "topic_predicted_key", "dim_topic", "topic_key"),
    ("fact_interactions", "case_created_date_key", "dim_date", "date_key"),
    ("fact_interactions", "case_closed_date_key", "dim_date", "date_key"),
    ("fact_sqm_monthly", "date_key", "dim_date", "date_key"),
    ("fact_sqm_monthly", "team_key", "dim_team", "team_key"),
    ("dim_agent", "team_key", "dim_team", "team_key"),
    ("dim_agent", "hire_date_key", "dim_date", "date_key"),
])
def test_foreign_key_resolves(model, fact, fk, dim, pk):
    orphans = set(model[fact][fk].dropna()) - set(model[dim][pk])
    assert not orphans, f"{fact}.{fk} -> {dim}.{pk}: {len(orphans)} orphans"


def test_active_fks_are_never_null(model):
    """Inactive-role FKs may be null; the active ones never may."""
    f = model["fact_interactions"]
    for col in ["date_key", "agent_key", "topic_key", "channel_key"]:
        assert f[col].notna().all(), f"{col} has nulls -- rows would vanish from visuals"


# --- Grain ----------------------------------------------------------------

def test_fact_interactions_grain_is_one_row_per_contact(model):
    f = model["fact_interactions"]
    assert f["interaction_id"].is_unique
    raw = pd.read_csv("data/raw/interactions.csv", usecols=["interaction_id"])
    assert set(f["interaction_id"]) == set(raw["interaction_id"])


def test_sqm_grain_is_month_by_team(model):
    s = model["fact_sqm_monthly"]
    assert not s.duplicated(["month_key", "team_key"]).any()


def test_dim_date_is_contiguous_and_whole_years(model):
    d = pd.to_datetime(model["dim_date"]["date"]).sort_values()
    assert (d.diff().dropna() == pd.Timedelta(days=1)).all(), "gap in the calendar"
    assert d.min().strftime("%m-%d") == "01-01"
    assert d.max().strftime("%m-%d") == "12-31"


def test_dim_date_covers_every_date_in_the_model(model):
    """Time intelligence silently blanks out if a fact date is missing."""
    keys = set(model["dim_date"]["date_key"])
    for table, col in [("fact_interactions", "date_key"),
                       ("fact_sqm_monthly", "date_key"),
                       ("dim_agent", "hire_date_key")]:
        assert set(model[table][col].dropna()) <= keys, f"{table}.{col} outside dim_date"


# --- Calendar correctness -------------------------------------------------

def test_business_day_excludes_weekends_and_holidays(model):
    d = model["dim_date"]
    assert not (d["is_business_day"] & d["is_weekend"]).any()
    assert not (d["is_business_day"] & d["is_holiday"]).any()
    assert (d["is_business_day"] | d["is_weekend"] | d["is_holiday"]).all()


def test_known_alberta_holidays_present(model):
    d = model["dim_date"].set_index("date")
    for day, name in [("2026-01-01", "New Year"), ("2026-02-16", "Family Day"),
                      ("2026-04-03", "Good Friday"), ("2026-05-18", "Victoria"),
                      ("2026-07-01", "Canada Day"), ("2026-09-07", "Labour Day"),
                      ("2026-10-12", "Thanksgiving"), ("2026-12-25", "Christmas")]:
        assert d.loc[day, "is_holiday"], f"{day} ({name}) not flagged"


def test_fiscal_year_starts_in_april(model):
    d = model["dim_date"].set_index("date")
    assert d.loc["2026-03-31", "fiscal_year"] == 2026
    assert d.loc["2026-04-01", "fiscal_year"] == 2027, "FY must roll on 1 April"
    assert d.loc["2026-04-01", "fiscal_quarter"] == 1
    assert d.loc["2026-04-01", "fiscal_month_number"] == 1
    assert d.loc["2026-03-31", "fiscal_month_number"] == 12


def test_monday_is_day_one(model):
    d = model["dim_date"]
    mondays = d[d["weekday_name"] == "Monday"]
    assert (mondays["day_of_week"] == 1).all()


# --- Measures and flags ---------------------------------------------------

def test_no_precomputed_ratios_in_fact_interactions(model):
    """A stored percentage cannot be re-aggregated; the components belong here."""
    bad = [c for c in model["fact_interactions"].columns
           if c.endswith(("_pct", "_percent", "_rate", "_ratio"))]
    assert not bad, f"pre-computed ratio columns: {bad}"


def test_durations_are_non_negative(model):
    f = model["fact_interactions"]
    assert (f["handle_time_sec"] >= 0).all()
    assert (f["hold_time_sec"] >= 0).all()
    assert (f["days_to_close"].dropna() >= 0).all()


def test_flags_are_boolean_not_text(model):
    f = model["fact_interactions"]
    for col in ["had_hold", "has_case", "has_survey_response",
                "is_repeat_contact_7d", "is_escalated", "is_csat_promoter"]:
        assert f[col].dtype == bool, f"{col} is {f[col].dtype}, expected bool"


def test_coverage_flags_match_their_payload(model):
    f = model["fact_interactions"]
    assert (f["has_case"] == f["case_id"].notna()).all()
    assert (f["has_survey_response"] == f["survey_response_id"].notna()).all()
    assert (f.loc[~f["has_survey_response"], "csat_score"].isna()).all()


def test_csat_bands_are_consistent(model):
    f = model["fact_interactions"]
    scored = f[f["csat_score"].notna()]
    assert (scored.loc[scored["is_csat_promoter"], "csat_score"] >= 4).all()
    assert (scored.loc[scored["is_csat_detractor"], "csat_score"] <= 2).all()
    assert not (f["is_csat_promoter"] & f["is_csat_detractor"]).any()


def test_sqm_percentages_are_in_range(model):
    s = model["fact_sqm_monthly"]
    assert s["fcr_pct"].between(0, 100).all()
    assert s["industry_benchmark_fcr_pct"].between(0, 100).all()


def test_team_name_lives_in_exactly_one_table(model):
    """Two places to filter team from is an ambiguity trap."""
    holders = [t for t in TABLES if "team_name" in model[t].columns]
    assert holders == ["dim_team"], f"team_name also on {holders}"
