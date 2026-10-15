"""Build the Power BI star schema into data/processed/.

Turns the raw synthetic tables plus the model scores into a dimensional model
a report can be built on directly:

    fact_interactions    one row per contact      (the main fact)
    fact_sqm_monthly     one row per month x team (the scorecard fact)
    dim_date             full calendar 2026
    dim_agent            the 20 CCRs
    dim_team             the 4 teams
    dim_topic            the 7 contact reasons
    dim_channel          voice / chat / email

dim_team is NOT in the original brief, and it is here because the model does
not work without it. fact_sqm_monthly is at month x TEAM grain, so team has to
be a dimension both facts can hang off. With team as a plain text column on
dim_agent, a team slicer could filter interactions but not the scorecard, and
the two would silently disagree. See powerbi/model_relationships.md.

SURROGATE KEYS
  Every dimension gets a meaningless integer key, assigned over the
  natural key in sorted order so the same input always produces the same key.
  The natural key (agent_id, interaction_id, ...) is kept alongside for
  traceability. dim_date uses the usual YYYYMMDD integer, which is a surrogate
  that happens to be readable.

WHAT IS DELIBERATELY NOT STORED
  Pre-computed ratios. handle_time vs target, FCR, accuracy -- none of them are
  stored as a percentage column, because a percentage cannot be averaged back
  up: the mean of per-call percentages is not the overall percentage. The
  numerator and denominator are stored instead and the ratio belongs in DAX.
  (fact_sqm_monthly is the one exception, and it is flagged as such -- the
  generator hands us a rate with no counts behind it.)

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/build_model_tables.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
SCORES_PATH = PROCESSED / "interaction_scores.csv"

# Same fixed "today" every other generator uses, so tenure and case age agree.
REFERENCE_DATE = date(2026, 10, 4)
NEW_HIRE_DAYS = 180

# Synthetic work emails for row-level security. The domain is a .example.com
# subdomain, which RFC 2606 reserves and guarantees can never resolve or
# receive mail -- so no synthetic agent can collide with a real mailbox
# (CLAUDE.md: no real people, no real email addresses).
EMAIL_DOMAIN = "prairieskyutilities.example.com"

# Fiscal year starts 1 April -- the Government of Alberta convention, and the
# usual one for Alberta utilities. Change this one constant to re-base the
# fiscal columns; nothing else needs touching.
FISCAL_YEAR_START_MONTH = 4


# --- Alberta statutory holidays -------------------------------------------

def easter_sunday(year):
    """Anonymous Gregorian algorithm -- needed for Good Friday."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f, g = (b + 8) // 25, (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    return date(year, (h + el - 7 * m + 114) // 31, ((h + el - 7 * m + 114) % 31) + 1)


def nth_weekday(year, month, weekday, n):
    """nth <weekday> of a month, e.g. the 3rd Monday of February."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def alberta_holidays(year):
    """Alberta general (statutory) holidays as {date: name}.

    Fixed-date holidays landing on a weekend are observed the following
    Monday, which is what decides whether the contact centre is open --
    so the observed day is what goes in, not the nominal one.
    """
    easter = easter_sunday(year)
    fixed = {
        date(year, 1, 1): "New Year's Day",
        date(year, 7, 1): "Canada Day",
        date(year, 11, 11): "Remembrance Day",
        date(year, 12, 25): "Christmas Day",
    }
    holidays = {}
    for day, name in fixed.items():
        if day.weekday() >= 5:  # Sat/Sun -> observed Monday
            day += timedelta(days=7 - day.weekday())
            name += " (observed)"
        holidays[day] = name

    holidays[nth_weekday(year, 2, 0, 3)] = "Alberta Family Day"
    holidays[easter - timedelta(days=2)] = "Good Friday"
    # Victoria Day: the Monday on or before 24 May.
    may24 = date(year, 5, 24)
    holidays[may24 - timedelta(days=(may24.weekday() - 0) % 7)] = "Victoria Day"
    holidays[nth_weekday(year, 9, 0, 1)] = "Labour Day"
    holidays[nth_weekday(year, 10, 0, 2)] = "Thanksgiving Day"
    return holidays


# --- Dimensions ------------------------------------------------------------

def build_dim_date(start, end):
    """Full calendar days from start to end, padded to whole years.

    Padded because Power BI time intelligence (YTD, same-period-last-year)
    needs contiguous whole years or it silently returns blanks at the edges.
    """
    first, last = date(start.year, 1, 1), date(end.year, 12, 31)
    days = pd.date_range(first, last, freq="D")

    holidays = {}
    for year in range(first.year, last.year + 1):
        holidays.update(alberta_holidays(year))

    d = pd.DataFrame({"date": days})
    d["date_key"] = d["date"].dt.strftime("%Y%m%d").astype("int64")
    d["year"] = d["date"].dt.year.astype("int16")
    d["quarter"] = d["date"].dt.quarter.astype("int8")
    d["quarter_label"] = "Q" + d["quarter"].astype(str) + " " + d["year"].astype(str)
    d["month_number"] = d["date"].dt.month.astype("int8")
    d["month_name"] = d["date"].dt.strftime("%B")
    d["month_short"] = d["date"].dt.strftime("%b")
    d["month_key"] = d["date"].dt.strftime("%Y%m").astype("int32")
    d["month_label"] = d["date"].dt.strftime("%b %Y")
    d["day_of_month"] = d["date"].dt.day.astype("int8")
    d["day_of_year"] = d["date"].dt.dayofyear.astype("int16")
    d["week_of_year"] = d["date"].dt.isocalendar().week.astype("int8")

    # Monday = 1, matching ISO and the way a rota is read.
    d["day_of_week"] = (d["date"].dt.dayofweek + 1).astype("int8")
    d["weekday_name"] = d["date"].dt.strftime("%A")
    d["weekday_short"] = d["date"].dt.strftime("%a")
    d["is_weekend"] = d["date"].dt.dayofweek >= 5

    holiday_series = d["date"].dt.date.map(holidays)
    d["holiday_name"] = holiday_series.fillna("")
    d["is_holiday"] = holiday_series.notna()
    d["is_business_day"] = ~d["is_weekend"] & ~d["is_holiday"]

    # Fiscal calendar. April start means Apr-Jun is FQ1 and the fiscal year is
    # named for the calendar year it ENDS in (FY2027 = Apr 2026 - Mar 2027).
    shifted = (d["month_number"] - FISCAL_YEAR_START_MONTH) % 12
    d["fiscal_month_number"] = (shifted + 1).astype("int8")
    d["fiscal_quarter"] = (shifted // 3 + 1).astype("int8")
    d["fiscal_year"] = np.where(d["month_number"] >= FISCAL_YEAR_START_MONTH,
                                d["year"] + 1, d["year"]).astype("int16")
    d["fiscal_quarter_label"] = ("FY" + d["fiscal_year"].astype(str)
                                 + " Q" + d["fiscal_quarter"].astype(str))
    d["fiscal_period_label"] = ("FY" + d["fiscal_year"].astype(str)
                                + " P" + d["fiscal_month_number"].astype(str).str.zfill(2))

    # Sort helpers: Power BI needs an integer to order text columns by.
    d["month_name_sort"] = d["month_number"]
    d["weekday_name_sort"] = d["day_of_week"]

    cols = ["date_key", "date", "year", "quarter", "quarter_label", "month_number",
            "month_name", "month_short", "month_key", "month_label", "month_name_sort",
            "day_of_month", "day_of_year", "week_of_year", "day_of_week",
            "weekday_name", "weekday_short", "weekday_name_sort", "is_weekend",
            "is_holiday", "holiday_name", "is_business_day", "fiscal_year",
            "fiscal_quarter", "fiscal_quarter_label", "fiscal_month_number",
            "fiscal_period_label"]
    return d[cols]


def build_dim_team(agents):
    t = (agents.groupby("team", as_index=False)
         .agg(team_lead=("team_lead", "first"), agent_count=("agent_id", "size"))
         .sort_values("team").reset_index(drop=True))
    t.insert(0, "team_key", (t.index + 1).astype("int16"))
    t["team_lead_email"] = agent_emails(t["team_lead"])
    return t.rename(columns={"team": "team_name"})


def agent_emails(names):
    """firstname.lastname@domain, de-duplicated.

    This column is the join key for row-level security: the CCR role matches
    it against USERPRINCIPALNAME(). Collisions would hand one agent another
    agent's data, so a repeated name gets a numeric suffix rather than being
    allowed to clash.
    """
    seen, out = {}, []
    for name in names:
        local = "".join(ch for ch in name.lower().replace(" ", ".") if ch.isalnum() or ch == ".")
        seen[local] = seen.get(local, 0) + 1
        if seen[local] > 1:
            local = f"{local}{seen[local]}"
        out.append(f"{local}@{EMAIL_DOMAIN}")
    return out


def build_dim_agent(agents, dim_team):
    a = agents.sort_values("agent_id").reset_index(drop=True).copy()
    a.insert(0, "agent_key", (a.index + 1).astype("int16"))
    a = a.merge(dim_team[["team_key", "team_name"]],
                left_on="team", right_on="team_name", how="left")

    a["agent_email"] = agent_emails(a["agent_name"])
    a["hire_date"] = pd.to_datetime(a["hire_date"])
    a["hire_date_key"] = a["hire_date"].dt.strftime("%Y%m%d").astype("int64")
    a["tenure_days"] = (pd.Timestamp(REFERENCE_DATE) - a["hire_date"]).dt.days.astype("int32")
    a["tenure_months"] = (a["tenure_days"] / 30.44).round(1)
    a["is_new_hire"] = a["tenure_days"] < NEW_HIRE_DAYS
    a["tenure_band"] = pd.cut(
        a["tenure_days"], [-1, 180, 730, 1825, 99999],
        labels=["Under 6 months", "6 months - 2 years", "2 - 5 years", "5 years +"]
    ).astype(str)

    return a[["agent_key", "agent_id", "agent_name", "agent_email", "team_key", "hire_date",
              "hire_date_key", "tenure_days", "tenure_months", "tenure_band",
              "is_new_hire", "csat_target", "aht_target_sec"]]


# Grouping so a report can roll the seven reasons up to something a manager
# reads at a glance. Flags mark the two that drive most of the pain.
TOPIC_META = {
    "billing question":        ("Billing",   False, False),
    "high bill dispute":       ("Billing",   True,  True),
    "meter reading":           ("Metering",  False, False),
    "move-in/move-out":        ("Lifecycle", False, False),
    "outage or service issue": ("Service",   False, True),
    "payment arrangement":     ("Credit",    True,  False),
    "general enquiry":         ("General",   False, False),
}


def build_dim_topic(interactions):
    names = sorted(set(interactions["topic"]) | set(TOPIC_META))
    rows = []
    for i, name in enumerate(names, start=1):
        group, is_financial_hardship, is_complaint_prone = TOPIC_META.get(
            name, ("Other", False, False))
        rows.append({
            "topic_key": i, "topic_name": name, "topic_group": group,
            "is_financial_hardship": is_financial_hardship,
            "is_complaint_prone": is_complaint_prone,
        })
    t = pd.DataFrame(rows)
    t["topic_key"] = t["topic_key"].astype("int16")
    return t


CHANNEL_META = {
    "voice": ("Assisted", True,  1),
    "chat":  ("Assisted", True,  2),
    "email": ("Deferred", False, 3),
}


def build_dim_channel(interactions):
    names = sorted(set(interactions["channel"]) | set(CHANNEL_META))
    rows = []
    for i, name in enumerate(names, start=1):
        group, is_realtime, order = CHANNEL_META.get(name, ("Other", False, 99))
        rows.append({"channel_key": i, "channel_name": name,
                     "channel_group": group, "is_realtime": is_realtime,
                     "channel_sort": order})
    c = pd.DataFrame(rows)
    c["channel_key"] = c["channel_key"].astype("int16")
    c["channel_sort"] = c["channel_sort"].astype("int8")
    return c


# --- Facts -----------------------------------------------------------------

def build_fact_interactions(interactions, scores, cases, survey,
                            dim_agent, dim_topic, dim_channel):
    f = interactions.sort_values("interaction_id").reset_index(drop=True).copy()
    f["start_datetime"] = pd.to_datetime(f["start_datetime"])

    f.insert(0, "interaction_key", (f.index + 1).astype("int32"))
    f["date_key"] = f["start_datetime"].dt.strftime("%Y%m%d").astype("int64")
    f["start_hour"] = f["start_datetime"].dt.hour.astype("int8")

    f = f.merge(dim_agent[["agent_key", "agent_id"]], on="agent_id", how="left")
    f = f.merge(dim_topic[["topic_key", "topic_name"]],
                left_on="topic", right_on="topic_name", how="left").drop(columns="topic_name")
    f = f.merge(dim_channel[["channel_key", "channel_name"]],
                left_on="channel", right_on="channel_name", how="left").drop(columns="channel_name")

    # --- model scores -----------------------------------------------------
    if scores is not None:
        f = f.merge(scores, on="interaction_id", how="left")
        f["sentiment_shift"] = (f["sentiment_end"] - f["sentiment_start"]).round(4)
        # Role-playing second topic key: what the classifier PREDICTED, as
        # opposed to how the contact was actually routed. Inactive
        # relationship in the model -- see model_relationships.md.
        f = f.merge(dim_topic[["topic_key", "topic_name"]].rename(
            columns={"topic_key": "topic_predicted_key", "topic_name": "topic_predicted"}),
            on="topic_predicted", how="left")
        f["sentiment_is_correct"] = f["sentiment_label"] == f["true_sentiment"]
        f["topic_is_correct"] = f["topic_predicted"] == f["topic"]
        f["has_score"] = f["sentiment_label"].notna()
    else:
        for c in ["sentiment_label", "sentiment_score", "sentiment_start",
                  "sentiment_end", "sentiment_shift", "topic_predicted",
                  "coaching_note", "generated_by", "model_version", "scored_at"]:
            f[c] = pd.NA
        f["topic_predicted_key"] = pd.NA
        f["sentiment_is_correct"] = pd.NA
        f["topic_is_correct"] = pd.NA
        f["has_score"] = False

    # --- case -------------------------------------------------------------
    c = cases.copy()
    c["created_date"] = pd.to_datetime(c["created_date"])
    c["closed_date"] = pd.to_datetime(c["closed_date"])
    c["case_created_date_key"] = c["created_date"].dt.strftime("%Y%m%d").astype("int64")
    c["case_closed_date_key"] = c["closed_date"].dt.strftime("%Y%m%d").astype("Int64")
    c["days_to_close"] = (c["closed_date"] - c["created_date"]).dt.days.astype("Int32")
    c["is_repeat_contact_7d"] = c["repeat_contact_7d"].eq("Y")
    c["is_escalated"] = c["status"].eq("escalated")
    c["is_case_open"] = c["status"].isin(["open", "escalated"])
    f = f.merge(
        c[["interaction_id", "case_id", "customer_id", "status",
           "case_created_date_key", "case_closed_date_key", "days_to_close",
           "is_repeat_contact_7d", "is_escalated", "is_case_open"]]
        .rename(columns={"status": "case_status"}),
        on="interaction_id", how="left")
    f["has_case"] = f["case_id"].notna()
    # A contact with no case cannot be a repeat contact; keep the flag boolean
    # rather than tri-state so DAX counts behave.
    for col in ["is_repeat_contact_7d", "is_escalated", "is_case_open"]:
        f[col] = f[col].fillna(False).astype(bool)

    # --- survey -----------------------------------------------------------
    s = survey.copy()
    s["is_resolved"] = s["resolved"].eq("Y")
    f = f.merge(
        s[["interaction_id", "response_id", "csat_score", "is_resolved", "comment"]]
        .rename(columns={"response_id": "survey_response_id",
                         "comment": "survey_comment"}),
        on="interaction_id", how="left")
    f["has_survey_response"] = f["survey_response_id"].notna()
    f["csat_score"] = f["csat_score"].astype("Int8")
    # Top-2-box is the standard CSAT headline; stored as a flag so it SUMs.
    f["is_csat_promoter"] = f["csat_score"].ge(4).fillna(False).astype(bool)
    f["is_csat_detractor"] = f["csat_score"].le(2).fillna(False).astype(bool)
    f["survey_comment"] = f["survey_comment"].fillna("")

    f["handle_time_sec"] = f["handle_time_sec"].astype("int32")
    f["hold_time_sec"] = f["hold_time_sec"].astype("int32")
    f["had_hold"] = f["hold_time_sec"] > 0

    cols = [
        "interaction_key", "interaction_id",
        "date_key", "agent_key", "topic_key", "topic_predicted_key", "channel_key",
        "start_datetime", "start_hour", "queue",
        "handle_time_sec", "hold_time_sec", "had_hold",
        "sentiment_label", "sentiment_score", "sentiment_start", "sentiment_end",
        "sentiment_shift", "true_sentiment", "sentiment_is_correct",
        "topic_predicted", "topic_is_correct", "has_score",
        "has_case", "case_id", "customer_id", "case_status",
        "case_created_date_key", "case_closed_date_key", "days_to_close",
        "is_repeat_contact_7d", "is_escalated", "is_case_open",
        "has_survey_response", "survey_response_id", "csat_score",
        "is_resolved", "is_csat_promoter", "is_csat_detractor", "survey_comment",
        "coaching_note", "generated_by", "model_version", "scored_at",
    ]
    return f[[c for c in cols if c in f.columns]]


def build_fact_sqm(sqm, dim_team):
    s = sqm.copy()
    s["month_start"] = pd.to_datetime(s["month"] + "-01")
    s["date_key"] = s["month_start"].dt.strftime("%Y%m%d").astype("int64")
    s["month_key"] = s["month_start"].dt.strftime("%Y%m").astype("int32")
    s = s.merge(dim_team[["team_key", "team_name"]],
                left_on="team", right_on="team_name", how="left")
    s = s.sort_values(["month_key", "team_key"]).reset_index(drop=True)
    s.insert(0, "sqm_key", (s.index + 1).astype("int16"))
    s["fcr_vs_benchmark_pp"] = (s["fcr_pct"] - s["industry_benchmark_fcr_pct"]).round(2)
    s["is_above_benchmark"] = s["fcr_pct"] > s["industry_benchmark_fcr_pct"]
    return s[["sqm_key", "date_key", "month_key", "team_key", "fcr_pct",
              "industry_benchmark_fcr_pct", "fcr_vs_benchmark_pp", "is_above_benchmark"]]


# --- Main ------------------------------------------------------------------

def main():
    for path in [RAW / "interactions.csv", RAW / "agents.csv", RAW / "cases.csv",
                 RAW / "csat_survey.csv", RAW / "sqm_monthly.csv"]:
        if not path.exists():
            raise SystemExit(f"{path} not found -- run: python src/generate_all.py")

    interactions = pd.read_csv(RAW / "interactions.csv")
    agents = pd.read_csv(RAW / "agents.csv")
    cases = pd.read_csv(RAW / "cases.csv")
    survey = pd.read_csv(RAW / "csat_survey.csv", keep_default_na=False, na_values=[""])
    sqm = pd.read_csv(RAW / "sqm_monthly.csv")

    scores = None
    if SCORES_PATH.exists():
        scores = pd.read_csv(SCORES_PATH)
        print(f"Using model scores from {SCORES_PATH} ({len(scores):,} rows)")
    else:
        print(f"{SCORES_PATH} not found -- score columns will be blank. "
              f"Run src/run_pipeline.py first for a complete model.")

    dim_team = build_dim_team(agents)
    dim_agent = build_dim_agent(agents, dim_team)
    dim_topic = build_dim_topic(interactions)
    dim_channel = build_dim_channel(interactions)

    fact_interactions = build_fact_interactions(
        interactions, scores, cases, survey, dim_agent, dim_topic, dim_channel)
    fact_sqm_monthly = build_fact_sqm(sqm, dim_team)

    # dim_date must span EVERY date the model references, not just the fact
    # dates -- hire_date_key reaches back ~6 years, and a key column that
    # cannot join is worse than no key column at all.
    span_start = min(pd.to_datetime(interactions["start_datetime"]).min().date(),
                     pd.to_datetime(agents["hire_date"]).min().date())
    span_end = max(pd.to_datetime(cases["closed_date"]).max().date(),
                   pd.to_datetime(interactions["start_datetime"]).max().date())
    dim_date = build_dim_date(span_start, span_end)

    tables = {
        "dim_date": dim_date, "dim_team": dim_team, "dim_agent": dim_agent,
        "dim_topic": dim_topic, "dim_channel": dim_channel,
        "fact_interactions": fact_interactions, "fact_sqm_monthly": fact_sqm_monthly,
    }

    # --- Referential integrity: never ship a model with orphan keys -------
    checks = [
        ("fact_interactions.date_key", fact_interactions["date_key"], dim_date["date_key"]),
        ("fact_interactions.agent_key", fact_interactions["agent_key"], dim_agent["agent_key"]),
        ("fact_interactions.topic_key", fact_interactions["topic_key"], dim_topic["topic_key"]),
        ("fact_interactions.channel_key", fact_interactions["channel_key"], dim_channel["channel_key"]),
        ("fact_sqm_monthly.date_key", fact_sqm_monthly["date_key"], dim_date["date_key"]),
        ("fact_sqm_monthly.team_key", fact_sqm_monthly["team_key"], dim_team["team_key"]),
        ("dim_agent.team_key", dim_agent["team_key"], dim_team["team_key"]),
        ("dim_agent.hire_date_key", dim_agent["hire_date_key"], dim_date["date_key"]),
    ]
    if fact_interactions["topic_predicted_key"].notna().any():
        checks.append(("fact_interactions.topic_predicted_key",
                       fact_interactions["topic_predicted_key"].dropna(), dim_topic["topic_key"]))
    for name, fk, pk in checks:
        orphans = set(fk.dropna()) - set(pk)
        if orphans:
            raise SystemExit(f"{name}: {len(orphans)} orphan key(s), e.g. {sorted(orphans)[:3]}")

    # Case date keys must also resolve, or a second date relationship breaks.
    for col in ["case_created_date_key", "case_closed_date_key"]:
        orphans = set(fact_interactions[col].dropna()) - set(dim_date["date_key"])
        if orphans:
            raise SystemExit(f"fact_interactions.{col}: {len(orphans)} orphan key(s)")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(PROCESSED / f"{name}.csv", index=False)

    # --- Summary ----------------------------------------------------------
    print(f"\nWrote {len(tables)} tables to {PROCESSED}/\n")
    summary = pd.DataFrame([
        {"table": n, "rows": len(t), "columns": t.shape[1],
         "grain": {
             "dim_date": "one row per calendar day",
             "dim_team": "one row per team",
             "dim_agent": "one row per CCR",
             "dim_topic": "one row per contact reason",
             "dim_channel": "one row per channel",
             "fact_interactions": "one row per contact",
             "fact_sqm_monthly": "one row per month x team",
         }[n]}
        for n, t in tables.items()
    ])
    print(summary.to_string(index=False))

    print(f"\ndim_date: {dim_date['date'].min().date()} -> {dim_date['date'].max().date()} "
          f"({len(dim_date):,} days, {int(dim_date['is_business_day'].sum()):,} business days, "
          f"{int(dim_date['is_holiday'].sum())} Alberta statutory holidays)")

    print("\nCoverage in fact_interactions:")
    for flag, label in [("has_score", "model scores"), ("has_case", "a case"),
                        ("has_survey_response", "a survey response")]:
        if flag in fact_interactions:
            n = int(fact_interactions[flag].sum())
            print(f"  {label:20s} {n:5,} of {len(fact_interactions):,} "
                  f"({n / len(fact_interactions):.0%})")

    # Contacts dated on a statutory holiday: the interaction generator uses
    # business days but does not exclude stat holidays, so this is expected
    # and worth stating rather than hiding.
    holiday_keys = set(dim_date.loc[dim_date["is_holiday"], "date_key"])
    on_holiday = fact_interactions["date_key"].isin(holiday_keys).sum()
    print(f"\nContacts dated on a statutory holiday: {on_holiday} "
          f"(generate_interactions.py uses business days but does not skip stat "
          f"holidays -- dim_date.is_business_day now makes them filterable)")

    print("\nRelationships are documented in powerbi/model_relationships.md")


if __name__ == "__main__":
    main()
