"""Generate the two synthetic survey / scorecard feeds.

  data/raw/csat_survey.csv  -- post-contact CSAT responses, one row per
                               interaction that actually answered the survey
                               (about 15% of them, skewed towards the angry).
  data/raw/sqm_monthly.csv  -- month x team service quality scorecard with
                               first contact resolution against an industry
                               benchmark.

Two things worth knowing about how these are built:

  * Non-response bias is deliberate. Unhappy customers answer surveys more
    often than happy ones, so a naive average of csat_score reads lower than
    the true population sentiment. That gap is the point -- it is what makes
    the transcript-based sentiment model worth building.

  * fcr_pct is a DESIGNED series, not a derived one. Each team has a base
    level reflecting how hard its work is, Team 3 gets an explicit step up in
    the last 2 months to match the sentiment improvement, and the
    month-to-month wobble comes from that cell's real repeat-contact rate.
    Deriving the level purely from repeat contacts was the first attempt and
    moved Team 3 by under 2 points -- invisible on a dashboard -- because
    moves resolve first time whatever the conversation was like.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/generate_surveys.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

# --- Reproducibility -------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INTERACTIONS_PATH = Path("data/raw/interactions.csv")
AGENTS_PATH = Path("data/raw/agents.csv")
CASES_PATH = Path("data/raw/cases.csv")
CSAT_PATH = Path("data/raw/csat_survey.csv")
SQM_PATH = Path("data/raw/sqm_monthly.csv")

# =========================================================================
# CSAT survey
# =========================================================================
# Chance of answering the survey at all, by true sentiment. Angry customers
# are the most motivated to respond -- that is the non-response bias above.
# Weighted against the actual sentiment mix this lands near 15% overall.
RESPONSE_RATE = {"negative": 0.21, "neutral": 0.13, "positive": 0.12}

# csat_score distributions over 1..5, by true sentiment. Deliberately noisy
# and overlapping: plenty of negative contacts still give a 3, and the odd
# positive one gives a 1, so the score is an imperfect label.
SCORE_PROBS = {
    "negative": [0.45, 0.28, 0.15, 0.08, 0.04],
    "neutral":  [0.08, 0.14, 0.34, 0.30, 0.14],
    "positive": [0.02, 0.04, 0.12, 0.34, 0.48],
}

# Chance the customer says the issue was resolved, by the score they gave.
RESOLVED_PROB = {1: 0.10, 2: 0.26, 3: 0.55, 4: 0.84, 5: 0.95}
# ... and if they had to contact us again within 7 days, they say no far more
# often. Keeps the survey consistent with cases.csv.
REPEAT_RESOLVED_MULT = 0.40

# Chance the comment box is left empty, by score band. People who are happy
# mostly just give the score and move on.
BLANK_COMMENT_PROB = {"low": 0.22, "mid": 0.45, "high": 0.55}

COMMENTS = {
    "low": [
        "Still not resolved. Third time I've had to explain this.",
        "Agent was polite but couldn't actually do anything.",
        "Nobody can give me a straight answer about my bill.",
        "Waited 25 minutes to be told to wait another two weeks.",
        "Charged for gas I didn't use and no one will admit it.",
        "I was promised a call back and never got one.",
        "Felt like I was being read a script.",
        "The estimated readings are a joke. Send someone out.",
        "Worst customer service I have dealt with in years.",
        "Transferred twice and then the line dropped.",
        "Shouldn't need a survey to tell you this is broken.",
        "No heat for two days and no real explanation.",
    ],
    "mid": [
        "Answered my question but took a while.",
        "Fine, I suppose. Nothing special.",
        "Agent was helpful, the system clearly isn't.",
        "Got there in the end.",
        "Reasonable service, hold time was too long.",
        "Information was correct but hard to follow.",
        "OK. Would prefer not to have to call at all.",
        "Resolved, though I had to push for it.",
    ],
    "high": [
        "Very helpful, explained everything clearly.",
        "Sorted out my payment plan in one call. Thank you.",
        "Agent was patient with all my questions.",
        "Quick, friendly and actually fixed the problem.",
        "Best call I've had with a utility. Genuinely.",
        "She knew exactly what to do. Great help.",
        "He stayed on the line until it was done.",
        "Clear answers and no run-around. Appreciated.",
        "Took the stress out of moving house.",
        "Fast and professional, no complaints.",
    ],
}


def score_band(score):
    """Map a 1-5 score to the comment/blank band it belongs to."""
    if score <= 2:
        return "low"
    return "mid" if score == 3 else "high"


def build_csat(interactions, repeat_by_interaction):
    """One row per interaction whose customer answered the survey."""
    rows = []
    for row in interactions.itertuples():
        if random.random() >= RESPONSE_RATE[row.true_sentiment]:
            continue

        score = int(np.random.choice([1, 2, 3, 4, 5], p=SCORE_PROBS[row.true_sentiment]))

        resolved_prob = RESOLVED_PROB[score]
        if repeat_by_interaction.get(row.interaction_id) == "Y":
            resolved_prob *= REPEAT_RESOLVED_MULT

        band = score_band(score)
        comment = "" if random.random() < BLANK_COMMENT_PROB[band] else random.choice(COMMENTS[band])

        rows.append(
            {
                "response_id": None,  # assigned after sorting, below
                "interaction_id": row.interaction_id,
                "csat_score": score,
                "resolved": "Y" if random.random() < resolved_prob else "N",
                "comment": comment,
            }
        )

    csat = pd.DataFrame(rows)
    # Responses are listed in interaction order, so ids read chronologically.
    csat["response_id"] = [f"R{i:04d}" for i in range(1, len(csat) + 1)]
    return csat


# =========================================================================
# SQM monthly scorecard
# =========================================================================
# Pre-improvement FCR level per team, in percent. These are ordered by how
# hard each team's work actually is -- Payments & Credit carries the disputes
# and arrears and sits below the industry benchmark, Billing Care sits above
# it. Team 3 (Move In / Move Out) starts LOW because it was the
# underperforming team, which is what its weaker early sentiment reflects.
TEAM_FCR_BASE = {
    "Payments & Credit": 71.5,
    "Move In / Move Out": 72.0,
    "Outage & Emergency": 74.2,
    "Billing Care": 76.0,
}

# The coaching step: Team 3 climbs by this much in the last 2 months, taking
# it from the bottom of the table to above the benchmark. Mirrors the
# sentiment improvement planted in generate_interactions.py.
IMPROVING_TEAM = "Move In / Move Out"
IMPROVEMENT_MONTHS = ["2026-08", "2026-09"]
IMPROVEMENT_PCT = 6.0

# Month-to-month movement is taken from each cell's real repeat-contact rate
# (FCR is the inverse of repeat contact) rather than being pure noise, plus a
# little measurement jitter on top.
WOBBLE_SLOPE = 10.0
JITTER_SD = 0.4

FCR_MIN, FCR_MAX = 70.0, 80.0

# The published industry figure: one number per month, the same for every team.
BENCHMARK_MEAN = 74.0
BENCHMARK_SD = 0.4

# Skip month x team cells with too little volume to score. The interaction
# window ends 2 October, so October is a two-day stub rather than a month.
MIN_MONTH_VOLUME = 20


def build_sqm(interactions, cases):
    """Month x team FCR scorecard against the industry benchmark.

    fcr_pct is a designed series, not a derived one: each team has a base
    level, Team 3 gets an explicit step up in the last 2 months, and the
    month-to-month wobble comes from that cell's actual repeat-contact rate
    relative to its team average. Deriving the level purely from repeat
    contacts was tried first and moved Team 3 by under 2 points -- too small
    to read on a dashboard -- because moves resolve first time regardless of
    how the conversation went.
    """
    merged = cases.merge(
        interactions[["interaction_id", "team", "start_datetime"]], on="interaction_id"
    )
    merged["month"] = merged["start_datetime"].dt.to_period("M").astype(str)

    cells = merged.groupby(["month", "team"]).agg(
        cases=("case_id", "size"),
        repeat_rate=("repeat_contact_7d", lambda s: (s == "Y").mean()),
    ).reset_index()
    cells = cells[cells["cases"] >= MIN_MONTH_VOLUME].copy()

    # Each team's own average repeat rate is the reference point, so the
    # wobble reflects a good or bad month for that team specifically.
    team_mean_repeat = cells.groupby("team")["repeat_rate"].transform("mean")

    base = cells["team"].map(TEAM_FCR_BASE)
    step = np.where(
        (cells["team"] == IMPROVING_TEAM) & cells["month"].isin(IMPROVEMENT_MONTHS),
        IMPROVEMENT_PCT,
        0.0,
    )
    wobble = -WOBBLE_SLOPE * (cells["repeat_rate"] - team_mean_repeat)
    jitter = np.random.normal(0.0, JITTER_SD, len(cells))

    cells["fcr_pct"] = (base + step + wobble + jitter).clip(FCR_MIN, FCR_MAX).round(1)

    # One benchmark per month, identical across teams (it is an industry
    # figure, not a team measurement).
    benchmarks = {
        m: round(float(np.clip(np.random.normal(BENCHMARK_MEAN, BENCHMARK_SD), 73.0, 75.0)), 1)
        for m in sorted(cells["month"].unique())
    }
    cells["industry_benchmark_fcr_pct"] = cells["month"].map(benchmarks)

    return cells[["month", "team", "fcr_pct", "industry_benchmark_fcr_pct"]].sort_values(
        ["month", "team"]
    ).reset_index(drop=True)


def main():
    for path in (INTERACTIONS_PATH, AGENTS_PATH, CASES_PATH):
        if not path.exists():
            raise SystemExit(f"{path} not found -- run the earlier generators first.")

    interactions = pd.read_csv(
        INTERACTIONS_PATH,
        usecols=["interaction_id", "agent_id", "start_datetime", "topic", "true_sentiment"],
        parse_dates=["start_datetime"],
    )
    agents = pd.read_csv(AGENTS_PATH, usecols=["agent_id", "team"])
    interactions = interactions.merge(agents, on="agent_id").sort_values(
        "start_datetime"
    ).reset_index(drop=True)

    cases = pd.read_csv(CASES_PATH, usecols=["case_id", "interaction_id", "repeat_contact_7d"])
    repeat_by_interaction = dict(zip(cases["interaction_id"], cases["repeat_contact_7d"]))

    csat = build_csat(interactions, repeat_by_interaction)
    sqm = build_sqm(interactions, cases)

    CSAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    csat.to_csv(CSAT_PATH, index=False)
    sqm.to_csv(SQM_PATH, index=False)

    # --- Summary ---------------------------------------------------------
    sentiment_of = dict(zip(interactions["interaction_id"], interactions["true_sentiment"]))
    csat["_sentiment"] = csat["interaction_id"].map(sentiment_of)

    print(f"Wrote {len(csat)} CSAT responses to {CSAT_PATH} "
          f"({len(csat) / len(interactions):.1%} of {len(interactions)} interactions)\n")

    print("Response rate by true sentiment (angry customers answer more):")
    responded = interactions["interaction_id"].isin(csat["interaction_id"])
    print(interactions.assign(responded=responded).groupby("true_sentiment")["responded"]
          .mean().round(3).to_string(), "\n")

    print("Mean csat_score by true sentiment (correlated, not deterministic):")
    print(csat.groupby("_sentiment")["csat_score"].agg(["size", "mean"]).round(2).to_string(), "\n")

    print("Score distribution:")
    print(csat["csat_score"].value_counts().sort_index().to_string(), "\n")

    print(f"resolved = Y: {(csat['resolved'] == 'Y').mean():.1%}   "
          f"blank comments: {(csat['comment'] == '').mean():.1%}")
    repeat_flag = csat["interaction_id"].map(repeat_by_interaction)
    print("resolved = Y, split by whether they contacted us again within 7 days:")
    print(csat.assign(repeat=repeat_flag).groupby("repeat")["resolved"]
          .apply(lambda s: round((s == "Y").mean(), 3)).to_string(), "\n")

    print(f"Wrote {len(sqm)} month x team rows to {SQM_PATH}")
    print(f"fcr_pct range: {sqm['fcr_pct'].min()} - {sqm['fcr_pct'].max()}, "
          f"benchmark range: {sqm['industry_benchmark_fcr_pct'].min()} - "
          f"{sqm['industry_benchmark_fcr_pct'].max()}\n")
    print(sqm.pivot(index="month", columns="team", values="fcr_pct").to_string())


if __name__ == "__main__":
    main()
