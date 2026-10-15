"""Generate synthetic contact centre interactions: data/raw/interactions.csv.

Creates 2,000 interactions over the last 6 months for the roster in
data/raw/agents.csv, with deliberate patterns baked in so the downstream
sentiment analysis and dashboard have something real to find:

  * Volume is higher on Mondays and in winter months (January peaks).
  * High bill disputes skew strongly negative and run long.
  * Agents still in ramp-up (< 6 months tenure) score lower sentiment.
  * Team 3 ("Move In / Move Out") improves noticeably in the last 2 months.

`true_sentiment` is the HIDDEN ground-truth label. It is what the NLP models
will be scored against -- never feed it to a model as a feature.
`transcript_text` is intentionally left empty; a later script fills it in.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/generate_interactions.py
"""

import random
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --- Reproducibility -------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Same fixed "today" as generate_agents.py, so tenure maths line up and the
# output stays byte-identical across runs.
REFERENCE_DATE = date(2026, 10, 4)

N_INTERACTIONS = 2_000
WINDOW_MONTHS = 6  # set to 12 to include a January in the window (see README note)

AGENTS_PATH = Path("data/raw/agents.csv")
OUTPUT_PATH = Path("data/raw/interactions.csv")

# Tenure below this counts as "new hire" / still ramping up, measured at the
# time of each interaction (not at REFERENCE_DATE).
NEW_HIRE_DAYS = 180

# Team 3 = third team on the roster. It starts slightly below average and then
# improves sharply in the final IMPROVEMENT_WINDOW_DAYS (e.g. after coaching).
IMPROVING_TEAM = "Move In / Move Out"
IMPROVEMENT_WINDOW_DAYS = 60  # "the last 2 months"

# --- Volume shape ----------------------------------------------------------
# Relative call volume by weekday. Monday is the classic contact centre spike
# (weekend backlog + bills noticed over the weekend).
DOW_WEIGHT = {0: 1.60, 1: 1.15, 2: 1.00, 3: 0.97, 4: 0.90}  # Mon..Fri

# Seasonality for an Alberta gas + electricity utility: January is the peak
# (highest winter heating bills land right after the holidays), summer is quiet.
MONTH_WEIGHT = {
    1: 1.70, 2: 1.35, 3: 1.10, 4: 1.00, 5: 0.90, 6: 0.85,
    7: 0.85, 8: 0.90, 9: 1.00, 10: 1.05, 11: 1.15, 12: 1.25,
}

# Business hours in Mountain Time, 07:00-18:xx, with a mid-morning peak and a
# lunch dip. start_datetime is local Alberta wall-clock time (America/Edmonton).
HOUR_WEIGHT = {
    7: 0.40, 8: 0.90, 9: 1.30, 10: 1.45, 11: 1.30, 12: 0.95,
    13: 1.10, 14: 1.20, 15: 1.15, 16: 1.00, 17: 0.70, 18: 0.35,
}

CHANNELS = ["voice", "chat", "email"]
CHANNEL_WEIGHTS = [0.65, 0.25, 0.10]

# --- Routing: queue -> owning team -> topic mix ----------------------------
# Each queue is owned by one team, so an interaction's agent always comes from
# the team that staffs that queue. Weights are the share of total volume.
ROUTING = [
    # (queue,            team,                 topic,                     weight)
    ("Q-BILLING-CARE",   "Billing Care",       "billing question",         0.15),
    ("Q-BILLING-CARE",   "Billing Care",       "meter reading",            0.07),
    ("Q-BILLING-CARE",   "Billing Care",       "general enquiry",          0.04),
    ("Q-OUTAGE-EMERG",   "Outage & Emergency", "outage or service issue",  0.20),
    ("Q-OUTAGE-EMERG",   "Outage & Emergency", "general enquiry",          0.04),
    ("Q-MOVES",          "Move In / Move Out", "move-in/move-out",         0.20),
    ("Q-MOVES",          "Move In / Move Out", "general enquiry",          0.04),
    ("Q-PAY-CREDIT",     "Payments & Credit",  "payment arrangement",      0.14),
    ("Q-PAY-CREDIT",     "Payments & Credit",  "high bill dispute",        0.12),
]

# --- Handle / hold time ----------------------------------------------------
# Median handle time in seconds for a voice contact, by topic. Disputes are the
# long ones (bill walkthroughs, usage history, escalation paths).
TOPIC_HANDLE_SEC = {
    "billing question": 380,
    "high bill dispute": 760,
    "move-in/move-out": 480,
    "payment arrangement": 540,
    "outage or service issue": 430,
    "meter reading": 350,
    "general enquiry": 290,
}

# Chat runs a bit longer than voice (typing, parallel contacts); email is a
# shorter single touch with no hold.
CHANNEL_HANDLE_MULT = {"voice": 1.00, "chat": 1.15, "email": 0.70}

# Agents still ramping up take longer to resolve the same contact.
NEW_HIRE_HANDLE_MULT = 1.20

# Likelihood a voice contact involves any hold at all, by topic.
TOPIC_HOLD_CHANCE = {
    "billing question": 0.45,
    "high bill dispute": 0.80,
    "move-in/move-out": 0.50,
    "payment arrangement": 0.60,
    "outage or service issue": 0.70,
    "meter reading": 0.40,
    "general enquiry": 0.30,
}

# --- Sentiment ------------------------------------------------------------
# Baseline [negative, neutral, positive] mix by topic. Disputes are mostly
# negative; moves and general enquiries are mostly fine.
TOPIC_SENTIMENT = {
    "high bill dispute":       [0.70, 0.22, 0.08],
    "outage or service issue": [0.40, 0.42, 0.18],
    "payment arrangement":     [0.33, 0.45, 0.22],
    "billing question":        [0.22, 0.50, 0.28],
    "meter reading":           [0.20, 0.55, 0.25],
    "move-in/move-out":        [0.15, 0.50, 0.35],
    "general enquiry":         [0.12, 0.53, 0.35],
}
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def weighted_choice(options, weights):
    """Pick one option with the given (unnormalised) weights."""
    weights = np.asarray(weights, dtype=float)
    return options[np.random.choice(len(options), p=weights / weights.sum())]


def business_day_weights(days):
    """Weight each business day by weekday and month seasonality."""
    return np.array([DOW_WEIGHT[d.weekday()] * MONTH_WEIGHT[d.month] for d in days])


def sentiment_probs(topic, is_new_hire, team, days_from_end, hold_time_sec):
    """Build the [neg, neu, pos] probabilities for one interaction.

    Starts from the topic baseline, then applies multiplicative nudges for the
    patterns we want the analysis to surface. Multipliers (rather than fixed
    numbers) keep every topic's own character intact.
    """
    neg, neu, pos = TOPIC_SENTIMENT[topic]

    # Ramp-up effect: newer agents de-escalate less well.
    if is_new_hire:
        neg *= 1.55
        pos *= 0.60

    if team == IMPROVING_TEAM:
        if days_from_end <= IMPROVEMENT_WINDOW_DAYS:
            # The last 2 months: noticeably better after coaching.
            neg *= 0.40
            pos *= 1.90
        else:
            # Before that, this team sat a little below the floor average.
            neg *= 1.30
            pos *= 0.80

    # Long holds sour a contact regardless of topic.
    if hold_time_sec >= 180:
        neg *= 1.35
        pos *= 0.70

    probs = np.array([neg, neu, pos], dtype=float)
    return probs / probs.sum()


def draw_handle_time(topic, channel, is_new_hire):
    """Handle time in seconds: topic median x channel x ramp-up, plus noise."""
    median = TOPIC_HANDLE_SEC[topic] * CHANNEL_HANDLE_MULT[channel]
    if is_new_hire:
        median *= NEW_HIRE_HANDLE_MULT
    # Lognormal gives the right shape: a hard floor, a long right tail.
    seconds = np.random.lognormal(mean=np.log(median), sigma=0.35)
    return int(np.clip(seconds, 45, 3600))


def draw_hold_time(topic, channel):
    """Hold time in seconds. Email has no hold; chat holds are short."""
    if channel == "email":
        return 0
    chance = TOPIC_HOLD_CHANCE[topic] * (0.5 if channel == "chat" else 1.0)
    if np.random.random() > chance:
        return 0
    median = 70 if channel == "chat" else 120
    return int(np.clip(np.random.lognormal(mean=np.log(median), sigma=0.8), 5, 1200))


def main():
    if not AGENTS_PATH.exists():
        raise SystemExit(f"{AGENTS_PATH} not found -- run src/generate_agents.py first.")

    agents = pd.read_csv(AGENTS_PATH, parse_dates=["hire_date"])
    agents["hire_date"] = agents["hire_date"].dt.date

    # Business days in the window (Mon-Fri; stat holidays are ignored to keep
    # this simple). pandas handles the month arithmetic.
    window_end = pd.Timestamp(REFERENCE_DATE)
    window_start = window_end - pd.DateOffset(months=WINDOW_MONTHS)
    days = [d.date() for d in pd.bdate_range(window_start, window_end)]
    day_weights = business_day_weights(days)

    queues = [(q, t, tp) for q, t, tp, _ in ROUTING]
    route_weights = [w for *_, w in ROUTING]

    rows = []
    for _ in range(N_INTERACTIONS):
        queue, team, topic = weighted_choice(queues, route_weights)
        day = weighted_choice(days, day_weights)

        # Only agents already hired on that day can take the contact.
        eligible = agents[(agents["team"] == team) & (agents["hire_date"] <= day)]
        agent = eligible.sample(1).iloc[0]

        tenure_days = (day - agent["hire_date"]).days
        is_new_hire = tenure_days < NEW_HIRE_DAYS

        hour = weighted_choice(list(HOUR_WEIGHT), list(HOUR_WEIGHT.values()))
        start = datetime.combine(day, datetime.min.time()) + timedelta(
            hours=int(hour),
            minutes=int(np.random.randint(0, 60)),
            seconds=int(np.random.randint(0, 60)),
        )

        channel = weighted_choice(CHANNELS, CHANNEL_WEIGHTS)
        handle_time_sec = draw_handle_time(topic, channel, is_new_hire)
        hold_time_sec = draw_hold_time(topic, channel)

        days_from_end = (REFERENCE_DATE - day).days
        probs = sentiment_probs(topic, is_new_hire, team, days_from_end, hold_time_sec)
        true_sentiment = SENTIMENT_LABELS[np.random.choice(3, p=probs)]

        rows.append(
            {
                "agent_id": agent["agent_id"],
                "start_datetime": start,
                "channel": channel,
                "queue": queue,
                "topic": topic,
                "handle_time_sec": handle_time_sec,
                "hold_time_sec": hold_time_sec,
                "transcript_text": "",  # filled in by a later script
                "true_sentiment": true_sentiment,  # hidden ground truth
            }
        )

    interactions = pd.DataFrame(rows).sort_values("start_datetime").reset_index(drop=True)
    # IDs are assigned after sorting so the file reads chronologically.
    interactions.insert(0, "interaction_id", [f"I{i:05d}" for i in range(1, len(interactions) + 1)])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(OUTPUT_PATH, index=False)

    # --- Summary: confirm the planted patterns actually show up -------------
    print(f"Wrote {len(interactions)} interactions to {OUTPUT_PATH}")
    print(f"Window: {interactions['start_datetime'].min()} -> {interactions['start_datetime'].max()} (Mountain Time)\n")

    print("Channel mix:")
    print((interactions["channel"].value_counts(normalize=True).round(3)).to_string(), "\n")

    print("Volume by weekday (Mon first):")
    dow = interactions["start_datetime"].dt.day_name()
    print(dow.value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]).to_string(), "\n")

    print("Negative share and mean handle time by topic:")
    by_topic = interactions.groupby("topic").agg(
        n=("interaction_id", "size"),
        pct_negative=("true_sentiment", lambda s: round((s == "negative").mean(), 3)),
        mean_handle_sec=("handle_time_sec", lambda s: int(s.mean())),
    )
    print(by_topic.sort_values("pct_negative", ascending=False).to_string(), "\n")

    merged = interactions.merge(agents[["agent_id", "team", "hire_date"]], on="agent_id")
    tenure = (merged["start_datetime"].dt.date - merged["hire_date"]).apply(lambda d: d.days)
    merged["ramping"] = tenure < NEW_HIRE_DAYS
    print("Positive share by ramp-up status:")
    print(merged.groupby("ramping")["true_sentiment"].apply(
        lambda s: round((s == "positive").mean(), 3)).to_string(), "\n")

    print(f"'{IMPROVING_TEAM}' positive share, before vs last 2 months:")
    last2 = (pd.Timestamp(REFERENCE_DATE) - merged["start_datetime"]).dt.days <= IMPROVEMENT_WINDOW_DAYS
    team3 = merged[merged["team"] == IMPROVING_TEAM]
    for label, mask in [("earlier", ~last2), ("last 2 months", last2)]:
        subset = merged[(merged["team"] == IMPROVING_TEAM) & mask]
        print(f"  {label:>14}: {round((subset['true_sentiment'] == 'positive').mean(), 3)} (n={len(subset)})")
    print(f"  (team total n={len(team3)})")


if __name__ == "__main__":
    main()
