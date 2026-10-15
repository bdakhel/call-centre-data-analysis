"""Generate synthetic case records: data/raw/cases.csv.

A case is the work item an agent opens off the back of a contact. Not every
interaction creates one -- a quick "what's my balance" call usually doesn't --
so this lands at roughly 1,800 cases for 2,000 interactions.

Patterns baked in:
  * Repeat contact within 7 days is much more likely after a negative
    interaction, and most likely of all on high bill disputes.
  * Escalations concentrate on negative disputes.
  * Open cases concentrate in the last couple of weeks; older cases are closed.
  * Customers are reused, so some have 2+ interactions (that is what makes a
    repeat contact possible in the first place).

How repeat contact is built (important for anyone reading the data):
  `repeat_contact_7d` is FORWARD looking -- Y means this customer contacted us
  again within 7 days of this interaction. That is the usual contact centre
  quality signal (the mirror of first contact resolution), not a flag for
  "this contact was itself a repeat". Rather than sampling the flag at random,
  we assign customers so that negative/dispute contacts really do get followed
  up, then read the flag off the resulting data -- so it reconciles exactly.
  Both halves of a repeat pair are forced to create a case, so every Y in this
  file has its matching follow-up case in this file too.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/generate_cases.py
"""

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --- Reproducibility -------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Same fixed "today" as the other generators.
REFERENCE_DATE = date(2026, 10, 4)

INTERACTIONS_PATH = Path("data/raw/interactions.csv")
OUTPUT_PATH = Path("data/raw/cases.csv")

REPEAT_WINDOW_DAYS = 7

# --- Which interactions open a case ---------------------------------------
# Informational contacts often need no follow-up work; disputes always do.
# These rates are tuned to land the file at about 1,800 rows.
CASE_RATE = {
    "high bill dispute": 1.00,
    "payment arrangement": 0.97,
    "move-in/move-out": 0.96,
    "outage or service issue": 0.92,
    "meter reading": 0.88,
    "billing question": 0.80,
    "general enquiry": 0.45,
}

# --- Repeat contact likelihood --------------------------------------------
# Chance an interaction is followed by another contact from the same customer
# inside the 7-day window. Disputes lead because one call rarely settles them.
TOPIC_REPEAT_BASE = {
    "high bill dispute": 0.34,
    "payment arrangement": 0.20,
    "outage or service issue": 0.18,
    "billing question": 0.13,
    "meter reading": 0.12,
    "move-in/move-out": 0.10,
    "general enquiry": 0.06,
}
# A badly handled contact is the main driver of a call-back.
SENTIMENT_REPEAT_MULT = {"negative": 2.1, "neutral": 1.0, "positive": 0.45}
MAX_REPEAT_PROB = 0.65

# --- Status ---------------------------------------------------------------
# Chance a case is escalated, by sentiment. Disputes get an extra push.
ESCALATION_PROB = {"negative": 0.13, "neutral": 0.035, "positive": 0.008}
DISPUTE_ESCALATION_MULT = 2.0

# Cases opened in the last FRESH_CASE_DAYS may still be sitting open; older
# ones have had time to be worked and closed.
FRESH_CASE_DAYS = 14
OPEN_PROB_FRESH = 0.40
OPEN_PROB_OLD = 0.02

# --- Resolution time (days from created to closed), by case type ----------
# (low, high) inclusive. Disputes need a verification read and a billing
# review, so they run for weeks; outages and quick questions close same-day.
RESOLUTION_DAYS = {
    "high bill dispute": (5, 30),
    "meter reading": (2, 14),
    "move-in/move-out": (1, 7),
    "payment arrangement": (0, 3),
    "outage or service issue": (0, 3),
    "billing question": (0, 2),
    "general enquiry": (0, 1),
}
# Escalated cases that do eventually close take noticeably longer.
ESCALATION_DELAY_DAYS = (7, 21)


def repeat_prob(topic, sentiment):
    """Chance this interaction gets a follow-up contact within 7 days."""
    p = TOPIC_REPEAT_BASE[topic] * SENTIMENT_REPEAT_MULT[sentiment]
    return min(p, MAX_REPEAT_PROB)


def assign_customers(interactions):
    """Give every interaction a customer_id, deliberately reusing some.

    Walks the interactions in time order. When an interaction is picked to
    generate a call-back, we hand its customer_id to a later interaction
    inside the 7-day window -- so the repeat contact is a real pair of rows
    rather than a flag we made up. Chains are allowed, which is how a few
    customers end up with three or more contacts.

    Returns (customer_of, linked) where `linked` marks the rows that are one
    half of a repeat pair.
    """
    n = len(interactions)
    dates = interactions["start_datetime"].dt.date.tolist()
    topics = interactions["topic"].tolist()
    sentiments = interactions["true_sentiment"].tolist()

    # Start with everyone being a distinct customer, then merge.
    customer_of = list(range(n))
    linked = [False] * n

    for i in range(n):
        if random.random() >= repeat_prob(topics[i], sentiments[i]):
            continue

        # Candidate follow-ups: a LATER DAY, still inside the window, and not
        # already claimed as somebody else's follow-up.
        deadline = dates[i] + timedelta(days=REPEAT_WINDOW_DAYS)
        candidates = []
        for j in range(i + 1, n):
            if dates[j] > deadline:
                break  # interactions are time-ordered, so we are past the window
            if dates[j] > dates[i] and not linked[j]:
                candidates.append(j)

        if candidates:
            j = random.choice(candidates)
            customer_of[j] = customer_of[i]  # propagates through chains
            linked[i] = linked[j] = True

    # Renumber to contiguous CUST- ids in order of first appearance.
    order = {}
    for raw in customer_of:
        if raw not in order:
            order[raw] = f"CUST-{len(order) + 1:05d}"
    return [order[raw] for raw in customer_of], linked


def pick_status(case_type, sentiment, created):
    """Choose closed / open / escalated for one case."""
    esc = ESCALATION_PROB[sentiment]
    if case_type == "high bill dispute":
        esc *= DISPUTE_ESCALATION_MULT
    if random.random() < esc:
        return "escalated"

    age_days = (REFERENCE_DATE - created).days
    open_prob = OPEN_PROB_FRESH if age_days <= FRESH_CASE_DAYS else OPEN_PROB_OLD
    return "open" if random.random() < open_prob else "closed"


def pick_closed_date(case_type, created):
    """Closed date for a closed case, or None if it could not have closed yet.

    Keeps the file self-consistent: a case whose resolution time would run
    past today simply hasn't closed, so the caller flips it back to open.
    """
    low, high = RESOLUTION_DAYS[case_type]
    closed = created + timedelta(days=random.randint(low, high))
    return closed if closed <= REFERENCE_DATE else None


def main():
    if not INTERACTIONS_PATH.exists():
        raise SystemExit(f"{INTERACTIONS_PATH} not found -- run src/generate_interactions.py first.")

    interactions = pd.read_csv(
        INTERACTIONS_PATH,
        usecols=["interaction_id", "start_datetime", "topic", "true_sentiment"],
        parse_dates=["start_datetime"],
    ).sort_values("start_datetime").reset_index(drop=True)

    customer_of, linked = assign_customers(interactions)

    # --- Decide which interactions open a case --------------------------
    # Both halves of a repeat pair always open one, so every repeat contact is
    # visible in this file and the flag reconciles.
    creates_case = [
        linked[i] or random.random() < CASE_RATE[row.topic]
        for i, row in enumerate(interactions.itertuples())
    ]

    # --- Repeat contact flag, read off the customer assignment ----------
    # Y when the same customer contacts again on a later day within 7 days.
    contacts_by_customer = {}
    for i, cust in enumerate(customer_of):
        contacts_by_customer.setdefault(cust, []).append(interactions["start_datetime"].iloc[i].date())

    rows = []
    for i, row in enumerate(interactions.itertuples()):
        if not creates_case[i]:
            continue

        cust = customer_of[i]
        created = row.start_datetime.date()
        later = [
            d for d in contacts_by_customer[cust]
            if created < d <= created + timedelta(days=REPEAT_WINDOW_DAYS)
        ]

        # case_type mirrors the interaction topic, so the two files join cleanly.
        case_type = row.topic
        status = pick_status(case_type, row.true_sentiment, created)

        closed_date = None
        if status == "closed":
            closed_date = pick_closed_date(case_type, created)
            if closed_date is None:
                status = "open"  # not enough time to have closed yet
        elif status == "escalated":
            # Some escalations have come back and closed; most are still live.
            if random.random() < 0.35:
                low, high = ESCALATION_DELAY_DAYS
                candidate = created + timedelta(days=random.randint(low, high))
                if candidate <= REFERENCE_DATE:
                    closed_date = candidate

        rows.append(
            {
                "case_id": None,  # assigned after sorting, below
                "interaction_id": row.interaction_id,
                "customer_id": cust,
                "case_type": case_type,
                "status": status,
                "created_date": created,
                "closed_date": closed_date,
                "repeat_contact_7d": "Y" if later else "N",
            }
        )

    cases = pd.DataFrame(rows)
    # IDs assigned in chronological order so the file reads like a case log.
    cases["case_id"] = [f"C{i:05d}" for i in range(1, len(cases) + 1)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(OUTPUT_PATH, index=False)

    # --- Summary: confirm the planted patterns ---------------------------
    sentiment_of = dict(zip(interactions["interaction_id"], interactions["true_sentiment"]))
    cases["_sentiment"] = cases["interaction_id"].map(sentiment_of)

    print(f"Wrote {len(cases)} cases to {OUTPUT_PATH} (from {len(interactions)} interactions, "
          f"{len(cases) / len(interactions):.0%} case rate)\n")

    print("Status mix:")
    print(cases["status"].value_counts(normalize=True).round(3).to_string(), "\n")

    print("Repeat contact within 7 days, by sentiment:")
    print(cases.groupby("_sentiment")["repeat_contact_7d"].apply(
        lambda s: round((s == "Y").mean(), 3)).to_string(), "\n")

    print("Repeat contact and escalation rate by case type:")
    by_type = cases.groupby("case_type").agg(
        n=("case_id", "size"),
        pct_repeat=("repeat_contact_7d", lambda s: round((s == "Y").mean(), 3)),
        pct_escalated=("status", lambda s: round((s == "escalated").mean(), 3)),
    )
    print(by_type.sort_values("pct_repeat", ascending=False).to_string(), "\n")

    counts = pd.Series(customer_of).value_counts()
    print(f"Customers: {counts.size} unique across {len(interactions)} interactions")
    print("Interactions per customer:")
    print(counts.value_counts().sort_index().rename_axis("interactions").to_string(), "\n")

    closed = cases[cases["status"] == "closed"]
    days_to_close = (pd.to_datetime(closed["closed_date"]) - pd.to_datetime(closed["created_date"])).dt.days
    print(f"Days to close (closed cases): mean {days_to_close.mean():.1f}, max {days_to_close.max()}")
    print(f"Open/escalated rows with a closed_date: "
          f"{cases[cases.status == 'open']['closed_date'].notna().sum()} open, "
          f"{cases[cases.status == 'escalated']['closed_date'].notna().sum()} escalated (escalations may close)")
    print(f"closed_date before created_date: {(pd.to_datetime(cases['closed_date']) < pd.to_datetime(cases['created_date'])).sum()}")
    print(f"closed_date after today: {(pd.to_datetime(cases['closed_date']).dt.date > REFERENCE_DATE).sum()}")


if __name__ == "__main__":
    main()
