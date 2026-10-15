"""Generate the synthetic contact centre roster: data/raw/agents.csv.

Creates 20 fictional Customer Care Representatives (CCRs) split evenly across
4 teams, each team fronted by a fictional team lead.

All data here is SYNTHETIC. No real people, no real company names (see CLAUDE.md).

Run from the project root:
    python src/generate_agents.py
"""

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# --- Reproducibility -------------------------------------------------------
# Seed 42 everywhere so re-running this script always produces the same roster.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_CA")  # Canadian locale -> plausible Canadian-sounding names
Faker.seed(SEED)

# A FIXED reference "today" keeps hire_date (and therefore who counts as a new
# hire) stable across runs. Bump this deliberately if you want to age the data.
REFERENCE_DATE = date(2026, 10, 4)

# --- Roster shape ----------------------------------------------------------
AGENTS_PER_TEAM = 5
# Functional team names for a natural gas + electricity contact centre (Alberta).
TEAMS = [
    "Billing Care",
    "Outage & Emergency",
    "Move In / Move Out",
    "Payments & Credit",
]
N_AGENTS = len(TEAMS) * AGENTS_PER_TEAM  # 20

# Three of the twenty are recent hires (< 6 months of tenure). Ramp-up effects
# for these agents show up later in CSAT / AHT analysis.
N_NEW_HIRES = 3

TENURE_MAX_DAYS = 6 * 365  # hire dates spread over the last ~6 years
NEW_HIRE_MAX_DAYS = 180  # "under 6 months" = fewer than 180 days of tenure

OUTPUT_PATH = Path("data/raw/agents.csv")


def unique_names(n, taken=None):
    """Return n distinct full names, avoiding anything already in `taken`.

    Faker can repeat names, so we loop until we have enough distinct ones.
    Built from first + last name rather than fake.name() because the latter
    sometimes tacks on titles/suffixes ("Dr.", "DDS") that look wrong on a
    contact centre roster.
    """
    seen = set(taken or [])
    names = []
    while len(names) < n:
        candidate = f"{fake.first_name()} {fake.last_name()}"
        if candidate not in seen:
            seen.add(candidate)
            names.append(candidate)
    return names


def main():
    # One fictional lead per team. Generated first so leads and CCRs never
    # share a name.
    lead_names = unique_names(len(TEAMS))
    team_leads = dict(zip(TEAMS, lead_names))

    agent_names = unique_names(N_AGENTS, taken=lead_names)

    # Decide which agent rows are the new hires, then build tenure accordingly.
    new_hire_rows = set(random.sample(range(N_AGENTS), N_NEW_HIRES))

    rows = []
    for i in range(N_AGENTS):
        if i in new_hire_rows:
            # Somewhere between ~2 weeks and just under 6 months of tenure.
            tenure_days = random.randint(14, NEW_HIRE_MAX_DAYS - 1)
        else:
            # Tenured staff: at least 6 months, up to ~6 years.
            tenure_days = random.randint(NEW_HIRE_MAX_DAYS, TENURE_MAX_DAYS)

        team = TEAMS[i // AGENTS_PER_TEAM]  # 5 agents per team, in order
        rows.append(
            {
                "agent_id": f"A{i + 1:03d}",  # A001 ... A020
                "agent_name": agent_names[i],
                "team": team,
                "team_lead": team_leads[team],
                "hire_date": REFERENCE_DATE - timedelta(days=tenure_days),
                # Per-agent service targets, the kind a WFM team would set.
                "csat_target": round(random.uniform(0.80, 0.85), 2),
                "aht_target_sec": random.randint(420, 480),
            }
        )

    agents = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    agents.to_csv(OUTPUT_PATH, index=False)

    # Quick console summary so you can eyeball the result without opening the file.
    tenure_months = (
        (pd.Timestamp(REFERENCE_DATE) - pd.to_datetime(agents["hire_date"])).dt.days / 30.44
    )
    print(f"Wrote {len(agents)} agents to {OUTPUT_PATH}")
    print(agents["team"].value_counts().to_string())
    print(f"New hires (< 6 months): {(tenure_months < 6).sum()}")
    print(agents.head().to_string(index=False))


if __name__ == "__main__":
    main()
