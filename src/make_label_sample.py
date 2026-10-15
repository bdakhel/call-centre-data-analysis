"""Pick 100 interactions for blind human labelling.

Writes data/labels/to_label.csv -- the work queue for src/label_app.py.

WHAT IS DELIBERATELY ABSENT FROM THE OUTPUT, AND WHY:
  The whole value of a human label set is that it is an INDEPENDENT yardstick.
  The moment the labeller can see what the model said, or what the generator
  recorded, the labels stop measuring the model and start measuring agreement
  with an anchor. So the file carries no `true_sentiment`, no
  `sentiment_label` / `sentiment_score`, no `topic_predicted`, and no
  `coaching_note`.

  `topic` is withheld too, which goes slightly beyond the brief. It is not a
  model prediction -- it is the generator's own field -- but it is a strong
  PRIOR on sentiment: high bill disputes run 73% negative in this data. Showing
  it would prime the labeller toward the answer the model is also likely to
  give, which is exactly the correlation that makes agreement look better than
  it is. The strata are kept in a sidecar manifest instead, joined back on
  interaction_id at analysis time.

  Rows are SHUFFLED. Sorted by stratum, the labeller would meet seven disputes
  in a row, then seven outages -- and run-length effects in human labelling are
  real. A shuffle costs nothing and removes them.

Sampling is PROPORTIONAL across the 21 topic x channel cells, using the
largest-remainder method to land on exactly 100 rows. Proportional (rather
than equal) allocation means the sample mirrors the live contact mix, so
overall agreement computed from it is an unbiased estimate of production
accuracy without reweighting. Every cell gets at least one row here; the
smallest (email x meter reading) is 0.6% of volume.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/make_label_sample.py
    python src/make_label_sample.py --n 200
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INTERACTIONS_PATH = Path("data/raw/interactions.csv")
OUTPUT_PATH = Path("data/labels/to_label.csv")
MANIFEST_PATH = Path("data/labels/to_label_manifest.csv")

DEFAULT_N = 100
STRATA = ["topic", "channel"]

# Exactly what the labeller sees. Adding a column here is a decision about
# bias, not a formatting choice -- see the module docstring.
LABEL_COLUMNS = ["interaction_id", "channel", "transcript_text", "human_label"]

# Columns that must never reach the labelling file.
FORBIDDEN = [
    "true_sentiment", "sentiment_label", "sentiment_score",
    "sentiment_start", "sentiment_end", "topic_predicted",
    "coaching_note", "generated_by", "topic",
]


def allocate(counts, n):
    """Split n across strata in proportion to `counts` (largest remainder).

    Plain rounding of proportional shares does not sum to n -- it drifts by a
    few either way. The largest-remainder method assigns every stratum its
    whole number of seats, then hands the leftovers to the strata with the
    biggest fractional parts, which lands on exactly n.
    """
    exact = counts / counts.sum() * n
    base = np.floor(exact).astype(int)
    remainder = n - base.sum()

    if remainder > 0:
        # Ties broken by stratum order, which is sorted and therefore stable.
        order = (exact - base).sort_values(ascending=False, kind="mergesort").index
        for stratum in order[:remainder]:
            base[stratum] += 1
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"how many interactions to sample (default {DEFAULT_N})")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not INTERACTIONS_PATH.exists():
        raise SystemExit(f"{INTERACTIONS_PATH} not found -- run: python src/generate_all.py")

    interactions = pd.read_csv(INTERACTIONS_PATH)
    if args.n > len(interactions):
        raise SystemExit(f"asked for {args.n} rows but only {len(interactions)} exist")

    # --- Allocate the sample across topic x channel ----------------------
    counts = interactions.groupby(STRATA).size().sort_index()
    quota = allocate(counts, args.n)

    picked = []
    for stratum, take in quota.items():
        if take == 0:
            continue
        cell = interactions[
            (interactions["topic"] == stratum[0]) & (interactions["channel"] == stratum[1])
        ]
        picked.append(cell.sample(n=take, random_state=SEED))
    sample = pd.concat(picked)

    # Shuffle so the labeller does not meet the strata in blocks.
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # --- Build the two outputs -------------------------------------------
    manifest = sample[["interaction_id", "topic", "channel"]].copy()
    manifest["stratum"] = manifest["topic"] + " | " + manifest["channel"]

    to_label = sample[["interaction_id", "channel", "transcript_text"]].copy()
    to_label["human_label"] = ""  # the column the labeller fills in

    # Belt and braces: assert nothing anchoring survived into the file.
    leaked = [c for c in FORBIDDEN if c in to_label.columns]
    assert not leaked, f"anchoring columns leaked into the labelling file: {leaked}"
    assert list(to_label.columns) == LABEL_COLUMNS, list(to_label.columns)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    to_label.to_csv(args.out, index=False)
    manifest.to_csv(MANIFEST_PATH, index=False)

    # --- Report ----------------------------------------------------------
    print(f"Wrote {len(to_label)} interactions to {args.out}")
    print(f"Columns: {', '.join(to_label.columns)}")
    print(f"Withheld from the labeller: {', '.join(FORBIDDEN)}")
    print(f"Strata recorded separately in {MANIFEST_PATH}\n")

    achieved = pd.crosstab(sample["topic"], sample["channel"])
    print("Sampled (topic x channel):")
    print(achieved.to_string())

    print("\nShare of sample vs share of population:")
    comparison = pd.DataFrame({
        "population_pct": (interactions.groupby(STRATA).size() / len(interactions) * 100),
        "sample_pct": (sample.groupby(STRATA).size() / len(sample) * 100),
    }).round(1).fillna(0.0)
    comparison["diff_pp"] = (comparison["sample_pct"] - comparison["population_pct"]).round(1)
    print(comparison.to_string())
    print(f"\nLargest deviation from the population mix: "
          f"{comparison['diff_pp'].abs().max():.1f} percentage points")

    # The hidden mix, printed for the ANALYST, never written to the file the
    # labeller opens. Useful to know what the ceiling looks like afterwards.
    print("\n(For reference only -- not in to_label.csv) hidden sentiment mix "
          "of the sample:")
    print(sample["true_sentiment"].value_counts().to_string())
    print(f"\nNext: streamlit run src/label_app.py")


if __name__ == "__main__":
    main()
