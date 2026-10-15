"""Compare every label source we have for the 100-row sample, side by side.

Four sources, kept strictly separate because they are NOT interchangeable:

  human        data/labels/human_labels.csv   -- you, in the Streamlit app.
                                                  The only independent reading.
  ai_judge     data/labels/ai_labels.csv      -- llama3.1:8b, blind. A second
                                                  opinion, not ground truth.
  roberta      data/processed/interaction_scores.csv -- the model under test.
  generator    data/raw/interactions.csv (true_sentiment) -- what the synthetic
                                                  data generator intended.

The last one deserves a caveat: `true_sentiment` is not truth about a real
conversation, it is the label the generator sampled BEFORE writing the
transcript from a phrase bank. If the bank's "neutral" lines read as mildly
positive to a human, the generator's label is the thing that is wrong. That is
precisely why the human labels matter.

Coverage differs by source (you may have labelled 37 of 100), so every
agreement figure prints the n it was computed on. Nothing is averaged across
sources and nothing is imputed.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/compare_labels.py
"""

from pathlib import Path

import pandas as pd

TO_LABEL = Path("data/labels/to_label.csv")
HUMAN = Path("data/labels/human_labels.csv")
AI = Path("data/labels/ai_labels.csv")
SCORES = Path("data/processed/interaction_scores.csv")
INTERACTIONS = Path("data/raw/interactions.csv")
MANIFEST = Path("data/labels/to_label_manifest.csv")

ORDER = ["negative", "neutral", "positive"]


def dedup(path, id_col="interaction_id"):
    """Last write wins -- the label files are append logs."""
    return pd.read_csv(path).drop_duplicates(id_col, keep="last")


def agreement(frame, a, b):
    """(pct, n) over rows where BOTH sources have a label."""
    both = frame[[a, b]].dropna()
    both = both[(both[a] != "") & (both[b] != "")]
    if both.empty:
        return None, 0
    return (both[a] == both[b]).mean(), len(both)


def confusion(frame, rows_col, cols_col):
    both = frame[[rows_col, cols_col]].dropna()
    both = both[(both[rows_col] != "") & (both[cols_col] != "")]
    if both.empty:
        return None
    return pd.crosstab(both[rows_col], both[cols_col]).reindex(
        index=ORDER, columns=ORDER, fill_value=0)


def direction(frame, a, b):
    """How often is `a` more negative than `b`, and vice versa?

    This is the question that matters for a miscalibrated classifier: errors
    scattered in both directions are noise, errors all pointing one way are a
    threshold set in the wrong place.
    """
    rank = {"negative": 0, "neutral": 1, "positive": 2}
    both = frame[[a, b]].dropna()
    both = both[(both[a] != "") & (both[b] != "")]
    if both.empty:
        return None
    delta = both[a].map(rank) - both[b].map(rank)
    return {
        "n": len(both),
        "agree": int((delta == 0).sum()),
        f"{a}_more_negative": int((delta < 0).sum()),
        f"{a}_more_positive": int((delta > 0).sum()),
    }


def main():
    if not TO_LABEL.exists():
        raise SystemExit(f"{TO_LABEL} not found -- run src/make_label_sample.py")

    frame = pd.read_csv(TO_LABEL, keep_default_na=False)[["interaction_id"]].copy()
    present = []

    if HUMAN.exists():
        frame = frame.merge(
            dedup(HUMAN)[["interaction_id", "human_label"]].rename(
                columns={"human_label": "human"}), on="interaction_id", how="left")
        present.append("human")
    if AI.exists():
        ai = dedup(AI)
        ai = ai[ai["ai_label"].notna() & (ai["ai_label"] != "")]
        frame = frame.merge(ai[["interaction_id", "ai_label"]].rename(
            columns={"ai_label": "ai_judge"}), on="interaction_id", how="left")
        present.append("ai_judge")
    if SCORES.exists():
        frame = frame.merge(
            pd.read_csv(SCORES)[["interaction_id", "sentiment_label"]].rename(
                columns={"sentiment_label": "roberta"}), on="interaction_id", how="left")
        present.append("roberta")
    if INTERACTIONS.exists():
        frame = frame.merge(
            pd.read_csv(INTERACTIONS)[["interaction_id", "true_sentiment"]].rename(
                columns={"true_sentiment": "generator"}), on="interaction_id", how="left")
        present.append("generator")

    print(f"Sample: {len(frame)} interactions\n")

    # --- Coverage --------------------------------------------------------
    coverage = pd.DataFrame({
        "labelled": [frame[c].replace("", pd.NA).notna().sum() for c in present],
    }, index=present)
    coverage["coverage"] = (coverage["labelled"] / len(frame)).map("{:.0%}".format)
    print("Coverage by source:")
    print(coverage.to_string(), "\n")

    # --- Label mix -------------------------------------------------------
    mix = pd.DataFrame({
        c: frame[c].replace("", pd.NA).value_counts(normalize=True)
        for c in present
    }).reindex(ORDER).fillna(0)
    print("Label mix (share of that source's own labels):")
    print(mix.map("{:.0%}".format).to_string(), "\n")

    # --- Pairwise agreement ----------------------------------------------
    print("Pairwise agreement (n = rows where both sources have a label):")
    pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1:]]
    table = []
    for a, b in pairs:
        pct, n = agreement(frame, a, b)
        if n:
            table.append({"pair": f"{a} vs {b}", "agreement": f"{pct:.0%}", "n": n})
    print(pd.DataFrame(table).to_string(index=False), "\n")

    # --- Direction of disagreement ---------------------------------------
    print("Direction of disagreement (is one source systematically harsher?):")
    for a, b in pairs:
        d = direction(frame, a, b)
        if not d or d["n"] == 0:
            continue
        wrong = d["n"] - d["agree"]
        if wrong == 0:
            continue
        harsher = d[f"{a}_more_negative"]
        softer = d[f"{a}_more_positive"]
        skew = ""
        # All-one-direction errors are calibration, not noise.
        if wrong >= 8 and (harsher == 0 or softer == 0):
            skew = "   <-- ALL disagreements point one way: a threshold problem"
        print(f"  {a} vs {b}: {wrong} disagree of {d['n']} | "
              f"{a} harsher {harsher}, {a} softer {softer}{skew}")
    print()

    # --- Confusion matrices against the human set ------------------------
    if "human" in present:
        for other in [c for c in present if c != "human"]:
            cm = confusion(frame, "human", other)
            if cm is None or cm.values.sum() == 0:
                continue
            print(f"human (rows) vs {other} (cols), n={cm.values.sum()}:")
            print(cm.to_string(), "\n")

    # --- Where to spend the next human minute ----------------------------
    if "human" in present and "ai_judge" in present and "roberta" in present:
        unlabelled = frame[frame["human"].replace("", pd.NA).isna()]
        contested = unlabelled[
            unlabelled["ai_judge"].notna() & unlabelled["roberta"].notna()
            & (unlabelled["ai_judge"] != unlabelled["roberta"])
        ]
        print(f"Rows still unlabelled by you: {len(unlabelled)}")
        print(f"  of those, the two models DISAGREE on {len(contested)}.")
        print("  Those are the highest-value rows to label next -- the ones "
              "where\n  a human breaks the tie. The rest the models already "
              "agree on.")
        if len(contested):
            out = Path("data/labels/to_label_priority.csv")
            contested[["interaction_id"]].to_csv(out, index=False)
            print(f"  Written to {out}")


if __name__ == "__main__":
    main()
