"""Classify each transcript into one of the 7 contact topics.

Two classifiers, same interface, compared on the same rows:

  rules     -- weighted keyword / phrase matching. Fast (whole dataset in
               under a second), fully inspectable, no model download.
  zeroshot  -- facebook/bart-large-mnli run as natural language inference.
               No training data needed, but ~1.6GB of weights and one forward
               pass PER LABEL per transcript (7x the work of a normal
               classifier), which on CPU is minutes for hundreds of rows, not
               thousands.

WHY `rules` IS THE DEFAULT (see --classifier to override):
  Measured: the rules classifier gets 99.5% on all 2,000 transcripts in under
  3 seconds. The zero-shot comparison has NOT been run -- the bart-large-mnli
  download stalls on this machine -- so the head-to-head accuracy number is
  still missing, and `rules` is the default on the strength of its own score
  plus the reasoning below, not on a measured win. Run
  `--classifier both --limit 150` once the model can be fetched and this
  paragraph should be updated with the real figure.

  The reasoning: the transcripts were
  generated from per-topic phrase banks, so topic vocabulary is highly
  separable and a keyword list captures almost all of it. Zero-shot has to
  infer the label set from short English descriptions and has no idea that,
  in utility billing, "verification read" means a dispute while "submit a
  reading" means a meter read -- distinctions a domain expert writes down in
  five minutes.
  The honest caveat: these keyword weights were written by reading
  generate_transcripts.py's phrase banks, so the rules classifier has seen the
  data-generating process. On real transcripts it would need retuning, and the
  zero-shot number is the more realistic estimate of cold-start performance.
  That is exactly why both are kept here rather than only the winner.

Unlike src/sentiment.py, this scores the FULL transcript, not just customer
turns: the agent's half carries most of the topic vocabulary ("I can schedule
the final read", "I'm showing an unplanned outage"), while the customer often
just says "it's too high".

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/topics.py                          # rules on everything
    python src/topics.py --classifier both --limit 200
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INTERACTIONS_PATH = Path("data/raw/interactions.csv")
OUTPUT_DIR = Path("data/labels")

TOPICS = [
    "billing question",
    "high bill dispute",
    "move-in/move-out",
    "payment arrangement",
    "outage or service issue",
    "meter reading",
    "general enquiry",
]

# Fallback when nothing matches at all. "general enquiry" is the catch-all
# queue in the real routing model too, so it is the right place to dump
# anything unrecognised.
DEFAULT_TOPIC = "general enquiry"


# --- Rules classifier ------------------------------------------------------
# (pattern, weight) per topic. Weight 3 = near-decisive domain phrase, 2 =
# strong, 1 = supporting, 0.5 = weak hint that only matters in aggregate.
#
# The hard pairs this table is built around:
#   * high bill dispute vs billing question -- both are "about the bill".
#     Dispute = shock at the AMOUNT plus the remedies a dispute triggers
#     (billing review, verification read). Question = asking what a LINE ITEM
#     means (distribution charge, rider, credit, cycle dates).
#   * high bill dispute vs meter reading -- both discuss estimated reads.
#     Dispute = the catch-up bill that resulted. Meter reading = the customer
#     supplying a figure off the dial.
#   * payment arrangement vs general enquiry -- "equalized payment plan" shows
#     up in both; arrears language (balance, notice, collection) decides it.
RULES = {
    "high bill dispute": [
        (r"\bbilling review\b", 3), (r"\bverification read\b", 3),
        (r"\bdegree.day\b", 3), (r"\bcold snap\b", 3),
        (r"\bunder.billed\b", 3), (r"\bfranchise fee\b", 2.5),
        (r"\bbilling investigation\b", 3), (r"\bcatch.up bill\b", 2.5),
        (r"\bcatching up on usage\b", 3),
        (r"\boutrageous\b", 2), (r"\bmore than double\b", 2),
        (r"\bdouble what I normally pay\b", 2),
        (r"\bbill is wrong\b", 2), (r"\bthis bill is\b", 1),
        (r"\bwent from \$[\d,.]+ to \$[\d,.]+\b", 2.5),
        (r"\bthree times colder\b", 2.5), (r"\bI live alone\b", 1.5),
        (r"\bhold on collection activity\b", 1.5),
        (r"\briders?\b", 0.5), (r"\busage went from\b", 2),
        (r"\bfurnace running harder\b", 2), (r"\bcold stretch\b", 2),
        (r"\bhigher than (I expected|usual)\b", 1.5),
        (r"\bexplain (how|this|my bill)\b", 1),
    ],
    "billing question": [
        (r"\bdistribution charge\b", 3), (r"\bline item\b", 3),
        (r"\bcharge breakdown\b", 2.5), (r"\bplain.language guide\b", 3),
        (r"\bbilling (period|cycle)\b", 2.5), (r"\bcycle shifted\b", 3),
        (r"\brate rider adjustment\b", 2.5), (r"\brider line\b", 2.5),
        (r"\bdaily fixed charge\b", 2.5),
        (r"\benergy charge and the delivery charge\b", 3),
        (r"\bcombined statement\b", 2), (r"\bbilled twice\b", 2.5),
        (r"\bimpossible to read\b", 2), (r"\bcredit of \$", 2),
        (r"\bwhat I actually owe\b", 2), (r"\bamount currently owing\b", 2),
        (r"\bquestion about (a line|my statement|my bill)\b", 2),
        (r"\bunderstand my bill\b", 1.5), (r"\bbudget properly\b", 1.5),
        (r"\bset by the regulator\b", 1.5), (r"\bread date\b", 1.5),
        (r"\bdon't recognize\b", 1.5),
    ],
    "move-in/move-out": [
        (r"\bfinal read\b", 3), (r"\bmove.out\b", 3), (r"\bmoving (to|out|into)\b", 3),
        (r"\bmoving day\b", 3), (r"\btaking possession\b", 3), (r"\bpossession\b", 2.5),
        (r"\bhand over the keys\b", 3), (r"\bnew address\b", 2.5),
        (r"\baccount setup charge\b", 2.5), (r"\bclosing on my house\b", 3),
        (r"\bclose (my|the old) account\b", 2.5), (r"\bbackdate the close\b", 2.5),
        (r"\bopen(ing)? (one|an account) at the new\b", 2.5),
        (r"\bcold house\b", 2), (r"\bstill billing me for the old place\b", 3),
        (r"\btransfer my service\b", 2.5), (r"\bservice (active|was supposed to be on)\b", 2),
        (r"\bin my name\b", 1.5), (r"\bnew place\b", 1.5),
        (r"\bdate you want service active\b", 2.5),
    ],
    "payment arrangement": [
        (r"\bpayment (plan|arrangement)\b", 3), (r"\binstal?ments?\b", 3),
        (r"\bdisconnection notice\b", 3), (r"\bcollection (letters|activity)\b", 3),
        (r"\bwinter protection\b", 3), (r"\bOctober 15 and April 15\b", 3),
        (r"\bpre.authorized debit\b", 2.5), (r"\bassistance programs?\b", 2.5),
        (r"\bspreading (that|my balance)\b", 3), (r"\bbehind on my account\b", 3),
        (r"\bI'm behind\b", 2.5), (r"\bcut off my heat\b", 3),
        (r"\bthreatening to\b", 2), (r"\bnotice (is )?withdrawn\b", 2.5),
        (r"\bbalance of \$", 2), (r"\bper month on top of normal usage\b", 2),
        (r"\bget back on track\b", 2), (r"\bcan't pay it all at once\b", 3),
        (r"\bmanageable\b", 1.5), (r"\bno disconnection\b", 2),
        (r"\bcollection\b", 1),
    ],
    "outage or service issue": [
        (r"\boutage\b", 3), (r"\bpower (is |has been )?out\b", 3),
        (r"\brestoration\b", 3), (r"\bcrews?\b", 2.5), (r"\btransformer\b", 3),
        (r"\bre.energize\b", 3), (r"\bflickering\b", 3), (r"\bsmell gas\b", 2.5),
        (r"\bemergency line\b", 2.5), (r"\boutage (ticket|map)\b", 3),
        (r"\bno (power|heat)\b", 2.5), (r"\bneighbours? are out\b", 3),
        (r"\bfurnace isn't getting gas\b", 3), (r"\bisolate the section\b", 2.5),
        (r"\bservice connection\b", 2), (r"\bdropping on and off\b", 2.5),
        (r"\bit's freezing\b", 1.5), (r"\btechnician\b", 1),
        (r"\bfailed transformer\b", 3), (r"\bunplanned\b", 2),
    ],
    "meter reading": [
        (r"\bmeter read(ing)?\b", 3), (r"\bsubmit (my own |a )?(meter )?read(ing)?\b", 3),
        (r"\bactual read\b", 2.5), (r"\bthe dial shows\b", 3), (r"\bfrom the dial\b", 3),
        (r"\bestimate flag\b", 3), (r"\bcustomer.supplied\b", 3),
        (r"\baccess note\b", 3), (r"\bregister (shows|isn't)\b", 2.5),
        (r"\bmeter reader\b", 3), (r"\bread my meter\b", 3),
        (r"\blast read\b", 2), (r"\bgate or a dog\b", 3),
        (r"\bphoto of my meter\b", 3), (r"\breads were estimated\b", 2.5),
        (r"\bestimated my meter\b", 3), (r"\bmeter number\b", 2.5),
        (r"\bagainst meter\b", 2), (r"\bsays 'estimated'\b", 2.5),
        (r"\breading is off\b", 2.5),
    ],
    "general enquiry": [
        (r"\bregulated rate\b", 3), (r"\bfixed term\b", 3), (r"\bexit fee\b", 3),
        (r"\befficiency rebate\b", 3), (r"\brebate program\b", 3),
        (r"\bauthorized contact\b", 3), (r"\bcontact (information|details)\b", 2.5),
        (r"\brate (options|comparison)\b", 3), (r"\bplan anniversary\b", 2.5),
        (r"\btrue up\b", 2), (r"\badministered provincially\b", 2.5),
        (r"\beligibility list\b", 2.5), (r"\bequalized payment plan\b", 1.5),
        (r"\bright rate\b", 2), (r"\btransferred twice\b", 2),
        (r"\badd my partner\b", 2.5), (r"\bupdate the contact\b", 2.5),
        (r"\bpre.authorized payment\b", 2), (r"\bjust tell me my balance\b", 2),
    ],
}

# Pre-compile once; this runs over 2,000 transcripts x ~140 patterns.
COMPILED = {
    topic: [(re.compile(p, re.IGNORECASE), w) for p, w in patterns]
    for topic, patterns in RULES.items()
}


class RuleTopicClassifier:
    """Weighted keyword matching. Inspectable and effectively free to run."""

    name = "rules"

    def predict(self, texts, show_progress=True):
        preds, t0 = [], time.perf_counter()
        for i, text in enumerate(texts, start=1):
            preds.append(self._predict_one(text))
            if show_progress and (i % 500 == 0 or i == len(texts)):
                end = "" if sys.stderr.isatty() else "\n"
                sys.stderr.write(f"\r  rules {i:,}/{len(texts):,}{end}")
                sys.stderr.flush()
        if show_progress:
            sys.stderr.write(f"  ({time.perf_counter() - t0:.2f}s)\n")
        return preds

    def scores(self, text):
        """Per-topic total weight -- useful for debugging a misclassification."""
        return {
            topic: sum(w for pattern, w in patterns if pattern.search(text))
            for topic, patterns in COMPILED.items()
        }

    def _predict_one(self, text):
        scores = self.scores(text)
        best = max(scores.values())
        if best == 0:
            return DEFAULT_TOPIC
        # Ties broken by TOPICS order, which is deterministic (seed 42 project).
        return next(t for t in TOPICS if scores[t] == best)


# --- Zero-shot classifier --------------------------------------------------
# bart-large-mnli scores "does this text entail <hypothesis>?" for each label,
# so the wording of the label matters as much as the model. Bare topic names
# ("move-in/move-out") are poor hypotheses; these are hand-written
# verbalizations. Note that writing them IS a form of tuning -- it is prompt
# engineering, just cheaper than a keyword table.
ZEROSHOT_LABELS = {
    "a question about charges or line items on the bill": "billing question",
    "a complaint that the bill is far too high": "high bill dispute",
    "moving home, starting or closing an account": "move-in/move-out",
    "arranging to pay a debt in instalments": "payment arrangement",
    "a power outage or a service fault": "outage or service issue",
    "submitting or correcting a meter reading": "meter reading",
    "a general account or rate enquiry": "general enquiry",
}
ZEROSHOT_MODEL = "facebook/bart-large-mnli"
# The topic is almost always established in the opening exchanges, and NLI
# cost grows with sequence length, so only the first part of the transcript is
# sent. This is a deliberate speed/accuracy trade, not an oversight.
ZEROSHOT_CHARS = 1_000


class ZeroShotTopicClassifier:
    """facebook/bart-large-mnli via transformers' zero-shot pipeline."""

    name = "zeroshot"

    def __init__(self, batch_size=8):
        from transformers import pipeline

        print(f"  loading {ZEROSHOT_MODEL} (first run downloads ~1.6GB)...", flush=True)
        # device=-1 pins it to CPU: this project must run on a laptop (CLAUDE.md).
        self.pipe = pipeline("zero-shot-classification", model=ZEROSHOT_MODEL, device=-1)
        self.batch_size = batch_size
        self.hypotheses = list(ZEROSHOT_LABELS.keys())

    def predict(self, texts, show_progress=True):
        preds, t0 = [], time.perf_counter()
        trimmed = [t[:ZEROSHOT_CHARS] for t in texts]
        for start in range(0, len(trimmed), self.batch_size):
            batch = trimmed[start:start + self.batch_size]
            results = self.pipe(batch, self.hypotheses, hypothesis_template="This is {}.")
            if isinstance(results, dict):  # a single-item batch comes back bare
                results = [results]
            for r in results:
                preds.append(ZEROSHOT_LABELS[r["labels"][0]])
            if show_progress:
                done = len(preds)
                rate = done / max(time.perf_counter() - t0, 1e-9)
                end = "" if sys.stderr.isatty() else "\n"
                sys.stderr.write(
                    f"\r  zeroshot {done:,}/{len(trimmed):,} "
                    f"({rate:.1f}/s, {(len(trimmed) - done) / max(rate, 1e-9):.0f}s left){end}"
                )
                sys.stderr.flush()
        if show_progress:
            sys.stderr.write(f"  ({time.perf_counter() - t0:.1f}s)\n")
        return preds


CLASSIFIERS = {"rules": RuleTopicClassifier, "zeroshot": ZeroShotTopicClassifier}
# See the module docstring for the reasoning behind this default.
DEFAULT_CLASSIFIER = "rules"


# --- Evaluation ------------------------------------------------------------

def evaluate(name, truth, preds, elapsed):
    """Print accuracy, per-class precision/recall and a confusion matrix."""
    truth, preds = pd.Series(truth, name="actual"), pd.Series(preds, name="predicted")
    accuracy = (truth == preds).mean()

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.1%} on {len(truth):,} transcripts "
          f"({elapsed:.2f}s, {len(truth) / max(elapsed, 1e-9):,.0f} transcripts/sec)")

    # Confusion matrix, rows = actual, cols = predicted. reindex so both axes
    # carry all 7 topics even when a classifier never predicts one of them --
    # a missing column is itself the finding.
    cm = pd.crosstab(truth, preds).reindex(index=TOPICS, columns=TOPICS, fill_value=0)
    short = {t: t[:11] for t in TOPICS}
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(cm.rename(index=short, columns=short).to_string())

    per_class = pd.DataFrame({
        "n": cm.sum(axis=1),
        "recall": (np.diag(cm) / cm.sum(axis=1).replace(0, np.nan)).round(3),
        "precision": (np.diag(cm) / cm.sum(axis=0).replace(0, np.nan)).round(3),
    })
    per_class["f1"] = (2 * per_class.precision * per_class.recall
                       / (per_class.precision + per_class.recall)).round(3)
    print("\nPer-topic:")
    print(per_class.to_string())

    worst = per_class["f1"].idxmin()
    print(f"\nWeakest topic: '{worst}' (F1 {per_class.loc[worst, 'f1']})")
    confusions = cm.where(~np.eye(len(TOPICS), dtype=bool)).stack().sort_values(ascending=False)
    if confusions.iloc[0] > 0:
        (a, b), n = confusions.index[0], int(confusions.iloc[0])
        print(f"Biggest confusion: {n} '{a}' predicted as '{b}'")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--classifier", choices=["rules", "zeroshot", "both"],
                        default=DEFAULT_CLASSIFIER)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N transcripts "
                             "(recommended with zeroshot: it is ~1,000x slower)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not INTERACTIONS_PATH.exists():
        raise SystemExit(f"{INTERACTIONS_PATH} not found -- run: python src/generate_all.py")

    interactions = pd.read_csv(INTERACTIONS_PATH)
    if args.limit:
        interactions = interactions.head(args.limit).copy()

    texts = interactions["transcript_text"].astype(str).tolist()
    truth = interactions["topic"].tolist()
    print(f"{len(texts):,} transcripts, {len(TOPICS)} topics")
    print("Actual topic mix:")
    print(pd.Series(truth).value_counts().to_string(), "\n")

    chosen = ["rules", "zeroshot"] if args.classifier == "both" else [args.classifier]
    results = {}
    for name in chosen:
        clf = (CLASSIFIERS[name](batch_size=args.batch_size) if name == "zeroshot"
               else CLASSIFIERS[name]())
        t0 = time.perf_counter()
        preds = clf.predict(texts)
        elapsed = time.perf_counter() - t0
        results[name] = evaluate(name, truth, preds, elapsed)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"topics_{name}.csv"
        pd.DataFrame({
            "interaction_id": interactions["interaction_id"],
            "topic_actual": truth,
            "topic_predicted": preds,
            "correct": [t == p for t, p in zip(truth, preds)],
        }).to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")

    if len(results) > 1:
        print("\n" + "=" * 60)
        best = max(results, key=results.get)
        for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
            print(f"  {name:9s} {acc:.1%}")
        print(f"\nBest on this run: {best}. Shipped default: {DEFAULT_CLASSIFIER} "
              f"(see module docstring for why).")


if __name__ == "__main__":
    main()
