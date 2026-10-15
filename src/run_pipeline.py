"""End-to-end scoring: interactions.csv -> data/processed/interaction_scores.csv.

Runs the three scoring stages over one set of interactions and writes a single
joined table, one row per interaction:

    interaction_id, sentiment_label, sentiment_score, sentiment_start,
    sentiment_end, topic_predicted, coaching_note, generated_by,
    model_version, scored_at

Stage order matters -- coaching consumes the sentiment scores, so sentiment
runs first:

    1. src/sentiment.py   -> sentiment_label / _score / _start / _end
    2. src/topics.py      -> topic_predicted
    3. src/coaching.py    -> coaching_note + generated_by

This is the SCORING pipeline. It is the counterpart to src/generate_all.py,
which builds the synthetic raw data; this reads that data and models it. Unlike
generate_all.py, the stages are imported and called in-process rather than
shelled out: they are classes with a shared interface, and keeping one process
means the RoBERTa weights are loaded once instead of once per stage.

PROVENANCE COLUMNS:
  model_version records which models actually produced each row, so a scores
  file is self-describing -- six months from now you can tell a VADER run from
  a RoBERTa one without guessing.
  scored_at is a UTC timestamp. Note that it makes the output NOT byte-identical
  between runs, which is a deliberate exception to this project's
  reproducibility rule (CLAUDE.md): the scores themselves are reproducible, the
  timestamp recording when they were produced is not meant to be.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/run_pipeline.py                      # everything, best models
    python src/run_pipeline.py --sample 50          # quick trial
    python src/run_pipeline.py --sentiment vader --coaching rules
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# The stage modules live alongside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import coaching  # noqa: E402
import sentiment  # noqa: E402
import topics  # noqa: E402

SEED = 42

INTERACTIONS_PATH = Path("data/raw/interactions.csv")
AGENTS_PATH = Path("data/raw/agents.csv")
OUTPUT_PATH = Path("data/processed/interaction_scores.csv")

# Exact column order of the output contract. tests/ asserts against this.
OUTPUT_COLUMNS = [
    "interaction_id",
    "sentiment_label",
    "sentiment_score",
    "sentiment_start",
    "sentiment_end",
    "topic_predicted",
    "coaching_note",
    "generated_by",
    "model_version",
    "scored_at",
]


def run_sentiment(interactions, which, batch_size, use_cache):
    """Stage 1. Returns a frame keyed by interaction_id."""
    scorer = sentiment.SCORERS[which](use_cache=use_cache, batch_size=batch_size)
    scored = scorer.score_interactions(interactions)
    version = (f"roberta:{sentiment.ROBERTA_MODEL}" if which == "roberta"
               else f"vader:compound_band={sentiment.VADER_COMPOUND_BAND}")
    return scored, version


def run_topics(interactions, which, batch_size):
    """Stage 2. Returns a Series of predicted topics aligned to interactions."""
    classifier = (topics.ZeroShotTopicClassifier(batch_size=batch_size)
                  if which == "zeroshot" else topics.RuleTopicClassifier())
    preds = classifier.predict(interactions["transcript_text"].astype(str).tolist())
    version = (f"zeroshot:{topics.ZEROSHOT_MODEL}" if which == "zeroshot"
               else "topics:rules")
    return pd.Series(preds, index=interactions.index), version


def run_coaching(interactions, agents, scored_sentiment, which, ollama_url, use_cache):
    """Stage 3. Returns a frame with coaching_note + generated_by."""
    use_ollama = False
    if which in ("auto", "ollama"):
        up, message = coaching.OllamaGenerator.available(ollama_url)
        print(f"  {message}")
        if up:
            use_ollama = True
        elif which == "ollama":
            raise SystemExit("--coaching ollama requested but Ollama is unavailable.")

    generator = coaching.OllamaGenerator(ollama_url) if use_ollama else None

    # coaching.build_facts expects the sentiment columns joined on, which is
    # exactly what stage 1 produced -- so no re-reading from disk.
    facts = coaching.build_facts(interactions, agents, scored_sentiment)
    cache = coaching.load_cache() if use_cache else {}
    notes, fallbacks = coaching.generate_notes(
        facts, generator, cache, flush_cache=use_cache and generator is not None)
    if use_cache:
        coaching.save_cache(cache)

    if fallbacks:
        print("\n  rows that fell back to templates:")
        for reason, n in sorted(fallbacks.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>5,}  {reason}")

    version = generator.name if use_ollama else "coaching:rules"
    return notes[["interaction_id", "coaching_note", "generated_by"]], version


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="score only the first N interactions (quick runs)")
    parser.add_argument("--sentiment", choices=["roberta", "vader"], default="roberta")
    parser.add_argument("--topic", choices=["rules", "zeroshot"], default="rules")
    parser.add_argument("--coaching", choices=["auto", "ollama", "rules"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--ollama-url", default=coaching.OLLAMA_URL)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    for path in (INTERACTIONS_PATH, AGENTS_PATH):
        if not path.exists():
            raise SystemExit(f"{path} not found -- run: python src/generate_all.py")

    interactions = pd.read_csv(INTERACTIONS_PATH)
    agents = pd.read_csv(AGENTS_PATH)
    if args.sample:
        interactions = interactions.head(args.sample).copy()

    use_cache = not args.no_cache
    started = time.perf_counter()
    print(f"Scoring {len(interactions):,} interactions "
          f"(sentiment={args.sentiment}, topic={args.topic}, coaching={args.coaching})\n")

    print("[1/3] sentiment")
    scored_sentiment, sentiment_version = run_sentiment(
        interactions, args.sentiment, args.batch_size, use_cache)

    print("\n[2/3] topics")
    topic_preds, topic_version = run_topics(interactions, args.topic, args.batch_size)

    print("\n[3/3] coaching")
    notes, coaching_version = run_coaching(
        interactions, agents, scored_sentiment, args.coaching,
        args.ollama_url, use_cache)

    # --- Join -------------------------------------------------------------
    scores = interactions[["interaction_id"]].copy()
    scores["topic_predicted"] = topic_preds.values
    scores = scores.merge(scored_sentiment, on="interaction_id", how="left")
    scores = scores.merge(notes, on="interaction_id", how="left")

    scores["model_version"] = "; ".join([sentiment_version, topic_version, coaching_version])
    scores["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    scores = scores[OUTPUT_COLUMNS]

    # --- Guard rails before writing ---------------------------------------
    # These mirror tests/ but run on every pipeline execution: a scores file
    # with missing rows or out-of-range values should never reach disk.
    missing = set(interactions["interaction_id"]) - set(scores["interaction_id"])
    if missing:
        raise SystemExit(f"{len(missing)} interactions lost during scoring, e.g. "
                         f"{sorted(missing)[:3]}")
    if scores["interaction_id"].duplicated().any():
        raise SystemExit("duplicate interaction_id in the scores table")
    for col in ("sentiment_score", "sentiment_start", "sentiment_end"):
        bad = scores[~scores[col].between(-1, 1)]
        if len(bad):
            raise SystemExit(f"{len(bad)} rows have {col} outside [-1, 1]")
    unknown = set(scores["topic_predicted"]) - set(topics.TOPICS)
    if unknown:
        raise SystemExit(f"unknown topic label(s) predicted: {unknown}")
    if scores[OUTPUT_COLUMNS].isna().any().any():
        nulls = scores[OUTPUT_COLUMNS].isna().sum()
        raise SystemExit(f"null values in the scores table:\n{nulls[nulls > 0]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out, index=False)

    # --- Summary ----------------------------------------------------------
    elapsed = time.perf_counter() - started
    print(f"\nWrote {len(scores):,} rows to {args.out} in {elapsed:.1f}s "
          f"({len(scores) / elapsed:.1f} interactions/sec)")
    print(f"model_version: {scores['model_version'].iloc[0]}")
    print(f"scored_at:     {scores['scored_at'].iloc[0]}\n")

    print("sentiment_label:")
    print(scores["sentiment_label"].value_counts().to_string())
    print("\ntopic_predicted:")
    print(scores["topic_predicted"].value_counts().to_string())
    print("\ngenerated_by:")
    print(scores["generated_by"].value_counts().to_string())

    # If the raw file still carries the hidden labels, report agreement -- it
    # is the quickest signal that a stage has silently regressed.
    if "true_sentiment" in interactions.columns:
        merged = scores.merge(interactions[["interaction_id", "true_sentiment", "topic"]],
                              on="interaction_id")
        s_acc = (merged["sentiment_label"] == merged["true_sentiment"]).mean()
        t_acc = (merged["topic_predicted"] == merged["topic"]).mean()
        print(f"\nAgainst the hidden labels: sentiment {s_acc:.1%}, topic {t_acc:.1%}")


if __name__ == "__main__":
    main()
