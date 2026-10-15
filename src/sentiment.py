"""Score the sentiment of synthetic contact centre transcripts.

Two scorers, deliberately kept on the same output schema so they can be
compared head to head:

  vader    -- VADER lexicon baseline. Instant, no model download, no training.
              The honest "do we even need a transformer?" control.
  roberta  -- cardiffnlp/twitter-roberta-base-sentiment-latest. A 3-class
              sentiment model fine-tuned on ~124M tweets. CPU-friendly
              (~125M params, no GPU needed) and handles the informal,
              typo-laden register of chat transcripts far better than a
              lexicon does.

Only CUSTOMER turns are scored. The agent's lines are scripted courtesy
("I'm sorry to hear that", "thank you for your patience") and scoring them
drags every transcript towards a polite neutral, washing out the signal we
actually want: how the customer felt.

Each transcript yields four columns:

  sentiment_label  -- label over ALL customer turns
  sentiment_score  -- P(positive) - P(negative) over all customer turns, in
                      [-1, 1]. Same definition for both scorers, so the two
                      are directly comparable.
  sentiment_start  -- same score over the FIRST third of customer turns
  sentiment_end    -- same score over the LAST third

start/end exist to catch the sentiment ARC. generate_transcripts.py plants
mixed conversations (angry opening, resolved ending), and a single average
hides them. end - start recovers the recovery.

Long text is chunked rather than truncated. Text over the model's 512-token
limit is split ON SENTENCE BOUNDARIES (which, since customer turns are joined
with a space, means on turn boundaries), each chunk is scored, and the
probabilities are averaged weighted by how many tokens each chunk contributed.
On this dataset chunking rarely fires -- customer turns run ~200 tokens -- but
truncation would silently drop the ending, which is exactly the part
sentiment_end depends on.

Results are cached by text hash in data/processed/, so a rerun that only
changes a few transcripts rescores only those.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/sentiment.py --scorer vader
    python src/sentiment.py --scorer roberta
    python src/sentiment.py --scorer both --limit 200
"""

import argparse
import hashlib
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- Reproducibility -------------------------------------------------------
# Inference is deterministic, but seeds are set everywhere in this project.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INTERACTIONS_PATH = Path("data/raw/interactions.csv")
CACHE_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/labels")

# Bump when scoring logic changes, so stale cache entries are ignored instead
# of silently mixing old and new numbers.
CACHE_VERSION = "v2"  # v2: sentence-boundary chunking, id-space batching

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# VADER's documented decision rule is a +/-0.05 band on `compound`, but that
# guidance is for SENTENCE-length text. `compound` is a sum-of-valences pushed
# through a norm, so it saturates towards +/-1 as text gets longer -- and here
# we hand it every customer turn in a call concatenated together. Measured on
# this dataset, truly negative transcripts still average compound +0.23,
# because polite filler ("thanks", "please", "appreciate") outnumbers the
# complaint words even in a complaint: 8.6% positive tokens vs 4.9% negative.
#
# So the band is recentred for document-length input. 0.95 was checked against
# true_sentiment (50.3% 3-class agreement, vs 43.6% for always guessing the
# majority class, vs 33.1% using the textbook 0.05 band). Retune if the
# transcript generator's register changes.
VADER_COMPOUND_BAND = 0.95

LABELS = ("negative", "neutral", "positive")


# --- Small helpers ---------------------------------------------------------

def progress(total, label, width=28):
    """Return an update(n) closure that draws an in-place progress bar.

    Hand-rolled rather than using tqdm: requirements.txt doesn't list it, and
    the VADER path should not need transformers installed just to show a bar.
    """
    state = {"done": 0, "t0": time.perf_counter(), "next_milestone": 0}
    # A \r-redrawn bar only works on a terminal. Piped into a file or a log it
    # leaves one line per tick, so fall back to occasional milestone lines.
    live = sys.stderr.isatty()
    milestone = max(1, total // 10)

    def update(n=1):
        state["done"] += n
        done, frac = state["done"], (state["done"] / total if total else 1.0)
        elapsed = time.perf_counter() - state["t0"]
        eta = (elapsed / frac - elapsed) if frac > 0 else 0.0

        if live:
            filled = int(width * frac)
            sys.stderr.write(
                f"\r  {label} [{'#' * filled}{'.' * (width - filled)}] "
                f"{done:>6,}/{total:,}  {elapsed:5.1f}s elapsed, {eta:5.1f}s left"
            )
            sys.stderr.flush()
            if done >= total:
                sys.stderr.write("\n")
                sys.stderr.flush()
        elif done >= state["next_milestone"] or done >= total:
            state["next_milestone"] = done + milestone
            sys.stderr.write(f"  {label} {done:>6,}/{total:,} ({frac:4.0%}) "
                             f"{elapsed:5.1f}s elapsed, {eta:5.1f}s left\n")
            sys.stderr.flush()

    return update


def text_key(text):
    """Stable cache key for one piece of text."""
    return hashlib.md5(f"{CACHE_VERSION}|{text}".encode("utf-8")).hexdigest()


# --- Transcript parsing ----------------------------------------------------

def customer_turns(transcript):
    """Pull out just the customer utterances, prefix stripped.

    Transcripts are line-per-turn, each line starting "Agent:" or "Customer:"
    (check_data.py asserts this holds for every row).
    """
    turns = []
    for line in str(transcript).split("\n"):
        line = line.strip()
        if line.startswith("Customer:"):
            utterance = line[len("Customer:"):].strip()
            if utterance:
                turns.append(utterance)
    return turns


def thirds(turns):
    """Split customer turns into (first third, last third).

    Uses ceil(n/3) so short conversations still give each window something to
    work with. The two windows never overlap for n >= 2; for a single-turn
    conversation both windows are that one turn, which is the only honest
    answer available.
    """
    if not turns:
        return [], []
    size = max(1, math.ceil(len(turns) / 3))
    return turns[:size], turns[-size:]


def transcript_windows(transcript):
    """Return the three texts to score for one transcript: full, start, end.

    Turns are joined with a space: these are separate utterances from one
    speaker, not a dialogue, so no speaker labels are needed in the input.
    """
    turns = customer_turns(transcript)
    start, end = thirds(turns)
    return {
        "full": " ".join(turns),
        "start": " ".join(start),
        "end": " ".join(end),
    }


# --- Scorer base class -----------------------------------------------------

class Scorer:
    """Shared caching / assembly logic. Subclasses implement _score_many()."""

    name = "base"

    def __init__(self, use_cache=True, batch_size=32):
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.cache_path = CACHE_DIR / f"sentiment_cache_{self.name}.csv"
        self._cache = self._load_cache() if use_cache else {}

    # -- cache ------------------------------------------------------------
    def _load_cache(self):
        if not self.cache_path.exists():
            return {}
        cached = pd.read_csv(self.cache_path)
        return {
            row.key: (row.label, float(row.score))
            for row in cached.itertuples()
        }

    def _save_cache(self):
        if not self.use_cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"key": k, "label": v[0], "score": v[1]} for k, v in self._cache.items()]
        ).to_csv(self.cache_path, index=False)

    # -- scoring ----------------------------------------------------------
    def _score_many(self, texts):
        """Score a list of texts. Returns [(label, score), ...]. Subclass job."""
        raise NotImplementedError

    def score_texts(self, texts):
        """Score texts, reusing cached results and scoring only what's new.

        Duplicate texts (common for the short start/end windows) collapse to a
        single model call.
        """
        results = {}
        todo = []
        for text in texts:
            key = text_key(text)
            if key in results or key in self._cache:
                continue
            todo.append(text)
            results[key] = None  # placeholder: marks it as already queued

        if todo:
            print(f"  {self.name}: {len(todo):,} new texts to score "
                  f"({len(texts) - len(todo):,} served from cache/duplicates)", flush=True)
            for text, outcome in zip(todo, self._score_many(todo)):
                self._cache[text_key(text)] = outcome
        else:
            print(f"  {self.name}: all {len(texts):,} texts already cached", flush=True)

        return [self._cache[text_key(t)] for t in texts]

    def score_interactions(self, interactions):
        """Score a whole interactions frame. Returns the output DataFrame."""
        windows = [transcript_windows(t) for t in interactions["transcript_text"]]

        # One flat list so the model batches across every window at once rather
        # than making three smaller passes.
        flat = [w[part] for w in windows for part in ("full", "start", "end")]
        scored = self.score_texts(flat)

        rows = []
        for i, interaction_id in enumerate(interactions["interaction_id"]):
            full, start, end = scored[i * 3: i * 3 + 3]
            rows.append({
                "interaction_id": interaction_id,
                "sentiment_label": full[0],
                "sentiment_score": round(full[1], 4),
                "sentiment_start": round(start[1], 4),
                "sentiment_end": round(end[1], 4),
            })

        self._save_cache()
        return pd.DataFrame(rows)


# --- VADER baseline --------------------------------------------------------

class VaderScorer(Scorer):
    """Lexicon baseline. No model weights, no download, effectively instant."""

    name = "vader"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()

    def _score_many(self, texts):
        # VADER is pure Python string matching -- there is nothing to batch, so
        # the "batch" here is only about how often the progress bar ticks.
        tick = progress(len(texts), "vader  ")
        out, since_tick = [], 0
        for text in texts:
            if not text.strip():
                out.append(("neutral", 0.0))
            else:
                s = self.analyzer.polarity_scores(text)
                # The SCORE is pos - neg, matching the transformer's definition
                # so the two scorers share one axis. The LABEL comes from
                # `compound`, which is what VADER actually calibrates its
                # decision rule on -- pos - neg ranks transcripts slightly
                # better (r=0.66 vs 0.59 against the true label) but has no
                # meaningful cut points, since even negative calls sit above
                # zero on it.
                score = float(s["pos"] - s["neg"])
                compound = float(s["compound"])
                if compound > VADER_COMPOUND_BAND:
                    label = "positive"
                elif compound < -VADER_COMPOUND_BAND:
                    label = "negative"
                else:
                    label = "neutral"
                out.append((label, score))
            since_tick += 1
            if since_tick == self.batch_size:
                tick(since_tick)
                since_tick = 0
        if since_tick:
            tick(since_tick)
        return out


# --- RoBERTa transformer ---------------------------------------------------

class RobertaScorer(Scorer):
    """cardiffnlp/twitter-roberta-base-sentiment-latest on CPU."""

    name = "roberta"

    def __init__(self, model_name=ROBERTA_MODEL, **kwargs):
        super().__init__(**kwargs)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        torch.manual_seed(SEED)

        print(f"  loading {model_name} (first run downloads ~500MB)...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # inference only: disables dropout
        # CPU is the only target (see CLAUDE.md) -- no .to("cuda") path here.

        # Read the label order off the config instead of assuming it. This model
        # ships {0: negative, 1: neutral, 2: positive}, but that is not universal.
        id2label = {i: str(l).lower() for i, l in self.model.config.id2label.items()}
        self.label_of = id2label
        self.neg_idx = next(i for i, l in id2label.items() if "neg" in l)
        self.pos_idx = next(i for i, l in id2label.items() if "pos" in l)

        # Some tokenizer configs ship a sentinel (1e30) rather than a real
        # limit; falling back to 512 keeps chunking active instead of silently
        # sending an over-long sequence into the model.
        reported = self.tokenizer.model_max_length
        self.max_length = 512 if reported is None or reported > 4096 else int(reported)

        # Work out what the tokenizer wraps a sequence in by encoding nothing:
        # for RoBERTa that is [<s>, </s>]. Derived rather than hardcoded as -2,
        # because the overhead differs by model -- and transformers 5 removed
        # the tokenizer helper that used to do this, so we add them ourselves.
        wrapper = self.tokenizer("", add_special_tokens=True)["input_ids"]
        self.prefix_ids = wrapper[:1] if wrapper else []
        self.suffix_ids = wrapper[-1:] if len(wrapper) > 1 else []
        self.max_content = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)

    def _wrap(self, ids):
        """Add the model's special tokens around a chunk of content ids."""
        return self.prefix_ids + list(ids) + self.suffix_ids

    # Sentence boundary: . ! or ? followed by whitespace. Customer turns are
    # joined with a space and each one ends in punctuation, so in practice this
    # splits on TURN boundaries -- the natural unit here.
    SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def _segments(self, text):
        """[(sentence, token_ids)] in order. One batched tokenizer call."""
        parts = [p for p in self.SENTENCE_SPLIT.split(text) if p.strip()]
        if not parts:
            parts = [text]
        encoded = self.tokenizer(parts, add_special_tokens=False)["input_ids"]
        return list(zip(parts, encoded))

    def _chunk_ids(self, text):
        """Pack text into <= max_content token chunks, split on sentences.

        Returns [(input_ids_including_specials, n_content_tokens), ...].

        Three choices worth stating, because the naive version of this is
        subtly wrong in all three:

        1. Chunks break on SENTENCE boundaries, not every 510 tokens. A blind
           token window cuts mid-clause -- "I am absolutely furious about this"
           / "bill and nobody has helped me" -- and the two halves do not score
           like the whole. Splitting where the speaker stopped keeps each chunk
           a complete thought.
        2. Everything stays in TOKEN-ID space. The previous version decoded ids
           back to text and re-encoded them in the batch, which is lossy
           (whitespace and unusual characters do not always survive the round
           trip) and costs a second tokenization of the entire corpus.
        3. A single sentence longer than the window is hard-split rather than
           dropped or truncated. Rare in dialogue, but silent data loss is the
           worst possible failure here.

        No overlap/stride between chunks: that matters when you cut mid-thought,
        and we do not. It would also double-count the overlapped tokens in the
        weighted average below.
        """
        chunks, current, current_len = [], [], 0

        def flush():
            nonlocal current, current_len
            if current:
                chunks.append(
                    (self._wrap(current), current_len)
                )
                current, current_len = [], 0

        for _, ids in self._segments(text):
            if len(ids) > self.max_content:
                flush()  # oversized single sentence: emit it on its own
                for i in range(0, len(ids), self.max_content):
                    piece = ids[i:i + self.max_content]
                    chunks.append(
                        (self._wrap(piece), len(piece))
                    )
                continue
            if current_len + len(ids) > self.max_content:
                flush()
            current.extend(ids)
            current_len += len(ids)
        flush()

        if not chunks:  # whitespace-only text
            chunks = [(self._wrap([]), 1)]
        return chunks

    def _score_many(self, texts):
        torch = self.torch
        pad_id = self.tokenizer.pad_token_id or 0

        # Expand every text into its chunks, remembering the parent so the
        # probabilities can be recombined afterwards.
        chunk_ids, chunk_parent, chunk_weight = [], [], []
        for parent, text in enumerate(texts):
            if not text.strip():
                continue  # empty window -> neutral, handled below
            for ids, weight in self._chunk_ids(text):
                chunk_ids.append(ids)
                chunk_parent.append(parent)
                chunk_weight.append(weight)

        n_nonempty = sum(1 for t in texts if t.strip())
        extra = len(chunk_ids) - n_nonempty
        if extra > 0:
            over = sum(1 for p in set(chunk_parent)
                       if chunk_parent.count(p) > 1)
            print(f"  {over} text(s) exceeded {self.max_content} tokens -> "
                  f"{extra} extra chunk(s)", flush=True)

        # Sort by length so each batch pads to a similar width. Padding short
        # sequences up to the longest in the batch is wasted compute, and on
        # CPU that waste is the difference between minutes and tens of minutes.
        order = sorted(range(len(chunk_ids)), key=lambda i: len(chunk_ids[i]))

        probs = np.zeros((len(chunk_ids), self.model.config.num_labels))
        tick = progress(len(chunk_ids), "roberta")

        with torch.no_grad():  # inference only: no gradient graph, big memory win
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                width = max(len(chunk_ids[i]) for i in idx)

                # Build the padded batch directly from ids -- no decode/encode
                # round trip, so what the model sees is exactly what _chunk_ids
                # produced.
                input_ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
                attention = torch.zeros((len(idx), width), dtype=torch.long)
                for row, i in enumerate(idx):
                    ids = chunk_ids[i]
                    input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attention[row, :len(ids)] = 1

                logits = self.model(input_ids=input_ids, attention_mask=attention).logits
                probs[idx] = torch.softmax(logits, dim=-1).numpy()
                tick(len(idx))

        # Recombine chunks into one distribution per text, weighted by how many
        # tokens each chunk actually contributed (a 500-token chunk should not
        # count the same as a 12-token tail).
        by_parent = {}
        for i, parent in enumerate(chunk_parent):
            by_parent.setdefault(parent, []).append(i)

        out = []
        for parent in range(len(texts)):
            rows = by_parent.get(parent)
            if not rows:
                out.append(("neutral", 0.0))  # empty window
                continue
            mean = np.average(probs[rows], axis=0,
                              weights=[chunk_weight[i] for i in rows])
            label = self.label_of[int(mean.argmax())]
            score = float(mean[self.pos_idx] - mean[self.neg_idx])
            out.append((label, score))
        return out


SCORERS = {"vader": VaderScorer, "roberta": RobertaScorer}


# --- CLI -------------------------------------------------------------------

def summarise(scored, interactions, scorer_name):
    """Print how the scores line up against the hidden true_sentiment."""
    df = scored.merge(
        interactions[["interaction_id", "true_sentiment", "topic", "channel"]],
        on="interaction_id",
    )
    print(f"\n--- {scorer_name} vs true_sentiment ---")
    print("Mean sentiment_score by true label:")
    print(df.groupby("true_sentiment")["sentiment_score"].agg(["mean", "size"]).round(3).to_string())

    print("\nConfusion (rows = true, cols = predicted):")
    print(pd.crosstab(df["true_sentiment"], df["sentiment_label"]).to_string())

    agree = (df["true_sentiment"] == df["sentiment_label"]).mean()
    print(f"\n3-class agreement with true_sentiment: {agree:.1%}")

    # The arc: how much the conversation moved from start to end.
    df["delta"] = df["sentiment_end"] - df["sentiment_start"]
    print("\nMean start -> end movement by true label:")
    print(df.groupby("true_sentiment")[["sentiment_start", "sentiment_end", "delta"]]
          .mean().round(3).to_string())
    recovered = (df["delta"] > 0.4).sum()
    print(f"\nTranscripts scored as recovering (end - start > 0.4): {recovered} "
          f"({recovered / len(df):.1%})")
    print("  Note: closing pleasantries ('thanks, all good') lift the end window on"
          "\n  their own, so read this against the other scorer rather than alone.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scorer", choices=["vader", "roberta", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N interactions (for a quick trial)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore and overwrite the cache")
    args = parser.parse_args()

    if not INTERACTIONS_PATH.exists():
        raise SystemExit(f"{INTERACTIONS_PATH} not found -- run: python src/generate_all.py")

    interactions = pd.read_csv(INTERACTIONS_PATH)
    if args.limit:
        interactions = interactions.head(args.limit).copy()

    turn_counts = [len(customer_turns(t)) for t in interactions["transcript_text"]]
    print(f"{len(interactions):,} interactions, "
          f"{sum(turn_counts):,} customer turns "
          f"(mean {np.mean(turn_counts):.1f} per transcript)\n")

    chosen = ["vader", "roberta"] if args.scorer == "both" else [args.scorer]
    for name in chosen:
        print(f"=== {name} ===", flush=True)
        scorer = SCORERS[name](use_cache=not args.no_cache, batch_size=args.batch_size)
        scored = scorer.score_interactions(interactions)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"sentiment_{name}.csv"
        scored.to_csv(out_path, index=False)
        print(f"  wrote {len(scored):,} rows to {out_path}")

        summarise(scored, interactions, name)
        print()


if __name__ == "__main__":
    main()
