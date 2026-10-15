"""Label the sample with an LLM judge -- a SECOND OPINION, not a substitute for
human labels.

Reads data/labels/to_label.csv (blind by construction -- no true_sentiment, no
model prediction, no topic) and writes data/labels/ai_labels.csv.

WHY THIS IS A SEPARATE FILE FROM human_labels.csv:
  The human label set exists to be an INDEPENDENT yardstick for the sentiment
  model. An LLM judge is not independent in the same way -- it is another
  language model reading the same text, prone to broadly the same failure
  modes (and the transcripts were themselves template-generated, which both
  models may read similarly). Writing these into human_labels.csv would give
  you a file that LOOKS like human ground truth and silently is not, which is
  worse than having no labels at all.

  So the two never mix. Keep whatever human labels you have; this file adds a
  third column to the comparison, and the analysis reports all three
  separately with their counts, so nobody can mistake one for another.

WHAT IT IS GOOD FOR:
  * Covering all 100 rows cheaply, so you can see whether RoBERTa's negative
    skew shows up against a second reader as well as against a human.
  * Flagging the rows where the two models disagree -- those are the ones most
    worth a human minute, if you only have a few to spare.

The judge is llama3.1:8b via local Ollama: a different architecture and
training set from the RoBERTa scorer, so its errors are at least not the same
errors. It sees only the transcript. Temperature 0 and seed 42 for
reproducibility (CLAUDE.md).

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/ai_label.py
    python src/ai_label.py --limit 10
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SEED = 42

TO_LABEL_PATH = Path("data/labels/to_label.csv")
OUTPUT_PATH = Path("data/labels/ai_labels.csv")
HUMAN_PATH = Path("data/labels/human_labels.csv")

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.1:8b"
TIMEOUT = 90

LABELS = ("positive", "neutral", "negative")

# Deliberately mirrors the question the Streamlit app puts to a human, word for
# word, so the two label sets are answering the SAME question. If the prompt
# asked something subtly different ("was this resolved?") the comparison
# between them would be meaningless.
SYSTEM_PROMPT = """You read customer service transcripts from a utility company contact centre and judge how the CUSTOMER felt about the contact overall.

Answer with exactly one word, lowercase, nothing else:
  positive  - the customer ended up satisfied, grateful, or pleased
  neutral   - routine; the customer was neither notably happy nor unhappy
  negative  - the customer was frustrated, angry, or left dissatisfied

Judge the CUSTOMER's feeling, not the agent's politeness. Agents are trained to
sound warm regardless, so their tone tells you little.

Weigh the whole conversation. A customer who opens angry but is genuinely
satisfied by the end is positive. A closing "thanks" alone is just manners, not
satisfaction.

Output only the single word."""


def ask(transcript, url, model):
    """Return (label, error). label is None when the call or parse failed."""
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": f"Transcript:\n\n{transcript}\n\nOne word:",
        "stream": False,
        "options": {"temperature": 0, "seed": SEED, "num_predict": 8},
    }
    try:
        r = requests.post(f"{url}/api/generate", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        raw = r.json().get("response", "")
    except Exception as exc:
        return None, f"request_failed:{type(exc).__name__}"

    # Models like to pad a one-word answer. Take the first label word present.
    text = raw.strip().lower()
    for label in LABELS:
        if re.search(rf"\b{label}\b", text):
            return label, ""
    return None, f"unparseable:{text[:30]!r}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--url", default=OLLAMA_URL)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if not TO_LABEL_PATH.exists():
        raise SystemExit(f"{TO_LABEL_PATH} not found -- run src/make_label_sample.py")

    queue = pd.read_csv(TO_LABEL_PATH, keep_default_na=False)
    if args.limit:
        queue = queue.head(args.limit).copy()

    # Resume: never redo work already on disk.
    done = {}
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        done = dict(zip(existing["interaction_id"], existing["ai_label"]))
        print(f"Resuming: {len(done)} already labelled")

    todo = [r for r in queue.itertuples() if r.interaction_id not in done]
    print(f"Judging {len(todo)} transcripts with {args.model} "
          f"(blind -- transcript text only)\n")

    rows, failures, t0 = [], {}, time.perf_counter()
    for i, row in enumerate(todo, start=1):
        label, error = ask(row.transcript_text, args.url, args.model)
        if error:
            failures[error.split(":")[0]] = failures.get(error.split(":")[0], 0) + 1
        rows.append({
            "interaction_id": row.interaction_id,
            "ai_label": label or "",          # blank, never guessed
            "labelled_by": f"ollama:{args.model}" if label else "failed",
            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        # Write after every row: this takes minutes and should survive a Ctrl-C.
        pd.DataFrame(rows[-1:]).to_csv(
            OUTPUT_PATH, mode="a", header=not OUTPUT_PATH.exists(), index=False)

        if i % 10 == 0 or i == len(todo):
            rate = i / max(time.perf_counter() - t0, 1e-9)
            sys.stderr.write(f"  {i}/{len(todo)} ({rate:.2f}/s, "
                             f"{(len(todo)-i)/max(rate,1e-9):.0f}s left)\n")
            sys.stderr.flush()

    labelled = pd.read_csv(OUTPUT_PATH).drop_duplicates("interaction_id", keep="last")
    print(f"\nWrote {len(labelled)} labels to {OUTPUT_PATH} "
          f"in {time.perf_counter()-t0:.0f}s")
    print(f"\nLabel mix:\n{labelled['ai_label'].value_counts().to_string()}")
    if failures:
        print(f"\nFailures: {failures}")

    if HUMAN_PATH.exists():
        human = pd.read_csv(HUMAN_PATH).drop_duplicates("interaction_id", keep="last")
        print(f"\nNote: {len(human)} human labels exist in {HUMAN_PATH}. "
              f"They are untouched -- run src/compare_labels.py to see all "
              f"three label sources side by side.")


if __name__ == "__main__":
    main()
