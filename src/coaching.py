"""Write a short coaching note for every interaction.

One note per contact, 1-2 sentences: ONE thing that went well, then ONE
specific thing to try next time. Built from five signals we already have --
sentiment, how sentiment MOVED across the call, topic, hold time, and handle
time against that agent's own AHT target.

Two generators:

  ollama  -- llama3.1:8b via a local Ollama server, if one is listening on
             http://localhost:11434 and has the model pulled. Produces more
             natural, less repetitive notes.
  rules   -- a template engine. Always available, deterministic, and the
             automatic fallback whenever Ollama is missing, slow, or returns
             something that fails the safety check below.

Every row records which one produced it in `generated_by`, so a reviewer can
tell a model-written note from a templated one at a glance.

ON TONE -- the thing that matters most here:
  A coaching note attached to individual agents is a performance artefact, and
  it is trivially easy to produce something that reads as a reprimand. Both
  generators are constrained to be developmental, never disciplinary. For the
  LLM that is a prompt instruction AND a post-hoc filter (BANNED_PHRASES): any
  note containing disciplinary or absolute language is thrown away and the
  rule-based note is used instead, rather than being shown to anyone. The
  prompt cannot be trusted on its own, so the filter is the actual guarantee.

  These notes are coaching prompts for a conversation between a team lead and
  an agent. They are not evidence, not a ranking, and not a substitute for a
  human looking at the contact. All the agents here are synthetic anyway.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/coaching.py                 # auto-detect Ollama, else rules
    python src/coaching.py --generator rules --limit 20
"""

import argparse
import hashlib
import json
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
AGENTS_PATH = Path("data/raw/agents.csv")
LABELS_DIR = Path("data/labels")
OUTPUT_PATH = LABELS_DIR / "coaching_notes.csv"
CACHE_PATH = Path("data/processed/coaching_cache.csv")

# --- Ollama ----------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 60  # seconds per note; 8B on CPU is not fast
OLLAMA_PROBE_TIMEOUT = 3

# --- Thresholds ------------------------------------------------------------
# A hold over 3 minutes is where customers start narrating the wait back to
# you; under 45s is barely noticed.
LONG_HOLD_SEC = 180
SHORT_HOLD_SEC = 45
# AHT within +/-10% of target is "on target" -- tighter than that is noise.
AHT_OVER = 1.25
AHT_ON_TARGET = 1.10
# Sentiment movement that counts as a real turnaround rather than a sign-off.
RECOVERY_SHIFT = 0.50
# A rising score is NOT by itself a recovery: sign-off pleasantries lift the
# end window on almost every call (measured in src/sentiment.py). To claim the
# customer was turned around, the call has to have started genuinely negative.
NEGATIVE_START_FOR_RECOVERY = -0.20
DECLINE_SHIFT = -0.30
POSITIVE_END = 0.30
NEGATIVE_END = -0.30

# Any of these in a generated note means it has drifted into performance
# management. The note is discarded and the templated one used instead.
# Trailing characters that may legitimately follow a sentence's full stop.
CLOSING_MARKS = "\"')"

BANNED_PHRASES = [
    "disciplin", "warning", "write-up", "written up", "reprimand",
    "unacceptable", "failure", "failed to", "poor performance", "underperform",
    "must improve", "needs to improve immediately", "not good enough",
    "escalate to management", "formal", "probation", "consequence",
    "expectations were not met", "fell short", "inadequate", "sub-par",
    "subpar", "concerning", "worrying", "problematic",
]


# --- Fact extraction -------------------------------------------------------

def a_topic(topic):
    """'a billing question' / 'an outage or service issue'.

    Topic names are injected mid-sentence, and three of the seven start with a
    vowel, so a hardcoded "a" in a template reads as a typo.
    """
    return f"{'an' if topic[0].lower() in 'aeiou' else 'a'} {topic}"


def mmss(seconds):
    """Format seconds the way a team lead would say them out loud."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes}m {rest:02d}s" if rest else f"{minutes}m"


def build_facts(interactions, agents, sentiment):
    """Join everything a note needs into one row per interaction."""
    df = interactions.merge(
        agents[["agent_id", "agent_name", "team", "aht_target_sec", "csat_target"]],
        on="agent_id", how="left",
    )
    if sentiment is not None:
        df = df.merge(sentiment[["interaction_id"] + SENTIMENT_COLUMNS],
                      on="interaction_id", how="left")
        # Backfill whatever the scored file did not cover, so no rule ever
        # silently evaluates against NaN.
        missing = df["sentiment_label"].isna()
        if missing.any():
            fallback = df.loc[missing, "true_sentiment"]
            df.loc[missing, "sentiment_label"] = fallback
            df.loc[missing, "sentiment_score"] = fallback.map(
                {"negative": -0.6, "neutral": 0.0, "positive": 0.6})
            df.loc[missing, "sentiment_start"] = df.loc[missing, "sentiment_score"]
            df.loc[missing, "sentiment_end"] = df.loc[missing, "sentiment_score"]
        df.attrs["n_backfilled"] = int(missing.sum())
    else:
        # No scored sentiment on disk: fall back to the generator's hidden
        # label so the module still runs standalone. Flagged in the output.
        df["sentiment_label"] = df["true_sentiment"]
        df["sentiment_score"] = df["true_sentiment"].map(
            {"negative": -0.6, "neutral": 0.0, "positive": 0.6})
        df["sentiment_start"] = df["sentiment_score"]
        df["sentiment_end"] = df["sentiment_score"]
        df.attrs["n_backfilled"] = len(df)

    assert df["sentiment_label"].notna().all(), "sentiment must be fully populated"
    df["shift"] = df["sentiment_end"] - df["sentiment_start"]
    df["aht_ratio"] = df["handle_time_sec"] / df["aht_target_sec"]
    return df


# --- Rule-based note engine ------------------------------------------------
# Each entry is (signal name, test, [templates]). Order is priority: the first
# test that passes wins, so the most specific/strongest signal is the one the
# note is built around. Templates get .format(**facts-ish) applied.

STRENGTHS = [
    ("recovery", lambda f: f["shift"] >= RECOVERY_SHIFT and f["sentiment_end"] > 0
     and f["sentiment_start"] <= NEGATIVE_START_FOR_RECOVERY, [
        "You turned this one around — the customer opened unhappy and was in a noticeably better place by the end.",
        "Really nice recovery here: the tone at the start was rough and you brought it back before the close.",
        "The customer came in frustrated and left settled, which is the hardest version of this call to get right.",
    ]),
    ("tough_topic_held", lambda f: f["topic"] in ("high bill dispute", "payment arrangement")
     and f["sentiment_label"] != "negative", [
        "Handling {a_topic} without the conversation souring is genuinely difficult, and you managed it.",
        "You kept {a_topic} calm and factual, which is exactly what this contact type needs.",
    ]),
    ("ended_positive", lambda f: f["sentiment_end"] >= POSITIVE_END, [
        "The customer sounded genuinely satisfied by the end of this one.",
        "You closed this on a clearly positive note.",
        "Good close — the customer's last few turns were warm.",
    ]),
    ("efficient", lambda f: f["aht_ratio"] <= 1.0 and f["sentiment_label"] != "negative"
     and f["hold_time_sec"] < LONG_HOLD_SEC, [
        "You came in under your AHT target at {handle} without the customer feeling rushed.",
        "Efficient handling here — {handle} against a {target} target, and the tone stayed fine.",
    ]),
    ("no_hold", lambda f: f["hold_time_sec"] <= SHORT_HOLD_SEC, [
        "You kept the customer with you throughout rather than parking them on hold.",
        "Almost no hold time on this one, which customers notice more than we give them credit for.",
    ]),
    ("steady", lambda f: True, [  # catch-all
        "You worked through {a_topic} steadily and got to a resolution.",
        "Solid, straightforward handling of {a_topic}.",
    ]),
]

# Improvement suggestions. Topic-specific where the advice would genuinely
# differ; generic otherwise. Every one is phrased as something to TRY.
IMPROVEMENTS = [
    ("ended_negative_dispute", lambda f: f["sentiment_end"] <= NEGATIVE_END
     and f["topic"] == "high bill dispute", [
        "Next time, try naming the billing review and the verification read in your first response — customers settle faster once they hear a concrete next step, before the explanation of why the bill rose.",
        "One to try: lead with what you can action (review, verification read, due-date extension) and explain the degree-day detail after, rather than before.",
    ]),
    ("ended_negative_payment", lambda f: f["sentiment_end"] <= NEGATIVE_END
     and f["topic"] == "payment arrangement", [
        "Worth trying next time: mention the winter protection window early. People carrying a disconnection notice often don't know it exists, and it changes the whole temperature of the call.",
        "Next time, try confirming the collection activity stops before working through the monthly figure — the reassurance lands better first.",
    ]),
    ("ended_negative", lambda f: f["sentiment_end"] <= NEGATIVE_END, [
        "For next time: before you close {a_topic}, try a short recap of what you've done and what happens next — unresolved-feeling endings are usually missing that, not missing the fix.",
        "One thing to try: when the tone is still tense near the end, ask directly whether anything is still unanswered rather than moving to the close.",
    ]),
    ("declined", lambda f: f["shift"] <= DECLINE_SHIFT, [
        "The mood slipped in the back half — next time try checking in mid-call ('does that make sense so far?') so you catch it while it's still small.",
        "Worth watching: this one started better than it ended. A quick temperature check before the resolution can stop that drift.",
    ]),
    ("long_hold", lambda f: f["hold_time_sec"] >= LONG_HOLD_SEC, [
        "The hold ran to {hold} — next time try setting the expectation up front ('this will take about two minutes') and checking back at the halfway mark.",
        "One to try: at {hold}, that hold was long enough to need a mid-hold check-in. Customers wait more patiently when they hear from you once during it.",
    ]),
    ("over_aht", lambda f: f["aht_ratio"] >= AHT_OVER, [
        "This ran to {handle} against your {target} target — next time try summarising the resolution once at the end rather than re-explaining it as you go.",
        "Something to try: at {aht_pct} over target, there's usually a repeated explanation in there. Picking one clear framing early tends to shorten the whole call.",
    ]),
    ("negative_throughout", lambda f: f["sentiment_label"] == "negative", [
        "Next time on {a_topic} like this, try acknowledging the frustration explicitly once, early — it doesn't fix the bill but it reliably lowers the heat.",
        "One to try: with a customer this unhappy from the opening, naming what you CAN do in your first reply gives them something to hold onto.",
    ]),
    ("polish", lambda f: True, [  # catch-all
        "To build on it: try closing with an explicit 'is there anything else outstanding?' — it catches the second issue customers often mention only when prompted.",
        "One small thing to try next time: confirm the reference number back to the customer at the close so they leave with something concrete.",
        "Next time, try naming the specific follow-up timeframe rather than 'shortly' — it sets expectations and cuts call-backs.",
    ]),
]


def pick(bank, facts, rng):
    """Return (signal_name, rendered_line) for the first matching rule."""
    for signal, test, templates in bank:
        if test(facts):
            template = rng.choice(templates)
            return signal, template.format(
                topic=facts["topic"],
                a_topic=a_topic(facts["topic"]),
                handle=mmss(facts["handle_time_sec"]),
                hold=mmss(facts["hold_time_sec"]),
                target=mmss(facts["aht_target_sec"]),
                aht_pct=f"{abs((facts['aht_ratio'] - 1) * 100):.0f}%",
            )
    raise AssertionError("banks must end with a catch-all rule")


def rule_note(facts):
    """Build one templated note: a strength, then an improvement.

    The RNG is seeded from the interaction id, not drawn from one shared
    stream, so a given interaction gets the same note whether you score 12
    rows or 2,000 -- and reruns stay stable (seed 42, per CLAUDE.md).
    """
    rng = random.Random(f"{SEED}|{facts['interaction_id']}")
    _, strength = pick(STRENGTHS, facts, rng)
    _, improvement = pick(IMPROVEMENTS, facts, rng)
    return f"{strength} {improvement}"


# --- Ollama note generator -------------------------------------------------

SYSTEM_PROMPT = """You are a supportive contact centre team leader writing a short coaching note about ONE customer interaction, addressed to the agent who handled it.

Rules you must follow:
- Exactly 1-2 sentences. Maximum 45 words total.
- Name ONE specific thing that went well, then ONE specific thing to try next time.
- Be warm, concrete and developmental. Write to the agent as "you".
- The "try next time" part must be an actionable technique, not a judgement.
- Use ONLY the facts given. Never invent details about the conversation.
- NEVER use disciplinary or performance-management language. Do not mention
  warnings, discipline, formal processes, consequences, or failure. Do not say
  the agent was bad, wrong, concerning, or unacceptable.
- Do not use headings, bullet points, labels or quotation marks.

Output only the note itself."""


def fact_block(facts):
    """The human-readable fact summary handed to the model."""
    direction = ("improved during the call" if facts["shift"] >= RECOVERY_SHIFT
                 else "got worse during the call" if facts["shift"] <= DECLINE_SHIFT
                 else "stayed roughly level")
    return (
        f"Topic: {facts['topic']}\n"
        f"Channel: {facts['channel']}\n"
        f"Overall customer sentiment: {facts['sentiment_label']} "
        f"(score {facts['sentiment_score']:+.2f} on a -1 to +1 scale)\n"
        f"Sentiment at the start: {facts['sentiment_start']:+.2f}; "
        f"at the end: {facts['sentiment_end']:+.2f}; so it {direction}\n"
        f"Hold time: {mmss(facts['hold_time_sec'])}\n"
        f"Handle time: {mmss(facts['handle_time_sec'])} against this agent's "
        f"target of {mmss(facts['aht_target_sec'])} "
        f"({(facts['aht_ratio'] - 1) * 100:+.0f}% vs target)"
    )


def is_safe(note):
    """Reject notes that drift into disciplinary territory, or malformed ones.

    Returns (ok, reason). This is the real guarantee on tone -- the prompt is
    a request, this is enforcement.
    """
    if not note or len(note.split()) < 8:
        return False, "too_short"
    if len(note.split()) > 70:
        return False, "too_long"
    lowered = note.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            return False, f"banned:{phrase}"
    if note.count(".") > 4:  # more than ~2 sentences of content
        return False, "too_many_sentences"
    # Must end on a finished sentence. Strip any trailing closing quote or
    # bracket first: a note ending `... is outstanding?"` IS complete, and an
    # earlier version of this check rejected it for ending on the quote mark.
    if not note.rstrip().rstrip(CLOSING_MARKS).rstrip().endswith((".", "!", "?")):
        return False, "truncated"
    # Unbalanced DOUBLE quotes mean the model was cut off mid-example
    # ('... such as "Can I clarify'). Apostrophes are deliberately NOT checked:
    # in English prose they are contractions and possessives ("don't", "the
    # customer's"), so an odd count is normal. Checking them rejected ~10% of
    # otherwise perfect notes.
    if note.count('"') % 2:
        return False, "unbalanced_quote"
    if any(marker in note for marker in ("- ", "*", "#", "Note:", "1.")):
        return False, "formatted_output"
    return True, ""


class OllamaGenerator:
    """Calls a local Ollama server. Falls back per-row, never crashes the run."""

    name = f"ollama:{OLLAMA_MODEL}"

    def __init__(self, url=OLLAMA_URL, model=OLLAMA_MODEL):
        import requests

        self.requests = requests
        self.url = url.rstrip("/")
        self.model = model

    @staticmethod
    def available(url=OLLAMA_URL, model=OLLAMA_MODEL):
        """(is_up, message). Checks the server AND that the model is pulled."""
        try:
            import requests
        except ImportError:
            return False, "the `requests` package is not installed"
        try:
            r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=OLLAMA_PROBE_TIMEOUT)
            r.raise_for_status()
        except Exception as exc:  # connection refused, DNS, timeout, bad status
            return False, f"no Ollama server at {url} ({type(exc).__name__})"

        names = [m.get("name", "") for m in r.json().get("models", [])]
        if not any(n == model or n.startswith(model.split(":")[0] + ":") for n in names):
            return False, (f"Ollama is running but '{model}' is not pulled "
                           f"(found: {', '.join(names) or 'nothing'}). "
                           f"Run: ollama pull {model}")
        return True, f"Ollama up at {url} with {model}"

    def generate(self, facts):
        """Return (note, error). note is None when the call failed."""
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": fact_block(facts),
            "stream": False,
            "options": {
                "temperature": 0.4,  # some variety, but keep it on-task
                "num_predict": 200,
                "seed": SEED,  # reproducible, per CLAUDE.md
            },
        }
        try:
            r = self.requests.post(f"{self.url}/api/generate", json=payload,
                                   timeout=OLLAMA_TIMEOUT)
            r.raise_for_status()
            note = r.json().get("response", "").strip()
        except Exception as exc:
            return None, f"request_failed:{type(exc).__name__}"

        # Models like to wrap things in quotes or add a preamble; strip that.
        note = note.strip().strip('"').strip()
        note = re.sub(r"^(coaching note|note)\s*:\s*", "", note, flags=re.IGNORECASE)
        note = " ".join(note.split())
        return note, ""


# --- Cache -----------------------------------------------------------------

def facts_key(facts, generator_name):
    """Cache key: the generator, the interaction, and the inputs to the note.

    The generator MUST be part of the key. Without it a rules run populates
    the cache and a later --generator ollama run serves those same templated
    notes straight back, doing no work and silently producing the wrong
    artefact -- which is exactly what happened before this was fixed.
    """
    parts = [generator_name, str(facts["interaction_id"]), facts["topic"],
             str(facts["sentiment_label"]), f"{facts['sentiment_start']:.3f}",
             f"{facts['sentiment_end']:.3f}", str(facts["hold_time_sec"]),
             str(facts["handle_time_sec"]), str(facts["aht_target_sec"])]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    df = pd.read_csv(CACHE_PATH, keep_default_na=False)
    return {r.key: (r.note, r.generated_by) for r in df.itertuples()}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"key": k, "note": v[0], "generated_by": v[1]}
                  for k, v in cache.items()]).to_csv(CACHE_PATH, index=False)


# --- Main ------------------------------------------------------------------

SENTIMENT_COLUMNS = ["sentiment_label", "sentiment_score",
                     "sentiment_start", "sentiment_end"]


def load_sentiment(needed_ids):
    """Pick the scored-sentiment file that actually COVERS the rows we score.

    roberta is the better model, so it is preferred -- but only if it covers
    the interactions being scored. A partial run (src/sentiment.py --limit)
    leaves a short file behind, and silently joining it would set sentiment to
    NaN on every other row; because NaN compares False, every sentiment-based
    rule would quietly switch itself off and the notes would be written from
    hold and handle time alone. So coverage is checked, not assumed.
    """
    best = None
    for name in ("roberta", "vader"):
        path = LABELS_DIR / f"sentiment_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        coverage = df["interaction_id"].isin(needed_ids).sum() / max(len(needed_ids), 1)
        print(f"  found {path.name}: covers {coverage:.1%} of the rows to score")
        if coverage >= 0.99:
            print(f"Using sentiment from {path}")
            return df, name
        if best is None or coverage > best[2]:
            best = (df, name, coverage)

    if best is not None:
        df, name, coverage = best
        print(f"WARNING: best available is {name} at {coverage:.1%} coverage. "
              f"Rows it does not cover fall back to the generator's "
              f"true_sentiment. Run src/sentiment.py without --limit to fix.")
        return df, f"{name} (partial, {coverage:.0%})"

    print("No scored sentiment in data/labels/ -- falling back to the "
          "generator's hidden true_sentiment. Run src/sentiment.py for real scores.")
    return None, "true_sentiment"


# Flush the cache to disk every N notes. An Ollama pass over 2,000 rows runs
# for hours; saving only at the end means a crash, a Ctrl-C or a laptop sleep
# throws away every note generated so far.
CACHE_FLUSH_EVERY = 100


def generate_notes(facts_df, generator=None, cache=None, show_progress=True,
                   flush_cache=False):
    """Write a coaching note for every row of `facts_df`.

    Shared by this module's CLI and by src/run_pipeline.py -- the generation
    loop lives here rather than inside main() so the pipeline does not have to
    reimplement the fallback and caching rules (CLAUDE.md: promote reusable
    logic into src/).

    `generator` is an OllamaGenerator or None for templates only. `cache` is a
    dict from facts_key() -> (note, generated_by); pass {} to disable reuse.
    Set `flush_cache` to persist progress periodically on long LLM runs.
    Returns (notes_df, fallback_reason_counts).
    """
    if cache is None:
        cache = {}

    rows, fallbacks, t0 = [], {}, time.perf_counter()
    for i, facts in enumerate(facts_df.to_dict("records"), start=1):
        key = facts_key(facts, generator.name if generator is not None else "rules")
        if key in cache:
            note, generated_by = cache[key]
            reason = "cached"
        else:
            note, generated_by, reason = None, "rules", ""
            if generator is not None:
                candidate, error = generator.generate(facts)
                if error:
                    reason = error
                else:
                    ok, why = is_safe(candidate)
                    if ok:
                        note, generated_by = candidate, generator.name
                    else:
                        reason = f"rejected:{why}"
            if note is None:
                note = rule_note(facts)
                generated_by = "rules"
                # Store under the rules key: a fallback is a rules note, and
                # caching it as an ollama result would make the fallback
                # permanent on every future run.
                key = facts_key(facts, "rules")
            cache[key] = (note, generated_by)

        if reason and reason != "cached":
            fallbacks[reason] = fallbacks.get(reason, 0) + 1

        rows.append({
            "interaction_id": facts["interaction_id"],
            "agent_id": facts["agent_id"],
            "agent_name": facts["agent_name"],
            "team": facts["team"],
            "topic": facts["topic"],
            "sentiment_label": facts["sentiment_label"],
            "sentiment_shift": round(facts["shift"], 3),
            "hold_time_sec": facts["hold_time_sec"],
            "handle_vs_target_pct": round((facts["aht_ratio"] - 1) * 100, 1),
            "coaching_note": note,
            "generated_by": generated_by,
        })

        if flush_cache and i % CACHE_FLUSH_EVERY == 0:
            save_cache(cache)  # crash-safety on multi-hour runs

        if show_progress and (i % 25 == 0 or i == len(facts_df)):
            rate = i / max(time.perf_counter() - t0, 1e-9)
            if sys.stderr.isatty():
                sys.stderr.write(f"\r  {i:,}/{len(facts_df):,} notes ({rate:.1f}/s)")
                sys.stderr.flush()
            elif i == len(facts_df) or i % 500 == 0:
                sys.stderr.write(f"  {i:,}/{len(facts_df):,} notes ({rate:.1f}/s)\n")
    if show_progress and sys.stderr.isatty():
        sys.stderr.write("\n")

    return pd.DataFrame(rows), fallbacks


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generator", choices=["auto", "ollama", "rules"], default="auto",
                        help="auto = use Ollama if it is running, else rules")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    args = parser.parse_args()

    for path in (INTERACTIONS_PATH, AGENTS_PATH):
        if not path.exists():
            raise SystemExit(f"{path} not found -- run: python src/generate_all.py")

    interactions = pd.read_csv(INTERACTIONS_PATH)
    agents = pd.read_csv(AGENTS_PATH)
    sentiment, sentiment_source = load_sentiment(set(interactions['interaction_id']))

    facts_df = build_facts(interactions, agents, sentiment)
    attrs = dict(facts_df.attrs)  # .head()/.copy() do not reliably carry attrs
    if args.limit:
        facts_df = facts_df.head(args.limit).copy()
    facts_df.attrs.update(attrs)

    # --- Decide the generator -------------------------------------------
    use_ollama = False
    if args.generator in ("auto", "ollama"):
        up, message = OllamaGenerator.available(args.ollama_url)
        print(message)
        if up:
            use_ollama = True
        elif args.generator == "ollama":
            raise SystemExit("--generator ollama was requested but Ollama is unavailable.")
        else:
            print("Falling back to the rule-based template engine.")

    generator = OllamaGenerator(args.ollama_url) if use_ollama else None
    cache = {} if args.no_cache else load_cache()

    print(f"\nWriting {len(facts_df):,} coaching notes "
          f"({'ollama + rules fallback' if use_ollama else 'rules'})...\n")

    t0 = time.perf_counter()
    notes, fallbacks = generate_notes(facts_df, generator, cache,
                                      flush_cache=generator is not None)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    notes.to_csv(OUTPUT_PATH, index=False)
    if not args.no_cache:
        save_cache(cache)

    # --- Summary ---------------------------------------------------------
    print(f"\nWrote {len(notes):,} notes to {OUTPUT_PATH} "
          f"in {time.perf_counter() - t0:.1f}s")
    backfilled = facts_df.attrs.get("n_backfilled", 0)
    print(f"Sentiment source: {sentiment_source}"
          + (f" ({backfilled:,} rows backfilled from true_sentiment)" if backfilled else "")
          + "\n")
    print("generated_by:")
    print(notes["generated_by"].value_counts().to_string())
    if fallbacks:
        print("\nWhy rows fell back to rules:")
        for reason, n in sorted(fallbacks.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>5,}  {reason}")

    lengths = notes["coaching_note"].str.split().str.len()
    print(f"\nNote length (words): min {lengths.min()}, "
          f"median {lengths.median():.0f}, max {lengths.max()}")

    print("\n--- 6 sample notes ---")
    for row in notes.sample(min(6, len(notes)), random_state=SEED).itertuples():
        print(f"\n[{row.interaction_id}] {row.topic} | {row.sentiment_label} | "
              f"shift {row.sentiment_shift:+.2f} | hold {mmss(row.hold_time_sec)} | "
              f"AHT {row.handle_vs_target_pct:+.0f}% | via {row.generated_by}")
        print(f"  {row.coaching_note}")


if __name__ == "__main__":
    main()
