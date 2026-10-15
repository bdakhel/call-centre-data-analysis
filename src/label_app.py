"""Streamlit helper for hand-labelling transcript sentiment.

Shows one transcript at a time with Positive / Neutral / Negative buttons and
appends to data/labels/human_labels.csv after every click.

Deliberately blind: this app only ever reads data/labels/to_label.csv, which
src/make_label_sample.py built without true_sentiment, without any model
prediction, and without the topic. Nothing on screen tells you what the
machine thought, because a human label is only worth having if it was formed
independently. Please do not open interactions.csv in another tab.

Saving is immediate -- one write per label -- so closing the browser never
costs more than the row you were on. Reopening resumes at the first unlabelled
transcript.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    streamlit run src/label_app.py
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

TO_LABEL_PATH = Path("data/labels/to_label.csv")
# Rows where the RoBERTa scorer and the LLM judge disagree, written by
# src/compare_labels.py. Labelling only these is the high-value short path:
# on the rows both models already agree on, a human vote changes little.
PRIORITY_PATH = Path("data/labels/to_label_priority.csv")
LABELS_PATH = Path("data/labels/human_labels.csv")

LABELS = ["positive", "neutral", "negative"]
# Button styling: Streamlit only offers primary/secondary, so the emoji does
# the colour-coding work.
BUTTONS = [("positive", "🙂  Positive"), ("neutral", "😐  Neutral"), ("negative", "🙁  Negative")]

st.set_page_config(page_title="Transcript labelling", page_icon="🏷️", layout="centered")

# NOTE ON use_container_width
# Streamlit 1.6x renamed this to width="stretch" and deprecates the old
# spelling, but it did NOT add `width` to every function at the same release:
# st.altair_chart and st.button on 1.37 reject it outright with a TypeError.
# use_container_width works on both, so it stays until the floor version is
# raised. If you see a deprecation warning, that is the trade — not a bug.





@st.cache_data
def load_queue():
    """The sample to label. Cached so a rerun does not re-read the file."""
    if not TO_LABEL_PATH.exists():
        return None
    return pd.read_csv(TO_LABEL_PATH)


def load_labels():
    """Labels recorded so far, as {interaction_id: label}.

    Read fresh (not cached) every rerun: it is written on every click, and a
    stale read here would show the wrong progress.
    """
    if not LABELS_PATH.exists():
        return {}
    done = pd.read_csv(LABELS_PATH)
    # Last write wins, so a changed mind overwrites the earlier answer.
    return dict(zip(done["interaction_id"], done["human_label"]))


def save_label(interaction_id, label):
    """Append one decision. Immediate write keeps progress crash-safe."""
    row = pd.DataFrame([{
        "interaction_id": interaction_id,
        "human_label": label,
        "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }])
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = not LABELS_PATH.exists()
    row.to_csv(LABELS_PATH, mode="a", header=header, index=False)


def render_transcript(text):
    """Draw the dialogue as a chat, so turn-taking is easy to follow."""
    for line in str(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Customer:"):
            with st.chat_message("user"):
                st.write(line[len("Customer:"):].strip())
        elif line.startswith("Agent:"):
            with st.chat_message("assistant"):
                st.write(line[len("Agent:"):].strip())
        else:  # shouldn't happen, but never hide data from the labeller
            st.write(line)


def first_unlabelled(queue, labels):
    for i, interaction_id in enumerate(queue["interaction_id"]):
        if interaction_id not in labels:
            return i
    return len(queue) - 1  # everything done: park on the last one


# --- App -------------------------------------------------------------------

full_queue = load_queue()
queue = full_queue
if queue is None:
    st.error(f"`{TO_LABEL_PATH}` not found.")
    st.code("python src/make_label_sample.py", language="bash")
    st.stop()

labels = load_labels()

# --- Optional: narrow the queue to the contested rows ---------------------
priority_ids = None
if PRIORITY_PATH.exists():
    priority_ids = set(pd.read_csv(PRIORITY_PATH)["interaction_id"])
    remaining_priority = len([i for i in priority_ids if i not in labels])
    only_contested = st.sidebar.toggle(
        f"Only rows the models disagree on ({remaining_priority} left)",
        value=False,
        help="The scorer and the LLM judge disagree on these. Your vote breaks "
             "the tie; on the rest they already agree.",
    )
    if only_contested:
        narrowed = full_queue[full_queue["interaction_id"].isin(priority_ids)]
        if len(narrowed):
            queue = narrowed.reset_index(drop=True)

if "position" not in st.session_state or st.session_state.get("_queue_len") != len(queue):
    # Reset the cursor when the queue changes size, or the saved index would
    # point at a different transcript than it did before the toggle.
    st.session_state._queue_len = len(queue)
    st.session_state.position = first_unlabelled(queue, labels)

# Clamp: the queue could have been regenerated shorter since last session.
st.session_state.position = max(0, min(st.session_state.position, len(queue) - 1))
position = st.session_state.position
row = queue.iloc[position]
existing = labels.get(row["interaction_id"])


def go(delta):
    st.session_state.position += delta


def record(label):
    save_label(row["interaction_id"], label)
    # Advance to the next UNLABELLED row rather than simply position + 1, so
    # revisiting one row mid-session does not drop you back into finished work.
    refreshed = load_labels()
    refreshed[row["interaction_id"]] = label
    remaining = [i for i in range(position + 1, len(queue))
                 if queue.iloc[i]["interaction_id"] not in refreshed]
    st.session_state.position = remaining[0] if remaining else position


# --- Header ---------------------------------------------------------------
st.title("🏷️ Transcript sentiment labelling")

total = len(queue)
done = sum(1 for i in queue["interaction_id"] if i in labels)
st.progress(done / total, text=f"{done} of {total} labelled")

if done == total:
    st.success("All transcripts labelled. You can close this tab.")

left, mid, right = st.columns([1, 2, 1])
left.caption(f"**#{position + 1}** of {total}")
mid.caption(f"`{row['interaction_id']}`  ·  channel: **{row['channel']}**")
if existing:
    right.caption(f"labelled: **{existing}**")

st.divider()

# --- The transcript -------------------------------------------------------
render_transcript(row["transcript_text"])
st.divider()

# --- The decision ---------------------------------------------------------
st.caption("How did **the customer** feel about this contact overall?")
cols = st.columns(3)
for col, (value, caption) in zip(cols, BUTTONS):
    col.button(
        caption,
        key=f"btn_{value}",
        use_container_width=True,
        type="primary" if existing == value else "secondary",
        on_click=record,
        args=(value,),
    )

# --- Navigation -----------------------------------------------------------
st.divider()
back, skip, jump = st.columns([1, 1, 2])
back.button("← Previous", use_container_width=True, disabled=position == 0,
            on_click=go, args=(-1,))
skip.button("Skip →", use_container_width=True, disabled=position >= total - 1,
            on_click=go, args=(1,))
if done < total:
    jump.button("Jump to next unlabelled", use_container_width=True,
                on_click=lambda: st.session_state.update(
                    position=first_unlabelled(queue, load_labels())))

with st.sidebar:
    st.header("Progress")
    st.metric("Labelled", f"{done} / {total}")
    if labels:
        counts = pd.Series(list(labels.values())).value_counts()
        st.bar_chart(counts)
        st.caption("Your label mix so far. A heavy skew early on is worth "
                   "noticing -- it is usually fatigue, not the data.")
    st.divider()
    st.caption(f"Saving to `{LABELS_PATH}` after every click.")
    st.caption("This app shows no model output and no generated labels, "
               "so your judgement stays independent.")
