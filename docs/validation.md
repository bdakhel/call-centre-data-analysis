# Can we trust the AI to score customer sentiment?

**Short answer: for spotting unhappy customers, yes — today. For anything else, not yet.**

*A note on the data: this is a portfolio demonstration built on synthetic contact centre
records. No real customers, agents or conversations are involved. The method is what
matters here; the numbers describe made-up data.*

---

## What we did

We took 100 conversations, picked to match the real mix of topics and contact channels,
and had a person read each one and judge how the customer felt: positive, neutral or
negative.

That person saw only the transcript. Not the AI's answer, not the topic, nothing else —
because once you have seen the machine's guess, you cannot un-see it, and the check stops
being a check.

Then we compared two automated scorers against those human judgements:

- **The word-list** — a simple tool that counts positive and negative words. Our baseline:
  if this works, we do not need anything cleverer.
- **The AI model** — a language model that reads the conversation in context.

---

## The results

| | Agrees with the human | Score after removing lucky guesses |
|---|---|---|
| Word-list | 47% | 0.19 — barely better than guessing |
| **AI model** | **61%** | **0.42 — moderate** |
| *The "right answer" in our test data* | *88%* | *0.82 — near-perfect* |

The second column matters because with three choices you get roughly a third right by
luck alone. It strips that out: 0 means coin-flipping, 1 means perfect.

That last row is the reality check. Our synthetic conversations were built with an
intended mood, and the human reader independently matched it **88%** of the time. So two
things are true: the labelling was done carefully, and **even a perfect system would only
score around 90%**, because some conversations genuinely are ambiguous. The AI's 61% should
be read against 90%, not against 100%.

---

## The one thing worth remembering

The AI's mistakes are not random. They all go the same way.

Of 100 conversations, it got 61 right and 39 wrong — and **all 39 errors were the AI
judging the customer as unhappier than the human did. Not one error went the other way.**

In practice:

- **It never missed an unhappy customer.** All 31 conversations the human called negative,
  the AI also called negative. Perfect.
- **It missed most of the happy ones.** Of 34 conversations the human called positive, the
  AI found 9.

This is good news, because a consistent one-directional bias is a dial set wrong, not a
system that fails to understand. Random errors would be much harder to fix.

---

## Why it gets things wrong

Reading the mistakes, five causes come up repeatedly:

1. **The customer calmed down and the AI did not notice.** Someone opens furious about
   twelve months of estimated meter readings, the agent fixes it, and they close with
   "that's great, you've been really helpful." The AI averages the whole call and reports
   an unhappy customer. This is the single biggest cause.
2. **It misses the word "not".** "I'm *not* panicking" and "there's *no* gas smell" both get
   counted as alarm.
3. **Ordinary speech sounds like conflict.** "Look, okay, I can work with that" is just how
   people talk, but reads as confrontational on the page.
4. **Short messages have no balance.** A three-line email containing one complaint gets
   judged entirely on that one line.
5. **It cannot tell a worry from a complaint.** "If the review finds nothing, I'm still
   stuck with the bill" is someone thinking ahead calmly, not someone angry.

Accuracy also varies a lot by contact type — **85% on move-in/move-out, but 29% on
outages**. Outage calls are the hardest shape: they start with a real problem and often end
satisfied once the customer gets an ETA.

---

## A fix we can make now

The AI already scores the beginning and the end of each conversation separately. We just
were not using it.

When we judge a conversation by **how the customer sounded at the end** rather than by
averaging the whole thing, agreement rises from **56% to 77%** on conversations the method
had never seen.

That is the right question anyway. "How did this customer feel about the contact?" is
answered by how they felt when they hung up — not by averaging their opening complaint
with their closing thanks.

**This needs no new technology, no retraining and no extra cost.** It is a change to how a
number we already calculate gets read.

---

## Recommendation on human review

**Do not switch off human review. Use the AI to decide who gets it.**

Three specific recommendations:

**1. Use it now for spotting unhappy customers — on its own.**
It caught every unhappy customer in our test. For flagging calls that need a follow-up or a
manager's attention, it is ready, and it is better than the post-call survey, which only
15% of customers answer. Treat a negative flag as "worth a look", not as a verdict.

**2. Do not use it yet for satisfaction reporting.**
It finds roughly one in four genuinely happy customers. Any dashboard built on it would
show satisfaction far below reality and would make teams look like they are failing when
they are not. Hold this back until the fix above is in place and re-tested.

**3. Review about 30% of contacts, chosen by the AI's own confidence.**

| If we send a person... | The AI handles | At this accuracy |
|---|---|---|
| nothing (full automation) | 100% | 81% |
| the least certain 16% | 84% | 82% |
| **the least certain 29%** | **71%** | **85%** |
| the least certain 41% | 59% | 83% |

The middle option is the sweet spot: roughly **three in ten conversations go to a person**,
the rest are scored automatically at 85% accuracy, and reviewing more than that stops
helping. A reviewer confirms or corrects, and those corrections become training data for
the next round.

---

## What this does not tell us

Three honest limits:

- **One hundred conversations, one reader.** Enough to see a clear pattern, not enough for
  precision — and with only one labeller we cannot measure how much two people would
  disagree with each other, which is the real benchmark for "good enough".
- **Per-topic figures rest on 8 to 21 conversations each.** Treat them as a steer on where
  to look, not as measurements.
- **Synthetic data.** Real transcripts are messier. Expect the real figures to be lower and
  re-run this check on live data before relying on any of it.

**Suggested next step:** apply the end-of-conversation fix, re-run this same 100-conversation
check to confirm the gain holds, and repeat it whenever the model changes. The whole
exercise took an afternoon and is the cheapest insurance available against quietly shipping
a broken metric.

---

*Full workings, charts and the conversations quoted above: `notebooks/validation.ipynb`.*
