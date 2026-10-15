# Recommendation: close every contact properly

**To:** Contact Centre Manager · **From:** CX Analytics · **Decision needed:** approve a 6-week pilot on one team

*Based on 2,000 contacts, April–September. This is a demonstration built on
synthetic data — the method is real, the customers are not.*

---

## The problem

We judge a contact by how long it took and whether the customer answered a survey.
Neither tells us the thing that actually costs us money: **whether the customer had
to ring back.**

Almost a quarter of our cases — **22.6%** — produce a second contact within seven
days. That is work we have already been paid to do once, done again. And it is not
spread evenly. It concentrates almost entirely in contacts that *ended* badly:
where the customer was still unhappy at the point we said goodbye.

We have not been measuring that moment at all.

---

## The evidence

**1. How a contact ends predicts whether it happens again — 3× over.**

| | Came back within 7 days |
|---|---|
| Contacts that ended well (1,238) | **13.5%** |
| Contacts that ended badly (585) | **41.9%** |

Not how the call *started*, and not its average mood. How it **ended**.

**2. It is the work, not the channel — and not the speed.**

Holding everything else constant, an outage call has **5.5×** the odds of going
negative and a high bill dispute **2.7×**. Channel has **1.06×** — effectively zero.
Phone, chat and email all land within four points of each other.

Hold time matters, but less than people assume: long holds lift the repeat rate
from 22.1% to 26.5%. Ending badly lifts it from 13.5% to 41.9%. **The ending is
seven times the lever that hold time is.**

**3. One of our teams already fixed this.**

Move In / Move Out halved their negative contacts between July and August —
**21.5% down to 11.1%** — and took first contact resolution from below the industry
benchmark to clearly above it, **+7 points**. Over the same period the other three
teams moved **1.4 points**.

Their handle time did not change. Their hold time did not change. Their mix of work
did not change. **Something about how they run a conversation changed, and we do not
currently know what it was.**

---

## What I recommend

**A mandatory close-out step on every contact, starting with the two topics that
generate the most rework.**

Before ending any outage or high-bill-dispute contact, the agent:

1. **Recaps** what has been done, in one sentence.
2. **Names the next step and when** — a date, not "shortly".
3. **Asks directly** whether anything is still unresolved.

Thirty seconds. No new system, no new headcount, no technology spend.

**Before we design the detail, though: ask Move In / Move Out what they started
doing in August.** The data proves they improved and cannot say why. That
conversation is the cheapest and most valuable half-hour available, and the answer
should shape the pilot rather than this document.

### Why this and not the obvious alternatives

- **Not "reduce hold time."** Worth doing, but it moves repeat contacts by four
  points against twenty-eight for the ending.
- **Not "push customers to chat."** Channel has no measurable effect. We would spend
  real money and move nothing.
- **Not "handle calls faster."** Nothing suggests speed drives satisfaction here, and
  pressure on handle time is the most likely way to make close-outs *worse*.

---

## Expected impact

Deliberately conservative — assume we close only **a third** of the gap between
contacts that end badly and those that do not:

| | Now | After |
|---|---|---|
| Repeat contact rate | 22.6% | **19.6%** |
| Repeat contacts per 2,000 | 412 | **357** |

That is **55 fewer repeat contacts per 2,000**, about 8 agent-hours back per
quarter at our current volume. Scale that to real volume for the actual figure.

The secondary gains are likely larger than the direct saving: fewer repeat contacts
raises FCR, which is the number already reported externally against the 74%
benchmark, and Team 3's experience suggests sentiment moves with it.

**I would not promise more than this.** If the pilot closes half the gap we save 83
contacts; if it changes nothing we have spent six weeks and no money.

---

## How we will measure it

One team pilots. The other three carry on unchanged and act as the comparison —
that is how we proved Team 3's gain was real rather than seasonal, and it is the
only way to know whether the pilot worked or the month was just quieter.

| Measure | Now | Target at 6 weeks | Source |
|---|---|---|---|
| Repeat contact rate (7-day) | 22.6% | ≤ 20% | Case log — counts, so it is trustworthy at any cut |
| Contacts ending negative | 29% | ≤ 25% | End-of-call sentiment score |
| FCR | varies by team | +2pp vs the other teams | Monthly scorecard |
| AHT | 8.3 min | **no more than +30 sec** | Guardrail, not a target |

That last row matters. A close-out step adds a few seconds, and if we let AHT
pressure quietly kill it we will conclude the idea failed when what failed was our
own priorities. **If AHT rises by thirty seconds and repeat contacts fall, that is a
win and should be said out loud at the start.**

Weekly review, decision at six weeks: roll out, adjust, or stop.

---

## How we introduce it

The risk here is not that the step is hard. It is that it reads as another
compliance tick, gets done mechanically, and stops working.

**Week 0 — Start with the team that already solved it.**
Sit down with Move In / Move Out and find out what changed in August. Build the
pilot from what they say, and let them present it. A change that comes from a peer
team lands differently from one that comes from a dashboard.

**Week 1 — Make the case with the numbers, to the people doing the work.**
Show the 13.5% against 41.9%. Agents already know that some calls end unresolved;
this gives them evidence that it matters and that it is not their fault. Frame it as
removing rework from their own queue, because that is what it is.

**Week 1 — Ask for volunteers, do not assign.**
One pilot team, ideally one that put its hand up. Agree the wording *with* them
rather than issuing a script — a close-out in someone's own words survives contact
with a real customer; a mandated sentence does not.

**Weeks 2–6 — Coach it, do not police it.**
Team leads review a handful of contacts a week using the coaching queue, which ranks
by how the contact ended. The notes are developmental by design. **The moment this
becomes a scorecard metric with consequences attached, agents will optimise the
metric instead of the conversation** and we will lose both.

**Week 3 — Fix what the pilot exposes.**
There will be contacts where the agent cannot honestly name a next step because the
process does not have one. Those are findings, not failures, and they are worth more
than the pilot itself. Collect them.

**Week 6 — Decide in public.**
Share the result whichever way it goes. If it worked, the pilot team rolls it out to
the others. If it did not, say so — the credibility of the next change depends on it.

**Ongoing — Keep the guardrail visible.**
Report repeat contact rate and AHT side by side, permanently. The failure mode for
this change is quiet erosion under handle-time pressure, and the only defence is
that both numbers stay on the same page.

---

## What to be careful about

- **The sentiment scores are imperfect.** Validated against 100 hand-read contacts,
  the model agrees with a human 61% of the time and consistently reads customers as
  unhappier than they are. The *end-of-call* score used here is the reliable part
  (77%). Good enough to prioritise who gets a listen; **not** good enough to appear
  on anyone's performance review. See `docs/validation.md`.
- **Survey CSAT understates us.** Only 15% of customers respond and unhappy ones
  respond more. Do not set a CSAT target off this baseline without correcting for it.
- **Three of twenty agents have under six months' tenure** and carry 1.5× the odds of
  a negative contact. That is a ramp-up effect, not a performance problem, and their
  numbers should be read separately.
- **Our longest-serving agents look worst on paper and are not.** They sit entirely
  in Payments & Credit handling nothing but disputes and payment arrangements — the
  two hardest topics. Any league table that ignores caseload will libel them.

---

**The ask:** approve a six-week pilot on one volunteer team, starting with a
conversation with Move In / Move Out. No budget required.

*Workings: `notebooks/insights.ipynb`. Model validation: `docs/validation.md`.*
