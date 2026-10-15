# Your sentiment scores — a quick guide

You'll start seeing a sentiment score and a short coaching note against your
contacts. Here's what they are, what they aren't, and what to do if one looks
wrong.

**The short version:** these are conversation starters for you and your team lead.
They are not a grade, and they are not used for discipline.

---

## What the score actually measures

After each contact, a computer model reads **what the customer typed or said** —
not your half of the conversation — and estimates how they were feeling.

You'll see three things:

| | |
|---|---|
| **Overall** | positive, neutral or negative across the whole contact |
| **Start** | how they sounded in their first few messages |
| **End** | how they sounded by the time you finished |

**The "end" one is the one worth looking at.** Lots of contacts start badly — that's
usually why the customer rang. What matters is where it got to. A call that starts
at "furious" and ends at "thanks, that's sorted" is a good call, and the end score
is what shows that.

---

## It gets things wrong, and we know how

We checked the model against 100 contacts read by a person. Being straight with
you about the results:

- **It agrees with a human about 6 times out of 10.** Not 10 out of 10.
- **When it's wrong, it's almost always wrong in the same direction: it thinks
  customers are unhappier than they actually are.** Of 39 mistakes, all 39 were
  that way round. None went the other way.
- **It never missed a genuinely unhappy customer.** If someone really was upset, it
  caught it every time.

So: a negative score doesn't necessarily mean a bad contact. It means *worth a
look*. A string of them might mean something. One on its own often means nothing.

### Things that reliably confuse it

- **"Not" and "no".** *"I'm not panicking"* and *"there's no gas smell"* both get
  read as alarm. It counts the worrying words and misses the word that cancels them.
- **Normal speech.** *"Look, okay, I can work with that"* scans as confrontational
  on the page. It's just how people talk.
- **Short emails.** Three lines with one complaint in them get judged entirely on
  that one line.
- **Thinking out loud.** *"If the review finds nothing, I'm still stuck with the
  bill"* is someone working through a worry calmly — the model hears anger.
- **Sarcasm.** It almost never catches it.
- **Tone of voice.** It only reads the transcript. Everything you did with your
  voice — warmth, patience, the pause in the right place — is invisible to it.

That last one matters. **The model cannot hear you being good at your job.**

---

## How to read your coaching note

Each note has two halves, always in the same order:

> *"You kept the customer calm through a difficult billing dispute, which is not
> easy. Next time, try naming the billing review earlier so they hear a concrete
> next step sooner."*

**One thing that went well. One thing to try next time.** That's the whole format.

A few things worth knowing:

- They're generated from the facts of the contact — the topic, how long the hold
  was, how the sentiment moved. **Nobody sat and watched you.**
- The "try next time" half is a suggestion, not a correction. Sometimes it'll be
  obvious, sometimes you'll have already done it, sometimes it won't fit the
  situation at all. That's fine.
- If a note describes a contact that doesn't sound like the one you remember, it's
  probably wrong. Say so.

---

## If a score looks wrong

**Please flag it. Genuinely — it helps.** Every correction makes the model better,
and we'd rather know.

1. **Note the contact ID** (the `I00000`-style reference on the row).
2. **Tell your team lead** what the score says and what actually happened. One
   sentence is enough: *"This one's marked negative but the customer was fine — they
   were quoting what their neighbour said."*
3. **Your team lead logs it** against that contact.
4. **It's reviewed** — a person reads the transcript and decides.
5. **If the score was wrong, it's corrected**, and the contact is added to the set
   we use to retrain and re-check the model.

**No justification needed and no limit on how many you flag.** If you think it's
wrong, flag it. If several agents flag the same kind of contact, that's a pattern
we need to fix in the model, not in you.

You can also flag a coaching note that reads oddly, feels unfair, or doesn't match
the contact. Same process.

---

## What these scores are *not* used for

Being explicit, because this is the part people worry about:

**These scores are not used for:**

- ❌ Performance reviews or ratings
- ❌ Disciplinary processes of any kind
- ❌ Pay, bonus or progression decisions
- ❌ Ranking agents against each other
- ❌ Any automated decision about you

**They are used for:**

- ✅ Picking which contacts are worth listening to together
- ✅ Spotting where a *process* is making contacts harder than they need to be
- ✅ Finding what's working — some of the best coaching comes from a contact that
  recovered well

A model that's right 6 times out of 10 and systematically pessimistic has no
business anywhere near someone's employment. It's nowhere near good enough for
that, and it never will be. **A person reads the contact before any conclusion is
drawn about how you handled it.**

If anyone ever uses these scores against you in a performance context, that is a
misuse of the tool and you should escalate it.

---

## The one thing most worth your attention

Across every contact we looked at, one pattern stood out:

> **Contacts that end badly are three times more likely to have the customer ring
> back** — 42% against 13%.

Not contacts that *start* badly. Contacts that *end* badly.

The most useful thirty seconds in any contact is the last thirty: recap what you've
done, say what happens next and when, and ask directly whether anything's still
outstanding. That's it. It's the single biggest thing in the data, and it's
entirely within your control.

---

## Questions

Ask your team lead first — they can see the same data you can, plus the context.
Anything about how the model works, or if you think it's getting a whole category
of contact wrong, comes to the CX Analytics team.

*The full honest write-up of how accurate the model is, including everything it
gets wrong, is in `docs/validation.md`. You're welcome to read it.*

---

*This is a demonstration project built on synthetic data — no real customers,
agents or conversations are involved.*
