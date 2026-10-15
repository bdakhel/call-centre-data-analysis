"""Fill transcript_text in data/raw/interactions.csv with synthetic dialogues.

Template-and-phrase-bank generator: every transcript is assembled from banks of
greetings, verification lines, topic-specific probes, resolutions and closings,
so the output varies a lot without needing an LLM.

Each dialogue matches its row's topic, channel and true_sentiment:

  * topic          -> which utility-specific details appear (meter reads,
                      estimated bills, winter disconnection rules, move dates,
                      payment arrangements, outage ETAs ...)
  * channel        -> voice gets spoken disfluencies; chat gets lowercase,
                      shorthand and typos; email is formal with a few typos
  * true_sentiment -> the emotional arc of the conversation

Realistic noise on purpose:
  * mixed arcs (angry start / happy end, calm start / angry end)
  * occasional sarcasm
  * typos in chat and email only

Safety: no real company, no real people, and NO payment card or bank numbers --
anywhere a card/account-for-payment would be spoken we emit "[REDACTED]".

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/generate_transcripts.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

# --- Reproducibility -------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INTERACTIONS_PATH = Path("data/raw/interactions.csv")
AGENTS_PATH = Path("data/raw/agents.csv")

# Fictional utility. Never use a real company name (see CLAUDE.md).
BRAND = "Prairie Sky Utilities"

# How many topic-specific middle exchanges (one agent + one customer turn each)
# a transcript gets, by channel. Email threads are shorter than calls.
# Total turns = 2 (greeting + opening) + 2 if verified + 2*pairs + 3 or 4 (close),
# which keeps every transcript inside the 6-14 turn contract.
MIDDLE_PAIRS = {"voice": (1, 3), "chat": (1, 3), "email": (0, 1)}

REDACTED = "[REDACTED]"

# =========================================================================
# Context values dropped into templates (all synthetic)
# =========================================================================
AB_CITIES = [
    "Calgary", "Edmonton", "Red Deer", "Lethbridge", "Medicine Hat",
    "Grande Prairie", "Airdrie", "Spruce Grove", "Okotoks", "Fort Saskatchewan",
    "Camrose", "Cochrane", "Leduc", "Lloydminster", "Canmore",
]
STREETS = [
    "Aspen Ridge Way", "Willow Creek Close", "Hawkstone Drive", "Sundance Boulevard",
    "Birchwood Lane", "Prairie Rose Crescent", "Cedarglen Road", "Thornbury Place",
    "Coyote Run", "Maple Bend Avenue",
]
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def build_context(row, agent_first_name):
    """Every placeholder any template might use, pre-filled for one row."""
    prev = round(random.uniform(85, 210), 2)
    spike = round(prev * random.uniform(1.8, 3.4), 2)
    return {
        "brand": BRAND,
        "agent": agent_first_name,
        # Clearly fictional account/site identifiers (brand-prefixed).
        "acct": f"PSU-{random.randint(1000000, 9999999)}",
        "site": f"SITE-{random.randint(100000, 999999)}",
        "meter": f"{random.choice('GE')}{random.randint(1000000, 9999999)}",
        "ref": f"REF-{random.randint(10000, 99999)}",
        "city": random.choice(AB_CITIES),
        "street": f"{random.randint(2, 480)} {random.choice(STREETS)}",
        "prev_amount": f"{prev:.2f}",
        "amount": f"{spike:.2f}",
        "balance": f"{round(random.uniform(120, 1850), 2):.2f}",
        "plan_amount": f"{round(random.uniform(45, 240), 2):.2f}",
        "credit": f"{round(random.uniform(15, 140), 2):.2f}",
        "months": random.choice([3, 4, 6, 8, 12]),
        "gj": round(random.uniform(4.5, 28.0), 1),
        "kwh": random.randint(280, 1900),
        "day": random.randint(1, 28),
        "month": random.choice(MONTH_NAMES),
        "days": random.choice([2, 3, 5, 7, 10, 14]),
        "eta_hours": random.choice([2, 3, 4, 6]),
        "eta_clock": f"{random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])}:{random.choice(['00', '15', '30', '45'])}",
        "redacted": REDACTED,
    }


# =========================================================================
# Phrase banks
# =========================================================================
GREETINGS = {
    "voice": [
        "Thank you for calling {brand}, my name is {agent}. Who do I have the pleasure of speaking with?",
        "{brand}, this is {agent} speaking. How can I help you today?",
        "Good morning, you've reached {brand}. My name is {agent} — what can I do for you?",
        "Thanks for holding, you're through to {agent} at {brand}. How can I help?",
        "{brand} customer care, {agent} speaking. What brings you in today?",
    ],
    "chat": [
        "Hi, you're chatting with {agent} at {brand}. How can I help today?",
        "Hello and welcome to {brand} chat support. My name is {agent}. What can I help you with?",
        "Hi there, {agent} here from {brand}. What's going on with your account?",
        "Thanks for waiting — {agent} from {brand} here. How can I help?",
    ],
    "email": [
        "Hello, thank you for contacting {brand}. My name is {agent} and I will be looking after your enquiry.",
        "Good day, this is {agent} from the {brand} customer care team. Thank you for your message.",
        "Hello, thank you for writing in to {brand}. I am {agent} and I have reviewed your enquiry.",
    ],
}

# Customer's first message, by topic and opening tone.
OPENINGS = {
    "high bill dispute": {
        "negative": [
            "Yeah, I need someone to explain how my gas bill went from ${prev_amount} to ${amount} in one month. Nobody in this house changed a thing.",
            "I'm looking at a bill for ${amount}. That is more than double what I normally pay and frankly it's outrageous.",
            "This bill is wrong. ${amount}? I live alone. I want this fixed today, not another investigation.",
            "I've had enough. Third month in a row the bill has gone up and now it's ${amount}. Somebody is going to explain this to me.",
        ],
        "neutral": [
            "Hi, I got my statement and it's ${amount} when it's usually around ${prev_amount}. I'd like to understand why.",
            "I'd like someone to go through my bill with me. The amount looks much higher than usual.",
            "Morning. My bill came in at ${amount} this cycle and I'm trying to work out what changed.",
        ],
        "positive": [
            "Hi, no emergency — I just want to understand my bill. It came in at ${amount} and I'd like to know what drove that.",
            "Good morning. My statement is higher than I expected and I was hoping you could walk me through it.",
        ],
    },
    "billing question": {
        "negative": [
            "I've got a charge on here I don't recognize and the website is useless. What is this rider line for ${credit}?",
            "Why am I being billed twice for the same period? I'm not paying for the same gas twice.",
            "Your bill is impossible to read. I just want to know what I actually owe.",
        ],
        "neutral": [
            "Hi, I have a question about a line item on my statement — the distribution charge.",
            "I'm trying to figure out the difference between the energy charge and the delivery charge on my bill.",
            "Quick question: my statement shows a credit of ${credit} and I'm not sure where it came from.",
            "Hello, I'd like to confirm my billing period dates. I think my cycle shifted.",
        ],
        "positive": [
            "Hi! Nothing's wrong, I just want to understand my bill a bit better so I can budget properly.",
            "Hello, I had a quick question about my statement — I think I'm reading it wrong.",
        ],
    },
    "move-in/move-out": {
        "negative": [
            "I set up a move-out on the {day}th and you're still billing me for the old place. Why?",
            "I'm closing on my house in {days} days and nobody has confirmed anything. This should not be hard.",
            "My service was supposed to be on when I took possession and I walked into a cold house. In {city}. In this weather.",
        ],
        "neutral": [
            "Hi, I'm moving to {city} on the {day}th and I need to set up service at the new address.",
            "I need to close my account at {street} and open one at the new place.",
            "Hello, I'm moving out on the {day}th of {month} and want to make sure the final read is scheduled.",
            "I'm taking possession of a place in {city} and need gas and power in my name.",
        ],
        "positive": [
            "Hi! We're moving into a new place in {city} on the {day}th and I want to get everything lined up early.",
            "Good morning — moving day is the {day}th and I'd like to transfer my service over.",
        ],
    },
    "payment arrangement": {
        "negative": [
            "I got a disconnection notice. I have a balance of ${balance} and I can't pay it all at once, and I'm not being treated like a criminal over it.",
            "I called about this last week and nothing was set up. Now I'm getting collection letters for ${balance}.",
            "You people sent me a notice threatening to cut off my heat. I have kids in the house.",
        ],
        "neutral": [
            "Hi, I've got a balance of ${balance} and I'd like to set up a payment plan.",
            "I'm behind on my account and want to arrange instalments before it goes further.",
            "Hello, I need to look at spreading my balance over a few months if that's possible.",
        ],
        "positive": [
            "Hi, I know I'm behind — I'd like to sort out a plan so I can get back on track.",
            "Good morning. I've had a rough couple of months and want to set up something manageable on my balance of ${balance}.",
        ],
    },
    "outage or service issue": {
        "negative": [
            "My power has been out for {eta_hours} hours in {city} and your outage map says nothing. What is going on?",
            "This is the third outage this month on {street}. I work from home. This is costing me money.",
            "No heat, it's freezing, and I've been on hold forever. When is someone actually coming?",
        ],
        "neutral": [
            "Hi, I'm at {street} in {city} and the power is out. I wanted to report it and get an ETA.",
            "I've got no power at my place in {city} — the neighbours are out too.",
            "Hello, my furnace isn't getting gas and I wanted to check whether there's an outage in the area.",
            "There's a flickering issue with my power at {street} — it's been dropping on and off since yesterday.",
        ],
        "positive": [
            "Hi, power's out at {street} — just wanted to report it in case you didn't know yet.",
            "Morning! No power in our part of {city}. Not panicking, just looking for an update.",
        ],
    },
    "meter reading": {
        "negative": [
            "You've estimated my meter for {months} months straight and now you've hit me with a catch-up bill. That's not acceptable.",
            "Nobody has read my meter in months. I sent in a reading and it was ignored.",
            "Your meter reader never came and now the numbers are completely wrong.",
        ],
        "neutral": [
            "Hi, I'd like to submit my own meter reading. The dial shows {gj} on the gas side.",
            "My statement says 'estimated' — I want to give you an actual read.",
            "Hello, I want to check when my meter was last read. Meter number is {meter}.",
            "I think my electric meter reading is off. It shows {kwh} kilowatt hours used and that seems high.",
        ],
        "positive": [
            "Hi, nothing urgent — I took a photo of my meter and want to pass on the actual reading.",
            "Good afternoon, I'd like to submit a reading so my next bill is accurate.",
        ],
    },
    "general enquiry": {
        "negative": [
            "I've been transferred twice already. I just want to know if I'm on the right rate.",
            "Your website logged me out three times. Can you just tell me my balance?",
        ],
        "neutral": [
            "Hi, I wanted to ask about the rate options — I'm on the regulated rate and wondering about a fixed term.",
            "Hello, I'd like to update the contact information on my account.",
            "Quick question about pre-authorized payment — how do I get set up on it?",
            "I heard there's an efficiency rebate program. Can you tell me how it works?",
            "Hi, I'd like to add my partner as an authorized contact on the account.",
        ],
        "positive": [
            "Hi! Just a quick question about the equalized payment plan — is it something I can join mid-year?",
            "Good morning, I wanted to ask about the home efficiency rebate I saw mentioned on my bill.",
        ],
    },
}

VERIFY_ASK = [
    "Happy to look into that. Can I get the account number or the service address to pull it up?",
    "Let me pull up the account. Could you confirm the account number and the name on the file?",
    "Before I go further I just need to verify the account — can you give me the service address and postal code?",
    "I can check that for you. What's the account number on the top right of your statement?",
    "Sure. To protect your information, can you confirm the service address and the last bill amount?",
]

VERIFY_GIVE = [
    "It's {acct}, and the address is {street}, {city}.",
    "Account number is {acct}. Service address {street}.",
    "Sure — {street} in {city}, and I think the account is {acct}.",
    "{acct}. The last bill was ${prev_amount} if that helps.",
    "I don't have the statement in front of me, but the address is {street}, {city}.",
]

VERIFY_CONFIRM = [
    "Thank you, I have the account open now — site ID {site}.",
    "Perfect, that matches. I can see the account.",
    "Got it, thanks. I'm in the account now.",
    "Thanks for confirming. I have your file up.",
]

# Topic-specific middle exchanges: (agent line, customer line).
MIDDLE = {
    "high bill dispute": [
        ("I can see the last three cycles here. Your usage went from {gj} gigajoules up noticeably this period — was there a cold snap or anyone extra in the house?",
         "We had that cold stretch, sure, but it wasn't three times colder."),
        ("One thing I'm noticing is that the previous two reads were estimated and this one was an actual read, so this bill is catching up on usage that was under-billed earlier.",
         "So you guessed low for two months and now I eat the difference in one bill?"),
        ("The energy charge itself is only part of it — there's also delivery, the municipal franchise fee and the riders, and those scale with usage.",
         "Nobody explains that anywhere on the statement."),
        ("Your furnace running harder in a cold month is the usual driver. Degree-day data for {city} shows this period was well below normal.",
         "That might explain some of it, I suppose."),
        ("I can request a billing review and a verification read on meter {meter} — that gets a technician out to confirm the register isn't faulty.",
         "How long does that take?"),
        ("If the review finds an error the correction is backdated, so you wouldn't lose out.",
         "And if it doesn't find an error, I'm still stuck with ${amount}."),
    ],
    "billing question": [
        ("The distribution charge covers the pipes and wires that get the energy to your home — it's set by the regulator, not by us.",
         "So it's fixed no matter how much I use?"),
        ("Partly fixed, partly variable. There's a daily fixed charge and then a variable portion tied to your usage of {gj} gigajoules.",
         "Okay, that's clearer than the statement makes it."),
        ("The credit of ${credit} came from a rate rider adjustment applied in {month}.",
         "That would have been nice to know on the bill itself."),
        ("Your billing cycle shifted by a few days because of the read date, which is why the period looks longer this time.",
         "Right, that explains the bigger number."),
        ("Looking at the detail, you're billed {kwh} kilowatt hours on the electricity side and {gj} gigajoules on gas, on one combined statement.",
         "I didn't realize both were on the same bill."),
        ("The amount currently owing is ${balance}, and the due date is the {day}th.",
         "Okay, I can work with that."),
    ],
    "move-in/move-out": [
        ("I can schedule the final read at {street} for the {day}th. Is that the day you hand over the keys?",
         "Yes, possession is that day."),
        ("For the new address I'll need the full address in {city} and the date you want service active.",
         "It's effective the {day}th of {month}."),
        ("I'd recommend setting the start date a day early so you're not walking into a cold house.",
         "That's a good idea, let's do that."),
        ("There's a one-time account setup charge and your final statement will come about {days} days after the move-out read.",
         "Fine, as long as there are no surprises."),
        ("I'm showing the old account is still open — that's why billing continued. I'll backdate the close to the {day}th.",
         "Thank you, that's what should have happened the first time."),
        ("Do you want to keep the same equalized payment plan at the new place, or start fresh with actual billing?",
         "Let's start fresh and see what the usage looks like."),
    ],
    "payment arrangement": [
        ("The balance on the account is ${balance}. I can look at spreading that over {months} months alongside your current bills.",
         "That would be about ${plan_amount} a month on top of normal usage?"),
        ("Roughly, yes. The plan would be ${plan_amount} a month. Does that work with your budget?",
         "It's tight, but it's better than a disconnection notice."),
        ("I also want to flag that we're inside the winter protection period, so residential heat isn't disconnected between October 15 and April 15 as long as you're engaged with us.",
         "Nobody told me that. I've been panicking for two weeks."),
        ("Once the arrangement is in place the collection activity stops and the notice is withdrawn.",
         "So the letters stop coming?"),
        ("For the payment method I'd set up pre-authorized debit. I won't read any numbers back to you — the card and bank details show as {redacted} on my side.",
         "Good, I'd rather they weren't read out anyway."),
        ("There are also utility assistance programs for Alberta households — I can email you the referral information with your confirmation.",
         "Yes please, I'd appreciate that."),
    ],
    "outage or service issue": [
        ("I'm showing an unplanned outage affecting your area of {city} — about 340 sites, first reported earlier today.",
         "So it's not just us."),
        ("Crews are on site now. The current estimate for restoration is around {eta_clock}, though that can move once they identify the fault.",
         "Any idea what caused it?"),
        ("The initial report points to a failed transformer on {street}. The crew has to isolate the section before they can re-energize.",
         "As long as somebody is actually out there."),
        ("If you smell gas at any point, leave the building and call the emergency line from outside — that's a different queue and it's answered immediately.",
         "No, no gas smell, it's just the power."),
        ("I'll add your premise to the outage ticket as {ref} so you get the automated restoration notification.",
         "That's helpful, thanks."),
        ("For the flickering, that pattern usually points to a service connection issue rather than the grid. I can book a technician within {days} business days.",
         "Let's get that booked."),
    ],
    "meter reading": [
        ("I can take that reading now. For gas I need the figure in gigajoules from the dial — you said {gj}?",
         "That's right, {gj}."),
        ("Thank you. I've entered the read against meter {meter} and flagged it as a customer-supplied actual.",
         "So the next bill will use the real number?"),
        ("It will. I've also removed the estimate flag so the next statement reconciles against your actual usage.",
         "Good, because the estimates have been nowhere close."),
        ("I can see why you were frustrated — the last {months} reads were estimated because the reader couldn't access the meter. Is there a gate or a dog?",
         "There's a gate, yes, but it's not locked."),
        ("I'll add an access note to the file so the reader knows. That should stop the estimating.",
         "That would help a lot."),
        ("On the electric side the register shows {kwh} kilowatt hours, which lines up with the same period last year.",
         "Okay, so that one is fine."),
    ],
    "general enquiry": [
        ("You're currently on the regulated rate, which changes monthly. A fixed term locks your energy rate but usually has an exit fee.",
         "And the delivery charges stay the same either way?"),
        ("Correct, delivery is regulated regardless of who you buy your energy from.",
         "Okay, that's the part I was confused about."),
        ("The equalized payment plan averages your annual usage into even monthly amounts — yours would be about ${plan_amount}.",
         "That's much easier to budget for."),
        ("You can join mid-year; we true up the difference at the plan anniversary.",
         "Good to know."),
        ("For the efficiency rebate, the program is administered provincially — I can send you the link and the eligibility list.",
         "Yes, please send that over."),
        ("I've updated the contact details and added the authorized contact to the file.",
         "Perfect, thank you."),
    ],
}

# Agent resolution, by topic.
RESOLUTIONS = {
    "high bill dispute": [
        "Here's what I've done: billing review opened under {ref}, verification read ordered on meter {meter}, and I've placed a hold on collection activity until it closes.",
        "I've opened a formal billing investigation as {ref} and extended your due date by {days} days while it's reviewed.",
        "I can't change the usage, but I've ordered the verification read and set you up on an equalized plan at ${plan_amount} so you're not hit like this again.",
    ],
    "billing question": [
        "I've emailed you the detailed charge breakdown and noted the explanation on the account under {ref} in case you call back.",
        "So to confirm: ${balance} owing, due the {day}th, and the ${credit} credit is already applied.",
        "I've added a note to your file and sent the plain-language guide to the statement charges.",
    ],
    "move-in/move-out": [
        "All set: final read at {street} on the {day}th, new account open in {city} effective the day before, confirmation reference {ref}.",
        "I've closed the old account effective the {day}th, backdated the billing, and opened the new one. You'll see a final statement in about {days} days.",
        "Service is scheduled to be live at the new address on the {day}th and I've emailed the confirmation to you as {ref}.",
    ],
    "payment arrangement": [
        "The arrangement is in place: ${plan_amount} a month over {months} months, notice withdrawn, and the payment method on file shows as {redacted}.",
        "I've set the instalment plan at ${plan_amount} monthly, stopped the collection activity, and emailed you the assistance program referral.",
        "Plan confirmed under {ref} — ${plan_amount} a month, first payment on the {day}th, and no disconnection while you're on the arrangement.",
    ],
    "outage or service issue": [
        "You're on the outage ticket as {ref}, estimated restoration around {eta_clock}, and you'll get a text when it's back on.",
        "Technician booked within {days} business days for the service connection, reference {ref}. If it goes fully out before then, call us and we'll escalate.",
        "Crews are working it now. I've added your premise to ticket {ref} so you get the automated updates rather than having to call back.",
    ],
    "meter reading": [
        "Your reading of {gj} is in against meter {meter}, the estimate flag is off, and I've added the access note for the reader.",
        "Actual read accepted and the account re-billed — you'll see the adjustment on the next statement under {ref}.",
        "I've submitted the customer read and requested a technician check on the register, reference {ref}.",
    ],
    "general enquiry": [
        "I've emailed you the rate comparison and the rebate program link, and noted the conversation under {ref}.",
        "The equalized plan is set up at ${plan_amount} a month and the account details are updated.",
        "Account updated and the information is on its way to your email. Reference is {ref} if you need it.",
    ],
}

# Customer's reaction to the resolution, by closing tone.
REACTIONS = {
    "negative": [
        "That's not really a resolution, is it. I still have a bill for ${amount} sitting there.",
        "So basically I wait again. I'll be filing a complaint with the regulator.",
        "I'm not satisfied with that at all, but I can see you're not going to do anything else.",
        "Honestly, this is why people switch providers.",
        "Fine. But if this isn't fixed in {days} days I'm calling back and I want a supervisor.",
    ],
    "neutral": [
        "Okay, that works. Thanks for checking.",
        "Alright, I'll watch for the email.",
        "That's fine. I appreciate you explaining it.",
        "Okay, noted. Thanks.",
        "Good, that answers my question.",
    ],
    "positive": [
        "That's great, thank you — you've been really helpful.",
        "Honestly, that's a huge relief. Thank you for taking the time.",
        "Perfect. You've explained that better than anyone has so far, thank you.",
        "Thank you so much, that's exactly what I needed.",
        "I really appreciate you sorting that out so quickly.",
    ],
}

CLOSINGS = {
    "voice": {
        "negative": [
            "I understand you're not happy with the outcome, and I've documented that on the file. Is there anything else I can look at today?",
            "I'm sorry I couldn't give you a better answer. The reference is {ref} if you do call back.",
        ],
        "neutral": [
            "Anything else I can help with while I have the account open?",
            "You should see that come through shortly. Thanks for calling {brand}.",
        ],
        "positive": [
            "That's what we're here for. Thanks for calling {brand}, and have a good rest of your day.",
            "Glad we got that sorted. Take care, and thanks for being a {brand} customer.",
        ],
    },
    "chat": {
        "negative": [
            "I've logged your dissatisfaction on the file under {ref}. Is there anything else before I close the chat?",
            "Sorry again that this wasn't the answer you wanted. The chat transcript will be emailed to you.",
        ],
        "neutral": [
            "Anything else I can help with today?",
            "I'll email the transcript over. Thanks for chatting with {brand}.",
        ],
        "positive": [
            "Happy to help! I'll send the transcript to your email. Have a great day.",
            "Glad that worked out. Thanks for chatting with {brand}!",
        ],
    },
    "email": {
        "negative": [
            "I am sorry this did not resolve the way you hoped. Your file is noted under {ref} and you may reply to this email to escalate.",
            "I appreciate your patience. If you wish to take this further, quote reference {ref} in your reply.",
        ],
        "neutral": [
            "If anything is unclear, reply to this email and quote reference {ref}. Thank you for contacting {brand}.",
            "Please reply if you need anything further. Reference {ref}.",
        ],
        "positive": [
            "Thank you for your patience and for being a {brand} customer. Reference {ref} if you need us again.",
            "It was a pleasure to help. Reply any time quoting {ref}. Kind regards, {agent}.",
        ],
    },
}

# Optional final customer turn, so transcripts don't all end agent-last.
SIGNOFFS = {
    "negative": ["Yeah. Goodbye.", "No. That's all.", "Nothing else. Thanks for nothing.",
                 "No, I'm done. Bye."],
    "neutral": ["No, that's everything. Thanks.", "That's all, thank you.",
                "Nope, that covers it. Bye.", "All good, thanks."],
    "positive": ["No, that's everything — thanks again, {agent}.", "That's all, you've been great. Bye!",
                 "Nothing else, thank you so much.", "All sorted, thanks for your help."],
}

# Email customers usually quote their own account details up front, so the
# verification exchange gets folded into their opening message.
EMAIL_ACCOUNT_SUFFIX = [
    " My account number is {acct} and the service address is {street}, {city}.",
    " For reference, the account is {acct} ({street}, {city}).",
    " Account {acct}, service address {street}.",
]

# Extra customer colour, sprinkled in mid-conversation.
TONE_SPICE = {
    "negative": [
        "This is ridiculous.",
        "I've been a customer for eleven years, by the way.",
        "I shouldn't have to call three times for this.",
        "And I was on hold for twenty minutes to hear that.",
        "You can imagine how that lands when money is tight.",
    ],
    "neutral": [
        "Okay.",
        "Right, I see.",
        "Mm-hm, go on.",
        "That makes sense.",
    ],
    "positive": [
        "Okay, that's good to hear.",
        "Thanks for being patient with my questions.",
        "You're being very clear, I appreciate it.",
    ],
}

# Sarcasm, used sparingly and only from customers.
SARCASM = [
    "Oh fantastic. Another month of guessing games.",
    "Wonderful. Can't wait for that email I'll never receive.",
    "Great system you've got there, really world class.",
    "Sure, because waiting another {days} days is exactly what I hoped for.",
    "Lovely. I'll just heat the house with the statement, shall I.",
]

# Agent empathy lines used when the customer opens angry.
EMPATHY = [
    "I hear you, and I'd be frustrated too. Let me dig into the detail with you.",
    "I'm sorry — that's not the experience we want you to have. Let's get to the bottom of it.",
    "Completely understandable. Give me a moment and I'll go through it line by line with you.",
    "I apologize for the run-around. I'll stay with this until we have an answer.",
]

# Agent acknowledgement when the customer turns the corner (mixed arcs).
TURNAROUND = [
    "I'm glad that helps. Let me make sure the rest of it is sorted too.",
    "That's what I'm here for — let me finish setting it up properly.",
    "Good, I'm pleased we got somewhere with it.",
]


# =========================================================================
# Channel styling: disfluencies for voice, typos for chat and email
# =========================================================================
TYPO_WORDS = {
    "the": "teh", "you": "yuo", "and": "adn", "because": "becuase",
    "receive": "recieve", "definitely": "definately", "payment": "paymnet",
    "account": "acount", "really": "realy", "about": "abotu",
    "think": "thnik", "that": "taht", "with": "wiht", "would": "woudl",
    "please": "plase", "thanks": "thakns", "service": "servcie",
    "reading": "readign", "bill": "bil", "month": "momth",
}

VOICE_FILLERS = ["um, ", "uh, ", "so, ", "I mean, ", "look, "]

CHAT_SHORTHAND = {"please": "pls", "thanks": "thx", "okay": "ok", "you": "u"}

# Chance a line picks up channel-specific noise, by channel and speaker.
NOISE_CHANCE = {
    ("chat", "Customer"): 0.45,
    ("chat", "Agent"): 0.12,
    ("email", "Customer"): 0.25,
    ("email", "Agent"): 0.05,
    ("voice", "Customer"): 0.22,
    ("voice", "Agent"): 0.08,
}


def typo_word(word):
    """Mangle one word: known misspelling, dropped letter, or swapped pair."""
    bare = word.strip(".,!?").lower()
    if bare in TYPO_WORDS and len(bare) > 2:
        return word.lower().replace(bare, TYPO_WORDS[bare])
    if len(word) < 4:
        return word
    i = random.randint(1, len(word) - 2)
    style = random.random()
    if style < 0.4:  # drop a letter
        return word[:i] + word[i + 1:]
    if style < 0.75:  # swap two letters
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]  # double a letter


def add_typos(text, n=2):
    """Introduce up to n typos into a line of text.

    Dollar amounts, account/meter identifiers, [REDACTED] and proper nouns are
    left alone -- mangling them produces nonsense ("the 25th of May" -> "of my")
    rather than the look of someone typing quickly.
    """
    words = text.split()
    eligible = [
        i for i, w in enumerate(words)
        if len(w) > 3
        and not any(c.isdigit() for c in w)
        and not any(c in w for c in "$[]@")
        and (i == 0 or w[0].islower())
    ]
    if not eligible:
        return text
    for idx in random.sample(eligible, min(n, len(eligible))):
        words[idx] = typo_word(words[idx])
    return " ".join(words)


def style_line(text, speaker, channel):
    """Apply channel-realistic noise to a finished line."""
    if random.random() > NOISE_CHANCE[(channel, speaker)]:
        return text

    if channel == "voice":
        # Transcribed speech: filler words, not typos.
        return random.choice(VOICE_FILLERS) + text[0].lower() + text[1:]

    if channel == "chat":
        text = add_typos(text, n=random.choice([1, 1, 2]))
        if speaker == "Customer":
            for long, short in CHAT_SHORTHAND.items():
                if random.random() < 0.3:
                    text = text.replace(long, short).replace(long.capitalize(), short)
            if random.random() < 0.5:
                text = text.lower()
            if random.random() < 0.4:
                text = text.rstrip(".")
        return text

    # email: a few typos, no lowercasing or shorthand
    return add_typos(text, n=1)


# =========================================================================
# Transcript assembly
# =========================================================================
def pick_arc(true_sentiment):
    """Return (opening_tone, closing_tone).

    Most conversations hold one tone, but a deliberate minority are mixed --
    an angry opening that ends well, or a calm one that sours. That is the
    noise a sentiment model has to cope with in real transcripts.
    """
    roll = random.random()
    if true_sentiment == "positive":
        return ("negative", "positive") if roll < 0.25 else ("positive", "positive")
    if true_sentiment == "negative":
        return ("neutral", "negative") if roll < 0.20 else ("negative", "negative")
    return ("negative", "neutral") if roll < 0.15 else ("neutral", "neutral")


def n_middle_pairs(channel, handle_rank):
    """Longer contacts get more middle exchanges (handle_rank is 0..1)."""
    low, high = MIDDLE_PAIRS[channel]
    return int(round(low + handle_rank * (high - low)))


def build_transcript(row, agent_first_name, handle_rank):
    """Assemble one dialogue as a newline-joined list of labelled turns.

    Turns are appended in strict Agent/Customer alternation -- extras like
    sarcasm and the mixed-arc hinge are folded into an existing turn rather
    than added as their own, so the speakers never double up.
    """
    ctx = build_context(row, agent_first_name)
    channel, topic = row["channel"], row["topic"]
    open_tone, close_tone = pick_arc(row["true_sentiment"])

    turns = []  # list of (speaker, raw template)

    # 1. Greeting + the customer's reason for contact.
    turns.append(("Agent", random.choice(GREETINGS[channel])))
    opening = random.choice(OPENINGS[topic][open_tone])

    # Email customers normally quote their account details in the first
    # message, so those threads skip the verification back-and-forth.
    verified_inline = channel == "email" and random.random() < 0.55
    if verified_inline:
        opening += random.choice(EMAIL_ACCOUNT_SUFFIX)
    turns.append(("Customer", opening))

    # 2. Verification exchange (empathy first if they opened hot).
    if not verified_inline:
        ask = random.choice(VERIFY_ASK)
        if open_tone == "negative":
            ask = random.choice(EMPATHY) + " " + ask
        turns.append(("Agent", ask))
        turns.append(("Customer", random.choice(VERIFY_GIVE)))

    # 3. Topic-specific middle. Indices are sampled without replacement (no
    #    repeated probe) and then SORTED, because each topic's bank is written
    #    in narrative order -- an agent must offer the billing review before
    #    referring back to it.
    n_pairs = n_middle_pairs(channel, handle_rank)
    bank = MIDDLE[topic]
    chosen = sorted(random.sample(range(len(bank)), min(n_pairs, len(bank))))

    # One sarcastic customer line, rare and negative-leaning. Appended to a
    # real turn so alternation holds.
    used_spice = set()  # so the same aside never appears twice in one dialogue
    sarcasm_at = None
    if open_tone == "negative" and chosen and random.random() < 0.12:
        sarcasm_at = random.choice(chosen)

    for position, idx in enumerate(chosen):
        agent_line, customer_line = bank[idx]
        if position == 0 and not agent_line.startswith("Thank"):
            agent_line = random.choice(VERIFY_CONFIRM) + " " + agent_line
        turns.append(("Agent", agent_line))

        # Tone colour: the opening tone governs early turns, the closing tone
        # the later ones, which is what makes a mixed arc readable.
        tone_now = open_tone if position < len(chosen) / 2 else close_tone
        if random.random() < 0.30:
            fresh = [s for s in TONE_SPICE[tone_now] if s not in used_spice]
            if fresh:
                spice = random.choice(fresh)
                used_spice.add(spice)
                customer_line += " " + spice
        if idx == sarcasm_at:
            customer_line += " " + random.choice(SARCASM)
        turns.append(("Customer", customer_line))

    # 4. Resolution (prefixed with the hinge line on a mixed arc), the
    #    customer's reaction, and the agent's close.
    resolution = random.choice(RESOLUTIONS[topic])
    if open_tone != close_tone and close_tone == "positive":
        resolution = random.choice(TURNAROUND) + " " + resolution
    turns.append(("Agent", resolution))
    turns.append(("Customer", random.choice(REACTIONS[close_tone])))
    turns.append(("Agent", random.choice(CLOSINGS[channel][close_tone])))

    # 5. Optional customer sign-off, so not every transcript ends agent-last.
    #    Forced when we are one turn short of the 6-turn floor.
    if len(turns) < 6 or random.random() < 0.45:
        turns.append(("Customer", random.choice(SIGNOFFS[close_tone])))

    # Fill placeholders, then apply channel styling.
    lines = []
    for speaker, template in turns:
        text = template.format(**ctx)
        lines.append(f"{speaker}: {style_line(text, speaker, channel)}")
    return "\n".join(lines)


def main():
    if not INTERACTIONS_PATH.exists():
        raise SystemExit(f"{INTERACTIONS_PATH} not found -- run src/generate_interactions.py first.")

    interactions = pd.read_csv(INTERACTIONS_PATH)
    agents = pd.read_csv(AGENTS_PATH)

    # Agents introduce themselves by first name only.
    first_names = dict(
        zip(agents["agent_id"], agents["agent_name"].str.split().str[0])
    )

    # Rank handle time within each channel (0..1) so long contacts get long
    # transcripts. Deterministic -- no randomness involved.
    interactions["_rank"] = interactions.groupby("channel")["handle_time_sec"].rank(pct=True)

    transcripts = [
        build_transcript(row, first_names[row["agent_id"]], row["_rank"])
        for _, row in interactions.iterrows()
    ]
    interactions["transcript_text"] = transcripts
    interactions = interactions.drop(columns="_rank")
    interactions.to_csv(INTERACTIONS_PATH, index=False)

    # --- Summary ---------------------------------------------------------
    turn_counts = [t.count("\n") + 1 for t in transcripts]
    print(f"Filled {len(transcripts)} transcripts in {INTERACTIONS_PATH}")
    print(f"Turns per transcript: min {min(turn_counts)}, max {max(turn_counts)}, "
          f"mean {sum(turn_counts) / len(turn_counts):.1f}")
    print(f"Mean characters: {int(np.mean([len(t) for t in transcripts]))}")
    print(f"Transcripts containing {REDACTED}: {sum(REDACTED in t for t in transcripts)}")

    # Every transcript must strictly alternate Agent / Customer and open with
    # the agent. Cheap invariant, catches assembly bugs immediately.
    bad_alternation = 0
    for t in transcripts:
        speakers = [line.split(":", 1)[0] for line in t.split("\n")]
        if speakers[0] != "Agent" or any(a == b for a, b in zip(speakers, speakers[1:])):
            bad_alternation += 1
    print(f"Transcripts with broken Agent/Customer alternation: {bad_alternation}")
    print(f"Turn counts inside the 6-14 contract: "
          f"{all(6 <= c <= 14 for c in turn_counts)}")
    print(f"Digit-run check (no 13-19 digit sequences): "
          f"{not interactions['transcript_text'].str.contains(r'\d{13,19}').any()}\n")

    # --- 5 samples, deliberately spread across channel / topic / sentiment --
    print("=" * 78)
    print("SAMPLE TRANSCRIPTS")
    print("=" * 78)
    samples = pd.concat([
        interactions[(interactions.channel == "voice") & (interactions.topic == "high bill dispute") & (interactions.true_sentiment == "negative")].head(1),
        interactions[(interactions.channel == "chat") & (interactions.topic == "move-in/move-out") & (interactions.true_sentiment == "positive")].head(1),
        interactions[(interactions.channel == "email") & (interactions.topic == "payment arrangement")].head(1),
        interactions[(interactions.channel == "voice") & (interactions.topic == "outage or service issue") & (interactions.true_sentiment == "neutral")].head(1),
        interactions[(interactions.channel == "chat") & (interactions.topic == "meter reading")].head(1),
    ])
    for _, r in samples.iterrows():
        print(f"\n--- {r.interaction_id} | {r.channel} | {r.topic} | "
              f"true_sentiment={r.true_sentiment} | agent={r.agent_id} | "
              f"AHT={r.handle_time_sec}s ---")
        print(r.transcript_text)
    print()


if __name__ == "__main__":
    main()
