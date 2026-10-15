# Data dictionary

Every table and every field in `utility-cx-insights`, plus the exact definition of
each KPI.

**All data is synthetic** (see `CLAUDE.md`). There is no real utility, no real
customers and no real agents. Each table below names the kind of system it
*mimics*, so the shape of the model is recognisable to someone who works with the
real thing.

**How the data flows**

```
  raw/          five source extracts, as a real centre would receive them
    ↓           src/run_pipeline.py      — sentiment, topic and coaching scores
  labels/       working artefacts: model output, human labels, validation
    ↓           src/build_model_tables.py — star schema
  processed/    seven tables the Power BI report is built on
```

**Conventions used throughout**

- `*_key` is a surrogate integer key; `*_id` is the natural key from the source.
- `is_*` / `has_*` / `had_*` are booleans. Sum them to count.
- ⚠️ marks a field that is easy to misuse. ✅ marks the one to prefer.
- Seconds are stored, never minutes. Percentages are not stored at all — see
  *Why there are almost no percentage columns* at the end.

---

## 1. Source extracts — `data/raw/`

These five files stand in for what a contact centre would receive from its
operational systems each night.

| File | Mimics |
|---|---|
| `agents.csv` | Workforce management / HR system |
| `interactions.csv` | ACD / contact routing platform + speech-to-text |
| `cases.csv` | CRM case management |
| `csat_survey.csv` | Post-contact survey platform |
| `sqm_monthly.csv` | Quality management scorecard |

### `agents`

**Grain:** one row per agent · **Rows:** 20 · **Fields:** 7

The CCR roster: who works here, on which team, with what targets.

*Mimics:* Workforce management / HR system

| Field | Type | Description |
|---|---|---|
| `agent_id` | text | Natural key for the agent, `A001`–`A020`. |
| `agent_name` | text | Full name. Synthetic (Faker, Canadian locale). |
| `team` | text | Which of the four teams the agent belongs to. |
| `team_lead` | text | Name of the team's leader. |
| `hire_date` | text | Start date. Spread over roughly six years. |
| `csat_target` | decimal | The agent's CSAT target as a proportion, 0.80–0.85. |
| `aht_target_sec` | integer | The agent's average handle time target in seconds, 420–480. **Varies per agent** — never average it across agents. |

### `interactions`

**Grain:** one row per contact · **Rows:** 22,120 · **Fields:** 10

One row per customer contact. The spine of everything else.

*Mimics:* ACD / contact routing platform, plus a speech-to-text engine for `transcript_text`

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | Natural key for the contact, `I00001`–`I02000`. |
| `agent_id` | text | Who handled it. Joins `agents`. |
| `start_datetime` | text | When the contact started. Business hours, Mountain Time, weekdays only. |
| `channel` | text | `voice`, `chat` or `email`. Roughly 65 / 25 / 10. |
| `queue` | text | The routing queue. Each queue is owned by exactly one team. |
| `topic` | text | The contact reason, one of seven. This is the **actual** topic, as routed. |
| `handle_time_sec` | integer | Time spent handling the contact. **Excludes hold** in this dataset. |
| `hold_time_sec` | integer | Time the customer spent on hold. 0 for most contacts. |
| `transcript_text` | text | The conversation, one turn per line, each prefixed `Agent:` or `Customer:`. Payment details always appear as `[REDACTED]`. |
| `true_sentiment` | text | ⚠️ **Synthetic ground truth.** The mood the generator intended before writing the transcript. Has no real-world equivalent — it exists only to measure the model. Never expose it in an operational report. |

### `cases`

**Grain:** one row per case · **Rows:** 1,823 · **Fields:** 8

Work items opened off the back of a contact. Not every contact creates one.

*Mimics:* CRM case management

| Field | Type | Description |
|---|---|---|
| `case_id` | text | Natural key, `C00001`+. |
| `interaction_id` | text | The contact that opened the case. Joins `interactions`. |
| `customer_id` | text | The customer. Repeats across contacts — that is what makes a repeat contact possible. |
| `case_type` | text | Mirrors the interaction `topic`, so the two files join cleanly. |
| `status` | text | `closed`, `open` or `escalated`. |
| `created_date` | text | Date the case was opened (the contact date). |
| `closed_date` | text | Date it closed. Empty while open. |
| `repeat_contact_7d` | text | `Y` if this customer contacted again within 7 days. **Forward-looking** — it marks the first contact of a pair, not the follow-up. |

### `csat_survey`

**Grain:** one row per survey response · **Rows:** 300 · **Fields:** 5

Survey responses. Roughly 15% of contacts, and not a random 15%.

*Mimics:* Post-contact survey platform

| Field | Type | Description |
|---|---|---|
| `response_id` | text | Natural key, `R0001`+. |
| `interaction_id` | text | The contact being rated. |
| `csat_score` | integer | 1–5. Correlated with true sentiment but deliberately noisy. |
| `resolved` | text | `Y`/`N` — did the customer consider it resolved? |
| `comment` | text | Free-text comment. Often blank. |

### `sqm_monthly`

**Grain:** one row per month × team · **Rows:** 24 · **Fields:** 4

Monthly first-contact-resolution by team against an industry benchmark.

*Mimics:* Quality management / service quality scorecard

| Field | Type | Description |
|---|---|---|
| `month` | text | `YYYY-MM`. |
| `team` | text | Team name. |
| `fcr_pct` | decimal | First contact resolution, 0–100. ⚠️ A **designed series**, not derived from the case log — see the KPI section. |
| `industry_benchmark_fcr_pct` | decimal | External benchmark, around 74%. |

---

## 2. Model tables — `data/processed/`

The star schema the Power BI report reads. Built by `src/build_model_tables.py`;
relationships are documented in `powerbi/model_relationships.md`.

*Mimics:* nothing external — this is the analytical layer this project builds.

### `dim_date`

**Grain:** one row per calendar day · **Rows:** 2,557 · **Fields:** 27

| Field | Type | Description |
|---|---|---|
| `date_key` | integer | **PK.** `YYYYMMDD` as an integer. |
| `date` | text | The calendar date. Mark the table as a date table on this column. |
| `year` | integer | Calendar year. |
| `quarter` | integer | Calendar quarter 1–4. |
| `quarter_label` | text | e.g. `Q2 2026`. |
| `month_number` | integer | 1–12. |
| `month_name` | text | e.g. `April`. Sort by `month_name_sort`. |
| `month_short` | text | e.g. `Apr`. |
| `month_key` | integer | `YYYYMM` for month-grain joins. |
| `month_label` | text | e.g. `Apr 2026`. |
| `month_name_sort` | integer | Sort-by column for `month_name`. |
| `day_of_month` | integer | 1–31. |
| `day_of_year` | integer | 1–366. |
| `week_of_year` | integer | ISO week number. |
| `day_of_week` | integer | **Monday = 1**, ISO convention. |
| `weekday_name` | text | e.g. `Monday`. Sort by `weekday_name_sort`. |
| `weekday_short` | text | e.g. `Mon`. |
| `weekday_name_sort` | integer | Sort-by column for `weekday_name`. |
| `is_weekend` | true/false | Saturday or Sunday. |
| `is_holiday` | true/false | An Alberta statutory holiday, with weekend observance shifted to the Monday. |
| `holiday_name` | text | Name of the holiday, blank otherwise. |
| `is_business_day` | true/false | Weekday **and** not a statutory holiday. 71 contacts fall on a stat holiday — the generator uses business days but does not skip them. |
| `fiscal_year` | integer | **Fiscal year starts 1 April**, named for the year it ends in. April 2026 is FY2027. |
| `fiscal_quarter` | integer | 1–4 with April = Q1. |
| `fiscal_quarter_label` | text | e.g. `FY2027 Q1`. |
| `fiscal_month_number` | integer | 1–12 with April = 1. |
| `fiscal_period_label` | text | e.g. `FY2027 P01`. |

### `dim_team`

**Grain:** one row per team · **Rows:** 4 · **Fields:** 5

| Field | Type | Description |
|---|---|---|
| `team_key` | integer | **PK.** |
| `team_name` | text | The only place a team name lives in the model. |
| `team_lead` | text | Team leader's name. |
| `agent_count` | integer | CCRs on the team. |
| `team_lead_email` | text | Identity key for the **Team Lead** RLS role. Synthetic, on a reserved `.example.com` domain. |

### `dim_agent`

**Grain:** one row per agent · **Rows:** 20 · **Fields:** 13

| Field | Type | Description |
|---|---|---|
| `agent_key` | integer | **PK.** |
| `agent_id` | text | Natural key from the source system. |
| `agent_name` | text | Full name. |
| `agent_email` | text | Identity key for the **CCR** RLS role, matched against `USERPRINCIPALNAME()`. Unique by construction. Synthetic, reserved domain. |
| `team_key` | integer | **FK →** `dim_team`. |
| `hire_date` | text | Start date. |
| `hire_date_key` | integer | **FK →** `dim_date` (inactive relationship). |
| `tenure_days` | integer | Days of service as at 2026-10-04. |
| `tenure_months` | decimal | The same in months. |
| `tenure_band` | text | `Under 6 months` / `6 months – 2 years` / `2 – 5 years` / `5 years +`. |
| `is_new_hire` | true/false | Under 180 days. Three of twenty. Read their figures separately — new hires carry ~1.5× the odds of a negative contact. |
| `csat_target` | decimal | Per-agent CSAT target, 0.80–0.85. |
| `aht_target_sec` | integer | Per-agent AHT target. **Sum per row, never average across agents.** |

### `dim_topic`

**Grain:** one row per contact reason · **Rows:** 7 · **Fields:** 5

| Field | Type | Description |
|---|---|---|
| `topic_key` | integer | **PK.** |
| `topic_name` | text | One of the seven contact reasons. |
| `topic_group` | text | Roll-up: Billing / Metering / Lifecycle / Service / Credit / General. |
| `is_financial_hardship` | true/false | High bill dispute and payment arrangement. |
| `is_complaint_prone` | true/false | High bill dispute and outage — the two with the highest negative rates. |

### `dim_channel`

**Grain:** one row per channel · **Rows:** 3 · **Fields:** 5

| Field | Type | Description |
|---|---|---|
| `channel_key` | integer | **PK.** |
| `channel_name` | text | voice / chat / email. |
| `channel_group` | text | `Assisted` (voice, chat) or `Deferred` (email). |
| `is_realtime` | true/false | True for voice and chat. |
| `channel_sort` | integer | Sort-by column for `channel_name`. |

### `fact_interactions`

**Grain:** one row per contact · **Rows:** 2,000 · **Fields:** 44

| Field | Type | Description |
|---|---|---|
| `interaction_key` | integer | **PK**, surrogate. |
| `interaction_id` | text | Degenerate dimension — the source system's contact reference. |
| `date_key` | integer | **FK →** `dim_date`, active. |
| `agent_key` | integer | **FK →** `dim_agent`, active. |
| `topic_key` | integer | **FK →** `dim_topic`, active. The **actual** topic. |
| `topic_predicted_key` | integer | **FK →** `dim_topic`, **inactive**. What the classifier predicted. |
| `channel_key` | integer | **FK →** `dim_channel`, active. |
| `start_datetime` | text | Contact start. |
| `start_hour` | integer | Hour of day, 0–23, for intraday analysis. |
| `queue` | text | Routing queue. |
| `handle_time_sec` | integer | Additive. Safe to sum. |
| `hold_time_sec` | integer | Additive. Safe to sum. |
| `had_hold` | true/false | Any hold at all. |
| `sentiment_label` | text | Model's overall verdict: positive / neutral / negative. ⚠️ Reads systematically too negative — 61% agreement with a human reader. Prefer `sentiment_end`. |
| `sentiment_score` | decimal | P(positive) − P(negative) across all customer turns, −1 to +1. |
| `sentiment_start` | decimal | The same score over the **first third** of the customer's turns. |
| `sentiment_end` | decimal | The same over the **last third**. ✅ The most trustworthy sentiment field — 77% agreement with a human against 56% for the whole-conversation score. |
| `sentiment_shift` | decimal | `sentiment_end − sentiment_start`. Positive means the contact recovered. |
| `true_sentiment` | text | ⚠️ Synthetic ground truth. Hide from reports except on an accuracy visual. |
| `sentiment_is_correct` | true/false | Did `sentiment_label` match `true_sentiment`? Synthetic-only. |
| `topic_predicted` | text | The rules classifier's topic. 99.5% accurate. |
| `topic_is_correct` | true/false | Did the prediction match the actual topic? |
| `has_score` | true/false | The contact carries model scores. |
| `has_case` | true/false | A case was opened. ~91% of contacts. |
| `case_id` | text | Case reference, blank if none. |
| `customer_id` | text | Customer reference. |
| `case_status` | text | closed / open / escalated. |
| `case_created_date_key` | decimal | **FK →** `dim_date`, inactive. |
| `case_closed_date_key` | decimal | **FK →** `dim_date`, inactive. Blank while open. |
| `days_to_close` | decimal | Days from open to close. Blank while open — excluded from averages, not counted as zero. |
| `is_repeat_contact_7d` | true/false | Customer contacted again within 7 days. Forward-looking. |
| `is_escalated` | true/false | Case was escalated. |
| `is_case_open` | true/false | Case is open or escalated. |
| `has_survey_response` | true/false | The customer answered the survey. ~15%. |
| `survey_response_id` | text | Survey reference, blank if none. |
| `csat_score` | decimal | 1–5, blank if no response. |
| `is_resolved` | text | Customer said it was resolved. **Tri-state** — blank where there is no response. |
| `is_csat_promoter` | true/false | CSAT 4 or 5. |
| `is_csat_detractor` | true/false | CSAT 1 or 2. |
| `survey_comment` | text | Free-text survey comment. |
| `coaching_note` | text | The 1–2 sentence coaching note. The most sensitive field in the model — it is about an individual's performance. |
| `generated_by` | text | `ollama:llama3.1:8b` or `rules`, recording which engine wrote the note. |
| `model_version` | text | Which models produced the scores on this row. |
| `scored_at` | text | UTC timestamp of scoring. The one non-reproducible column by design. |

### `fact_sqm_monthly`

**Grain:** one row per month × team · **Rows:** 24 · **Fields:** 8

| Field | Type | Description |
|---|---|---|
| `sqm_key` | integer | **PK.** |
| `date_key` | integer | **FK →** `dim_date`, the **first of the month**. |
| `month_key` | integer | `YYYYMM`. |
| `team_key` | integer | **FK →** `dim_team`. |
| `fcr_pct` | decimal | ⚠️ A stored rate with no counts behind it. Never sum; average only with care. |
| `industry_benchmark_fcr_pct` | decimal | External benchmark. |
| `fcr_vs_benchmark_pp` | decimal | Gap to benchmark in percentage points. |
| `is_above_benchmark` | true/false | Beat the benchmark this month. |

### `interaction_scores`

**Grain:** one row per contact · **Rows:** 2,000 · **Fields:** 10

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | Joins `interactions`. |
| `sentiment_label` | text | See `fact_interactions`. |
| `sentiment_score` | decimal | See `fact_interactions`. |
| `sentiment_start` | decimal | See `fact_interactions`. |
| `sentiment_end` | decimal | See `fact_interactions`. |
| `topic_predicted` | text | See `fact_interactions`. |
| `coaching_note` | text | See `fact_interactions`. |
| `generated_by` | text | See `fact_interactions`. |
| `model_version` | text | See `fact_interactions`. |
| `scored_at` | text | See `fact_interactions`. |

---

## 3. Working artefacts — `data/labels/`

Intermediate output from the modelling and validation steps. **Not part of the
Power BI model** — kept because the validation argument depends on them.

### `sentiment_roberta`

**Rows:** 2,000 · **Fields:** 5

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | See `fact_interactions`. |
| `sentiment_label` | text | See `fact_interactions`. |
| `sentiment_score` | decimal | See `fact_interactions`. |
| `sentiment_start` | decimal | See `fact_interactions`. |
| `sentiment_end` | decimal | See `fact_interactions`. |

### `sentiment_vader`

**Rows:** 2,000 · **Fields:** 5

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | See `fact_interactions`. |
| `sentiment_label` | text | See `fact_interactions`. |
| `sentiment_score` | decimal | See `fact_interactions`. |
| `sentiment_start` | decimal | See `fact_interactions`. |
| `sentiment_end` | decimal | See `fact_interactions`. |

### `topics_rules`

**Rows:** 2,000 · **Fields:** 4

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | Joins `interactions`. |
| `topic_actual` | text | The routed topic. |
| `topic_predicted` | text | The classifier's answer. |
| `correct` | true/false | Did they match? |

### `coaching_notes`

**Rows:** 2,000 · **Fields:** 11

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | Joins `interactions`. |
| `agent_id` | text | Who handled it. |
| `agent_name` | text | Agent name. |
| `team` | text | Team. |
| `topic` | text | Contact reason. |
| `sentiment_label` | text | Model verdict. |
| `sentiment_shift` | decimal | End minus start. |
| `hold_time_sec` | integer | Hold seconds. |
| `handle_vs_target_pct` | decimal | Handle time against the agent's own target. |
| `coaching_note` | text | The note. |
| `generated_by` | text | Which engine wrote it. |

### `to_label`

**Rows:** 1,114 · **Fields:** 4

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | Joins `interactions`. |
| `channel` | text | Shown for context. |
| `transcript_text` | text | The conversation to read. |
| `human_label` | decimal | Empty on issue; the labeller fills it in. **No model output or topic is included** — the labels must be independent. |

### `to_label_manifest`

**Rows:** 100 · **Fields:** 4

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | Joins `to_label`. |
| `topic` | text | Withheld from the labeller; kept here for analysis. |
| `channel` | text | Stratum component. |
| `stratum` | text | `topic | channel`, the sampling cell. |

### `to_label_priority`

**Rows:** 38 · **Fields:** 1

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | Contacts where the two models disagree — the highest-value rows for a human to label next. |

### `human_labels`

**Rows:** 104 · **Fields:** 3

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | Joins `to_label`. |
| `human_label` | text | The human verdict. Append log — last write wins. |
| `labelled_at` | text | UTC timestamp. |

### `ai_labels`

**Rows:** 100 · **Fields:** 4

| Field | Type | Description |
|---|---|---|
| `interaction_id` | text | Joins `to_label`. |
| `ai_label` | text | An LLM judge's blind verdict. **A second opinion, not ground truth.** |
| `labelled_by` | text | `ollama:llama3.1:8b` or `failed`. |
| `labelled_at` | text | UTC timestamp. |
---

## 4. KPI definitions

Every measure, with its exact formula. The DAX lives in `powerbi/measures.dax`.

**Reading the formulas:** `DIVIDE(a, b)` returns blank rather than erroring when
`b` is zero. "pp" means percentage points — the change in a percentage is not
itself a percentage, and conflating the two is the most common error on a
contact centre dashboard.

### Volume

| KPI | Definition | Formula |
|---|---|---|
| **Contacts** | Number of contacts handled. | `COUNTROWS(fact_interactions)` |
| **Customers** | Distinct customers who made contact. Lower than Contacts, because some come back. | `DISTINCTCOUNT(customer_id)` |
| **Contacts per Business Day** | Daily average, excluding weekends and Alberta statutory holidays. | `Contacts ÷ COUNTROWS(dim_date WHERE is_business_day)` |

### Efficiency

| KPI | Definition | Formula |
|---|---|---|
| **AHT (minutes)** | Average handle time. **Excludes hold** in this dataset. | `SUM(handle_time_sec) ÷ (Contacts × 60)` |
| **Hold (minutes)** | Average hold time, across all contacts including those with no hold. | `SUM(hold_time_sec) ÷ (Contacts × 60)` |
| **Hold Rate %** | Share of contacts involving any hold. | `Contacts WHERE had_hold ÷ Contacts` |
| **AHT Target (minutes)** | The weighted target for the contacts in view. | `SUMX(fact, RELATED(aht_target_sec)) ÷ (Contacts × 60)` |
| **AHT vs Target %** | How far handle time runs over (+) or under (−) target. | `(SUM(handle_time_sec) − SUMX(fact, RELATED(aht_target_sec))) ÷ SUMX(fact, RELATED(aht_target_sec))` |
| **% Within AHT Target** | Share of contacts at or under the agent's own target. Catches a long tail the average hides. | `COUNTROWS(fact WHERE handle_time_sec ≤ RELATED(aht_target_sec)) ÷ Contacts` |

> **Why AHT vs Target sums the targets rather than averaging them.** Every agent
> has their own target (420–480s). Averaging `aht_target_sec` across agents would
> weight a 10-contact agent the same as a 170-contact one. The formula sums each
> contact's own target instead.

### Survey

| KPI | Definition | Formula |
|---|---|---|
| **Survey Responses** | Contacts where the customer answered. | `Contacts WHERE has_survey_response` |
| **Survey Response Rate %** | Share of contacts producing a response. | `Survey Responses ÷ Contacts` |
| **CSAT %** | Top-2-box: share of **respondents** scoring 4 or 5. | `Contacts WHERE is_csat_promoter ÷ Survey Responses` |
| **CSAT Detractor %** | Share of respondents scoring 1 or 2. | `Contacts WHERE is_csat_detractor ÷ Survey Responses` |
| **Avg CSAT Score** | Mean 1–5 score among respondents. | `AVERAGE(csat_score)` |
| **CSAT Target %** | The target for the agents in view, weighted by their response volume. | `SUMX(respondents, RELATED(csat_target)) ÷ Survey Responses` |
| **CSAT vs Target (pp)** | Performance against target. | `(CSAT % − CSAT Target %) × 100` |
| **Survey Resolved %** | Share of respondents who said it was resolved. Not an FCR rate. | `Contacts WHERE is_resolved ÷ Survey Responses` |

> ⚠️ **Every survey KPI is biased low.** Only ~15% of customers respond, and
> unhappy ones respond more often. The denominator is respondents, not customers.
> Measured CSAT is 36.3%; the true population sentiment is better than that.

### First contact resolution

| KPI | Definition | Formula |
|---|---|---|
| **FCR % (SQM)** | First contact resolution from the monthly scorecard. | `AVERAGE(fcr_pct)` |
| **FCR Benchmark %** | The external benchmark, ~74%. | `AVERAGE(industry_benchmark_fcr_pct)` |
| **FCR vs Benchmark (pp)** | Gap to benchmark. | `FCR % − FCR Benchmark %` |
| **Months Above Benchmark** | Count of month × team cells beating it. Shows whether a gain is broad or one team carrying the average. | `COUNTROWS(fact_sqm WHERE is_above_benchmark)` |

> ⚠️ **FCR is an unweighted mean, and it is the one KPI here that cannot be
> re-aggregated safely.** The source supplies a finished percentage with no
> numerator or denominator, so rolling months or teams together treats a
> 500-contact team the same as a 50-contact one. Trustworthy at month × team
> grain; treat any total as indicative. A production feed should carry
> `resolved_contacts` and `total_contacts` and let the measure compute the ratio.

### Repeat contact and cases

| KPI | Definition | Formula |
|---|---|---|
| **Cases** | Contacts that opened a case (~91%). | `Contacts WHERE has_case` |
| **Repeat Contacts** | Contacts followed by another from the same customer within 7 days. | `Contacts WHERE is_repeat_contact_7d` |
| **Repeat Contact Rate %** | The operational mirror of FCR. Unlike FCR it is built from counts, so it re-aggregates correctly at any grain. | `Repeat Contacts ÷ Cases` |
| **Escalation Rate %** | Share of cases escalated. | `Contacts WHERE is_escalated ÷ Cases` |
| **Avg Days to Close** | Mean days open to close, **closed cases only**. | `AVERAGE(days_to_close)` |

> **The denominator is Cases, not Contacts.** The repeat flag is only meaningful
> where a case exists. And it is **forward-looking** — it marks the first contact
> of a pair, not the follow-up.

### Sentiment

| KPI | Definition | Formula |
|---|---|---|
| **% Negative** | Share of scored contacts labelled negative. | `Contacts WHERE sentiment_label = "negative" ÷ Scored Interactions` |
| **% Positive** | Share labelled positive. | `Contacts WHERE sentiment_label = "positive" ÷ Scored Interactions` |
| **Net Sentiment** | % positive minus % negative, on a −100…+100 scale. Reads like NPS. | `(% Positive − % Negative) × 100` |
| **Avg Sentiment Score** | Mean of the continuous score. Moves before a contact crosses a label boundary, so better for trends. | `AVERAGE(sentiment_score)` |
| **Sentiment Shift** | How far sentiment travelled from the first third of the customer's turns to the last. | `AVERAGE(sentiment_end − sentiment_start)` |
| **Recovered Contacts** | Started poorly, ended well. | `Contacts WHERE sentiment_shift > 0.4` |
| **Deteriorated Contacts** | Started fine, ended badly. | `Contacts WHERE sentiment_shift < −0.3` |
| ✅ **% Negative (Calibrated)** | Negative share judged on the **end** of the contact. | `Contacts WHERE sentiment_end < −0.25 ÷ Scored Interactions` |
| ✅ **Net Sentiment (Calibrated)** | Net sentiment judged on the end of the contact. **The recommended headline.** | `(Contacts WHERE sentiment_end > 0.9 − Contacts WHERE sentiment_end < −0.25) ÷ Scored Interactions × 100` |

> **Prefer the calibrated pair.** Validated against 100 hand-read contacts,
> `sentiment_label` agrees with a human 61% of the time, and **every one of its 39
> errors judged the customer gloomier than the human did**. Scoring the end of the
> conversation lifts held-out agreement to 77%. The −0.25 / +0.90 thresholds are
> the validated band. See `docs/validation.md`.

### Coaching queue

| KPI | Definition | Formula |
|---|---|---|
| **Coaching Priority** | 0–100 score for how much a contact deserves a listen. | `40 if sentiment_end < −0.25` `+ 20 if sentiment_shift < −0.3` `+ 15 if hold > 3 min` `+ 10 if AHT > 25% over target` `+ 10 if repeat contact` `+ 5 if escalated` |
| **Top 10 Coaching Flag** | 1 for the ten highest-priority contacts in the current filter context. Use as a visual filter. | `TOPN(10, ALLSELECTED(interaction_id), [Coaching Priority], DESC)` |
| **Coaching Rank** | Position in the queue, 1 = most urgent. | `RANKX(ALLSELECTED(interaction_id), [Coaching Priority], DESC, DENSE)` |

> The weights are a **judgement call, not a fitted model**. Tune them with a team
> lead. `ALLSELECTED` means the queue re-ranks inside whatever the slicers are set
> to, so a lead filtering to their own team gets their top ten.

### Change over time

Each KPI has a prior-month partner and a change measure.

| Pattern | Formula |
|---|---|
| Prior month (most) | `CALCULATE([KPI], DATEADD(dim_date[date], -1, MONTH))` |
| Prior month (**FCR only**) | `CALCULATE([FCR % (SQM)], PREVIOUSMONTH(dim_date[date]))` |
| Change, counts | `[KPI] − [KPI PM]`, and `÷ [KPI PM]` for a percentage |
| Change, rates | `([KPI] − [KPI PM]) × 100`, reported in **pp** |

> **FCR uses `PREVIOUSMONTH`, not `DATEADD`.** `fact_sqm_monthly` only holds the
> first of each month, so `DATEADD` from any other day lands on a date with no
> scorecard row and returns blank.

MoM measures: `Contacts MoM`, `Contacts MoM %`, `AHT MoM (minutes)`,
`CSAT MoM (pp)`, `Net Sentiment MoM`, `% Negative MoM (pp)`,
`Repeat Contact Rate MoM (pp)`, `FCR MoM (pp)`.

> **Direction matters in conditional formatting.** For AHT, repeat contact rate and
> % negative, **down is good**. Getting this backwards is the most common
> scorecard error.

### Report furniture

| KPI | Definition |
|---|---|
| **Data As Of** | Freshness stamp. Reads the latest contact in the model, **not today's date**, so a failed refresh is visible rather than hidden behind a ticking clock. |
| **Synthetic Data Label** | The provenance banner required on every page. |
| **Model Version** | Which models produced the scores. |

---

## Why there are almost no percentage columns

Only one rate is stored in this model: `fcr_pct`, and only because the source
gives nothing else.

Everything else is stored as its components, because **the mean of per-row
percentages is not the overall percentage**. If a 10-minute call and a 2-minute
call are each 20% over target, the pair is not "20% over" in any useful sense —
the right answer depends on the totals.

```
WRONG   AVERAGE(handle_time_sec / aht_target_sec) − 1
RIGHT   SUM(handle_time_sec) / SUMX(fact, RELATED(aht_target_sec)) − 1
```

A test in `tests/test_powerbi_artifacts.py` fails the build if a `*_pct`, `*_rate`
or `*_ratio` column appears in `fact_interactions`.

---

## Known limitations

| | |
|---|---|
| **Sentiment reads too negative** | 61% agreement with a human; all errors in one direction. Use `sentiment_end`. See `docs/validation.md`. |
| **Survey CSAT is biased low** | 15% response rate, skewed to the unhappy. |
| **FCR cannot be re-aggregated** | Stored as a rate with no counts. |
| **71 contacts fall on a statutory holiday** | The generator uses business days but does not skip stat holidays. `dim_date[is_business_day]` makes them filterable. |
| **Handle time excludes hold** | They are drawn independently, so hold can exceed handle on 56 contacts. |
| **`true_sentiment` has no real-world equivalent** | It exists only to measure the model. Hide it. |
| **Three agents have under 6 months' tenure** | Their figures carry ~1.5× the odds of a negative contact for that reason alone. |
| **Long-tenure agents look worst and are not** | They sit entirely in Payments & Credit on the two hardest topics. Any league table ignoring caseload will libel them. |

---

*Generated against the live schema; every one of the 192 fields is documented.
Regenerate the data with `src/generate_all.py` → `src/run_pipeline.py` →
`src/build_model_tables.py`.*
