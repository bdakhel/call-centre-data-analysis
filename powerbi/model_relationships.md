# Power BI data model

Star schema produced by `src/build_model_tables.py` into `data/processed/`.
Two fact tables, five dimensions, all conformed.

All data is **synthetic** — see `CLAUDE.md`. No real customers, agents or company.

---

## Schema at a glance

```
                      ┌───────────────┐
                      │   dim_date    │
                      │  (2,557 rows) │
                      └───┬───────┬───┘
            date_key  1:* │       │ 1:*  date_key
         ┌────────────────┘       └──────────────┐
         │                                       │
┌────────▼─────────────┐              ┌──────────▼────────┐
│  fact_interactions   │              │ fact_sqm_monthly  │
│     (2,000 rows)     │              │     (24 rows)     │
└──┬────────┬────────┬─┘              └─────────┬─────────┘
   │        │        │                          │ team_key
   │        │        │                          │ 1:*
   │ 1:*    │ 1:*    │ 1:*                      │
┌──▼─────┐ ┌▼───────┐ ┌▼──────────┐   ┌─────────▼─────────┐
│dim_agent│ │dim_topic│ │dim_channel│   │     dim_team      │
│ (20)    │ │  (7)    │ │   (3)     │   │       (4)         │
└──┬──────┘ └─────────┘ └───────────┘   └─────────▲─────────┘
   │                       team_key  1:*          │
   └──────────────────────────────────────────────┘
```

---

## Relationships

Every relationship is **one-to-many** from the dimension to the fact, with
**single** cross-filter direction (dimension filters fact, never the reverse).
That is the default for a reason: bidirectional filters create ambiguous paths
and can silently change what a measure means.

| # | From (one) | To (many) | Cardinality | Cross-filter | Active | Join on |
|---|---|---|---|---|---|---|
| 1 | `dim_date[date_key]` | `fact_interactions[date_key]` | 1:* | Single → | **Yes** | contact date |
| 2 | `dim_agent[agent_key]` | `fact_interactions[agent_key]` | 1:* | Single → | **Yes** | handling agent |
| 3 | `dim_topic[topic_key]` | `fact_interactions[topic_key]` | 1:* | Single → | **Yes** | actual (routed) topic |
| 4 | `dim_channel[channel_key]` | `fact_interactions[channel_key]` | 1:* | Single → | **Yes** | contact channel |
| 5 | `dim_team[team_key]` | `dim_agent[team_key]` | 1:* | Single → | **Yes** | agent's team |
| 6 | `dim_date[date_key]` | `fact_sqm_monthly[date_key]` | 1:* | Single → | **Yes** | first day of month |
| 7 | `dim_team[team_key]` | `fact_sqm_monthly[team_key]` | 1:* | Single → | **Yes** | scorecard team |
| 8 | `dim_topic[topic_key]` | `fact_interactions[topic_predicted_key]` | 1:* | Single → | **No** | model-predicted topic |
| 9 | `dim_date[date_key]` | `fact_interactions[case_created_date_key]` | 1:* | Single → | **No** | case opened |
| 10 | `dim_date[date_key]` | `fact_interactions[case_closed_date_key]` | 1:* | Single → | **No** | case closed |
| 11 | `dim_date[date_key]` | `dim_agent[hire_date_key]` | 1:* | Single → | **No** | agent hire date |

### The inactive relationships (8–11)

Power BI allows only one active path between two tables. Relationships 8–11 are
**role-playing**: the same dimension used in a second role. Activate one inside a
measure with `USERELATIONSHIP`:

```dax
Cases Opened =
CALCULATE (
    COUNTROWS ( fact_interactions ),
    USERELATIONSHIP ( dim_date[date_key], fact_interactions[case_created_date_key] )
)

Topic Model Accuracy =
DIVIDE (
    CALCULATE ( COUNTROWS ( fact_interactions ), fact_interactions[topic_is_correct] = TRUE () ),
    COUNTROWS ( fact_interactions )
)
```

Relationship 8 matters for a specific question: *"what did the classifier think
this was?"* versus *"what was it actually?"*. Slicing by `dim_topic` normally
means the actual topic; activate 8 to slice by the prediction instead.

> **If you prefer explicit date roles**, duplicate `dim_date` as `dim_date_case_opened`
> / `dim_date_case_closed` with active relationships, rather than relying on
> `USERELATIONSHIP`. Costs two small tables, saves every measure author from
> remembering. Either is defensible; the single-table version is shipped here
> because the model is small.

### Why `dim_team` exists

It is not in the original table list, and the model is broken without it.

`fact_sqm_monthly` is at **month × team** grain — there is no agent on it. With
team as a plain text column on `dim_agent`, a team slicer would filter
interactions but leave the scorecard untouched, and the two visuals on the same
page would disagree with no warning. `dim_team` is the **conformed dimension**
that filters both: directly for `fact_sqm_monthly`, and through `dim_agent` for
`fact_interactions`.

That makes `dim_team → dim_agent → fact_interactions` a **snowflake**. It is a
deliberate trade: snowflaking costs one hop, and flattening team onto `dim_agent`
would cost cross-fact consistency, which is worse. `team_name` lives only on
`dim_team` so there is exactly one place to filter it from.

---

## Tables

### `dim_date` — 2,557 rows, one per calendar day

Spans **2020-01-01 to 2026-12-31**. Padded to whole calendar years because Power BI
time intelligence (`TOTALYTD`, `SAMEPERIODLASTYEAR`) returns blanks at the edges of
a partial year. The 2020 start covers the earliest agent `hire_date`, so
relationship 11 resolves — a key column that cannot join is worse than no key.

**Mark this table as a date table in Power BI** (`Table tools → Mark as date table`,
using `date`). Without it, time intelligence falls back to auto-generated hierarchies
and the fiscal columns are ignored.

| Column | Type | Notes |
|---|---|---|
| `date_key` | int64 | **PK.** `YYYYMMDD` |
| `date` | date | the actual date — use this for *Mark as date table* |
| `year`, `quarter`, `month_number`, `day_of_month`, `day_of_year`, `week_of_year` | int | calendar parts |
| `quarter_label`, `month_name`, `month_short`, `month_label` | text | display |
| `month_key` | int32 | `YYYYMM`, for month-grain joins |
| `month_name_sort`, `weekday_name_sort` | int | set as *Sort by column* so "April" sorts before "August" |
| `day_of_week` | int8 | **Monday = 1** (ISO) |
| `weekday_name`, `weekday_short` | text | |
| `is_weekend` | bool | Sat/Sun |
| `is_holiday`, `holiday_name` | bool, text | Alberta statutory holidays |
| `is_business_day` | bool | weekday **and** not a statutory holiday |
| `fiscal_year` | int16 | **FY starts 1 April**, named for the year it ends in — Apr 2026 is FY2027 |
| `fiscal_quarter`, `fiscal_month_number` | int8 | Apr = Q1, period 1 |
| `fiscal_quarter_label`, `fiscal_period_label` | text | `FY2027 Q1`, `FY2027 P01` |

Holidays: New Year's Day, Alberta Family Day, Good Friday, Victoria Day, Canada
Day, Labour Day, Thanksgiving, Remembrance Day, Christmas Day. Fixed-date holidays
falling on a weekend shift to the observed Monday. Boxing Day and Heritage Day are
optional in Alberta and are **not** included.

> ⚠️ **71 contacts fall on a statutory holiday.** `generate_interactions.py` draws
> business days but does not skip stat holidays. The data is the way it is;
> `is_business_day` now makes those contacts filterable rather than invisible.
> Filter them out for staffing analysis, or leave them in and note it.

### `dim_team` — 4 rows

| Column | Type | Notes |
|---|---|---|
| `team_key` | int16 | **PK** |
| `team_name` | text | the only place team name lives |
| `team_lead` | text | a team attribute, not an agent one |
| `agent_count` | int | CCRs on the team |
| `team_lead_email` | text | RLS key for the Team Lead role — see `rls_roles.md` |

### `dim_agent` — 20 rows

| Column | Type | Notes |
|---|---|---|
| `agent_key` | int16 | **PK** |
| `agent_id` | text | natural key (`A001`…) |
| `agent_name` | text | |
| `agent_email` | text | RLS key for the CCR role, matched to `USERPRINCIPALNAME()`. Unique — a collision would hand one agent another's data. Hide from report view. |
| `team_key` | int16 | **FK →** `dim_team` |
| `hire_date` | date | |
| `hire_date_key` | int64 | **FK →** `dim_date` (inactive, rel. 11) |
| `tenure_days`, `tenure_months` | int32, float | as at **2026-10-04** |
| `tenure_band` | text | Under 6 months / 6 months–2 years / 2–5 years / 5 years+ |
| `is_new_hire` | bool | tenure < 180 days |
| `csat_target` | float | 0.80–0.85 |
| `aht_target_sec` | int | 420–480; per-agent, so AHT variance must be computed per row |

### `dim_topic` — 7 rows

| Column | Type | Notes |
|---|---|---|
| `topic_key` | int16 | **PK** |
| `topic_name` | text | |
| `topic_group` | text | Billing / Metering / Lifecycle / Service / Credit / General |
| `is_financial_hardship` | bool | high bill dispute, payment arrangement |
| `is_complaint_prone` | bool | high bill dispute, outage or service issue |

### `dim_channel` — 3 rows

| Column | Type | Notes |
|---|---|---|
| `channel_key` | int16 | **PK** |
| `channel_name` | text | voice / chat / email |
| `channel_group` | text | Assisted (voice, chat) / Deferred (email) |
| `is_realtime` | bool | |
| `channel_sort` | int8 | *Sort by column* for `channel_name` |

### `fact_interactions` — 2,000 rows, one per contact

The main fact. Left-joined to cases (91% match) and surveys (15% match), so the
`has_*` flags matter — see the gotchas below.

**Keys**

| Column | Type | Notes |
|---|---|---|
| `interaction_key` | int32 | **PK**, surrogate |
| `interaction_id` | text | degenerate dimension (`I00001`…) |
| `date_key`, `agent_key`, `topic_key`, `channel_key` | int | **FKs**, active |
| `topic_predicted_key`, `case_created_date_key`, `case_closed_date_key` | int | **FKs**, inactive |

**Measures (additive — safe to SUM)**

`handle_time_sec`, `hold_time_sec` (int32), `days_to_close` (Int32, null when open).

**Model scores**

`sentiment_label`, `sentiment_score`, `sentiment_start`, `sentiment_end`,
`sentiment_shift` (= end − start), `topic_predicted`, `coaching_note`,
`generated_by`, `model_version`, `scored_at`.

`true_sentiment` is the generator's hidden label — **synthetic ground truth that
would not exist in production.** It is here so accuracy visuals are possible.
Do not build an operational metric on it.

**Flags (bool — SUM them to count)**

`had_hold`, `has_score`, `has_case`, `has_survey_response`, `is_repeat_contact_7d`,
`is_escalated`, `is_case_open`, `is_resolved`, `is_csat_promoter` (CSAT ≥ 4),
`is_csat_detractor` (CSAT ≤ 2), `sentiment_is_correct`, `topic_is_correct`.

**Other attributes:** `start_datetime`, `start_hour`, `queue`, `case_id`,
`customer_id`, `case_status`, `survey_response_id`, `csat_score` (Int8, null when
no response), `survey_comment`.

### `fact_sqm_monthly` — 24 rows, one per month × team

| Column | Type | Notes |
|---|---|---|
| `sqm_key` | int16 | **PK** |
| `date_key` | int64 | **FK →** `dim_date`, first day of the month |
| `month_key` | int32 | `YYYYMM` |
| `team_key` | int16 | **FK →** `dim_team` |
| `fcr_pct` | float | first contact resolution, % |
| `industry_benchmark_fcr_pct` | float | ~74% |
| `fcr_vs_benchmark_pp` | float | difference in percentage points |
| `is_above_benchmark` | bool | |

---

## Gotchas

**1. `fcr_pct` is a rate with no counts behind it — never SUM it, and average it
with care.** The source hands us a percentage, not a numerator and denominator, so
a weighted roll-up is impossible: `AVERAGE(fcr_pct)` across teams treats a 500-contact
team the same as a 50-contact one. Use it at month × team grain, and if a total is
needed, state that it is an unweighted mean. *A production model would store
`resolved_contacts` and `total_contacts` and compute the ratio in DAX.* This is the
one place the shipped model stores a pre-computed rate, and only because the source
gives nothing else.

**2. No percentages are stored anywhere else, deliberately.** AHT-vs-target, accuracy
and CSAT are all stored as their components. The mean of per-row percentages is not
the overall percentage, and storing one invites exactly that error:

```dax
-- right
AHT vs Target % =
DIVIDE ( SUM ( fact_interactions[handle_time_sec] ),
         SUMX ( fact_interactions, RELATED ( dim_agent[aht_target_sec] ) ) ) - 1
```

**3. Filter on the `has_*` flags before averaging the joined columns.** 1,823 of 2,000
contacts have a case and only 300 have a survey. `AVERAGE(csat_score)` ignores blanks
and is therefore correct for *respondents* — but respondents are not customers:
unhappy ones answer more often, so survey CSAT reads materially below true sentiment.
That gap is a finding, not a bug (see `docs/validation.md`).

**4. `is_repeat_contact_7d` is forward-looking.** True means *this customer contacted
again within 7 days*, the mirror of first contact resolution — not "this contact was
itself a repeat". It reconciles exactly against the case log.

**5. Sentiment labels are miscalibrated toward negative.** Validation against 100
human labels put RoBERTa at 61% agreement with **every one of its 39 errors** judging
customers gloomier than the human did. `sentiment_end` is the better signal (77%
held-out vs 56%). Prefer `sentiment_end` over `sentiment_label` for anything
customer-facing, and read `docs/validation.md` before publishing a satisfaction metric.

---

## Loading into Power BI

1. **Get Data → Text/CSV** for each of the seven files in `data/processed/`.
2. Check types on import — `date_key` must land as **Whole Number**, not text, or the
   relationships will not form.
3. Create relationships 1–7 as **active**, 8–11 as **inactive** (uncheck *Make this
   relationship active*). All single-direction.
4. `dim_date` → **Mark as date table**, using `date`.
5. Set *Sort by column*: `month_name` → `month_name_sort`, `weekday_name` →
   `weekday_name_sort`, `channel_name` → `channel_sort`.
6. Hide every `*_key` column from report view, plus `dim_agent[agent_email]` and
   `dim_team[team_lead_email]`. Hide `true_sentiment` unless the report is
   specifically about model accuracy.

See `powerbi/BUILD_GUIDE.md` for the full browser-based walkthrough (Power BI
Desktop is Windows-only) and `powerbi/rls_roles.md` for the security roles.

Rebuild the tables any time upstream data or scores change:

```bash
python src/generate_all.py      # synthetic source data
python src/run_pipeline.py      # model scores
python src/build_model_tables.py
```

`build_model_tables.py` fails loudly on any orphan foreign key, so a broken model
cannot reach the `.pbix`.
