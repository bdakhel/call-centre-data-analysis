# Building the report in Power BI (browser)

Step-by-step for **Power BI in the browser** at `app.powerbi.com`.

**Why the browser:** Power BI Desktop is Windows-only — there is no Mac build and
there will not be one. Everything here is done in Safari or Chrome on macOS. The
browser can now do the three things that used to force people onto Desktop:
relationships, DAX measures, and row-level security roles.

**What you need**
- A Power BI account with a **Pro** licence (or a Fabric capacity workspace).
- A workspace you can create items in — *not* **My workspace**, which cannot host
  RLS role assignments properly.
- Roughly 90 minutes for the first build.

**What you cannot do in the browser** — worth knowing up front so you do not hunt
for them: custom visuals from the marketplace are limited, bookmarks and
selections are more awkward than on Desktop, and there is no Performance Analyzer.
None of it blocks this report.

All data is **synthetic** (see `CLAUDE.md`).

---

## Step 1 — Produce the data (5 min, on your Mac)

```bash
cd "/Users/bassamd/Downloads/Sentiment analysis/utility-cx-insights"

.venv/bin/python src/generate_all.py            # synthetic source data
.venv/bin/python src/run_pipeline.py            # sentiment, topics, coaching
.venv/bin/python src/build_model_tables.py      # the seven star-schema tables
.venv/bin/python src/export_powerbi_workbook.py # one workbook to upload
```

The last step writes **`data/processed/utility_cx_model.xlsx`** (~0.7 MB) — seven
named Excel tables, one per model table. Named tables matter: Power BI imports
them under those exact names instead of guessing at sheet ranges.

> **Why a workbook and not the seven CSVs?** Uploading CSVs one at a time gives
> you seven *separate* semantic models that cannot be related to each other. One
> workbook gives one model with seven tables, which is what a star schema needs.
>
> `powerbi/load_queries.m` is still the reference for correct column typing, and
> is what you would use in a Dataflow if you later want scheduled refresh from
> OneDrive. For a portfolio build, the workbook is the shortest honest path.

---

## Step 2 — Upload and create the semantic model (5 min)

1. Go to **app.powerbi.com** → your workspace.
2. **+ New item → Semantic model** (older tenants: **+ New → Upload a file**).
3. Choose **Excel**, browse to `utility_cx_model.xlsx`, upload.
4. Power BI creates a semantic model with all seven tables. Rename it something
   like `Utility CX Insights`.

**Check before moving on.** Open the model and confirm all seven tables arrived
with sensible row counts:

| Table | Rows |
|---|---|
| `dim_date` | 2,557 |
| `dim_team` | 4 |
| `dim_agent` | 20 |
| `dim_topic` | 7 |
| `dim_channel` | 3 |
| `fact_interactions` | 2,000 |
| `fact_sqm_monthly` | 24 |

If a table is missing, the Excel table name did not survive the upload — re-run
the export and check the sheet tabs are named exactly as above.

### Fix the data types

Excel upload guesses types, and it guesses `date_key` wrong often enough to be
worth checking every time. In **Open data model**, for each table click a column
and check the **Data type** in the Properties pane:

- All `*_key` columns → **Whole number**. *If any lands as Text, relationships
  silently refuse to form later and the error message will not tell you why.*
- `date`, `hire_date` → **Date**
- `start_datetime` → **Date/time**
- `scored_at` → **Date/time** (the offset is dropped; harmless here)
- `sentiment_*`, `fcr_pct`, `csat_target`, `tenure_months` → **Decimal number**
- Every `is_*` / `has_*` / `had_hold` → **True/False**

---

## Step 3 — Relationships (10 min)

Workspace → the semantic model → **⋯ → Open data model**. You get a diagram view.
Drag from the dimension column to the fact column to create each one.

Create these **seven active** relationships. All are **one-to-many**, dimension →
fact, with **single** cross-filter direction. Leave *Assume referential integrity*
unchecked.

| From (one) | To (many) |
|---|---|
| `dim_date[date_key]` | `fact_interactions[date_key]` |
| `dim_agent[agent_key]` | `fact_interactions[agent_key]` |
| `dim_topic[topic_key]` | `fact_interactions[topic_key]` |
| `dim_channel[channel_key]` | `fact_interactions[channel_key]` |
| `dim_team[team_key]` | `dim_agent[team_key]` |
| `dim_date[date_key]` | `fact_sqm_monthly[date_key]` |
| `dim_team[team_key]` | `fact_sqm_monthly[team_key]` |

Then **four inactive** ones. Create each the same way, then open it and
**uncheck "Make this relationship active"**:

| From | To | Used for |
|---|---|---|
| `dim_topic[topic_key]` | `fact_interactions[topic_predicted_key]` | what the model predicted |
| `dim_date[date_key]` | `fact_interactions[case_created_date_key]` | case opened |
| `dim_date[date_key]` | `fact_interactions[case_closed_date_key]` | case closed |
| `dim_date[date_key]` | `dim_agent[hire_date_key]` | hire date |

Power BI will offer to make some of these active automatically. Say no — two
active paths between the same tables is ambiguous and it will refuse anyway.

### Mark the date table

Select `dim_date` → **⋯ → Mark as date table** → choose `dim_date[date]`.
Skip this and every time-intelligence measure returns blank.

### Sort-by columns

Select the column, then Properties → **Sort by column**:

| Column | Sort by |
|---|---|
| `dim_date[month_name]` | `month_name_sort` |
| `dim_date[weekday_name]` | `weekday_name_sort` |
| `dim_channel[channel_name]` | `channel_sort` |

Without these, months sort alphabetically and April leads the year.

### Hide the plumbing

Right-click → **Hide** on every `*_key` column in all tables, plus
`fact_interactions[true_sentiment]` (synthetic ground truth — it should not be
browsable alongside the model's own output) and `dim_agent[agent_email]`.

---

## Step 4 — Measures (20 min)

In **Open data model**, select `fact_interactions` → **New measure** → paste one
definition → tick the checkmark. Repeat.

Open `powerbi/measures.dax` and work top to bottom — **order matters**, because
later measures reference earlier ones. Start with everything under *99 Base*; a
measure that references an undefined one shows a red squiggle that clears as soon
as its dependency exists.

Set each measure's **Format** and **Display folder** in the Properties pane as you
go (the folder for each section is in the file's banner comments). Doing it later
means revisiting 60+ measures.

Minimum set to get the report standing — add the rest after:

```
Interactions · Scored Interactions · Cases · Survey Responses
Handle Seconds · Target Handle Seconds
AHT (minutes) · AHT vs Target % · CSAT % · Survey Response Rate %
FCR % (SQM) · FCR vs Benchmark (pp) · Net Sentiment · % Negative
Repeat Contact Rate % · Sentiment Shift
Data As Of · Synthetic Data Label
Coaching Priority · Top 10 Coaching Flag · Coaching Rank
```

Format strings are in each measure's `///` comment. The important ones:
percentages `0.0%`, percentage-point changes `+0.0;-0.0`, AHT `0.0`,
Net Sentiment `+0.0;-0.0`.

---

## Step 5 — Theme (2 min)

1. Workspace → **+ New item → Report** → pick your semantic model.
2. In the report: **View → Theme → Browse for themes**.
3. Upload `powerbi/theme.json`.

The palette is validated for colour-vision deficiency and WCAG contrast. Two
consequences you should not override:

- **Data colours 3, 4 and 5** (aqua, yellow, pink) sit below 3:1 against white.
  They are fine as chart fills *with data labels showing* — the theme turns labels
  on by default for bar and column charts. Do not turn them off on those visuals.
- **Never recolour a series by hand.** The slot order is what keeps adjacent
  colours distinguishable for colourblind readers; reordering breaks it. I tested
  a "nicer" order while building this and it failed the check outright
  (green↔orange collapsed to ΔE 3.2 for protanopia, against a floor of 8).

---

## Step 6 — Row-level security (10 min)

Follow **`powerbi/rls_roles.md`**. Short version:

**Open data model → Manage roles → New**

| Role | Table | Filter (use the **DAX editor**, not the dropdown) |
|---|---|---|
| `CCR` | `dim_agent` | `[agent_email] = USERPRINCIPALNAME()` |
| `Team Lead` | `dim_team` | `[team_lead_email] = USERPRINCIPALNAME()` |

The dropdown editor cannot express `USERPRINCIPALNAME()` — you must switch to the
DAX editor.

Test with **Manage roles → Test as role**. Note that **RLS does not apply to you**
as workspace Admin/Member/Contributor — it only constrains Viewers, so opening the
report yourself proves nothing.

---

## Step 7 — The pages

Five pages. Canvas is **16:9, 1280 × 720** throughout (View → Page view → Actual
size; Format page → Canvas settings → 1280 × 720).

### Furniture on every page

Build it once on the Executive Scorecard, then copy-paste onto each page.

| Element | Position (x, y, w, h) | Content |
|---|---|---|
| Header bar | 0, 0, 1280, 56 | Shape → Rectangle, fill `#f5f4f1`, no border |
| Page title | 16, 12, 460, 32 | Textbox, 18pt Segoe UI Semibold, `#0b0b0b` |
| **Synthetic data label** | 486, 16, 530, 24 | Card → `[Synthetic Data Label]`, 10pt, `#d03b3b`, no category label |
| **Data as of** | 1030, 16, 234, 24 | Card → `[Data As Of]`, 10pt, `#52514e`, right-aligned, no category label |

Both cards use measures rather than typed text so they cannot go stale — the
"as of" date reads the latest contact in the model, not today's clock, so a failed
refresh is visible instead of hidden.

### Shared slicers

Put these at the top of pages 1–3, directly under the header (y = 64, h = 40):

| Slicer | Field | Style | x, w |
|---|---|---|---|
| Date | `dim_date[month_label]` | Dropdown | 16, 200 |
| Team | `dim_team[team_name]` | Dropdown | 228, 200 |
| Channel | `dim_channel[channel_name]` | Tile, horizontal | 440, 280 |
| Topic | `dim_topic[topic_name]` | Dropdown | 732, 220 |

Then **View → Sync slicers** and sync Date, Team and Channel across pages 1–3
(sync *and* visible on all three).

---

### Page 1 — Executive Scorecard

*The "how are we doing" page. Readable in ten seconds from the doorway.*

**KPI row** — six cards, y = 116, h = 110, w = 200, gap 12, starting x = 16:

| # | Measure | Title | Format |
|---|---|---|---|
| 1 | `[Interactions]` | Contacts | `#,0` |
| 2 | `[AHT (minutes)]` | AHT (min) | `0.0` |
| 3 | `[CSAT %]` | CSAT | `0.0%` |
| 4 | `[FCR % (SQM)]` | FCR | `0.0` |
| 5 | `[Net Sentiment]` | Net sentiment | `+0.0;-0.0` |
| 6 | `[Repeat Contact Rate %]` | Repeat 7d | `0.0%` |

Under each, a small textbox (h = 20) with its MoM measure: `[Contacts MoM %]`,
`[AHT MoM (minutes)]`, `[CSAT MoM (pp)]`, `[FCR MoM (pp)]`,
`[Net Sentiment MoM]`, `[Repeat Contact Rate MoM (pp)]`.

> Set conditional formatting on the MoM values — but check the direction. For AHT,
> repeat contact and % negative, **down is good**. Getting this backwards is the
> most common error on a scorecard like this.

**Row 2 — trend**, y = 240, h = 230:

- *Contact volume and sentiment* — **Line and clustered column chart**, x = 16, w = 620.
  X: `dim_date[month_label]`. Column: `[Interactions]`. Line: `[Net Sentiment]`.
  Title: "Volume and sentiment by month".
- *FCR vs benchmark* — **Line chart**, x = 648, w = 616.
  X: `dim_date[month_label]`. Y: `[FCR % (SQM)]` and `[FCR Benchmark %]`.
  Legend on. Title: "First contact resolution vs industry benchmark".

**Row 3 — breakdown**, y = 484, h = 220:

- *Contacts by topic* — **Clustered bar chart**, x = 16, w = 400.
  Y: `dim_topic[topic_name]`. X: `[Interactions]`. Sort descending. Data labels on.
- *Team scorecard* — **Table**, x = 428, w = 836.
  Rows: `dim_team[team_name]`. Columns: `[Interactions]`, `[AHT (minutes)]`,
  `[CSAT %]`, `[FCR % (SQM)]`, `[FCR vs Benchmark (pp)]`, `[Net Sentiment]`,
  `[Repeat Contact Rate %]`.
  Conditional formatting: background scale on `[FCR vs Benchmark (pp)]`, red
  `#f6dcdc` → green `#dcf0dc`.

---

### Page 2 — Sentiment & Topics

*Where the unhappiness is concentrated.*

**KPI row**, y = 116, h = 100, w = 240:

`[% Negative]` · `[% Positive]` · `[Net Sentiment]` · `[Sentiment Shift]` ·
`[Recovered Contacts]`

**Row 2**, y = 228, h = 250:

- *Sentiment by topic* — **100% stacked bar chart**, x = 16, w = 620.
  Y: `dim_topic[topic_name]`. X: `[Interactions]`.
  Legend: `fact_interactions[sentiment_label]`. Sort by `[% Negative]` desc.
  Data labels on. Title: "Sentiment mix by topic".
- *Negative share over time* — **Line chart**, x = 648, w = 616.
  X: `dim_date[month_label]`. Y: `[% Negative]`. Legend: `dim_team[team_name]`.
  Title: "% negative by team — watch Move In / Move Out from August".

**Row 3**, y = 492, h = 212:

- *Sentiment arc* — **Scatter chart**, x = 16, w = 500.
  X: `fact_interactions[sentiment_start]`, Y: `fact_interactions[sentiment_end]`,
  Details: `fact_interactions[interaction_id]`, Legend: `dim_topic[topic_name]`.
  Add an X and Y constant line at 0. Points above the diagonal recovered.
  *Cap it:* visual-level Top N filter, 400 by `[Coaching Priority]` — 2,000 points
  renders slowly in the browser and reads as a smear.
- *Channel mix* — **Clustered column chart**, x = 528, w = 340.
  X: `dim_channel[channel_name]`, Y: `[% Negative]`. Data labels on.
- *Model accuracy* — **Card** + **Table**, x = 880, w = 384.
  Card: a measure for sentiment accuracy. Table: `fact_interactions[sentiment_label]`
  rows × `fact_interactions[true_sentiment]` columns, values `[Interactions]`.
  Title: "Model vs generator label — synthetic ground truth only".

> This last visual is the one place `true_sentiment` is legitimately used. Keep it
> on this page and keep the title explicit — it would not exist in production.

---

### Page 3 — Team & Coaching

*The team lead's page. Includes the coaching queue.*

**Row 1 — agent scorecard**, y = 116, h = 280:

**Table**, x = 16, w = 1248. Rows: `dim_agent[agent_name]`. Columns:

`[Interactions]` · `[AHT (minutes)]` · `[AHT vs Target %]` · `[CSAT %]` ·
`[CSAT vs Target (pp)]` · `[Net Sentiment]` · `[% Negative]` ·
`[Repeat Contact Rate %]` · `[Sentiment Shift]`

Add `dim_agent[tenure_band]` as a column so a new hire's numbers are read in
context — three of the twenty have under six months' tenure and their figures look
worse for that reason alone.

Conditional formatting: data bars on `[Interactions]`; background scale on
`[AHT vs Target %]` and `[CSAT vs Target (pp)]`.

**Row 2 — the top-10 coaching queue**, y = 408, h = 296:

**Table**, x = 16, w = 1248. Columns in this order:

| Field | Note |
|---|---|
| `[Coaching Rank]` | sort ascending by this |
| `dim_agent[agent_name]` | |
| `fact_interactions[interaction_id]` | the grain — required |
| `dim_topic[topic_name]` | |
| `[Coaching Priority]` | format `0` |
| `fact_interactions[sentiment_end]` | format `+0.00;-0.00` |
| `fact_interactions[sentiment_shift]` | format `+0.00;-0.00` |
| `[Hold (minutes)]` | format `0.0` |
| `[AHT vs Target %]` | format `+0.0%;-0.0%` |
| `fact_interactions[coaching_note]` | word wrap on, widest column |

**Then apply the filter that makes it a top 10:** Filters pane → *Filters on this
visual* → drag in **`[Top 10 Coaching Flag]`** → *Show items when the value* **is 1**.

The flag uses `ALLSELECTED`, so the queue re-ranks inside whatever the slicers are
set to. A lead who picks their own team gets *their* top ten, not ten rows that all
belong to someone else. That is the whole point of the measure — a plain visual-level
"Top N" filter would rank against the unfiltered model.

> **Alternative without the measure:** visual-level filter on
> `fact_interactions[interaction_id]` → *Top N* → Top 10 → By value
> `[Coaching Priority]`. Simpler, and it respects slicers too. The measure version
> is here because it also gives you `[Coaching Rank]` as a sortable column and can
> be reused on other visuals.

Set **Drill through** on this table → to the Interaction Detail page.

---

### Page 4 — CCR Self-View

*What an individual agent sees. Build it as if you were them.*

With the `CCR` role applied, `dim_agent` filters to one row, so every visual here
is automatically personal — no page-level filter needed, and none should be added
(a page filter would be a second, weaker lock that masks an RLS mistake).

**Row 1 — personal KPIs**, y = 116, h = 120, w = 244:

`[Interactions]` · `[AHT (minutes)]` vs `[AHT Target (minutes)]` ·
`[CSAT %]` vs `[CSAT Target %]` · `[Net Sentiment]` · `[Sentiment Shift]`

Use **KPI visuals** for the two with targets: Indicator = the measure,
Target goals = the target measure, Trend axis = `dim_date[month_label]`.

**Row 2**, y = 248, h = 220:

- *My trend* — **Line chart**, x = 16, w = 620.
  X: `dim_date[month_label]`. Y: `[CSAT %]` and `[Net Sentiment]`.
- *My topic mix* — **Donut chart**, x = 648, w = 300.
  Legend: `dim_topic[topic_name]`, Values: `[Interactions]`.
- *How I compare* — **Card**, x = 964, w = 300. `[AHT vs Target %]`, large.
  Subtitle textbox: "vs your own AHT target".

**Row 3 — my coaching notes**, y = 480, h = 224:

**Table**, x = 16, w = 1248. Columns: `dim_date[date]`,
`fact_interactions[interaction_id]`, `dim_topic[topic_name]`,
`fact_interactions[sentiment_label]`, `fact_interactions[coaching_note]` (wrapped).
Sort by date descending. Filter: `[Coaching Priority]` is not blank.

> **Tone matters on this page.** These notes are written to be developmental, and
> the page should read that way — lead with the trend and the wins, not a ranked
> list of failures. There is deliberately no league table here. Agents comparing
> themselves to colleagues is the fastest way to make a coaching tool resented.

---

### Page 5 — Interaction Detail (drill-through)

*One contact, in full. Reached by right-clicking a row elsewhere.*

**Set it up as a drill-through page:**

1. Format page → **Page information → Allow drill through from: Used as category**.
2. Drag `fact_interactions[interaction_id]` into **Drill through** in the Filters pane.
3. Keep the auto-generated **back button** (top-left, 16, 64).
4. Format page → **Page information → Hide page** once tested — a drill-through
   page is meaningless without its filter context.

**Layout:**

| Element | Position | Content |
|---|---|---|
| Context strip | 16, 112, 1248, 90 | **Multi-row card**: `dim_agent[agent_name]`, `dim_team[team_name]`, `dim_topic[topic_name]`, `dim_channel[channel_name]`, `dim_date[date]`, `fact_interactions[queue]` |
| Metric strip | 16, 214, 1248, 90 | **Multi-row card**: `[AHT (minutes)]`, `[Hold (minutes)]`, `[AHT vs Target %]`, `fact_interactions[sentiment_label]`, `fact_interactions[sentiment_score]`, `fact_interactions[sentiment_shift]` |
| Sentiment arc | 16, 316, 400, 180 | **Clustered column**: `fact_interactions[sentiment_start]` and `fact_interactions[sentiment_end]`, data labels on |
| Case & survey | 428, 316, 400, 180 | **Table**: `case_id`, `case_status`, `days_to_close`, `is_repeat_contact_7d`, `csat_score`, `is_resolved` |
| Coaching note | 840, 316, 424, 180 | **Card (text)** or 1-row table: `fact_interactions[coaching_note]`, wrapped |
| Survey comment | 16, 508, 608, 196 | **Table**: `fact_interactions[survey_comment]`, wrapped |
| Provenance | 636, 508, 628, 196 | **Table**: `generated_by`, `model_version`, `scored_at` |

> The **transcript itself is deliberately not on this page.** It is not in
> `fact_interactions` — the model carries the scores and the note, not the
> conversation. If you add it, treat it as the most sensitive field in the report
> and check the RLS behaviour on this page specifically.

---

## Step 8 — Before you share

- [ ] Every page has the synthetic-data label and the "data as of" card.
- [ ] MoM conditional formatting points the right way — **down is good** for AHT,
      repeat contact and % negative.
- [ ] Months sort chronologically, not alphabetically (sort-by columns, Step 3).
- [ ] All `*_key` columns hidden; `true_sentiment` hidden except on the accuracy visual.
- [ ] Drill-through works from both the coaching queue and the agent scorecard.
- [ ] Tested as a **Viewer** in both roles — not just *Test as role*, and not as
      yourself as workspace admin.
- [ ] A CCR cannot see another agent anywhere, including the drill-through page.
- [ ] Decided whether team FCR is sensitive to a CCR (see `rls_roles.md`).

### Refreshing later

```bash
.venv/bin/python src/generate_all.py
.venv/bin/python src/run_pipeline.py
.venv/bin/python src/build_model_tables.py
.venv/bin/python src/export_powerbi_workbook.py
```

Then upload the workbook again, replacing the file. Measures, relationships,
theme and RLS roles all survive — they live on the semantic model, not the file.

For scheduled refresh instead of manual upload, put the CSVs in OneDrive for
Business and build a **Dataflow** using the M in `powerbi/load_queries.m`; its
Power Query Online editor takes that code directly.

---

## If something does not work

**Relationships will not form.** A `*_key` column imported as Text. Fix the type
in the data model (Step 2), then retry.

**Time intelligence returns blank.** `dim_date` was not marked as a date table, or
it was marked on `date_key` rather than `date`.

**A measure shows a red squiggle.** It references a measure you have not created
yet. Work through `measures.dax` top to bottom.

**Months read April, August, December…** Sort-by columns not set (Step 3).

**FCR MoM is always blank.** `fact_sqm_monthly` only holds the first of each month.
That measure uses `PREVIOUSMONTH` for exactly this reason — check you pasted it and
not a `DATEADD` version.

**RLS does nothing when you test it.** You are an Admin/Member/Contributor on the
workspace, and RLS only constrains Viewers.

**The coaching queue shows more than 10 rows.** The visual-level filter on
`[Top 10 Coaching Flag]` is missing or not set to *is 1*. Ties at the boundary can
also return 11 — `TOPN` keeps all tied rows.
