// =====================================================================
//  utility-cx-insights — Power Query (M) load layer
//  Loads the seven star-schema CSVs from data/processed/.
//  Model + relationships: powerbi/model_relationships.md
//  All data is SYNTHETIC (see CLAUDE.md).
// =====================================================================
//
//  HOW TO USE
//  Each block below is a SEPARATE query. In Power BI Desktop:
//    Home → Transform data → New Source → Blank Query → Advanced Editor,
//    paste one block, and name the query exactly as its header says.
//
//  Create them in this order — the later queries depend on the first two:
//    1.  DataFolder        (parameter)
//    2.  fnLoadTable       (shared function)
//    3.  the seven tables
//
//  WHY A FUNCTION RATHER THAN SEVEN COPIES
//  The CSV options below are not boilerplate — each one is load-bearing:
//    * Encoding = 65001 (UTF-8). The transcripts and coaching notes contain
//      em-dashes and curly quotes. Without this they arrive as mojibake.
//    * QuoteStyle.Csv. Coaching notes and survey comments contain commas and
//      embedded newlines; this keeps a quoted field as one value instead of
//      silently shearing rows.
//    * Culture "en-US" on the type step. Dates are ISO (yyyy-MM-dd). On a
//      machine with a dd/MM locale, an unqualified conversion reads
//      2026-04-06 inconsistently or errors. Pinning the culture removes the
//      dependency on whoever opens the file.
//  Getting any of these wrong fails quietly, which is the worst way to fail.
//
// =====================================================================


// ─────────────────────────────────────────────────────────────────────
//  QUERY 1 of 9  ·  name it:  DataFolder        (Parameter)
// ─────────────────────────────────────────────────────────────────────
//  Create via Home → Manage Parameters → New Parameter:
//      Name:      DataFolder
//      Type:      Text
//      Required:  yes
//      Current:   the absolute path to data/processed
//
//  Keeping the path in a parameter means the report moves between machines
//  (and between dev and a gateway) by editing one value, not seven queries.
//
//  If you prefer to define it as a query instead of through the dialog:

"C:\path\to\utility-cx-insights\data\processed"
    meta [
        IsParameterQuery = true,
        Type = type text,
        IsParameterQueryRequired = true
    ]


// ─────────────────────────────────────────────────────────────────────
//  QUERY 2 of 9  ·  name it:  fnLoadTable       (Function)
// ─────────────────────────────────────────────────────────────────────

let
    Load = (FileName as text, ColumnTypes as list) as table =>
        let
            // Accept the folder with or without a trailing separator.
            Separator =
                if Text.EndsWith(DataFolder, "\") or Text.EndsWith(DataFolder, "/")
                then "" else "\",
            FullPath = DataFolder & Separator & FileName,

            Source = Csv.Document(
                File.Contents(FullPath),
                [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
            ),
            Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),

            // Fail loudly and by name if the file's shape has drifted from
            // what the model expects, rather than loading a half-typed table.
            Expected = List.Transform(ColumnTypes, each _{0}),
            Missing  = List.Difference(Expected, Table.ColumnNames(Promoted)),
            Checked  =
                if List.IsEmpty(Missing) then Promoted
                else error Error.Record(
                    "Column missing from " & FileName,
                    Text.Combine(Missing, ", "),
                    "Re-run: python src/build_model_tables.py"
                ),

            Typed = Table.TransformColumnTypes(Checked, ColumnTypes, "en-US")
        in
            Typed
in
    Load


// ─────────────────────────────────────────────────────────────────────
//  QUERY 3 of 9  ·  name it:  dim_date
// ─────────────────────────────────────────────────────────────────────
//  After loading: Table tools → Mark as date table → dim_date[date].
//  Then set Sort by column:  month_name → month_name_sort,
//                            weekday_name → weekday_name_sort.

fnLoadTable(
    "dim_date.csv",
    {
        {"date_key", Int64.Type},              // PK, YYYYMMDD
        {"date", type date},                   // mark-as-date-table column
        {"year", Int64.Type},
        {"quarter", Int64.Type},
        {"quarter_label", type text},
        {"month_number", Int64.Type},
        {"month_name", type text},
        {"month_short", type text},
        {"month_key", Int64.Type},             // YYYYMM
        {"month_label", type text},
        {"month_name_sort", Int64.Type},       // Sort by column for month_name
        {"day_of_month", Int64.Type},
        {"day_of_year", Int64.Type},
        {"week_of_year", Int64.Type},
        {"day_of_week", Int64.Type},           // Monday = 1
        {"weekday_name", type text},
        {"weekday_short", type text},
        {"weekday_name_sort", Int64.Type},     // Sort by column for weekday_name
        {"is_weekend", type logical},
        {"is_holiday", type logical},
        {"holiday_name", type text},
        {"is_business_day", type logical},     // weekday AND not a stat holiday
        {"fiscal_year", Int64.Type},           // FY starts 1 April
        {"fiscal_quarter", Int64.Type},
        {"fiscal_quarter_label", type text},
        {"fiscal_month_number", Int64.Type},
        {"fiscal_period_label", type text}
    }
)


// ─────────────────────────────────────────────────────────────────────
//  QUERY 4 of 9  ·  name it:  dim_team
// ─────────────────────────────────────────────────────────────────────

fnLoadTable(
    "dim_team.csv",
    {
        {"team_key", Int64.Type},              // PK
        {"team_name", type text},              // the ONLY place team name lives
        {"team_lead", type text},
        {"agent_count", Int64.Type},
        {"team_lead_email", type text}         // RLS: matched to USERPRINCIPALNAME()
    }
)


// ─────────────────────────────────────────────────────────────────────
//  QUERY 5 of 9  ·  name it:  dim_agent
// ─────────────────────────────────────────────────────────────────────

fnLoadTable(
    "dim_agent.csv",
    {
        {"agent_key", Int64.Type},             // PK
        {"agent_id", type text},               // natural key, A001…
        {"agent_name", type text},
        {"agent_email", type text},            // RLS: matched to USERPRINCIPALNAME()
        {"team_key", Int64.Type},              // FK → dim_team
        {"hire_date", type date},
        {"hire_date_key", Int64.Type},         // FK → dim_date (inactive)
        {"tenure_days", Int64.Type},
        {"tenure_months", type number},
        {"tenure_band", type text},
        {"is_new_hire", type logical},
        {"csat_target", type number},          // 0.80–0.85, format as %
        {"aht_target_sec", Int64.Type}         // per-agent; see AHT vs Target %
    }
)


// ─────────────────────────────────────────────────────────────────────
//  QUERY 6 of 9  ·  name it:  dim_topic
// ─────────────────────────────────────────────────────────────────────

fnLoadTable(
    "dim_topic.csv",
    {
        {"topic_key", Int64.Type},             // PK
        {"topic_name", type text},
        {"topic_group", type text},            // Billing / Metering / …
        {"is_financial_hardship", type logical},
        {"is_complaint_prone", type logical}
    }
)


// ─────────────────────────────────────────────────────────────────────
//  QUERY 7 of 9  ·  name it:  dim_channel
// ─────────────────────────────────────────────────────────────────────
//  After loading: set Sort by column  channel_name → channel_sort.

fnLoadTable(
    "dim_channel.csv",
    {
        {"channel_key", Int64.Type},           // PK
        {"channel_name", type text},
        {"channel_group", type text},          // Assisted / Deferred
        {"is_realtime", type logical},
        {"channel_sort", Int64.Type}           // Sort by column for channel_name
    }
)


// ─────────────────────────────────────────────────────────────────────
//  QUERY 8 of 9  ·  name it:  fact_interactions
// ─────────────────────────────────────────────────────────────────────
//  The grain is one row per contact. Columns that come from the LEFT JOINs
//  to cases and surveys are null where there is no match — that is expected,
//  not dirty data. Filter on the has_* flags before averaging them.

fnLoadTable(
    "fact_interactions.csv",
    {
        // ── keys ──
        {"interaction_key", Int64.Type},       // PK
        {"interaction_id", type text},         // degenerate dimension
        {"date_key", Int64.Type},              // FK → dim_date       (active)
        {"agent_key", Int64.Type},             // FK → dim_agent      (active)
        {"topic_key", Int64.Type},             // FK → dim_topic      (active)
        {"topic_predicted_key", Int64.Type},   // FK → dim_topic      (inactive)
        {"channel_key", Int64.Type},           // FK → dim_channel    (active)

        // ── when and where ──
        {"start_datetime", type datetime},
        {"start_hour", Int64.Type},
        {"queue", type text},

        // ── additive measures ──
        {"handle_time_sec", Int64.Type},
        {"hold_time_sec", Int64.Type},
        {"had_hold", type logical},

        // ── model sentiment ──
        {"sentiment_label", type text},
        {"sentiment_score", type number},      // −1 … +1
        {"sentiment_start", type number},      // first third of customer turns
        {"sentiment_end", type number},        // last third — better calibrated
        {"sentiment_shift", type number},      // end − start
        {"true_sentiment", type text},         // SYNTHETIC ground truth; hide
        {"sentiment_is_correct", type logical},

        // ── model topic ──
        {"topic_predicted", type text},
        {"topic_is_correct", type logical},
        {"has_score", type logical},

        // ── case (left join, ~91% match) ──
        {"has_case", type logical},
        {"case_id", type text},
        {"customer_id", type text},
        {"case_status", type text},
        {"case_created_date_key", Int64.Type}, // FK → dim_date (inactive)
        {"case_closed_date_key", Int64.Type},  // FK → dim_date (inactive), null if open
        {"days_to_close", Int64.Type},         // null while open
        {"is_repeat_contact_7d", type logical},// forward-looking
        {"is_escalated", type logical},
        {"is_case_open", type logical},

        // ── survey (left join, ~15% match) ──
        {"has_survey_response", type logical},
        {"survey_response_id", type text},
        {"csat_score", Int64.Type},            // 1–5, null if no response
        {"is_resolved", type logical},         // TRI-STATE: null if no response
        {"is_csat_promoter", type logical},    // score ≥ 4
        {"is_csat_detractor", type logical},   // score ≤ 2
        {"survey_comment", type text},

        // ── coaching + provenance ──
        {"coaching_note", type text},
        {"generated_by", type text},
        {"model_version", type text},
        {"scored_at", type datetimezone}       // ISO 8601 with offset
    }
)


// ─────────────────────────────────────────────────────────────────────
//  QUERY 9 of 9  ·  name it:  fact_sqm_monthly
// ─────────────────────────────────────────────────────────────────────
//  Grain: one row per month × team. date_key is the FIRST of the month, so
//  this fact only joins dim_date on 12 days a year — which is why the FCR
//  MoM measure uses PREVIOUSMONTH rather than DATEADD.

fnLoadTable(
    "fact_sqm_monthly.csv",
    {
        {"sqm_key", Int64.Type},               // PK
        {"date_key", Int64.Type},              // FK → dim_date, 1st of month
        {"month_key", Int64.Type},             // YYYYMM
        {"team_key", Int64.Type},              // FK → dim_team
        {"fcr_pct", type number},              // ⚠ a stored rate — never SUM
        {"industry_benchmark_fcr_pct", type number},
        {"fcr_vs_benchmark_pp", type number},
        {"is_above_benchmark", type logical}
    }
)


// =====================================================================
//  AFTER LOADING
// =====================================================================
//
//  1. Create the relationships listed in powerbi/model_relationships.md.
//     Seven are active, four inactive, all one-to-many, all single-direction.
//     M does not create relationships — that is a model step.
//
//  2. dim_date → Mark as date table (dim_date[date]).
//
//  3. Sort by column:
//       dim_date[month_name]    → dim_date[month_name_sort]
//       dim_date[weekday_name]  → dim_date[weekday_name_sort]
//       dim_channel[channel_name] → dim_channel[channel_sort]
//
//  4. Hide every *_key column from report view, plus
//     fact_interactions[true_sentiment] unless the page is about model accuracy.
//
//  5. Paste powerbi/measures.dax.
//
//  TROUBLESHOOTING
//    "We couldn't parse the input" on a logical column — pandas writes
//    True/False, which Power Query accepts case-insensitively. If a locale
//    rejects it, insert before the type step:
//        Table.ReplaceValue(Promoted, "True", "TRUE", Replacer.ReplaceText, {"col"})
//
//    Row counts too high and text columns look shredded — QuoteStyle was lost.
//    Coaching notes contain newlines; QuoteStyle.Csv is mandatory.
//
//    Dates arrive as text or error — the culture argument was dropped from
//    Table.TransformColumnTypes. It must stay "en-US" for ISO dates.
//
//  REFRESHING THE SOURCE DATA
//      python src/generate_all.py        # synthetic source data
//      python src/run_pipeline.py        # model scores
//      python src/build_model_tables.py  # these seven CSVs
