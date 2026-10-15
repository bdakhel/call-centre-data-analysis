"""The DAX measure library, reimplemented in pandas.

Every function here mirrors a measure in `powerbi/measures.dax`. The Streamlit
report and the Power BI report must agree, so there is exactly one definition
of each metric per tool and this file is the pandas half.

Each function takes an already-filtered frame and returns a scalar, which is
the pandas equivalent of a measure evaluating inside a filter context. Call
them after slicing, never before.

The semantics that are easy to get wrong, and are deliberately preserved:

  * AHT vs target SUMS each contact's own target. Agents have different
    targets (420-480s), so averaging dim_agent would weight a 10-contact agent
    like a 170-contact one.
  * Repeat contact rate is over CASES, not contacts. The flag is only
    meaningful where a case exists.
  * CSAT is over RESPONDENTS, not contacts. ~15% answer, skewed to the unhappy.
  * FCR is an unweighted mean of a stored rate. The source gives no counts, so
    no weighted roll-up is possible. Flagged, not silently averaged.
  * Nothing returns a pre-computed ratio from a stored column; every rate is
    computed from its components at the grain being displayed.

All data is SYNTHETIC (see CLAUDE.md).
"""

import numpy as np
import pandas as pd

# Validated end-of-call decision band (docs/validation.md).
NEG_BAND, POS_BAND = -0.25, 0.90
RECOVERY_SHIFT, DECLINE_SHIFT = 0.4, -0.3


def _safe_divide(numerator, denominator):
    """DIVIDE() — blank rather than an error on a zero denominator."""
    if denominator in (0, None) or pd.isna(denominator):
        return np.nan
    return numerator / denominator


# --- Base -----------------------------------------------------------------

def interactions(df):
    return len(df)


def scored_interactions(df):
    return int(df["has_score"].sum())


def cases(df):
    return int(df["has_case"].sum())


def survey_responses(df):
    return int(df["has_survey_response"].sum())


def handle_seconds(df):
    return float(df["handle_time_sec"].sum())


def target_handle_seconds(df):
    """SUMX over each row's OWN target — see the module docstring."""
    return float(df["aht_target_sec"].sum())


# --- Volume ---------------------------------------------------------------

def customers(df):
    return int(df["customer_id"].nunique())


def contacts_per_business_day(df, business_days):
    return _safe_divide(len(df), business_days)


# --- Efficiency -----------------------------------------------------------

def aht_minutes(df):
    return _safe_divide(handle_seconds(df), len(df) * 60)


def hold_minutes(df):
    return _safe_divide(float(df["hold_time_sec"].sum()), len(df) * 60)


def hold_rate_pct(df):
    return _safe_divide(int(df["had_hold"].sum()), len(df))


def aht_target_minutes(df):
    return _safe_divide(target_handle_seconds(df), len(df) * 60)


def aht_vs_target_pct(df):
    target = target_handle_seconds(df)
    return _safe_divide(handle_seconds(df) - target, target)


def aht_vs_target_minutes(df):
    return _safe_divide(handle_seconds(df) - target_handle_seconds(df), len(df) * 60)


def pct_within_aht_target(df):
    within = (df["handle_time_sec"] <= df["aht_target_sec"]).sum()
    return _safe_divide(int(within), len(df))


# --- Survey ---------------------------------------------------------------

def csat_pct(df):
    return _safe_divide(int(df["is_csat_promoter"].sum()), survey_responses(df))


def csat_detractor_pct(df):
    return _safe_divide(int(df["is_csat_detractor"].sum()), survey_responses(df))


def avg_csat_score(df):
    scores = df["csat_score"].dropna()
    return float(scores.mean()) if len(scores) else np.nan


def survey_response_rate_pct(df):
    return _safe_divide(survey_responses(df), len(df))


def csat_target_pct(df):
    """Weighted by response volume, not averaged across the roster."""
    responders = df[df["has_survey_response"]]
    return _safe_divide(float(responders["csat_target"].sum()), len(responders))


def csat_vs_target_pp(df):
    actual, target = csat_pct(df), csat_target_pct(df)
    if pd.isna(actual) or pd.isna(target):
        return np.nan
    return (actual - target) * 100


def survey_resolved_pct(df):
    resolved = df["is_resolved"].fillna(False).astype(bool).sum()
    return _safe_divide(int(resolved), survey_responses(df))


# --- FCR (from the monthly scorecard, a separate fact) --------------------

def fcr_pct(sqm):
    """⚠️ Unweighted mean of a stored rate. See the module docstring."""
    return float(sqm["fcr_pct"].mean()) if len(sqm) else np.nan


def fcr_benchmark_pct(sqm):
    return float(sqm["industry_benchmark_fcr_pct"].mean()) if len(sqm) else np.nan


def fcr_vs_benchmark_pp(sqm):
    if not len(sqm):
        return np.nan
    return fcr_pct(sqm) - fcr_benchmark_pct(sqm)


# --- Repeat contact and cases ---------------------------------------------

def repeat_contacts(df):
    return int(df["is_repeat_contact_7d"].sum())


def repeat_contact_rate_pct(df):
    return _safe_divide(repeat_contacts(df), cases(df))


def escalation_rate_pct(df):
    return _safe_divide(int(df["is_escalated"].sum()), cases(df))


def avg_days_to_close(df):
    days = df["days_to_close"].dropna()
    return float(days.mean()) if len(days) else np.nan


# --- Sentiment ------------------------------------------------------------

def pct_negative(df):
    return _safe_divide(int((df["sentiment_label"] == "negative").sum()),
                        scored_interactions(df))


def pct_positive(df):
    return _safe_divide(int((df["sentiment_label"] == "positive").sum()),
                        scored_interactions(df))


def net_sentiment(df):
    pos, neg = pct_positive(df), pct_negative(df)
    if pd.isna(pos) or pd.isna(neg):
        return np.nan
    return (pos - neg) * 100


def avg_sentiment_score(df):
    return float(df["sentiment_score"].mean()) if len(df) else np.nan


def sentiment_shift(df):
    return float(df["sentiment_shift"].mean()) if len(df) else np.nan


def recovered_contacts(df):
    return int((df["sentiment_shift"] > RECOVERY_SHIFT).sum())


def deteriorated_contacts(df):
    return int((df["sentiment_shift"] < DECLINE_SHIFT).sum())


def pct_negative_calibrated(df):
    """✅ The reading validation supports — judged on how the contact ENDED."""
    return _safe_divide(int((df["sentiment_end"] < NEG_BAND).sum()),
                        scored_interactions(df))


def net_sentiment_calibrated(df):
    pos = int((df["sentiment_end"] > POS_BAND).sum())
    neg = int((df["sentiment_end"] < NEG_BAND).sum())
    return _safe_divide(pos - neg, scored_interactions(df)) * 100


# --- Coaching queue -------------------------------------------------------

def coaching_priority(df):
    """0-100 per contact. Weights are a judgement call, not a fitted model."""
    score = pd.Series(0, index=df.index, dtype="float64")
    score += np.where(df["sentiment_end"] < NEG_BAND, 40, 0)
    score += np.where(df["sentiment_shift"] < DECLINE_SHIFT, 20, 0)
    score += np.where(df["hold_time_sec"] > 180, 15, 0)
    score += np.where(df["handle_time_sec"] > df["aht_target_sec"] * 1.25, 10, 0)
    score += np.where(df["is_repeat_contact_7d"], 10, 0)
    score += np.where(df["is_escalated"], 5, 0)
    return score


def coaching_queue(df, top_n=10):
    """The top N contacts by priority, ranked.

    Ranks inside whatever `df` has already been filtered to — the pandas
    equivalent of ALLSELECTED, so a lead filtering to their own team gets
    their top ten rather than ten rows belonging to someone else.
    """
    out = df.copy()
    out["coaching_priority"] = coaching_priority(out)
    out = out.sort_values("coaching_priority", ascending=False).head(top_n)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


# --- Month-on-month -------------------------------------------------------

def by_month(df, measure, month_col="month"):
    """Run a measure per month. Returns a Series indexed by month."""
    return df.groupby(month_col, observed=True).apply(
        measure, include_groups=False)


def mom_change(series):
    """Previous-month delta for a monthly series. First month is NaN."""
    return series.diff()


# --- Report furniture -----------------------------------------------------

def data_as_of(df_unfiltered):
    """Latest contact in the model — NOT today's date.

    Reading the clock would hide a failed refresh behind a date that keeps
    advancing. This goes stale visibly.
    """
    return pd.to_datetime(df_unfiltered["start_datetime"]).max()


# Everything a KPI card needs, in one call, so pages stay declarative.
HEADLINE = {
    "Contacts": (interactions, "{:,.0f}"),
    "AHT (min)": (aht_minutes, "{:.1f}"),
    "CSAT %": (csat_pct, "{:.1%}"),
    "Net sentiment": (net_sentiment, "{:+.0f}"),
    "% negative": (pct_negative, "{:.1%}"),
    "Repeat 7d": (repeat_contact_rate_pct, "{:.1%}"),
}
