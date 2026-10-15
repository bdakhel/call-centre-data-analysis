"""Streamlit companion to the Power BI report — and a live transcript scorer.

Two things in one app:

  * **The report.** The five pages specified in powerbi/BUILD_GUIDE.md, built
    on the same star schema and the same measure definitions, so you can see
    the thing working without a Power BI licence or a Windows machine. Every
    number comes from src/measures.py, which is the pandas half of
    powerbi/measures.dax — one definition per metric, per tool.
  * **The scorer.** Paste a transcript and watch the pipeline run on it live.

It also simulates row-level security, so you can see what a CCR or a team lead
would actually be allowed to see (powerbi/rls_roles.md).

Models load once per session via st.cache_resource, so only the first score is
slow. The report pages load no models at all.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    streamlit run src/demo_app.py
"""

import re
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import measures as M  # noqa: E402

PROCESSED = ROOT / "data/processed"
INTERACTIONS = ROOT / "data/raw/interactions.csv"
VALIDATION = ROOT / "docs/validation.md"

# Palette from powerbi/theme.json — validated for colour-vision deficiency.
BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, MUTED, GREY, SURFACE = "#0b0b0b", "#52514e", "#c9c8c3", "#fcfcfb"
GOOD, BAD = "#0ca30c", "#d03b3b"
SENTIMENT_COLOUR = {"negative": BAD, "neutral": MUTED, "positive": GOOD}

st.set_page_config(page_title="Utility CX Insights", page_icon="⚡", layout="wide")

# NOTE ON use_container_width
# Streamlit 1.6x renamed this to width="stretch" and deprecates the old
# spelling, but it did NOT add `width` to every function at the same release:
# st.altair_chart and st.button on 1.37 reject it outright with a TypeError.
# use_container_width works on both, so it stays until the floor version is
# raised. If you see a deprecation warning, that is the trade — not a bug.





# --- Data ------------------------------------------------------------------

@st.cache_data(show_spinner="Loading the model tables…")
def load_model():
    """The star schema, flattened once for the whole app."""
    need = ["fact_interactions", "dim_agent", "dim_team", "dim_topic",
            "dim_channel", "dim_date", "fact_sqm_monthly"]
    if any(not (PROCESSED / f"{t}.csv").exists() for t in need):
        return None

    fact = pd.read_csv(PROCESSED / "fact_interactions.csv", parse_dates=["start_datetime"])
    agent = pd.read_csv(PROCESSED / "dim_agent.csv")
    team = pd.read_csv(PROCESSED / "dim_team.csv")
    topic = pd.read_csv(PROCESSED / "dim_topic.csv")
    channel = pd.read_csv(PROCESSED / "dim_channel.csv")
    date = pd.read_csv(PROCESSED / "dim_date.csv", parse_dates=["date"])
    sqm = pd.read_csv(PROCESSED / "fact_sqm_monthly.csv")

    df = (fact
          .merge(agent[["agent_key", "agent_id", "agent_name", "agent_email",
                        "team_key", "tenure_band", "is_new_hire",
                        "csat_target", "aht_target_sec"]], on="agent_key")
          .merge(team[["team_key", "team_name", "team_lead", "team_lead_email"]], on="team_key")
          .merge(topic[["topic_key", "topic_name", "topic_group"]], on="topic_key")
          .merge(channel[["channel_key", "channel_name"]], on="channel_key"))
    df["month"] = df["start_datetime"].dt.to_period("M").astype(str)
    df["is_resolved"] = df["is_resolved"].astype("object")

    sqm = sqm.merge(team[["team_key", "team_name"]], on="team_key")
    sqm["month"] = (sqm["month_key"].astype(str).str[:4] + "-"
                    + sqm["month_key"].astype(str).str[4:])
    return {"fact": df, "sqm": sqm, "date": date, "agent": agent, "team": team}


@st.cache_data
def load_accuracy():
    """Measured accuracy, parsed from docs/validation.md rather than hardcoded.

    If the validation is re-run and the numbers move, the app stops quoting
    the old ones. Degrades to empty, never to a stale figure.
    """
    facts = {}
    if not VALIDATION.exists():
        return facts
    text = VALIDATION.read_text()
    row = re.search(r"\|\s*\*\*AI model\*\*\s*\|\s*\*\*(\d+)%\*\*\s*\|\s*\*\*([\d.]+)", text)
    if row:
        facts["accuracy"], facts["kappa"] = int(row.group(1)), float(row.group(2))
    ceiling = re.search(r'\|\s*\*The "right answer" in our test data\*\s*\|\s*\*(\d+)%\*', text)
    if ceiling:
        facts["ceiling"] = int(ceiling.group(1))
    lift = re.search(r"agreement rises from \*\*(\d+)% to (\d+)%\*\*", text)
    if lift:
        facts["whole_call"], facts["end_only"] = int(lift.group(1)), int(lift.group(2))
    sample = re.search(r"We took (\d+) conversations", text)
    if sample:
        facts["n"] = int(sample.group(1))
    return facts


@st.cache_data
def load_samples():
    if not INTERACTIONS.exists():
        return pd.DataFrame()
    df = pd.read_csv(INTERACTIONS)
    picks = [g[g["true_sentiment"] == mood].iloc[0]
             for _, g in df.groupby("topic")
             for mood in ["negative", "neutral", "positive"]
             if len(g[g["true_sentiment"] == mood])]
    out = pd.DataFrame(picks)
    out["label"] = out["topic"] + "  ·  " + out["channel"] + "  ·  " + out["interaction_id"]
    return out.reset_index(drop=True)


# --- Cached models (scorer page only) -------------------------------------

@st.cache_resource(show_spinner="Loading the sentiment model (first run only)…")
def load_scorer():
    import sentiment
    return sentiment.RobertaScorer(use_cache=False, batch_size=8)


@st.cache_resource
def load_classifier():
    import topics
    return topics.RuleTopicClassifier()


@st.cache_resource
def load_coaching():
    import coaching
    return coaching


# --- Chart helpers ---------------------------------------------------------

def base_chart(data, height=260):
    return alt.Chart(data).properties(height=height).configure_view(
        strokeWidth=0).configure_axis(
        labelColor=MUTED, titleColor=MUTED, gridColor="#ebeae6",
        domainColor="#d8d7d2", tickColor="#d8d7d2")


def kpi_row(items):
    """items: list of (label, value_string, delta_string or None, help)."""
    for col, (label, value, delta, helptext) in zip(st.columns(len(items)), items):
        col.metric(label, value, delta, help=helptext)


def fmt(value, pattern, blank="—"):
    return blank if value is None or (isinstance(value, float) and pd.isna(value)) \
        else pattern.format(value)


# --- Sidebar: provenance, RLS, filters -------------------------------------

def sidebar(model):
    accuracy = load_accuracy()

    with st.sidebar:
        st.error("**All data is synthetic.** No real customers, agents or "
                 "conversations.", icon="⚠️")

        if model is None:
            st.stop()

        fact = model["fact"]
        st.caption(f"**Data as of {M.data_as_of(fact):%d %b %Y}** · "
                   f"{len(fact):,} contacts")

        # --- Row-level security simulation -------------------------------
        st.subheader("View as")
        role = st.selectbox(
            "Role", ["Full access (no RLS)", "Team Lead", "CCR"],
            label_visibility="collapsed",
            help="Simulates the roles in powerbi/rls_roles.md. RLS filters "
                 "dim_agent or dim_team; everything downstream follows.")

        scoped, identity = fact, None
        if role == "Team Lead":
            lead = st.selectbox("Which lead", sorted(model["team"]["team_lead"]))
            scoped = fact[fact["team_lead"] == lead]
            identity = fact.loc[fact["team_lead"] == lead, "team_lead_email"].iloc[0] \
                if len(scoped) else "—"
        elif role == "CCR":
            who = st.selectbox("Which agent", sorted(model["agent"]["agent_name"]))
            scoped = fact[fact["agent_name"] == who]
            identity = fact.loc[fact["agent_name"] == who, "agent_email"].iloc[0] \
                if len(scoped) else "—"

        if identity:
            st.caption(f"`USERPRINCIPALNAME()` → `{identity}`")
            st.caption(f"Sees **{len(scoped):,}** of {len(fact):,} contacts.")

        # --- Slicers ------------------------------------------------------
        st.subheader("Filters")
        months = sorted(scoped["month"].unique())
        chosen_months = st.multiselect("Month", months, default=months)
        teams = sorted(scoped["team_name"].unique())
        chosen_teams = st.multiselect("Team", teams, default=teams)
        channels = sorted(scoped["channel_name"].unique())
        chosen_channels = st.multiselect("Channel", channels, default=channels)
        topics_all = sorted(scoped["topic_name"].unique())
        chosen_topics = st.multiselect("Topic", topics_all, default=topics_all)

        filtered = scoped[
            scoped["month"].isin(chosen_months)
            & scoped["team_name"].isin(chosen_teams)
            & scoped["channel_name"].isin(chosen_channels)
            & scoped["topic_name"].isin(chosen_topics)]

        st.caption(f"**{len(filtered):,}** contacts in scope")

        # --- Accuracy -----------------------------------------------------
        st.divider()
        st.subheader("How accurate is the model?")
        if accuracy:
            st.metric("Agreement with a human reader", f"{accuracy.get('accuracy','–')}%",
                      help=f"Measured on {accuracy.get('n',100)} blind-read transcripts.")
            st.caption(
                f"Cohen's κ **{accuracy.get('kappa','–')}**. The generator's own label "
                f"agrees **{accuracy.get('ceiling','–')}%** of the time, so ~90% is the "
                f"ceiling. Judging the **end** of a contact lifts held-out agreement "
                f"from **{accuracy.get('whole_call','–')}% to {accuracy.get('end_only','–')}%**.")
        st.warning("The model reads customers as **unhappier than they are** — every "
                   "validation error went that way. A negative label means *worth a "
                   "look*, never a verdict.", icon="📉")
        st.caption("`docs/validation.md` · `powerbi/BUILD_GUIDE.md`")

    sqm = model["sqm"]
    sqm_filtered = sqm[sqm["month"].isin(chosen_months) & sqm["team_name"].isin(chosen_teams)]
    business_days = int(model["date"]["is_business_day"].sum())
    return filtered, sqm_filtered, business_days, role


# --- Page: Executive Scorecard --------------------------------------------

def page_executive():
    df, sqm, business_days, _ = CONTEXT
    st.title("📊 Executive Scorecard")
    st.caption("How the centre is doing. Every figure is defined in "
               "`powerbi/measures.dax` and computed identically here.")

    if df.empty:
        st.info("No contacts match the current filters.")
        return

    monthly = {m: g for m, g in df.groupby("month")}
    ordered = sorted(monthly)
    this_m = monthly[ordered[-1]] if ordered else df
    prev_m = monthly[ordered[-2]] if len(ordered) > 1 else None

    def delta(measure, pattern="{:+.1f}", scale=1.0):
        if prev_m is None:
            return None
        now, before = measure(this_m), measure(prev_m)
        if pd.isna(now) or pd.isna(before):
            return None
        return pattern.format((now - before) * scale)

    kpi_row([
        ("Contacts", f"{M.interactions(df):,}", delta(M.interactions, "{:+,.0f}"),
         "COUNTROWS(fact_interactions)"),
        ("AHT (min)", fmt(M.aht_minutes(df), "{:.1f}"), delta(M.aht_minutes),
         "Handle seconds ÷ (contacts × 60). Down is good."),
        ("CSAT %", fmt(M.csat_pct(df), "{:.1%}"), delta(M.csat_pct, "{:+.1f}", 100),
         "Share of RESPONDENTS scoring 4–5. ~15% respond, skewed unhappy."),
        ("FCR %", fmt(M.fcr_pct(sqm), "{:.1f}"), None,
         "Unweighted mean of a stored rate — trustworthy at month × team only."),
        ("Net sentiment", fmt(M.net_sentiment(df), "{:+.0f}"), delta(M.net_sentiment),
         "% positive − % negative, −100…+100."),
        ("Repeat 7d", fmt(M.repeat_contact_rate_pct(df), "{:.1%}"),
         delta(M.repeat_contact_rate_pct, "{:+.1f}", 100),
         "Repeat contacts ÷ CASES. Down is good."),
    ])
    if prev_m is not None:
        st.caption(f"Deltas compare **{ordered[-1]}** with **{ordered[-2]}**. "
                   f"For AHT and Repeat 7d, a negative delta is an improvement.")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("**Volume and sentiment by month**")
        vol = pd.DataFrame({
            "month": ordered,
            "contacts": [M.interactions(monthly[m]) for m in ordered],
            "net_sentiment": [M.net_sentiment(monthly[m]) for m in ordered]})
        bars = alt.Chart(vol).mark_bar(color=BLUE, size=28).encode(
            x=alt.X("month:N", title=None),
            y=alt.Y("contacts:Q", title="contacts"),
            tooltip=["month", "contacts"])
        line = alt.Chart(vol).mark_line(color=ORANGE, strokeWidth=2.5,
                                        point=alt.OverlayMarkDef(size=55)).encode(
            x="month:N",
            y=alt.Y("net_sentiment:Q", title="net sentiment",
                    scale=alt.Scale(zero=False)),
            tooltip=["month", alt.Tooltip("net_sentiment:Q", format="+.1f")])
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent")
                        .properties(height=280), use_container_width=True)
        st.caption("Blue bars: contacts. Orange line: net sentiment.")

    with right:
        st.markdown("**First contact resolution vs industry benchmark**")
        if sqm.empty:
            st.info("No scorecard rows for the current filters.")
        else:
            fcr = sqm.groupby("month", as_index=False).agg(
                fcr=("fcr_pct", "mean"), benchmark=("industry_benchmark_fcr_pct", "mean"))
            melted = fcr.melt("month", var_name="series", value_name="value")
            chart = alt.Chart(melted).mark_line(strokeWidth=2.5,
                                                point=alt.OverlayMarkDef(size=55)).encode(
                x=alt.X("month:N", title=None),
                y=alt.Y("value:Q", title="FCR %", scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", scale=alt.Scale(
                    domain=["fcr", "benchmark"], range=[BLUE, GREY]),
                    legend=alt.Legend(title=None, orient="top")),
                tooltip=["month", "series", alt.Tooltip("value:Q", format=".1f")])
            st.altair_chart(chart.properties(height=280), use_container_width=True)
            st.caption("⚠️ Unweighted mean — see the KPI tooltip.")

    st.divider()
    low, high = st.columns([2, 3])

    with low:
        st.markdown("**Contacts by topic**")
        by_topic = (df.groupby("topic_name").size().sort_values()
                    .rename("contacts").reset_index())
        chart = alt.Chart(by_topic).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
            y=alt.Y("topic_name:N", sort="-x", title=None),
            x=alt.X("contacts:Q", title=None),
            tooltip=["topic_name", "contacts"])
        labels = chart.mark_text(align="left", dx=4, color=MUTED).encode(text="contacts:Q")
        st.altair_chart((chart + labels).properties(height=240), use_container_width=True)

    with high:
        st.markdown("**Team scorecard**")
        rows = []
        for name, g in df.groupby("team_name"):
            team_sqm = sqm[sqm["team_name"] == name]
            rows.append({
                "Team": name, "Contacts": M.interactions(g),
                "AHT (min)": M.aht_minutes(g), "CSAT %": M.csat_pct(g),
                "FCR %": M.fcr_pct(team_sqm),
                "vs benchmark (pp)": M.fcr_vs_benchmark_pp(team_sqm),
                "Net sentiment": M.net_sentiment(g),
                "Repeat 7d": M.repeat_contact_rate_pct(g)})
        st.dataframe(
            pd.DataFrame(rows).set_index("Team"),
            use_container_width=True, height=240,
            column_config={
                "Contacts": st.column_config.NumberColumn(format="%d"),
                "AHT (min)": st.column_config.NumberColumn(format="%.1f"),
                "CSAT %": st.column_config.NumberColumn(format="%.1f%%"),
                "FCR %": st.column_config.NumberColumn(format="%.1f"),
                "vs benchmark (pp)": st.column_config.NumberColumn(format="%+.1f"),
                "Net sentiment": st.column_config.NumberColumn(format="%+.0f"),
                "Repeat 7d": st.column_config.NumberColumn(format="%.1f%%")})


# --- Page: Sentiment & Topics ---------------------------------------------

def page_sentiment():
    df, _, _, _ = CONTEXT
    st.title("💬 Sentiment & Topics")
    st.caption("Where the unhappiness concentrates — and how far the model can "
               "be trusted about it.")

    if df.empty:
        st.info("No contacts match the current filters.")
        return

    kpi_row([
        ("% negative", fmt(M.pct_negative(df), "{:.1%}"), None, "Uncalibrated label."),
        ("% negative (calibrated)", fmt(M.pct_negative_calibrated(df), "{:.1%}"), None,
         "Judged on the END of the contact — the reading validation supports."),
        ("Net sentiment", fmt(M.net_sentiment(df), "{:+.0f}"), None, None),
        ("Sentiment shift", fmt(M.sentiment_shift(df), "{:+.2f}"), None,
         "How far sentiment travels from the first third of the customer's turns to the last."),
        ("Recovered", f"{M.recovered_contacts(df):,}", None, "Shift > +0.4 — started rough, ended well."),
        ("Deteriorated", f"{M.deteriorated_contacts(df):,}", None, "Shift < −0.3 — worth listening to."),
    ])

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("**Sentiment mix by topic**")
        mix = (df.groupby(["topic_name", "sentiment_label"]).size()
               .rename("contacts").reset_index())
        order = (df.assign(neg=df["sentiment_label"].eq("negative"))
                 .groupby("topic_name")["neg"].mean().sort_values(ascending=False).index.tolist())
        chart = alt.Chart(mix).mark_bar().encode(
            y=alt.Y("topic_name:N", sort=order, title=None),
            x=alt.X("contacts:Q", stack="normalize", title="share of contacts",
                    axis=alt.Axis(format="%")),
            color=alt.Color("sentiment_label:N", scale=alt.Scale(
                domain=list(SENTIMENT_COLOUR), range=list(SENTIMENT_COLOUR.values())),
                legend=alt.Legend(title=None, orient="top")),
            tooltip=["topic_name", "sentiment_label", "contacts"])
        st.altair_chart(chart.properties(height=300), use_container_width=True)

    with right:
        st.markdown("**% negative over time, by team**")
        trend = (df.assign(neg=df["sentiment_label"].eq("negative"))
                 .groupby(["month", "team_name"], as_index=False)["neg"].mean())
        trend["neg"] *= 100
        chart = alt.Chart(trend).mark_line(strokeWidth=2.2,
                                           point=alt.OverlayMarkDef(size=45)).encode(
            x=alt.X("month:N", title=None),
            y=alt.Y("neg:Q", title="% negative"),
            color=alt.Color("team_name:N", scale=alt.Scale(
                range=[BLUE, ORANGE, AQUA, YELLOW]),
                legend=alt.Legend(title=None, orient="top", columns=2)),
            tooltip=["month", "team_name", alt.Tooltip("neg:Q", format=".1f")])
        st.altair_chart(chart.properties(height=300), use_container_width=True)

    st.divider()
    arc_col, acc_col = st.columns([3, 2])

    with arc_col:
        st.markdown("**The sentiment arc** — where each contact started and ended")
        sample = df.nlargest(min(400, len(df)), "sentiment_shift").copy()
        sample = pd.concat([sample, df.nsmallest(min(200, len(df)), "sentiment_shift")]
                           ).drop_duplicates("interaction_id")
        scatter = alt.Chart(sample).mark_circle(size=45, opacity=0.55).encode(
            x=alt.X("sentiment_start:Q", title="sentiment at the start",
                    scale=alt.Scale(domain=[-1, 1])),
            y=alt.Y("sentiment_end:Q", title="sentiment at the end",
                    scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("topic_name:N", scale=alt.Scale(
                range=[BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, "#6250d6"]),
                legend=alt.Legend(title=None, orient="bottom", columns=3)),
            tooltip=["interaction_id", "topic_name", "agent_name",
                     alt.Tooltip("sentiment_start:Q", format="+.2f"),
                     alt.Tooltip("sentiment_end:Q", format="+.2f")])
        diagonal = alt.Chart(pd.DataFrame({"x": [-1, 1]})).mark_line(
            color=MUTED, strokeDash=[4, 4], strokeWidth=1).encode(x="x:Q", y="x:Q")
        st.altair_chart((scatter + diagonal).properties(height=340),
                        use_container_width=True)
        st.caption("Above the dashed line = the contact improved. Capped at ~600 "
                   "points; 2,000 renders as a smear.")

    with acc_col:
        st.markdown("**Model vs the generator's label**")
        st.caption("Synthetic ground truth only — this visual could not exist in "
                   "production.")
        if "true_sentiment" in df:
            cm = pd.crosstab(df["true_sentiment"], df["sentiment_label"]).reindex(
                index=["negative", "neutral", "positive"],
                columns=["negative", "neutral", "positive"], fill_value=0)
            st.dataframe(cm, use_container_width=True)
            agree = (df["sentiment_label"] == df["true_sentiment"]).mean()
            st.metric("Agreement with the generator", f"{agree:.1%}")
            st.caption("Rows = generator, columns = model. Mass below the diagonal "
                       "is the model being gloomier than the data intended.")


# --- Page: Team & Coaching -------------------------------------------------

def page_team():
    df, _, _, role = CONTEXT
    st.title("👥 Team & Coaching")
    st.caption("The team lead's view, including the coaching queue.")

    if df.empty:
        st.info("No contacts match the current filters.")
        return

    st.markdown("**Agent scorecard**")
    rows = []
    for name, g in df.groupby("agent_name"):
        rows.append({
            "Agent": name, "Tenure": g["tenure_band"].iloc[0],
            "Contacts": M.interactions(g), "AHT (min)": M.aht_minutes(g),
            "vs target": M.aht_vs_target_pct(g), "CSAT %": M.csat_pct(g),
            "CSAT vs target (pp)": M.csat_vs_target_pp(g),
            "Net sentiment": M.net_sentiment(g), "% negative": M.pct_negative(g),
            "Repeat 7d": M.repeat_contact_rate_pct(g),
            "Shift": M.sentiment_shift(g)})
    scorecard = pd.DataFrame(rows).sort_values("Contacts", ascending=False)
    st.dataframe(
        scorecard.set_index("Agent"), use_container_width=True, height=300,
        column_config={
            "Contacts": st.column_config.ProgressColumn(
                format="%d", min_value=0, max_value=int(scorecard["Contacts"].max())),
            "AHT (min)": st.column_config.NumberColumn(format="%.1f"),
            "vs target": st.column_config.NumberColumn(format="%+.1f%%"),
            "CSAT %": st.column_config.NumberColumn(format="%.1f%%"),
            "CSAT vs target (pp)": st.column_config.NumberColumn(format="%+.1f"),
            "Net sentiment": st.column_config.NumberColumn(format="%+.0f"),
            "% negative": st.column_config.NumberColumn(format="%.1f%%"),
            "Repeat 7d": st.column_config.NumberColumn(format="%.1f%%"),
            "Shift": st.column_config.NumberColumn(format="%+.2f")})
    st.caption("Three agents have under six months' tenure and carry ~1.5× the odds "
               "of a negative contact for that reason alone — read the Tenure column "
               "before the rest of the row.")

    st.divider()
    st.markdown("**Coaching queue — the ten contacts most worth a listen**")
    top_n = st.slider("Queue length", 5, 25, 10, 5, label_visibility="collapsed")
    queue = M.coaching_queue(df, top_n)
    if queue.empty:
        st.info("Nothing in the queue for the current filters.")
        return

    display = queue[["rank", "agent_name", "interaction_id", "topic_name",
                     "coaching_priority", "sentiment_end", "sentiment_shift",
                     "hold_time_sec", "coaching_note"]].copy()
    display["hold_time_sec"] = display["hold_time_sec"] / 60
    display.columns = ["#", "Agent", "Contact", "Topic", "Priority", "Ended",
                       "Shift", "Hold (min)", "Coaching note"]
    st.dataframe(
        display.set_index("#"), use_container_width=True, height=360,
        column_config={
            "Priority": st.column_config.ProgressColumn(
                format="%d", min_value=0, max_value=100),
            "Ended": st.column_config.NumberColumn(format="%+.2f"),
            "Shift": st.column_config.NumberColumn(format="%+.2f"),
            "Hold (min)": st.column_config.NumberColumn(format="%.1f"),
            "Coaching note": st.column_config.TextColumn(width="large")})
    st.caption("Priority is weighted from how the contact ended (40), whether it "
               "deteriorated (20), hold over 3 min (15), AHT over target (10), a "
               "repeat contact (10) and escalation (5). It re-ranks inside the "
               "current filters — the pandas equivalent of `ALLSELECTED`.")

    picked = st.selectbox("Open one in the drill-through",
                          ["—"] + queue["interaction_id"].tolist())
    if picked != "—":
        st.session_state["drill_id"] = picked
        st.success(f"`{picked}` loaded — open **🔍 Interaction Detail**.")


# --- Page: CCR Self-View ---------------------------------------------------

def page_ccr():
    df, _, _, role = CONTEXT
    st.title("🙋 CCR Self-View")

    if role != "CCR":
        st.info("Set **View as → CCR** in the sidebar to see this page the way an "
                "agent would. With the CCR role applied, row-level security filters "
                "`dim_agent` to one row and every visual below becomes personal "
                "automatically — no page filter needed.", icon="🔒")

    if df.empty:
        st.info("No contacts in scope.")
        return

    agents_in_scope = df["agent_name"].nunique()
    if agents_in_scope > 1:
        st.caption(f"Currently showing **{agents_in_scope} agents** combined.")

    kpi_row([
        ("My contacts", f"{M.interactions(df):,}", None, None),
        ("My AHT (min)", fmt(M.aht_minutes(df), "{:.1f}"),
         fmt(M.aht_vs_target_minutes(df), "{:+.1f}"), "Delta is against your own target."),
        ("My CSAT %", fmt(M.csat_pct(df), "{:.1%}"),
         fmt(M.csat_vs_target_pp(df), "{:+.1f}"), "Delta is percentage points vs your target."),
        ("Net sentiment", fmt(M.net_sentiment(df), "{:+.0f}"), None, None),
        ("Sentiment shift", fmt(M.sentiment_shift(df), "{:+.2f}"), None,
         "Positive means your contacts end better than they start."),
    ])

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.markdown("**My trend**")
        trend = pd.DataFrame({
            "month": sorted(df["month"].unique()),
            "CSAT %": [M.csat_pct(g) * 100 if not pd.isna(M.csat_pct(g)) else None
                       for _, g in df.groupby("month")],
            "Net sentiment": [M.net_sentiment(g) for _, g in df.groupby("month")]})
        melted = trend.melt("month", var_name="measure", value_name="value").dropna()
        chart = alt.Chart(melted).mark_line(strokeWidth=2.5,
                                            point=alt.OverlayMarkDef(size=55)).encode(
            x=alt.X("month:N", title=None), y=alt.Y("value:Q", title=None),
            color=alt.Color("measure:N", scale=alt.Scale(range=[BLUE, ORANGE]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["month", "measure", alt.Tooltip("value:Q", format=".1f")])
        st.altair_chart(chart.properties(height=260), use_container_width=True)

    with right:
        st.markdown("**My topic mix**")
        mix = df.groupby("topic_name").size().rename("contacts").reset_index()
        chart = alt.Chart(mix).mark_arc(innerRadius=55).encode(
            theta="contacts:Q",
            color=alt.Color("topic_name:N", scale=alt.Scale(
                range=[BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, "#6250d6"]),
                legend=alt.Legend(title=None, orient="right")),
            tooltip=["topic_name", "contacts"])
        st.altair_chart(chart.properties(height=260), use_container_width=True)

    st.divider()
    st.markdown("**My coaching notes**")
    notes = df[df["coaching_note"].notna()].nlargest(
        min(25, len(df)), "start_datetime")[
        ["start_datetime", "interaction_id", "topic_name", "sentiment_label",
         "coaching_note"]].copy()
    notes["start_datetime"] = notes["start_datetime"].dt.strftime("%d %b")
    notes.columns = ["Date", "Contact", "Topic", "Sentiment", "Note"]
    st.dataframe(notes.set_index("Date"), use_container_width=True, height=300,
                 column_config={"Note": st.column_config.TextColumn(width="large")})
    st.info("These notes are developmental. They are **not** used for performance "
            "reviews, pay or discipline — see `docs/ccr_quick_reference.md`, which "
            "also explains how to dispute one.", icon="💬")


# --- Page: Interaction Detail ----------------------------------------------

def page_detail():
    df, _, _, _ = CONTEXT
    st.title("🔍 Interaction Detail")
    st.caption("One contact in full. In Power BI this is a hidden drill-through "
               "page reached by right-clicking a row.")

    if df.empty:
        st.info("No contacts in scope.")
        return

    ids = df["interaction_id"].tolist()
    default = st.session_state.get("drill_id")
    index = ids.index(default) if default in ids else 0
    chosen = st.selectbox("Contact", ids, index=index)
    row = df[df["interaction_id"] == chosen].iloc[0]

    c = st.columns(6)
    for col, (label, value) in zip(c, [
            ("Agent", row["agent_name"]), ("Team", row["team_name"]),
            ("Topic", row["topic_name"]), ("Channel", row["channel_name"]),
            ("Date", f"{row['start_datetime']:%d %b %Y}"), ("Queue", row["queue"])]):
        col.metric(label, value)

    c = st.columns(6)
    over = (row["handle_time_sec"] / row["aht_target_sec"] - 1) * 100
    for col, (label, value, helptext) in zip(c, [
            ("Handle", f"{row['handle_time_sec']/60:.1f} min", None),
            ("Hold", f"{row['hold_time_sec']/60:.1f} min", None),
            ("vs target", f"{over:+.0f}%", f"Target {row['aht_target_sec']}s"),
            ("Label", str(row["sentiment_label"]).title(), "Uncalibrated"),
            ("Score", f"{row['sentiment_score']:+.2f}", "−1 … +1"),
            ("Shift", f"{row['sentiment_shift']:+.2f}", "End minus start")]):
        col.metric(label, value, help=helptext)

    st.divider()
    left, mid, right = st.columns([2, 2, 3])

    with left:
        st.markdown("**How it moved**")
        arc = pd.DataFrame({"point": ["Start", "End"],
                            "value": [row["sentiment_start"], row["sentiment_end"]]})
        chart = alt.Chart(arc).mark_bar(size=50, cornerRadiusEnd=4).encode(
            x=alt.X("point:N", title=None, sort=["Start", "End"]),
            y=alt.Y("value:Q", title="sentiment", scale=alt.Scale(domain=[-1, 1])),
            color=alt.condition(alt.datum.value < 0, alt.value(BAD), alt.value(GOOD)),
            tooltip=[alt.Tooltip("value:Q", format="+.2f")])
        st.altair_chart(chart.properties(height=220), use_container_width=True)

    with mid:
        st.markdown("**Case & survey**")
        detail = {
            "Case": row["case_id"] if row["has_case"] else "— none —",
            "Status": row["case_status"] if row["has_case"] else "—",
            "Days to close": ("—" if pd.isna(row["days_to_close"])
                              else f"{row['days_to_close']:.0f}"),
            "Repeat within 7d": "Yes" if row["is_repeat_contact_7d"] else "No",
            "Escalated": "Yes" if row["is_escalated"] else "No",
            "CSAT": ("— no response —" if pd.isna(row["csat_score"])
                     else f"{row['csat_score']:.0f} / 5"),
        }
        st.dataframe(pd.Series(detail).rename("value").to_frame(),
                     use_container_width=True, height=220)

    with right:
        st.markdown("**Coaching note**")
        st.info(row["coaching_note"] if pd.notna(row["coaching_note"]) else "—",
                icon="💬")
        st.caption(f"Written by `{row['generated_by']}`")
        if isinstance(row.get("survey_comment"), str) and row["survey_comment"].strip():
            st.markdown("**Survey comment**")
            st.write(f"> {row['survey_comment']}")

    with st.expander("Provenance"):
        st.code(f"model_version : {row['model_version']}\n"
                f"scored_at     : {row['scored_at']}\n"
                f"topic actual  : {row['topic_name']}\n"
                f"topic predicted: {row['topic_predicted']}", language="text")

    st.caption("The transcript itself is deliberately **not** in the model — it "
               "carries the scores and the note, not the conversation.")


# --- Page: Live scorer ------------------------------------------------------

def verdict(end_score):
    if end_score > M.POS_BAND:
        return "positive", GOOD
    if end_score < M.NEG_BAND:
        return "negative", BAD
    return "neutral", MUTED


def score_transcript(transcript, hold_sec, handle_sec, target_sec):
    import sentiment as sentiment_mod
    scorer, classifier, coaching = load_scorer(), load_classifier(), load_coaching()

    windows = sentiment_mod.transcript_windows(transcript)
    turns = sentiment_mod.customer_turns(transcript)
    full, start, end = scorer.score_texts(
        [windows["full"], windows["start"], windows["end"]])
    topic = classifier.predict([transcript], show_progress=False)[0]

    facts = {"interaction_id": "DEMO-0001", "topic": topic,
             "sentiment_label": full[0], "sentiment_score": full[1],
             "sentiment_start": start[1], "sentiment_end": end[1],
             "true_sentiment": full[0], "shift": end[1] - start[1],
             "hold_time_sec": hold_sec, "handle_time_sec": handle_sec,
             "aht_target_sec": target_sec,
             "aht_ratio": handle_sec / target_sec if target_sec else 1.0}
    return {"label": full[0], "score": full[1], "start": start[1], "end": end[1],
            "shift": end[1] - start[1], "topic": topic, "turns": turns,
            "note": coaching.rule_note(facts),
            "topic_scores": classifier.scores(transcript)}


def scorer_available():
    """Is this interpreter able to run the models at all?

    The report pages need only pandas and altair, but scoring needs torch,
    transformers and vaderSentiment. Running the app from a Python that lacks
    them should say so plainly rather than throwing ImportError mid-click.
    """
    import importlib.util

    missing = [m for m in ("torch", "transformers", "vaderSentiment")
               if importlib.util.find_spec(m) is None]
    return missing


def page_scorer():
    st.title("🎧 Score a transcript")

    missing = scorer_available()
    if missing:
        st.warning(
            f"This page needs **{', '.join(missing)}**, which the Python running "
            f"this app does not have. The report pages work fine without them — "
            f"only live scoring needs the models.",
            icon="📦")
        st.markdown("Run the app from the project's virtual environment instead:")
        st.code("cd /Users/bassamd/Downloads/Sentiment\\ analysis/utility-cx-insights\n"
                ".venv/bin/streamlit run src/demo_app.py", language="bash")
        st.caption(f"Currently running: `{sys.executable}`")
        return

    st.caption("Paste a contact centre transcript, or pick a sample. Lines start "
               "with `Agent:` or `Customer:`. This page runs the real models — the "
               "first score takes a few seconds while they load.")

    samples = load_samples()
    left, right = st.columns([3, 2])

    with left:
        source = st.radio("Source", ["Pick a sample", "Paste your own"],
                          horizontal=True, label_visibility="collapsed")
        if source == "Pick a sample" and len(samples):
            choice = st.selectbox("Sample", samples["label"], label_visibility="collapsed")
            transcript = samples.loc[samples["label"] == choice, "transcript_text"].iloc[0]
            st.text_area("Transcript", transcript, height=240, disabled=True,
                         label_visibility="collapsed")
        else:
            transcript = st.text_area(
                "Transcript", height=240, label_visibility="collapsed",
                placeholder="Agent: Thanks for calling, how can I help?\n"
                            "Customer: My bill has doubled and nobody can tell me why.\n"
                            "Agent: I'm sorry — let me open a billing review.\n"
                            "Customer: Thank you, that would help.")

    with right:
        st.markdown("**Contact details** — these shape the coaching note, not the sentiment.")
        hold_sec = st.slider("Hold (seconds)", 0, 900, 0, 15)
        handle_sec = st.slider("Handle (seconds)", 60, 1800, 480, 30)
        target_sec = st.slider("Agent's AHT target (seconds)", 420, 480, 450, 10)

    if st.button("Score it", type="primary", use_container_width=True):
        import sentiment as sentiment_mod
        if not transcript or not transcript.strip():
            st.warning("Paste a transcript or pick a sample first.")
            return
        if not sentiment_mod.customer_turns(transcript):
            st.error("No customer turns found — each line needs an `Agent:` or "
                     "`Customer:` prefix. The model reads the customer's half only.")
            return

        with st.spinner("Scoring…"):
            result = score_transcript(transcript, hold_sec, handle_sec, target_sec)

        st.divider()
        calibrated, colour = verdict(result["end"])
        kpi_row([
            ("Overall label", result["label"].title(), None, "Uncalibrated — runs too negative."),
            ("Score", f"{result['score']:+.2f}", None, "P(positive) − P(negative)"),
            ("Predicted topic", result["topic"].title(), None, None),
            ("Customer turns", str(len(result["turns"])), None, None),
        ])
        st.markdown(f"#### Calibrated verdict: "
                    f"<span style='color:{colour}'>{calibrated.upper()}</span>",
                    unsafe_allow_html=True)
        st.caption(f"From the **end** of the conversation ({result['end']:+.2f}) against "
                   f"the validated band ({M.NEG_BAND:+.2f} / {M.POS_BAND:+.2f}).")

        chart_col, topic_col = st.columns([3, 2])
        with chart_col:
            arc = pd.DataFrame({"point": ["Start", "End"],
                                "value": [result["start"], result["end"]]})
            chart = alt.Chart(arc).mark_bar(size=60, cornerRadiusEnd=4).encode(
                x=alt.X("point:N", title=None, sort=["Start", "End"]),
                y=alt.Y("value:Q", title="sentiment", scale=alt.Scale(domain=[-1, 1])),
                color=alt.condition(alt.datum.value < 0, alt.value(BAD), alt.value(GOOD)))
            st.altair_chart(chart.properties(height=240), use_container_width=True)
            delta = result["shift"]
            if delta > M.RECOVERY_SHIFT:
                st.success(f"**Recovered (+{delta:.2f}).** Started rough, ended better.", icon="📈")
            elif delta < M.DECLINE_SHIFT:
                st.error(f"**Deteriorated ({delta:+.2f}).** Worth listening to.", icon="📉")
            else:
                st.info(f"**Steady ({delta:+.2f}).**", icon="➡️")

        with topic_col:
            st.markdown("**Why this topic**")
            hits = {k: v for k, v in result["topic_scores"].items() if v > 0}
            if hits:
                st.dataframe(pd.Series(hits).sort_values(ascending=False).head(4)
                             .rename("keyword score").to_frame(), use_container_width=True)
            else:
                st.caption("No keywords matched — routed to the catch-all.")

        st.markdown("**Coaching note**")
        st.info(result["note"], icon="💬")
        with st.expander("Customer turns the model read"):
            for turn in result["turns"]:
                st.markdown(f"- {turn}")


# --- Navigation -------------------------------------------------------------

# (filtered fact, filtered sqm, business days, role) — set by main(), and set
# directly by tests. Declared here so the page functions have something to read
# when the module is imported rather than run.
CONTEXT = None


def main():
    """Entry point. Guarded so `import demo_app` does not start a session.

    Streamlit executes the script with __name__ == "__main__", so this runs
    under `streamlit run` and stays quiet on import — which is what lets the
    page functions be tested directly.
    """
    global CONTEXT

    model = load_model()
    if model is None:
        st.title("⚡ Utility CX Insights")
        st.error("Model tables not found in `data/processed/`.")
        st.code("python src/generate_all.py\n"
                "python src/run_pipeline.py\n"
                "python src/build_model_tables.py", language="bash")
        st.stop()

    CONTEXT = sidebar(model)

    st.navigation([
        st.Page(page_executive, title="Executive Scorecard", icon="📊", default=True),
        st.Page(page_sentiment, title="Sentiment & Topics", icon="💬"),
        st.Page(page_team, title="Team & Coaching", icon="👥"),
        st.Page(page_ccr, title="CCR Self-View", icon="🙋"),
        st.Page(page_detail, title="Interaction Detail", icon="🔍"),
        st.Page(page_scorer, title="Score a transcript", icon="🎧"),
    ]).run()


if __name__ == "__main__":
    main()
