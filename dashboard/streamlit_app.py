from __future__ import annotations

from html import escape
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.service import ServiceError, SmallBizPulseService
from app.settings import Settings

load_dotenv()


def _format_location(city: str | None, state: str | None) -> str:
    parts = [part for part in [city, state] if part]
    return ", ".join(parts) if parts else "Unknown location"


def _first_of(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _format_prob(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "—"


def _format_not_scored_reason(reason: Any) -> str:
    text = str(reason or "").strip()
    if not text:
        return "not_scored"
    if text == "insufficient_history_for_windows_live_data_mismatch":
        return (
            "insufficient_history_for_windows_live_data_mismatch "
            "(live Yelp review file appears to be missing part of this business history)"
        )
    return text


def _to_datetime_series(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    return out.dropna(subset=[column])


def _render_trend_charts(result: dict[str, Any]) -> None:
    chart_data = result.get("chart_data") or {}

    ratings_df = pd.DataFrame(chart_data.get("ratings_by_month") or [])
    rating_bucket_df = pd.DataFrame(chart_data.get("rating_bucket_counts_by_month") or [])
    predicted_df = pd.DataFrame(chart_data.get("predicted_close_by_month") or [])
    topics_df = pd.DataFrame(chart_data.get("topics_by_month") or [])
    actual_close_month = chart_data.get("actual_close_month")

    if ratings_df.empty:
        windows = result.get("recent_windows") or []
        ratings_df = pd.DataFrame(
            [
                {"month": w.get("end_month"), "avg_stars": None, "review_count": None}
                for w in windows
                if w.get("end_month")
            ]
        )
    if predicted_df.empty:
        windows = result.get("recent_windows") or []
        predicted_df = pd.DataFrame(
            [
                {"month": w.get("end_month"), "p_closed": w.get("p_closed")}
                for w in windows
                if w.get("end_month") is not None and w.get("p_closed") is not None
            ]
        )

    st.markdown("<h3 class='sbp-subsection'>Trend views</h3>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Ratings by month")
        ratings_df = _to_datetime_series(ratings_df, "month")
        rating_bucket_df = _to_datetime_series(rating_bucket_df, "month")

        if not ratings_df.empty and ratings_df["avg_stars"].notna().any():
            ratings_df["avg_stars"] = pd.to_numeric(ratings_df["avg_stars"], errors="coerce")
            ratings_df["review_count"] = pd.to_numeric(ratings_df.get("review_count"), errors="coerce")
            ratings_line = (
                alt.Chart(ratings_df)
                .mark_line(color="#0ea5e9", point=True, strokeWidth=3)
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("avg_stars:Q", title="Avg stars", scale=alt.Scale(domain=[1, 5])),
                    tooltip=[
                        alt.Tooltip("month:T", title="Month"),
                        alt.Tooltip("avg_stars:Q", title="Avg stars", format=".2f"),
                        alt.Tooltip("review_count:Q", title="Reviews", format=".0f"),
                    ],
                )
            )
            st.altair_chart(ratings_line, use_container_width=True)
        else:
            st.caption("No monthly average-star series available.")

        st.markdown("#### Rating buckets by month (1-5 stars)")
        if not rating_bucket_df.empty:
            rating_bucket_df["stars_bucket"] = pd.to_numeric(
                rating_bucket_df.get("stars_bucket"),
                errors="coerce",
            ).astype("Int64")
            rating_bucket_df["count"] = pd.to_numeric(rating_bucket_df.get("count"), errors="coerce")
            rating_bucket_df = rating_bucket_df.dropna(subset=["stars_bucket", "count"]).copy()
            if not rating_bucket_df.empty:
                rating_bucket_df["stars_bucket_label"] = rating_bucket_df["stars_bucket"].astype(str) + "★"
                bucket_chart = (
                    alt.Chart(rating_bucket_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("month:T", title="Month"),
                        y=alt.Y("count:Q", title="Review count"),
                        color=alt.Color(
                            "stars_bucket_label:N",
                            title="Stars",
                            sort=["1★", "2★", "3★", "4★", "5★"],
                            scale=alt.Scale(
                                domain=["1★", "2★", "3★", "4★", "5★"],
                                range=["#ef4444", "#f97316", "#f59e0b", "#22c55e", "#16a34a"],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("month:T", title="Month"),
                            alt.Tooltip("stars_bucket_label:N", title="Stars"),
                            alt.Tooltip("count:Q", title="Count", format=".0f"),
                        ],
                    )
                )
                st.altair_chart(bucket_chart, use_container_width=True)
            else:
                st.caption("No star-bucket count series available.")
        else:
            st.caption("No star-bucket count series available.")

    with right:
        st.markdown("#### Predicted close by month")
        predicted_df = _to_datetime_series(predicted_df, "month")
        if not predicted_df.empty:
            predicted_df["p_closed"] = pd.to_numeric(predicted_df["p_closed"], errors="coerce")
            predicted_df = predicted_df.dropna(subset=["p_closed"]).sort_values("month")
            if not predicted_df.empty:
                # Densify to monthly cadence and interpolate for a cleaner historical trend.
                full_months = pd.DataFrame(
                    {"month": pd.date_range(predicted_df["month"].min(), predicted_df["month"].max(), freq="MS")}
                )
                dense = full_months.merge(predicted_df[["month", "p_closed"]], on="month", how="left")
                dense["p_closed_monthly"] = (
                    dense["p_closed"].interpolate(method="linear", limit_direction="both")
                )
                dense["p_closed_smooth"] = dense["p_closed_monthly"].rolling(3, min_periods=1).mean()

                observed = predicted_df.copy()
                observed["series"] = "window_end_observed"

                y_min = float(dense["p_closed_monthly"].min())
                y_max = float(dense["p_closed_monthly"].max())
                pad = max(0.03, (y_max - y_min) * 0.25)
                domain_min = max(0.0, y_min - pad)
                domain_max = min(1.0, y_max + pad)
                if domain_max - domain_min < 0.08:
                    center = (domain_min + domain_max) / 2
                    domain_min = max(0.0, center - 0.04)
                    domain_max = min(1.0, center + 0.04)

                monthly_line = (
                    alt.Chart(dense)
                    .mark_line(color="#f43f5e", strokeDash=[6, 4], strokeWidth=2)
                    .encode(
                        x=alt.X("month:T", title="Month"),
                        y=alt.Y(
                            "p_closed_monthly:Q",
                            title="p_closed (zoomed)",
                            scale=alt.Scale(domain=[domain_min, domain_max]),
                        ),
                        tooltip=[
                            alt.Tooltip("month:T", title="Month"),
                            alt.Tooltip("p_closed_monthly:Q", title="Monthly predicted", format=".3f"),
                        ],
                    )
                )
                smooth_line = (
                    alt.Chart(dense)
                    .mark_line(color="#be123c", strokeWidth=3)
                    .encode(
                        x=alt.X("month:T", title="Month"),
                        y=alt.Y(
                            "p_closed_smooth:Q",
                            title="p_closed (zoomed)",
                            scale=alt.Scale(domain=[domain_min, domain_max]),
                        ),
                        tooltip=[
                            alt.Tooltip("month:T", title="Month"),
                            alt.Tooltip("p_closed_smooth:Q", title="3-mo trend", format=".3f"),
                        ],
                    )
                )
                observed_points = (
                    alt.Chart(observed)
                    .mark_point(color="#7f1d1d", size=70, filled=True)
                    .encode(
                        x=alt.X("month:T", title="Month"),
                        y=alt.Y(
                            "p_closed:Q",
                            title="p_closed (zoomed)",
                            scale=alt.Scale(domain=[domain_min, domain_max]),
                        ),
                        tooltip=[
                            alt.Tooltip("month:T", title="Window end"),
                            alt.Tooltip("p_closed:Q", title="Observed window p_closed", format=".3f"),
                        ],
                    )
                )

                layers = [monthly_line, smooth_line, observed_points]

                close_month = pd.to_datetime(actual_close_month, errors="coerce")
                if pd.notna(close_month):
                    close_rule_df = pd.DataFrame({"close_month": [close_month]})
                    close_rule = (
                        alt.Chart(close_rule_df)
                        .mark_rule(color="#111827", strokeDash=[2, 2], strokeWidth=2)
                        .encode(x="close_month:T")
                    )
                    close_label = (
                        alt.Chart(close_rule_df)
                        .mark_text(
                            text="actual close",
                            dx=6,
                            dy=-8,
                            color="#111827",
                            fontSize=11,
                        )
                        .encode(
                            x="close_month:T",
                            y=alt.value(14),
                        )
                    )
                    layers.extend([close_rule, close_label])

                risk_chart = alt.layer(*layers).properties(height=320)
                st.altair_chart(risk_chart, use_container_width=True)
            else:
                st.caption("No predicted probability series available.")
        else:
            st.caption("No predicted probability series available.")

    st.markdown("#### Top topics by month")
    topics_df = _to_datetime_series(topics_df, "month")
    if not topics_df.empty:
        topics_df["rank"] = pd.to_numeric(topics_df.get("rank"), errors="coerce")
        topics_df["strength"] = pd.to_numeric(topics_df.get("strength"), errors="coerce")
        topics_df = topics_df.dropna(subset=["theme", "rank"]).copy()
        if not topics_df.empty:
            # Keep chart readable by limiting to most frequent themes.
            top_themes = topics_df["theme"].value_counts().head(8).index
            topics_df = topics_df[topics_df["theme"].isin(top_themes)]
            topics_df["theme_short"] = topics_df["theme"].astype(str).str.slice(0, 48)
            topic_chart = (
                alt.Chart(topics_df)
                .mark_rect(cornerRadius=2)
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("theme_short:N", title="BERTopic theme"),
                    color=alt.Color(
                        "strength:Q",
                        title="Top-3 weight",
                        scale=alt.Scale(domain=[1, 3], range=["#dbeafe", "#60a5fa", "#1d4ed8"]),
                    ),
                    tooltip=[
                        alt.Tooltip("month:T", title="Month"),
                        alt.Tooltip("theme:N", title="Theme"),
                        alt.Tooltip("rank:Q", title="Rank", format=".0f"),
                    ],
                )
            )
            st.altair_chart(topic_chart, use_container_width=True)
        else:
            st.caption("No monthly topic series available for this result.")
    else:
        st.caption("No monthly topic series available for this result.")


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Work+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #0f172a;
  --muted: #55637a;
  --line: #d8e1ec;
  --card: #ffffff;
  --teal: #0ea5e9;
  --mint: #22c55e;
  --rose: #f43f5e;
  --shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
  --radius: 10px;
}

.stApp {
  color: var(--ink);
  font-family: "Work Sans", -apple-system, BlinkMacSystemFont, sans-serif;
  background:
    radial-gradient(circle at 8% 0%, #eaf3ff 0, transparent 32%),
    radial-gradient(circle at 100% 0%, #fff8e7 0, transparent 28%),
    linear-gradient(180deg, #f8fafd 0%, #eef3f8 100%);
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stAppViewContainer"] > .main {
  padding-top: 1.2rem;
}

[data-testid="stMainBlockContainer"] {
  max-width: 1200px;
  padding-top: 0.8rem;
  padding-bottom: 2.8rem;
}

.sbp-shell {
  color: var(--ink);
}

.sbp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.sbp-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.sbp-mark {
  width: 42px;
  height: 42px;
  border-radius: 12px;
  background: linear-gradient(135deg, #0ea5e9, #67e8f9);
  color: #0f172a;
  display: grid;
  place-items: center;
  font-family: "Space Grotesk", sans-serif;
  font-weight: 700;
}

.sbp-title {
  margin: 0;
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.14rem;
}

.sbp-eyebrow {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-size: 11px;
  color: #94a3b8;
  font-weight: 600;
}

.sbp-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  color: #e2e8f0;
  background: rgba(255, 255, 255, 0.1);
  font-size: 13px;
  font-weight: 600;
}

.sbp-pill-light {
  color: var(--ink);
  border-color: rgba(14, 165, 233, 0.25);
  background: rgba(14, 165, 233, 0.12);
}

.sbp-pill-good {
  color: #14532d;
  border-color: rgba(34, 197, 94, 0.3);
  background: rgba(34, 197, 94, 0.15);
}

.sbp-pill-warn {
  color: #7c2d12;
  border-color: rgba(245, 158, 11, 0.35);
  background: rgba(245, 158, 11, 0.2);
}

.sbp-hero {
  margin-top: 10px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 22px;
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 16px;
}

.sbp-hero h1 {
  margin: 6px 0 8px;
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
  font-size: clamp(23px, 2.2vw, 28px);
}

.sbp-lede {
  margin: 0;
  color: var(--muted);
  font-size: 15px;
}

.sbp-chip {
  background: linear-gradient(135deg, #e6f4ff, #f1fbff);
  border-radius: var(--radius);
  color: #1e293b;
  padding: 14px;
  border: 1px solid #cfe3f7;
}

.sbp-chip p {
  margin: 0;
}

.sbp-chip .sbp-chip-label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.sbp-chip .sbp-chip-value {
  margin-top: 4px;
  margin-bottom: 6px;
  font-size: 16px;
  font-weight: 700;
}

.sbp-step {
  margin-top: 14px;
}

.sbp-step h2 {
  margin: 4px 0 0;
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.7rem;
}

.sbp-subsection {
  margin-top: 10px;
  margin-bottom: 7px;
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.05rem;
}

div[data-testid="stForm"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 14px 14px 6px 14px;
}

div[data-testid="stForm"] [data-testid="stMarkdownContainer"] > p {
  color: var(--muted);
  font-size: 13px;
}

div[data-testid="stTextInputRootElement"] input {
  border-radius: 10px !important;
  border: 1px solid var(--line) !important;
  font-family: "Work Sans", sans-serif;
}

div[data-testid="stTextInputRootElement"] input:focus {
  border-color: rgba(14, 165, 233, 0.55) !important;
  box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.18) !important;
}

.stButton > button, .stFormSubmitButton > button {
  border-radius: 12px !important;
  border: none !important;
  background: linear-gradient(135deg, #0ea5e9, #22c55e) !important;
  color: #0f172a !important;
  font-weight: 700 !important;
  transition: transform 120ms ease, box-shadow 120ms ease;
}

.stButton > button:hover, .stFormSubmitButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(14, 165, 233, 0.25);
}

.sbp-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 12px;
}

.sbp-candidate-name {
  margin: 0 0 4px;
  font-family: "Space Grotesk", sans-serif;
  font-size: 18px;
  color: var(--ink);
}

.sbp-small {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
}

.sbp-score-header {
  margin-top: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.sbp-score-title h2 {
  margin: 0;
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.7rem;
}

.sbp-flat {
  padding: 2px 0 10px;
}

.sbp-flat-title {
  margin: 0;
  font-size: 24px;
  line-height: 1.2;
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
}

.sbp-flat-meta {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 15px;
}

.sbp-flat-divider {
  height: 1px;
  background: var(--line);
  margin: 8px 0 10px;
}

.sbp-risk-inline {
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--card);
  padding: 12px;
}

.sbp-biz-card h3 {
  margin: 0;
  font-size: 24px;
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
}

.sbp-biz-card p {
  margin: 3px 0 0;
  color: var(--muted);
}

.sbp-risk-card .label {
  margin: 0;
  color: var(--muted);
  text-transform: uppercase;
  font-size: 12px;
}

.sbp-risk-card .value {
  margin: 6px 0 4px;
  font-size: 46px;
  line-height: 1;
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
}

.sbp-risk-card .bucket {
  margin: 0;
  color: var(--muted);
  font-size: 15px;
}

.sbp-card h4 {
  margin: 0 0 8px;
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
  font-size: 16px;
}

.sbp-list {
  margin: 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: var(--ink);
}

.sbp-list.ordered {
  padding-left: 20px;
}

.sbp-chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.sbp-chip-pill {
  display: inline-flex;
  align-items: center;
  padding: 5px 11px;
  border-radius: 999px;
  background: rgba(14, 165, 233, 0.15);
  border: 1px solid rgba(14, 165, 233, 0.25);
  color: #1f2937;
  font-weight: 600;
  font-size: 12px;
}

.sbp-keyword-label {
  margin: 10px 0 3px;
  color: var(--muted);
  font-size: 13px;
}

.sbp-keywords {
  margin: 0;
  color: var(--ink);
  line-height: 1.4;
  font-size: 14px;
}

.sbp-evidence-meta {
  margin: 0 0 8px;
  color: var(--muted);
  font-size: 15px;
}

.sbp-evidence-text {
  margin: 0;
  color: var(--ink);
  line-height: 1.5;
}

.sbp-empty {
  margin: 0;
  color: var(--muted);
}

.sbp-alert {
  margin-top: 12px;
  border: 1px solid rgba(244, 63, 94, 0.24);
  border-radius: 12px;
  background: rgba(244, 63, 94, 0.08);
  color: #9f1239;
  padding: 10px 12px;
  font-weight: 600;
}

[data-testid="stAlert"] {
  border-radius: 12px;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow);
}

div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
  border: none !important;
}

@media (max-width: 980px) {
  .sbp-hero {
    grid-template-columns: 1fr;
  }
  .sbp-flat-title {
    font-size: 22px;
  }
  .sbp-biz-card h3 {
    font-size: 22px;
  }
  .sbp-risk-card .value {
    font-size: 40px;
  }
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_shell_header() -> None:
    st.markdown(
        """
<div class="sbp-shell">
  <div class="sbp-header">
    <div class="sbp-brand">
      <div class="sbp-mark">SB</div>
      <div>
        <p class="sbp-eyebrow">Closure Risk Intelligence</p>
        <p class="sbp-title">SmallBizPulse Dashboard</p>
      </div>
    </div>
  </div>
  <div class="sbp-hero">
    <div>
      <p class="sbp-eyebrow">Artifact-first + live fallback</p>
      <h1>Find a restaurant, score closure risk, and see the likely drivers.</h1>
      <p class="sbp-lede">Search -> Select -> Score with live inference fallback when artifacts are missing.</p>
    </div>
    <div class="sbp-chip">
      <p class="sbp-chip-label">Runtime</p>
      <p class="sbp-chip-value">Search -> Select -> Score</p>
      <p>Artifact results load quickly. Live mode kicks in when dependencies and data are ready.</p>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _get_service() -> SmallBizPulseService:
    return SmallBizPulseService(Settings.from_env())


def _render_health_sidebar(service: SmallBizPulseService) -> None:
    checks = service.health()
    live_ready = bool(checks.get("live_fallback_ready"))
    pill_cls = "sbp-pill sbp-pill-good" if live_ready else "sbp-pill sbp-pill-warn"
    mode = "Live-first ready" if live_ready else "Live-first limited"
    with st.sidebar:
        st.markdown("### Runtime")
        st.markdown(
            f"<span class='{pill_cls}'>{escape(mode)}</span>",
            unsafe_allow_html=True,
        )
        with st.expander("Health checks"):
            st.json(checks)


def _render_score(result: dict[str, Any]) -> None:
    mode = result.get("scoring_mode") or "unknown"
    st.markdown(
        f"""
<div class="sbp-score-header">
  <div class="sbp-score-title">
    <p class="sbp-eyebrow">Step 2</p>
    <h2>Business scoring output</h2>
  </div>
  <span class="sbp-pill sbp-pill-light">Mode: {escape(str(mode))}</span>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.6, 1], gap="large")
    business_name = result.get("name") or result.get("business_id") or "Unknown business"
    business_id = result.get("business_id") or "—"
    status = result.get("status") or "Unknown"
    total_reviews = result.get("total_reviews")
    with left:
        st.markdown(
            f"""
<div class="sbp-flat">
  <h3 class="sbp-flat-title">{escape(str(business_name))}</h3>
  <p class="sbp-flat-meta">{escape(_format_location(result.get("city"), result.get("state")))}
  · ID: {escape(str(business_id))}</p>
  <p class="sbp-flat-meta">Status: {escape(str(status))}
  · Reviews: {escape(str(total_reviews) if total_reviews is not None else "—")}</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
<div class="sbp-risk-inline sbp-risk-card">
  <p class="label">Risk score</p>
  <p class="value">{escape(_format_prob(result.get("risk_score")))}</p>
  <p class="bucket">Bucket: {escape(str(result.get("risk_bucket") or "—"))}</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='sbp-flat-divider'></div>", unsafe_allow_html=True)

    windows_col, themes_col, recs_col = st.columns([1.05, 1.15, 1.35], gap="large")

    windows = result.get("recent_windows") or []
    with windows_col:
        st.markdown("#### Recent windows")
        if windows:
            rows: list[dict[str, Any]] = []
            for window in windows:
                month = _first_of(
                    window,
                    "end_month",
                    "window_end_month",
                    "window_end",
                    "month",
                    "date",
                )
                p_closed = _first_of(window, "p_closed", "probability", "score")
                rows.append(
                    {
                        "month": str(month) if month is not None else "—",
                        "p_closed": _format_prob(p_closed),
                    }
                )
            windows_df = pd.DataFrame(rows)
            st.dataframe(
                windows_df,
                use_container_width=True,
                hide_index=True,
                column_order=("month", "p_closed"),
            )
        else:
            st.caption("No window-level probabilities available.")

    themes = result.get("themes_top3") or []
    with themes_col:
        st.markdown("#### Themes")
        chip_html = (
            "<div class='sbp-chip-list'>"
            + "".join(
                f"<span class='sbp-chip-pill'>{escape(str(theme))}</span>"
                for theme in themes
            )
            + "</div>"
            if themes
            else "<p class='sbp-empty'>No themes.</p>"
        )
        keywords = result.get("problem_keywords") or "No keyword signal available."
        st.markdown(
            f"""
{chip_html}
<p class="sbp-keyword-label">Problem keywords</p>
<p class="sbp-keywords">{escape(str(keywords))}</p>
""",
            unsafe_allow_html=True,
        )

    recommendations = result.get("recommendations_top3") or []
    with recs_col:
        st.markdown("#### Top recommendations")
        if recommendations:
            rec_df = pd.DataFrame(
                {
                    "rank": list(range(1, len(recommendations) + 1)),
                    "recommendation": [str(rec) for rec in recommendations],
                }
            )
            st.dataframe(
                rec_df,
                use_container_width=True,
                hide_index=True,
                column_order=("rank", "recommendation"),
            )
        else:
            st.caption("No recommendations available.")
        notes = result.get("recommendation_notes")
        if notes:
            st.caption(str(notes))

    _render_trend_charts(result)

    st.markdown("<h3 class='sbp-subsection'>Evidence reviews</h3>", unsafe_allow_html=True)
    evidence = result.get("evidence_reviews") or []
    if evidence:
        rows: list[dict[str, Any]] = []
        for review in evidence:
            text = _first_of(review, "snippet", "text", "review_text", "body") or ""
            text_value = str(text)
            if len(text_value) > 600:
                text_value = f"{text_value[:597]}..."
            rows.append(
                {
                    "date": _first_of(review, "date", "review_date", "month"),
                    "stars": _first_of(review, "stars", "star_rating"),
                    "neg_prob": _first_of(
                        review,
                        "sentiment_neg_prob",
                        "neg_prob",
                        "negative_probability",
                        "p_neg",
                    ),
                    "review_text": text_value,
                }
            )

        evidence_df = pd.DataFrame(rows)
        if "neg_prob" in evidence_df.columns:
            evidence_df["neg_prob"] = evidence_df["neg_prob"].apply(_format_prob)

        st.dataframe(
            evidence_df,
            use_container_width=True,
            hide_index=True,
            column_order=("date", "stars", "neg_prob", "review_text"),
        )
    else:
        st.markdown(
            "<div class='sbp-card'><p class='sbp-empty'>No evidence reviews available for this result.</p></div>",
            unsafe_allow_html=True,
        )

    if result.get("not_scored_reason"):
        st.markdown(
            f"<div class='sbp-alert'>Not scored reason: "
            f"{escape(_format_not_scored_reason(result.get('not_scored_reason')))}</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="SmallBizPulse Dashboard",
        page_icon="📊",
        layout="wide",
    )
    _inject_styles()
    _render_shell_header()

    try:
        service = _get_service()
    except Exception as exc:
        st.error(f"Service failed to initialize: {exc}")
        st.stop()

    _render_health_sidebar(service)

    if "candidates" not in st.session_state:
        st.session_state["candidates"] = []
    if "score_result" not in st.session_state:
        st.session_state["score_result"] = None
    if "force_live_inference" not in st.session_state:
        st.session_state["force_live_inference"] = False

    st.markdown(
        """
<div class="sbp-step">
  <p class="sbp-eyebrow">Step 1</p>
  <h2>Search businesses</h2>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("search_form"):
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            name = st.text_input("Company name", placeholder="e.g., Five Guys")
        with col2:
            city = st.text_input("City (optional)")
        with col3:
            state = st.text_input("State (optional)", max_chars=2)
        include_unscorable = st.checkbox("Include unscorable matches", value=False)
        submitted = st.form_submit_button("Search")

    if submitted:
        if not name.strip():
            st.warning("Enter a company name.")
        else:
            try:
                with st.spinner("Searching..."):
                    st.session_state["candidates"] = service.search_businesses(
                        name=name.strip(),
                        city=city.strip() or None,
                        state=state.strip() or None,
                        limit=10,
                        scorable_only=not include_unscorable,
                    )
                st.session_state["score_result"] = None
            except ServiceError as exc:
                st.error(f"Search failed: {exc}")

    candidates = st.session_state.get("candidates", [])
    if candidates:
        st.markdown("<h3 class='sbp-subsection'>Candidates</h3>", unsafe_allow_html=True)
        for candidate in candidates:
            mode = "artifact ready" if candidate.get("risk_available") else "live only"
            st.markdown(
                f"""
<div class="sbp-card">
  <p class="sbp-candidate-name">{escape(str(candidate.get("name") or "Unknown business"))}</p>
  <p class="sbp-small">
    {escape(_format_location(candidate.get("city"), candidate.get("state")))}
    · ID: {escape(str(candidate.get("business_id") or "—"))}
  </p>
  <p class="sbp-small">
    Reviews: {escape(str(candidate.get("review_count") if candidate.get("review_count") is not None else "—"))}
    · {mode}
  </p>
</div>
""",
                unsafe_allow_html=True,
            )

        option_map = {
            f"{c.get('name') or 'Unknown'} | {_format_location(c.get('city'), c.get('state'))} "
            f"| reviews: {c.get('review_count') if c.get('review_count') is not None else '—'} "
            f"| {'artifact' if c.get('risk_available') else 'live'}": c
            for c in candidates
        }
        selected_label = st.selectbox("Select a business to score", list(option_map.keys()))
        selected = option_map[selected_label]
        force_live = st.checkbox(
            "Force live inference (ignore artifact cache)",
            key="force_live_inference",
            help="When enabled, artifact scores are bypassed and only live scoring is attempted for this click.",
        )
        if force_live:
            st.warning("Force live inference is ON: artifact fallback is bypassed for this score request.")
        if st.button("Score business", type="primary"):
            try:
                with st.spinner("Scoring..."):
                    result = service.score_business(
                        selected.get("business_id") or "",
                        force_live_inference=force_live,
                    )
                    st.session_state["score_result"] = result
                    if (
                        force_live
                        and result.get("availability") != "scored"
                        and selected.get("risk_available")
                    ):
                        st.info(
                            "Live scoring was forced and failed. "
                            "Artifact score exists for this business - run again with force-live OFF."
                        )
            except ServiceError as exc:
                st.error(f"Scoring failed: {exc}")
    else:
        st.markdown(
            "<div class='sbp-card'><p class='sbp-empty'>No candidates yet. Search for a business to begin.</p></div>",
            unsafe_allow_html=True,
        )

    result = st.session_state.get("score_result")
    if isinstance(result, dict):
        _render_score(result)


if __name__ == "__main__":
    main()
