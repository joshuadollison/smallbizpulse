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


def _format_pct(value: Any, digits: int = 1) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.{digits}f}%"
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


def _render_search_table_with_action(
    rows: list[dict[str, Any]],
    *,
    key_prefix: str,
    action_label: str,
) -> str | None:
    if not rows:
        return None

    frame = pd.DataFrame(rows)
    if frame.empty or "business_id" not in frame.columns:
        return None

    display_frame = frame.drop(columns=["business_id"], errors="ignore").copy()
    display_frame = display_frame.fillna("—")

    selection_rows: list[int] = []
    try:
        event = st.dataframe(
            display_frame,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"{key_prefix}_table",
        )
        if isinstance(event, dict):
            selection_rows = event.get("selection", {}).get("rows", []) or []
        else:
            selection = getattr(event, "selection", None)
            if isinstance(selection, dict):
                selection_rows = selection.get("rows", []) or []
    except TypeError:
        # Fallback for older Streamlit builds without row selection support.
        st.dataframe(display_frame, use_container_width=True, hide_index=True)

    selected_id_key = f"{key_prefix}_selected_business_id"
    if selection_rows:
        selected_idx = int(selection_rows[0])
        if 0 <= selected_idx < len(frame):
            st.session_state[selected_id_key] = str(frame.iloc[selected_idx]["business_id"])

    selected_business_id = st.session_state.get(selected_id_key)
    if selected_business_id:
        selected_row = frame[frame["business_id"].astype(str) == str(selected_business_id)]
        if not selected_row.empty:
            selected_name = str(selected_row.iloc[0].get("name") or "Unknown")
            st.caption(f"Selected: {selected_name} ({selected_business_id})")

    action_clicked = st.button(
        action_label,
        key=f"{key_prefix}_action_button",
        use_container_width=False,
        disabled=not selected_business_id,
    )
    if action_clicked and selected_business_id:
        return str(selected_business_id)
    return None


def _render_trend_charts(result: dict[str, Any]) -> None:
    chart_data = result.get("chart_data") or {}

    ratings_df = pd.DataFrame(chart_data.get("ratings_by_month") or [])
    rating_bucket_df = pd.DataFrame(chart_data.get("rating_bucket_counts_by_month") or [])
    predicted_df = pd.DataFrame(chart_data.get("predicted_close_by_month") or [])
    topics_df = pd.DataFrame(chart_data.get("topics_by_month") or [])
    topics_class_df = pd.DataFrame(chart_data.get("topics_per_class") or [])
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
                    .mark_line(point=True, strokeWidth=2.5)
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
    topics_class_df["strength"] = pd.to_numeric(topics_class_df.get("strength"), errors="coerce")
    topics_class_df["topic_id"] = pd.to_numeric(topics_class_df.get("topic_id"), errors="coerce")
    if "stars_bucket" in topics_class_df.columns:
        topics_class_df["stars_bucket"] = pd.to_numeric(
            topics_class_df.get("stars_bucket"),
            errors="coerce",
        ).astype("Int64")

    if not topics_df.empty:
        topics_df["rank"] = pd.to_numeric(topics_df.get("rank"), errors="coerce")
        topics_df["strength"] = pd.to_numeric(topics_df.get("strength"), errors="coerce")
        topics_df["topic_id"] = pd.to_numeric(topics_df.get("topic_id"), errors="coerce")
        topics_df = topics_df.dropna(subset=["theme", "rank", "strength"]).copy()

    if topics_df.empty and topics_class_df.empty:
        st.caption("No monthly topic series available for this result.")
    else:
        base_for_keys = topics_df if not topics_df.empty else topics_class_df
        extracted_topic_id = base_for_keys["theme"].astype(str).str.extract(r"topic[_\s-]*(\d+)")[0]
        base_for_keys["topic_key"] = base_for_keys["topic_id"].apply(
            lambda value: f"topic_{int(value)}" if pd.notna(value) else None
        )
        base_for_keys.loc[base_for_keys["topic_key"].isna(), "topic_key"] = extracted_topic_id.apply(
            lambda value: f"topic_{int(value)}" if pd.notna(value) else None
        )
        base_for_keys.loc[base_for_keys["topic_key"].isna(), "topic_key"] = base_for_keys["theme"].astype(str)

        if "topic_terms" not in base_for_keys.columns:
            base_for_keys["topic_terms"] = ""
        base_for_keys["topic_terms"] = base_for_keys["topic_terms"].fillna("").astype(str).str.strip()
        fallback_terms = base_for_keys["theme"].astype(str).str.extract(r":\s*(.+)$", expand=False).fillna("")
        missing_terms = base_for_keys["topic_terms"].eq("")
        base_for_keys.loc[missing_terms, "topic_terms"] = fallback_terms.loc[missing_terms]
        base_for_keys["topic_terms"] = base_for_keys["topic_terms"].apply(
            lambda value: ", ".join([part.strip() for part in str(value).split(",") if part.strip()][:6])
        )

        top_topic_keys = (
            base_for_keys.groupby("topic_key", as_index=False)["strength"]
            .sum()
            .sort_values("strength", ascending=False)
            .head(6)["topic_key"]
            .tolist()
        )

        label_rows = (
            base_for_keys.sort_values(["topic_key", "strength"], ascending=[True, False])
            .groupby("topic_key", as_index=False)
            .first()[["topic_key", "topic_terms"]]
        )
        topic_label_lookup: dict[str, str] = {}
        for row in label_rows.itertuples(index=False):
            key = str(row.topic_key)
            terms = str(row.topic_terms or "").strip()
            topic_label_lookup[key] = f"{key}: {terms}" if terms else key
        topic_order = [topic_label_lookup[key] for key in top_topic_keys if key in topic_label_lookup]

        st.markdown("##### Topics per class (top 6)")
        st.caption("BERTopic-style class view (class = stars).")
        if topics_class_df.empty:
            st.caption("No topic-by-stars class data available for this result.")
        else:
            class_df = topics_class_df.copy()
            extracted_topic_id = class_df["theme"].astype(str).str.extract(r"topic[_\s-]*(\d+)")[0]
            class_df["topic_key"] = class_df["topic_id"].apply(
                lambda value: f"topic_{int(value)}" if pd.notna(value) else None
            )
            class_df.loc[class_df["topic_key"].isna(), "topic_key"] = extracted_topic_id.apply(
                lambda value: f"topic_{int(value)}" if pd.notna(value) else None
            )
            class_df.loc[class_df["topic_key"].isna(), "topic_key"] = class_df["theme"].astype(str)
            if class_df.empty:
                st.caption("No topic-by-stars class data available for this result.")
            else:
                if "class_label" not in class_df.columns:
                    class_df["class_label"] = class_df["stars_bucket"].astype("Int64").astype(str) + "★"
                class_df["topic_label"] = class_df["topic_key"].map(topic_label_lookup)
                class_df["topic_label"] = class_df["topic_label"].fillna(class_df["theme"].astype(str))
                class_topic_order = (
                    class_df.groupby("topic_label", as_index=False)["strength"]
                    .sum()
                    .sort_values("strength", ascending=False)["topic_label"]
                    .tolist()
                )

                class_chart = (
                    alt.Chart(class_df)
                    .mark_rect(cornerRadius=2)
                    .encode(
                        x=alt.X(
                            "class_label:N",
                            title="Stars class",
                            sort=["1★", "5★"],
                        ),
                        y=alt.Y("topic_label:N", title="Topic", sort=class_topic_order),
                        color=alt.Color(
                            "strength:Q",
                            title="Strength",
                            scale=alt.Scale(range=["#dbeafe", "#60a5fa", "#1d4ed8"]),
                        ),
                        tooltip=[
                            alt.Tooltip("class_label:N", title="Stars class"),
                            alt.Tooltip("topic_label:N", title="Topic"),
                            alt.Tooltip("rank:Q", title="Rank", format=".0f"),
                            alt.Tooltip("strength:Q", title="Strength", format=".2f"),
                        ],
                    )
                )
                st.altair_chart(class_chart.properties(height=280), use_container_width=True)

        st.markdown("##### Topics over time (top 6)")
        if topics_df.empty:
            st.caption("No monthly topic series available for this result.")
        else:
            time_df = topics_df.copy()
            extracted_topic_id = time_df["theme"].astype(str).str.extract(r"topic[_\s-]*(\d+)")[0]
            time_df["topic_key"] = time_df["topic_id"].apply(
                lambda value: f"topic_{int(value)}" if pd.notna(value) else None
            )
            time_df.loc[time_df["topic_key"].isna(), "topic_key"] = extracted_topic_id.apply(
                lambda value: f"topic_{int(value)}" if pd.notna(value) else None
            )
            time_df.loc[time_df["topic_key"].isna(), "topic_key"] = time_df["theme"].astype(str)
            time_df = time_df[time_df["topic_key"].isin(top_topic_keys)].copy()
            if time_df.empty:
                st.caption("No monthly topic series available for this result.")
            else:
                timeline = (
                    time_df.groupby(["month", "topic_key"], as_index=False)["strength"]
                    .sum()
                    .sort_values(["month", "topic_key"])
                )
                if timeline.empty:
                    st.caption("No monthly topic series available for this result.")
                else:
                    all_months = pd.date_range(
                        timeline["month"].min(),
                        timeline["month"].max(),
                        freq="MS",
                    )
                    dense_index = pd.MultiIndex.from_product(
                        [all_months, top_topic_keys],
                        names=["month", "topic_key"],
                    ).to_frame(index=False)
                    timeline = dense_index.merge(
                        timeline,
                        on=["month", "topic_key"],
                        how="left",
                    )
                    timeline["strength"] = timeline["strength"].fillna(0.0)
                    timeline["topic_label"] = timeline["topic_key"].map(topic_label_lookup)

                    over_time_chart = (
                        alt.Chart(timeline)
                        .mark_line(point=True, strokeWidth=2.5)
                        .encode(
                            x=alt.X("month:T", title="Month"),
                            y=alt.Y("strength:Q", title="Topic strength"),
                            color=alt.Color("topic_label:N", title="Topic", sort=topic_order),
                            tooltip=[
                                alt.Tooltip("month:T", title="Month"),
                                alt.Tooltip("topic_label:N", title="Topic"),
                                alt.Tooltip("strength:Q", title="Strength", format=".2f"),
                            ],
                        )
                    )
                    st.altair_chart(over_time_chart.properties(height=320), use_container_width=True)


def _render_component_diagnostics(result: dict[str, Any]) -> None:
    component2 = result.get("component2_diagnostics") or {}
    resilience = result.get("resilience_context") or {}

    st.markdown("<h3 class='sbp-subsection'>Component diagnostics (v2)</h3>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Component 2/3: topic diagnostics")
        neg_count = component2.get("negative_review_count")
        st.caption(
            "Negative reviews used by BERTopic: "
            f"{int(neg_count) if isinstance(neg_count, (int, float)) else '—'} "
            "(stars <= 2 AND vader <= -0.05)"
        )

        terminal_topics = component2.get("terminal_topics") or []
        if terminal_topics:
            terminal_df = pd.DataFrame(
                [
                    {
                        "theme": topic.get("theme"),
                        "share": _format_pct(topic.get("share"), 1),
                        "count": topic.get("count"),
                        "recommendation": topic.get("recommendation") or "—",
                    }
                    for topic in terminal_topics
                ]
            )
            st.dataframe(
                terminal_df,
                use_container_width=True,
                hide_index=True,
                column_order=("theme", "share", "count", "recommendation"),
            )
        else:
            st.caption("No terminal-topic signal for this business in Component 2 outputs.")

        gaps = component2.get("topic_recovery_gaps") or []
        st.markdown("##### Recovery divergence by topic")
        if gaps:
            gap_df = pd.DataFrame(
                [
                    {
                        "theme": item.get("theme"),
                        "closed-open gap": _format_pct(item.get("closed_minus_open_share"), 2),
                        "closed share": _format_pct(item.get("closed_after_negative"), 2),
                        "open share": _format_pct(item.get("open_after_negative"), 2),
                    }
                    for item in gaps
                ]
            )
            st.dataframe(
                gap_df,
                use_container_width=True,
                hide_index=True,
                column_order=("theme", "closed-open gap", "closed share", "open share"),
            )
        else:
            st.caption("No topic-level recovery comparison rows matched this business.")

    with right:
        st.markdown("#### Component 4: resilience context")

        city_context = resilience.get("city_context") or {}
        if city_context:
            st.markdown(
                "<div class='sbp-card'>"
                f"<p class='sbp-small'><strong>{escape(str(city_context.get('city') or '—'))}, "
                f"{escape(str(city_context.get('state') or '—'))}</strong></p>"
                f"<p class='sbp-small'>City closure rate: {escape(_format_pct(city_context.get('closure_rate'), 1))}"
                " · "
                f"Percentile: {escape(_format_pct(city_context.get('closure_rate_percentile'), 0))}</p>"
                f"<p class='sbp-small'>Businesses: {escape(str(city_context.get('businesses') or '—'))}"
                " · "
                f"Closed: {escape(str(city_context.get('closed') or '—'))}</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No city-level resilience context available.")

        cuisine_context = resilience.get("cuisine_context") or []
        st.markdown("##### Matched cuisine closure rates")
        if cuisine_context:
            cuisine_df = pd.DataFrame(
                [
                    {
                        "category": item.get("category"),
                        "closure_rate": _format_pct(item.get("closure_rate"), 1),
                        "businesses": item.get("businesses"),
                    }
                    for item in cuisine_context
                ]
            )
            st.dataframe(
                cuisine_df,
                use_container_width=True,
                hide_index=True,
                column_order=("category", "closure_rate", "businesses"),
            )
        else:
            st.caption("No cuisine-category overlap found for this business.")

        checkin_context = resilience.get("checkin_floor_context") or {}
        st.markdown("##### Check-in floor context")
        if checkin_context:
            latest_checkins = _first_of(checkin_context, "latest_checkins")
            latest_checkins_text = "—"
            if isinstance(latest_checkins, (int, float)):
                if pd.notna(latest_checkins):
                    latest_checkins_text = str(int(round(float(latest_checkins))))
            st.markdown(
                "<div class='sbp-card'>"
                f"<p class='sbp-small'>Latest monthly check-ins: "
                f"<strong>{escape(latest_checkins_text)}</strong></p>"
                f"<p class='sbp-small'>Bin: {escape(str(checkin_context.get('bin_label') or '—'))}</p>"
                f"<p class='sbp-small'>Bin closure rate: {escape(_format_pct(checkin_context.get('closure_rate'), 1))}"
                " · "
                f"Delta vs floor bin: {escape(_format_pct(checkin_context.get('closure_rate_delta'), 1))}</p>"
                f"<p class='sbp-small'>Activity floor: {escape(str(checkin_context.get('activity_floor') or '—'))}</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No check-in floor context available.")

        recovery_pattern = resilience.get("recovery_pattern") or {}
        st.markdown("##### Recovery pattern")
        if recovery_pattern:
            st.markdown(
                "<div class='sbp-card'>"
                f"<p class='sbp-small'>Had negative phase: "
                f"<strong>{'yes' if recovery_pattern.get('had_negative_phase') else 'no'}</strong></p>"
                f"<p class='sbp-small'>Recovered pattern: "
                f"<strong>{'yes' if recovery_pattern.get('recovered_pattern') else 'no'}</strong></p>"
                f"<p class='sbp-small'>Sentiment delta: {escape(_format_prob(recovery_pattern.get('sentiment_delta')))}"
                " · "
                f"Check-in delta: {escape(_format_prob(recovery_pattern.get('checkin_delta')))}</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No recovery-pattern row available for this business.")


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

.stFormSubmitButton > button {
  border-radius: 12px !important;
  border: none !important;
  background: linear-gradient(135deg, #0ea5e9, #22c55e) !important;
  color: #0f172a !important;
  font-weight: 700 !important;
  transition: transform 120ms ease, box-shadow 120ms ease;
}

.stFormSubmitButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(14, 165, 233, 0.25);
}

.stButton > button {
  border-radius: 0 !important;
  border: 1px solid var(--line) !important;
  background: #f8fafc !important;
  color: var(--ink) !important;
  font-weight: 600 !important;
  padding: 6px 10px !important;
  transition: border-color 120ms ease, background-color 120ms ease;
}

.stButton > button:hover {
  transform: none;
  box-shadow: none;
  background: #f1f5f9 !important;
  border-color: rgba(14, 165, 233, 0.45) !important;
}

.stButton > button p {
  margin: 0 !important;
}

.sbp-table-head {
  margin-top: 8px;
}

.sbp-table-head-cell {
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 0;
  padding: 8px 10px;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.02em;
  color: var(--muted);
}

.sbp-table-cell {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 0;
  padding: 8px 10px;
  font-size: 14px;
  color: var(--ink);
}

.sbp-table-cell-muted {
  color: var(--muted);
}

.sbp-selected-tag {
  font-weight: 700;
  color: #0b7285;
  font-size: 12px;
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
  border-radius: 0;
  overflow: hidden;
  box-shadow: none;
}

div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
  border: none !important;
}

div[data-testid="stDataFrame"] table {
  border-collapse: collapse !important;
}

div[data-testid="stDataFrame"] th,
div[data-testid="stDataFrame"] td {
  border: 1px solid var(--line) !important;
  text-align: left !important;
  padding: 8px 10px !important;
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


def _render_shell_header(page: str) -> None:
    if page == "Artifact explorer":
        eyebrow = "Artifact score explorer"
        title = "Search scored artifact rows and inspect stored model outputs."
        lede = "Browse precomputed scored businesses by name, location, and risk band."
        chip_value = "Artifact browse -> Select -> Inspect"
        chip_body = "Uses scored artifact outputs only. No live inference on this page."
    else:
        eyebrow = "Live scoring workflow"
        title = "Find a restaurant and score closure risk live from Yelp source data."
        lede = "Search -> Select -> Live score with configurable 12-month window quotas."
        chip_value = "Live search -> Live score"
        chip_body = "Runs sentiment + feature pipeline + GRU locally on this machine."

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
      <p class="sbp-eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <p class="sbp-lede">{lede}</p>
    </div>
    <div class="sbp-chip">
      <p class="sbp-chip-label">Runtime</p>
      <p class="sbp-chip-value">{chip_value}</p>
      <p>{chip_body}</p>
    </div>
  </div>
</div>
""".format(
            eyebrow=escape(eyebrow),
            title=escape(title),
            lede=escape(lede),
            chip_value=escape(chip_value),
            chip_body=escape(chip_body),
        ),
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _get_service() -> SmallBizPulseService:
    return SmallBizPulseService(Settings.from_env())


def _render_health_sidebar(service: SmallBizPulseService) -> str:
    checks = service.health()
    live_ready = bool(checks.get("live_scoring_ready") or checks.get("live_fallback_ready"))
    pill_cls = "sbp-pill sbp-pill-good" if live_ready else "sbp-pill sbp-pill-warn"
    mode = "Live scoring ready" if live_ready else "Live scoring limited"
    dependency_status = checks.get("dependency_status") or {}
    page = "Live scoring"
    with st.sidebar:
        page = st.radio(
            "Page",
            ["Live scoring", "Artifact explorer"],
            index=0,
            key="sbp_page_selector",
        )
        st.markdown("### Runtime")
        st.markdown(
            f"<span class='{pill_cls}'>{escape(mode)}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("### Dependency indicators")
        indicator_labels = [
            ("tensorflow_package", "TensorFlow package"),
            ("tensorflow_runtime", "TensorFlow runtime"),
            ("vader_sentiment_package", "vaderSentiment package"),
            ("nltk_package", "NLTK package"),
            ("v2_runtime_ready", "V2 runtime ready"),
            ("v2_gru_model_file", "V2 GRU model file"),
            ("yelp_business_file", "Yelp business file"),
            ("yelp_review_file", "Yelp review file"),
            ("yelp_tip_file", "Yelp tip file"),
            ("yelp_checkin_file", "Yelp check-in file"),
            ("v2_bundle_dir", "V2 bundle dir"),
            ("component2_topics_table", "Component 2 topic table"),
            ("component3_recommendations_table", "Component 3 recommendation table"),
            ("component4_city_table", "Component 4 city table"),
            ("component4_cuisine_table", "Component 4 cuisine table"),
            ("component4_checkin_table", "Component 4 check-in table"),
            ("component4_recovery_table", "Component 4 recovery table"),
        ]
        indicators_df = pd.DataFrame(
            [
                {
                    "dependency": label,
                    "status": "ok" if bool(dependency_status.get(key)) else "missing",
                }
                for key, label in indicator_labels
            ]
        )
        st.dataframe(indicators_df, use_container_width=True, hide_index=True)
        with st.expander("Health checks"):
            st.json(checks)
    return page


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

    data_quality = result.get("data_quality") or {}
    if data_quality.get("has_mismatch"):
        expected_reviews = data_quality.get("expected_reviews")
        observed_reviews = data_quality.get("observed_reviews")
        coverage_ratio = data_quality.get("coverage_ratio")
        coverage_text = _format_pct(coverage_ratio, 1) if coverage_ratio is not None else "—"
        st.warning(
            "Live data coverage mismatch: "
            f"observed reviews={observed_reviews if observed_reviews is not None else '—'} vs "
            f"expected metadata reviews={expected_reviews if expected_reviews is not None else '—'} "
            f"(coverage={coverage_text}). "
            "Scores and diagnostics are computed from observed live reviews only."
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
    _render_component_diagnostics(result)

    st.markdown("<h3 class='sbp-subsection'>Evidence reviews</h3>", unsafe_allow_html=True)
    evidence = result.get("evidence_reviews") or []
    if evidence:
        rows: list[dict[str, Any]] = []
        for review in evidence:
            text = _first_of(review, "snippet", "text", "review_text", "body") or ""
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
                    "review_text": str(text),
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


def _render_live_scoring_page(service: SmallBizPulseService) -> None:
    if "live_candidates" not in st.session_state:
        st.session_state["live_candidates"] = []
    if "live_score_result" not in st.session_state:
        st.session_state["live_score_result"] = None
    if "live_selected_business_id" not in st.session_state:
        st.session_state["live_selected_business_id"] = None
    if "live_min_active_months" not in st.session_state:
        st.session_state["live_min_active_months"] = int(service.DEFAULT_MIN_ACTIVE_MONTHS)
    if "live_min_reviews_in_window" not in st.session_state:
        st.session_state["live_min_reviews_in_window"] = int(service.DEFAULT_MIN_REVIEWS_IN_WINDOW)

    st.markdown(
        """
<div class="sbp-step">
  <p class="sbp-eyebrow">Step 1</p>
  <h2>Search businesses (live source)</h2>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("live_search_form"):
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            name = st.text_input("Company name", placeholder="e.g., Five Guys")
        with col2:
            city = st.text_input("City (optional)")
        with col3:
            state = st.text_input("State (optional)", max_chars=2)
        include_unscorable = st.checkbox("Include unscorable matches", value=False)
        quota_col1, quota_col2 = st.columns(2)
        with quota_col1:
            min_active_months = st.slider(
                "Min active months in 12-month window",
                min_value=0,
                max_value=12,
                value=int(st.session_state["live_min_active_months"]),
                key="live_min_active_months",
            )
        with quota_col2:
            min_reviews_in_window = st.slider(
                "Min reviews in 12-month window",
                min_value=0,
                max_value=120,
                value=int(st.session_state["live_min_reviews_in_window"]),
                key="live_min_reviews_in_window",
            )
        submitted = st.form_submit_button("Search live")

    if submitted:
        if not name.strip():
            st.warning("Enter a company name.")
        else:
            try:
                with st.spinner("Searching live businesses..."):
                    st.session_state["live_candidates"] = service.search_businesses(
                        name=name.strip(),
                        city=city.strip() or None,
                        state=state.strip() or None,
                        limit=10,
                        scorable_only=not include_unscorable,
                        min_active_months=min_active_months,
                        min_reviews_in_window=min_reviews_in_window,
                        live_only=True,
                    )
                st.session_state["live_score_result"] = None
                st.session_state["live_selected_business_id"] = None
            except ServiceError as exc:
                st.error(f"Search failed: {exc}")

    candidates = st.session_state.get("live_candidates", [])
    if candidates:
        st.markdown("<h3 class='sbp-subsection'>Candidates</h3>", unsafe_allow_html=True)
        st.caption("Select one row in the table, then click `Score selected`.")
        table_rows = [
            {
                "name": str(candidate.get("name") or "Unknown business"),
                "location": _format_location(candidate.get("city"), candidate.get("state")),
                "status": str(candidate.get("status") or "Unknown"),
                "reviews": str(candidate.get("review_count") if candidate.get("review_count") is not None else "—"),
                "last_review_month": str(candidate.get("last_review_month") or "—"),
                "business_id": str(candidate.get("business_id") or ""),
            }
            for candidate in candidates
        ]
        selected_business_id = _render_search_table_with_action(
            table_rows,
            key_prefix="live_candidates",
            action_label="Score selected",
        )
        if selected_business_id:
            st.session_state["live_selected_business_id"] = selected_business_id
            try:
                with st.spinner("Running live scoring..."):
                    result = service.score_business_live(
                        selected_business_id,
                        min_active_months=int(st.session_state["live_min_active_months"]),
                        min_reviews_in_window=int(st.session_state["live_min_reviews_in_window"]),
                    )
                    st.session_state["live_score_result"] = result
            except ServiceError as exc:
                st.error(f"Scoring failed: {exc}")
    else:
        st.markdown(
            "<div class='sbp-card'><p class='sbp-empty'>No candidates yet. Search for a business to begin.</p></div>",
            unsafe_allow_html=True,
        )

    result = st.session_state.get("live_score_result")
    if isinstance(result, dict):
        _render_score(result)


def _render_artifact_explorer_page(service: SmallBizPulseService) -> None:
    if "artifact_candidates" not in st.session_state:
        st.session_state["artifact_candidates"] = []
    if "artifact_score_result" not in st.session_state:
        st.session_state["artifact_score_result"] = None
    if "artifact_selected_business_id" not in st.session_state:
        st.session_state["artifact_selected_business_id"] = None

    st.markdown(
        """
<div class="sbp-step">
  <p class="sbp-eyebrow">Artifact Explorer</p>
  <h2>Search scored artifacts</h2>
</div>
""",
        unsafe_allow_html=True,
    )

    risk_bands = ["Any"] + service.artifact_risk_bands()
    with st.form("artifact_search_form"):
        col1, col2, col3, col4 = st.columns([3, 2, 1, 2])
        with col1:
            name = st.text_input("Company name (optional)", placeholder="e.g., Five Guys")
        with col2:
            city = st.text_input("City (optional)", key="artifact_city")
        with col3:
            state = st.text_input("State (optional)", max_chars=2, key="artifact_state")
        with col4:
            risk_band = st.selectbox("Risk band", options=risk_bands, index=0)
        limit = st.slider("Max rows", min_value=10, max_value=200, value=25, step=5)
        submitted = st.form_submit_button("Search artifacts")

    if submitted:
        try:
            with st.spinner("Searching scored artifacts..."):
                st.session_state["artifact_candidates"] = service.search_scored_artifacts(
                    name=name.strip(),
                    city=city.strip() or None,
                    state=state.strip() or None,
                    risk_bucket=None if risk_band == "Any" else risk_band,
                    limit=limit,
                )
            st.session_state["artifact_score_result"] = None
            st.session_state["artifact_selected_business_id"] = None
        except ServiceError as exc:
            st.error(f"Artifact search failed: {exc}")

    candidates = st.session_state.get("artifact_candidates", [])
    if candidates:
        st.markdown("<h3 class='sbp-subsection'>Scored artifact rows</h3>", unsafe_allow_html=True)
        st.caption("Select one row in the table, then click `View selected`.")
        table_rows = [
            {
                "name": str(candidate.get("name") or "Unknown business"),
                "location": _format_location(candidate.get("city"), candidate.get("state")),
                "status": str(candidate.get("status") or "Unknown"),
                "reviews": str(candidate.get("review_count") if candidate.get("review_count") is not None else "—"),
                "last_review_month": str(candidate.get("last_review_month") or "—"),
                "risk_score": _format_prob(candidate.get("risk_score")),
                "risk_band": str(candidate.get("risk_bucket") or "—"),
                "business_id": str(candidate.get("business_id") or ""),
            }
            for candidate in candidates
        ]
        selected_business_id = _render_search_table_with_action(
            table_rows,
            key_prefix="artifact_candidates",
            action_label="View selected",
        )
        if selected_business_id:
            st.session_state["artifact_selected_business_id"] = selected_business_id
            try:
                with st.spinner("Loading artifact score..."):
                    result = service.score_business_artifact(selected_business_id)
                    st.session_state["artifact_score_result"] = result
            except ServiceError as exc:
                st.error(f"Artifact score load failed: {exc}")
    else:
        st.markdown(
            "<div class='sbp-card'><p class='sbp-empty'>No artifact rows yet. Search to load scored businesses.</p></div>",
            unsafe_allow_html=True,
        )

    result = st.session_state.get("artifact_score_result")
    if isinstance(result, dict):
        _render_score(result)


def main() -> None:
    st.set_page_config(
        page_title="SmallBizPulse Dashboard",
        page_icon="📊",
        layout="wide",
    )
    _inject_styles()

    try:
        service = _get_service()
    except Exception as exc:
        st.error(f"Service failed to initialize: {exc}")
        st.stop()

    page = _render_health_sidebar(service)
    _render_shell_header(page)

    if page == "Artifact explorer":
        _render_artifact_explorer_page(service)
    else:
        _render_live_scoring_page(service)


if __name__ == "__main__":
    main()
