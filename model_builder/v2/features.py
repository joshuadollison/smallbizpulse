from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .io import RestaurantTables
from .sentiment import VaderSentimentScorer, add_vader_compound


@dataclass(frozen=True)
class MonthlyFeatureConfig:
    """Controls monthly signal generation from raw Yelp tables."""

    negative_review_star_threshold: float = 2.0
    negative_vader_threshold: float = -0.05


@dataclass(frozen=True)
class MonthlySignalArtifacts:
    """Intermediate monthly tables used by downstream model builders."""

    review_monthly: pd.DataFrame
    tip_monthly: pd.DataFrame
    checkin_monthly: pd.DataFrame
    monthly_panel: pd.DataFrame


@dataclass(frozen=True)
class SequenceWindowConfig:
    """Controls GRU sequence window construction."""

    seq_len: int = 12
    horizon_months: int = 6
    min_total_reviews_for_sequence: int = 10
    min_active_months: int = 6
    min_reviews_in_window: int = 10
    inactive_months_for_zombie: int = 12


@dataclass(frozen=True)
class SequenceWindows:
    """Windowed sequence dataset for temporal closure-risk modeling."""

    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    feature_columns: list[str]


MONTHLY_BASE_NUMERIC = [
    "review_count",
    "avg_stars",
    "vader_mean",
    "vader_std",
    "vader_neg_share",
    "tip_count",
    "checkin_count",
]


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return float(values.mean()) if len(values) else 0.0
    return float((values * weights).sum() / weight_sum)


def _series_slope(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(values) < 2:
        return 0.0
    x_axis = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x_axis, values, 1)
    return float(slope)


def _parse_checkin_timestamps(checkin_value: str | None) -> pd.Series:
    if not checkin_value:
        return pd.Series([], dtype="datetime64[ns]")

    parts = [part.strip() for part in str(checkin_value).split(",") if part.strip()]
    if not parts:
        return pd.Series([], dtype="datetime64[ns]")

    timestamps = pd.to_datetime(parts, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if pd.Series(timestamps).notna().any():
        return pd.Series(timestamps).dropna().reset_index(drop=True)

    # Fallback for unexpected non-standard timestamp formats.
    fallback = pd.to_datetime(parts, errors="coerce")
    return pd.Series(fallback).dropna().reset_index(drop=True)


def aggregate_monthly_reviews(
    review_df: pd.DataFrame,
    *,
    sentiment_column: str = "vader_compound",
    negative_star_threshold: float,
    negative_vader_threshold: float,
) -> pd.DataFrame:
    """Aggregate review-level metrics to business-month rows."""
    required = {"business_id", "date", "stars", sentiment_column}
    missing = sorted(required - set(review_df.columns))
    if missing:
        raise ValueError(f"review_df missing required columns for monthly aggregation: {missing}")

    frame = review_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["stars"] = pd.to_numeric(frame["stars"], errors="coerce")
    frame[sentiment_column] = pd.to_numeric(frame[sentiment_column], errors="coerce")
    frame = frame.dropna(subset=["stars", sentiment_column]).copy()

    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    frame["negative_review"] = (
        (frame["stars"] <= float(negative_star_threshold))
        & (frame[sentiment_column] <= float(negative_vader_threshold))
    )

    monthly = (
        frame.groupby(["business_id", "month"], as_index=False)
        .agg(
            review_count=("stars", "count"),
            avg_stars=("stars", "mean"),
            vader_mean=(sentiment_column, "mean"),
            vader_std=(sentiment_column, "std"),
            vader_neg_share=("negative_review", "mean"),
        )
        .sort_values(["business_id", "month"])
        .reset_index(drop=True)
    )

    monthly["vader_std"] = pd.to_numeric(monthly["vader_std"], errors="coerce").fillna(0.0)
    return monthly


def aggregate_monthly_tips(tip_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tip volume to business-month rows."""
    required = {"business_id", "date"}
    missing = sorted(required - set(tip_df.columns))
    if missing:
        raise ValueError(f"tip_df missing required columns: {missing}")

    frame = tip_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()

    return (
        frame.groupby(["business_id", "month"], as_index=False)
        .size()
        .rename(columns={"size": "tip_count"})
        .sort_values(["business_id", "month"])
        .reset_index(drop=True)
    )


def aggregate_monthly_checkins(checkin_df: pd.DataFrame) -> pd.DataFrame:
    """Explode check-in timestamp lists and aggregate them monthly."""
    required = {"business_id", "date"}
    missing = sorted(required - set(checkin_df.columns))
    if missing:
        raise ValueError(f"checkin_df missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for row in checkin_df.itertuples(index=False):
        business_id = str(getattr(row, "business_id"))
        timestamps = _parse_checkin_timestamps(getattr(row, "date", None))
        if timestamps.empty:
            continue

        months = timestamps.dt.to_period("M").dt.to_timestamp()
        month_counts = months.value_counts()
        for month, count in month_counts.items():
            rows.append(
                {
                    "business_id": business_id,
                    "month": pd.Timestamp(month),
                    "checkin_count": int(count),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["business_id", "month", "checkin_count"])

    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["business_id", "month"], as_index=False)["checkin_count"]
        .sum()
        .sort_values(["business_id", "month"])
        .reset_index(drop=True)
    )


def _merge_monthly_tables(
    business_df: pd.DataFrame,
    review_monthly: pd.DataFrame,
    tip_monthly: pd.DataFrame,
    checkin_monthly: pd.DataFrame,
) -> pd.DataFrame:
    merged = review_monthly.merge(tip_monthly, on=["business_id", "month"], how="outer")
    merged = merged.merge(checkin_monthly, on=["business_id", "month"], how="outer")

    business_columns = [
        "business_id",
        "name",
        "city",
        "state",
        "status",
        "categories",
        "stars",
        "review_count",
    ]
    business_meta = business_df[business_columns].copy().rename(
        columns={
            "stars": "business_stars",
            "review_count": "business_review_count",
        }
    )

    merged = merged.merge(business_meta, on="business_id", how="left")

    for column in MONTHLY_BASE_NUMERIC:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    merged["month"] = pd.to_datetime(merged["month"], errors="coerce")
    merged = merged.dropna(subset=["month"]).copy()

    merged["status"] = merged["status"].fillna("Open")
    merged["name"] = merged["name"].fillna("Unknown")
    merged["city"] = merged["city"].fillna("Unknown")
    merged["state"] = merged["state"].fillna("Unknown")
    merged["categories"] = merged["categories"].fillna("")
    merged["business_stars"] = pd.to_numeric(merged["business_stars"], errors="coerce").fillna(0.0)
    merged["business_review_count"] = pd.to_numeric(
        merged["business_review_count"], errors="coerce"
    ).fillna(0.0)

    merged = merged.sort_values(["business_id", "month"]).reset_index(drop=True)
    return merged


def build_monthly_signal_panel(
    tables: RestaurantTables,
    *,
    config: MonthlyFeatureConfig | None = None,
    sentiment_scorer: VaderSentimentScorer | None = None,
) -> MonthlySignalArtifacts:
    """Build monthly multi-signal panel required for v2 modeling."""
    cfg = config or MonthlyFeatureConfig()

    scored_reviews = add_vader_compound(
        tables.review,
        text_column="text",
        output_column="vader_compound",
        scorer=sentiment_scorer,
    )

    review_monthly = aggregate_monthly_reviews(
        scored_reviews,
        sentiment_column="vader_compound",
        negative_star_threshold=cfg.negative_review_star_threshold,
        negative_vader_threshold=cfg.negative_vader_threshold,
    )
    tip_monthly = aggregate_monthly_tips(tables.tip)
    checkin_monthly = aggregate_monthly_checkins(tables.checkin)

    panel = _merge_monthly_tables(tables.business, review_monthly, tip_monthly, checkin_monthly)

    return MonthlySignalArtifacts(
        review_monthly=review_monthly,
        tip_monthly=tip_monthly,
        checkin_monthly=checkin_monthly,
        monthly_panel=panel,
    )


def compute_business_snapshot_features(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-business features for classical closure-risk baselines."""
    required = {
        "business_id",
        "status",
        "month",
        "review_count",
        "avg_stars",
        "vader_mean",
        "tip_count",
        "checkin_count",
        "city",
        "state",
        "categories",
        "name",
    }
    missing = sorted(required - set(monthly_panel.columns))
    if missing:
        raise ValueError(f"monthly_panel missing required columns for snapshots: {missing}")

    frame = monthly_panel.copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
    frame = frame.dropna(subset=["month"]).copy()

    snapshots: list[dict[str, Any]] = []

    for business_id, group in frame.groupby("business_id"):
        group = group.sort_values("month").reset_index(drop=True)

        review_counts = pd.to_numeric(group["review_count"], errors="coerce").fillna(0.0)
        checkins = pd.to_numeric(group["checkin_count"], errors="coerce").fillna(0.0)
        tips = pd.to_numeric(group["tip_count"], errors="coerce").fillna(0.0)
        stars = pd.to_numeric(group["avg_stars"], errors="coerce").fillna(0.0)
        sentiments = pd.to_numeric(group["vader_mean"], errors="coerce").fillna(0.0)

        recent_checkins = float(checkins.tail(3).mean()) if len(checkins) else 0.0
        prior_checkins = float(checkins.iloc[-6:-3].mean()) if len(checkins) >= 6 else float(checkins.mean())

        recent_sentiments = sentiments.tail(36)
        quarter_frame = group[["month", "vader_mean", "review_count"]].copy()
        quarter_frame["quarter"] = quarter_frame["month"].dt.to_period("Q")

        quarterly = (
            quarter_frame.groupby("quarter", as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "quarterly_sentiment": _weighted_average(g["vader_mean"], g["review_count"]),
                    }
                )
            )
            .reset_index(drop=True)
        )
        quarter_values = pd.to_numeric(quarterly["quarterly_sentiment"], errors="coerce").dropna().to_numpy()
        decline_three_quarters = False
        if len(quarter_values) >= 4:
            tail = quarter_values[-4:]
            decline_three_quarters = bool(np.all(np.diff(tail) < 0.0))

        snapshot = {
            "business_id": business_id,
            "name": str(group["name"].iloc[-1]),
            "city": str(group["city"].iloc[-1]),
            "state": str(group["state"].iloc[-1]),
            "categories": str(group["categories"].iloc[-1]),
            "status": str(group["status"].iloc[-1]),
            "label_closed": int(str(group["status"].iloc[-1]) == "Closed"),
            "active_months": int((review_counts > 0).sum()),
            "months_observed": int(len(group)),
            "total_reviews": float(review_counts.sum()),
            "total_tips": float(tips.sum()),
            "total_checkins": float(checkins.sum()),
            "avg_monthly_checkins": float(checkins.mean()) if len(checkins) else 0.0,
            "checkin_velocity_3m": float(recent_checkins - prior_checkins),
            "recent_checkins_3m": float(recent_checkins),
            "stars_mean_weighted": _weighted_average(stars, review_counts),
            "sentiment_mean_weighted": _weighted_average(sentiments, review_counts),
            "sentiment_slope_36m": _series_slope(recent_sentiments),
            "sentiment_decline_three_quarters": int(decline_three_quarters),
            "tip_per_review": float(tips.sum() / max(review_counts.sum(), 1.0)),
            "review_volume_std": float(review_counts.std(ddof=0)) if len(review_counts) else 0.0,
            "last_month": pd.Timestamp(group["month"].iloc[-1]),
        }
        snapshots.append(snapshot)

    return pd.DataFrame(snapshots)


def _reindex_to_contiguous_months(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("month").copy()
    month_range = pd.date_range(group["month"].min(), group["month"].max(), freq="MS")

    static_columns = ["business_id", "status", "name", "city", "state", "categories"]
    static_values = {column: group[column].iloc[-1] for column in static_columns if column in group.columns}

    reindexed = group.set_index("month").reindex(month_range).reset_index().rename(columns={"index": "month"})
    for column, value in static_values.items():
        reindexed[column] = reindexed[column].fillna(value)

    for column in MONTHLY_BASE_NUMERIC:
        if column not in reindexed.columns:
            reindexed[column] = 0.0
        reindexed[column] = pd.to_numeric(reindexed[column], errors="coerce").fillna(0.0)

    return reindexed


def _add_trajectory_features(group: pd.DataFrame, *, base_columns: Sequence[str]) -> pd.DataFrame:
    """Create per-business trajectory features used by the GRU model."""
    local = group.sort_values("month").copy()

    values = local[list(base_columns)].astype(float)
    means = values.mean(axis=0)
    stds = values.std(axis=0).replace(0.0, 1.0)

    for column in base_columns:
        local[f"{column}_z"] = (local[column].astype(float) - float(means[column])) / float(stds[column])

    z_columns = [f"{column}_z" for column in base_columns]
    for column in z_columns:
        local[f"{column}_d1"] = local[column].diff(1)
        local[f"{column}_rm3"] = local[column].rolling(3, min_periods=1).mean()
        local[f"{column}_rs3"] = local[column].rolling(3, min_periods=1).std()
        local[f"{column}_rs6"] = local[column].rolling(6, min_periods=1).std()

    month_origin = int(local["month"].iloc[0].year) * 12 + int(local["month"].iloc[0].month)
    month_index = local["month"].dt.year.astype(int) * 12 + local["month"].dt.month.astype(int)
    local["months_since_first"] = (month_index - month_origin).astype(float)

    engineered_columns = [
        column
        for column in local.columns
        if column.endswith("_z")
        or column.endswith("_d1")
        or column.endswith("_rm3")
        or column.endswith("_rs3")
        or column.endswith("_rs6")
    ]

    local[engineered_columns] = local[engineered_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return local


def build_sequence_windows(
    monthly_panel: pd.DataFrame,
    *,
    config: SequenceWindowConfig | None = None,
) -> SequenceWindows:
    """Build GRU-ready sequence windows labeled for near-term closure risk."""
    cfg = config or SequenceWindowConfig()

    required = {
        "business_id",
        "status",
        "month",
        "review_count",
        "avg_stars",
        "vader_mean",
        "vader_std",
        "vader_neg_share",
        "tip_count",
        "checkin_count",
    }
    missing = sorted(required - set(monthly_panel.columns))
    if missing:
        raise ValueError(f"monthly_panel missing required columns for sequence windows: {missing}")

    panel = monthly_panel.copy()
    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")
    panel = panel.dropna(subset=["month"]).copy()

    global_last_month = panel["month"].max()
    base_feature_columns = [
        "checkin_count",
        "review_count",
        "avg_stars",
        "vader_mean",
        "vader_std",
        "vader_neg_share",
        "tip_count",
    ]

    feature_columns: list[str] = []

    business_snapshot = (
        panel.groupby(["business_id", "status"], as_index=False)["month"]
        .max()
        .rename(columns={"month": "last_review_month"})
    )
    zombie_cutoff = global_last_month - pd.DateOffset(months=int(cfg.inactive_months_for_zombie))
    business_snapshot["is_zombie_open"] = (
        business_snapshot["status"].astype(str).eq("Open")
        & (business_snapshot["last_review_month"] <= zombie_cutoff)
    )

    panel = panel.merge(
        business_snapshot[["business_id", "status", "last_review_month", "is_zombie_open"]],
        on=["business_id", "status"],
        how="left",
    )

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for business_id, raw_group in panel.groupby("business_id"):
        group = _reindex_to_contiguous_months(raw_group)
        group = group.sort_values("month").reset_index(drop=True)

        status = str(group["status"].iloc[-1])
        total_reviews = float(group["review_count"].sum())
        is_zombie_open = bool(raw_group["is_zombie_open"].iloc[-1])
        last_review_month = pd.Timestamp(raw_group["last_review_month"].iloc[-1])

        if status == "Open" and is_zombie_open:
            continue

        if total_reviews <= float(cfg.min_total_reviews_for_sequence):
            continue

        group = _add_trajectory_features(group, base_columns=base_feature_columns)
        if not feature_columns:
            feature_columns = [
                column
                for column in group.columns
                if column.endswith("_z")
                or column.endswith("_d1")
                or column.endswith("_rm3")
                or column.endswith("_rs3")
                or column.endswith("_rs6")
            ]
            if "months_since_first" not in feature_columns:
                feature_columns.append("months_since_first")

        closure_month = pd.NaT
        if status == "Closed":
            closure_month = pd.Timestamp(group["month"].iloc[-1])

        for start in range(0, len(group) - cfg.seq_len + 1):
            end = start + cfg.seq_len
            window = group.iloc[start:end].copy()
            window_end = pd.Timestamp(window["month"].iloc[-1])
            horizon_end = window_end + pd.DateOffset(months=int(cfg.horizon_months))

            active_months = int((window["review_count"] > 0).sum())
            if active_months < int(cfg.min_active_months):
                continue

            window_reviews = float(window["review_count"].sum())
            if window_reviews < float(cfg.min_reviews_in_window):
                continue

            if status == "Closed" and pd.notna(closure_month) and window_end >= closure_month:
                continue

            if status == "Open" and horizon_end > global_last_month:
                continue

            if status == "Closed" and pd.notna(closure_month):
                label = int((closure_month > window_end) and (closure_month <= horizon_end))
            else:
                label = 0

            X_rows.append(window[feature_columns].astype(float).to_numpy(dtype=np.float32))
            y_rows.append(label)
            meta_rows.append(
                {
                    "business_id": str(business_id),
                    "status": status,
                    "start_month": pd.Timestamp(window["month"].iloc[0]),
                    "end_month": window_end,
                    "horizon_end": horizon_end,
                    "closure_month": closure_month,
                    "label": int(label),
                    "total_reviews": total_reviews,
                    "window_reviews": window_reviews,
                    "last_review_month": last_review_month,
                }
            )

    if not X_rows:
        return SequenceWindows(
            X=np.empty((0, cfg.seq_len, len(feature_columns)), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            meta=pd.DataFrame(
                columns=[
                    "business_id",
                    "status",
                    "start_month",
                    "end_month",
                    "horizon_end",
                    "closure_month",
                    "label",
                    "total_reviews",
                    "window_reviews",
                    "last_review_month",
                ]
            ),
            feature_columns=feature_columns,
        )

    return SequenceWindows(
        X=np.stack(X_rows, axis=0),
        y=np.asarray(y_rows, dtype=np.int64),
        meta=pd.DataFrame(meta_rows),
        feature_columns=feature_columns,
    )
