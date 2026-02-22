from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResilienceArtifacts:
    """Persisted outputs for Component 4 resilience/vulnerability analysis."""

    output_dir: Path
    city_rates_path: Path
    cuisine_rates_path: Path
    checkin_floor_path: Path
    recovery_patterns_path: Path



def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return float(values.mean()) if len(values) else 0.0
    return float((values * weights).sum() / total)



def city_level_closure_rates(business_df: pd.DataFrame) -> pd.DataFrame:
    """Compute closure prevalence by city/state."""
    required = {"city", "state", "status"}
    missing = sorted(required - set(business_df.columns))
    if missing:
        raise ValueError(f"business_df missing required columns: {missing}")

    frame = business_df.copy()
    frame["is_closed"] = frame["status"].astype(str).eq("Closed").astype(int)

    result = (
        frame.groupby(["city", "state"], as_index=False)
        .agg(
            businesses=("status", "size"),
            closed=("is_closed", "sum"),
        )
        .sort_values("businesses", ascending=False)
        .reset_index(drop=True)
    )
    result["closure_rate"] = result["closed"] / result["businesses"].clip(lower=1)
    return result.sort_values("closure_rate", ascending=False).reset_index(drop=True)



def cuisine_level_closure_rates(business_df: pd.DataFrame) -> pd.DataFrame:
    """Compute closure prevalence by cuisine/category token."""
    required = {"business_id", "status", "categories"}
    missing = sorted(required - set(business_df.columns))
    if missing:
        raise ValueError(f"business_df missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for row in business_df.itertuples(index=False):
        categories = str(getattr(row, "categories", "") or "")
        tokens = [token.strip() for token in categories.split(",") if token.strip()]
        is_closed = int(str(getattr(row, "status", "")) == "Closed")
        business_id = str(getattr(row, "business_id"))

        for token in tokens:
            rows.append(
                {
                    "business_id": business_id,
                    "category": token,
                    "is_closed": is_closed,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["category", "businesses", "closed", "closure_rate"])

    frame = pd.DataFrame(rows).drop_duplicates(["business_id", "category"])

    result = (
        frame.groupby("category", as_index=False)
        .agg(
            businesses=("business_id", "nunique"),
            closed=("is_closed", "sum"),
        )
        .sort_values("businesses", ascending=False)
        .reset_index(drop=True)
    )
    result["closure_rate"] = result["closed"] / result["businesses"].clip(lower=1)
    return result.sort_values("closure_rate", ascending=False).reset_index(drop=True)



def checkin_floor_analysis(snapshot_features: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """
    Estimate closure-rate jump regions across check-in activity bins.

    The resulting "activity_floor" is the upper edge of the bucket where
    closure-rate increase is strongest relative to the previous bucket.
    """
    required = {"avg_monthly_checkins", "label_closed"}
    missing = sorted(required - set(snapshot_features.columns))
    if missing:
        raise ValueError(f"snapshot_features missing required columns: {missing}")

    frame = snapshot_features.copy()
    frame["avg_monthly_checkins"] = pd.to_numeric(frame["avg_monthly_checkins"], errors="coerce").fillna(0.0)
    frame["label_closed"] = pd.to_numeric(frame["label_closed"], errors="coerce").fillna(0).astype(int)

    unique_values = frame["avg_monthly_checkins"].nunique()
    if unique_values < 2:
        return pd.DataFrame(
            [
                {
                    "bin_label": "all",
                    "bin_left": float(frame["avg_monthly_checkins"].min()),
                    "bin_right": float(frame["avg_monthly_checkins"].max()),
                    "businesses": int(len(frame)),
                    "closure_rate": float(frame["label_closed"].mean()),
                    "closure_rate_delta": 0.0,
                    "activity_floor": float(frame["avg_monthly_checkins"].max()),
                }
            ]
        )

    q = min(max(2, int(n_bins)), int(unique_values))
    frame["checkin_bin"] = pd.qcut(frame["avg_monthly_checkins"], q=q, duplicates="drop")

    grouped = (
        frame.groupby("checkin_bin", as_index=False, observed=False)
        .agg(
            businesses=("label_closed", "size"),
            closure_rate=("label_closed", "mean"),
        )
        .sort_values("checkin_bin")
        .reset_index(drop=True)
    )

    grouped["closure_rate_delta"] = grouped["closure_rate"].diff().fillna(0.0)

    max_jump_idx = int(grouped["closure_rate_delta"].idxmax()) if len(grouped) else 0
    activity_floor = float(grouped.loc[max_jump_idx, "checkin_bin"].right) if len(grouped) else 0.0

    grouped["bin_label"] = grouped["checkin_bin"].astype(str)
    grouped["bin_left"] = grouped["checkin_bin"].apply(lambda interval: float(interval.left))
    grouped["bin_right"] = grouped["checkin_bin"].apply(lambda interval: float(interval.right))
    grouped["activity_floor"] = activity_floor

    return grouped[
        [
            "bin_label",
            "bin_left",
            "bin_right",
            "businesses",
            "closure_rate",
            "closure_rate_delta",
            "activity_floor",
        ]
    ]



def recovery_pattern_analysis(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Identify businesses that experienced a negative phase and whether they recovered.

    Recovery rule (open businesses):
    - hit a quarterly sentiment low <= -0.05
    - final quarter sentiment improves by >= 0.15 from trough
    - final quarter check-ins >= trough quarter check-ins
    """
    required = {"business_id", "status", "month", "vader_mean", "review_count", "checkin_count"}
    missing = sorted(required - set(monthly_panel.columns))
    if missing:
        raise ValueError(f"monthly_panel missing required columns: {missing}")

    frame = monthly_panel.copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
    frame = frame.dropna(subset=["month"]).copy()

    frame["quarter"] = frame["month"].dt.to_period("Q")

    quarter = (
        frame.groupby(["business_id", "status", "quarter"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "quarter_sentiment": _weighted_average(g["vader_mean"], g["review_count"]),
                    "quarter_checkins": float(pd.to_numeric(g["checkin_count"], errors="coerce").fillna(0.0).sum()),
                }
            )
        )
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for business_id, group in quarter.groupby("business_id"):
        group = group.sort_values("quarter").reset_index(drop=True)
        status = str(group["status"].iloc[-1])

        sent_values = pd.to_numeric(group["quarter_sentiment"], errors="coerce").fillna(0.0)
        checkins = pd.to_numeric(group["quarter_checkins"], errors="coerce").fillna(0.0)

        trough_idx = int(sent_values.idxmin())
        trough_sent = float(sent_values.iloc[trough_idx])
        trough_checkins = float(checkins.iloc[trough_idx])

        final_sent = float(sent_values.iloc[-1])
        final_checkins = float(checkins.iloc[-1])

        had_negative_phase = bool(trough_sent <= -0.05)
        recovered = bool(
            status == "Open"
            and had_negative_phase
            and (final_sent - trough_sent) >= 0.15
            and final_checkins >= trough_checkins
        )

        rows.append(
            {
                "business_id": str(business_id),
                "status": status,
                "had_negative_phase": int(had_negative_phase),
                "recovered_pattern": int(recovered),
                "trough_sentiment": trough_sent,
                "final_sentiment": final_sent,
                "sentiment_delta": float(final_sent - trough_sent),
                "trough_checkins": trough_checkins,
                "final_checkins": final_checkins,
                "checkin_delta": float(final_checkins - trough_checkins),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["had_negative_phase", "recovered_pattern", "sentiment_delta"],
        ascending=[False, False, False],
    ).reset_index(drop=True)



def build_resilience_artifacts(
    *,
    business_df: pd.DataFrame,
    snapshot_features: pd.DataFrame,
    monthly_panel: pd.DataFrame,
    output_dir: Path | str,
) -> ResilienceArtifacts:
    """Build and persist Component 4 analysis tables."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    city_df = city_level_closure_rates(business_df)
    cuisine_df = cuisine_level_closure_rates(business_df)
    floor_df = checkin_floor_analysis(snapshot_features)
    recovery_df = recovery_pattern_analysis(monthly_panel)

    city_rates_path = out_dir / "city_closure_rates.csv"
    cuisine_rates_path = out_dir / "cuisine_closure_rates.csv"
    checkin_floor_path = out_dir / "checkin_floor_analysis.csv"
    recovery_patterns_path = out_dir / "recovery_patterns.csv"

    city_df.to_csv(city_rates_path, index=False)
    cuisine_df.to_csv(cuisine_rates_path, index=False)
    floor_df.to_csv(checkin_floor_path, index=False)
    recovery_df.to_csv(recovery_patterns_path, index=False)

    return ResilienceArtifacts(
        output_dir=out_dir,
        city_rates_path=city_rates_path,
        cuisine_rates_path=cuisine_rates_path,
        checkin_floor_path=checkin_floor_path,
        recovery_patterns_path=recovery_patterns_path,
    )
