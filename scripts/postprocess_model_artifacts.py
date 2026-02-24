#!/usr/bin/env python3
"""Build post-GRU artifact tables used by the dashboard.

This script codifies the notebook's post-processing merge path:
1) `gru_business_triage.csv` (risk outputs)
2) optional business metadata (Yelp business JSON/JSONL or existing A-table)
3) problem keywords + recommendations outputs

Outputs:
- `A_closure_risk_table.csv`
- `final_closure_risk_problems_recommendations.csv`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


RISK_BASE_COLUMNS = [
    "business_id",
    "status",
    "risk_score",
    "risk_bucket",
    "y_business",
    "p_recent_max",
    "p_max",
    "p_last",
    "p_mean",
    "n_windows",
    "end_month_last",
    "p_last3_max",
]

BUSINESS_META_COLUMNS = [
    "name",
    "city",
    "state",
    "stars",
    "review_count",
    "categories",
]

A_OUTPUT_COLUMNS = [
    "business_id",
    "name",
    "city",
    "state",
    "status",
    "risk_score",
    "risk_bucket",
    "y_business",
    "p_recent_max",
    "p_max",
    "p_last",
    "p_mean",
    "n_windows",
    "end_month_last",
    "p_last3_max",
    "stars",
    "review_count",
    "categories",
]

FINAL_COLUMNS = [
    "business_id",
    "status",
    "risk_score",
    "risk_bucket",
    "problem_keywords",
    "recommendations_top3",
    "themes_top3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SmallBizPulse post-GRU artifact CSVs.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("models/artifacts"),
        help="Folder holding GRU triage and post-processing artifacts.",
    )
    parser.add_argument(
        "--triage-csv",
        type=Path,
        default=None,
        help="Path to gru_business_triage.csv. Defaults to <artifact-root>/gru_business_triage.csv",
    )
    parser.add_argument(
        "--business-json",
        type=Path,
        default=None,
        help=(
            "Optional Yelp business JSON/JSONL file used to add name/city/state/stars/review_count/categories."
        ),
    )
    parser.add_argument(
        "--problem-csv",
        type=Path,
        default=None,
        help="Path to problem keyword table (defaults to <artifact-root>/B_business_problem_terms.csv).",
    )
    parser.add_argument(
        "--recommendations-csv",
        type=Path,
        default=None,
        help="Path to recommendations table (defaults to <artifact-root>/biz_recommendations.csv).",
    )
    parser.add_argument(
        "--a-out",
        type=Path,
        default=None,
        help="Path for A_closure_risk_table.csv output (defaults to <artifact-root>/A_closure_risk_table.csv).",
    )
    parser.add_argument(
        "--final-out",
        type=Path,
        default=None,
        help=(
            "Path for final_closure_risk_problems_recommendations.csv output "
            "(defaults to <artifact-root>/final_closure_risk_problems_recommendations.csv)."
        ),
    )
    return parser.parse_args()


def load_json_auto(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        first_char = handle.read(1)
        handle.seek(0)
        if first_char == "[":
            return pd.DataFrame(json.load(handle))
        return pd.read_json(handle, lines=True)


def _path_or_default(path: Path | None, default: Path) -> Path:
    return path if path is not None else default


def _normalize_business_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["business_id"] = out["business_id"].astype(str).str.strip()
    out = out[out["business_id"] != ""]
    return out


def _validate_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def load_business_metadata_from_json(path: Path) -> pd.DataFrame:
    business_df = load_json_auto(path)
    _validate_columns(business_df, ["business_id"], "business_json")

    business_df = _normalize_business_id(business_df)
    out = pd.DataFrame({"business_id": business_df["business_id"]})

    for column in BUSINESS_META_COLUMNS:
        if column in business_df.columns:
            out[column] = business_df[column]

    if "status" in business_df.columns:
        out["status_meta"] = business_df["status"].astype(str)
    elif "is_open" in business_df.columns:
        mapped = business_df["is_open"].map({1: "Open", 0: "Closed"})
        out["status_meta"] = mapped.where(mapped.notna(), "Unknown")

    out = out.drop_duplicates(subset=["business_id"], keep="first")
    return out


def load_business_metadata_from_a_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["business_id"] + BUSINESS_META_COLUMNS + ["status_meta"])

    a_df = pd.read_csv(path)
    _validate_columns(a_df, ["business_id"], "existing A-table")
    a_df = _normalize_business_id(a_df)

    out = pd.DataFrame({"business_id": a_df["business_id"]})
    if "status" in a_df.columns:
        out["status_meta"] = a_df["status"]
    for column in BUSINESS_META_COLUMNS:
        if column in a_df.columns:
            out[column] = a_df[column]
    return out.drop_duplicates(subset=["business_id"], keep="first")


def build_a_table(
    triage_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    triage = _normalize_business_id(triage_df)
    _validate_columns(triage, ["business_id", "risk_score", "risk_bucket"], "triage")

    merged = triage.merge(metadata_df, on="business_id", how="left")

    if "status" not in merged.columns:
        merged["status"] = pd.NA
    if "status_meta" in merged.columns:
        merged["status"] = merged["status"].fillna(merged["status_meta"])
    merged["status"] = merged["status"].fillna("Unknown")
    if "status_meta" in merged.columns:
        merged = merged.drop(columns=["status_meta"])

    preferred_order = A_OUTPUT_COLUMNS
    for column in preferred_order:
        if column not in merged.columns:
            merged[column] = pd.NA

    merged = merged[preferred_order]
    merged = merged.sort_values(["risk_score", "business_id"], ascending=[False, True]).reset_index(drop=True)
    return merged


def load_problem_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["business_id", "problem_keywords"])

    df = pd.read_csv(path)
    _validate_columns(df, ["business_id", "problem_keywords"], "problem_csv")
    df = _normalize_business_id(df)
    return df[["business_id", "problem_keywords"]].drop_duplicates(subset=["business_id"], keep="first")


def load_recommendation_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["business_id", "recommendations_top3", "themes_top3"])

    df = pd.read_csv(path)
    _validate_columns(df, ["business_id", "recommendations_top3", "themes_top3"], "recommendations_csv")
    df = _normalize_business_id(df)
    return df[["business_id", "recommendations_top3", "themes_top3"]].drop_duplicates(
        subset=["business_id"],
        keep="first",
    )


def build_final_table(
    a_table: pd.DataFrame,
    problem_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
) -> pd.DataFrame:
    core = a_table[["business_id", "status", "risk_score", "risk_bucket"]].copy()
    merged = core.merge(problem_df, on="business_id", how="left")
    merged = merged.merge(recommendation_df, on="business_id", how="left")

    for column in FINAL_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged[FINAL_COLUMNS].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root

    a_existing = artifact_root / "A_closure_risk_table.csv"
    triage_csv = _path_or_default(args.triage_csv, artifact_root / "gru_business_triage.csv")
    problem_csv = _path_or_default(args.problem_csv, artifact_root / "B_business_problem_terms.csv")
    recommendations_csv = _path_or_default(
        args.recommendations_csv,
        artifact_root / "biz_recommendations.csv",
    )
    a_out = _path_or_default(args.a_out, artifact_root / "A_closure_risk_table.csv")
    final_out = _path_or_default(
        args.final_out,
        artifact_root / "final_closure_risk_problems_recommendations.csv",
    )

    if not triage_csv.exists():
        raise FileNotFoundError(f"Missing triage CSV: {triage_csv}")

    triage_df = pd.read_csv(triage_csv)
    triage_df = _normalize_business_id(triage_df)

    if args.business_json is not None:
        metadata_df = load_business_metadata_from_json(args.business_json)
        metadata_source = f"business_json ({args.business_json})"
    else:
        metadata_df = load_business_metadata_from_a_table(a_existing)
        metadata_source = f"existing A-table fallback ({a_existing})"

    a_table = build_a_table(triage_df, metadata_df)

    problem_df = load_problem_table(problem_csv)
    recommendation_df = load_recommendation_table(recommendations_csv)
    final_table = build_final_table(a_table, problem_df, recommendation_df)

    a_out.parent.mkdir(parents=True, exist_ok=True)
    final_out.parent.mkdir(parents=True, exist_ok=True)

    a_table.to_csv(a_out, index=False)
    final_table.to_csv(final_out, index=False)

    print(f"Built A-table rows: {len(a_table)} -> {a_out}")
    print(f"Built final rows: {len(final_table)} -> {final_out}")
    print(f"Metadata source: {metadata_source}")
    print(
        "Coverage: "
        f"problem_keywords={final_table['problem_keywords'].notna().sum()}/{len(final_table)}, "
        f"recommendations_top3={final_table['recommendations_top3'].notna().sum()}/{len(final_table)}, "
        f"themes_top3={final_table['themes_top3'].notna().sum()}/{len(final_table)}",
    )


if __name__ == "__main__":
    main()
