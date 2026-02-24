from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class YelpDatasetPaths:
    """Filesystem paths for the Yelp source tables."""

    root: Path

    @property
    def business(self) -> Path:
        return self.root / "yelp_academic_dataset_business.json"

    @property
    def review(self) -> Path:
        return self.root / "yelp_academic_dataset_review.json"

    @property
    def tip(self) -> Path:
        return self.root / "yelp_academic_dataset_tip.json"

    @property
    def checkin(self) -> Path:
        return self.root / "yelp_academic_dataset_checkin.json"

    @property
    def user(self) -> Path:
        return self.root / "yelp_academic_dataset_user.json"

    def required_paths(self) -> list[Path]:
        return [self.business, self.review, self.tip, self.checkin]


@dataclass(frozen=True)
class YelpTables:
    """Raw Yelp tables loaded from disk."""

    business: pd.DataFrame
    review: pd.DataFrame
    tip: pd.DataFrame
    checkin: pd.DataFrame


@dataclass(frozen=True)
class RestaurantTables:
    """Restaurant-scoped Yelp tables used by model building."""

    business: pd.DataFrame
    review: pd.DataFrame
    tip: pd.DataFrame
    checkin: pd.DataFrame


class DatasetLoadError(RuntimeError):
    """Raised when source datasets are missing or malformed."""


def _read_first_non_whitespace_char(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1)
            if not chunk:
                return ""
            if not chunk.isspace():
                return chunk


def load_json_auto(path: Path) -> pd.DataFrame:
    """
    Load either JSON array files or newline-delimited JSON files.

    The local Yelp exports are array-style JSON. We support both formats so the
    same code works across source variants.
    """
    if not path.exists():
        raise DatasetLoadError(f"missing dataset file: {path}")

    first_char = _read_first_non_whitespace_char(path)
    if first_char == "[":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            nested = payload.get("data")
            if isinstance(nested, list):
                return pd.DataFrame(nested)
        raise DatasetLoadError(f"unsupported JSON array payload shape in {path}")

    return pd.read_json(path, lines=True)


def validate_required_columns(frame: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DatasetLoadError(f"{frame_name} missing required columns: {missing}")


def load_yelp_tables(paths: YelpDatasetPaths) -> YelpTables:
    """Load the minimum Yelp tables needed by the v2 modeling stack."""
    for required in paths.required_paths():
        if not required.exists():
            raise DatasetLoadError(f"missing required file: {required}")

    business_df = load_json_auto(paths.business)
    review_df = load_json_auto(paths.review)
    tip_df = load_json_auto(paths.tip)
    checkin_df = load_json_auto(paths.checkin)

    validate_required_columns(
        business_df,
        ["business_id", "name", "city", "state", "is_open", "categories", "stars", "review_count"],
        "business",
    )
    validate_required_columns(review_df, ["review_id", "business_id", "stars", "text", "date"], "review")
    validate_required_columns(tip_df, ["business_id", "text", "date"], "tip")
    validate_required_columns(checkin_df, ["business_id", "date"], "checkin")

    return YelpTables(
        business=business_df,
        review=review_df,
        tip=tip_df,
        checkin=checkin_df,
    )


def filter_restaurants(tables: YelpTables) -> RestaurantTables:
    """Restrict all tables to businesses categorized as restaurants."""
    business_df = tables.business.copy()
    business_df["categories"] = business_df["categories"].fillna("")

    restaurant_df = business_df[
        business_df["categories"].str.contains("Restaurants", case=False, na=False)
    ].copy()

    restaurant_ids = set(restaurant_df["business_id"].astype(str).tolist())

    review_df = tables.review[tables.review["business_id"].isin(restaurant_ids)].copy()
    tip_df = tables.tip[tables.tip["business_id"].isin(restaurant_ids)].copy()
    checkin_df = tables.checkin[tables.checkin["business_id"].isin(restaurant_ids)].copy()

    review_df["date"] = pd.to_datetime(review_df["date"], errors="coerce")
    review_df = review_df.dropna(subset=["date"]).copy()

    tip_df["date"] = pd.to_datetime(tip_df["date"], errors="coerce")
    tip_df = tip_df.dropna(subset=["date"]).copy()

    status_map = restaurant_df.set_index("business_id")["is_open"].to_dict()
    review_df["status"] = review_df["business_id"].map(status_map).map({1: "Open", 0: "Closed"})

    restaurant_df["status"] = restaurant_df["is_open"].map({1: "Open", 0: "Closed"})

    return RestaurantTables(
        business=restaurant_df.reset_index(drop=True),
        review=review_df.reset_index(drop=True),
        tip=tip_df.reset_index(drop=True),
        checkin=checkin_df.reset_index(drop=True),
    )


def load_restaurant_tables(data_root: Path | str) -> RestaurantTables:
    """Convenience wrapper used by the orchestrator."""
    paths = YelpDatasetPaths(root=Path(data_root))
    raw_tables = load_yelp_tables(paths)
    return filter_restaurants(raw_tables)


def summarize_tables(tables: RestaurantTables) -> dict[str, Any]:
    """Basic row-count diagnostics for run logs."""
    return {
        "business_rows": int(len(tables.business)),
        "review_rows": int(len(tables.review)),
        "tip_rows": int(len(tables.tip)),
        "checkin_rows": int(len(tables.checkin)),
        "open_businesses": int((tables.business["status"] == "Open").sum()),
        "closed_businesses": int((tables.business["status"] == "Closed").sum()),
    }
