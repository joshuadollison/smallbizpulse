from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .settings import Settings

logger = logging.getLogger(__name__)

RECO_FALLBACK_NOTE = (
    "Not enough signal in keywords - review recent negative feedback and tag issues "
    "into service, food, cleanliness, value, or accuracy."
)

NAME_NORMALIZER = re.compile(r"[^a-z0-9]+")


class ServiceError(RuntimeError):
    """Raised when a required service dependency is unavailable."""


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    value = value.casefold().strip()
    value = NAME_NORMALIZER.sub(" ", value)
    return " ".join(value.split())


def as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def as_optional_int(value: Any) -> int | None:
    number = as_optional_float(value)
    if number is None:
        return None
    return int(round(number))


def assign_risk_bucket(score: float | None, bins: list[float], labels: list[str]) -> str | None:
    if score is None or not bins or not labels or len(bins) != len(labels) + 1:
        return None

    if score <= bins[0]:
        return labels[0]
    if score >= bins[-1]:
        return labels[-1]

    for i, label in enumerate(labels):
        left = bins[i]
        right = bins[i + 1]
        if i == 0 and left <= score <= right:
            return label
        if i > 0 and left < score <= right:
            return label
    return labels[-1]


def parse_csv_list(value: str | None) -> list[str]:
    text = as_optional_str(value)
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_recommendations(value: str | None) -> list[str]:
    text = as_optional_str(value)
    if not text:
        return []

    # Handles strings like: "1. foo  2. bar  3. baz"
    parts = [part.strip() for part in re.split(r"\s*\d+\.\s*", text) if part.strip()]
    if parts:
        return parts

    # Fallback when list numbering is missing.
    fallback_parts = [part.strip() for part in re.split(r"\s{2,}|\n+", text) if part.strip()]
    return fallback_parts if fallback_parts else [text]


def load_json_auto(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        first_char = handle.read(1)
        handle.seek(0)
        if first_char == "[":
            data = json.load(handle)
            return pd.DataFrame(data)
        return pd.read_json(handle, lines=True)


@dataclass
class LiveScoringOutcome:
    ok: bool
    payload: dict[str, Any]
    reason: str | None = None


@dataclass
class LiveWorkflowRepresentation:
    """In-memory representation for the latest live inference workflow."""

    business_id: str
    created_at: str
    sentiment_df: pd.DataFrame
    monthly_df: pd.DataFrame
    meta_windows: pd.DataFrame
    probabilities: np.ndarray
    risk_score: float
    risk_bucket: str | None


class SmallBizPulseService:
    REQUIRED_RISK_COLUMNS = {
        "business_id",
        "name",
        "city",
        "state",
        "status",
        "risk_score",
        "risk_bucket",
        "review_count",
        "end_month_last",
    }

    REQUIRED_PROBLEM_COLUMNS = {
        "business_id",
        "problem_keywords",
        "recommendations_top3",
        "themes_top3",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self._tensorflow_available = module_available("tensorflow")
        self._transformers_available = module_available("transformers")
        self._torch_available = module_available("torch")
        self._joblib_available = module_available("joblib")
        self._tensorflow_runtime_available: bool | None = None

        self._risk_bins: list[float] = [0.0, 0.5, 0.65, 0.75, 0.85, 1.0]
        self._risk_labels: list[str] = ["low", "medium", "elevated", "high", "very_high"]
        self._playbook: list[tuple[str, list[str], str]] = []
        self._bertopic_topic_terms: dict[int, list[str]] = {}

        self._scored_df: pd.DataFrame | None = None
        self._scored_by_id: dict[str, dict[str, Any]] = {}

        self._live_business_df: pd.DataFrame | None = None
        self._live_reviews_df: pd.DataFrame | None = None
        self._global_last_month: pd.Timestamp | None = None
        self._live_window_scoreable_by_business: dict[str, bool] = {}

        self._sentiment_model: Any = None
        self._sentiment_tokenizer: Any = None
        self._sentiment_pos_idx: int | None = None
        self._sentiment_neg_idx: int | None = None

        self._gru_model: Any = None
        self._vec_model: Any = None
        self._nmf_model: Any = None

        # Live path keeps intermediate representations in memory only.
        self._last_live_workflow: LiveWorkflowRepresentation | None = None

        self._load_artifacts()

    @property
    def yelp_data_available(self) -> bool:
        return self._required_yelp_files_present()

    def health(self) -> dict[str, Any]:
        yelp_present = self._required_yelp_files_present()
        model_path = self.settings.model_dir / "model_gru.keras"

        live_ready = (
            self.settings.enable_live_fallback
            and self._tensorflow_available
            and self._is_tensorflow_runtime_available()
            and yelp_present
            and model_path.exists()
        )

        return {
            "artifact_root": str(self.settings.artifact_root),
            "artifact_root_exists": self.settings.artifact_root.exists(),
            "model_dir": str(self.settings.model_dir),
            "model_dir_exists": self.settings.model_dir.exists(),
            "sentiment_dir": str(self.settings.sentiment_dir),
            "sentiment_dir_exists": self.settings.sentiment_dir.exists(),
            "yelp_data_dir": str(self.settings.yelp_data_dir) if self.settings.yelp_data_dir else None,
            "yelp_data_available": yelp_present,
            "live_fallback_enabled": self.settings.enable_live_fallback,
            "prefer_live_scoring": self.settings.prefer_live_scoring,
            "tensorflow_available": self._tensorflow_available,
            "tensorflow_runtime_available": self._is_tensorflow_runtime_available(),
            "transformers_available": self._transformers_available,
            "torch_available": self._torch_available,
            "joblib_available": self._joblib_available,
            "gru_inference_mode": "subprocess_worker",
            "gru_worker_timeout_seconds": self.settings.gru_worker_timeout_seconds,
            "gru_worker_conda_env": self.settings.gru_worker_conda_env,
            "gru_worker_python": self.settings.gru_worker_python,
            "live_fallback_ready": live_ready,
            "scored_businesses": int(len(self._scored_df)) if self._scored_df is not None else 0,
        }

    def search_businesses(
        self,
        name: str,
        *,
        city: str | None = None,
        state: str | None = None,
        limit: int = 10,
        scorable_only: bool = True,
    ) -> list[dict[str, Any]]:
        query = normalize_text(name)
        if not query:
            return []
        query_tokens = [token for token in query.split() if token]
        query_token_set = set(query_tokens)

        city_q = normalize_text(city)
        state_q = normalize_text(state)

        candidates_df = self._candidate_df()
        ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        for _, row in candidates_df.iterrows():
            candidate_name = as_optional_str(row.get("name")) or ""
            candidate_city = as_optional_str(row.get("city"))
            candidate_state = as_optional_str(row.get("state"))
            candidate_name_norm = normalize_text(candidate_name)
            candidate_tokens = [token for token in candidate_name_norm.split() if token]
            candidate_token_set = set(candidate_tokens)
            candidate_city_norm = normalize_text(candidate_city)
            candidate_state_norm = normalize_text(candidate_state)

            exact = int(candidate_name_norm == query)
            starts = int(bool(query) and candidate_name_norm.startswith(query))
            contains = int(bool(query) and query in candidate_name_norm)
            fuzzy = SequenceMatcher(None, query, candidate_name_norm).ratio() if candidate_name_norm else 0.0
            token_overlap = len(query_token_set & candidate_token_set)
            token_prefix_overlap = int(
                bool(query_tokens)
                and any(
                    any(candidate_token.startswith(query_token) for candidate_token in candidate_tokens)
                    for query_token in query_tokens
                )
            )
            all_query_tokens_present = int(
                bool(query_tokens)
                and all(
                    any(candidate_token.startswith(query_token) for candidate_token in candidate_tokens)
                    for query_token in query_tokens
                )
            )

            city_bonus = int(bool(city_q) and city_q == candidate_city_norm)
            state_bonus = int(bool(state_q) and state_q == candidate_state_norm)

            city_filter_ok = not city_q or city_q in candidate_city_norm
            state_filter_ok = not state_q or state_q == candidate_state_norm
            if not city_filter_ok or not state_filter_ok:
                continue

            fuzzy_match_ok = fuzzy >= 0.78 and token_prefix_overlap
            if not (exact or starts or contains or all_query_tokens_present or fuzzy_match_ok):
                continue

            payload = {
                "business_id": as_optional_str(row.get("business_id")),
                "name": candidate_name,
                "city": candidate_city,
                "state": candidate_state,
                "status": as_optional_str(row.get("status")) or "Unknown",
                "review_count": as_optional_int(row.get("total_reviews")),
                "last_review_month": self._format_month(row.get("last_review_month")),
                "risk_available": bool(row.get("risk_available", False)),
            }

            rank = (
                exact,
                starts,
                contains,
                all_query_tokens_present,
                token_overlap,
                city_bonus,
                state_bonus,
                round(fuzzy, 6),
                as_optional_int(row.get("total_reviews")) or 0,
                payload["name"] or "",
            )
            if scorable_only and not self._candidate_is_scorable(payload):
                continue
            ranked.append((rank, payload))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[: max(1, min(limit, 50))]]

    def _candidate_is_scorable(self, candidate: dict[str, Any]) -> bool:
        if bool(candidate.get("risk_available", False)):
            return True

        if not self.settings.enable_live_fallback:
            return False
        if not self.settings.yelp_data_dir:
            return False
        if not self._required_yelp_files_present():
            return False

        business_id = as_optional_str(candidate.get("business_id"))
        if not business_id:
            return False
        status = as_optional_str(candidate.get("status")) or "Unknown"
        return self._can_build_live_windows_for_business(business_id, status)

    def _can_build_live_windows_for_business(self, business_id: str, status: str) -> bool:
        cached = self._live_window_scoreable_by_business.get(business_id)
        if cached is not None:
            return cached

        reviews_df = self._load_live_reviews_df()
        if reviews_df is None or reviews_df.empty:
            self._live_window_scoreable_by_business[business_id] = False
            return False

        business_reviews = reviews_df[reviews_df["business_id"] == business_id]
        if business_reviews.empty:
            self._live_window_scoreable_by_business[business_id] = False
            return False

        monthly_df = (
            business_reviews.groupby("month", as_index=False)
            .agg(
                review_count=("review_id", "count"),
                avg_stars=("stars", "mean"),
            )
            .sort_values("month")
            .reset_index(drop=True)
        )
        if monthly_df.empty:
            self._live_window_scoreable_by_business[business_id] = False
            return False

        monthly_df["business_id"] = business_id
        monthly_df["status"] = status
        monthly_df["tx_sent_mean"] = 0.5
        monthly_df["tx_sent_std"] = 0.0
        monthly_df["tx_neg_share"] = 0.5
        monthly_df["tx_pos_share"] = 0.5

        windows = self._build_inference_windows(
            monthly_df[
                [
                    "business_id",
                    "status",
                    "month",
                    "review_count",
                    "avg_stars",
                    "tx_sent_mean",
                    "tx_sent_std",
                    "tx_neg_share",
                    "tx_pos_share",
                ]
            ],
            {
                "business_id": business_id,
                "status": status,
            },
        )
        can_score = windows is not None and len(windows[1]) > 0
        self._live_window_scoreable_by_business[business_id] = can_score
        return can_score

    def score_business(
        self,
        business_id: str,
        *,
        force_live_inference: bool = False,
    ) -> dict[str, Any]:
        business_id = (business_id or "").strip()
        if not business_id:
            return self._not_scored_payload(reason="missing_business_id")

        artifact_row = self._scored_by_id.get(business_id)
        artifact_identity = self._identity_from_row(artifact_row) if artifact_row is not None else None

        if force_live_inference:
            live_outcome = self._score_live_fallback(business_id)
            if live_outcome.ok:
                return live_outcome.payload

            live_identity = None
            if isinstance(live_outcome.payload, dict):
                candidate = live_outcome.payload.get("identity")
                if isinstance(candidate, dict):
                    live_identity = candidate
            identity = live_identity or self._lookup_live_business_identity(business_id) or artifact_identity
            return self._not_scored_payload(
                business_id=business_id,
                identity=identity,
                reason=live_outcome.reason or "not_scored",
                scoring_mode="live_fallback",
            )

        if self.settings.prefer_live_scoring:
            live_outcome = self._score_live_fallback(business_id)
            if live_outcome.ok:
                return live_outcome.payload
            if artifact_row is not None:
                return self._artifact_payload(artifact_row)

            live_identity = None
            if isinstance(live_outcome.payload, dict):
                candidate = live_outcome.payload.get("identity")
                if isinstance(candidate, dict):
                    live_identity = candidate
            identity = live_identity or self._lookup_live_business_identity(business_id) or artifact_identity
            return self._not_scored_payload(
                business_id=business_id,
                identity=identity,
                reason=live_outcome.reason or "not_scored",
                scoring_mode="live_fallback",
            )

        if artifact_row is not None:
            return self._artifact_payload(artifact_row)

        live_outcome = self._score_live_fallback(business_id)
        if live_outcome.ok:
            return live_outcome.payload

        live_identity = None
        if isinstance(live_outcome.payload, dict):
            candidate = live_outcome.payload.get("identity")
            if isinstance(candidate, dict):
                live_identity = candidate
        identity = live_identity or self._lookup_live_business_identity(business_id)
        return self._not_scored_payload(
            business_id=business_id,
            identity=identity,
            reason=live_outcome.reason or "not_scored",
            scoring_mode="live_fallback",
        )

    def _load_artifacts(self) -> None:
        risk_path = self.settings.artifact_root / "A_closure_risk_table.csv"
        problem_path = self.settings.artifact_root / "final_closure_risk_problems_recommendations.csv"
        triage_path = self.settings.artifact_root / "gru_business_triage.csv"
        topic_terms_path = self.settings.artifact_root / "B_topic_terms.csv"
        metadata_path = self.settings.model_dir / "model_metadata.json"

        if not risk_path.exists():
            raise ServiceError(f"Missing artifact: {risk_path}")
        if not problem_path.exists():
            raise ServiceError(f"Missing artifact: {problem_path}")
        if not metadata_path.exists():
            raise ServiceError(f"Missing metadata: {metadata_path}")

        risk_df = pd.read_csv(risk_path)
        missing_risk = sorted(self.REQUIRED_RISK_COLUMNS - set(risk_df.columns))
        if missing_risk:
            raise ServiceError(f"A_closure_risk_table.csv missing columns: {missing_risk}")

        risk_df = risk_df.copy()
        risk_df["business_id"] = risk_df["business_id"].astype(str)
        risk_df = risk_df.rename(
            columns={
                "review_count": "total_reviews",
                "end_month_last": "last_review_month",
            }
        )

        problem_df = pd.read_csv(problem_path)
        missing_problem = sorted(self.REQUIRED_PROBLEM_COLUMNS - set(problem_df.columns))
        if missing_problem:
            raise ServiceError(
                "final_closure_risk_problems_recommendations.csv missing columns: "
                f"{missing_problem}"
            )
        problem_df = problem_df.copy()
        problem_df["business_id"] = problem_df["business_id"].astype(str)

        merged = risk_df.merge(
            problem_df[
                [
                    "business_id",
                    "problem_keywords",
                    "recommendations_top3",
                    "themes_top3",
                ]
            ],
            on="business_id",
            how="left",
        )

        if triage_path.exists():
            triage_df = pd.read_csv(triage_path)
            triage_df = triage_df.copy()
            triage_df["business_id"] = triage_df["business_id"].astype(str)
            for column in [
                "business_id",
                "p_recent_max",
                "p_last3_max",
                "p_last",
                "p_max",
                "p_mean",
                "n_windows",
                "end_month_last",
            ]:
                if column not in triage_df.columns:
                    triage_df[column] = np.nan

            merged = merged.merge(
                triage_df[
                    [
                        "business_id",
                        "p_recent_max",
                        "p_last3_max",
                        "p_last",
                        "p_max",
                        "p_mean",
                        "n_windows",
                        "end_month_last",
                    ]
                ],
                on="business_id",
                how="left",
            )

            merged["last_review_month"] = merged["last_review_month"].fillna(merged["end_month_last"])
            merged = merged.drop(columns=["end_month_last"], errors="ignore")

        self._scored_df = merged
        self._scored_by_id = {
            row["business_id"]: row
            for row in merged.to_dict(orient="records")
            if as_optional_str(row.get("business_id"))
        }

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        risk_bins = metadata.get("risk_bins")
        risk_labels = metadata.get("risk_labels")
        if (
            isinstance(risk_bins, list)
            and isinstance(risk_labels, list)
            and len(risk_bins) == len(risk_labels) + 1
        ):
            self._risk_bins = [float(x) for x in risk_bins]
            self._risk_labels = [str(x) for x in risk_labels]

        playbook_items: list[tuple[str, list[str], str]] = []
        for item in metadata.get("PLAYBOOK", []):
            if not isinstance(item, list) or len(item) < 3:
                continue
            theme = str(item[0]).strip()
            cues = [str(cue).strip().casefold() for cue in item[1] if str(cue).strip()]
            recommendation = str(item[2]).strip()
            if theme and recommendation:
                playbook_items.append((theme, cues, recommendation))
        self._playbook = playbook_items

        self._bertopic_topic_terms = {}
        if topic_terms_path.exists():
            try:
                topic_terms_df = pd.read_csv(topic_terms_path)
                if {"topic_id", "top_terms"} <= set(topic_terms_df.columns):
                    parsed: dict[int, list[str]] = {}
                    for row in topic_terms_df.itertuples(index=False):
                        topic_id = as_optional_int(getattr(row, "topic_id", None))
                        top_terms = as_optional_str(getattr(row, "top_terms", None))
                        if topic_id is None or not top_terms:
                            continue
                        terms = [
                            term.strip().casefold()
                            for term in top_terms.split(",")
                            if term.strip()
                        ]
                        if terms:
                            parsed[topic_id] = terms
                    self._bertopic_topic_terms = parsed
            except Exception as exc:  # pragma: no cover - optional artifact
                logger.warning("Failed to parse B_topic_terms.csv: %s", exc)

    def _candidate_df(self) -> pd.DataFrame:
        assert self._scored_df is not None

        scored = self._scored_df[
            [
                "business_id",
                "name",
                "city",
                "state",
                "status",
                "total_reviews",
                "last_review_month",
            ]
        ].copy()
        scored["risk_available"] = True

        live = self._load_live_business_df()
        if live is None or live.empty:
            return scored

        unscored = live[~live["business_id"].isin(scored["business_id"])].copy()
        merged = pd.concat([scored, unscored], ignore_index=True)
        return merged

    def _load_live_business_df(self) -> pd.DataFrame | None:
        if self._live_business_df is not None:
            return self._live_business_df

        if not self.settings.yelp_data_dir:
            return None

        business_path = self.settings.yelp_data_dir / "yelp_academic_dataset_business.json"
        if not business_path.exists():
            return None

        try:
            business_df = load_json_auto(business_path)
        except Exception as exc:  # pragma: no cover - data-dependent
            logger.warning("Failed loading live business data: %s", exc)
            return None

        required_columns = {"business_id", "name", "city", "state", "categories", "is_open"}
        missing = required_columns - set(business_df.columns)
        if missing:
            logger.warning("Live business data missing columns: %s", sorted(missing))
            return None

        business_df = business_df.copy()
        business_df = business_df[
            business_df["categories"].astype(str).str.contains("Restaurants", case=False, na=False)
        ]
        business_df["business_id"] = business_df["business_id"].astype(str)
        business_df["status"] = business_df["is_open"].map({1: "Open", 0: "Closed"}).fillna("Unknown")
        if "review_count" in business_df.columns:
            business_df["total_reviews"] = pd.to_numeric(business_df["review_count"], errors="coerce")
        else:
            business_df["total_reviews"] = np.nan

        business_df["last_review_month"] = np.nan
        business_df["risk_available"] = False

        self._live_business_df = business_df[
            [
                "business_id",
                "name",
                "city",
                "state",
                "status",
                "total_reviews",
                "last_review_month",
                "risk_available",
            ]
        ].copy()
        return self._live_business_df

    def _lookup_live_business_identity(self, business_id: str) -> dict[str, Any] | None:
        live_df = self._load_live_business_df()
        if live_df is None or live_df.empty:
            return None

        row = live_df[live_df["business_id"] == business_id]
        if row.empty:
            return None

        record = row.iloc[0].to_dict()
        return {
            "business_id": as_optional_str(record.get("business_id")),
            "name": as_optional_str(record.get("name")),
            "city": as_optional_str(record.get("city")),
            "state": as_optional_str(record.get("state")),
            "status": as_optional_str(record.get("status")) or "Unknown",
            "total_reviews": as_optional_int(record.get("total_reviews")),
            "last_review_month": self._format_month(record.get("last_review_month")),
        }

    def _identity_from_row(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "business_id": as_optional_str(row.get("business_id")),
            "name": as_optional_str(row.get("name")),
            "city": as_optional_str(row.get("city")),
            "state": as_optional_str(row.get("state")),
            "status": as_optional_str(row.get("status")) or "Unknown",
            "total_reviews": as_optional_int(row.get("total_reviews")),
            "last_review_month": self._format_month(row.get("last_review_month")),
        }

    def _artifact_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        risk_score = as_optional_float(row.get("risk_score"))
        risk_bucket = as_optional_str(row.get("risk_bucket")) or assign_risk_bucket(
            risk_score, self._risk_bins, self._risk_labels
        )
        status = as_optional_str(row.get("status")) or "Unknown"

        themes = parse_csv_list(row.get("themes_top3"))
        recommendations = parse_recommendations(row.get("recommendations_top3"))
        problem_keywords = as_optional_str(row.get("problem_keywords"))
        recommendation_notes = None

        if not themes or not recommendations:
            themes = ["needs_review"]
            recommendations = [RECO_FALLBACK_NOTE]
            recommendation_notes = RECO_FALLBACK_NOTE

        recent_windows: list[dict[str, Any]] = []
        end_month = self._format_month(row.get("last_review_month"))
        p_last = as_optional_float(row.get("p_last"))
        if end_month is not None and p_last is not None:
            recent_windows.append(
                {
                    "end_month": end_month,
                    "p_closed": p_last,
                }
            )

        chart_data = {
            "ratings_by_month": [],
            "rating_bucket_counts_by_month": [],
            "predicted_close_by_month": [
                {
                    "month": window.get("end_month"),
                    "p_closed": as_optional_float(window.get("p_closed")),
                }
                for window in recent_windows
            ],
            "topics_by_month": [],
            "actual_close_month": end_month if status.casefold() == "closed" else None,
        }

        return {
            "business_id": as_optional_str(row.get("business_id")),
            "name": as_optional_str(row.get("name")),
            "city": as_optional_str(row.get("city")),
            "state": as_optional_str(row.get("state")),
            "status": status,
            "total_reviews": as_optional_int(row.get("total_reviews")),
            "last_review_month": end_month,
            "risk_score": risk_score,
            "risk_bucket": risk_bucket,
            "recent_windows": recent_windows,
            "themes_top3": themes[:3],
            "problem_keywords": problem_keywords,
            "evidence_reviews": [],
            "recommendations_top3": recommendations[:3],
            "recommendation_notes": recommendation_notes,
            "chart_data": chart_data,
            "scoring_mode": "artifact",
            "availability": "scored",
            "not_scored_reason": None,
        }

    def _score_live_fallback(self, business_id: str) -> LiveScoringOutcome:
        if not self.settings.enable_live_fallback:
            return LiveScoringOutcome(False, {}, "live_fallback_disabled")
        if not self.settings.yelp_data_dir:
            return LiveScoringOutcome(False, {}, "missing_yelp_data_dir")
        if not self._required_yelp_files_present():
            return LiveScoringOutcome(False, {}, "yelp_files_missing")
        if not self._tensorflow_available:
            return LiveScoringOutcome(False, {}, "tensorflow_not_installed")
        if not self._is_tensorflow_runtime_available():
            return LiveScoringOutcome(False, {}, "tensorflow_runtime_unavailable")

        identity = self._lookup_live_business_identity(business_id)
        if identity is None:
            return LiveScoringOutcome(False, {}, "business_not_found")

        try:
            reviews_df = self._load_live_reviews_df()
            if reviews_df is None:
                return LiveScoringOutcome(False, {}, "reviews_unavailable")

            business_reviews = reviews_df[reviews_df["business_id"] == business_id].copy()
            if business_reviews.empty:
                return LiveScoringOutcome(False, {}, "no_reviews_for_business")

            business_reviews["date"] = pd.to_datetime(business_reviews["date"], errors="coerce")
            business_reviews = business_reviews.dropna(subset=["date"])
            if business_reviews.empty:
                return LiveScoringOutcome(False, {}, "invalid_review_dates")

            live_identity = dict(identity)
            live_identity["total_reviews"] = int(len(business_reviews))
            live_identity["last_review_month"] = self._format_month(business_reviews["date"].max())

            sentiment_df = self._score_sentiment_for_reviews(business_reviews)
            monthly_df = self._build_monthly_panel(sentiment_df, identity)
            windows = self._build_inference_windows(monthly_df, identity)
            if windows is None:
                expected_reviews = as_optional_int(identity.get("total_reviews"))
                observed_reviews = as_optional_int(live_identity.get("total_reviews"))
                reason = "insufficient_history_for_windows"
                if (
                    expected_reviews is not None
                    and observed_reviews is not None
                    and observed_reviews < expected_reviews
                ):
                    reason = "insufficient_history_for_windows_live_data_mismatch"
                return LiveScoringOutcome(False, {"identity": live_identity}, reason)

            X_windows, meta_windows = windows
            if len(meta_windows) == 0:
                expected_reviews = as_optional_int(identity.get("total_reviews"))
                observed_reviews = as_optional_int(live_identity.get("total_reviews"))
                reason = "insufficient_history_for_windows"
                if (
                    expected_reviews is not None
                    and observed_reviews is not None
                    and observed_reviews < expected_reviews
                ):
                    reason = "insufficient_history_for_windows_live_data_mismatch"
                return LiveScoringOutcome(False, {"identity": live_identity}, reason)

            p_windows = self._predict_gru_probabilities(X_windows)
            risk_score, risk_bucket, recent_windows = self._aggregate_window_risk(meta_windows, p_windows)
            self._store_live_workflow_representation(
                business_id=business_id,
                sentiment_df=sentiment_df,
                monthly_df=monthly_df,
                meta_windows=meta_windows,
                probabilities=p_windows,
                risk_score=risk_score,
                risk_bucket=risk_bucket,
            )
            evidence = self._extract_evidence_reviews(sentiment_df)
            problem_keywords, themes_top3, recommendations_top3, recommendation_notes = (
                self._derive_problems_and_recommendations(sentiment_df, evidence)
            )

            payload = {
                "business_id": identity.get("business_id"),
                "name": identity.get("name"),
                "city": identity.get("city"),
                "state": identity.get("state"),
                "status": identity.get("status") or "Unknown",
                "total_reviews": len(sentiment_df),
                "last_review_month": self._format_month(monthly_df["month"].max()),
                "risk_score": risk_score,
                "risk_bucket": risk_bucket,
                "recent_windows": recent_windows,
                "themes_top3": themes_top3,
                "problem_keywords": problem_keywords,
                "evidence_reviews": evidence,
                "recommendations_top3": recommendations_top3,
                "recommendation_notes": recommendation_notes,
                "chart_data": self._build_live_chart_data(
                    identity=identity,
                    monthly_df=monthly_df,
                    meta_windows=meta_windows,
                    probabilities=p_windows,
                    sentiment_df=sentiment_df,
                ),
                "scoring_mode": "live_fallback",
                "availability": "scored",
                "not_scored_reason": None,
            }
            return LiveScoringOutcome(True, payload)
        except ServiceError as exc:
            return LiveScoringOutcome(False, {}, str(exc))
        except Exception as exc:  # pragma: no cover - data and env dependent
            logger.exception("Live fallback scoring failed")
            return LiveScoringOutcome(False, {}, f"live_fallback_failed:{exc.__class__.__name__}")

    def _load_live_reviews_df(self) -> pd.DataFrame | None:
        if self._live_reviews_df is not None:
            return self._live_reviews_df

        if not self.settings.yelp_data_dir:
            return None

        review_path = self.settings.yelp_data_dir / "yelp_academic_dataset_review.json"
        if not review_path.exists():
            return None

        review_df = load_json_auto(review_path)
        required = {"business_id", "review_id", "date", "stars", "text"}
        missing = required - set(review_df.columns)
        if missing:
            raise ServiceError(f"review dataset missing columns: {sorted(missing)}")

        review_df = review_df.copy()
        review_df["business_id"] = review_df["business_id"].astype(str)
        review_df["date"] = pd.to_datetime(review_df["date"], errors="coerce")
        review_df = review_df.dropna(subset=["date"])
        review_df["month"] = review_df["date"].dt.to_period("M").dt.to_timestamp()
        self._live_reviews_df = review_df[
            ["business_id", "review_id", "date", "month", "stars", "text"]
        ].copy()

        if self._global_last_month is None:
            max_date = self._live_reviews_df["date"].max()
            if pd.notna(max_date):
                self._global_last_month = max_date.to_period("M").to_timestamp()

        return self._live_reviews_df

    def _store_live_workflow_representation(
        self,
        *,
        business_id: str,
        sentiment_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
        meta_windows: pd.DataFrame,
        probabilities: np.ndarray,
        risk_score: float,
        risk_bucket: str | None,
    ) -> None:
        """Keep the latest live scoring intermediate state in memory."""
        self._last_live_workflow = LiveWorkflowRepresentation(
            business_id=business_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            sentiment_df=sentiment_df.copy(),
            monthly_df=monthly_df.copy(),
            meta_windows=meta_windows.copy(),
            probabilities=np.asarray(probabilities, dtype=float).copy(),
            risk_score=float(risk_score),
            risk_bucket=risk_bucket,
        )

    def _score_sentiment_for_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        if not self._transformers_available or not self._torch_available:
            raise ServiceError("transformer_dependencies_missing")
        if not self.settings.sentiment_dir.exists():
            raise ServiceError("sentiment_model_missing")

        if self._sentiment_model is None or self._sentiment_tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._sentiment_tokenizer = AutoTokenizer.from_pretrained(self.settings.sentiment_dir)
            self._sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.sentiment_dir
            )
            self._sentiment_model.eval()

            config = self._sentiment_model.config
            id2label = {
                int(k): str(v)
                for k, v in getattr(config, "id2label", {0: "NEG", 1: "POS"}).items()
            }

            pos_idx = None
            neg_idx = None
            for idx, label in id2label.items():
                label_cf = label.casefold()
                if label_cf in {"pos", "positive", "label_1"}:
                    pos_idx = idx
                if label_cf in {"neg", "negative", "label_0"}:
                    neg_idx = idx

            if pos_idx is None:
                pos_idx = 1 if 1 in id2label else 0
            if neg_idx is None:
                neg_idx = 0 if 0 in id2label else 1

            self._sentiment_pos_idx = pos_idx
            self._sentiment_neg_idx = neg_idx

        import torch

        tokenized_model = self._sentiment_model
        tokenizer = self._sentiment_tokenizer
        assert tokenized_model is not None and tokenizer is not None
        assert self._sentiment_pos_idx is not None and self._sentiment_neg_idx is not None

        texts = reviews_df["text"].fillna("").astype(str).tolist()
        p_pos: list[float] = []
        p_neg: list[float] = []

        batch_size = 32
        for offset in range(0, len(texts), batch_size):
            batch_texts = texts[offset : offset + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = tokenized_model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            p_pos.extend(probs[:, self._sentiment_pos_idx].tolist())
            p_neg.extend(probs[:, self._sentiment_neg_idx].tolist())

        out = reviews_df.copy()
        out["p_pos"] = p_pos
        out["p_neg"] = p_neg
        return out

    def _build_monthly_panel(self, sentiment_df: pd.DataFrame, identity: dict[str, Any]) -> pd.DataFrame:
        df = sentiment_df.copy()
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce")

        monthly = (
            df.groupby("month", as_index=False)
            .agg(
                review_count=("review_id", "count"),
                avg_stars=("stars", "mean"),
                tx_sent_mean=("p_pos", "mean"),
                tx_sent_std=("p_pos", "std"),
                tx_neg_share=("p_neg", lambda s: float((s >= 0.5).mean())),
                tx_pos_share=("p_pos", lambda s: float((s >= 0.5).mean())),
            )
            .sort_values("month")
            .reset_index(drop=True)
        )

        # Keep sparse month rows (months with activity only) to mirror model training
        # and candidate scoreability checks.
        monthly["review_count"] = pd.to_numeric(monthly["review_count"], errors="coerce").fillna(0).astype(int)
        monthly["avg_stars"] = pd.to_numeric(monthly["avg_stars"], errors="coerce").fillna(3.0)
        monthly["tx_sent_mean"] = pd.to_numeric(monthly["tx_sent_mean"], errors="coerce").fillna(0.5)

        monthly["business_id"] = identity.get("business_id")
        monthly["status"] = identity.get("status") or "Unknown"
        monthly["tx_sent_std"] = pd.to_numeric(monthly["tx_sent_std"], errors="coerce").fillna(0.0)
        monthly["tx_neg_share"] = pd.to_numeric(monthly["tx_neg_share"], errors="coerce").fillna(0.5)
        monthly["tx_pos_share"] = pd.to_numeric(monthly["tx_pos_share"], errors="coerce").fillna(0.5)

        return monthly[
            [
                "business_id",
                "status",
                "month",
                "review_count",
                "avg_stars",
                "tx_sent_mean",
                "tx_sent_std",
                "tx_neg_share",
                "tx_pos_share",
            ]
        ]

    def _build_inference_windows(
        self,
        monthly_df: pd.DataFrame,
        identity: dict[str, Any],
    ) -> tuple[np.ndarray, pd.DataFrame] | None:
        if monthly_df.empty:
            return None

        seq_len = 12
        horizon_months = 6
        inactive_k = 12
        min_active_months = 6
        min_reviews = 10

        base_feats = [
            "review_count",
            "avg_stars",
            "tx_sent_mean",
            "tx_sent_std",
            "tx_neg_share",
            "tx_pos_share",
        ]

        g = monthly_df.copy().sort_values("month").reset_index(drop=True)

        # Per-business standardization and trajectory features.
        x = g[base_feats].astype(float)
        mu = x.mean(axis=0)
        sd = x.std(axis=0).replace(0.0, 1.0)

        for column in base_feats:
            g[f"{column}_z"] = (g[column].astype(float) - float(mu[column])) / float(sd[column])

        z_cols = [f"{column}_z" for column in base_feats]
        for column in z_cols:
            g[f"{column}_d1"] = g[column].diff(1)
            g[f"{column}_rm3"] = g[column].rolling(3, min_periods=1).mean()
            g[f"{column}_rs3"] = g[column].rolling(3, min_periods=1).std()
            g[f"{column}_rs6"] = g[column].rolling(6, min_periods=1).std()

        month0 = int(g["month"].iloc[0].year) * 12 + int(g["month"].iloc[0].month)
        month_index = g["month"].dt.year.astype(int) * 12 + g["month"].dt.month.astype(int)
        g["months_since_first"] = (month_index - month0).astype(float)

        feat_cols = [
            column
            for column in g.columns
            if column.endswith("_z")
            or column.endswith("_d1")
            or column.endswith("_rm3")
            or column.endswith("_rs3")
            or column.endswith("_rs6")
        ] + ["months_since_first"]

        g[feat_cols] = g[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if len(g) < seq_len:
            return None

        global_last_month = self._global_last_month
        if global_last_month is None:
            global_last_month = g["month"].max()

        last_review_month = g["month"].max()
        status = identity.get("status") or "Unknown"
        closure_month = last_review_month if status == "Closed" else pd.NaT
        zombie_cutoff = global_last_month - pd.DateOffset(months=inactive_k)
        is_zombie_open = bool(status == "Open" and last_review_month <= zombie_cutoff)
        if is_zombie_open:
            return None

        x_windows: list[np.ndarray] = []
        meta_rows: list[dict[str, Any]] = []

        for start in range(0, len(g) - seq_len + 1):
            end = start + seq_len
            window = g.iloc[start:end].copy()
            window_end = window["month"].iloc[-1]

            active_months = int((window["review_count"] > 0).sum())
            total_reviews = float(window["review_count"].sum())
            if active_months < min_active_months:
                continue
            if total_reviews < min_reviews:
                continue

            if status == "Closed" and pd.notna(closure_month) and window_end >= closure_month:
                continue

            # Match notebook right-censoring rule for Open businesses.
            horizon_end = window_end + pd.DateOffset(months=horizon_months)
            if status == "Open" and horizon_end > global_last_month:
                continue

            x_windows.append(window[feat_cols].to_numpy(dtype=np.float32))
            meta_rows.append(
                {
                    "business_id": identity.get("business_id"),
                    "start_month": window["month"].iloc[0],
                    "end_month": window_end,
                }
            )

        if not x_windows:
            return None

        return np.stack(x_windows, axis=0), pd.DataFrame(meta_rows)

    def _predict_gru_probabilities(self, x_windows: np.ndarray) -> np.ndarray:
        model_path = self.settings.model_dir / "model_gru.keras"
        if not model_path.exists():
            raise ServiceError("gru_model_missing")

        if x_windows.size == 0:
            return np.asarray([], dtype=float)

        return self._predict_gru_probabilities_subprocess(x_windows, model_path)

    def _predict_gru_probabilities_subprocess(
        self,
        x_windows: np.ndarray,
        model_path: Path,
    ) -> np.ndarray:
        worker_path = Path(__file__).with_name("gru_worker.py")
        if not worker_path.exists():
            raise ServiceError("gru_worker_missing")

        with tempfile.TemporaryDirectory(prefix="sbp_gru_") as temp_dir:
            input_path = Path(temp_dir) / "x_windows.npy"
            output_path = Path(temp_dir) / "p_windows.npy"

            np.save(input_path, np.asarray(x_windows, dtype=np.float32))

            worker_args = [
                str(worker_path),
                "--model-path",
                str(model_path),
                "--input-path",
                str(input_path),
                "--output-path",
                str(output_path),
            ]
            command = self._build_gru_worker_command(worker_args)

            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.settings.gru_worker_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise ServiceError("gru_worker_timeout") from exc

            if result.returncode != 0:
                reason = self._gru_worker_failure_reason(result.returncode)
                stderr_tail = self._tail_text(result.stderr)
                stdout_tail = self._tail_text(result.stdout)

                if stderr_tail:
                    logger.error("GRU worker failed (%s) stderr:\n%s", reason, stderr_tail)
                elif stdout_tail:
                    logger.error("GRU worker failed (%s) stdout:\n%s", reason, stdout_tail)
                else:
                    logger.error("GRU worker failed (%s) with no output", reason)

                raise ServiceError(reason)

            if not output_path.exists():
                raise ServiceError("gru_worker_no_output")

            try:
                probs = np.load(output_path)
            except Exception as exc:
                raise ServiceError("gru_worker_invalid_output") from exc

        return np.asarray(probs, dtype=float).reshape(-1)

    def _build_gru_worker_command(self, worker_args: list[str]) -> list[str]:
        if self.settings.gru_worker_conda_env:
            return [
                "conda",
                "run",
                "-n",
                self.settings.gru_worker_conda_env,
                "python",
                *worker_args,
            ]

        worker_python = self.settings.gru_worker_python or sys.executable
        return [worker_python, *worker_args]

    @staticmethod
    def _gru_worker_failure_reason(returncode: int) -> str:
        if returncode < 0:
            return f"gru_worker_signal_{abs(returncode)}"
        if returncode > 128:
            # Shell-wrapped signal exit code pattern (128 + signal).
            sig_num = returncode - 128
            if sig_num > 0:
                return f"gru_worker_signal_{sig_num}"
        return f"gru_worker_exit_{returncode}"

    @staticmethod
    def _tail_text(text: str | None, *, max_lines: int = 20) -> str:
        if not text:
            return ""
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        return "\n".join(lines[-max_lines:])

    def _load_gru_model(self, model_path: Path) -> Any:
        import tensorflow as tf

        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            return self._prepare_loaded_gru_model(model, tf)
        except ValueError as exc:
            message = str(exc)
            if "Lambda layer" not in message and "safe_mode=False" not in message:
                raise ServiceError(f"gru_model_load_failed:{exc.__class__.__name__}") from exc

            # Artifact is local and versioned with this repo; retry with trusted deserialization.
            logger.warning("Retrying GRU model load with safe_mode=False: %s", model_path)
            try:
                model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                return self._prepare_loaded_gru_model(model, tf)
            except NotImplementedError as retry_exc:
                # Keras 3 may fail on Lambda layers missing output_shape metadata.
                retry_text = str(retry_exc).casefold()
                if "output_shape" not in retry_text and "infer the shape" not in retry_text:
                    raise ServiceError(
                        f"gru_model_load_failed:{retry_exc.__class__.__name__}"
                    ) from retry_exc

                patched_path = self._build_lambda_shape_patched_archive(model_path)
                if patched_path is None:
                    raise ServiceError("gru_model_lambda_output_shape_missing") from retry_exc

                try:
                    logger.warning(
                        "Retrying GRU model load with Lambda output_shape patch: %s",
                        patched_path,
                    )
                    model = tf.keras.models.load_model(patched_path, compile=False, safe_mode=False)
                    return self._prepare_loaded_gru_model(model, tf)
                except Exception as patched_exc:  # pragma: no cover - local TF/Keras dependent
                    raise ServiceError(
                        f"gru_model_load_failed:{patched_exc.__class__.__name__}"
                    ) from patched_exc
                finally:
                    patched_path.unlink(missing_ok=True)
            except TypeError as retry_exc:
                raise ServiceError("gru_model_requires_safe_mode_override") from retry_exc
            except Exception as retry_exc:  # pragma: no cover - depends on local TF/Keras build
                raise ServiceError(
                    f"gru_model_load_failed:{retry_exc.__class__.__name__}"
                ) from retry_exc
        except ServiceError:
            raise
        except Exception as exc:  # pragma: no cover - depends on local TF/Keras build
            raise ServiceError(f"gru_model_load_failed:{exc.__class__.__name__}") from exc

    def _prepare_loaded_gru_model(self, model: Any, tf_module: Any) -> Any:
        try:
            lambda_layer_type = tf_module.keras.layers.Lambda
        except Exception:
            return model

        for layer in getattr(model, "layers", []):
            if not isinstance(layer, lambda_layer_type):
                continue
            fn = getattr(layer, "function", None)
            globals_dict = getattr(fn, "__globals__", None)
            if isinstance(globals_dict, dict):
                globals_dict.setdefault("tf", tf_module)

        return model

    def _build_lambda_shape_patched_archive(self, model_path: Path) -> Path | None:
        try:
            with zipfile.ZipFile(model_path, "r") as archive:
                config = json.loads(archive.read("config.json"))
                changed = False

                for layer in config.get("config", {}).get("layers", []):
                    if layer.get("class_name") != "Lambda":
                        continue

                    layer_config = layer.setdefault("config", {})
                    if "output_shape" in layer_config:
                        continue

                    input_shape = layer.get("build_config", {}).get("input_shape")
                    if not isinstance(input_shape, list) or not input_shape:
                        continue

                    last_dim = input_shape[-1]
                    if not isinstance(last_dim, int):
                        continue

                    layer_config["output_shape"] = [last_dim]
                    changed = True

                if not changed:
                    return None

                with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as handle:
                    patched_path = Path(handle.name)

                with zipfile.ZipFile(patched_path, "w") as patched:
                    for member in archive.infolist():
                        payload = archive.read(member.filename)
                        if member.filename == "config.json":
                            payload = json.dumps(config).encode("utf-8")
                        patched.writestr(member, payload)

                return patched_path
        except Exception as exc:  # pragma: no cover - filesystem/env dependent
            logger.exception("Failed to build Lambda-patched archive for %s", model_path)
            raise ServiceError(f"gru_model_patch_failed:{exc.__class__.__name__}") from exc

    def _aggregate_window_risk(
        self,
        meta_windows: pd.DataFrame,
        probabilities: np.ndarray,
    ) -> tuple[float, str | None, list[dict[str, Any]]]:
        scored = meta_windows.copy()
        scored["p_closed"] = probabilities
        scored = scored.sort_values("end_month")

        recent = scored.tail(3)
        recent_windows = [
            {
                "end_month": self._format_month(row.end_month),
                "p_closed": float(row.p_closed),
            }
            for row in recent.itertuples()
        ]

        risk_score = float(recent["p_closed"].max()) if len(recent) else float(scored["p_closed"].max())
        risk_bucket = assign_risk_bucket(risk_score, self._risk_bins, self._risk_labels)
        return risk_score, risk_bucket, recent_windows

    def _build_live_chart_data(
        self,
        *,
        identity: dict[str, Any],
        monthly_df: pd.DataFrame,
        meta_windows: pd.DataFrame,
        probabilities: np.ndarray,
        sentiment_df: pd.DataFrame,
    ) -> dict[str, Any]:
        ratings_by_month: list[dict[str, Any]] = []
        if not monthly_df.empty:
            ordered_monthly = monthly_df.sort_values("month")
            for row in ordered_monthly.itertuples():
                ratings_by_month.append(
                    {
                        "month": self._format_month(row.month),
                        "avg_stars": as_optional_float(row.avg_stars),
                        "review_count": as_optional_int(row.review_count),
                    }
                )

        rating_bucket_counts_by_month: list[dict[str, Any]] = []
        if not sentiment_df.empty:
            rating_df = sentiment_df.copy()
            rating_df["month"] = (
                pd.to_datetime(rating_df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            )
            rating_df["stars"] = pd.to_numeric(rating_df["stars"], errors="coerce")
            rating_df = rating_df.dropna(subset=["month", "stars"])
            if not rating_df.empty:
                rating_df["stars_bucket"] = (
                    rating_df["stars"].round().clip(lower=1, upper=5).astype(int)
                )
                months = sorted(pd.to_datetime(rating_df["month"]).dt.to_period("M").dt.to_timestamp().unique())
                full_index = pd.MultiIndex.from_product(
                    [months, [1, 2, 3, 4, 5]],
                    names=["month", "stars_bucket"],
                )
                bucket_counts = (
                    rating_df.groupby(["month", "stars_bucket"], as_index=True)
                    .size()
                    .reindex(full_index, fill_value=0)
                    .reset_index(name="count")
                    .sort_values(["month", "stars_bucket"])
                )
                rating_bucket_counts_by_month = [
                    {
                        "month": self._format_month(row.month),
                        "stars_bucket": int(row.stars_bucket),
                        "count": int(row.count),
                    }
                    for row in bucket_counts.itertuples()
                ]

        predicted_close_by_month: list[dict[str, Any]] = []
        if len(meta_windows) and len(probabilities):
            scored = meta_windows.copy()
            scored["p_closed"] = probabilities
            scored = scored.sort_values("end_month")
            predicted_close_by_month = [
                {
                    "month": self._format_month(row.end_month),
                    "p_closed": as_optional_float(row.p_closed),
                }
                    for row in scored.itertuples()
            ]

        topics_by_month = self._derive_topics_by_month(sentiment_df)
        status = as_optional_str(identity.get("status")) or "Unknown"
        actual_close_month = (
            self._format_month(monthly_df["month"].max())
            if status.casefold() == "closed" and not monthly_df.empty
            else None
        )

        return {
            "ratings_by_month": ratings_by_month,
            "rating_bucket_counts_by_month": rating_bucket_counts_by_month,
            "predicted_close_by_month": predicted_close_by_month,
            "topics_by_month": topics_by_month,
            "actual_close_month": actual_close_month,
        }

    def _derive_topics_by_month(self, sentiment_df: pd.DataFrame) -> list[dict[str, Any]]:
        if sentiment_df.empty:
            return []

        df = sentiment_df.copy()
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        df = df.dropna(subset=["month"])
        if df.empty:
            return []

        topic_rows: list[dict[str, Any]] = []
        grouped = df.sort_values("month").groupby("month", as_index=False)

        for month_value, month_df in grouped:
            ranked = month_df.sort_values(["p_neg", "stars"], ascending=[False, True])
            negatives = ranked[ranked["p_neg"] >= 0.5].head(120)
            if negatives.empty:
                negatives = ranked.head(60)

            month_str = self._format_month(month_value)
            month_text = " ".join(negatives["text"].fillna("").astype(str)).casefold()

            if self._bertopic_topic_terms and month_text:
                topic_scores: list[tuple[float, int, list[str]]] = []
                for topic_id, terms in self._bertopic_topic_terms.items():
                    score = 0.0
                    for term in terms[:12]:
                        if " " in term:
                            score += float(month_text.count(term))
                        else:
                            score += float(len(re.findall(rf"\b{re.escape(term)}\b", month_text)))
                    if score > 0:
                        topic_scores.append((score, topic_id, terms))

                topic_scores.sort(key=lambda item: item[0], reverse=True)
                for rank, (score, topic_id, terms) in enumerate(topic_scores[:3], start=1):
                    topic_rows.append(
                        {
                            "month": month_str,
                            "theme": f"topic_{topic_id}: {', '.join(terms[:3])}",
                            "rank": rank,
                            "strength": float(score),
                        }
                    )
                continue

            texts = negatives["text"].fillna("").astype(str).tolist()
            keywords = self._extract_keywords_simple(texts) if texts else []
            evidence = []
            for row in negatives.head(12).itertuples():
                text = re.sub(r"\s+", " ", str(row.text or "")).strip()
                evidence.append({"snippet": text[:220]})

            themes, _ = self._map_themes_and_recommendations(keywords, evidence)
            if not themes and keywords:
                themes = ["needs_review"]

            for rank, theme in enumerate(themes[:3], start=1):
                topic_rows.append(
                    {
                        "month": month_str,
                        "theme": theme,
                        "rank": rank,
                        "strength": 4 - rank,
                    }
                )

        return topic_rows

    def _extract_evidence_reviews(self, sentiment_df: pd.DataFrame) -> list[dict[str, Any]]:
        ranked = sentiment_df.copy()
        ranked["stars"] = pd.to_numeric(ranked["stars"], errors="coerce")
        ranked = ranked.sort_values(["p_neg", "stars"], ascending=[False, True]).head(5)

        evidence: list[dict[str, Any]] = []
        for row in ranked.itertuples():
            text = re.sub(r"\s+", " ", str(row.text or "")).strip()
            snippet = text[:220] + ("..." if len(text) > 220 else "")
            evidence.append(
                {
                    "review_id": as_optional_str(row.review_id),
                    "date": self._format_date(row.date),
                    "stars": as_optional_int(row.stars),
                    "sentiment_neg_prob": as_optional_float(row.p_neg),
                    "snippet": snippet,
                }
            )
        return evidence

    def _derive_problems_and_recommendations(
        self,
        sentiment_df: pd.DataFrame,
        evidence_reviews: list[dict[str, Any]],
    ) -> tuple[str | None, list[str], list[str], str | None]:
        negative_texts = (
            sentiment_df.sort_values(["p_neg", "stars"], ascending=[False, True])["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        keywords = self._extract_problem_keywords(negative_texts)
        themes, recommendations = self._map_themes_and_recommendations(keywords, evidence_reviews)

        recommendation_notes = None
        if not themes:
            themes = ["needs_review"]
            recommendations = [RECO_FALLBACK_NOTE]
            recommendation_notes = RECO_FALLBACK_NOTE

        keywords_text = ", ".join(keywords[:20]) if keywords else None
        return keywords_text, themes[:3], recommendations[:3], recommendation_notes

    def _extract_problem_keywords(self, texts: list[str]) -> list[str]:
        texts = [text for text in texts if text.strip()]
        if not texts:
            return []

        if self._joblib_available:
            try:
                keywords = self._extract_keywords_with_topics(texts)
                if keywords:
                    return keywords
            except Exception:  # pragma: no cover - model artifact dependent
                logger.exception("Topic keyword extraction failed; using fallback keyword extraction")

        return self._extract_keywords_simple(texts)

    def _extract_keywords_with_topics(self, texts: list[str]) -> list[str]:
        vec_path = self.settings.model_dir / "vec.joblib"
        nmf_path = self.settings.model_dir / "nmf.joblib"
        if not vec_path.exists() or not nmf_path.exists():
            return []

        if self._vec_model is None or self._nmf_model is None:
            import joblib

            self._vec_model = joblib.load(vec_path)
            self._nmf_model = joblib.load(nmf_path)

        vectorized = self._vec_model.transform(texts)
        if vectorized.shape[0] == 0:
            return []

        weights = self._nmf_model.transform(vectorized)
        if weights.size == 0:
            return []

        topic_strength = np.asarray(weights.mean(axis=0)).ravel()
        top_topic_ids = np.argsort(topic_strength)[::-1][:3]

        vocab = self._vec_model.get_feature_names_out()
        keywords: list[str] = []
        for topic_idx in top_topic_ids:
            component = self._nmf_model.components_[topic_idx]
            top_terms = np.argsort(component)[::-1][:8]
            for term_idx in top_terms:
                term = str(vocab[term_idx]).strip()
                if term and term not in keywords:
                    keywords.append(term)

        return keywords[:20]

    def _extract_keywords_simple(self, texts: list[str]) -> list[str]:
        token_pattern = re.compile(r"[a-z]{3,}")
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "was",
            "were",
            "have",
            "from",
            "just",
            "they",
            "them",
            "very",
            "really",
            "restaurant",
            "place",
            "food",
            "service",
        }

        counts: dict[str, int] = {}
        for text in texts[:250]:
            for token in token_pattern.findall(text.casefold()):
                if token in stop_words:
                    continue
                counts[token] = counts.get(token, 0) + 1

        return [term for term, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]]

    def _map_themes_and_recommendations(
        self,
        keywords: list[str],
        evidence_reviews: list[dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        if not self._playbook:
            return [], []

        joined_keywords = " ".join(keywords).casefold()
        joined_evidence = " ".join((item.get("snippet") or "") for item in evidence_reviews).casefold()

        scored: list[tuple[int, str, str]] = []
        for theme, cues, recommendation in self._playbook:
            cue_hits = 0
            for cue in cues:
                cue_norm = cue.casefold()
                if cue_norm and (cue_norm in joined_keywords or cue_norm in joined_evidence):
                    cue_hits += 1
            if cue_hits > 0:
                scored.append((cue_hits, theme, recommendation))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        themes: list[str] = []
        recommendations: list[str] = []
        for _, theme, recommendation in scored:
            if theme not in themes:
                themes.append(theme)
                recommendations.append(recommendation)
            if len(themes) >= 3:
                break

        return themes, recommendations

    def _required_yelp_files_present(self) -> bool:
        if not self.settings.yelp_data_dir:
            return False
        required = [
            "yelp_academic_dataset_business.json",
            "yelp_academic_dataset_review.json",
        ]
        return all((self.settings.yelp_data_dir / filename).exists() for filename in required)

    def _is_tensorflow_runtime_available(self) -> bool:
        if self._tensorflow_runtime_available is not None:
            return self._tensorflow_runtime_available

        if not self._tensorflow_available:
            self._tensorflow_runtime_available = False
            return False

        probe_cmd = [sys.executable, "-c", "import tensorflow as tf; print(tf.__version__)"]
        process_env = os.environ.copy()
        process_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        process_env.setdefault("CUDA_VISIBLE_DEVICES", "-1")

        try:
            probe = subprocess.run(
                probe_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=20,
                env=process_env,
            )
            self._tensorflow_runtime_available = probe.returncode == 0
        except Exception:
            self._tensorflow_runtime_available = False

        return bool(self._tensorflow_runtime_available)

    def _format_month(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
                return text
            parsed = pd.to_datetime(text, errors="coerce")
            if pd.isna(parsed):
                return text
            return parsed.to_period("M").to_timestamp().strftime("%Y-%m-%d")

        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_period("M").to_timestamp().strftime("%Y-%m-%d")

    def _format_date(self, value: Any) -> str | None:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.strftime("%Y-%m-%d")

    def _not_scored_payload(
        self,
        *,
        reason: str,
        business_id: str | None = None,
        identity: dict[str, Any] | None = None,
        scoring_mode: str = "artifact",
    ) -> dict[str, Any]:
        identity = identity or {}
        return {
            "business_id": identity.get("business_id") or business_id,
            "name": identity.get("name"),
            "city": identity.get("city"),
            "state": identity.get("state"),
            "status": identity.get("status") or "Unknown",
            "total_reviews": identity.get("total_reviews"),
            "last_review_month": identity.get("last_review_month"),
            "risk_score": None,
            "risk_bucket": None,
            "recent_windows": [],
            "themes_top3": [],
            "problem_keywords": None,
            "evidence_reviews": [],
            "recommendations_top3": [],
            "recommendation_notes": None,
            "chart_data": {
                "ratings_by_month": [],
                "rating_bucket_counts_by_month": [],
                "predicted_close_by_month": [],
                "topics_by_month": [],
                "actual_close_month": None,
            },
            "scoring_mode": scoring_mode,
            "availability": "not_scored_yet",
            "not_scored_reason": reason,
        }


__all__ = [
    "RECO_FALLBACK_NOTE",
    "ServiceError",
    "SmallBizPulseService",
    "assign_risk_bucket",
    "normalize_text",
    "parse_recommendations",
]
