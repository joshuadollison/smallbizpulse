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
from dataclasses import dataclass, replace
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
GENERIC_CUISINE_CATEGORIES = {
    "restaurants",
    "food",
    "fast food",
    "american (traditional)",
    "american (new)",
    "nightlife",
    "bars",
    "coffee & tea",
    "cafes",
    "sandwiches",
    "breakfast & brunch",
    "desserts",
    "bakeries",
    "caterers",
}


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


def parse_categories(value: Any) -> list[str]:
    text = as_optional_str(value)
    if not text:
        return []
    items = [part.strip() for part in str(text).split(",") if part and part.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


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


TOPIC_TERM_TOKEN_PATTERN = re.compile(r"[a-z]+")
TOPIC_STOP_WORDS_BASE = {
    "dont",
    "didnt",
    "ive",
    "im",
    "youre",
    "thats",
    "cant",
    "couldnt",
    "wont",
    "wouldnt",
}


def _load_topic_stop_words() -> set[str]:
    stop_words = set(TOPIC_STOP_WORDS_BASE)
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        stop_words.update(str(word).casefold() for word in ENGLISH_STOP_WORDS)
    except Exception:
        pass
    return stop_words


TOPIC_STOP_WORDS = _load_topic_stop_words()


def normalize_topic_term(value: str | None) -> str:
    text = as_optional_str(value)
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def is_stop_word_term(term: str) -> bool:
    tokens = TOPIC_TERM_TOKEN_PATTERN.findall(term)
    if not tokens:
        return True
    return all(token in TOPIC_STOP_WORDS for token in tokens)


def sanitize_topic_terms(terms: Iterable[str], *, limit: int | None = None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_term in terms:
        term = normalize_topic_term(raw_term)
        if not term:
            continue
        if is_stop_word_term(term):
            continue
        if term in seen:
            continue
        seen.add(term)
        cleaned.append(term)
        if limit is not None and len(cleaned) >= int(limit):
            break
    return cleaned


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

    V2_RUNTIME_MODE = "v2_runtime"
    LEGACY_RUNTIME_MODE = "legacy_artifact"
    DEFAULT_MIN_ACTIVE_MONTHS = 6
    DEFAULT_MIN_REVIEWS_IN_WINDOW = 10

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self._tensorflow_available = module_available("tensorflow")
        self._transformers_available = module_available("transformers")
        self._torch_available = module_available("torch")
        self._joblib_available = module_available("joblib")
        self._bertopic_available = module_available("bertopic")
        self._vader_sentiment_available = module_available("vaderSentiment")
        self._nltk_available = module_available("nltk")
        self._tensorflow_runtime_available: bool | None = None

        self._risk_bins: list[float] = [0.0, 0.5, 0.65, 0.75, 0.85, 1.0]
        self._risk_labels: list[str] = ["low", "medium", "elevated", "high", "very_high"]
        self._playbook: list[tuple[str, list[str], str]] = []
        self._bertopic_topic_terms: dict[int, list[str]] = {}

        self._scored_df: pd.DataFrame | None = None
        self._scored_by_id: dict[str, dict[str, Any]] = {}
        self._runtime_mode: str = self.LEGACY_RUNTIME_MODE

        self._live_business_df: pd.DataFrame | None = None
        self._live_reviews_df: pd.DataFrame | None = None
        self._live_tips_df: pd.DataFrame | None = None
        self._live_checkins_df: pd.DataFrame | None = None
        self._global_last_month: pd.Timestamp | None = None
        self._live_window_scoreable_by_business: dict[tuple[str, int, int], bool] = {}

        self._sentiment_model: Any = None
        self._sentiment_tokenizer: Any = None
        self._sentiment_pos_idx: int | None = None
        self._sentiment_neg_idx: int | None = None
        self._vader_scorer: Any = None

        self._gru_model: Any = None
        self._vec_model: Any = None
        self._nmf_model: Any = None

        self._v2_bundle_dir: Path | None = None
        self._v2_runtime: Any = None
        self._v2_runtime_error: str | None = None
        self._v2_monthly_panel_df: pd.DataFrame | None = None
        self._v2_reviews_df: pd.DataFrame | None = None
        self._v2_negative_topics_df: pd.DataFrame | None = None
        self._v2_terminal_topics_df: pd.DataFrame | None = None
        self._v2_recovery_comparison_df: pd.DataFrame | None = None
        self._v2_city_closure_rates_df: pd.DataFrame | None = None
        self._v2_cuisine_closure_rates_df: pd.DataFrame | None = None
        self._v2_checkin_floor_df: pd.DataFrame | None = None
        self._v2_recovery_patterns_df: pd.DataFrame | None = None
        self._v2_topic_terms_by_id: dict[int, list[str]] = {}
        self._v2_recommendation_by_topic: dict[int, tuple[str | None, str | None]] = {}
        self._v2_recommendation_match_by_topic: dict[int, float | None] = {}
        self._v2_sequence_window_cache: dict[str, pd.DataFrame] = {}
        self._v2_runtime_score_cache: dict[str, dict[str, Any]] = {}
        self._v2_resilience_cache: dict[str, dict[str, Any]] = {}
        self._v2_global_last_month: pd.Timestamp | None = None
        self._v2_sequence_config: Any = None
        self._v2_gru_available: bool = False
        self._v2_bertopic_model_path: Path | None = None
        self._v2_bertopic_model: Any = None
        self._v2_bertopic_load_attempted: bool = False

        # Live path keeps intermediate representations in memory only.
        self._last_live_workflow: LiveWorkflowRepresentation | None = None

        self._load_artifacts()

    @property
    def yelp_data_available(self) -> bool:
        return self._required_yelp_files_present()

    def health(self) -> dict[str, Any]:
        yelp_present = self._required_yelp_files_present()
        yelp_business_file_exists = bool(
            self.settings.yelp_data_dir
            and (self.settings.yelp_data_dir / "yelp_academic_dataset_business.json").exists()
        )
        yelp_review_file_exists = bool(
            self.settings.yelp_data_dir
            and (self.settings.yelp_data_dir / "yelp_academic_dataset_review.json").exists()
        )
        yelp_tip_file_exists = bool(
            self.settings.yelp_data_dir
            and (self.settings.yelp_data_dir / "yelp_academic_dataset_tip.json").exists()
        )
        yelp_checkin_file_exists = bool(
            self.settings.yelp_data_dir
            and (self.settings.yelp_data_dir / "yelp_academic_dataset_checkin.json").exists()
        )

        live_ready = (
            self._runtime_mode == self.V2_RUNTIME_MODE
            and yelp_present
            and self._v2_gru_available
            and self._is_tensorflow_runtime_available()
        )

        dependency_status = {
            "tensorflow_package": bool(self._tensorflow_available),
            "tensorflow_runtime": bool(self._is_tensorflow_runtime_available()),
            "vader_sentiment_package": bool(self._vader_sentiment_available),
            "nltk_package": bool(self._nltk_available),
            "v2_runtime_ready": self._runtime_mode == self.V2_RUNTIME_MODE,
            "v2_gru_model_file": bool(self._v2_gru_available),
            "yelp_business_file": yelp_business_file_exists,
            "yelp_review_file": yelp_review_file_exists,
            "yelp_tip_file": yelp_tip_file_exists,
            "yelp_checkin_file": yelp_checkin_file_exists,
            "v2_bundle_dir": bool(self._v2_bundle_dir and self._v2_bundle_dir.exists()),
            "component2_topics_table": bool(
                self._v2_negative_topics_df is not None and not self._v2_negative_topics_df.empty
            ),
            "component3_recommendations_table": bool(len(self._v2_recommendation_by_topic)),
            "component4_city_table": bool(
                self._v2_city_closure_rates_df is not None and not self._v2_city_closure_rates_df.empty
            ),
            "component4_cuisine_table": bool(
                self._v2_cuisine_closure_rates_df is not None and not self._v2_cuisine_closure_rates_df.empty
            ),
            "component4_checkin_table": bool(
                self._v2_checkin_floor_df is not None and not self._v2_checkin_floor_df.empty
            ),
            "component4_recovery_table": bool(
                self._v2_recovery_patterns_df is not None and not self._v2_recovery_patterns_df.empty
            ),
        }

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
            "vader_sentiment_available": self._vader_sentiment_available,
            "nltk_available": self._nltk_available,
            "transformers_available": self._transformers_available,
            "torch_available": self._torch_available,
            "joblib_available": self._joblib_available,
            "gru_inference_mode": "subprocess_worker",
            "gru_worker_timeout_seconds": self.settings.gru_worker_timeout_seconds,
            "gru_worker_conda_env": self.settings.gru_worker_conda_env,
            "gru_worker_python": self.settings.gru_worker_python,
            "live_fallback_ready": live_ready,
            "live_scoring_ready": live_ready,
            "runtime_mode": self._runtime_mode,
            "v2_bundle_dir": str(self._v2_bundle_dir) if self._v2_bundle_dir else None,
            "v2_runtime_ready": self._runtime_mode == self.V2_RUNTIME_MODE,
            "v2_runtime_error": self._v2_runtime_error,
            "v2_gru_available": self._v2_gru_available,
            "dependency_status": dependency_status,
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
        min_active_months: int | None = None,
        min_reviews_in_window: int | None = None,
        live_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = normalize_text(name)
        if not query:
            return []
        query_tokens = [token for token in query.split() if token]
        query_token_set = set(query_tokens)

        city_q = normalize_text(city)
        state_q = normalize_text(state)

        if live_only:
            live_df = self._load_live_business_df()
            if live_df is None or live_df.empty:
                return []
            candidates_df = live_df.copy()
        else:
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
            if scorable_only:
                try:
                    is_scorable = self._candidate_is_scorable(
                        payload,
                        min_active_months=min_active_months,
                        min_reviews_in_window=min_reviews_in_window,
                        live_only=live_only,
                    )
                except TypeError:
                    # Backward compatibility for tests/patches that monkeypatch this method.
                    is_scorable = self._candidate_is_scorable(payload)
                if not is_scorable:
                    continue
            ranked.append((rank, payload))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[: max(1, min(limit, 50))]]

    def _candidate_is_scorable(
        self,
        candidate: dict[str, Any],
        *,
        min_active_months: int | None = None,
        min_reviews_in_window: int | None = None,
        live_only: bool = False,
    ) -> bool:
        if bool(candidate.get("risk_available", False)) and not live_only:
            return True

        if not self.settings.yelp_data_dir:
            return False
        if not self._required_yelp_files_present():
            return False

        business_id = as_optional_str(candidate.get("business_id"))
        if not business_id:
            return False
        status = as_optional_str(candidate.get("status")) or "Unknown"
        return self._can_build_live_windows_for_business(
            business_id,
            status,
            min_active_months=min_active_months,
            min_reviews_in_window=min_reviews_in_window,
        )

    def _can_build_live_windows_for_business(
        self,
        business_id: str,
        status: str,
        *,
        min_active_months: int | None = None,
        min_reviews_in_window: int | None = None,
    ) -> bool:
        resolved_min_active = (
            self.DEFAULT_MIN_ACTIVE_MONTHS
            if min_active_months is None
            else max(0, int(min_active_months))
        )
        resolved_min_reviews = (
            self.DEFAULT_MIN_REVIEWS_IN_WINDOW
            if min_reviews_in_window is None
            else max(0, int(min_reviews_in_window))
        )
        cache_key = (business_id, resolved_min_active, resolved_min_reviews)
        cached = self._live_window_scoreable_by_business.get(cache_key)
        if cached is not None:
            return cached

        reviews_df = self._load_live_reviews_df()
        if reviews_df is None or reviews_df.empty:
            self._live_window_scoreable_by_business[cache_key] = False
            return False

        try:
            monthly_panel = self._build_live_v2_monthly_panel(
                business_id=business_id,
                identity={
                    "business_id": business_id,
                    "status": status,
                },
            )
            if monthly_panel.empty:
                self._live_window_scoreable_by_business[cache_key] = False
                return False

            self._ensure_repo_root_on_path()
            from model_builder.v2.features import build_sequence_windows

            if self._v2_runtime is None or self._v2_runtime.sequence_config is None:
                self._live_window_scoreable_by_business[cache_key] = False
                return False

            sequence_config = replace(
                self._v2_runtime.sequence_config,
                min_active_months=resolved_min_active,
                min_reviews_in_window=resolved_min_reviews,
            )
            windows = build_sequence_windows(monthly_panel, config=sequence_config)
            can_score = bool(windows.X.size and not windows.meta.empty)
        except Exception:
            can_score = False

        self._live_window_scoreable_by_business[cache_key] = can_score
        return can_score

    def score_business(
        self,
        business_id: str,
        *,
        force_live_inference: bool = False,
        min_active_months: int | None = None,
        min_reviews_in_window: int | None = None,
    ) -> dict[str, Any]:
        # Compatibility entry point for API/tests.
        business_id = (business_id or "").strip()
        if not business_id:
            return self._not_scored_payload(reason="missing_business_id")

        if force_live_inference:
            if not self.settings.enable_live_fallback:
                artifact_row = self._scored_by_id.get(business_id)
                identity = self._lookup_live_business_identity(business_id) or self._identity_from_row(
                    artifact_row
                )
                return self._not_scored_payload(
                    business_id=business_id,
                    identity=identity,
                    reason="live_fallback_disabled",
                    scoring_mode="live_fallback",
                )
            result = self.score_business_live(
                business_id,
                min_active_months=min_active_months,
                min_reviews_in_window=min_reviews_in_window,
            )
            if result.get("availability") != "scored":
                result = dict(result)
                result["scoring_mode"] = "live_fallback"
            return result

        if self.settings.prefer_live_scoring:
            live_result = self.score_business_live(
                business_id,
                min_active_months=min_active_months,
                min_reviews_in_window=min_reviews_in_window,
            )
            if live_result.get("availability") == "scored":
                return live_result

        artifact_row = self._scored_by_id.get(business_id)
        if artifact_row is not None:
            if self._runtime_mode == self.V2_RUNTIME_MODE:
                return self._v2_payload(artifact_row)
            return self._artifact_payload(artifact_row)

        return self.score_business_live(
            business_id,
            min_active_months=min_active_months,
            min_reviews_in_window=min_reviews_in_window,
        )

    def score_business_live(
        self,
        business_id: str,
        *,
        min_active_months: int | None = None,
        min_reviews_in_window: int | None = None,
    ) -> dict[str, Any]:
        business_id = (business_id or "").strip()
        if not business_id:
            return self._not_scored_payload(reason="missing_business_id")

        live_outcome = self._invoke_live_scoring(
            business_id,
            min_active_months=min_active_months,
            min_reviews_in_window=min_reviews_in_window,
        )
        if live_outcome.ok:
            return live_outcome.payload

        live_identity = None
        live_quality = None
        if isinstance(live_outcome.payload, dict):
            candidate = live_outcome.payload.get("identity")
            if isinstance(candidate, dict):
                live_identity = candidate
            quality_candidate = live_outcome.payload.get("data_quality")
            if isinstance(quality_candidate, dict):
                live_quality = quality_candidate
        identity = live_identity or self._lookup_live_business_identity(business_id)
        return self._not_scored_payload(
            business_id=business_id,
            identity=identity,
            reason=live_outcome.reason or "not_scored",
            scoring_mode="live",
            data_quality=live_quality,
        )

    def _invoke_live_scoring(
        self,
        business_id: str,
        *,
        min_active_months: int | None = None,
        min_reviews_in_window: int | None = None,
    ) -> LiveScoringOutcome:
        try:
            return self._score_live_fallback(
                business_id,
                min_active_months=min_active_months,
                min_reviews_in_window=min_reviews_in_window,
            )
        except TypeError:
            # Backward compatibility for tests/patches monkeypatching legacy signature.
            return self._score_live_fallback(business_id)

    def artifact_risk_bands(self) -> list[str]:
        return list(self._risk_labels)

    def _ensure_v2_artifact_risk_for_business(
        self,
        business_id: str | None,
        *,
        fallback_score: float | None = None,
        fallback_bucket: str | None = None,
    ) -> tuple[float | None, str | None]:
        business_key = (business_id or "").strip()
        if not business_key:
            return fallback_score, fallback_bucket

        score = as_optional_float(fallback_score)
        bucket = as_optional_str(fallback_bucket) or assign_risk_bucket(
            score, self._risk_bins, self._risk_labels
        )
        if score is not None and bucket:
            return score, bucket

        cached_row = self._v2_runtime_score_cache.get(business_key)
        if isinstance(cached_row, dict):
            cached_score = as_optional_float(cached_row.get("risk_score"))
            cached_bucket = as_optional_str(cached_row.get("risk_bucket")) or assign_risk_bucket(
                cached_score, self._risk_bins, self._risk_labels
            )
            if cached_score is not None and cached_bucket:
                return cached_score, cached_bucket

        if self._runtime_mode != self.V2_RUNTIME_MODE:
            return score, bucket

        if self._v2_runtime is not None and self._v2_monthly_panel_df is not None:
            business_monthly = self._v2_monthly_panel_df[
                self._v2_monthly_panel_df["business_id"] == business_key
            ]
            if not business_monthly.empty:
                original_sequence_config = self._v2_runtime.sequence_config
                self._v2_runtime.sequence_config = None
                try:
                    runtime_scores = self._v2_runtime.score_monthly_panel(business_monthly)
                except Exception as exc:  # pragma: no cover - runtime dependent
                    logger.warning(
                        "Failed v2 runtime baseline/rule scoring for %s during search: %s",
                        business_key,
                        exc,
                    )
                    runtime_scores = pd.DataFrame()
                finally:
                    self._v2_runtime.sequence_config = original_sequence_config

                if not runtime_scores.empty:
                    runtime_row = runtime_scores.iloc[0].to_dict()
                    runtime_score = as_optional_float(runtime_row.get("risk_score"))
                    runtime_bucket = as_optional_str(runtime_row.get("risk_bucket")) or assign_risk_bucket(
                        runtime_score, self._risk_bins, self._risk_labels
                    )
                    if runtime_score is not None and runtime_bucket:
                        score = runtime_score
                        bucket = runtime_bucket
                        self._v2_runtime_score_cache[business_key] = dict(runtime_row)

        window_scores = self._build_v2_sequence_windows_for_business(business_key)
        if window_scores.empty:
            if score is not None and bucket:
                if self._scored_df is not None and not self._scored_df.empty:
                    mask = self._scored_df["business_id"].astype(str).eq(business_key)
                    if bool(mask.any()):
                        self._scored_df.loc[mask, "risk_score"] = score
                        self._scored_df.loc[mask, "risk_bucket"] = bucket
                scored_row = self._scored_by_id.get(business_key)
                if isinstance(scored_row, dict):
                    scored_row["risk_score"] = score
                    scored_row["risk_bucket"] = bucket
            return score, bucket

        recent_k = max(1, int(getattr(self._v2_runtime, "gru_recent_k_windows", 3)))
        recent_scores = pd.to_numeric(
            window_scores.tail(recent_k)["p_closed"],
            errors="coerce",
        ).dropna()
        if recent_scores.empty:
            return score, bucket

        score = float(recent_scores.max())
        bucket = assign_risk_bucket(score, self._risk_bins, self._risk_labels)
        cache_payload = {
            "risk_score": score,
            "risk_bucket": bucket,
            "risk_source": "gru",
        }
        self._v2_runtime_score_cache[business_key] = cache_payload

        if self._scored_df is not None and not self._scored_df.empty:
            mask = self._scored_df["business_id"].astype(str).eq(business_key)
            if bool(mask.any()):
                self._scored_df.loc[mask, "risk_score"] = score
                self._scored_df.loc[mask, "risk_bucket"] = bucket
                self._scored_df.loc[mask, "risk_source"] = "gru"

        scored_row = self._scored_by_id.get(business_key)
        if isinstance(scored_row, dict):
            scored_row["risk_score"] = score
            scored_row["risk_bucket"] = bucket
            scored_row["risk_source"] = "gru"

        return score, bucket

    def search_scored_artifacts(
        self,
        name: str,
        *,
        city: str | None = None,
        state: str | None = None,
        risk_bucket: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        if self._scored_df is None or self._scored_df.empty:
            return []

        query = normalize_text(name)
        query_tokens = [token for token in query.split() if token]
        query_token_set = set(query_tokens)
        city_q = normalize_text(city)
        state_q = normalize_text(state)
        risk_bucket_q = normalize_text(risk_bucket)
        allow_runtime_fill = bool(query_tokens or city_q or state_q)

        ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for _, row in self._scored_df.iterrows():
            candidate_name = as_optional_str(row.get("name")) or ""
            candidate_city = as_optional_str(row.get("city"))
            candidate_state = as_optional_str(row.get("state"))
            candidate_name_norm = normalize_text(candidate_name)
            candidate_tokens = [token for token in candidate_name_norm.split() if token]
            candidate_token_set = set(candidate_tokens)
            candidate_city_norm = normalize_text(candidate_city)
            candidate_state_norm = normalize_text(candidate_state)

            city_filter_ok = not city_q or city_q in candidate_city_norm
            state_filter_ok = not state_q or state_q == candidate_state_norm
            if not city_filter_ok or not state_filter_ok:
                continue

            exact = starts = contains = all_query_tokens_present = token_overlap = 0
            fuzzy = 0.0
            city_bonus = int(bool(city_q) and city_q == candidate_city_norm)
            state_bonus = int(bool(state_q) and state_q == candidate_state_norm)

            if query:
                exact = int(candidate_name_norm == query)
                starts = int(bool(query) and candidate_name_norm.startswith(query))
                contains = int(bool(query) and query in candidate_name_norm)
                fuzzy = SequenceMatcher(None, query, candidate_name_norm).ratio() if candidate_name_norm else 0.0
                token_overlap = len(query_token_set & candidate_token_set)
                all_query_tokens_present = int(
                    bool(query_tokens)
                    and all(
                        any(candidate_token.startswith(query_token) for candidate_token in candidate_tokens)
                        for query_token in query_tokens
                    )
                )
                token_prefix_overlap = int(
                    bool(query_tokens)
                    and any(
                        any(candidate_token.startswith(query_token) for candidate_token in candidate_tokens)
                        for query_token in query_tokens
                    )
                )
                fuzzy_match_ok = fuzzy >= 0.78 and token_prefix_overlap
                if not (exact or starts or contains or all_query_tokens_present or fuzzy_match_ok):
                    continue

            risk_score = as_optional_float(row.get("risk_score"))
            candidate_bucket = as_optional_str(row.get("risk_bucket")) or assign_risk_bucket(
                risk_score,
                self._risk_bins,
                self._risk_labels,
            )
            if risk_bucket_q:
                if allow_runtime_fill and candidate_bucket is None:
                    risk_score, candidate_bucket = self._ensure_v2_artifact_risk_for_business(
                        as_optional_str(row.get("business_id")),
                        fallback_score=risk_score,
                        fallback_bucket=candidate_bucket,
                    )
                candidate_bucket_norm = normalize_text(candidate_bucket)
                if candidate_bucket_norm != risk_bucket_q:
                    continue

            payload = {
                "business_id": as_optional_str(row.get("business_id")),
                "name": candidate_name,
                "city": candidate_city,
                "state": candidate_state,
                "status": as_optional_str(row.get("status")) or "Unknown",
                "review_count": as_optional_int(row.get("total_reviews")),
                "last_review_month": self._format_month(row.get("last_review_month")),
                "risk_score": risk_score,
                "risk_bucket": candidate_bucket,
            }

            if query:
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
            else:
                rank = (
                    risk_score if risk_score is not None else -1.0,
                    as_optional_int(row.get("total_reviews")) or 0,
                    payload["name"] or "",
                )
            ranked.append((rank, payload))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[: max(1, min(limit, 200))]]

    def score_business_artifact(self, business_id: str) -> dict[str, Any]:
        business_id = (business_id or "").strip()
        if not business_id:
            return self._not_scored_payload(reason="missing_business_id")

        artifact_row = self._scored_by_id.get(business_id)
        if artifact_row is None:
            return self._not_scored_payload(
                business_id=business_id,
                identity=self._lookup_live_business_identity(business_id),
                reason="artifact_business_not_found",
                scoring_mode="artifact",
            )

        if self._runtime_mode == self.V2_RUNTIME_MODE:
            return self._v2_payload(artifact_row)
        return self._artifact_payload(artifact_row)

    def _load_artifacts(self) -> None:
        v2_error = self._load_v2_artifacts()
        self._v2_runtime_error = v2_error
        if self._runtime_mode == self.V2_RUNTIME_MODE:
            return

        # Legacy artifact runtime is intentionally disabled.
        self._runtime_mode = "v2_unavailable"
        self._scored_df = pd.DataFrame(
            columns=[
                "business_id",
                "name",
                "city",
                "state",
                "status",
                "total_reviews",
                "last_review_month",
                "risk_score",
                "risk_bucket",
                "risk_source",
                "themes_top3",
                "recommendations_top3",
                "problem_keywords",
            ]
        )
        self._scored_by_id = {}

    def _load_v2_artifacts(self) -> str | None:
        bundle_root = self._resolve_v2_bundle_root()
        if bundle_root is None:
            return "v2_bundle_not_found"

        component1_dir = bundle_root / "component1_survival"
        monthly_panel_path = bundle_root / "monthly_signal_panel.csv"
        baseline_path = component1_dir / "baseline_models.joblib"
        rule_model_path = component1_dir / "rule_model.json"
        if not monthly_panel_path.exists():
            return f"v2_missing_monthly_panel:{monthly_panel_path}"
        if not baseline_path.exists() or not rule_model_path.exists():
            return f"v2_missing_component1_models:{component1_dir}"

        self._ensure_repo_root_on_path()
        try:
            from model_builder.v2 import InferenceDependencyError, SurvivalRuntime
        except Exception as exc:
            return f"v2_import_failed:{exc.__class__.__name__}"

        try:
            runtime = SurvivalRuntime.from_output_dir(component1_dir)
        except InferenceDependencyError as exc:
            return f"v2_dependency_error:{exc}"
        except Exception as exc:
            return f"v2_runtime_init_failed:{exc.__class__.__name__}"

        try:
            monthly_panel = pd.read_csv(monthly_panel_path)
        except Exception as exc:
            return f"v2_monthly_panel_load_failed:{exc.__class__.__name__}"

        required_columns = {
            "business_id",
            "month",
            "review_count",
            "avg_stars",
            "vader_mean",
            "vader_std",
            "vader_neg_share",
            "tip_count",
            "checkin_count",
            "name",
            "city",
            "state",
            "status",
        }
        missing_columns = sorted(required_columns - set(monthly_panel.columns))
        if missing_columns:
            return f"v2_monthly_panel_missing_columns:{missing_columns}"

        monthly_panel = monthly_panel.copy()
        monthly_panel["business_id"] = monthly_panel["business_id"].astype(str)
        monthly_panel["month"] = pd.to_datetime(monthly_panel["month"], errors="coerce")
        monthly_panel = monthly_panel.dropna(subset=["month"]).reset_index(drop=True)
        if monthly_panel.empty:
            return "v2_monthly_panel_empty"

        startup_sequence_config = runtime.sequence_config
        scored_df = (
            monthly_panel.sort_values(["business_id", "month"])
            .groupby("business_id", as_index=False)
            .agg(
                name=("name", "last"),
                city=("city", "last"),
                state=("state", "last"),
                status=("status", "last"),
                total_reviews=("review_count", "sum"),
                last_review_month=("month", "max"),
            )
        )
        scored_df["risk_score"] = np.nan
        scored_df["risk_bucket"] = np.nan
        scored_df["risk_source"] = "pending"
        scored_df["baseline_score"] = np.nan
        scored_df["rule_score"] = np.nan
        scored_df["gru_score"] = np.nan

        # Prefer precomputed GRU triage scores from component output when present.
        triage_path = component1_dir / "gru_business_triage.csv"
        if triage_path.exists():
            try:
                triage_df = pd.read_csv(triage_path).copy()
                if {"business_id", "risk_score"} <= set(triage_df.columns):
                    triage_df["business_id"] = triage_df["business_id"].astype(str)
                    triage_df["risk_score"] = pd.to_numeric(triage_df["risk_score"], errors="coerce")
                    triage_df["status"] = triage_df.get("status")
                    triage_df["end_month_last"] = pd.to_datetime(
                        triage_df.get("end_month_last"), errors="coerce"
                    )
                    triage_df = triage_df[
                        ["business_id", "risk_score", "status", "end_month_last"]
                    ].drop_duplicates(subset=["business_id"], keep="last")

                    scored_df = scored_df.merge(
                        triage_df,
                        on="business_id",
                        how="left",
                        suffixes=("", "_triage"),
                    )
                    scored_df["risk_score"] = pd.to_numeric(
                        scored_df["risk_score"], errors="coerce"
                    ).combine_first(
                        pd.to_numeric(scored_df.get("risk_score_triage"), errors="coerce")
                    )
                    scored_df["risk_source"] = np.where(
                        scored_df["risk_score"].notna(),
                        "gru",
                        scored_df["risk_source"],
                    )
                    scored_df["status"] = scored_df["status"].fillna(scored_df.get("status_triage"))
                    scored_df["last_review_month"] = scored_df["last_review_month"].fillna(
                        scored_df.get("end_month_last")
                    )
                    scored_df["risk_bucket"] = scored_df["risk_score"].apply(
                        lambda value: assign_risk_bucket(
                            as_optional_float(value), self._risk_bins, self._risk_labels
                        )
                    )
                    scored_df = scored_df.drop(
                        columns=[
                            "risk_score_triage",
                            "status_triage",
                            "end_month_last",
                        ],
                        errors="ignore",
                    )
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 GRU triage scores: %s", exc)

        self._v2_bundle_dir = bundle_root
        self._v2_runtime = runtime
        self._v2_monthly_panel_df = monthly_panel
        self._v2_global_last_month = monthly_panel["month"].max()
        self._v2_sequence_config = startup_sequence_config
        self._v2_gru_available = bool(
            runtime.artifacts.gru_model_path is not None and startup_sequence_config is not None
        )

        self._load_v2_optional_tables(bundle_root)
        topic_summary = self._build_v2_topic_summary_df()
        if topic_summary is not None and not topic_summary.empty:
            scored_df = scored_df.merge(topic_summary, on="business_id", how="left")
        else:
            scored_df["themes_top3"] = np.nan
            scored_df["recommendations_top3"] = np.nan
            scored_df["problem_keywords"] = np.nan

        self._scored_df = scored_df
        self._scored_by_id = {
            row["business_id"]: row
            for row in scored_df.to_dict(orient="records")
            if as_optional_str(row.get("business_id"))
        }
        self._runtime_mode = self.V2_RUNTIME_MODE
        self._v2_runtime_error = None

        logger.info("Loaded v2 runtime bundle from %s", bundle_root)
        return None

    def _resolve_v2_bundle_root(self) -> Path | None:
        configured = self.settings.v2_bundle_dir
        if configured is None:
            return None

        root = configured.expanduser()
        candidates: list[Path] = []

        def _is_bundle_dir(path: Path) -> bool:
            return (
                path.is_dir()
                and (path / "component1_survival").exists()
                and (path / "monthly_signal_panel.csv").exists()
            )

        if _is_bundle_dir(root):
            candidates.append(root)

        named = root / "v2_artifacts_colab_bundle"
        if _is_bundle_dir(named):
            candidates.append(named)

        if root.exists() and root.is_dir():
            for child in root.iterdir():
                if _is_bundle_dir(child):
                    candidates.append(child)

        if not candidates:
            return None

        # Prefer newest generated bundle when multiple are present.
        deduped: dict[str, Path] = {str(path.resolve()): path for path in candidates}
        ordered = sorted(
            deduped.values(),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return ordered[0]

    def _ensure_repo_root_on_path(self) -> None:
        repo_root_str = str(self.settings.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

    def _load_v2_optional_tables(self, bundle_root: Path) -> None:
        self._v2_reviews_df = None
        self._v2_negative_topics_df = None
        self._v2_terminal_topics_df = None
        self._v2_recovery_comparison_df = None
        self._v2_city_closure_rates_df = None
        self._v2_cuisine_closure_rates_df = None
        self._v2_checkin_floor_df = None
        self._v2_recovery_patterns_df = None
        self._v2_topic_terms_by_id = {}
        self._v2_recommendation_by_topic = {}
        self._v2_recommendation_match_by_topic = {}
        self._v2_sequence_window_cache = {}
        self._v2_runtime_score_cache = {}
        self._v2_resilience_cache = {}
        self._v2_bertopic_model_path = None
        self._v2_bertopic_model = None
        self._v2_bertopic_load_attempted = False

        bertopic_model_path = bundle_root / "component2_topics" / "bertopic_model"
        if bertopic_model_path.exists():
            self._v2_bertopic_model_path = bertopic_model_path

        reviews_path = bundle_root / "reviews_with_vader.csv"
        if reviews_path.exists():
            try:
                reviews_df = pd.read_csv(reviews_path)
                reviews_df = reviews_df.copy()
                reviews_df["business_id"] = reviews_df["business_id"].astype(str)
                reviews_df["date"] = pd.to_datetime(reviews_df.get("date"), errors="coerce")
                reviews_df = reviews_df.dropna(subset=["date"]).reset_index(drop=True)
                self._v2_reviews_df = reviews_df
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 reviews_with_vader.csv: %s", exc)

        negative_topics_path = bundle_root / "component2_topics" / "negative_review_topics.csv"
        if negative_topics_path.exists():
            try:
                negative_topics_df = pd.read_csv(negative_topics_path)
                negative_topics_df = negative_topics_df.copy()
                if "business_id" in negative_topics_df.columns:
                    negative_topics_df["business_id"] = negative_topics_df["business_id"].astype(str)
                if "date" in negative_topics_df.columns:
                    negative_topics_df["date"] = pd.to_datetime(
                        negative_topics_df["date"],
                        errors="coerce",
                    )
                if "topic_id" in negative_topics_df.columns:
                    negative_topics_df["topic_id"] = pd.to_numeric(
                        negative_topics_df["topic_id"],
                        errors="coerce",
                    ).astype("Int64")
                self._v2_negative_topics_df = negative_topics_df
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 negative_review_topics.csv: %s", exc)

        terminal_topics_path = bundle_root / "component2_topics" / "terminal_topics.csv"
        if terminal_topics_path.exists():
            try:
                terminal_df = pd.read_csv(terminal_topics_path).copy()
                if "business_id" in terminal_df.columns:
                    terminal_df["business_id"] = terminal_df["business_id"].astype(str)
                if "topic_id" in terminal_df.columns:
                    terminal_df["topic_id"] = pd.to_numeric(terminal_df["topic_id"], errors="coerce").astype("Int64")
                for numeric_column in ("share", "count"):
                    if numeric_column in terminal_df.columns:
                        terminal_df[numeric_column] = pd.to_numeric(
                            terminal_df[numeric_column], errors="coerce"
                        )
                self._v2_terminal_topics_df = terminal_df
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 terminal_topics.csv: %s", exc)

        recovery_comparison_path = bundle_root / "component2_topics" / "recovery_comparison.csv"
        if recovery_comparison_path.exists():
            try:
                recovery_df = pd.read_csv(recovery_comparison_path).copy()
                if "topic_id" in recovery_df.columns:
                    recovery_df["topic_id"] = pd.to_numeric(
                        recovery_df["topic_id"], errors="coerce"
                    ).astype("Int64")
                for numeric_column in (
                    "closed_after_negative",
                    "open_after_negative",
                    "closed_minus_open_share",
                ):
                    if numeric_column in recovery_df.columns:
                        recovery_df[numeric_column] = pd.to_numeric(
                            recovery_df[numeric_column], errors="coerce"
                        )
                self._v2_recovery_comparison_df = recovery_df
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 recovery_comparison.csv: %s", exc)

        topic_terms_path = bundle_root / "component2_topics" / "topic_terms.csv"
        if topic_terms_path.exists():
            try:
                topic_terms_df = pd.read_csv(topic_terms_path)
                parsed: dict[int, list[str]] = {}
                if {"topic_id", "top_terms"} <= set(topic_terms_df.columns):
                    for row in topic_terms_df.itertuples(index=False):
                        topic_id = as_optional_int(getattr(row, "topic_id", None))
                        terms_text = as_optional_str(getattr(row, "top_terms", None))
                        if topic_id is None or not terms_text:
                            continue
                        terms = sanitize_topic_terms(terms_text.split(","), limit=20)
                        if terms:
                            parsed[topic_id] = terms
                self._v2_topic_terms_by_id = parsed
                self._bertopic_topic_terms = parsed
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 topic_terms.csv: %s", exc)

        recommendations_path = bundle_root / "component3_recommendations" / "topic_recommendations.csv"
        if recommendations_path.exists():
            try:
                recommendation_df = pd.read_csv(recommendations_path)
                mapping: dict[int, tuple[str | None, str | None]] = {}
                match_scores: dict[int, float | None] = {}
                if "topic_id" in recommendation_df.columns:
                    recommendation_df["topic_id"] = pd.to_numeric(
                        recommendation_df["topic_id"], errors="coerce"
                    ).astype("Int64")
                    for row in recommendation_df.itertuples(index=False):
                        topic_id = as_optional_int(getattr(row, "topic_id", None))
                        if topic_id is None:
                            continue
                        theme = as_optional_str(getattr(row, "theme", None))
                        recommendation = as_optional_str(getattr(row, "recommendation", None))
                        match_score = as_optional_float(getattr(row, "match_score", None))
                        mapping[topic_id] = (theme, recommendation)
                        match_scores[topic_id] = match_score
                self._v2_recommendation_by_topic = mapping
                self._v2_recommendation_match_by_topic = match_scores
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 topic_recommendations.csv: %s", exc)

        city_rates_path = bundle_root / "component4_resilience" / "city_closure_rates.csv"
        if city_rates_path.exists():
            try:
                city_df = pd.read_csv(city_rates_path).copy()
                for name in ("city", "state"):
                    if name in city_df.columns:
                        city_df[name] = city_df[name].fillna("").astype(str).str.strip()
                for numeric_column in ("businesses", "closed", "closure_rate"):
                    if numeric_column in city_df.columns:
                        city_df[numeric_column] = pd.to_numeric(city_df[numeric_column], errors="coerce")
                self._v2_city_closure_rates_df = city_df
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 city_closure_rates.csv: %s", exc)

        cuisine_rates_path = bundle_root / "component4_resilience" / "cuisine_closure_rates.csv"
        if cuisine_rates_path.exists():
            try:
                cuisine_df = pd.read_csv(cuisine_rates_path).copy()
                if "category" in cuisine_df.columns:
                    cuisine_df["category"] = cuisine_df["category"].fillna("").astype(str).str.strip()
                for numeric_column in ("businesses", "closed", "closure_rate"):
                    if numeric_column in cuisine_df.columns:
                        cuisine_df[numeric_column] = pd.to_numeric(cuisine_df[numeric_column], errors="coerce")
                self._v2_cuisine_closure_rates_df = cuisine_df
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 cuisine_closure_rates.csv: %s", exc)

        checkin_floor_path = bundle_root / "component4_resilience" / "checkin_floor_analysis.csv"
        if checkin_floor_path.exists():
            try:
                floor_df = pd.read_csv(checkin_floor_path).copy()
                for numeric_column in (
                    "bin_left",
                    "bin_right",
                    "businesses",
                    "closure_rate",
                    "closure_rate_delta",
                    "activity_floor",
                ):
                    if numeric_column in floor_df.columns:
                        floor_df[numeric_column] = pd.to_numeric(floor_df[numeric_column], errors="coerce")
                self._v2_checkin_floor_df = floor_df
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 checkin_floor_analysis.csv: %s", exc)

        recovery_patterns_path = bundle_root / "component4_resilience" / "recovery_patterns.csv"
        if recovery_patterns_path.exists():
            try:
                pattern_df = pd.read_csv(recovery_patterns_path).copy()
                if "business_id" in pattern_df.columns:
                    pattern_df["business_id"] = pattern_df["business_id"].astype(str)
                for numeric_column in (
                    "had_negative_phase",
                    "recovered_pattern",
                    "trough_sentiment",
                    "final_sentiment",
                    "sentiment_delta",
                    "trough_checkins",
                    "final_checkins",
                    "checkin_delta",
                ):
                    if numeric_column in pattern_df.columns:
                        pattern_df[numeric_column] = pd.to_numeric(
                            pattern_df[numeric_column], errors="coerce"
                        )
                self._v2_recovery_patterns_df = pattern_df
            except Exception as exc:  # pragma: no cover - data dependent
                logger.warning("Failed loading v2 recovery_patterns.csv: %s", exc)

    def _build_v2_topic_summary_df(self) -> pd.DataFrame | None:
        if self._v2_negative_topics_df is None or self._v2_negative_topics_df.empty:
            return None
        if "topic_id" not in self._v2_negative_topics_df.columns:
            return None

        rows: list[dict[str, Any]] = []
        grouped = (
            self._v2_negative_topics_df.dropna(subset=["business_id", "topic_id"])
            .groupby(["business_id", "topic_id"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["business_id", "count"], ascending=[True, False])
        )

        for business_id, business_topics in grouped.groupby("business_id"):
            themes: list[str] = []
            recommendations: list[str] = []
            keywords: list[str] = []

            for row in business_topics.head(3).itertuples(index=False):
                topic_id = as_optional_int(getattr(row, "topic_id", None))
                if topic_id is None:
                    continue

                mapped_theme, mapped_recommendation = self._v2_recommendation_by_topic.get(topic_id, (None, None))
                if mapped_theme is None:
                    mapped_theme = f"topic_{topic_id}"
                if mapped_theme not in themes:
                    themes.append(mapped_theme)

                if mapped_recommendation and mapped_recommendation not in recommendations:
                    recommendations.append(mapped_recommendation)

                topic_terms = sanitize_topic_terms(
                    self._v2_topic_terms_by_id.get(topic_id, []),
                    limit=8,
                )
                for term in topic_terms:
                    if term not in keywords:
                        keywords.append(term)

            rows.append(
                {
                    "business_id": str(business_id),
                    "themes_top3": ", ".join(themes[:3]) if themes else None,
                    "recommendations_top3": "  ".join(
                        f"{idx}. {item}" for idx, item in enumerate(recommendations[:3], start=1)
                    )
                    if recommendations
                    else None,
                    "problem_keywords": ", ".join(keywords[:20]) if keywords else None,
                }
            )

        if not rows:
            return None
        return pd.DataFrame(rows)

    def _load_legacy_artifacts(self) -> None:
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
                        terms = sanitize_topic_terms(top_terms.split(","), limit=20)
                        if terms:
                            parsed[topic_id] = terms
                    self._bertopic_topic_terms = parsed
            except Exception as exc:  # pragma: no cover - optional artifact
                logger.warning("Failed to parse B_topic_terms.csv: %s", exc)

        self._runtime_mode = self.LEGACY_RUNTIME_MODE

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
        if unscored.empty:
            return scored

        # Keep concat schema-stable so pandas does not infer dtypes from all-NA extras.
        unscored = unscored.reindex(columns=scored.columns)
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
        business_df["categories"] = business_df["categories"].fillna("").astype(str)
        business_df["business_stars"] = pd.to_numeric(business_df.get("stars"), errors="coerce")
        business_df["business_review_count"] = pd.to_numeric(
            business_df.get("review_count"), errors="coerce"
        )
        business_df["total_reviews"] = pd.to_numeric(business_df.get("review_count"), errors="coerce")

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
                "categories",
                "business_stars",
                "business_review_count",
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

    def _build_data_quality(
        self,
        *,
        expected_reviews: int | None,
        observed_reviews: int | None,
    ) -> dict[str, Any]:
        expected = as_optional_int(expected_reviews)
        observed = as_optional_int(observed_reviews)

        coverage_ratio: float | None = None
        if expected is not None and expected > 0 and observed is not None:
            coverage_ratio = float(observed) / float(expected)

        has_mismatch = bool(
            expected is not None
            and observed is not None
            and expected > 0
            and observed < expected
        )

        return {
            "expected_reviews": expected,
            "observed_reviews": observed,
            "coverage_ratio": coverage_ratio,
            "has_mismatch": has_mismatch,
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
            "component2_diagnostics": {
                "negative_review_count": None,
                "terminal_topics": [],
                "topic_recovery_gaps": [],
            },
            "resilience_context": {
                "city_context": None,
                "cuisine_context": [],
                "checkin_floor_context": None,
                "recovery_pattern": None,
            },
            "chart_data": chart_data,
            "scoring_mode": "artifact",
            "availability": "scored",
            "data_quality": {
                "expected_reviews": None,
                "observed_reviews": None,
                "coverage_ratio": None,
                "has_mismatch": False,
            },
            "not_scored_reason": None,
        }

    def _v2_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        business_id = as_optional_str(row.get("business_id"))
        status = as_optional_str(row.get("status")) or "Unknown"
        risk_score = as_optional_float(row.get("risk_score"))
        risk_bucket = as_optional_str(row.get("risk_bucket")) or assign_risk_bucket(
            risk_score, self._risk_bins, self._risk_labels
        )
        risk_source = as_optional_str(row.get("risk_source")) or "unknown"

        runtime_row = self._v2_runtime_score_cache.get(business_id or "")
        if runtime_row is None and business_id and self._v2_runtime is not None and self._v2_monthly_panel_df is not None:
            business_monthly = self._v2_monthly_panel_df[
                self._v2_monthly_panel_df["business_id"] == business_id
            ]
            if not business_monthly.empty:
                original_sequence_config = self._v2_runtime.sequence_config
                self._v2_runtime.sequence_config = None
                try:
                    runtime_scores = self._v2_runtime.score_monthly_panel(business_monthly)
                except Exception as exc:  # pragma: no cover - runtime dependent
                    logger.warning("Failed v2 runtime baseline/rule scoring for %s: %s", business_id, exc)
                    runtime_scores = pd.DataFrame()
                finally:
                    self._v2_runtime.sequence_config = original_sequence_config

                if not runtime_scores.empty:
                    runtime_row = runtime_scores.iloc[0].to_dict()
                    self._v2_runtime_score_cache[business_id] = runtime_row

        if runtime_row is not None:
            runtime_score = as_optional_float(runtime_row.get("risk_score"))
            runtime_bucket = as_optional_str(runtime_row.get("risk_bucket")) or assign_risk_bucket(
                runtime_score, self._risk_bins, self._risk_labels
            )
            runtime_source = as_optional_str(runtime_row.get("risk_source"))
            if runtime_score is not None:
                risk_score = runtime_score
                risk_bucket = runtime_bucket
            if runtime_source:
                risk_source = runtime_source

        component2_diagnostics = self._build_v2_component2_diagnostics(business_id)
        terminal_topics = component2_diagnostics.get("terminal_topics", [])

        themes = parse_csv_list(row.get("themes_top3"))
        recommendations = parse_recommendations(row.get("recommendations_top3"))
        problem_keywords = as_optional_str(row.get("problem_keywords"))
        recommendation_notes = None

        if not themes and terminal_topics:
            themes = [
                as_optional_str(topic.get("theme"))
                for topic in terminal_topics
                if as_optional_str(topic.get("theme"))
            ]

        if not recommendations and terminal_topics:
            inferred_recommendations: list[str] = []
            for topic in terminal_topics:
                topic_id = as_optional_int(topic.get("topic_id"))
                if topic_id is None:
                    continue
                _, mapped_recommendation = self._v2_recommendation_by_topic.get(topic_id, (None, None))
                if mapped_recommendation and mapped_recommendation not in inferred_recommendations:
                    inferred_recommendations.append(mapped_recommendation)
            recommendations = inferred_recommendations[:3]

        if not problem_keywords and terminal_topics:
            keyword_terms: list[str] = []
            for topic in terminal_topics:
                topic_id = as_optional_int(topic.get("topic_id"))
                if topic_id is None:
                    continue
                topic_terms = sanitize_topic_terms(
                    self._v2_topic_terms_by_id.get(topic_id, []),
                    limit=8,
                )
                for term in topic_terms:
                    if term not in keyword_terms:
                        keyword_terms.append(term)
            if keyword_terms:
                problem_keywords = ", ".join(keyword_terms[:20])

        if not themes:
            recommendation_notes = (
                "Component 2 strict-negative filter found no topic diagnostics for this business. "
                f"{RECO_FALLBACK_NOTE}"
            )
            themes = ["needs_review"]

        if not recommendations:
            recommendation_notes = recommendation_notes or RECO_FALLBACK_NOTE
            recommendations = [RECO_FALLBACK_NOTE]

        window_scores = self._build_v2_sequence_windows_for_business(business_id)
        if not window_scores.empty:
            recent_gru_scores = window_scores.tail(max(1, int(getattr(self._v2_runtime, "gru_recent_k_windows", 3))))
            risk_score = float(pd.to_numeric(recent_gru_scores["p_closed"], errors="coerce").max())
            risk_bucket = assign_risk_bucket(risk_score, self._risk_bins, self._risk_labels)
            risk_source = "gru"

        recent_windows = [
            {
                "end_month": self._format_month(row_data.end_month),
                "p_closed": as_optional_float(row_data.p_closed),
            }
            for row_data in window_scores.tail(3).itertuples(index=False)
        ]

        evidence_reviews = self._build_v2_evidence_reviews(business_id)
        chart_data = self._build_v2_chart_data(
            business_id=business_id,
            status=status,
            window_scores=window_scores,
        )
        resilience_context = self._build_v2_resilience_context(business_id, row)

        business_monthly = None
        if self._v2_monthly_panel_df is not None and business_id is not None:
            business_monthly = self._v2_monthly_panel_df[
                self._v2_monthly_panel_df["business_id"] == business_id
            ]

        total_reviews = as_optional_int(row.get("total_reviews"))
        if total_reviews is None and business_monthly is not None and not business_monthly.empty:
            total_reviews = int(
                pd.to_numeric(business_monthly["review_count"], errors="coerce").fillna(0).sum()
            )

        expected_reviews = None
        if business_monthly is not None and not business_monthly.empty and "business_review_count" in business_monthly.columns:
            expected_reviews = as_optional_int(
                pd.to_numeric(business_monthly["business_review_count"], errors="coerce").iloc[-1]
            )
        if expected_reviews is None:
            expected_reviews = total_reviews

        data_quality = self._build_data_quality(
            expected_reviews=expected_reviews,
            observed_reviews=total_reviews,
        )

        return {
            "business_id": business_id,
            "name": as_optional_str(row.get("name")),
            "city": as_optional_str(row.get("city")),
            "state": as_optional_str(row.get("state")),
            "status": status,
            "total_reviews": total_reviews,
            "last_review_month": self._format_month(row.get("last_review_month")),
            "risk_score": risk_score,
            "risk_bucket": risk_bucket,
            "recent_windows": recent_windows,
            "themes_top3": themes[:3],
            "problem_keywords": problem_keywords,
            "evidence_reviews": evidence_reviews,
            "recommendations_top3": recommendations[:3],
            "recommendation_notes": recommendation_notes,
            "component2_diagnostics": component2_diagnostics,
            "resilience_context": resilience_context,
            "chart_data": chart_data,
            "scoring_mode": f"{self.V2_RUNTIME_MODE}:{risk_source}",
            "availability": "scored",
            "data_quality": data_quality,
            "not_scored_reason": None,
        }

    def _build_v2_chart_data(
        self,
        *,
        business_id: str | None,
        status: str,
        window_scores: pd.DataFrame,
    ) -> dict[str, Any]:
        ratings_by_month: list[dict[str, Any]] = []
        rating_bucket_counts_by_month: list[dict[str, Any]] = []
        predicted_close_by_month: list[dict[str, Any]] = []
        topics_by_month: list[dict[str, Any]] = []
        topics_per_class: list[dict[str, Any]] = []
        actual_close_month: str | None = None

        if business_id and self._v2_monthly_panel_df is not None:
            business_monthly = self._v2_monthly_panel_df[
                self._v2_monthly_panel_df["business_id"] == business_id
            ].copy()
            if not business_monthly.empty:
                business_monthly = business_monthly.sort_values("month")
                active_monthly = business_monthly[
                    pd.to_numeric(business_monthly["review_count"], errors="coerce").fillna(0.0) > 0.0
                ]
                ratings_by_month = [
                    {
                        "month": self._format_month(record.month),
                        "avg_stars": as_optional_float(record.avg_stars),
                        "review_count": as_optional_int(record.review_count),
                    }
                    for record in active_monthly.itertuples(index=False)
                ]
                if status.casefold() == "closed":
                    actual_close_month = self._format_month(business_monthly["month"].max())

        business_reviews = self._v2_business_reviews(business_id)
        if business_reviews is not None and not business_reviews.empty:
            bucket_df = business_reviews.copy()
            bucket_df["month"] = pd.to_datetime(bucket_df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            bucket_df["stars"] = pd.to_numeric(bucket_df["stars"], errors="coerce")
            bucket_df = bucket_df.dropna(subset=["month", "stars"])
            if not bucket_df.empty:
                bucket_df["stars_bucket"] = bucket_df["stars"].round().clip(lower=1, upper=5).astype(int)
                months = sorted(bucket_df["month"].unique())
                full_index = pd.MultiIndex.from_product(
                    [months, [1, 2, 3, 4, 5]],
                    names=["month", "stars_bucket"],
                )
                counts = (
                    bucket_df.groupby(["month", "stars_bucket"], as_index=True)
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
                    for row in counts.itertuples(index=False)
                ]

        if not window_scores.empty:
            predicted_close_by_month = [
                {
                    "month": self._format_month(row.end_month),
                    "p_closed": as_optional_float(row.p_closed),
                }
                for row in window_scores.sort_values("end_month").itertuples(index=False)
            ]

        topics_by_month = self._build_v2_topics_by_month(business_id)
        topics_per_class = self._build_v2_topics_per_class(business_id)

        return {
            "ratings_by_month": ratings_by_month,
            "rating_bucket_counts_by_month": rating_bucket_counts_by_month,
            "predicted_close_by_month": predicted_close_by_month,
            "topics_by_month": topics_by_month,
            "topics_per_class": topics_per_class,
            "actual_close_month": actual_close_month,
        }

    def _build_v2_sequence_windows_for_business(self, business_id: str | None) -> pd.DataFrame:
        if not business_id:
            return pd.DataFrame(columns=["end_month", "p_closed"])

        cached = self._v2_sequence_window_cache.get(business_id)
        if cached is not None:
            return cached.copy()

        if (
            self._v2_runtime is None
            or self._v2_sequence_config is None
            or self._v2_monthly_panel_df is None
        ):
            return pd.DataFrame(columns=["end_month", "p_closed"])

        self._ensure_repo_root_on_path()
        try:
            from model_builder.v2.features import build_sequence_windows
        except Exception:
            return pd.DataFrame(columns=["end_month", "p_closed"])

        business_monthly = self._v2_monthly_panel_df[
            self._v2_monthly_panel_df["business_id"] == business_id
        ].copy()
        if business_monthly.empty:
            return pd.DataFrame(columns=["end_month", "p_closed"])

        panel_input = business_monthly.copy()
        if self._v2_global_last_month is not None:
            business_last_month = panel_input["month"].max()
            if pd.notna(business_last_month) and self._v2_global_last_month > business_last_month:
                pad_row = panel_input.iloc[[0]].copy()
                pad_row["business_id"] = "__sbp_global_month_pad__"
                pad_row["name"] = "__sbp_global_month_pad__"
                pad_row["status"] = "Open"
                pad_row["month"] = self._v2_global_last_month
                for column in [
                    "review_count",
                    "avg_stars",
                    "vader_mean",
                    "vader_std",
                    "vader_neg_share",
                    "tip_count",
                    "checkin_count",
                    "business_stars",
                    "business_review_count",
                ]:
                    if column in pad_row.columns:
                        pad_row[column] = 0.0
                panel_input = pd.concat([panel_input, pad_row], ignore_index=True)

        try:
            windows = build_sequence_windows(panel_input, config=self._v2_sequence_config)
        except Exception as exc:  # pragma: no cover - data dependent
            logger.warning("Failed building v2 sequence windows for %s: %s", business_id, exc)
            return pd.DataFrame(columns=["end_month", "p_closed"])

        if windows.X.size == 0 or windows.meta.empty:
            result = pd.DataFrame(columns=["end_month", "p_closed"])
            self._v2_sequence_window_cache[business_id] = result
            return result.copy()

        model = self._v2_runtime._load_gru_model()  # Runtime caches model internally.
        if model is None:
            result = pd.DataFrame(columns=["end_month", "p_closed"])
            self._v2_sequence_window_cache[business_id] = result
            return result.copy()

        meta = windows.meta.reset_index(drop=True).copy()
        business_mask = meta["business_id"].astype(str).eq(business_id).to_numpy()
        if not business_mask.any():
            result = pd.DataFrame(columns=["end_month", "p_closed"])
            self._v2_sequence_window_cache[business_id] = result
            return result.copy()

        x_business = windows.X[business_mask]
        meta_business = meta.loc[business_mask].copy().reset_index(drop=True)

        try:
            probabilities = model.predict(x_business, batch_size=512, verbose=0).reshape(-1)
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.warning("Failed scoring v2 GRU windows for %s: %s", business_id, exc)
            result = pd.DataFrame(columns=["end_month", "p_closed"])
            self._v2_sequence_window_cache[business_id] = result
            return result.copy()

        meta_business["p_closed"] = probabilities
        result = (
            meta_business[["end_month", "p_closed"]]
            .sort_values("end_month")
            .reset_index(drop=True)
        )
        self._v2_sequence_window_cache[business_id] = result
        return result.copy()

    def _v2_business_reviews(self, business_id: str | None) -> pd.DataFrame | None:
        if not business_id or self._v2_reviews_df is None:
            return None
        subset = self._v2_reviews_df[self._v2_reviews_df["business_id"] == business_id]
        return subset.copy()

    def _load_v2_bertopic_model(self) -> Any | None:
        if self._v2_bertopic_model is not None:
            return self._v2_bertopic_model
        if self._v2_bertopic_load_attempted:
            return None
        self._v2_bertopic_load_attempted = True

        if not self._bertopic_available or self._v2_bertopic_model_path is None:
            return None

        try:
            from bertopic import BERTopic

            self._v2_bertopic_model = BERTopic.load(str(self._v2_bertopic_model_path))
        except Exception as exc:  # pragma: no cover - optional dependency / artifact dependent
            logger.warning("Failed loading v2 BERTopic model: %s", exc)
            self._v2_bertopic_model = None
        return self._v2_bertopic_model

    def _assign_topics_from_terms(self, texts: list[str]) -> list[int]:
        if not texts:
            return []

        topic_terms: list[tuple[int, list[str]]] = []
        for topic_id, terms in self._v2_topic_terms_by_id.items():
            cleaned = sanitize_topic_terms(terms, limit=12)
            if cleaned:
                topic_terms.append((int(topic_id), cleaned))

        if not topic_terms:
            return [-1 for _ in texts]

        assigned: list[int] = []
        for text in texts:
            text_norm = re.sub(r"\s+", " ", str(text).casefold())
            best_topic = -1
            best_score = 0.0
            for topic_id, terms in topic_terms:
                score = 0.0
                for term in terms:
                    if " " in term:
                        score += float(text_norm.count(term))
                    else:
                        score += float(len(re.findall(rf"\b{re.escape(term)}\b", text_norm)))
                if score > best_score:
                    best_score = score
                    best_topic = topic_id
            assigned.append(best_topic if best_score > 0 else -1)
        return assigned

    def _assign_topics_for_reviews(self, texts: list[str]) -> list[int]:
        if not texts:
            return []

        model = self._load_v2_bertopic_model()
        if model is not None:
            try:
                topics, _ = model.transform(texts)
                assigned: list[int] = []
                for topic in topics:
                    topic_id = as_optional_int(topic)
                    assigned.append(topic_id if topic_id is not None else -1)
                return assigned
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.warning("Failed BERTopic.transform for class diagnostics; falling back to term matching: %s", exc)

        return self._assign_topics_from_terms(texts)

    def _build_v2_evidence_reviews(self, business_id: str | None) -> list[dict[str, Any]]:
        if not business_id:
            return []

        source_df: pd.DataFrame | None = None
        if self._v2_negative_topics_df is not None and not self._v2_negative_topics_df.empty:
            candidate = self._v2_negative_topics_df[self._v2_negative_topics_df["business_id"] == business_id]
            if not candidate.empty:
                source_df = candidate.copy()

        if source_df is None:
            source_df = self._v2_business_reviews(business_id)

        if source_df is None or source_df.empty:
            return []

        scored = source_df.copy()
        scored["stars"] = pd.to_numeric(scored.get("stars"), errors="coerce")
        scored["vader_compound"] = pd.to_numeric(scored.get("vader_compound"), errors="coerce")
        scored = scored.sort_values(["stars", "vader_compound"], ascending=[True, True])

        evidence: list[dict[str, Any]] = []
        for row in scored.itertuples(index=False):
            raw_text = as_optional_str(getattr(row, "text", None)) or ""
            text = re.sub(r"\s+", " ", raw_text).strip()
            snippet = text
            vader = as_optional_float(getattr(row, "vader_compound", None))
            neg_prob = None if vader is None else float(np.clip((1.0 - vader) / 2.0, 0.0, 1.0))
            evidence.append(
                {
                    "review_id": as_optional_str(getattr(row, "review_id", None)),
                    "date": self._format_date(getattr(row, "date", None)),
                    "stars": as_optional_int(getattr(row, "stars", None)),
                    "sentiment_neg_prob": neg_prob,
                    "snippet": snippet,
                }
            )
        return evidence

    def _build_v2_topics_by_month(self, business_id: str | None) -> list[dict[str, Any]]:
        if not business_id:
            return []
        if self._v2_negative_topics_df is None or self._v2_negative_topics_df.empty:
            return []
        if "topic_id" not in self._v2_negative_topics_df.columns:
            return []

        topic_df = self._v2_negative_topics_df[
            self._v2_negative_topics_df["business_id"] == business_id
        ].copy()
        if topic_df.empty:
            return []

        topic_df["date"] = pd.to_datetime(topic_df.get("date"), errors="coerce")
        topic_df["month"] = topic_df["date"].dt.to_period("M").dt.to_timestamp()
        topic_df = topic_df.dropna(subset=["month", "topic_id"])
        if topic_df.empty:
            return []

        grouped = (
            topic_df.groupby(["month", "topic_id"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["month", "count"], ascending=[True, False])
        )

        rows: list[dict[str, Any]] = []
        for month_value, month_topics in grouped.groupby("month"):
            for rank, record in enumerate(month_topics.head(10).itertuples(index=False), start=1):
                topic_id = as_optional_int(record.topic_id)
                if topic_id is None:
                    continue
                terms = sanitize_topic_terms(self._v2_topic_terms_by_id.get(topic_id, []), limit=6)
                theme, _ = self._v2_recommendation_by_topic.get(topic_id, (None, None))
                if not theme:
                    if terms:
                        theme = f"topic_{topic_id}: {', '.join(terms[:6])}"
                    else:
                        theme = f"topic_{topic_id}"
                rows.append(
                    {
                        "month": self._format_month(month_value),
                        "topic_id": topic_id,
                        "topic_terms": ", ".join(terms) if terms else None,
                        "theme": theme,
                        "rank": rank,
                        "strength": float(record.count),
                    }
                )
        return rows

    def _build_v2_topics_per_class(self, business_id: str | None) -> list[dict[str, Any]]:
        if not business_id:
            return []
        reviews_df = self._v2_business_reviews(business_id)
        if reviews_df is None or reviews_df.empty:
            return []

        topic_df = reviews_df.copy()
        if topic_df.empty:
            return []

        topic_df["stars"] = pd.to_numeric(topic_df["stars"], errors="coerce")
        topic_df["text"] = topic_df["text"].fillna("").astype(str)
        topic_df = topic_df.dropna(subset=["stars"])
        topic_df = topic_df[topic_df["text"].str.strip().str.len() > 0]
        if topic_df.empty:
            return []

        topic_df["stars_bucket"] = topic_df["stars"].round().clip(lower=1, upper=5).astype(int)
        topic_df = topic_df[topic_df["stars_bucket"].isin([1, 5])].copy()
        if topic_df.empty:
            return []

        assigned_topics = self._assign_topics_for_reviews(topic_df["text"].tolist())
        if not assigned_topics:
            return []
        topic_df = topic_df.reset_index(drop=True)
        topic_df["topic_id"] = pd.Series(assigned_topics, index=topic_df.index)
        topic_df = topic_df[pd.to_numeric(topic_df["topic_id"], errors="coerce").fillna(-1).astype(int) >= 0].copy()
        if topic_df.empty:
            return []
        topic_df["topic_id"] = pd.to_numeric(topic_df["topic_id"], errors="coerce").astype(int)

        topics_per_class = (
            topic_df.groupby(["stars_bucket", "topic_id"], as_index=False)
            .size()
            .rename(columns={"size": "frequency"})
            .sort_values(["stars_bucket", "frequency"], ascending=[True, False])
        )
        topics_per_class = topics_per_class.groupby("stars_bucket", as_index=False).head(6).reset_index(drop=True)
        if topics_per_class.empty:
            return []

        rows: list[dict[str, Any]] = []
        for stars_bucket, class_rows in topics_per_class.groupby("stars_bucket", sort=True):
            class_rows = class_rows.sort_values("frequency", ascending=False).reset_index(drop=True)
            class_rank = 1
            for record in class_rows.itertuples(index=False):
                topic_id = as_optional_int(getattr(record, "topic_id", None))
                if topic_id is None or stars_bucket is None:
                    continue
                terms = sanitize_topic_terms(self._v2_topic_terms_by_id.get(topic_id, []), limit=6)
                theme = self._topic_theme_label(topic_id) or f"topic_{topic_id}"
                rows.append(
                    {
                        "stars_bucket": int(stars_bucket),
                        "class_label": f"{int(stars_bucket)}★",
                        "rank": class_rank,
                        "topic_id": topic_id,
                        "topic_terms": ", ".join(terms) if terms else None,
                        "theme": theme,
                        "strength": float(getattr(record, "frequency", 0.0)),
                    }
                )
                class_rank += 1
        return rows

    def _topic_theme_label(self, topic_id: int | None) -> str | None:
        if topic_id is None:
            return None
        theme, _ = self._v2_recommendation_by_topic.get(topic_id, (None, None))
        if theme:
            return theme
        terms = sanitize_topic_terms(self._v2_topic_terms_by_id.get(topic_id, []), limit=6)
        if terms:
            return f"topic_{topic_id}: {', '.join(terms[:6])}"
        return f"topic_{topic_id}"

    def _build_v2_component2_diagnostics(self, business_id: str | None) -> dict[str, Any]:
        diagnostics = {
            "negative_review_count": 0,
            "terminal_topics": [],
            "topic_recovery_gaps": [],
        }
        if not business_id:
            return diagnostics

        if self._v2_negative_topics_df is not None and not self._v2_negative_topics_df.empty:
            diagnostics["negative_review_count"] = int(
                len(self._v2_negative_topics_df[self._v2_negative_topics_df["business_id"] == business_id])
            )

        terminal_topics: list[dict[str, Any]] = []
        if self._v2_terminal_topics_df is not None and not self._v2_terminal_topics_df.empty:
            subset = self._v2_terminal_topics_df[
                self._v2_terminal_topics_df["business_id"] == business_id
            ].copy()
            if not subset.empty:
                subset = subset.sort_values(["share", "count"], ascending=[False, False]).head(3)
                for row in subset.itertuples(index=False):
                    topic_id = as_optional_int(getattr(row, "topic_id", None))
                    if topic_id is None:
                        continue
                    theme = self._topic_theme_label(topic_id)
                    recommendation = as_optional_str(
                        self._v2_recommendation_by_topic.get(topic_id, (None, None))[1]
                    )
                    terminal_topics.append(
                        {
                            "topic_id": topic_id,
                            "theme": theme,
                            "share": as_optional_float(getattr(row, "share", None)),
                            "count": as_optional_int(getattr(row, "count", None)),
                            "recommendation": recommendation,
                            "match_score": self._v2_recommendation_match_by_topic.get(topic_id),
                        }
                    )
        diagnostics["terminal_topics"] = terminal_topics

        if terminal_topics and self._v2_recovery_comparison_df is not None and not self._v2_recovery_comparison_df.empty:
            topic_ids = {
                as_optional_int(topic.get("topic_id"))
                for topic in terminal_topics
                if as_optional_int(topic.get("topic_id")) is not None
            }
            comparison_rows: list[dict[str, Any]] = []
            for topic_id in topic_ids:
                if topic_id is None:
                    continue
                subset = self._v2_recovery_comparison_df[
                    self._v2_recovery_comparison_df["topic_id"] == topic_id
                ]
                if subset.empty:
                    continue
                record = subset.iloc[0].to_dict()
                comparison_rows.append(
                    {
                        "topic_id": topic_id,
                        "theme": self._topic_theme_label(topic_id),
                        "closed_minus_open_share": as_optional_float(record.get("closed_minus_open_share")),
                        "closed_after_negative": as_optional_float(record.get("closed_after_negative")),
                        "open_after_negative": as_optional_float(record.get("open_after_negative")),
                    }
                )
            diagnostics["topic_recovery_gaps"] = sorted(
                comparison_rows,
                key=lambda item: abs(as_optional_float(item.get("closed_minus_open_share")) or 0.0),
                reverse=True,
            )[:3]

        return diagnostics

    def _build_v2_resilience_context(self, business_id: str | None, row: dict[str, Any]) -> dict[str, Any]:
        default_payload = {
            "city_context": None,
            "cuisine_context": [],
            "checkin_floor_context": None,
            "recovery_pattern": None,
        }
        if not business_id:
            return default_payload

        cached = self._v2_resilience_cache.get(business_id)
        if cached is not None:
            return cached

        city_value = as_optional_str(row.get("city"))
        state_value = as_optional_str(row.get("state"))
        city_norm = normalize_text(city_value)
        state_norm = normalize_text(state_value)

        city_context: dict[str, Any] | None = None
        if (
            city_norm
            and state_norm
            and self._v2_city_closure_rates_df is not None
            and not self._v2_city_closure_rates_df.empty
        ):
            city_df = self._v2_city_closure_rates_df
            mask = (
                city_df["city"].astype(str).map(normalize_text).eq(city_norm)
                & city_df["state"].astype(str).map(normalize_text).eq(state_norm)
            )
            subset = city_df[mask]
            if not subset.empty:
                record = subset.iloc[0]
                closure_rate = as_optional_float(record.get("closure_rate"))
                all_rates = pd.to_numeric(city_df.get("closure_rate"), errors="coerce").dropna()
                percentile = None
                if closure_rate is not None and not all_rates.empty:
                    percentile = float((all_rates <= closure_rate).mean())
                city_context = {
                    "city": as_optional_str(record.get("city")),
                    "state": as_optional_str(record.get("state")),
                    "businesses": as_optional_int(record.get("businesses")),
                    "closed": as_optional_int(record.get("closed")),
                    "closure_rate": closure_rate,
                    "closure_rate_percentile": percentile,
                }

        business_monthly = None
        if self._v2_monthly_panel_df is not None:
            subset = self._v2_monthly_panel_df[self._v2_monthly_panel_df["business_id"] == business_id]
            if not subset.empty:
                business_monthly = subset.sort_values("month").copy()

        categories: list[str] = []
        if business_monthly is not None and "categories" in business_monthly.columns:
            category_text = as_optional_str(business_monthly.iloc[-1].get("categories"))
            categories = parse_categories(category_text)

        cuisine_context: list[dict[str, Any]] = []
        if (
            categories
            and self._v2_cuisine_closure_rates_df is not None
            and not self._v2_cuisine_closure_rates_df.empty
        ):
            category_keys = {normalize_text(item) for item in categories if normalize_text(item)}
            category_keys = {
                item for item in category_keys if item and item not in GENERIC_CUISINE_CATEGORIES
            }
            cuisine_df = self._v2_cuisine_closure_rates_df.copy()
            cuisine_df["category_key"] = cuisine_df["category"].astype(str).map(normalize_text)
            matched = cuisine_df[cuisine_df["category_key"].isin(category_keys)].copy()
            if not matched.empty:
                matched = matched.sort_values(["closure_rate", "businesses"], ascending=[False, False])
                for record in matched.head(3).itertuples(index=False):
                    cuisine_context.append(
                        {
                            "category": as_optional_str(getattr(record, "category", None)),
                            "businesses": as_optional_int(getattr(record, "businesses", None)),
                            "closed": as_optional_int(getattr(record, "closed", None)),
                            "closure_rate": as_optional_float(getattr(record, "closure_rate", None)),
                        }
                    )

        checkin_floor_context: dict[str, Any] | None = None
        if business_monthly is not None and "checkin_count" in business_monthly.columns:
            latest_checkins = as_optional_float(
                pd.to_numeric(business_monthly["checkin_count"], errors="coerce").fillna(0.0).iloc[-1]
            )
            if (
                latest_checkins is not None
                and self._v2_checkin_floor_df is not None
                and not self._v2_checkin_floor_df.empty
            ):
                floor_df = self._v2_checkin_floor_df.copy()
                floor_df["bin_left"] = pd.to_numeric(floor_df.get("bin_left"), errors="coerce")
                floor_df["bin_right"] = pd.to_numeric(floor_df.get("bin_right"), errors="coerce")
                mask = (latest_checkins > floor_df["bin_left"]) & (latest_checkins <= floor_df["bin_right"])
                matched = floor_df[mask]
                if matched.empty:
                    matched = floor_df.sort_values("bin_right").tail(1)
                floor_row = matched.iloc[0]
                checkin_floor_context = {
                    "latest_checkins": latest_checkins,
                    "bin_label": as_optional_str(floor_row.get("bin_label")),
                    "closure_rate": as_optional_float(floor_row.get("closure_rate")),
                    "closure_rate_delta": as_optional_float(floor_row.get("closure_rate_delta")),
                    "activity_floor": as_optional_float(floor_row.get("activity_floor")),
                }

        recovery_pattern: dict[str, Any] | None = None
        if self._v2_recovery_patterns_df is not None and not self._v2_recovery_patterns_df.empty:
            subset = self._v2_recovery_patterns_df[
                self._v2_recovery_patterns_df["business_id"] == business_id
            ]
            if not subset.empty:
                record = subset.iloc[0].to_dict()
                recovery_pattern = {
                    "had_negative_phase": bool(as_optional_int(record.get("had_negative_phase")) or 0),
                    "recovered_pattern": bool(as_optional_int(record.get("recovered_pattern")) or 0),
                    "sentiment_delta": as_optional_float(record.get("sentiment_delta")),
                    "checkin_delta": as_optional_float(record.get("checkin_delta")),
                    "trough_sentiment": as_optional_float(record.get("trough_sentiment")),
                    "final_sentiment": as_optional_float(record.get("final_sentiment")),
                    "trough_checkins": as_optional_float(record.get("trough_checkins")),
                    "final_checkins": as_optional_float(record.get("final_checkins")),
                }

        payload = {
            "city_context": city_context,
            "cuisine_context": cuisine_context,
            "checkin_floor_context": checkin_floor_context,
            "recovery_pattern": recovery_pattern,
        }
        self._v2_resilience_cache[business_id] = payload
        return payload

    def _score_live_fallback(
        self,
        business_id: str,
        *,
        min_active_months: int | None = None,
        min_reviews_in_window: int | None = None,
    ) -> LiveScoringOutcome:
        if self._runtime_mode != self.V2_RUNTIME_MODE or self._v2_runtime is None:
            return LiveScoringOutcome(False, {}, "v2_runtime_unavailable")
        if not self.settings.yelp_data_dir:
            return LiveScoringOutcome(False, {}, "missing_yelp_data_dir")
        if not self._required_yelp_files_present():
            return LiveScoringOutcome(False, {}, "yelp_files_missing")

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
            data_quality = self._build_data_quality(
                expected_reviews=as_optional_int(identity.get("total_reviews")),
                observed_reviews=as_optional_int(live_identity.get("total_reviews")),
            )

            sentiment_df = self._score_sentiment_for_reviews(business_reviews)
            monthly_df = self._build_live_v2_monthly_panel(
                business_id=business_id,
                identity=live_identity,
            )
            if monthly_df.empty:
                return LiveScoringOutcome(
                    False,
                    {"identity": live_identity, "data_quality": data_quality},
                    "no_monthly_panel_for_business",
                )

            windows = self._build_live_v2_windows(
                monthly_df,
                business_id,
                min_active_months=min_active_months,
                min_reviews_in_window=min_reviews_in_window,
            )
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
                return LiveScoringOutcome(
                    False,
                    {"identity": live_identity, "data_quality": data_quality},
                    reason,
                )

            meta_windows, p_windows = windows
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
                return LiveScoringOutcome(
                    False,
                    {"identity": live_identity, "data_quality": data_quality},
                    reason,
                )

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
            component2_diagnostics = {
                "negative_review_count": None,
                "terminal_topics": [],
                "topic_recovery_gaps": [],
            }
            resilience_context = {
                "city_context": None,
                "cuisine_context": [],
                "checkin_floor_context": None,
                "recovery_pattern": None,
            }
            if self._runtime_mode == self.V2_RUNTIME_MODE:
                component2_diagnostics = self._build_v2_component2_diagnostics(business_id)
                resilience_context = self._build_v2_resilience_context(business_id, live_identity)
                terminal_topics = component2_diagnostics.get("terminal_topics") or []
                if terminal_topics:
                    inferred_themes: list[str] = []
                    inferred_recommendations: list[str] = []
                    for topic in terminal_topics:
                        theme_name = as_optional_str(topic.get("theme"))
                        if theme_name and theme_name not in inferred_themes:
                            inferred_themes.append(theme_name)

                        recommendation_text = as_optional_str(topic.get("recommendation"))
                        if recommendation_text and recommendation_text not in inferred_recommendations:
                            inferred_recommendations.append(recommendation_text)

                    if (not themes_top3 or themes_top3 == ["needs_review"]) and inferred_themes:
                        themes_top3 = inferred_themes[:3]

                    fallback_only = (
                        not recommendations_top3
                        or all(
                            (as_optional_str(recommendation) or "") == RECO_FALLBACK_NOTE
                            for recommendation in recommendations_top3
                        )
                    )
                    if fallback_only and inferred_recommendations:
                        recommendations_top3 = inferred_recommendations[:3]
                        if recommendation_notes and RECO_FALLBACK_NOTE in recommendation_notes:
                            recommendation_notes = None

            payload = {
                "business_id": live_identity.get("business_id"),
                "name": live_identity.get("name"),
                "city": live_identity.get("city"),
                "state": live_identity.get("state"),
                "status": live_identity.get("status") or "Unknown",
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
                "component2_diagnostics": component2_diagnostics,
                "resilience_context": resilience_context,
                "chart_data": self._build_live_chart_data(
                    identity=live_identity,
                    monthly_df=monthly_df,
                    meta_windows=meta_windows,
                    probabilities=p_windows,
                    sentiment_df=sentiment_df,
                ),
                "scoring_mode": "live_v2",
                "availability": "scored",
                "data_quality": data_quality,
                "not_scored_reason": None,
            }
            return LiveScoringOutcome(True, payload)
        except ServiceError as exc:
            return LiveScoringOutcome(False, {}, str(exc))
        except Exception as exc:  # pragma: no cover - data and env dependent
            logger.exception("Live v2 scoring failed")
            return LiveScoringOutcome(False, {}, f"live_v2_failed:{exc.__class__.__name__}")

    def _build_live_v2_monthly_panel(
        self,
        *,
        business_id: str,
        identity: dict[str, Any],
    ) -> pd.DataFrame:
        self._ensure_repo_root_on_path()
        try:
            from model_builder.v2.features import MonthlyFeatureConfig, build_monthly_signal_panel
            from model_builder.v2.io import RestaurantTables
            from model_builder.v2.sentiment import SentimentDependencyError, VaderSentimentScorer
        except Exception as exc:
            raise ServiceError("v2_feature_import_failed") from exc

        live_business = self._load_live_business_df()
        reviews = self._load_live_reviews_df()
        if live_business is None or live_business.empty or reviews is None or reviews.empty:
            return pd.DataFrame()

        business_row = live_business[live_business["business_id"] == business_id]
        if business_row.empty:
            return pd.DataFrame()

        business_value = business_row.iloc[0]
        business_frame = pd.DataFrame(
            [
                {
                    "business_id": business_id,
                    "name": as_optional_str(business_value.get("name")) or "Unknown",
                    "city": as_optional_str(business_value.get("city")) or "Unknown",
                    "state": as_optional_str(business_value.get("state")) or "Unknown",
                    "status": as_optional_str(business_value.get("status")) or "Unknown",
                    "categories": as_optional_str(business_value.get("categories")) or "",
                    "stars": as_optional_float(business_value.get("business_stars")) or 0.0,
                    "review_count": as_optional_float(business_value.get("business_review_count")) or 0.0,
                }
            ]
        )

        review_frame = reviews[reviews["business_id"] == business_id][
            ["review_id", "business_id", "stars", "text", "date"]
        ].copy()
        tip_frame = self._load_live_tips_df()
        tip_frame = tip_frame[tip_frame["business_id"] == business_id][["business_id", "text", "date"]].copy()
        checkin_frame = self._load_live_checkins_df()
        checkin_frame = checkin_frame[checkin_frame["business_id"] == business_id][["business_id", "date"]].copy()

        if self._vader_scorer is None:
            try:
                self._vader_scorer = VaderSentimentScorer()
            except SentimentDependencyError as exc:
                raise ServiceError("vader_unavailable") from exc

        tables = RestaurantTables(
            business=business_frame.reset_index(drop=True),
            review=review_frame.reset_index(drop=True),
            tip=tip_frame.reset_index(drop=True),
            checkin=checkin_frame.reset_index(drop=True),
        )
        artifacts = build_monthly_signal_panel(
            tables,
            config=MonthlyFeatureConfig(),
            sentiment_scorer=self._vader_scorer,
        )
        monthly_panel = artifacts.monthly_panel.copy()
        monthly_panel["business_id"] = monthly_panel["business_id"].astype(str)
        monthly_panel["month"] = pd.to_datetime(monthly_panel["month"], errors="coerce")
        monthly_panel = monthly_panel.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)
        if monthly_panel.empty:
            return monthly_panel

        # Ensure status aligns with live identity.
        monthly_panel["status"] = identity.get("status") or monthly_panel.get("status")
        return monthly_panel

    def _build_live_v2_windows(
        self,
        monthly_panel: pd.DataFrame,
        business_id: str,
        *,
        min_active_months: int | None = None,
        min_reviews_in_window: int | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray] | None:
        self._ensure_repo_root_on_path()
        try:
            from model_builder.v2.features import build_sequence_windows
        except Exception as exc:
            raise ServiceError("v2_sequence_import_failed") from exc

        if self._v2_runtime is None or self._v2_runtime.sequence_config is None:
            return None

        resolved_min_active = (
            self.DEFAULT_MIN_ACTIVE_MONTHS
            if min_active_months is None
            else max(0, int(min_active_months))
        )
        resolved_min_reviews = (
            self.DEFAULT_MIN_REVIEWS_IN_WINDOW
            if min_reviews_in_window is None
            else max(0, int(min_reviews_in_window))
        )
        sequence_config = replace(
            self._v2_runtime.sequence_config,
            min_active_months=resolved_min_active,
            min_reviews_in_window=resolved_min_reviews,
        )

        windows = build_sequence_windows(monthly_panel, config=sequence_config)
        if windows.X.size == 0 or windows.meta.empty:
            return None

        meta = windows.meta.copy()
        meta["business_id"] = meta["business_id"].astype(str)
        business_mask = meta["business_id"] == business_id
        if not business_mask.any():
            return None

        x_business = windows.X[business_mask.to_numpy()]
        meta_business = meta.loc[business_mask].reset_index(drop=True)

        model = self._v2_runtime._load_gru_model()
        if model is None:
            raise ServiceError("v2_gru_model_unavailable")
        if not self._is_tensorflow_runtime_available():
            raise ServiceError("tensorflow_runtime_unavailable")

        try:
            probabilities = model.predict(x_business, batch_size=512, verbose=0).reshape(-1)
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise ServiceError(f"v2_gru_scoring_failed:{exc.__class__.__name__}") from exc

        return meta_business, np.asarray(probabilities, dtype=float)

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

    def _load_live_tips_df(self) -> pd.DataFrame:
        if self._live_tips_df is not None:
            return self._live_tips_df

        empty = pd.DataFrame(columns=["business_id", "text", "date"])
        if not self.settings.yelp_data_dir:
            self._live_tips_df = empty
            return self._live_tips_df

        tip_path = self.settings.yelp_data_dir / "yelp_academic_dataset_tip.json"
        if not tip_path.exists():
            self._live_tips_df = empty
            return self._live_tips_df

        try:
            tip_df = load_json_auto(tip_path)
        except Exception as exc:  # pragma: no cover - data-dependent
            logger.warning("Failed loading live tip data: %s", exc)
            self._live_tips_df = empty
            return self._live_tips_df

        required = {"business_id", "text", "date"}
        missing = required - set(tip_df.columns)
        if missing:
            logger.warning("Live tip data missing columns: %s", sorted(missing))
            self._live_tips_df = empty
            return self._live_tips_df

        tip_df = tip_df.copy()
        tip_df["business_id"] = tip_df["business_id"].astype(str)
        tip_df["date"] = pd.to_datetime(tip_df["date"], errors="coerce")
        tip_df = tip_df.dropna(subset=["date"]).reset_index(drop=True)
        self._live_tips_df = tip_df[["business_id", "text", "date"]].copy()
        return self._live_tips_df

    def _load_live_checkins_df(self) -> pd.DataFrame:
        if self._live_checkins_df is not None:
            return self._live_checkins_df

        empty = pd.DataFrame(columns=["business_id", "date"])
        if not self.settings.yelp_data_dir:
            self._live_checkins_df = empty
            return self._live_checkins_df

        checkin_path = self.settings.yelp_data_dir / "yelp_academic_dataset_checkin.json"
        if not checkin_path.exists():
            self._live_checkins_df = empty
            return self._live_checkins_df

        try:
            checkin_df = load_json_auto(checkin_path)
        except Exception as exc:  # pragma: no cover - data-dependent
            logger.warning("Failed loading live checkin data: %s", exc)
            self._live_checkins_df = empty
            return self._live_checkins_df

        required = {"business_id", "date"}
        missing = required - set(checkin_df.columns)
        if missing:
            logger.warning("Live checkin data missing columns: %s", sorted(missing))
            self._live_checkins_df = empty
            return self._live_checkins_df

        checkin_df = checkin_df.copy()
        checkin_df["business_id"] = checkin_df["business_id"].astype(str)
        checkin_df["date"] = checkin_df["date"].astype(str)
        self._live_checkins_df = checkin_df[["business_id", "date"]].copy()
        return self._live_checkins_df

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
        self._ensure_repo_root_on_path()
        try:
            from model_builder.v2.sentiment import SentimentDependencyError, VaderSentimentScorer, add_vader_compound
        except Exception as exc:
            raise ServiceError("v2_sentiment_import_failed") from exc

        if self._vader_scorer is None:
            try:
                self._vader_scorer = VaderSentimentScorer()
            except SentimentDependencyError as exc:
                raise ServiceError("vader_unavailable") from exc

        scored = add_vader_compound(
            reviews_df,
            text_column="text",
            output_column="vader_compound",
            scorer=self._vader_scorer,
        ).copy()

        scored["p_neg"] = pd.to_numeric((1.0 - scored["vader_compound"]) / 2.0, errors="coerce").clip(0.0, 1.0)
        scored["p_pos"] = (1.0 - scored["p_neg"]).clip(0.0, 1.0)
        return scored

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
        *,
        min_active_months: int | None = None,
        min_reviews_in_window: int | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame] | None:
        if monthly_df.empty:
            return None

        seq_len = 12
        horizon_months = 6
        inactive_k = 12
        resolved_min_active_months = (
            self.DEFAULT_MIN_ACTIVE_MONTHS
            if min_active_months is None
            else max(0, int(min_active_months))
        )
        resolved_min_reviews = (
            self.DEFAULT_MIN_REVIEWS_IN_WINDOW
            if min_reviews_in_window is None
            else max(0, int(min_reviews_in_window))
        )
        resolved_min_active_months = min(seq_len, resolved_min_active_months)

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
            if active_months < resolved_min_active_months:
                continue
            if total_reviews < resolved_min_reviews:
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
                    cleaned_terms = sanitize_topic_terms(terms, limit=12)
                    if not cleaned_terms:
                        continue
                    score = 0.0
                    for term in cleaned_terms:
                        if " " in term:
                            score += float(month_text.count(term))
                        else:
                            score += float(len(re.findall(rf"\b{re.escape(term)}\b", month_text)))
                    if score > 0:
                        topic_scores.append((score, topic_id, cleaned_terms))

                topic_scores.sort(key=lambda item: item[0], reverse=True)
                for rank, (score, topic_id, terms) in enumerate(topic_scores[:10], start=1):
                    topic_rows.append(
                        {
                            "month": month_str,
                            "topic_id": topic_id,
                            "topic_terms": ", ".join(terms[:6]),
                            "theme": f"topic_{topic_id}: {', '.join(terms[:6])}",
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

            for rank, theme in enumerate(themes[:10], start=1):
                topic_rows.append(
                    {
                        "month": month_str,
                        "topic_id": None,
                        "topic_terms": None,
                        "theme": theme,
                        "rank": rank,
                        "strength": float(max(1, 11 - rank)),
                    }
                )

        return topic_rows

    def _extract_evidence_reviews(self, sentiment_df: pd.DataFrame) -> list[dict[str, Any]]:
        ranked = sentiment_df.copy()
        ranked["stars"] = pd.to_numeric(ranked["stars"], errors="coerce")
        ranked = ranked.sort_values(["p_neg", "stars"], ascending=[False, True])

        evidence: list[dict[str, Any]] = []
        for row in ranked.itertuples():
            raw_text = as_optional_str(getattr(row, "text", None)) or ""
            text = re.sub(r"\s+", " ", raw_text).strip()
            snippet = text
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
                normalized = normalize_topic_term(str(vocab[term_idx]))
                if not normalized:
                    continue
                if is_stop_word_term(normalized):
                    continue
                if normalized not in keywords:
                    keywords.append(normalized)

        return keywords[:20]

    def _extract_keywords_simple(self, texts: list[str]) -> list[str]:
        token_pattern = re.compile(r"[a-z]{3,}")
        stop_words = set(TOPIC_STOP_WORDS)
        stop_words.update({"restaurant", "place", "food", "service"})

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
        data_quality: dict[str, Any] | None = None,
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
            "component2_diagnostics": {
                "negative_review_count": None,
                "terminal_topics": [],
                "topic_recovery_gaps": [],
            },
            "resilience_context": {
                "city_context": None,
                "cuisine_context": [],
                "checkin_floor_context": None,
                "recovery_pattern": None,
            },
            "chart_data": {
                "ratings_by_month": [],
                "rating_bucket_counts_by_month": [],
                "predicted_close_by_month": [],
                "topics_by_month": [],
                "actual_close_month": None,
            },
            "scoring_mode": scoring_mode,
            "availability": "not_scored_yet",
            "data_quality": data_quality
            or {
                "expected_reviews": None,
                "observed_reviews": None,
                "coverage_ratio": None,
                "has_mismatch": False,
            },
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
