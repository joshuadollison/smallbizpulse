from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    MonthlyFeatureConfig,
    SequenceWindowConfig,
    build_monthly_signal_panel,
    build_sequence_windows,
    compute_business_snapshot_features,
)
from .io import RestaurantTables, load_restaurant_tables
from .sentiment import VaderSentimentScorer
from .survival import RuleBasedRiskModel


DEFAULT_RISK_BINS = (0.0, 0.50, 0.65, 0.75, 0.85, 1.0)
DEFAULT_RISK_LABELS = ("low", "medium", "elevated", "high", "very_high")


class InferenceDependencyError(RuntimeError):
    """Raised when runtime inference dependencies are unavailable."""


@dataclass(frozen=True)
class SurvivalRuntimeArtifacts:
    output_dir: Path
    baseline_bundle_path: Path
    rule_model_path: Path
    gru_model_path: Path | None
    gru_metadata_path: Path | None


@dataclass
class SurvivalRuntime:
    """Live inference service for Component 1 models."""

    artifacts: SurvivalRuntimeArtifacts
    baseline_bundle: dict[str, Any]
    rule_model: RuleBasedRiskModel
    sequence_config: SequenceWindowConfig | None = None
    gru_threshold: float | None = None
    gru_recent_k_windows: int = 3

    _gru_model: Any | None = None

    @classmethod
    def from_output_dir(cls, output_dir: Path | str) -> "SurvivalRuntime":
        try:
            import joblib
        except Exception as exc:  # pragma: no cover
            raise InferenceDependencyError("joblib is required for loading baseline models") from exc

        root = Path(output_dir)

        baseline_bundle_path = root / "baseline_models.joblib"
        rule_model_path = root / "rule_model.json"
        gru_model_path = root / "gru_survival_model.keras"
        gru_metadata_path = root / "gru_metadata.json"

        if not baseline_bundle_path.exists():
            raise FileNotFoundError(f"missing baseline model bundle: {baseline_bundle_path}")
        if not rule_model_path.exists():
            raise FileNotFoundError(f"missing rule model file: {rule_model_path}")

        baseline_bundle = joblib.load(baseline_bundle_path)

        rule_payload = json.loads(rule_model_path.read_text(encoding="utf-8"))
        rule_model = RuleBasedRiskModel(
            checkin_floor=float(rule_payload["checkin_floor"]),
            star_floor=float(rule_payload["star_floor"]),
            velocity_floor=float(rule_payload["velocity_floor"]),
        )

        sequence_config: SequenceWindowConfig | None = None
        gru_threshold: float | None = None
        gru_recent_k_windows = 3
        if gru_metadata_path.exists():
            payload = json.loads(gru_metadata_path.read_text(encoding="utf-8"))
            sequence_config = SequenceWindowConfig(
                seq_len=int(payload.get("seq_len", 12)),
                horizon_months=int(payload.get("horizon_months", 6)),
                min_total_reviews_for_sequence=int(payload.get("min_total_reviews_for_sequence", 10)),
                min_active_months=int(payload.get("min_active_months", 6)),
                min_reviews_in_window=int(payload.get("min_reviews_in_window", 10)),
                inactive_months_for_zombie=int(payload.get("inactive_months_for_zombie", 12)),
            )
            threshold_value = payload.get("threshold")
            if threshold_value is not None:
                gru_threshold = float(threshold_value)
            gru_recent_k_windows = int(payload.get("gru_recent_k_windows", 3))

        artifacts = SurvivalRuntimeArtifacts(
            output_dir=root,
            baseline_bundle_path=baseline_bundle_path,
            rule_model_path=rule_model_path,
            gru_model_path=gru_model_path if gru_model_path.exists() else None,
            gru_metadata_path=gru_metadata_path if gru_metadata_path.exists() else None,
        )

        return cls(
            artifacts=artifacts,
            baseline_bundle=baseline_bundle,
            rule_model=rule_model,
            sequence_config=sequence_config,
            gru_threshold=gru_threshold,
            gru_recent_k_windows=gru_recent_k_windows,
        )

    def _load_gru_model(self) -> Any | None:
        if self.artifacts.gru_model_path is None:
            return None
        if self._gru_model is not None:
            return self._gru_model

        try:
            from tensorflow import keras
        except Exception:
            return None

        self._gru_model = keras.models.load_model(self.artifacts.gru_model_path)
        return self._gru_model

    def _assign_risk_bucket(self, score: float | None) -> str | None:
        if score is None:
            return None

        bins = list(DEFAULT_RISK_BINS)
        labels = list(DEFAULT_RISK_LABELS)

        if score <= bins[0]:
            return labels[0]
        if score >= bins[-1]:
            return labels[-1]

        for idx, label in enumerate(labels):
            left = bins[idx]
            right = bins[idx + 1]
            if idx == 0 and left <= score <= right:
                return label
            if idx > 0 and left < score <= right:
                return label

        return labels[-1]

    def _baseline_scores(self, snapshot_features: pd.DataFrame) -> pd.DataFrame:
        models = self.baseline_bundle.get("models", {})
        feature_columns = self.baseline_bundle.get("feature_columns", [])
        thresholds = self.baseline_bundle.get("thresholds", {})

        if not models or not feature_columns:
            raise ValueError("baseline bundle missing models or feature columns")

        frame = snapshot_features.copy()
        X = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

        model_names = sorted(models.keys())
        preferred = "gradient_boosting" if "gradient_boosting" in models else model_names[0]

        frame["baseline_model"] = preferred
        frame["baseline_score"] = models[preferred].predict_proba(X)[:, 1]
        frame["baseline_threshold"] = float(thresholds.get(preferred, 0.5))
        frame["baseline_flagged"] = (frame["baseline_score"] >= frame["baseline_threshold"]).astype(int)

        return frame

    def _rule_scores(self, snapshot_features: pd.DataFrame) -> pd.Series:
        scores = self.rule_model.score_frame(snapshot_features)
        return pd.Series(scores, index=snapshot_features.index, dtype=float)

    def _gru_scores(self, monthly_panel: pd.DataFrame) -> pd.DataFrame:
        if self.sequence_config is None:
            return pd.DataFrame(columns=["business_id", "gru_score", "gru_window_count"])

        model = self._load_gru_model()
        if model is None:
            return pd.DataFrame(columns=["business_id", "gru_score", "gru_window_count"])

        windows = build_sequence_windows(monthly_panel, config=self.sequence_config)
        if windows.X.size == 0 or windows.meta.empty:
            return pd.DataFrame(columns=["business_id", "gru_score", "gru_window_count"])

        probabilities = model.predict(windows.X, batch_size=512, verbose=0).reshape(-1)

        frame = windows.meta.copy()
        frame["p_closed"] = probabilities
        frame = frame.sort_values(["business_id", "end_month"]).reset_index(drop=True)

        recent_max = (
            frame.groupby("business_id", group_keys=False)
            .tail(max(1, int(self.gru_recent_k_windows)))
            .groupby("business_id", as_index=False)
            .agg(
                gru_score=("p_closed", "max"),
                gru_window_count=("p_closed", "size"),
            )
        )
        return recent_max

    def score_restaurant_tables(
        self,
        tables: RestaurantTables,
        *,
        monthly_config: MonthlyFeatureConfig | None = None,
        sentiment_scorer: VaderSentimentScorer | None = None,
    ) -> pd.DataFrame:
        """
        Build monthly features live from raw restaurant tables and score directly.

        This keeps runtime inference model-driven and avoids any dependency on
        precomputed risk artifact tables.
        """
        panel = build_monthly_signal_panel(
            tables,
            config=monthly_config,
            sentiment_scorer=sentiment_scorer,
        ).monthly_panel
        return self.score_monthly_panel(panel)

    def score_data_root(
        self,
        data_root: Path | str,
        *,
        monthly_config: MonthlyFeatureConfig | None = None,
        sentiment_scorer: VaderSentimentScorer | None = None,
    ) -> pd.DataFrame:
        """Convenience wrapper to score directly from a Yelp dataset folder."""
        tables = load_restaurant_tables(data_root)
        return self.score_restaurant_tables(
            tables,
            monthly_config=monthly_config,
            sentiment_scorer=sentiment_scorer,
        )

    def score_monthly_panel(self, monthly_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Live scoring from model files and raw monthly panel.

        No dependency on precomputed risk artifacts.
        """
        snapshot = compute_business_snapshot_features(monthly_panel)
        if snapshot.empty:
            return pd.DataFrame(
                columns=[
                    "business_id",
                    "name",
                    "city",
                    "state",
                    "status",
                    "risk_score",
                    "risk_bucket",
                    "risk_source",
                    "baseline_score",
                    "rule_score",
                    "gru_score",
                    "total_reviews",
                ]
            )

        baseline = self._baseline_scores(snapshot)
        baseline["rule_score"] = self._rule_scores(baseline)

        low_data_mask = pd.to_numeric(baseline["total_reviews"], errors="coerce").fillna(0.0) <= 5
        baseline["risk_score"] = baseline["baseline_score"].astype(float)
        baseline.loc[low_data_mask, "risk_score"] = baseline.loc[low_data_mask, "rule_score"].astype(float)
        baseline["risk_source"] = np.where(low_data_mask, "rule", "baseline")

        gru_scores = self._gru_scores(monthly_panel)
        if not gru_scores.empty:
            baseline = baseline.merge(gru_scores, on="business_id", how="left")
            has_gru = baseline["gru_score"].notna()
            baseline.loc[has_gru, "risk_score"] = baseline.loc[has_gru, "gru_score"].astype(float)
            baseline.loc[has_gru, "risk_source"] = "gru"
        else:
            baseline["gru_score"] = np.nan

        baseline["risk_bucket"] = baseline["risk_score"].apply(self._assign_risk_bucket)

        return baseline[
            [
                "business_id",
                "name",
                "city",
                "state",
                "status",
                "risk_score",
                "risk_bucket",
                "risk_source",
                "baseline_score",
                "rule_score",
                "gru_score",
                "total_reviews",
            ]
        ].sort_values("risk_score", ascending=False).reset_index(drop=True)
