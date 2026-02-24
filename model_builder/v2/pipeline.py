from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .features import MonthlyFeatureConfig, build_monthly_signal_panel
from .io import RestaurantTables, load_restaurant_tables, summarize_tables
from .recommendation_engine import RecommendationArtifacts, build_recommendation_artifacts
from .resilience import ResilienceArtifacts, build_resilience_artifacts
from .sentiment import VaderSentimentScorer, add_vader_compound
from .survival import SurvivalTrainingArtifacts, SurvivalTrainingConfig, train_survival_models
from .topic_modeling import TopicModelArtifacts, TopicModelConfig, train_diagnostic_topic_model


@dataclass(frozen=True)
class ModelBuilderV2Config:
    """Top-level configuration for methodology-aligned model building."""

    output_root: Path = Path("models/v2_artifacts")
    run_topic_model: bool = True
    run_recommendation_mapping: bool = True
    run_resilience_analysis: bool = True

    monthly: MonthlyFeatureConfig = field(default_factory=MonthlyFeatureConfig)
    survival: SurvivalTrainingConfig = field(default_factory=SurvivalTrainingConfig)
    topic: TopicModelConfig = field(default_factory=TopicModelConfig)


@dataclass(frozen=True)
class ModelBuilderV2Artifacts:
    """Output contract for the complete v2 model-building run."""

    output_root: Path
    monthly_panel_path: Path
    scored_reviews_path: Path
    run_summary_path: Path
    survival: SurvivalTrainingArtifacts
    topic_model: TopicModelArtifacts | None
    recommendations: RecommendationArtifacts | None
    resilience: ResilienceArtifacts | None


class ModelBuilderV2:
    """Orchestrates Components 1-4 from `docs/methodology.md`."""

    def __init__(self, config: ModelBuilderV2Config | None = None) -> None:
        self.config = config or ModelBuilderV2Config()

    def _ensure_output_dirs(self) -> dict[str, Path]:
        root = Path(self.config.output_root)
        root.mkdir(parents=True, exist_ok=True)

        dirs = {
            "root": root,
            "component1": root / "component1_survival",
            "component2": root / "component2_topics",
            "component3": root / "component3_recommendations",
            "component4": root / "component4_resilience",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs

    def _resolve_flagged_businesses(
        self,
        snapshot_features: pd.DataFrame,
        survival_artifacts: SurvivalTrainingArtifacts,
    ) -> list[str]:
        """
        Derive flagged businesses from the strongest baseline model.

        Uses gradient boosting by default, then falls back to the first available model.
        """
        try:
            import joblib
        except Exception:
            return []

        bundle = joblib.load(survival_artifacts.baseline_model_path)
        models = bundle.get("models", {})
        feature_columns = bundle.get("feature_columns", [])
        thresholds = bundle.get("thresholds", {})

        if not models or not feature_columns:
            return []

        preferred_name = "gradient_boosting"
        model_name = preferred_name if preferred_name in models else next(iter(models.keys()))
        model = models[model_name]
        threshold = float(thresholds.get(model_name, 0.5))

        frame = snapshot_features.copy()
        X = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        probabilities = model.predict_proba(X)[:, 1]

        flagged = frame.loc[probabilities >= threshold, "business_id"].astype(str).tolist()
        return flagged

    def run(
        self,
        *,
        data_root: Path | str,
        flagged_business_ids: Iterable[str] | None = None,
    ) -> ModelBuilderV2Artifacts:
        dirs = self._ensure_output_dirs()

        tables: RestaurantTables = load_restaurant_tables(data_root)
        table_summary = summarize_tables(tables)

        sentiment_scorer = VaderSentimentScorer()
        scored_reviews = add_vader_compound(tables.review, scorer=sentiment_scorer)

        monthly_artifacts = build_monthly_signal_panel(
            tables,
            config=self.config.monthly,
            sentiment_scorer=sentiment_scorer,
        )

        monthly_panel_path = dirs["root"] / "monthly_signal_panel.csv"
        scored_reviews_path = dirs["root"] / "reviews_with_vader.csv"

        monthly_artifacts.monthly_panel.to_csv(monthly_panel_path, index=False)
        scored_reviews.to_csv(scored_reviews_path, index=False)

        survival_artifacts = train_survival_models(
            monthly_artifacts.monthly_panel,
            output_dir=dirs["component1"],
            config=self.config.survival,
        )

        snapshot_features = pd.read_csv(survival_artifacts.snapshot_features_path)

        resolved_flagged_ids = list(flagged_business_ids) if flagged_business_ids is not None else []
        if not resolved_flagged_ids:
            resolved_flagged_ids = self._resolve_flagged_businesses(snapshot_features, survival_artifacts)

        topic_artifacts: TopicModelArtifacts | None = None
        recommendation_artifacts: RecommendationArtifacts | None = None
        resilience_artifacts: ResilienceArtifacts | None = None

        if self.config.run_topic_model:
            topic_artifacts = train_diagnostic_topic_model(
                scored_reviews,
                output_dir=dirs["component2"],
                config=self.config.topic,
                flagged_business_ids=resolved_flagged_ids,
            )

            if self.config.run_recommendation_mapping:
                topic_terms_df = pd.read_csv(topic_artifacts.topic_terms_path)
                recommendation_artifacts = build_recommendation_artifacts(
                    topic_terms_df,
                    output_dir=dirs["component3"],
                )

        if self.config.run_resilience_analysis:
            resilience_artifacts = build_resilience_artifacts(
                business_df=tables.business,
                snapshot_features=snapshot_features,
                monthly_panel=monthly_artifacts.monthly_panel,
                output_dir=dirs["component4"],
            )

        run_summary = {
            "table_summary": table_summary,
            "monthly_panel_rows": int(len(monthly_artifacts.monthly_panel)),
            "scored_reviews_rows": int(len(scored_reviews)),
            "flagged_business_count": int(len(resolved_flagged_ids)),
            "component1": {
                "baseline_models": sorted(survival_artifacts.baseline_results.keys()),
                "rule_metrics": survival_artifacts.rule_metrics,
                "gru_trained": survival_artifacts.gru_model_path is not None,
            },
            "component2": {
                "ran": topic_artifacts is not None,
                "output_dir": str(topic_artifacts.output_dir) if topic_artifacts else None,
            },
            "component3": {
                "ran": recommendation_artifacts is not None,
                "output_dir": str(recommendation_artifacts.output_dir) if recommendation_artifacts else None,
            },
            "component4": {
                "ran": resilience_artifacts is not None,
                "output_dir": str(resilience_artifacts.output_dir) if resilience_artifacts else None,
            },
        }

        run_summary_path = dirs["root"] / "run_summary.json"
        run_summary_path.write_text(json.dumps(run_summary, indent=2, default=str), encoding="utf-8")

        return ModelBuilderV2Artifacts(
            output_root=dirs["root"],
            monthly_panel_path=monthly_panel_path,
            scored_reviews_path=scored_reviews_path,
            run_summary_path=run_summary_path,
            survival=survival_artifacts,
            topic_model=topic_artifacts,
            recommendations=recommendation_artifacts,
            resilience=resilience_artifacts,
        )
