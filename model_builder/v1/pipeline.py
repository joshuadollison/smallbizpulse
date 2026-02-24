from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .gru_training import GruTrainingArtifacts, GruTrainingConfig, train_gru_end_to_end
from .sentiment_training import (
    DistilBertSentimentTrainer,
    SentimentTrainingArtifacts,
    SentimentTrainingConfig,
    aggregate_business_month_sentiment,
)


@dataclass
class ModelBuilderV1:
    """High-level API that orchestrates sentiment + GRU model building."""

    sentiment_config: SentimentTrainingConfig = field(default_factory=SentimentTrainingConfig)
    gru_config: GruTrainingConfig = field(default_factory=GruTrainingConfig)

    def __post_init__(self) -> None:
        self._sentiment_trainer = DistilBertSentimentTrainer(self.sentiment_config)

    def train_sentiment_model(
        self,
        reviews: pd.DataFrame,
        *,
        output_dir: Path | str,
    ) -> SentimentTrainingArtifacts:
        return self._sentiment_trainer.train(reviews, output_dir=output_dir)

    def score_reviews_with_sentiment(
        self,
        reviews: pd.DataFrame,
        *,
        model_dir: Path | str,
        temperature: float | None = None,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        return self._sentiment_trainer.score_reviews(
            reviews,
            model_dir=model_dir,
            temperature=temperature,
            batch_size=batch_size,
        )

    def build_monthly_panel_from_scored_reviews(
        self,
        scored_reviews: pd.DataFrame,
    ) -> pd.DataFrame:
        return aggregate_business_month_sentiment(
            scored_reviews,
            min_reviews_for_share=self.sentiment_config.min_reviews_for_share,
        )

    def train_gru_model(
        self,
        monthly_panel: pd.DataFrame,
        *,
        output_dir: Path | str,
        playbook: Iterable[Any] | None = None,
    ) -> GruTrainingArtifacts:
        return train_gru_end_to_end(
            monthly_panel,
            output_dir=output_dir,
            config=self.gru_config,
            playbook=playbook,
        )

    def run_full_pipeline(
        self,
        reviews: pd.DataFrame,
        *,
        sentiment_output_dir: Path | str,
        gru_output_dir: Path | str,
        playbook: Iterable[Any] | None = None,
    ) -> dict[str, Any]:
        """Train sentiment model, aggregate monthly features, then train GRU model."""
        sentiment_artifacts = self.train_sentiment_model(reviews, output_dir=sentiment_output_dir)

        scored_reviews = self.score_reviews_with_sentiment(
            reviews,
            model_dir=sentiment_artifacts.model_dir,
            temperature=sentiment_artifacts.best_temperature,
        )
        monthly_panel = self.build_monthly_panel_from_scored_reviews(scored_reviews)

        gru_artifacts = self.train_gru_model(
            monthly_panel,
            output_dir=gru_output_dir,
            playbook=playbook,
        )

        return {
            "sentiment_artifacts": sentiment_artifacts,
            "scored_reviews": scored_reviews,
            "monthly_panel": monthly_panel,
            "gru_artifacts": gru_artifacts,
        }
