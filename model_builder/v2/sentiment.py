from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


class SentimentDependencyError(RuntimeError):
    """Raised when a required sentiment dependency is unavailable."""


def _build_vader_analyzer() -> object:
    """
    Build a VADER analyzer.

    Preference order:
    1) vaderSentiment package
    2) nltk.sentiment.vader (auto-download lexicon if needed)
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except Exception:
        pass

    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        return SentimentIntensityAnalyzer()
    except Exception as exc:  # pragma: no cover - dependency guard
        raise SentimentDependencyError(
            "VADER analyzer is unavailable. Install `vaderSentiment` or `nltk` with vader_lexicon."
        ) from exc


@dataclass
class VaderSentimentScorer:
    """Simple wrapper around a VADER compound sentiment analyzer."""

    analyzer: object | None = None

    def __post_init__(self) -> None:
        if self.analyzer is None:
            self.analyzer = _build_vader_analyzer()

    def score_text(self, text: str) -> float:
        payload = self.analyzer.polarity_scores(text or "")
        return float(payload.get("compound", 0.0))

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self.score_text(str(text)) for text in texts], dtype=float)



def add_vader_compound(
    reviews: pd.DataFrame,
    *,
    text_column: str = "text",
    output_column: str = "vader_compound",
    scorer: VaderSentimentScorer | None = None,
) -> pd.DataFrame:
    """Attach VADER compound scores to review rows."""
    if text_column not in reviews.columns:
        raise ValueError(f"reviews missing required text column: {text_column}")

    local_scorer = scorer or VaderSentimentScorer()

    frame = reviews.copy()
    frame[text_column] = frame[text_column].fillna("").astype(str)
    frame[output_column] = local_scorer.score_texts(frame[text_column].tolist())

    return frame
