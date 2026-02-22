from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class TopicModelDependencyError(RuntimeError):
    """Raised when BERTopic dependencies are unavailable."""


@dataclass(frozen=True)
class TopicModelConfig:
    """Training controls for Component 2 diagnostic BERTopic modeling."""

    negative_star_threshold: float = 2.0
    negative_vader_threshold: float = -0.05
    min_documents: int = 100
    min_topic_size: int = 20
    terminal_review_count: int = 10
    nr_topics: str | int = "auto"


@dataclass(frozen=True)
class TopicModelArtifacts:
    """Persisted outputs for diagnostic topic modeling."""

    output_dir: Path
    topic_terms_path: Path
    document_topics_path: Path
    terminal_topics_path: Path
    recovery_comparison_path: Path
    model_path: Path | None



def _require_bertopic() -> Any:
    try:
        from bertopic import BERTopic

        return BERTopic
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise TopicModelDependencyError(
            "BERTopic is required for Component 2. Install `bertopic` before training."
        ) from exc



def filter_negative_reviews(
    reviews: pd.DataFrame,
    *,
    config: TopicModelConfig | None = None,
) -> pd.DataFrame:
    """Apply methodology-defined negative review filter."""
    cfg = config or TopicModelConfig()

    required = {"business_id", "date", "status", "stars", "vader_compound", "text"}
    missing = sorted(required - set(reviews.columns))
    if missing:
        raise ValueError(f"reviews missing required columns for topic filtering: {missing}")

    frame = reviews.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["stars"] = pd.to_numeric(frame["stars"], errors="coerce")
    frame["vader_compound"] = pd.to_numeric(frame["vader_compound"], errors="coerce")
    frame["text"] = frame["text"].fillna("").astype(str)

    frame = frame.dropna(subset=["date", "stars", "vader_compound"]).copy()

    is_negative = (
        (frame["stars"] <= float(cfg.negative_star_threshold))
        & (frame["vader_compound"] <= float(cfg.negative_vader_threshold))
    )

    return frame.loc[is_negative].sort_values(["business_id", "date"]).reset_index(drop=True)



def _build_topic_terms_table(topic_model: Any) -> pd.DataFrame:
    topic_rows: list[dict[str, Any]] = []

    topics = topic_model.get_topics()
    for topic_id, term_payload in topics.items():
        if topic_id == -1:
            continue
        if not term_payload:
            continue

        terms = [term for term, _ in term_payload[:20]]
        top_weights = [float(weight) for _, weight in term_payload[:20]]
        topic_rows.append(
            {
                "topic_id": int(topic_id),
                "top_terms": ", ".join(terms),
                "top_weights": ", ".join(f"{value:.6f}" for value in top_weights),
            }
        )

    return pd.DataFrame(topic_rows).sort_values("topic_id").reset_index(drop=True)



def _build_terminal_topics_table(
    negative_df: pd.DataFrame,
    *,
    assigned_topics: np.ndarray,
    terminal_review_count: int,
) -> pd.DataFrame:
    frame = negative_df.copy()
    frame["topic_id"] = assigned_topics.astype(int)

    terminal = (
        frame.sort_values(["business_id", "date"])
        .groupby("business_id", group_keys=False)
        .tail(int(terminal_review_count))
        .copy()
    )

    if terminal.empty:
        return pd.DataFrame(columns=["business_id", "status", "topic_id", "count", "share"])

    grouped = (
        terminal.groupby(["business_id", "status", "topic_id"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    grouped["share"] = grouped["count"] / grouped.groupby("business_id")["count"].transform("sum")
    return grouped.sort_values(["business_id", "count"], ascending=[True, False]).reset_index(drop=True)



def _build_recovery_comparison_table(
    negative_df: pd.DataFrame,
    *,
    assigned_topics: np.ndarray,
) -> pd.DataFrame:
    frame = negative_df.copy()
    frame["topic_id"] = assigned_topics.astype(int)

    business_status = frame.groupby("business_id", as_index=False)["status"].last()
    business_status["cohort"] = np.where(
        business_status["status"].eq("Closed"),
        "closed_after_negative",
        "open_after_negative",
    )

    frame = frame.merge(business_status[["business_id", "cohort"]], on="business_id", how="left")

    grouped = (
        frame.groupby(["cohort", "topic_id"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    grouped["share"] = grouped["count"] / grouped.groupby("cohort")["count"].transform("sum")

    pivot = grouped.pivot_table(
        index="topic_id",
        columns="cohort",
        values="share",
        fill_value=0.0,
    ).reset_index()

    if "closed_after_negative" not in pivot.columns:
        pivot["closed_after_negative"] = 0.0
    if "open_after_negative" not in pivot.columns:
        pivot["open_after_negative"] = 0.0

    pivot["closed_minus_open_share"] = (
        pd.to_numeric(pivot["closed_after_negative"], errors="coerce").fillna(0.0)
        - pd.to_numeric(pivot["open_after_negative"], errors="coerce").fillna(0.0)
    )

    return pivot.sort_values("closed_minus_open_share", ascending=False).reset_index(drop=True)



def train_diagnostic_topic_model(
    reviews: pd.DataFrame,
    *,
    output_dir: Path | str,
    config: TopicModelConfig | None = None,
    flagged_business_ids: Iterable[str] | None = None,
) -> TopicModelArtifacts:
    """
    Train BERTopic diagnostics for negative reviews of flagged businesses.

    Methodology constraints applied:
    - Negative definition: stars <= 2 AND vader_compound <= -0.05.
    - Terminal analysis: final 10 reviews before a business's last observed date.
    - Recovery comparison: open-after-negative vs closed-after-negative cohorts.
    """
    cfg = config or TopicModelConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    negative_df = filter_negative_reviews(reviews, config=cfg)

    if flagged_business_ids is not None:
        flagged_set = {str(business_id) for business_id in flagged_business_ids}
        negative_df = negative_df[negative_df["business_id"].astype(str).isin(flagged_set)].copy()

    negative_df = negative_df.dropna(subset=["text"]).copy()
    negative_df["text"] = negative_df["text"].astype(str).str.strip()
    negative_df = negative_df[negative_df["text"].str.len() > 0].copy()

    if len(negative_df) < int(cfg.min_documents):
        raise ValueError(
            f"insufficient negative documents for BERTopic: {len(negative_df)} < {cfg.min_documents}"
        )

    BERTopic = _require_bertopic()

    topic_model = BERTopic(
        nr_topics=cfg.nr_topics,
        min_topic_size=int(cfg.min_topic_size),
        calculate_probabilities=True,
        verbose=False,
    )

    texts = negative_df["text"].tolist()
    topics, probabilities = topic_model.fit_transform(texts)

    topic_terms_df = _build_topic_terms_table(topic_model)

    topic_confidence: np.ndarray
    if probabilities is None:
        topic_confidence = np.full(len(topics), np.nan, dtype=float)
    else:
        probabilities_array = np.asarray(probabilities)
        topic_confidence = probabilities_array.max(axis=1)

    document_topics_df = negative_df.copy()
    document_topics_df["topic_id"] = np.asarray(topics, dtype=int)
    document_topics_df["topic_confidence"] = topic_confidence

    terminal_topics_df = _build_terminal_topics_table(
        document_topics_df,
        assigned_topics=document_topics_df["topic_id"].to_numpy(),
        terminal_review_count=int(cfg.terminal_review_count),
    )

    recovery_comparison_df = _build_recovery_comparison_table(
        document_topics_df,
        assigned_topics=document_topics_df["topic_id"].to_numpy(),
    )

    topic_terms_path = out_dir / "topic_terms.csv"
    document_topics_path = out_dir / "negative_review_topics.csv"
    terminal_topics_path = out_dir / "terminal_topics.csv"
    recovery_comparison_path = out_dir / "recovery_comparison.csv"

    topic_terms_df.to_csv(topic_terms_path, index=False)
    document_topics_df.to_csv(document_topics_path, index=False)
    terminal_topics_df.to_csv(terminal_topics_path, index=False)
    recovery_comparison_df.to_csv(recovery_comparison_path, index=False)

    model_path: Path | None = None
    try:
        model_path = out_dir / "bertopic_model"
        topic_model.save(str(model_path), save_embedding_model=False)
    except Exception:
        model_path = None

    return TopicModelArtifacts(
        output_dir=out_dir,
        topic_terms_path=topic_terms_path,
        document_topics_path=document_topics_path,
        terminal_topics_path=terminal_topics_path,
        recovery_comparison_path=recovery_comparison_path,
        model_path=model_path,
    )
