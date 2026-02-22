from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass(frozen=True)
class SentimentTrainingConfig:
    """Configuration for DistilBERT sentiment training and scoring."""

    seed: int = 42
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    negative_stars: tuple[float, ...] = (1.0, 2.0)
    positive_stars: tuple[float, ...] = (4.0, 5.0)
    min_reviews_for_share: int = 5
    temperature_grid_start: float = 0.5
    temperature_grid_stop: float = 5.0
    temperature_grid_step: float = 0.1


@dataclass(frozen=True)
class SentimentTrainingArtifacts:
    """Artifacts produced by sentiment training."""

    model_dir: Path
    best_temperature: float
    eval_metrics: dict[str, float]


class ReviewTorchDataset(TorchDataset):
    """Torch dataset wrapper that tokenizes raw review text on access."""

    def __init__(
        self,
        *,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self._texts = list(texts)
        self._labels = list(labels)
        self._tokenizer = tokenizer
        self._max_length = int(max_length)

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        encoded = self._tokenizer(
            self._texts[idx],
            truncation=True,
            max_length=self._max_length,
        )
        encoded["labels"] = int(self._labels[idx])
        return encoded


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_sentiment_training_frame(
    reviews: pd.DataFrame,
    *,
    negative_stars: Sequence[float],
    positive_stars: Sequence[float],
) -> pd.DataFrame:
    """Build a binary sentiment frame from review stars and text."""
    required = {"stars", "text", "date"}
    missing = sorted(required - set(reviews.columns))
    if missing:
        raise ValueError(f"reviews missing required columns: {missing}")

    negative_set = {float(value) for value in negative_stars}
    positive_set = {float(value) for value in positive_stars}
    if not negative_set or not positive_set:
        raise ValueError("negative_stars and positive_stars must be non-empty")
    if negative_set & positive_set:
        raise ValueError("negative_stars and positive_stars must not overlap")

    frame = reviews.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["stars"] = pd.to_numeric(frame["stars"], errors="coerce")
    frame["text"] = frame["text"].fillna("").astype(str)

    allowed = negative_set | positive_set
    frame = frame[frame["stars"].isin(allowed)].copy()
    frame["label"] = frame["stars"].isin(positive_set).astype(int)

    if frame.empty:
        raise ValueError("no rows left after sentiment label filtering")

    label_counts = frame["label"].value_counts().to_dict()
    if 0 not in label_counts or 1 not in label_counts:
        raise ValueError("both label classes are required for training")

    return frame.reset_index(drop=True)


def stratified_train_val_split(
    frame: pd.DataFrame,
    *,
    label_column: str = "label",
    train_fraction: float = 0.8,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic stratified split without sklearn."""
    if label_column not in frame.columns:
        raise ValueError(f"{label_column!r} not found in frame")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")

    labels = frame[label_column].to_numpy()
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("both classes are required for stratified split")

    rng = np.random.default_rng(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_cutoff = int(train_fraction * len(pos_idx))
    neg_cutoff = int(train_fraction * len(neg_idx))

    train_idx = np.concatenate([pos_idx[:pos_cutoff], neg_idx[:neg_cutoff]])
    val_idx = np.concatenate([pos_idx[pos_cutoff:], neg_idx[neg_cutoff:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_split = frame.iloc[train_idx].reset_index(drop=True)
    val_split = frame.iloc[val_idx].reset_index(drop=True)

    return train_split, val_split


def _compute_binary_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _fit_temperature(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    start: float,
    stop: float,
    step: float,
) -> float:
    """Find the temperature that minimizes NLL on validation logits."""
    if step <= 0:
        raise ValueError("temperature step must be > 0")

    val_logits = torch.tensor(logits, dtype=torch.float32)
    val_labels = torch.tensor(labels, dtype=torch.long)

    def nll_at_temperature(temperature: float) -> float:
        scaled_logits = val_logits / float(temperature)
        probabilities = torch.softmax(scaled_logits, dim=1)
        label_probabilities = probabilities[torch.arange(len(val_labels)), val_labels]
        return float((-torch.log(label_probabilities.clamp_min(1e-12))).mean().item())

    n_steps = int(round((stop - start) / step)) + 1
    candidates = np.linspace(start, stop, n_steps)
    losses = [nll_at_temperature(float(candidate)) for candidate in candidates]
    best_idx = int(np.argmin(losses))

    return float(candidates[best_idx])


class DistilBertSentimentTrainer:
    """Train, calibrate, save, and reuse a DistilBERT binary sentiment model."""

    def __init__(self, config: SentimentTrainingConfig | None = None) -> None:
        self.config = config or SentimentTrainingConfig()

    def train(self, reviews: pd.DataFrame, *, output_dir: Path | str) -> SentimentTrainingArtifacts:
        """Train a model from review text and star-derived labels."""
        set_global_seed(self.config.seed)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        labeled = prepare_sentiment_training_frame(
            reviews,
            negative_stars=self.config.negative_stars,
            positive_stars=self.config.positive_stars,
        )

        train_split, val_split = stratified_train_val_split(
            labeled,
            label_column="label",
            train_fraction=0.8,
            seed=self.config.seed,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, num_labels=2)

        train_dataset = ReviewTorchDataset(
            texts=train_split["text"],
            labels=train_split["label"],
            tokenizer=tokenizer,
            max_length=self.config.max_length,
        )
        val_dataset = ReviewTorchDataset(
            texts=val_split["text"],
            labels=val_split["label"],
            tokenizer=tokenizer,
            max_length=self.config.max_length,
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

        steps_per_epoch = math.ceil(
            len(train_dataset) / (self.config.train_batch_size * self.config.gradient_accumulation_steps)
        )
        total_steps = max(1, steps_per_epoch * self.config.epochs)
        warmup_steps = int(self.config.warmup_ratio * total_steps)

        training_args = TrainingArguments(
            output_dir=str(output_path),
            seed=self.config.seed,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=100,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_metrics=_compute_binary_metrics,
        )

        trainer.train()
        eval_metrics = {key: float(value) for key, value in trainer.evaluate().items()}

        trainer.save_model(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        prediction_output = trainer.predict(val_dataset)
        best_temperature = _fit_temperature(
            logits=prediction_output.predictions,
            labels=prediction_output.label_ids,
            start=self.config.temperature_grid_start,
            stop=self.config.temperature_grid_stop,
            step=self.config.temperature_grid_step,
        )

        calibration_path = output_path / "calibration.json"
        calibration_path.write_text(
            json.dumps({"temperature": best_temperature}, indent=2),
            encoding="utf-8",
        )

        return SentimentTrainingArtifacts(
            model_dir=output_path,
            best_temperature=best_temperature,
            eval_metrics=eval_metrics,
        )

    def score_reviews(
        self,
        reviews: pd.DataFrame,
        *,
        model_dir: Path | str,
        temperature: float | None = None,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """Score raw reviews with calibrated positive sentiment probability."""
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"sentiment model directory not found: {model_path}")

        required = {"text"}
        missing = sorted(required - set(reviews.columns))
        if missing:
            raise ValueError(f"reviews missing required columns: {missing}")

        calibrated_temperature = temperature
        if calibrated_temperature is None:
            calibration_file = model_path / "calibration.json"
            if calibration_file.exists():
                payload = json.loads(calibration_file.read_text(encoding="utf-8"))
                calibrated_temperature = float(payload.get("temperature", 1.0))
            else:
                calibrated_temperature = 1.0

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model.to(device)
        model.eval()

        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        texts = reviews["text"].fillna("").astype(str).tolist()

        probabilities: list[float] = []
        with torch.no_grad():
            for offset in range(0, len(texts), batch_size):
                batch_texts = texts[offset : offset + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=self.config.max_length,
                )
                features = [
                    {key: encoded[key][index] for key in encoded.keys()}
                    for index in range(len(batch_texts))
                ]
                batch = collator(features)
                batch = {key: value.to(device) for key, value in batch.items()}

                logits = model(**batch).logits
                pos_probs = torch.softmax(logits / float(calibrated_temperature), dim=1)[:, 1]
                probabilities.extend(pos_probs.detach().cpu().numpy().tolist())

        scored = reviews.copy()
        scored["tx_sent"] = np.asarray(probabilities, dtype=float)
        return scored


def aggregate_business_month_sentiment(
    scored_reviews: pd.DataFrame,
    *,
    min_reviews_for_share: int = 5,
) -> pd.DataFrame:
    """Aggregate scored reviews into monthly business panel features."""
    required = {"business_id", "status", "date", "review_id", "stars", "tx_sent"}
    missing = sorted(required - set(scored_reviews.columns))
    if missing:
        raise ValueError(f"scored_reviews missing required columns: {missing}")

    frame = scored_reviews.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        frame.groupby(["business_id", "status", "month"], as_index=False)
        .agg(
            review_count=("review_id", "count"),
            avg_stars=("stars", "mean"),
            tx_sent_mean=("tx_sent", "mean"),
            tx_sent_std=("tx_sent", "std"),
            tx_neg_share=("tx_sent", lambda series: float((series < 0.30).mean())),
            tx_pos_share=("tx_sent", lambda series: float((series > 0.70).mean())),
        )
        .sort_values(["business_id", "month"])
        .reset_index(drop=True)
    )

    monthly["tx_pos_share"] = np.where(
        monthly["review_count"] >= int(min_reviews_for_share),
        monthly["tx_pos_share"],
        np.nan,
    )
    monthly["tx_neg_share"] = np.where(
        monthly["review_count"] >= int(min_reviews_for_share),
        monthly["tx_neg_share"],
        np.nan,
    )
    monthly["tx_sent_std"] = pd.to_numeric(monthly["tx_sent_std"], errors="coerce").fillna(0.0)

    return monthly
