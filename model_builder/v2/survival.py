from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import SequenceWindowConfig, SequenceWindows, build_sequence_windows, compute_business_snapshot_features


@dataclass(frozen=True)
class SurvivalTrainingConfig:
    """Training controls for Component 1 multi-signal survival modeling."""

    random_seed: int = 42
    train_fraction: float = 0.8
    threshold_beta: float = 2.0
    rule_low_review_cutoff: int = 5

    sequence: SequenceWindowConfig = field(default_factory=SequenceWindowConfig)

    enable_gru: bool = True
    gru_epochs: int = 40
    gru_batch_size: int = 256
    gru_learning_rate: float = 1e-3
    gru_weight_decay: float = 1e-4
    gru_label_smoothing: float = 0.01
    gru_clipnorm: float = 1.0
    gru_pos_batch_rate: float = 0.30
    gru_early_stopping_patience: int = 6
    gru_lr_plateau_patience: int = 3
    gru_lr_plateau_factor: float = 0.5
    gru_min_learning_rate: float = 1e-5
    gru_recent_k_windows: int = 3


@dataclass(frozen=True)
class BaselineModelResult:
    model_name: str
    threshold: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class GruModelResult:
    threshold: float
    metrics: dict[str, float]
    window_topk_metrics: pd.DataFrame
    business_topk_metrics: pd.DataFrame
    business_triage: pd.DataFrame


@dataclass(frozen=True)
class RuleBasedRiskModel:
    """Low-data fallback model for businesses with <=5 reviews."""

    checkin_floor: float
    star_floor: float
    velocity_floor: float

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        required = {"avg_monthly_checkins", "stars_mean_weighted", "checkin_velocity_3m"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"rule model input missing required columns: {missing}")

        checkins = pd.to_numeric(frame["avg_monthly_checkins"], errors="coerce").fillna(0.0)
        stars = pd.to_numeric(frame["stars_mean_weighted"], errors="coerce").fillna(0.0)
        velocity = pd.to_numeric(frame["checkin_velocity_3m"], errors="coerce").fillna(0.0)

        checkin_component = ((self.checkin_floor - checkins) / max(self.checkin_floor, 1.0)).clip(lower=0.0)
        star_component = ((self.star_floor - stars) / 5.0).clip(lower=0.0)
        velocity_component = ((self.velocity_floor - velocity) / max(abs(self.velocity_floor), 1.0)).clip(lower=0.0)

        score = 0.60 * checkin_component + 0.25 * star_component + 0.15 * velocity_component
        return np.clip(score.to_numpy(dtype=float), 0.0, 1.0)


@dataclass(frozen=True)
class SurvivalTrainingArtifacts:
    """Output paths and metrics for Component 1."""

    output_dir: Path
    baseline_model_path: Path
    baseline_metrics_path: Path
    rule_model_path: Path
    snapshot_features_path: Path
    sequence_meta_path: Path
    gru_model_path: Path | None
    gru_metrics_path: Path | None
    gru_metadata_path: Path | None
    gru_window_topk_metrics_path: Path | None
    gru_business_topk_metrics_path: Path | None
    gru_business_triage_path: Path | None
    baseline_results: dict[str, BaselineModelResult]
    rule_metrics: dict[str, float]
    gru_result: GruModelResult | None


@dataclass(frozen=True)
class WindowTrainValidationSplit:
    """Business-stratified train/validation partition for sequence windows."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    meta_val: pd.DataFrame


FEATURE_EXCLUDE_COLUMNS = {
    "business_id",
    "name",
    "city",
    "state",
    "categories",
    "status",
    "last_month",
    "label_closed",
}


class SurvivalDependencyError(RuntimeError):
    """Raised when required survival modeling dependencies are missing."""


def _select_numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in FEATURE_EXCLUDE_COLUMNS:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().sum() == 0:
            continue
        columns.append(column)
    return columns


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray, beta: float) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5

    beta_sq = float(beta) ** 2
    best_threshold = 0.5
    best_score = -1.0

    for idx, threshold in enumerate(thresholds):
        p = float(precision[idx + 1])
        r = float(recall[idx + 1])
        denom = beta_sq * p + r
        if denom <= 1e-12:
            continue
        f_beta = (1.0 + beta_sq) * p * r / denom
        if f_beta > best_score:
            best_score = f_beta
            best_threshold = float(threshold)

    return float(best_threshold)


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

    y_pred = (probabilities >= float(threshold)).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, probabilities)) if len(y_true) else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }

    return metrics


def _train_baseline_estimators(
    snapshot_features: pd.DataFrame,
    *,
    config: SurvivalTrainingConfig,
) -> tuple[dict[str, Any], dict[str, BaselineModelResult], list[str]]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    feature_columns = _select_numeric_feature_columns(snapshot_features)
    if not feature_columns:
        raise ValueError("no numeric baseline feature columns were detected")

    frame = snapshot_features.copy()
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = frame[feature_columns].to_numpy(dtype=float)
    y = frame["label_closed"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("baseline training requires both open and closed classes")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=float(config.train_fraction),
        random_state=int(config.random_seed),
        stratify=y,
    )

    estimators: dict[str, Any] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        random_state=int(config.random_seed),
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=int(config.random_seed),
                    ),
                ),
            ]
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            random_state=int(config.random_seed),
            learning_rate=0.05,
            max_depth=6,
            max_iter=400,
        ),
    }

    results: dict[str, BaselineModelResult] = {}
    fitted_models: dict[str, Any] = {}

    for model_name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_val)[:, 1]
        threshold = _select_threshold(y_val, probabilities, config.threshold_beta)
        metrics = _classification_metrics(y_val, probabilities, threshold)

        fitted_models[model_name] = estimator
        results[model_name] = BaselineModelResult(
            model_name=model_name,
            threshold=float(threshold),
            metrics=metrics,
        )

    return fitted_models, results, feature_columns


def _split_windows_by_business(
    windows: SequenceWindows,
    *,
    train_fraction: float,
    seed: int,
) -> WindowTrainValidationSplit | None:
    from sklearn.model_selection import train_test_split

    if windows.X.size == 0 or windows.meta.empty:
        return None

    meta = windows.meta.copy()
    business_outcomes = (
        meta.groupby("business_id", as_index=False)["label"]
        .max()
        .rename(columns={"label": "business_label"})
    )

    if business_outcomes["business_label"].nunique() < 2:
        return None

    train_business_ids, val_business_ids = train_test_split(
        business_outcomes["business_id"].to_numpy(),
        train_size=float(train_fraction),
        random_state=int(seed),
        stratify=business_outcomes["business_label"].to_numpy(),
    )

    train_mask = meta["business_id"].isin(set(train_business_ids)).to_numpy()
    val_mask = meta["business_id"].isin(set(val_business_ids)).to_numpy()

    X_train = windows.X[train_mask]
    y_train = windows.y[train_mask]
    X_val = windows.X[val_mask]
    y_val = windows.y[val_mask]
    meta_val = meta.loc[val_mask].reset_index(drop=True)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return None

    return WindowTrainValidationSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        meta_val=meta_val,
    )


def _build_balanced_training_datasets(
    split: WindowTrainValidationSplit,
    *,
    config: SurvivalTrainingConfig,
) -> tuple[Any, Any, int]:
    import tensorflow as tf

    pos_idx = np.where(split.y_train == 1)[0]
    neg_idx = np.where(split.y_train == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("training split must include both positive and negative windows")

    ds_pos = tf.data.Dataset.from_tensor_slices((split.X_train[pos_idx], split.y_train[pos_idx]))
    ds_neg = tf.data.Dataset.from_tensor_slices((split.X_train[neg_idx], split.y_train[neg_idx]))

    ds_pos = ds_pos.shuffle(min(len(pos_idx), 20_000), seed=config.random_seed, reshuffle_each_iteration=True).repeat()
    ds_neg = ds_neg.shuffle(min(len(neg_idx), 20_000), seed=config.random_seed, reshuffle_each_iteration=True).repeat()

    train_ds = tf.data.Dataset.sample_from_datasets(
        [ds_pos, ds_neg],
        weights=[float(config.gru_pos_batch_rate), 1.0 - float(config.gru_pos_batch_rate)],
        seed=config.random_seed,
    )
    train_ds = train_ds.batch(int(config.gru_batch_size), drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((split.X_val, split.y_val))
    val_ds = val_ds.batch(int(config.gru_batch_size)).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = int(np.ceil(len(split.y_train) / max(1, int(config.gru_batch_size))))
    return train_ds, val_ds, steps_per_epoch


def _build_gru_model(*, sequence_length: int, n_features: int, base_rate_bias: float) -> Any:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(sequence_length, n_features))

    hidden = layers.GRU(128, return_sequences=True)(inputs)
    hidden = layers.LayerNormalization()(hidden)
    hidden = layers.Dropout(0.20)(hidden)

    hidden = layers.GRU(64, return_sequences=True)(hidden)
    hidden = layers.LayerNormalization()(hidden)
    hidden = layers.Dropout(0.20)(hidden)

    attention = layers.Dense(1)(hidden)
    attention = layers.Softmax(axis=1, name="attn_softmax")(attention)
    hidden = layers.Multiply()([hidden, attention])
    hidden = layers.Lambda(lambda tensor: tf.reduce_sum(tensor, axis=1), name="attn_pool_sum")(hidden)

    hidden = layers.Dense(64, activation="relu")(hidden)
    hidden = layers.Dropout(0.25)(hidden)

    output = layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=keras.initializers.Constant(base_rate_bias),
    )(hidden)

    return keras.Model(inputs, output)


def _compile_gru_model(model: Any, *, config: SurvivalTrainingConfig) -> None:
    from tensorflow import keras

    try:
        optimizer = keras.optimizers.AdamW(
            learning_rate=float(config.gru_learning_rate),
            weight_decay=float(config.gru_weight_decay),
            clipnorm=float(config.gru_clipnorm),
        )
    except Exception:
        optimizer = keras.optimizers.Adam(
            learning_rate=float(config.gru_learning_rate),
            clipnorm=float(config.gru_clipnorm),
        )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(label_smoothing=float(config.gru_label_smoothing)),
        metrics=[
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def _compute_topk_metrics(
    *,
    ranked_frame: pd.DataFrame,
    label_column: str,
    score_column: str,
    percentiles: tuple[float, ...] = (0.5, 1, 2, 5, 10, 15, 20),
) -> pd.DataFrame:
    frame = ranked_frame.sort_values(score_column, ascending=False).reset_index(drop=True)
    total = len(frame)
    positives = int((frame[label_column] == 1).sum())

    rows: list[dict[str, Any]] = []
    for percentile in percentiles:
        k = max(1, int(total * (float(percentile) / 100.0)))
        topk = frame.head(k)
        tp = int((topk[label_column] == 1).sum())

        rows.append(
            {
                "percentile": float(percentile),
                "k": int(k),
                "precision": float(tp / max(1, k)),
                "recall": float(tp / max(1, positives)),
                "tp": int(tp),
            }
        )

    return pd.DataFrame(rows)


def _compute_window_topk_metrics(
    *,
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    percentiles: tuple[float, ...] = (0.5, 1, 2, 5, 10, 15, 20),
) -> pd.DataFrame:
    frame = metadata.copy()
    frame["p_closed"] = np.asarray(probabilities, dtype=float)
    frame["y_true"] = np.asarray(labels, dtype=int)
    return _compute_topk_metrics(
        ranked_frame=frame,
        label_column="y_true",
        score_column="p_closed",
        percentiles=percentiles,
    )


def _compute_business_triage(
    *,
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    recent_k_windows: int,
) -> pd.DataFrame:
    frame = metadata.copy()
    frame["p_closed"] = np.asarray(probabilities, dtype=float)
    frame["y_true"] = np.asarray(labels, dtype=int)
    frame = frame.sort_values(["business_id", "end_month"]).reset_index(drop=True)

    grouped = frame.groupby("business_id", as_index=False).agg(
        status=("status", "last"),
        end_month_last=("end_month", "max"),
        p_last=("p_closed", "last"),
        p_mean=("p_closed", "mean"),
        p_max=("p_closed", "max"),
        n_windows=("p_closed", "size"),
        y_business=("y_true", "max"),
    )

    recent_last_k = (
        frame.groupby("business_id", group_keys=False)
        .tail(max(1, int(recent_k_windows)))
        .groupby("business_id", as_index=False)["p_closed"]
        .max()
        .rename(columns={"p_closed": "p_recent_max"})
    )
    triage = grouped.merge(recent_last_k, on="business_id", how="left")
    triage["risk_score"] = triage["p_recent_max"].fillna(triage["p_max"])
    return triage.sort_values("risk_score", ascending=False).reset_index(drop=True)


def _compute_business_topk_metrics(
    triage: pd.DataFrame,
    *,
    percentiles: tuple[float, ...] = (0.5, 1, 2, 5, 10, 15, 20),
) -> pd.DataFrame:
    return _compute_topk_metrics(
        ranked_frame=triage,
        label_column="y_business",
        score_column="risk_score",
        percentiles=percentiles,
    )


def _train_gru_model(
    windows: SequenceWindows,
    *,
    config: SurvivalTrainingConfig,
) -> tuple[Any, GruModelResult] | tuple[None, None]:
    if windows.X.size == 0 or windows.meta.empty:
        return None, None

    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        return None, None

    split = _split_windows_by_business(
        windows,
        train_fraction=float(config.train_fraction),
        seed=int(config.random_seed),
    )
    if split is None:
        return None, None

    np.random.seed(int(config.random_seed))
    tf.random.set_seed(int(config.random_seed))

    train_ds, val_ds, steps_per_epoch = _build_balanced_training_datasets(split, config=config)

    pos_rate = float((split.y_train == 1).mean())
    epsilon = 1e-7
    base_rate_bias = float(np.log((pos_rate + epsilon) / (1.0 - pos_rate + epsilon)))

    model = _build_gru_model(
        sequence_length=split.X_train.shape[1],
        n_features=split.X_train.shape[2],
        base_rate_bias=base_rate_bias,
    )
    _compile_gru_model(model, config=config)

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=int(config.gru_early_stopping_patience),
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=float(config.gru_lr_plateau_factor),
            patience=int(config.gru_lr_plateau_patience),
            min_lr=float(config.gru_min_learning_rate),
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config.gru_epochs),
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(split.X_val, batch_size=512, verbose=0).reshape(-1)
    threshold = _select_threshold(split.y_val, probabilities, config.threshold_beta)

    metrics = _classification_metrics(split.y_val, probabilities, threshold)
    raw_eval = model.evaluate(val_ds, verbose=0)
    for name, value in zip(model.metrics_names, raw_eval):
        metrics[f"keras_{name}"] = float(value)

    window_topk = _compute_window_topk_metrics(
        metadata=split.meta_val,
        probabilities=probabilities,
        labels=split.y_val,
    )
    business_triage = _compute_business_triage(
        metadata=split.meta_val,
        probabilities=probabilities,
        labels=split.y_val,
        recent_k_windows=int(config.gru_recent_k_windows),
    )
    business_topk = _compute_business_topk_metrics(business_triage)

    result = GruModelResult(
        threshold=float(threshold),
        metrics=metrics,
        window_topk_metrics=window_topk,
        business_topk_metrics=business_topk,
        business_triage=business_triage,
    )
    return model, result


def fit_rule_based_model(
    snapshot_features: pd.DataFrame,
    *,
    config: SurvivalTrainingConfig,
) -> RuleBasedRiskModel:
    """Fit a simple interpretable low-data rule model."""
    low_data = snapshot_features[
        pd.to_numeric(snapshot_features["total_reviews"], errors="coerce").fillna(0.0)
        <= float(config.rule_low_review_cutoff)
    ].copy()

    if low_data.empty:
        low_data = snapshot_features.copy()

    closed_low = low_data[low_data["label_closed"] == 1]
    checkin_source = closed_low if not closed_low.empty else low_data

    checkin_floor = float(
        pd.to_numeric(checkin_source["avg_monthly_checkins"], errors="coerce").fillna(0.0).quantile(0.60)
    )
    star_floor = float(
        pd.to_numeric(checkin_source["stars_mean_weighted"], errors="coerce").fillna(0.0).quantile(0.60)
    )
    velocity_floor = float(
        pd.to_numeric(checkin_source["checkin_velocity_3m"], errors="coerce").fillna(0.0).quantile(0.40)
    )

    return RuleBasedRiskModel(
        checkin_floor=max(checkin_floor, 1e-6),
        star_floor=max(star_floor, 1e-6),
        velocity_floor=velocity_floor,
    )


def _evaluate_rule_model(
    model: RuleBasedRiskModel,
    snapshot_features: pd.DataFrame,
    *,
    config: SurvivalTrainingConfig,
) -> dict[str, float]:
    low_data = snapshot_features[
        pd.to_numeric(snapshot_features["total_reviews"], errors="coerce").fillna(0.0)
        <= float(config.rule_low_review_cutoff)
    ].copy()

    if low_data.empty:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "threshold": 0.5,
            "n_rows": 0.0,
        }

    y_true = low_data["label_closed"].astype(int).to_numpy()
    probabilities = model.score_frame(low_data)

    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "threshold": 0.5,
            "n_rows": float(len(low_data)),
        }

    threshold = _select_threshold(y_true, probabilities, config.threshold_beta)
    metrics = _classification_metrics(y_true, probabilities, threshold)
    metrics["n_rows"] = float(len(low_data))
    return metrics


def _persist_baseline_bundle(
    output_dir: Path,
    *,
    models: dict[str, Any],
    feature_columns: list[str],
    baseline_results: dict[str, BaselineModelResult],
) -> tuple[Path, Path]:
    payload = {
        "models": models,
        "feature_columns": feature_columns,
        "thresholds": {name: result.threshold for name, result in baseline_results.items()},
    }

    model_path = output_dir / "baseline_models.joblib"
    metrics_path = output_dir / "baseline_metrics.json"

    try:
        import joblib

        joblib.dump(payload, model_path)
    except Exception as exc:  # pragma: no cover
        raise SurvivalDependencyError("joblib is required to persist baseline models") from exc

    metrics_payload = {name: result.metrics for name, result in baseline_results.items()}
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return model_path, metrics_path


def _persist_rule_model(output_dir: Path, model: RuleBasedRiskModel, metrics: dict[str, float]) -> Path:
    path = output_dir / "rule_model.json"
    payload = {
        "checkin_floor": model.checkin_floor,
        "star_floor": model.star_floor,
        "velocity_floor": model.velocity_floor,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _persist_gru_outputs(
    output_dir: Path,
    *,
    model: Any,
    config: SurvivalTrainingConfig,
    windows: SequenceWindows,
    result: GruModelResult,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    model_path = output_dir / "gru_survival_model.keras"
    metrics_path = output_dir / "gru_metrics.json"
    metadata_path = output_dir / "gru_metadata.json"
    window_topk_path = output_dir / "gru_window_topk.csv"
    business_topk_path = output_dir / "gru_business_topk.csv"
    business_triage_path = output_dir / "gru_business_triage.csv"

    model.save(model_path)

    metrics_path.write_text(
        json.dumps({"threshold": result.threshold, "metrics": result.metrics}, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "feature_columns": list(windows.feature_columns),
        "seq_len": int(config.sequence.seq_len),
        "horizon_months": int(config.sequence.horizon_months),
        "min_total_reviews_for_sequence": int(config.sequence.min_total_reviews_for_sequence),
        "min_active_months": int(config.sequence.min_active_months),
        "min_reviews_in_window": int(config.sequence.min_reviews_in_window),
        "inactive_months_for_zombie": int(config.sequence.inactive_months_for_zombie),
        "gru_pos_batch_rate": float(config.gru_pos_batch_rate),
        "gru_batch_size": int(config.gru_batch_size),
        "gru_epochs": int(config.gru_epochs),
        "gru_learning_rate": float(config.gru_learning_rate),
        "gru_weight_decay": float(config.gru_weight_decay),
        "gru_label_smoothing": float(config.gru_label_smoothing),
        "gru_clipnorm": float(config.gru_clipnorm),
        "gru_recent_k_windows": int(config.gru_recent_k_windows),
        "threshold": float(result.threshold),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result.window_topk_metrics.to_csv(window_topk_path, index=False)
    result.business_topk_metrics.to_csv(business_topk_path, index=False)
    result.business_triage.to_csv(business_triage_path, index=False)

    return (
        model_path,
        metrics_path,
        metadata_path,
        window_topk_path,
        business_topk_path,
        business_triage_path,
    )


def train_survival_models(
    monthly_panel: pd.DataFrame,
    *,
    output_dir: Path | str,
    config: SurvivalTrainingConfig | None = None,
) -> SurvivalTrainingArtifacts:
    """
    Train Component 1 models from the methodology:
    - Logistic Regression baseline
    - SVM baseline
    - Gradient Boosting baseline
    - GRU sequence model for businesses with >10 reviews
    - Rule-based scorer for businesses with <=5 reviews
    """
    cfg = config or SurvivalTrainingConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_features = compute_business_snapshot_features(monthly_panel)
    snapshot_features_path = out_dir / "business_snapshot_features.csv"
    snapshot_features.to_csv(snapshot_features_path, index=False)

    fitted_models, baseline_results, feature_columns = _train_baseline_estimators(
        snapshot_features,
        config=cfg,
    )
    baseline_model_path, baseline_metrics_path = _persist_baseline_bundle(
        out_dir,
        models=fitted_models,
        feature_columns=feature_columns,
        baseline_results=baseline_results,
    )

    rule_model = fit_rule_based_model(snapshot_features, config=cfg)
    rule_metrics = _evaluate_rule_model(rule_model, snapshot_features, config=cfg)
    rule_model_path = _persist_rule_model(out_dir, rule_model, rule_metrics)

    windows = build_sequence_windows(monthly_panel, config=cfg.sequence)
    sequence_meta_path = out_dir / "sequence_window_meta.csv"
    windows.meta.to_csv(sequence_meta_path, index=False)

    gru_model_path: Path | None = None
    gru_metrics_path: Path | None = None
    gru_metadata_path: Path | None = None
    gru_window_topk_metrics_path: Path | None = None
    gru_business_topk_metrics_path: Path | None = None
    gru_business_triage_path: Path | None = None
    gru_result: GruModelResult | None = None

    if cfg.enable_gru:
        model, gru_result = _train_gru_model(windows, config=cfg)
        if model is not None and gru_result is not None:
            (
                gru_model_path,
                gru_metrics_path,
                gru_metadata_path,
                gru_window_topk_metrics_path,
                gru_business_topk_metrics_path,
                gru_business_triage_path,
            ) = _persist_gru_outputs(
                out_dir,
                model=model,
                config=cfg,
                windows=windows,
                result=gru_result,
            )

    return SurvivalTrainingArtifacts(
        output_dir=out_dir,
        baseline_model_path=baseline_model_path,
        baseline_metrics_path=baseline_metrics_path,
        rule_model_path=rule_model_path,
        snapshot_features_path=snapshot_features_path,
        sequence_meta_path=sequence_meta_path,
        gru_model_path=gru_model_path,
        gru_metrics_path=gru_metrics_path,
        gru_metadata_path=gru_metadata_path,
        gru_window_topk_metrics_path=gru_window_topk_metrics_path,
        gru_business_topk_metrics_path=gru_business_topk_metrics_path,
        gru_business_triage_path=gru_business_triage_path,
        baseline_results=baseline_results,
        rule_metrics=rule_metrics,
        gru_result=gru_result,
    )
