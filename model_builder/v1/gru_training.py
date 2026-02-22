from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import tensorflow as tf
    from tensorflow import keras


REQUIRED_MONTHLY_COLUMNS = {
    "business_id",
    "status",
    "month",
    "review_count",
    "avg_stars",
    "tx_sent_mean",
    "tx_sent_std",
    "tx_neg_share",
    "tx_pos_share",
}

BASE_FEATURE_COLUMNS = [
    "review_count",
    "avg_stars",
    "tx_sent_mean",
    "tx_sent_std",
    "tx_neg_share",
    "tx_pos_share",
]


@dataclass(frozen=True)
class GruTrainingConfig:
    """Configuration for GRU closure-risk model training."""

    seed: int = 42
    sequence_length: int = 12
    horizon_months: int = 6
    inactive_months_for_zombie: int = 12
    min_active_months: int = 6
    min_reviews_in_window: int = 10
    positive_batch_rate: float = 0.30
    batch_size: int = 256
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.01
    clipnorm: float = 1.0
    early_stopping_patience: int = 6
    lr_plateau_patience: int = 3
    lr_plateau_factor: float = 0.5
    min_learning_rate: float = 1e-5
    recent_k_months: int = 3
    risk_bins: tuple[float, ...] = (0.0, 0.50, 0.65, 0.75, 0.85, 1.0)
    risk_labels: tuple[str, ...] = ("low", "medium", "elevated", "high", "very_high")


@dataclass(frozen=True)
class WindowBuildResult:
    """Result of converting a monthly panel into sequence windows."""

    features: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    feature_columns: list[str]
    global_last_month: pd.Timestamp


@dataclass(frozen=True)
class TrainValidationSplit:
    """Business-stratified training and validation partitions."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    metadata_train: pd.DataFrame
    metadata_val: pd.DataFrame


@dataclass(frozen=True)
class GruTrainingArtifacts:
    """Files and frames produced by end-to-end GRU training."""

    model_path: Path
    metadata_path: Path
    triage_path: Path
    top5_path: Path
    top10_path: Path
    validation_metrics: dict[str, float]
    window_topk_metrics: pd.DataFrame
    business_triage: pd.DataFrame


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for reproducible training runs."""
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_monthly_panel(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize monthly panel columns required by the GRU pipeline."""
    missing = sorted(REQUIRED_MONTHLY_COLUMNS - set(monthly_panel.columns))
    if missing:
        raise ValueError(f"monthly panel missing required columns: {missing}")

    frame = monthly_panel.copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
    frame = frame.dropna(subset=["month"]).copy()

    for column in BASE_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[BASE_FEATURE_COLUMNS] = frame[BASE_FEATURE_COLUMNS].fillna(0.0)

    frame = frame.sort_values(["business_id", "month"]).reset_index(drop=True)
    return frame


def _build_business_status_snapshot(
    panel: pd.DataFrame,
    *,
    inactive_months_for_zombie: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Compute business-level last review month, closure proxy, and zombie flags."""
    global_last_month = panel["month"].max()

    snapshot = (
        panel.groupby(["business_id", "status"], as_index=False)["month"]
        .max()
        .rename(columns={"month": "last_review_month"})
    )

    snapshot["closure_month"] = pd.NaT
    closed_mask = snapshot["status"] == "Closed"
    snapshot.loc[closed_mask, "closure_month"] = snapshot.loc[closed_mask, "last_review_month"].values

    zombie_cutoff = global_last_month - pd.DateOffset(months=int(inactive_months_for_zombie))
    snapshot["is_zombie_open"] = (snapshot["status"] == "Open") & (
        snapshot["last_review_month"] <= zombie_cutoff
    )

    return snapshot, global_last_month


def _add_trajectory_features(group: pd.DataFrame, *, base_columns: Sequence[str]) -> pd.DataFrame:
    """Create per-business standardized and trajectory features."""
    group = group.sort_values("month").copy()

    values = group[list(base_columns)].astype(float)
    means = values.mean(axis=0)
    stds = values.std(axis=0).replace(0.0, 1.0)

    for column in base_columns:
        group[f"{column}_z"] = (group[column].astype(float) - float(means[column])) / float(stds[column])

    z_columns = [f"{column}_z" for column in base_columns]
    for column in z_columns:
        group[f"{column}_d1"] = group[column].diff(1)
        group[f"{column}_rm3"] = group[column].rolling(3, min_periods=1).mean()
        group[f"{column}_rs3"] = group[column].rolling(3, min_periods=1).std()
        group[f"{column}_rs6"] = group[column].rolling(6, min_periods=1).std()

    month_origin = int(group["month"].iloc[0].year) * 12 + int(group["month"].iloc[0].month)
    month_index = group["month"].dt.year.astype(int) * 12 + group["month"].dt.month.astype(int)
    group["months_since_first"] = (month_index - month_origin).astype(float)

    engineered_columns = [
        column
        for column in group.columns
        if column.endswith("_z")
        or column.endswith("_d1")
        or column.endswith("_rm3")
        or column.endswith("_rs3")
        or column.endswith("_rs6")
    ]

    group[engineered_columns] = (
        group[engineered_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    return group


def engineer_feature_panel(monthly_panel: pd.DataFrame, config: GruTrainingConfig) -> tuple[pd.DataFrame, list[str], pd.Timestamp]:
    """Prepare monthly panel with business snapshot flags and engineered trajectory features."""
    prepared = prepare_monthly_panel(monthly_panel)
    snapshot, global_last_month = _build_business_status_snapshot(
        prepared,
        inactive_months_for_zombie=config.inactive_months_for_zombie,
    )

    merged = prepared.merge(
        snapshot[["business_id", "status", "last_review_month", "closure_month", "is_zombie_open"]],
        on=["business_id", "status"],
        how="left",
    )

    featured = merged.groupby("business_id", group_keys=False).apply(
        _add_trajectory_features,
        base_columns=BASE_FEATURE_COLUMNS,
    )

    feature_columns = [
        column
        for column in featured.columns
        if column.endswith("_z")
        or column.endswith("_d1")
        or column.endswith("_rm3")
        or column.endswith("_rs3")
        or column.endswith("_rs6")
    ] + ["months_since_first"]

    return featured.reset_index(drop=True), feature_columns, global_last_month


def build_windows_and_labels(
    feature_panel: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: GruTrainingConfig,
    global_last_month: pd.Timestamp,
) -> WindowBuildResult:
    """Convert feature panel into fixed-length windows with closure labels."""
    sequence_length = int(config.sequence_length)
    horizon_months = int(config.horizon_months)

    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, Any]] = []

    for (business_id, status), group in feature_panel.groupby(["business_id", "status"]):
        group = group.sort_values("month").reset_index(drop=True)

        is_zombie_open = bool(group["is_zombie_open"].iloc[0])
        closure_month = group["closure_month"].iloc[0]

        if status == "Open" and is_zombie_open:
            continue
        if len(group) < sequence_length:
            continue

        for start in range(0, len(group) - sequence_length + 1):
            end = start + sequence_length
            window = group.iloc[start:end].copy()
            window_end = window["month"].iloc[-1]

            active_months = int((window["review_count"] > 0).sum())
            total_reviews = float(window["review_count"].sum())
            if active_months < int(config.min_active_months):
                continue
            if total_reviews < float(config.min_reviews_in_window):
                continue

            if status == "Closed" and pd.notna(closure_month) and window_end >= closure_month:
                continue

            horizon_end = window_end + pd.DateOffset(months=horizon_months)
            if status == "Open" and horizon_end > global_last_month:
                continue

            if status == "Closed" and pd.notna(closure_month):
                label = int((closure_month > window_end) and (closure_month <= horizon_end))
            else:
                label = 0

            features.append(window[list(feature_columns)].to_numpy(dtype=np.float32))
            labels.append(label)
            metadata_rows.append(
                {
                    "business_id": business_id,
                    "status": status,
                    "start_month": window["month"].iloc[0],
                    "end_month": window_end,
                    "horizon_end": horizon_end,
                    "closure_month": closure_month,
                    "y": label,
                    "last_review_month": group["last_review_month"].iloc[0],
                }
            )

    if not features:
        raise ValueError("no windows available after filtering")

    return WindowBuildResult(
        features=np.stack(features, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        metadata=pd.DataFrame(metadata_rows),
        feature_columns=list(feature_columns),
        global_last_month=global_last_month,
    )


def split_windows_by_business(
    windows: WindowBuildResult,
    *,
    train_fraction: float,
    seed: int,
) -> TrainValidationSplit:
    """Split windows by business to avoid identity leakage."""
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")

    metadata = windows.metadata
    outcomes = (
        metadata.groupby("business_id", as_index=False)["y"]
        .max()
        .rename(columns={"y": "y_business"})
    )

    positive_ids = outcomes[outcomes["y_business"] == 1]["business_id"].to_numpy()
    negative_ids = outcomes[outcomes["y_business"] == 0]["business_id"].to_numpy()

    if len(positive_ids) == 0 or len(negative_ids) == 0:
        raise ValueError("both positive and negative businesses are required for split")

    rng = np.random.default_rng(seed)
    rng.shuffle(positive_ids)
    rng.shuffle(negative_ids)

    def compute_cutoff(size: int) -> int:
        cutoff = int(train_fraction * size)
        if size == 1:
            return 1
        return min(max(1, cutoff), size - 1)

    pos_cutoff = compute_cutoff(len(positive_ids))
    neg_cutoff = compute_cutoff(len(negative_ids))

    train_businesses = set(positive_ids[:pos_cutoff].tolist() + negative_ids[:neg_cutoff].tolist())
    val_businesses = set(positive_ids[pos_cutoff:].tolist() + negative_ids[neg_cutoff:].tolist())

    train_mask = metadata["business_id"].isin(train_businesses).to_numpy()
    val_mask = metadata["business_id"].isin(val_businesses).to_numpy()

    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise ValueError("train/validation split produced an empty partition")

    return TrainValidationSplit(
        X_train=windows.features[train_mask],
        y_train=windows.labels[train_mask],
        X_val=windows.features[val_mask],
        y_val=windows.labels[val_mask],
        metadata_train=metadata.loc[train_mask].reset_index(drop=True),
        metadata_val=metadata.loc[val_mask].reset_index(drop=True),
    )


def build_balanced_training_datasets(
    split: TrainValidationSplit,
    *,
    config: GruTrainingConfig,
) -> tuple["tf.data.Dataset", "tf.data.Dataset", int]:
    """Create balanced tf.data datasets for training and validation."""
    import tensorflow as tf

    pos_idx = np.where(split.y_train == 1)[0]
    neg_idx = np.where(split.y_train == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("training split must include both positive and negative windows")

    ds_pos = tf.data.Dataset.from_tensor_slices((split.X_train[pos_idx], split.y_train[pos_idx]))
    ds_neg = tf.data.Dataset.from_tensor_slices((split.X_train[neg_idx], split.y_train[neg_idx]))

    ds_pos = ds_pos.shuffle(min(len(pos_idx), 20_000), seed=config.seed, reshuffle_each_iteration=True).repeat()
    ds_neg = ds_neg.shuffle(min(len(neg_idx), 20_000), seed=config.seed, reshuffle_each_iteration=True).repeat()

    train_ds = tf.data.Dataset.sample_from_datasets(
        [ds_pos, ds_neg],
        weights=[float(config.positive_batch_rate), 1.0 - float(config.positive_batch_rate)],
        seed=config.seed,
    )
    train_ds = train_ds.batch(config.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((split.X_val, split.y_val))
    val_ds = val_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = int(np.ceil(len(split.y_train) / config.batch_size))
    return train_ds, val_ds, steps_per_epoch


def build_gru_model(*, sequence_length: int, n_features: int, base_rate_bias: float) -> "keras.Model":
    """Build the GRU closure-risk architecture."""
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


def compile_gru_model(model: "keras.Model", *, config: GruTrainingConfig) -> None:
    """Compile model with notebook-equivalent optimizer, loss, and metrics."""
    from tensorflow import keras

    try:
        optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            clipnorm=config.clipnorm,
        )
    except Exception:
        optimizer = keras.optimizers.Adam(
            learning_rate=config.learning_rate,
            clipnorm=config.clipnorm,
        )

    loss = keras.losses.BinaryCrossentropy(label_smoothing=config.label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def train_gru_model(
    split: TrainValidationSplit,
    *,
    config: GruTrainingConfig,
) -> tuple["keras.Model", dict[str, float]]:
    """Train GRU model and return fitted model + validation metrics."""
    from tensorflow import keras

    train_ds, val_ds, steps_per_epoch = build_balanced_training_datasets(split, config=config)

    pos_rate = float((split.y_train == 1).mean())
    epsilon = 1e-7
    base_rate_bias = float(np.log((pos_rate + epsilon) / (1.0 - pos_rate + epsilon)))

    model = build_gru_model(
        sequence_length=split.X_train.shape[1],
        n_features=split.X_train.shape[2],
        base_rate_bias=base_rate_bias,
    )
    compile_gru_model(model, config=config)

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=config.lr_plateau_factor,
            patience=config.lr_plateau_patience,
            min_lr=config.min_learning_rate,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    values = model.evaluate(val_ds, verbose=0)
    metrics = {name: float(value) for name, value in zip(model.metrics_names, values)}

    return model, metrics


def predict_window_probabilities(
    model: "keras.Model",
    features: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """Predict closure probabilities for sequence windows."""
    probabilities = model.predict(features, batch_size=batch_size, verbose=0).reshape(-1)
    return np.asarray(probabilities, dtype=float)


def compute_window_topk_metrics(
    *,
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    percentiles: Sequence[float] = (0.5, 1, 2, 5, 10, 15, 20),
) -> pd.DataFrame:
    """Compute top-K workload metrics at window level."""
    frame = metadata.copy()
    frame["p_closed"] = probabilities
    frame["y_true"] = labels.astype(int)
    frame = frame.sort_values("p_closed", ascending=False).reset_index(drop=True)

    total = len(frame)
    positives = int((frame["y_true"] == 1).sum())

    rows: list[dict[str, Any]] = []
    for percentile in percentiles:
        k = max(1, int(total * (float(percentile) / 100.0)))
        topk = frame.head(k)
        tp = int((topk["y_true"] == 1).sum())
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


def aggregate_business_risk(
    *,
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    global_last_month: pd.Timestamp,
    config: GruTrainingConfig,
) -> pd.DataFrame:
    """Aggregate window probabilities to business-level risk triage."""
    frame = metadata.copy().sort_values(["business_id", "end_month"]).reset_index(drop=True)
    frame["p_closed"] = probabilities
    frame["y_true"] = labels.astype(int)

    grouped = frame.groupby(["business_id", "status"], as_index=False).agg(
        end_month_last=("end_month", "max"),
        p_last=("p_closed", "last"),
        p_max=("p_closed", "max"),
        p_mean=("p_closed", "mean"),
        n_windows=("p_closed", "size"),
        y_business=("y_true", "max"),
    )

    recent_cutoff = global_last_month - pd.DateOffset(months=int(config.recent_k_months))
    recent_windows = frame[frame["end_month"] >= recent_cutoff].copy()

    recent_max = (
        recent_windows.sort_values(["business_id", "end_month"])
        .groupby("business_id", group_keys=False)
        .tail(3)
        .groupby("business_id", as_index=False)["p_closed"]
        .max()
        .rename(columns={"p_closed": "p_recent_max"})
    )

    overall_last3 = (
        frame.sort_values(["business_id", "end_month"])
        .groupby("business_id", group_keys=False)
        .tail(3)
        .groupby("business_id", as_index=False)["p_closed"]
        .max()
        .rename(columns={"p_closed": "p_last3_max"})
    )

    triage = grouped.merge(recent_max, on="business_id", how="left").merge(overall_last3, on="business_id", how="left")

    triage["risk_score"] = triage["p_recent_max"].fillna(triage["p_last3_max"]).fillna(triage["p_max"])
    triage["risk_bucket"] = pd.cut(
        triage["risk_score"],
        bins=list(config.risk_bins),
        labels=list(config.risk_labels),
        include_lowest=True,
    )

    return triage.sort_values("risk_score", ascending=False).reset_index(drop=True)


def save_gru_outputs(
    *,
    model: Any,
    feature_columns: Sequence[str],
    config: GruTrainingConfig,
    output_dir: Path | str,
    business_triage: pd.DataFrame,
    playbook: Iterable[Any] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Persist trained model, metadata, and business triage CSV outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_dir = output_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model_gru.keras"
    model.save(model_path)

    triage_path = output_path / "gru_business_triage.csv"
    top5_path = output_path / "gru_business_triage_top5pct.csv"
    top10_path = output_path / "gru_business_triage_top10pct.csv"

    business_triage.to_csv(triage_path, index=False)
    business_triage.head(max(1, int(0.05 * len(business_triage)))).to_csv(top5_path, index=False)
    business_triage.head(max(1, int(0.10 * len(business_triage)))).to_csv(top10_path, index=False)

    metadata_path = model_dir / "model_metadata.json"
    metadata = {
        "FEAT_COLS": list(feature_columns),
        "SEED": config.seed,
        "SEQ_LEN": config.sequence_length,
        "H": config.horizon_months,
        "INACTIVE_K": config.inactive_months_for_zombie,
        "MIN_ACTIVE_MONTHS": config.min_active_months,
        "MIN_REVIEWS_IN_WINDOW": config.min_reviews_in_window,
        "POS_BATCH_RATE": config.positive_batch_rate,
        "BATCH_SIZE": config.batch_size,
        "EPOCHS": config.epochs,
        "LR": config.learning_rate,
        "WEIGHT_DECAY": config.weight_decay,
        "LABEL_SMOOTHING": config.label_smoothing,
        "CLIPNORM": config.clipnorm,
        "REQ": sorted(REQUIRED_MONTHLY_COLUMNS),
        "NUM_COLS": list(BASE_FEATURE_COLUMNS),
        "BASE_FEATS": list(BASE_FEATURE_COLUMNS),
        "risk_bins": list(config.risk_bins),
        "risk_labels": list(config.risk_labels),
        "PLAYBOOK": list(playbook) if playbook is not None else [],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return model_path, metadata_path, triage_path, top5_path, top10_path


def train_gru_end_to_end(
    monthly_panel: pd.DataFrame,
    *,
    output_dir: Path | str,
    config: GruTrainingConfig | None = None,
    train_fraction: float = 0.8,
    playbook: Iterable[Any] | None = None,
) -> GruTrainingArtifacts:
    """Run the full GRU workflow from monthly panel to saved artifacts."""
    config = config or GruTrainingConfig()
    set_global_seed(config.seed)

    feature_panel, feature_columns, global_last_month = engineer_feature_panel(monthly_panel, config)
    windows = build_windows_and_labels(
        feature_panel,
        feature_columns=feature_columns,
        config=config,
        global_last_month=global_last_month,
    )
    split = split_windows_by_business(windows, train_fraction=train_fraction, seed=config.seed)

    model, validation_metrics = train_gru_model(split, config=config)

    probabilities = predict_window_probabilities(model, split.X_val)
    window_topk = compute_window_topk_metrics(
        metadata=split.metadata_val,
        probabilities=probabilities,
        labels=split.y_val,
    )
    business_triage = aggregate_business_risk(
        metadata=split.metadata_val,
        probabilities=probabilities,
        labels=split.y_val,
        global_last_month=windows.global_last_month,
        config=config,
    )

    model_path, metadata_path, triage_path, top5_path, top10_path = save_gru_outputs(
        model=model,
        feature_columns=feature_columns,
        config=config,
        output_dir=output_dir,
        business_triage=business_triage,
        playbook=playbook,
    )

    return GruTrainingArtifacts(
        model_path=model_path,
        metadata_path=metadata_path,
        triage_path=triage_path,
        top5_path=top5_path,
        top10_path=top10_path,
        validation_metrics=validation_metrics,
        window_topk_metrics=window_topk,
        business_triage=business_triage,
    )
