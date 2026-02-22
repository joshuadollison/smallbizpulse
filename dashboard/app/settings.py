from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    artifact_root: Path
    model_dir: Path
    sentiment_dir: Path
    yelp_data_dir: Path | None
    enable_live_fallback: bool
    prefer_live_scoring: bool
    gru_worker_timeout_seconds: int
    gru_worker_conda_env: str | None
    gru_worker_python: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        repo_root = Path(__file__).resolve().parents[2]

        artifact_root = Path(
            os.getenv("SBP_ARTIFACT_ROOT", str(repo_root / "models" / "artifacts"))
        )
        model_dir = Path(
            os.getenv("SBP_MODEL_DIR", str(repo_root / "models" / "artifacts" / "models"))
        )
        sentiment_dir = Path(
            os.getenv(
                "SBP_SENTIMENT_DIR",
                str(repo_root / "models" / "artifacts" / "transformer_sentiment_distilbert"),
            )
        )

        yelp_raw = os.getenv("SBP_YELP_DATA_DIR")
        yelp_data_dir = Path(yelp_raw).expanduser() if yelp_raw else None
        worker_conda_env = os.getenv("SBP_GRU_WORKER_CONDA_ENV")
        worker_python = os.getenv("SBP_GRU_WORKER_PYTHON")

        return cls(
            repo_root=repo_root,
            artifact_root=artifact_root.expanduser(),
            model_dir=model_dir.expanduser(),
            sentiment_dir=sentiment_dir.expanduser(),
            yelp_data_dir=yelp_data_dir,
            enable_live_fallback=_parse_bool(
                os.getenv("SBP_ENABLE_LIVE_FALLBACK"),
                default=True,
            ),
            prefer_live_scoring=_parse_bool(
                os.getenv("SBP_PREFER_LIVE_SCORING"),
                default=True,
            ),
            gru_worker_timeout_seconds=_parse_int(
                os.getenv("SBP_GRU_WORKER_TIMEOUT_SECONDS"),
                default=240,
            ),
            gru_worker_conda_env=worker_conda_env.strip() if worker_conda_env else None,
            gru_worker_python=worker_python.strip() if worker_python else None,
        )
