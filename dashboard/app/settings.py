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
    v2_bundle_dir: Path | None
    yelp_data_dir: Path | None
    enable_live_fallback: bool
    prefer_live_scoring: bool
    gru_worker_timeout_seconds: int
    gru_worker_conda_env: str | None
    gru_worker_python: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        repo_root = Path(__file__).resolve().parents[2]
        default_v2_component1_dir = (
            repo_root / "models" / "v2_artifacts" / "v2_artifacts_colab_bundle" / "component1_survival"
        )
        default_v2_component2_dir = (
            repo_root / "models" / "v2_artifacts" / "v2_artifacts_colab_bundle" / "component2_topics"
        )

        artifact_root = Path(
            os.getenv("SBP_ARTIFACT_ROOT", str(repo_root / "models" / "artifacts"))
        )
        model_dir = Path(
            os.getenv("SBP_MODEL_DIR", str(default_v2_component1_dir))
        )
        sentiment_dir = Path(
            os.getenv(
                "SBP_SENTIMENT_DIR",
                str(default_v2_component2_dir),
            )
        )

        v2_bundle_raw = os.getenv("SBP_V2_BUNDLE_DIR")
        artifact_root_raw = os.getenv("SBP_ARTIFACT_ROOT")
        custom_artifact_root = bool(artifact_root_raw)
        if v2_bundle_raw:
            v2_bundle_dir = Path(v2_bundle_raw).expanduser()
        else:
            preferred_v2_bundle = repo_root / "models" / "v2_artifacts" / "v2_artifacts_colab_bundle"
            fallback_v2_bundle = repo_root / "models" / "v2_artifacts"
            if custom_artifact_root:
                v2_bundle_dir = None
            elif preferred_v2_bundle.exists():
                v2_bundle_dir = preferred_v2_bundle
            elif fallback_v2_bundle.exists():
                v2_bundle_dir = fallback_v2_bundle
            else:
                v2_bundle_dir = preferred_v2_bundle

        yelp_raw = os.getenv("SBP_YELP_DATA_DIR")
        if yelp_raw:
            yelp_data_dir = Path(yelp_raw).expanduser()
        else:
            default_yelp_dir = repo_root / "data" / "external" / "yelp_dataset_new"
            yelp_data_dir = default_yelp_dir if default_yelp_dir.exists() else None
        worker_conda_env = os.getenv("SBP_GRU_WORKER_CONDA_ENV")
        worker_python = os.getenv("SBP_GRU_WORKER_PYTHON")

        return cls(
            repo_root=repo_root,
            artifact_root=artifact_root.expanduser(),
            model_dir=model_dir.expanduser(),
            sentiment_dir=sentiment_dir.expanduser(),
            v2_bundle_dir=v2_bundle_dir,
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
