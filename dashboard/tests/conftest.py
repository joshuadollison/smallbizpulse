from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))


@pytest.fixture
def artifact_bundle(tmp_path: Path) -> dict[str, Path]:
    artifact_root = tmp_path / "artifacts"
    model_dir = tmp_path / "models"
    sentiment_dir = tmp_path / "sentiment"

    artifact_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    sentiment_dir.mkdir(parents=True, exist_ok=True)

    (artifact_root / "A_closure_risk_table.csv").write_text(
        "\n".join(
            [
                "business_id,name,city,state,status,risk_score,risk_bucket,review_count,end_month_last",
                "b1,Alpha Grill,Phoenix,AZ,Open,0.72,elevated,120,2018-09-01",
                "b2,Bravo Tacos,Tempe,AZ,Open,0.31,low,50,2018-08-01",
            ]
        ),
        encoding="utf-8",
    )

    (artifact_root / "final_closure_risk_problems_recommendations.csv").write_text(
        "\n".join(
            [
                "business_id,problem_keywords,recommendations_top3,themes_top3",
                "b1,wait slow,\"1. Improve staffing  2. Tighten prep tickets\",\"service_speed, order_accuracy\"",
                "b2,,,",
            ]
        ),
        encoding="utf-8",
    )

    (artifact_root / "gru_business_triage.csv").write_text(
        "\n".join(
            [
                "business_id,p_recent_max,p_last3_max,p_last,p_max,p_mean,n_windows,end_month_last",
                "b1,0.72,0.71,0.68,0.74,0.55,6,2018-09-01",
                "b2,0.31,0.30,0.29,0.33,0.24,4,2018-08-01",
            ]
        ),
        encoding="utf-8",
    )

    metadata = {
        "risk_bins": [0.0, 0.5, 0.65, 0.75, 0.85, 1.0],
        "risk_labels": ["low", "medium", "elevated", "high", "very_high"],
        "PLAYBOOK": [
            [
                "service_speed",
                ["slow", "wait", "waiting", "minutes"],
                "Service speed recommendation",
            ],
            [
                "order_accuracy",
                ["wrong", "missing", "forgot"],
                "Order accuracy recommendation",
            ],
        ],
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    return {
        "artifact_root": artifact_root,
        "model_dir": model_dir,
        "sentiment_dir": sentiment_dir,
    }
