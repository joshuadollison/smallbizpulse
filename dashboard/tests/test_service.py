from __future__ import annotations

import pandas as pd
import pytest

from app.service import (
    RECO_FALLBACK_NOTE,
    LiveScoringOutcome,
    ServiceError,
    SmallBizPulseService,
    assign_risk_bucket,
    normalize_text,
)
from app.settings import Settings


def _service_from_bundle(
    bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *,
    enable_live_fallback: bool = True,
) -> SmallBizPulseService:
    monkeypatch.setenv("SBP_ARTIFACT_ROOT", str(bundle["artifact_root"]))
    monkeypatch.setenv("SBP_MODEL_DIR", str(bundle["model_dir"]))
    monkeypatch.setenv("SBP_SENTIMENT_DIR", str(bundle["sentiment_dir"]))
    monkeypatch.setenv("SBP_ENABLE_LIVE_FALLBACK", "true" if enable_live_fallback else "false")
    monkeypatch.setenv("SBP_PREFER_LIVE_SCORING", "true")
    monkeypatch.delenv("SBP_YELP_DATA_DIR", raising=False)

    return SmallBizPulseService(Settings.from_env())


def test_normalize_text_casefold_and_punctuation() -> None:
    assert normalize_text("  FIVE-Guys!! ") == "five guys"


def test_assign_risk_bucket_with_valid_bins() -> None:
    bins = [0.0, 0.5, 0.65, 0.75, 0.85, 1.0]
    labels = ["low", "medium", "elevated", "high", "very_high"]
    assert assign_risk_bucket(0.49, bins, labels) == "low"
    assert assign_risk_bucket(0.66, bins, labels) == "elevated"


def test_search_ranking_prefers_exact_match(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_from_bundle(artifact_bundle, monkeypatch)

    results = service.search_businesses("Alpha Grill", city="Phoenix", state="AZ", limit=10)
    assert results
    assert results[0]["business_id"] == "b1"
    assert results[0]["risk_available"] is True


def test_search_filters_unscorable_candidates_by_default(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_from_bundle(artifact_bundle, monkeypatch)

    candidates = pd.DataFrame(
        [
            {
                "business_id": "u1",
                "name": "Live Spot One",
                "city": "Phoenix",
                "state": "AZ",
                "status": "Open",
                "total_reviews": 20,
                "last_review_month": "2018-01-01",
                "risk_available": False,
            },
            {
                "business_id": "u2",
                "name": "Live Spot Two",
                "city": "Phoenix",
                "state": "AZ",
                "status": "Open",
                "total_reviews": 20,
                "last_review_month": "2018-01-01",
                "risk_available": False,
            },
        ]
    )
    monkeypatch.setattr(service, "_candidate_df", lambda: candidates)
    monkeypatch.setattr(
        service,
        "_candidate_is_scorable",
        lambda payload: payload["business_id"] == "u2",
    )

    filtered = service.search_businesses("live spot", limit=10, scorable_only=True)
    unfiltered = service.search_businesses("live spot", limit=10, scorable_only=False)

    assert [item["business_id"] for item in filtered] == ["u2"]
    assert {item["business_id"] for item in unfiltered} == {"u1", "u2"}


def test_search_does_not_return_weak_fuzzy_noise(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_from_bundle(artifact_bundle, monkeypatch)

    candidates = pd.DataFrame(
        [
            {
                "business_id": "f1",
                "name": "Five Guys",
                "city": "Harahan",
                "state": "LA",
                "status": "Open",
                "total_reviews": 89,
                "last_review_month": "2018-01-01",
                "risk_available": False,
            },
            {
                "business_id": "e1",
                "name": "El Maguey",
                "city": "Harahan",
                "state": "LA",
                "status": "Open",
                "total_reviews": 20,
                "last_review_month": "2018-01-01",
                "risk_available": False,
            },
        ]
    )
    monkeypatch.setattr(service, "_candidate_df", lambda: candidates)
    monkeypatch.setattr(service, "_candidate_is_scorable", lambda _payload: True)

    results = service.search_businesses("five guys", limit=10, scorable_only=False)
    assert [item["business_id"] for item in results] == ["f1"]


def test_missing_problem_and_recs_use_fallback(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_from_bundle(artifact_bundle, monkeypatch)

    result = service.score_business("b2")
    assert result["availability"] == "scored"
    assert result["themes_top3"] == ["needs_review"]
    assert result["recommendations_top3"] == [RECO_FALLBACK_NOTE]


def test_force_live_inference_bypasses_artifact_short_circuit(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_from_bundle(artifact_bundle, monkeypatch)

    def fake_live(_business_id: str) -> LiveScoringOutcome:
        return LiveScoringOutcome(
            ok=True,
            payload={
                "business_id": "b1",
                "name": "Alpha Grill",
                "city": "Phoenix",
                "state": "AZ",
                "status": "Open",
                "total_reviews": 222,
                "last_review_month": "2018-09-01",
                "risk_score": 0.88,
                "risk_bucket": "very_high",
                "recent_windows": [],
                "themes_top3": ["service_speed"],
                "problem_keywords": "slow wait",
                "evidence_reviews": [],
                "recommendations_top3": ["hire more front-of-house staff"],
                "recommendation_notes": None,
                "scoring_mode": "live_fallback",
                "availability": "scored",
                "not_scored_reason": None,
            },
        )

    monkeypatch.setattr(service, "_score_live_fallback", fake_live)

    result = service.score_business("b1", force_live_inference=True)
    assert result["availability"] == "scored"
    assert result["scoring_mode"] == "live_fallback"
    assert result["risk_score"] == 0.88


def test_live_first_prefers_live_without_force_flag(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_from_bundle(artifact_bundle, monkeypatch)

    def fake_live(_business_id: str) -> LiveScoringOutcome:
        return LiveScoringOutcome(
            ok=True,
            payload={
                "business_id": "b1",
                "name": "Alpha Grill",
                "city": "Phoenix",
                "state": "AZ",
                "status": "Open",
                "total_reviews": 222,
                "last_review_month": "2018-09-01",
                "risk_score": 0.88,
                "risk_bucket": "very_high",
                "recent_windows": [],
                "themes_top3": ["service_speed"],
                "problem_keywords": "slow wait",
                "evidence_reviews": [],
                "recommendations_top3": ["hire more front-of-house staff"],
                "recommendation_notes": None,
                "scoring_mode": "live_fallback",
                "availability": "scored",
                "not_scored_reason": None,
            },
        )

    monkeypatch.setattr(service, "_score_live_fallback", fake_live)

    result = service.score_business("b1")
    assert result["availability"] == "scored"
    assert result["scoring_mode"] == "live_fallback"
    assert result["risk_score"] == 0.88


def test_force_live_inference_returns_not_scored_when_live_unavailable(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_from_bundle(artifact_bundle, monkeypatch, enable_live_fallback=False)

    result = service.score_business("b1", force_live_inference=True)
    assert result["availability"] == "not_scored_yet"
    assert result["scoring_mode"] == "live_fallback"
    assert result["not_scored_reason"] == "live_fallback_disabled"
    assert result["name"] == "Alpha Grill"


def test_force_live_inference_uses_live_identity_on_not_scored(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_from_bundle(artifact_bundle, monkeypatch)

    def fake_live(_business_id: str) -> LiveScoringOutcome:
        return LiveScoringOutcome(
            ok=False,
            payload={
                "identity": {
                    "business_id": "b1",
                    "name": "Alpha Grill",
                    "city": "Phoenix",
                    "state": "AZ",
                    "status": "Open",
                    "total_reviews": 18,
                    "last_review_month": "2018-07-01",
                }
            },
            reason="insufficient_history_for_windows_live_data_mismatch",
        )

    monkeypatch.setattr(service, "_score_live_fallback", fake_live)

    result = service.score_business("b1", force_live_inference=True)
    assert result["availability"] == "not_scored_yet"
    assert result["scoring_mode"] == "live_fallback"
    assert result["not_scored_reason"] == "insufficient_history_for_windows_live_data_mismatch"
    assert result["total_reviews"] == 18
    assert result["last_review_month"] == "2018-07-01"


def test_inference_windows_right_censor_open_latest_horizon(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_from_bundle(artifact_bundle, monkeypatch)

    monthly_df = pd.DataFrame(
        {
            "business_id": ["live-1"] * 12,
            "status": ["Open"] * 12,
            "month": pd.date_range("2018-01-01", periods=12, freq="MS"),
            "review_count": [1] * 12,
            "avg_stars": [3.5] * 12,
            "tx_sent_mean": [0.5] * 12,
            "tx_sent_std": [0.1] * 12,
            "tx_neg_share": [0.5] * 12,
            "tx_pos_share": [0.5] * 12,
        }
    )

    service._global_last_month = pd.Timestamp("2019-05-01")  # noqa: SLF001 - test setup

    windows = service._build_inference_windows(  # noqa: SLF001 - unit test for inference contract
        monthly_df,
        {"business_id": "live-1", "status": "Open"},
    )
    assert windows is None


def test_inference_windows_include_open_when_horizon_observed(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_from_bundle(artifact_bundle, monkeypatch)

    monthly_df = pd.DataFrame(
        {
            "business_id": ["live-1"] * 12,
            "status": ["Open"] * 12,
            "month": pd.date_range("2018-01-01", periods=12, freq="MS"),
            "review_count": [1] * 12,
            "avg_stars": [3.5] * 12,
            "tx_sent_mean": [0.5] * 12,
            "tx_sent_std": [0.1] * 12,
            "tx_neg_share": [0.5] * 12,
            "tx_pos_share": [0.5] * 12,
        }
    )

    service._global_last_month = pd.Timestamp("2019-11-01")  # noqa: SLF001 - test setup

    windows = service._build_inference_windows(  # noqa: SLF001 - unit test for inference contract
        monthly_df,
        {"business_id": "live-1", "status": "Open"},
    )
    assert windows is not None
    assert len(windows[1]) >= 1


def test_loader_validates_required_columns(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_root = tmp_path / "artifacts"
    model_dir = tmp_path / "models"
    sentiment_dir = tmp_path / "sentiment"

    artifact_root.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    sentiment_dir.mkdir(parents=True)

    # Missing required 'risk_score' column.
    (artifact_root / "A_closure_risk_table.csv").write_text(
        "business_id,name,city,state,status,risk_bucket,review_count,end_month_last\n"
        "x1,Broken Place,Phoenix,AZ,Open,low,10,2018-01-01\n",
        encoding="utf-8",
    )
    (artifact_root / "final_closure_risk_problems_recommendations.csv").write_text(
        "business_id,problem_keywords,recommendations_top3,themes_top3\n"
        "x1,,,\n",
        encoding="utf-8",
    )
    (model_dir / "model_metadata.json").write_text(
        '{"risk_bins":[0.0,0.5,1.0],"risk_labels":["low","high"],"PLAYBOOK":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("SBP_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("SBP_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("SBP_SENTIMENT_DIR", str(sentiment_dir))
    monkeypatch.setenv("SBP_ENABLE_LIVE_FALLBACK", "false")

    with pytest.raises(ServiceError):
        SmallBizPulseService(Settings.from_env())
