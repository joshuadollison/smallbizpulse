from __future__ import annotations

import pytest

from app import create_app



def _make_client(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *,
    enable_live_fallback: bool,
):
    monkeypatch.setenv("SBP_ARTIFACT_ROOT", str(artifact_bundle["artifact_root"]))
    monkeypatch.setenv("SBP_MODEL_DIR", str(artifact_bundle["model_dir"]))
    monkeypatch.setenv("SBP_SENTIMENT_DIR", str(artifact_bundle["sentiment_dir"]))
    monkeypatch.setenv("SBP_ENABLE_LIVE_FALLBACK", "true" if enable_live_fallback else "false")
    monkeypatch.setenv("SBP_PREFER_LIVE_SCORING", "true")
    monkeypatch.delenv("SBP_YELP_DATA_DIR", raising=False)

    app = create_app()
    app.testing = True
    return app.test_client()


def test_search_endpoint_returns_ranked_candidates(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(artifact_bundle, monkeypatch, enable_live_fallback=True)

    response = client.get("/api/search", query_string={"name": "alpha grill"})
    assert response.status_code == 200

    payload = response.get_json()
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["business_id"] == "b1"


def test_search_endpoint_can_include_unscorable_candidates(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(artifact_bundle, monkeypatch, enable_live_fallback=True)

    response = client.get(
        "/api/search",
        query_string={"name": "alpha grill", "include_unscorable": "1"},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["business_id"] == "b1"


def test_score_endpoint_returns_artifact_payload(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(artifact_bundle, monkeypatch, enable_live_fallback=True)

    response = client.post("/api/score", json={"business_id": "b1"})
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["availability"] == "scored"
    assert payload["scoring_mode"] == "artifact"
    assert payload["business_id"] == "b1"
    assert "risk_score" in payload
    assert "themes_top3" in payload
    assert "recommendations_top3" in payload


def test_score_endpoint_unscored_returns_not_scored_when_fallback_disabled(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(artifact_bundle, monkeypatch, enable_live_fallback=False)

    response = client.post("/api/score", json={"business_id": "not-in-artifacts"})
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["availability"] == "not_scored_yet"
    assert payload["not_scored_reason"] == "live_fallback_disabled"


def test_score_endpoint_force_live_inference_bypasses_artifact(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(artifact_bundle, monkeypatch, enable_live_fallback=False)

    response = client.post("/api/score", json={"business_id": "b1", "force_live_inference": True})
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["availability"] == "not_scored_yet"
    assert payload["scoring_mode"] == "live_fallback"
    assert payload["not_scored_reason"] == "live_fallback_disabled"
    assert payload["business_id"] == "b1"


def test_health_endpoint_includes_runtime_flags(
    artifact_bundle: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(artifact_bundle, monkeypatch, enable_live_fallback=True)

    response = client.get("/api/health")
    assert response.status_code == 200

    payload = response.get_json()
    checks = payload["checks"]
    assert "tensorflow_available" in checks
    assert "tensorflow_runtime_available" in checks
    assert "live_fallback_enabled" in checks
    assert "prefer_live_scoring" in checks
    assert "artifact_root_exists" in checks
