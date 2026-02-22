"""Flask app for SmallBizPulse artifact-first scoring with live fallback."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from .service import ServiceError, SmallBizPulseService
from .settings import Settings

# Load optional .env for local runs.
load_dotenv()


def _build_service() -> SmallBizPulseService:
    settings = Settings.from_env()
    return SmallBizPulseService(settings)


def _parse_bool_payload(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    app.config["APP_NAME"] = os.getenv("APP_NAME", "SmallBizPulse Dashboard")
    app.config["APP_TAGLINE"] = os.getenv(
        "APP_TAGLINE", "Closure-risk scoring, diagnostics, and action planning for restaurants"
    )

    service: SmallBizPulseService | None = None
    service_error: str | None = None
    try:
        service = _build_service()
    except Exception as exc:  # pragma: no cover - startup robustness guard
        service_error = str(exc)

    @app.route("/")
    def index() -> str:
        return render_template(
            "index.html",
            app_name=app.config["APP_NAME"],
            tagline=app.config["APP_TAGLINE"],
        )

    @app.route("/api/health")
    def health() -> Any:
        checks = service.health() if service is not None else {}
        return jsonify(
            {
                "status": "ok",
                "app": app.config["APP_NAME"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "checks": checks,
                "service_ready": service is not None,
                "service_error": service_error,
            }
        )

    @app.route("/api/search")
    def search() -> Any:
        if service is None:
            return jsonify({"error": "service_unavailable", "message": service_error}), 503

        name = request.args.get("name", "")
        city = request.args.get("city")
        state = request.args.get("state")
        try:
            limit = int(request.args.get("limit", 10))
        except ValueError:
            limit = 10
        include_unscorable = _parse_bool_payload(
            request.args.get("include_unscorable"),
            default=False,
        )

        if not name.strip():
            return jsonify([])

        results = service.search_businesses(
            name=name,
            city=city,
            state=state,
            limit=limit,
            scorable_only=not include_unscorable,
        )
        return jsonify(results)

    @app.route("/api/score", methods=["POST"])
    def score() -> Any:
        if service is None:
            return jsonify({"error": "service_unavailable", "message": service_error}), 503

        payload = request.get_json(silent=True) or {}
        force_live_inference = _parse_bool_payload(payload.get("force_live_inference"), default=False)

        business_id = str(payload.get("business_id") or "").strip()
        if not business_id:
            name = str(payload.get("name") or "").strip()
            city = str(payload.get("city") or "").strip() or None
            state = str(payload.get("state") or "").strip() or None

            if not name:
                return (
                    jsonify(
                        {
                            "error": "business_id_or_name_required",
                            "message": "Provide business_id, or provide name (with optional city/state).",
                        }
                    ),
                    400,
                )

            candidates = service.search_businesses(name=name, city=city, state=state, limit=5)
            if not candidates:
                return (
                    jsonify(
                        {
                            "error": "business_not_found",
                            "message": "No businesses matched the provided name/filter.",
                            "result": {
                                "business_id": None,
                                "name": name,
                                "city": city,
                                "state": state,
                                "status": "Unknown",
                                "total_reviews": None,
                                "last_review_month": None,
                                "risk_score": None,
                                "risk_bucket": None,
                                "recent_windows": [],
                                "themes_top3": [],
                                "problem_keywords": None,
                                "evidence_reviews": [],
                                "recommendations_top3": [],
                                "recommendation_notes": None,
                                "scoring_mode": "artifact",
                                "availability": "not_scored_yet",
                                "not_scored_reason": "business_not_found",
                            },
                        }
                    ),
                    404,
                )

            # No auto-pick on ambiguous names unless exactly one candidate remains.
            if len(candidates) > 1:
                return (
                    jsonify(
                        {
                            "error": "ambiguous_name",
                            "message": "Multiple businesses matched. Use /api/search and choose business_id.",
                            "candidates": candidates,
                        }
                    ),
                    409,
                )

            business_id = candidates[0]["business_id"]

        result = service.score_business(
            business_id,
            force_live_inference=force_live_inference,
        )
        return jsonify(result)

    @app.errorhandler(ServiceError)
    def handle_service_error(exc: ServiceError) -> Any:
        return jsonify({"error": "service_error", "message": str(exc)}), 500

    return app


# For `python -m flask --app app run` and gunicorn `app:app`.
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5050)), debug=True)
