"""Minimal Flask dashboard placeholder for SmallBizPulse.

The goal is to be deployable on Render with no external data sources yet.
All numbers come from `dashboard/data/sample_dashboard.json` so the UI works
end-to-end while modeling work is in progress.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template

# Load optional .env for local runs
load_dotenv()

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_dashboard.json"

# Lightweight fallback so the app never crashes if the file is missing
DEFAULT_DATA: Dict[str, Any] = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "kpis": [],
    "trend": [],
    "topics": [],
    "alerts": [],
}


def load_dashboard_data() -> Dict[str, Any]:
    """Load placeholder metrics from disk with a safe fallback."""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_DATA
    except json.JSONDecodeError:
        return DEFAULT_DATA


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    app.config["APP_NAME"] = os.getenv("APP_NAME", "SmallBizPulse Dashboard")
    app.config["APP_TAGLINE"] = os.getenv(
        "APP_TAGLINE", "Early warning & intervention signals for independent restaurants"
    )

    @app.route("/")
    def index():
        data = load_dashboard_data()
        return render_template(
            "index.html",
            app_name=app.config["APP_NAME"],
            tagline=app.config["APP_TAGLINE"],
            generated_at=data.get("generated_at"),
        )

    @app.route("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "app": app.config["APP_NAME"],
                "generated_at": load_dashboard_data().get("generated_at"),
            }
        )

    @app.route("/api/kpis")
    def kpis():
        return jsonify(load_dashboard_data().get("kpis", []))

    @app.route("/api/trend")
    def trend():
        return jsonify(load_dashboard_data().get("trend", []))

    @app.route("/api/topics")
    def topics():
        return jsonify(load_dashboard_data().get("topics", []))

    @app.route("/api/alerts")
    def alerts():
        return jsonify(load_dashboard_data().get("alerts", []))

    return app


# For `python -m flask --app app run` convenience
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
