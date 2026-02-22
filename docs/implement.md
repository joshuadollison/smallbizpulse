Implemented end-to-end.

**What I changed**
- Replaced placeholder backend with artifact-first scoring + optional live fallback service:
  - `dashboard/app/service.py`
  - `dashboard/app/settings.py`
- Rewired Flask app with new APIs and health/runtime checks:
  - `dashboard/app/__init__.py`
- Added required endpoints:
  - `GET /api/search`
  - `POST /api/score`
  - extended `GET /api/health`
- Replaced placeholder UI with search/select/score workflow:
  - `dashboard/app/templates/index.html`
  - `dashboard/app/static/app.js`
  - `dashboard/app/static/styles.css`
- Fixed deploy entrypoint and dependency set:
  - `dashboard/Procfile` now `web: gunicorn app:app`
  - `dashboard/requirements.txt` now includes `pandas`, `numpy`, `joblib` (core runtime deps)
- Updated dashboard docs for new runtime/env vars/API:
  - `dashboard/README.md`
- Added unit and API tests:
  - `dashboard/tests/conftest.py`
  - `dashboard/tests/test_service.py`
  - `dashboard/tests/test_api.py`

**Behavior now**
- Artifact-backed scoring works immediately from `models/artifacts`.
- Missing problem/recommendation rows now return `themes_top3=["needs_review"]` + fallback recommendation note.
- Unscored businesses return `availability="not_scored_yet"` with machine-friendly `not_scored_reason`.
- Live fallback is lazy/optional and only attempts when enabled + dependencies/data exist.

**Validation run**
- Syntax check passed for new/edited Python files.
- Test suite passed: `pytest -q dashboard/tests` -> `9 passed`.

**Important note**
- TensorFlow is not installed in your current environment, so live fallback will currently return `not_scored_yet` with a dependency reason until TF is installed and `SBP_YELP_DATA_DIR` is set.

**Natural next steps**
1. Install TensorFlow in the dashboard environment and set `SBP_YELP_DATA_DIR` to your Yelp JSON directory.
2. Run `cd dashboard && flask --app app run` and test `/api/search` + `/api/score` on an unscored business.