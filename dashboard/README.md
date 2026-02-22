# SmallBizPulse Dashboard

Streamlit dashboard for restaurant closure-risk scoring (backed by shared service logic in `app/`).

The runtime is live-first (with artifact fallback) when local Yelp data + optional ML dependencies are available.

## Structure
- `streamlit_app.py` Streamlit UI entrypoint
- `app/` shared backend service logic (and Flask API compatibility layer)
- `requirements.txt` core dashboard dependencies
- `Procfile` Render/Heroku-style startup (`web: streamlit run ...`)

## API (compatibility)
- `GET /api/health` runtime/dependency/data readiness flags
- `GET /api/search?name=<str>&city=<opt>&state=<opt>&limit=<opt>` business candidate lookup
- `POST /api/score` body: `{"business_id": "..."}` (preferred) or name/city/state

## Runtime Configuration
- `SBP_ARTIFACT_ROOT` default: `models/artifacts`
- `SBP_MODEL_DIR` default: `models/artifacts/models`
- `SBP_SENTIMENT_DIR` default: `models/artifacts/transformer_sentiment_distilbert`
- `SBP_YELP_DATA_DIR` optional path to Yelp JSON files for live fallback
- `SBP_ENABLE_LIVE_FALLBACK` default: `true`
- `SBP_PREFER_LIVE_SCORING` default: `true`
- `SBP_APP_SERVER` default: `streamlit` (`flask` optional)
- `APP_NAME`, `APP_TAGLINE`, `PORT` optional UI/server settings

Live fallback expects these files in `SBP_YELP_DATA_DIR`:
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`

## Run locally
```bash
./scripts/run_dashboard.sh
# open http://127.0.0.1:8501
```

## Deploy to Render
1. Push this repo to GitHub.
2. Create a Render Web Service with root directory `dashboard`.
3. Build command: `pip install -r requirements.txt`.
4. Start command: `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`.
5. Configure env vars as needed.
