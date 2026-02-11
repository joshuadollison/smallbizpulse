# SmallBizPulse Dashboard

A placeholder Flask dashboard you can deploy to Render while the modeling work is in progress. It serves sample data from `data/sample_dashboard.json` so the UI works end-to-end before wiring real predictions.

## Structure
- `app/` - Flask app, templates, and static assets
- `data/` - small JSON payload used by the placeholder API
- `requirements.txt` - minimal deps (Flask + gunicorn + python-dotenv)
- `Procfile` - `web: gunicorn app.main:app` for Render/Heroku style hosting

## Run locally
```bash
cd dashboard
pip install -r requirements.txt
flask --app app run
# open http://127.0.0.1:5000
```

## Deploy to Render
1) Push this repo to GitHub.  
2) In Render, create a **Web Service** and set **Root Directory** to `dashboard`.  
3) Build command: `pip install -r requirements.txt`  
4) Start command: `gunicorn app.main:app`  
5) Optional env vars: `APP_NAME`, `APP_TAGLINE`, `PORT`.

## Swap in real data
- Replace `data/sample_dashboard.json` with your export (same keys: `kpis`, `trend`, `topics`, `alerts`).
- Or update the API endpoints in `app/__init__.py` to query your model or database.
