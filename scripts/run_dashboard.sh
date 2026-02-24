#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/dashboard"

# Default to the known-good TensorFlow environment for live fallback.
CONDA_ENV="${SBP_CONDA_ENV:-sbp311}"
HOST="${HOST:-127.0.0.1}"
APP_SERVER="${SBP_APP_SERVER:-streamlit}"
if [[ "${APP_SERVER}" == "flask" ]]; then
  DEFAULT_PORT="5050"
else
  DEFAULT_PORT="8501"
fi
PORT="${PORT:-${DEFAULT_PORT}}"
DEFAULT_YELP_DIR="${ROOT_DIR}/data/external/yelp_dataset_new"
DEFAULT_V2_BUNDLE_DIR="${ROOT_DIR}/models/v2_artifacts/v2_artifacts_colab_bundle"

export SBP_ENABLE_LIVE_FALLBACK="${SBP_ENABLE_LIVE_FALLBACK:-true}"
export SBP_PREFER_LIVE_SCORING="${SBP_PREFER_LIVE_SCORING:-true}"
export SBP_YELP_DATA_DIR="${SBP_YELP_DATA_DIR:-${DEFAULT_YELP_DIR}}"
export SBP_V2_BUNDLE_DIR="${SBP_V2_BUNDLE_DIR:-${DEFAULT_V2_BUNDLE_DIR}}"

if [[ ! -f "${SBP_YELP_DATA_DIR}/yelp_academic_dataset_business.json" ]] || [[ ! -f "${SBP_YELP_DATA_DIR}/yelp_academic_dataset_review.json" ]]; then
  echo "Warning: Yelp files not found in SBP_YELP_DATA_DIR=${SBP_YELP_DATA_DIR}"
  echo "Live scoring will return not_scored_yet until those files are present."
fi

echo "Running dashboard with conda env: ${CONDA_ENV}"
echo "SBP_APP_SERVER=${APP_SERVER}"
echo "SBP_ENABLE_LIVE_FALLBACK=${SBP_ENABLE_LIVE_FALLBACK}"
echo "SBP_PREFER_LIVE_SCORING=${SBP_PREFER_LIVE_SCORING}"
echo "SBP_YELP_DATA_DIR=${SBP_YELP_DATA_DIR}"
echo "SBP_V2_BUNDLE_DIR=${SBP_V2_BUNDLE_DIR}"
echo "URL: http://${HOST}:${PORT}"
if [[ "${APP_SERVER}" == "flask" ]]; then
  exec conda run -n "${CONDA_ENV}" flask --app app run --host "${HOST}" --port "${PORT}"
fi
exec conda run -n "${CONDA_ENV}" streamlit run streamlit_app.py --server.address "${HOST}" --server.port "${PORT}" --server.headless true
