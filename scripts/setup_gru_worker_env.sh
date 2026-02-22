#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-sbp-gru-legacy311}"

echo "Creating conda env: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.11 pip

echo "Installing TensorFlow runtime"
conda run -n "${ENV_NAME}" pip install "tensorflow==2.15.1"

echo "Installing Keras 3.10.0 (model-compat override)"
conda run -n "${ENV_NAME}" pip install --no-deps --upgrade "keras==3.10.0"

echo "Installing Keras runtime extras"
conda run -n "${ENV_NAME}" pip install optree namex rich markdown-it-py mdurl pygments

echo "Done. Worker env ready: ${ENV_NAME}"
