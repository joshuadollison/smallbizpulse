#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_PATH="${1:-${ROOT_DIR}/outputs/model_builder_v2_colab_bundle.zip}"

mkdir -p "$(dirname "${OUT_PATH}")"

cd "${ROOT_DIR}"
zip -r "${OUT_PATH}" \
  model_builder \
  notebooks/model_builder_v2_colab.ipynb \
  -x '*/__pycache__/*' '*.pyc' '.ipynb_checkpoints/*'

echo "Created portable bundle: ${OUT_PATH}"
