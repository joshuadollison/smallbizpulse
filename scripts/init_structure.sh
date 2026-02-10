#!/usr/bin/env bash
set -euo pipefail

# Run this from the repo root:
#   bash scripts/init_structure.sh
# or:
#   chmod +x scripts/init_structure.sh && ./scripts/init_structure.sh

# Safety check: must be run from a git repo root (or at least a repo with .git/)
if [[ ! -d ".git" ]]; then
  echo "ERROR: .git directory not found.  Run this from the repo root." >&2
  exit 1
fi

# Create directories (idempotent)
mkdir -p \
  docs/proposal \
  docs/rubric \
  docs/presentation/figures \
  docs/reports \
  data/raw \
  data/interim \
  data/processed \
  data/external \
  notebooks/00_admin \
  notebooks/01_ingest \
  notebooks/02_eda \
  notebooks/03_modeling \
  notebooks/04_dashboard_prep \
  src/smallbizpulse/config \
  src/smallbizpulse/ingest \
  src/smallbizpulse/preprocess \
  src/smallbizpulse/features \
  src/smallbizpulse/models \
  src/smallbizpulse/viz \
  src/smallbizpulse/utils \
  src/scripts \
  src/tests \
  dashboard/app \
  dashboard/api \
  dashboard/assets \
  dashboard/data \
  dashboard/tests \
  outputs/figures \
  outputs/tables \
  outputs/models \
  outputs/logs \
  env

# Helper: create a file only if missing
touch_if_missing() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    touch "$path"
  fi
}

# Helper: write file only if missing
write_if_missing() {
  local path="$1"
  shift
  if [[ ! -f "$path" ]]; then
    cat > "$path" <<'EOF'
'"$@"'
EOF
  fi
}

# Package init
touch_if_missing "src/smallbizpulse/__init__.py"

# Config placeholder (only if missing)
if [[ ! -f "src/smallbizpulse/config/settings.yaml" ]]; then
  cat > "src/smallbizpulse/config/settings.yaml" <<'EOF'
project:
  name: smallbizpulse
paths:
  data_raw: data/raw
  data_interim: data/interim
  data_processed: data/processed
  outputs: outputs
EOF
fi

# data/README.md (only if missing)
if [[ ! -f "data/README.md" ]]; then
  cat > "data/README.md" <<'EOF'
# Data

Put large datasets in data/raw locally.  Do not commit large raw data to GitHub.

Suggested pattern:
- data/raw - original source files (local only)
- data/interim - intermediate transforms (local only)
- data/processed - clean, model-ready datasets (local only)
- data/external - external reference data (local only)

If you need to commit data, commit only tiny samples and document provenance.
EOF
fi

# dashboard/README.md (only if missing)
if [[ ! -f "dashboard/README.md" ]]; then
  cat > "dashboard/README.md" <<'EOF'
# Dashboard

Put your web dashboard here.

Recommended subfolders:
- app/ - UI code
- api/ - backend services (if any)
- assets/ - static assets
- data/ - dashboard-ready exports (small only, or generated locally)
- tests/ - dashboard tests
EOF
fi

# env placeholders (only if missing)
touch_if_missing "env/requirements.txt"

if [[ ! -f "env/environment.yml" ]]; then
  cat > "env/environment.yml" <<'EOF'
name: smallbizpulse
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - -r requirements.txt
EOF
fi

# .gitkeep for empty dirs so Git tracks them (only if missing)
touch_if_missing "docs/presentation/figures/.gitkeep"

touch_if_missing "data/raw/.gitkeep"
touch_if_missing "data/interim/.gitkeep"
touch_if_missing "data/processed/.gitkeep"
touch_if_missing "data/external/.gitkeep"

touch_if_missing "outputs/figures/.gitkeep"
touch_if_missing "outputs/tables/.gitkeep"
touch_if_missing "outputs/models/.gitkeep"
touch_if_missing "outputs/logs/.gitkeep"

touch_if_missing "dashboard/app/.gitkeep"
touch_if_missing "dashboard/api/.gitkeep"
touch_if_missing "dashboard/assets/.gitkeep"
touch_if_missing "dashboard/data/.gitkeep"
touch_if_missing "dashboard/tests/.gitkeep"

echo "Done.  Created missing project folders without overwriting existing files."
echo "Next:"
echo "  git status"
echo "  git add ."
echo "  git commit -m \"Add project folder structure\""