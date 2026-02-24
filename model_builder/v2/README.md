# model_builder/v2

Methodology-aligned model-building package based on `docs/methodology.md`.

## What v2 builds

### Component 1: Multi-signal survival prediction
- Baselines:
  - Logistic Regression
  - SVM (RBF)
  - Gradient Boosting
- Sequence model:
  - GRU (attention pooling) trained on 12-month windows for businesses with `>10` reviews
- Rule model:
  - Explicit low-data fallback for businesses with `<=5` reviews

Signals used:
- Check-ins (volume + velocity)
- Review sentiment trend (VADER compound)
- Supporting stars/review volume/tip activity

`v2` sequence training also applies key hardening from `v1`:
- zombie Open masking (`inactive_months_for_zombie`)
- right-censor masking for Open windows
- trajectory features (`z`, `d1`, `rm3`, `rs3`, `rs6`, `months_since_first`)
- business-stratified validation split (no identity leakage)
- balanced positive/negative tf.data sampling
- GRU window + business top-K triage metrics for operational workload planning

### Component 2: Diagnostic topic modeling
- BERTopic on negative reviews only:
  - `stars <= 2` **and** `vader_compound <= -0.05`
- Terminal analysis on final 10 negative reviews per business
- Recovery comparison:
  - open-after-negative vs closed-after-negative cohorts

### Component 3: Intervention recommendation mapping
- Topic term bundles mapped to intervention themes and recommended actions.

### Component 4: Resilience/vulnerability analysis
- City-level closure rates
- Cuisine-level closure rates
- Check-in floor analysis (where closure rates jump)
- Recovery pattern analysis after negative phases

## Package structure
- `io.py`: Yelp loading and restaurant filtering
- `sentiment.py`: VADER scoring wrappers
- `features.py`: monthly signal panel + snapshot + sequence windows
- `survival.py`: Component 1 model training and persistence
- `inference.py`: live model scoring runtime (no precomputed risk-table dependency)
- `topic_modeling.py`: Component 2 BERTopic training and outputs
- `recommendation_engine.py`: Component 3 topic->intervention mapping
- `resilience.py`: Component 4 analysis outputs
- `pipeline.py`: orchestration (`ModelBuilderV2`)
- `requirements-core.txt`: minimal Component 1 dependencies
- `requirements-colab.txt`: Colab-friendly full dependency set

## Quick start

```python
from pathlib import Path
from model_builder.v2 import ModelBuilderV2, ModelBuilderV2Config

builder = ModelBuilderV2(
    ModelBuilderV2Config(output_root=Path("models/v2_artifacts"))
)

artifacts = builder.run(
    data_root=Path("data/external/yelp_dataset_new")
)

print("summary:", artifacts.run_summary_path)
```

## Live inference (model-driven, no score artifacts)

```python
from model_builder.v2 import SurvivalRuntime

runtime = SurvivalRuntime.from_output_dir("models/v2_artifacts/component1_survival")
scores = runtime.score_data_root("data/external/yelp_dataset_new")
print(scores.head())
```

`SurvivalRuntime` only loads trained model files (`baseline_models.joblib`, `rule_model.json`, optional `gru_survival_model.keras`) and computes risk live from source data.

## Colab portability

To run in Colab on a larger GPU:
1. Copy the `model_builder/` folder to Drive.
2. Upload/open `notebooks/model_builder_v2_colab.ipynb` in Colab.
3. Set Drive paths in the notebook, run all cells, and train.
4. Copy `models/v2_artifacts` back to local for runtime use.

## Dependency notes
- Required:
  - `pandas`, `numpy`, `scikit-learn`
  - `joblib`
- Required for sequence model training:
  - `tensorflow`
- Required for BERTopic diagnostics:
  - `bertopic`
  - sentence-transformer stack compatible with your environment
- VADER sentiment:
  - `vaderSentiment` preferred
  - fallback to `nltk` VADER if available

If BERTopic is missing, Component 2 raises a clear dependency error.

## Outputs
All artifacts are written under `ModelBuilderV2Config.output_root`, by default:
- `component1_survival/`
  - `baseline_models.joblib`
  - `baseline_metrics.json`
  - `rule_model.json`
  - `business_snapshot_features.csv`
  - `sequence_window_meta.csv`
  - `gru_survival_model.keras` (if GRU training succeeds)
  - `gru_metadata.json` (if GRU training succeeds)
  - `gru_metrics.json` (if GRU training succeeds)
  - `gru_window_topk.csv` (if GRU training succeeds)
  - `gru_business_topk.csv` (if GRU training succeeds)
  - `gru_business_triage.csv` (if GRU training succeeds)
- `component2_topics/`
- `component3_recommendations/`
- `component4_resilience/`
- `monthly_signal_panel.csv`
- `reviews_with_vader.csv`
- `run_summary.json`
