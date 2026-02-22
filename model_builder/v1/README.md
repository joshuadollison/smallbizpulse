# model_builder/v1

Clean Python extraction of the model-building code from `notebooks/model_exploration.ipynb`.

## Scope
- DistilBERT sentiment model training + calibration + scoring.
- GRU closure-risk model training + window/business triage outputs.
- Programmatic orchestration API (`ModelBuilderV1`) to run both in sequence.

## Modules
- `sentiment_training.py`
  - `DistilBertSentimentTrainer`
  - `SentimentTrainingConfig`
  - `aggregate_business_month_sentiment(...)`
- `gru_training.py`
  - `GruTrainingConfig`
  - `train_gru_end_to_end(...)`
  - split/window/triage helpers
- `pipeline.py`
  - `ModelBuilderV1`

## Usage
```python
import pandas as pd
from model_builder.v1 import ModelBuilderV1

# reviews must include at least:
# business_id, status, date, review_id, stars, text
reviews = pd.read_csv("data/interim/rest_reviews.csv")

builder = ModelBuilderV1()

# 1) Sentiment training
sentiment = builder.train_sentiment_model(
    reviews,
    output_dir="models/artifacts/transformer_sentiment_distilbert",
)

# 2) Score all reviews + aggregate monthly panel
scored = builder.score_reviews_with_sentiment(
    reviews,
    model_dir=sentiment.model_dir,
    temperature=sentiment.best_temperature,
)
monthly_panel = builder.build_monthly_panel_from_scored_reviews(scored)

# 3) GRU training + triage artifacts
gru = builder.train_gru_model(
    monthly_panel,
    output_dir="models/artifacts",
)

print(gru.validation_metrics)
```

## Data contracts
### Sentiment trainer input
- Required columns: `stars`, `text`, `date`.
- Label mapping defaults:
  - negative stars: `1.0, 2.0`
  - positive stars: `4.0, 5.0`

### Monthly panel for GRU trainer
- Required columns:
  - `business_id`, `status`, `month`
  - `review_count`, `avg_stars`
  - `tx_sent_mean`, `tx_sent_std`, `tx_neg_share`, `tx_pos_share`

## Artifacts written by GRU pipeline
- `models/model_gru.keras`
- `models/model_metadata.json`
- `gru_business_triage.csv`
- `gru_business_triage_top5pct.csv`
- `gru_business_triage_top10pct.csv`
