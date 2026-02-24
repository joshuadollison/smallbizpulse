# Live Inference Flow (No Runtime CSV Writes)

This document captures the runtime path we use from the notebook pipeline, excluding the VADER comparison branch.

## Scope
- Runtime scoring path: memory-only intermediate representations.
- Offline artifact refresh path: optional CSV generation for audit/regression.
- VADER: excluded from production/runtime flow.

## Runtime Path (Memory-Only)
1. Resolve `business_id` from search.
2. Load all reviews for the business from Yelp review JSON.
3. Run DistilBERT sentiment on each review (`p_pos`, `p_neg`).
4. Aggregate review-level sentiment into monthly panel features:
   - `review_count`, `avg_stars`, `tx_sent_mean`, `tx_sent_std`, `tx_neg_share`, `tx_pos_share`
5. Build sequence features/windows (same family as notebook):
   - z-scores, deltas, rolling stats, `months_since_first`
   - apply notebook parity gates: minimum activity/reviews and right-censoring for open businesses
6. Run GRU on valid windows to produce window `p_closed`.
7. Aggregate to business risk (`risk_score`, `risk_bucket`, recent windows).
8. Build problems/recommendations:
   - evidence reviews -> keywords/topics -> themes -> recommendations.

Search note:
- `/api/search` now returns only candidates likely to produce a score by default (artifact-available
  or live-window-scorable under current gates).
- Use `include_unscorable=1` to disable this filter for debugging/exploration.

Runtime note:
- No intermediate CSVs are written in live scoring.
- The service stores the latest live workflow representation in memory on the service instance.
- GRU inference runs in an isolated subprocess worker; native TensorFlow crashes are converted to
  machine-readable `not_scored_reason` values (for example `gru_worker_signal_5`) instead of
  terminating the dashboard process.

## Notebook Lineage Mapping
- Yelp load/filter to restaurants + `rest_review_df`.
- DistilBERT fine-tune/save + full-review scoring.
- `biz_month_tx` creation.
- GRU feature engineering, censoring/windowing, inference, triage aggregation.

These are implemented in `dashboard/app/service.py` for runtime inference.

## Offline Artifact Refresh (Optional)
If you want to regenerate artifact CSVs for audit snapshots, use:

```bash
python scripts/postprocess_model_artifacts.py --artifact-root models/artifacts
```

This script is not part of live inference execution.

## Inputs and Outputs
- Runtime required inputs:
  - Yelp business JSON
  - Yelp review JSON
  - DistilBERT folder (`models/artifacts/transformer_sentiment_distilbert`)
  - GRU + topic artifacts (`models/artifacts/models`)
- Runtime outputs:
  - API response payload only (no persisted intermediate files)
- Optional offline outputs:
  - `models/artifacts/A_closure_risk_table.csv`
  - `models/artifacts/final_closure_risk_problems_recommendations.csv`

## Recommended Run Command
Use the launcher to avoid TensorFlow interpreter mismatches:

```bash
bash scripts/run_dashboard.sh
```

It runs Streamlit in `sbp311` by default on `127.0.0.1:8501` with live fallback enabled and Yelp path preconfigured.
Set `SBP_APP_SERVER=flask` if you need the Flask API UI/server path.
By default, scoring preference is live-first (`SBP_PREFER_LIVE_SCORING=true`) with artifact fallback if live
scoring is unavailable for a request.
By default, GRU subprocess inference is routed to `sbp-gru-legacy311` via
`SBP_GRU_WORKER_CONDA_ENV`.
