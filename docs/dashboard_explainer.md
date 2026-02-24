# Dashboard Explainer (Current Implementation)

This reflects the current Streamlit dashboard and service behavior in:
- `dashboard/streamlit_app.py`
- `dashboard/app/service.py`
- `dashboard/app/settings.py`

---

## 1) What happens when a user searches?

The dashboard has two separate pages and two separate search datasets.

### Live scoring page
- Search source: live Yelp business table (`yelp_academic_dataset_business.json`), restaurant rows only.
- Matching/ranking: exact/prefix/contains/token/fuzzy matching + city/state bonuses + review-count tie-break.
- Inputs:
  - `Company name` (required)
  - `City`, `State` (optional)
  - `Include unscorable matches`
  - `Min active months in 12-month window` slider
  - `Min reviews in 12-month window` slider
- Result table: selectable flat table, then `Score selected`.

### Artifact explorer page
- Search source: scored artifact table (`_scored_df`) loaded from v2 bundle outputs.
- Inputs:
  - `Company name`, `City`, `State`
  - `Risk band` (`Any`, `low`, `medium`, `elevated`, `high`, `very_high`)
  - `Max rows`
- Result table: selectable flat table, then `View selected`.

---

## 2) Live vs artifact scoring behavior

### Live page
The UI directly calls `score_business_live(...)` for the selected `business_id`.
- No artifact fallback is used in this page flow.
- Output is based on live Yelp files + v2 runtime + GRU.

### Artifact page
The UI directly calls `score_business_artifact(...)`.
- Output is loaded/computed from artifact-backed v2 data products.

### Important note
`service.score_business(...)` still exists for API compatibility and can do mixed/fallback logic, but Streamlit page flows now call the explicit live/artifact methods above.

---

## 3) Why artifact values can still differ from live values

Live and artifact can diverge because they do not always use identical input coverage.

Common causes:
- Live Yelp review file has fewer/more rows than the artifact build used.
- Window gating removes different windows after applying slider thresholds.
- Topic/recommendation optional tables may exist for artifact but not for a specific live-derived path.

Signal for mismatch:
- `data_quality.has_mismatch = true` and `coverage_ratio < 1` means observed live reviews are lower than expected business review count metadata.

---

## 4) Risk score and buckets

Risk is a probability-like value in `[0, 1]`.

Default bucket thresholds:
- `low`: `[0.00, 0.50]`
- `medium`: `(0.50, 0.65]`
- `elevated`: `(0.65, 0.75]`
- `high`: `(0.75, 0.85]`
- `very_high`: `(0.85, 1.00]`

For GRU window scoring, final business risk is:
- `max(p_closed)` over recent windows (`k` from runtime config, usually 3).

---

## 5) Recent windows: what they are and how they are built

`Recent windows` are sequence-level predictions:
- each row has `end_month`, `p_closed`
- these are the latest scored 12-month windows for the business

Window construction uses v2 feature logic:
- sequence length = 12 months
- sliding windows over monthly panel
- gating by:
  - `min_active_months` (slider, default 6)
  - `min_reviews_in_window` (slider, default 10)
- right-censor and sequence validity checks from v2 feature builder

If no window survives, you get `not_scored_reason` such as:
- `insufficient_history_for_windows`
- `insufficient_history_for_windows_live_data_mismatch`

---

## 6) What `_can_build_live_windows_for_business` does

`_can_build_live_windows_for_business(...)` is a pre-check used by search filtering.

What it does:
- Builds a quick monthly panel for the candidate.
- Runs v2 sequence-window builder with the selected slider thresholds.
- Returns `True` if at least one scorable window exists, else `False`.
- Caches result by `(business_id, min_active_months, min_reviews_in_window)`.

Why it exists:
- When `Include unscorable matches` is off, it prevents showing businesses that cannot produce a live score under current thresholds.

---

## 7) Themes, keywords, and recommendations

### Themes
- Source priority:
  - precomputed v2 topic summary fields, then
  - terminal topic diagnostics, then
  - fallback `needs_review`.

`needs_review` means no reliable thematic mapping was found. It is a workflow fallback marker, not a predicted class.

### Problem keywords
- Built from topic terms / mapped terms and de-duplicated.
- Stop-word filtering is applied to avoid junk tokens.

### Recommendations
- Source priority:
  - precomputed `recommendations_top3`, then
  - topic->recommendation mapping from v2 tables, then
  - fallback note:
    - "Not enough signal in keywords - review recent negative feedback and tag issues into service, food, cleanliness, value, or accuracy."

---

## 8) Trend charts

The trend section is driven by `chart_data`:
- `ratings_by_month` (avg stars)
- `rating_bucket_counts_by_month` (1-5 star lines)
- `predicted_close_by_month` (window-level risk over time)
- `topics_per_class` (class view, top topics)
- `topics_by_month` (topic timeline)

Charts can be empty when the relevant series is unavailable for that business/path.

---

## 9) Component diagnostics (v2)

### Component 2/3 diagnostics
- Negative-review count used by topic diagnostics
- Terminal topics table (theme/share/count/recommendation)
- Recovery divergence by topic

### Component 4 resilience context
- City closure context
- Matched cuisine closure rates
- Check-in floor context
- Recovery pattern

These sections depend on optional v2 output tables and business-level row coverage in those tables.

---

## 10) Dependency indicators in sidebar

The dashboard now reports:
- TensorFlow package
- TensorFlow runtime
- `vaderSentiment` package
- NLTK package
- V2 runtime readiness + GRU model presence
- Yelp source file presence
- Component 2/3/4 table presence

Interpretation:
- `TensorFlow package = ok` but `TensorFlow runtime = missing` means import/runtime probe failed (often binary/ABI mismatch).

---

## 11) Known failure reasons and what they mean

- `yelp_files_missing`: configured Yelp path does not contain required JSONs.
- `missing_yelp_data_dir`: no Yelp dir configured.
- `vader_unavailable`: neither `vaderSentiment` nor NLTK VADER path was usable.
- `tensorflow_runtime_unavailable`: TensorFlow import/runtime probe failed.
- `insufficient_history_for_windows`: not enough valid sequence windows under current slider thresholds.
- `insufficient_history_for_windows_live_data_mismatch`: window shortage plus observed-vs-expected review mismatch.

---

## 12) Operational guardrails

- Launch with `scripts/run_dashboard.sh` to get sane defaults for:
  - `SBP_YELP_DATA_DIR`
  - `SBP_V2_BUNDLE_DIR`
  - `SBP_PREFER_LIVE_SCORING`
- Keep `numpy<2` in dashboard runtime env to avoid TensorFlow 2.16 ABI failures.
- For stable live sentiment, install:
  - `vaderSentiment`
  - `nltk` + `vader_lexicon`

