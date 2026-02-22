# CIS509 - Yelp Closure-Risk Dashboard Handoff (IDE Agent Doc)

## 1. What we are building (project intent)
We are building an analytics workflow (and dashboard UI on top of it) that, given a restaurant (company) name, will:
1) Predict **closure risk** (near-term) using historical review-derived monthly trajectories.  
2) Identify **what is going wrong** (problems/themes) from text evidence.  
3) Generate **actionable recommendations** mapped to the detected problems.  

The dashboard should behave like a product.  Input a restaurant name.  Output risk, drivers, and next actions - without relying on “CSV glue” as the primary interface.  CSV artifacts are fine for audit/export, but the workflow should be callable as functions/services.

## 2. What we produced in the notebook (current state)
### 2.1 Data standardization and monthly panel
We built/used a monthly business panel (`biz_month_tx`) with required columns:
- `business_id`, `status`, `month`
- `review_count`, `avg_stars`
- `tx_sent_mean`, `tx_sent_std`, `tx_neg_share`, `tx_pos_share`

This is the canonical modeling table: one row per (business_id, month), with numeric signals plus text-derived sentiment aggregates.

### 2.2 Zombie masking (Open but inactive)
We implemented “zombie Open” exclusion to avoid contaminating labels with businesses that are Open in metadata but effectively inactive.

- `global_last_month = max(month)`  
- `zombie_cutoff = global_last_month - INACTIVE_K months` (INACTIVE_K = 12)  
- Zombie rule: `status == Open` AND `last_review_month <= zombie_cutoff`  
- Zombie Open businesses are excluded entirely (label is unknown, not 0).  

Observed output (latest run):
- Global last month: 2018-10-01  
- Zombie cutoff: 2017-10-01  
- Open zombies excluded: 854  

### 2.3 Right-censor masking (cannot observe full horizon)
We added right-censoring to prevent false negatives near the end of the dataset.

For each window ending at `window_end`, define:
- `horizon_end = window_end + H months` (H = 6)

Rule:
- If `status == Open` AND `horizon_end > global_last_month`, exclude window (unknown label).  

This reduced windows and improved label correctness by removing “Open windows” that cannot be verified as non-closing within the prediction horizon.

### 2.4 Feature engineering (levels + trajectories)
We engineered per-business normalized and trajectory features from base signals.

Base features:
- `review_count`, `avg_stars`, `tx_sent_mean`, `tx_sent_std`, `tx_neg_share`, `tx_pos_share`

Per business:
- z-scores: `*_z`
- lag-1 diffs: `*_d1`
- rolling mean (3): `*_rm3`
- rolling std (3): `*_rs3`
- rolling std (6): `*_rs6`
- time index: `months_since_first`

Total feature count: 31 (latest run).  

These features are explicitly designed to capture trend shifts and instability, not just absolute levels.

### 2.5 Windowed dataset construction (sequence model)
We build fixed-length windows:
- SEQ_LEN = 12 months per window  
- X shape (latest run): (18942, 12, 31)  
- y balance (latest run): 0 = 18222, 1 = 720  

Window filtering (to avoid weak/noisy sequences):
- MIN_ACTIVE_MONTHS = 6 (months with review_count > 0 in window)  
- MIN_REVIEWS_IN_WINDOW = 10 (total reviews in window)  

Labeling:
- Closed business has a proxy `closure_month = last_review_month`  
- Window is positive if closure_month in (window_end, window_end + H]  
- Windows at/after closure_month excluded for Closed businesses  

### 2.6 No-leakage split (business-stratified)
We split train/val by business_id (not by windows), stratified by whether a business has any positive window.

Latest run:
- Train windows: 15116, Val windows: 3826  
- Train pos rate: 0.03817  
- Val pos rate: 0.03738  
- Val businesses: 228, Val positive businesses: 61 (0.2675)  

### 2.7 Balanced batches for training
Training uses a balanced sampling tf.data pipeline:
- Build ds_pos and ds_neg, repeat and sample with weights.  
- POS_BATCH_RATE target = 0.30  
- BATCH_SIZE = 256  

This avoids the model collapsing to “always negative” given heavy class imbalance.

### 2.8 GRU risk model (closure-risk predictor)
We trained a GRU sequence model with:
- 2x GRU stack (128 then 64) with LayerNorm and Dropout
- Attention pooling across time (Dense -> Softmax over axis=1 -> weighted sum)
- Dense head (64 relu)
- Sigmoid output with **bias initializer** set to training base-rate logit for calibration

Optimizer:
- AdamW if available, otherwise Adam  
- LR = 1e-3 with ReduceLROnPlateau  
- Weight decay = 1e-4  
- Label smoothing = 0.01  
- Clipnorm = 1.0  

Metrics tracked:
- PR-AUC (primary), ROC-AUC, precision, recall

This model outputs `p_closed` per window.

### 2.9 Triage metrics (window-level and business-level)
We computed “top-K” precision/recall to represent operational triage (workload vs yield).

Window-level: rank by `p_closed` (highest risk windows).  
Business-level: aggregate windows into a business risk score.

Business risk score:
- `p_recent_max` = max probability over the most recent 3 windows (by end_month)  
- `risk_score = p_recent_max` fallback to p_max if missing  
- risk buckets with bins: [0.0, 0.50, 0.65, 0.75, 0.85, 1.0] -> low/medium/elevated/high/very_high  

We also introduced a “recent cutoff” concept for business triage (example output printed cutoff 2018-07-01).  This is used to emphasize recent behavior.

### 2.10 Problems and recommendations (text-to-themes + playbook)
We produced problem keywords per business and mapped them to themes and recommendations, resulting in a final table with:
- `business_id`, `status`
- `risk_score`, `risk_bucket`
- `problem_keywords`
- `recommendations_top3`
- `themes_top3`

Final artifact produced:
- `../artifacts/final_closure_risk_problems_recommendations.csv`  
- Rows: 228, Businesses: 228  

Example (confirmed working):
- risk_score present
- problem keywords present for many
- recommendations_top3 populated
- themes_top3 populated

## 3. Models we built (what exists and what to ship)
### 3.1 GRU closure-risk model (Keras / TensorFlow)
- This is the main predictive model: sequence -> closure risk probability.
- It should be saved and loaded as a Keras model (`.keras` or SavedModel directory).  

What the dashboard needs from it:
- A `predict_proba(windows)` producing probability per window.
- A business risk score aggregator (recent max, etc.).  

### 3.2 Transformer sentiment model (DistilBERT)
We have a local HF model folder `transformer_sentiment_distilbert`.  
Observed config:
- `id2label: {0: 'LABEL_0', 1: 'LABEL_1'}`
- `label2id: {'LABEL_0': 0, 'LABEL_1': 1}`
- `num_labels: 2`

Important: which label corresponds to POS vs NEG must be checked via test sentences.  We ran a sanity test and found:
- POS sentences: p(LABEL_1) ~ 1.000
- NEG sentences: p(LABEL_0) ~ 0.998

Therefore in this trained model:
- `LABEL_1 == POS`  
- `LABEL_0 == NEG`  

The dashboard uses this model for text scoring (review-level), then aggregates into monthly sentiment features:
- tx_sent_mean, tx_sent_std, tx_neg_share, tx_pos_share

## 4. What challenges we solved (and why they mattered)
1) **Label noise from zombies** - Open-but-inactive businesses were polluting negatives.  We excluded them as unknown.  
2) **Right-censoring** - Near the dataset end, Open windows cannot be verified for H-month outcomes.  We excluded those windows to avoid false negatives.  
3) **Leakage** - Window-level split leaks business identity.  We split by business_id, stratified by business outcome.  
4) **Class imbalance** - Balanced batch sampling prevented trivial “all negative” behavior.  
5) **Calibration** - Output-layer bias initialized to base-rate logit improved probability calibration stability.  
6) **Operational evaluation** - Top-K precision/recall at both window and business levels matches how this will be used in the real world (triage).

## 5. Current findings (what we can say right now)
- The GRU model is learning non-trivial signal (PR-AUC above base) but is still noisy - expected given weak labels and heavy imbalance.  
- Top-K triage is usable as a prioritization mechanism.  At small K, precision can be meaningfully above base rate.  
- The pipeline produces a coherent final output: risk + themes + recommendations for 228 validation businesses.  
- The problems/recommendations layer works end-to-end (recommendations_top3 and themes_top3 are populated in final output).

## 6. What is left (to make this dashboard-grade)
### 6.1 Replace CSV-first flow with model-first service
Right now notebook emits CSVs for inspection.  The dashboard should call functions directly and optionally export CSV for audit.

### 6.2 Name-based lookup and entity resolution
Dashboard input is “company name”.  Need deterministic lookup:
- normalize name (casefold, punctuation)  
- handle duplicates (multiple businesses with similar name)  
- return candidates with city/state, category, and review_count to disambiguate

### 6.3 Real-time feature build for a selected business
Given a business_id, build:
1) Pull all reviews -> run transformer sentiment -> per-review score  
2) Aggregate into monthly metrics -> produce `biz_month_tx` rows for that business  
3) Apply same feature engineering (`*_z`, `*_d1`, rolling stats, months_since_first)  
4) Build last SEQ_LEN windows that satisfy filters  
5) Run GRU -> window probabilities -> business risk score  

### 6.4 Evidence and explainability
For the “problems” section:
- show top negative reviews and representative phrases  
- show trend charts (stars, review_count, sentiment mean, neg share)  
- show which recent windows triggered risk (end_month, p_closed)

### 6.5 Recommendation mapping refinement
We currently map keywords -> themes -> playbook recommendations.  Improve by:
- weighting keywords by recency and negative sentiment context  
- adding a “needs_review” fallback when signal is weak  
- ensuring recommendations are grounded with citations (example review snippets)

## 7. How to use this locally (what the IDE agent should implement)
### 7.1 Expected local folder layout
You should have:
- `models/` (Keras SavedModel or .keras for GRU risk model)
- `transformer_sentiment_distilbert/` (HF model folder)
- data sources required to pull reviews for a business (Yelp dataset tables/files)

Artifacts currently live at:
- `../artifacts/` (relative to notebook working directory)

### 7.2 Minimal callable API (recommended)
Implement these functions in the dashboard backend:

```python
def find_business_candidates(name: str, *, city: str | None = None, state: str | None = None, limit: int = 10) -> list[dict]:
    """
    Returns candidate businesses with business_id and metadata for disambiguation.
    """

def build_monthly_panel_for_business(business_id: str) -> "pd.DataFrame":
    """
    Returns monthly panel with required columns:
    business_id, status, month, review_count, avg_stars,
    tx_sent_mean, tx_sent_std, tx_neg_share, tx_pos_share
    """

def build_features_and_windows(biz_month_tx_one: "pd.DataFrame", *, seq_len: int = 12, H: int = 6) -> tuple["np.ndarray", "pd.DataFrame"]:
    """
    Applies the exact same feature engineering as notebook and builds windows.
    Returns:
      X_windows: (n_windows, 12, 31)
      meta_windows: rows with start_month, end_month, etc.
    """

def predict_closure_risk(X_windows: "np.ndarray") -> "np.ndarray":
    """
    Loads GRU model and returns p_closed per window.
    """

def aggregate_business_risk(meta_windows: "pd.DataFrame", p_windows: "np.ndarray") -> dict:
    """
    Computes risk_score = max over most recent 3 windows (or configured rule),
    and assigns risk_bucket.
    """

def extract_problem_keywords(business_id: str) -> str:
    """
    Produces problem keywords from reviews (or cached).
    """

def map_recommendations(problem_keywords: str) -> tuple[str, str]:
    """
    Returns (recommendations_top3, themes_top3).
    """

def score_business(name: str, *, city: str | None = None, state: str | None = None) -> dict:
    """
    Full end-to-end:
    name -> candidate -> business_id -> reviews -> monthly -> windows -> risk -> problems -> recs
    Returns JSON payload for dashboard.
    """


## 7.3 Dashboard output contract (what UI should display)

Return **one JSON object** from your scoring endpoint.  This is the contract the UI renders.

### Business identity
- `business_id` (string)
- `name` (string)
- `city` (string | null)
- `state` (string | null)
- `status` ("Open" | "Closed" | "Unknown")
- `total_reviews` (int)
- `last_review_month` (YYYY-MM-01 string | null)

### Risk
- `risk_score` (float 0-1) - business-level score (ex: max over most recent 3 windows)
- `risk_bucket` ("low" | "medium" | "elevated" | "high" | "very_high")
- `recent_windows` (array) - most recent N windows used for scoring:
  - each item: `{ "end_month": "YYYY-MM-01", "p_closed": float }`

### Problems
- `themes_top3` (array of strings) - normalized themes like:
  - `["food_quality","service_speed","order_accuracy"]`
- `problem_keywords` (string | null) - comma-separated keywords/phrases (what we extracted)
- `evidence_reviews` (array) - 3-5 short snippets grounding the problems:
  - each item:
    - `review_id` (string | null)
    - `date` (YYYY-MM-DD | null)
    - `stars` (int | null)
    - `sentiment_neg_prob` (float | null) - optional
    - `snippet` (string) - short excerpt

### Recommendations
- `recommendations_top3` (array of strings) - actionable items mapped from themes
- `recommendation_notes` (string | null) - optional, for “needs_review” cases

---

## 7.4 End-to-end workflow (what the system does)

### Inputs the dashboard collects
- User enters: `company_name` (and optionally city/state filters)
- Optional filters:
  - time range (months)
  - minimum reviews
  - “restaurants only” toggle

### End-to-end pipeline (the thing we are actually building)
**name -> candidate -> business_id -> reviews -> monthly -> windows -> risk -> problems -> recs -> JSON**

1) **Resolve name to business_id**
- Take `company_name` + optional city/state.
- Search business table for candidates (name similarity).
- If multiple candidates:
  - return a short candidate list to the UI for user selection.
- Once selected: lock `business_id`.

2) **Fetch all reviews for business_id**
- Pull review text, stars, date/time, review_id.
- Clean text minimally (strip weird whitespace, normalize).

3) **Per-review sentiment inference (DistilBERT)**
- Run local transformer model (the fine-tuned DistilBERT folder).
- Produce per-review:
  - `p_neg`, `p_pos` (based on label mapping you validated: the label consistently higher on positive test sentences is POS).
- Store these scores in-memory for aggregation + evidence selection.

4) **Aggregate reviews to monthly features**
For each month:
- `review_count`
- `avg_stars`
- `tx_sent_mean` (mean sentiment score - your chosen convention, usually pos_prob or neg_prob)
- `tx_sent_std`
- `tx_neg_share` (fraction of reviews that are “negative” by threshold)
- `tx_pos_share`

5) **Build rolling windows for GRU**
- Build **SEQ_LEN=12** month windows.
- Apply the exact filters you used in notebook:
  - minimum active months
  - minimum reviews in window
- Apply censoring logic:
  - exclude zombie Open businesses (inactive for INACTIVE_K months relative to dataset max)
  - exclude right-censored Open windows (cannot observe full horizon after window end)

6) **Run GRU risk model**
- Load `models/model_gru.keras`.
- For each window, produce `p_closed`.
- Business-level `risk_score`:
  - max over most recent 3 windows (`p_recent_max`) or fallback to `p_max`.
- Bucket into `risk_bucket` via your bins.

7) **Problem extraction**
- Focus on the “why” layer for the dashboard:
  - select evidence reviews: top 3-5 by **negative sentiment** (and/or low stars) within recent time window.
  - extract `problem_keywords` from those reviews (your existing keyword pipeline).
  - map keywords -> normalized `themes_top3`.

8) **Recommendations**
- Use the rule-based playbook mapping:
  - `themes_top3` -> `recommendations_top3`.
- If insufficient signal:
  - set `themes_top3=["needs_review"]`
  - set `recommendation_notes="Not enough signal in keywords - review recent negatives manually."`

9) **Return JSON**
Return the object from section 7.3 exactly.  No CSVs required for dashboard runtime.

---

## 7.5 What the UI should look like (minimum viable dashboard)

### Screen 1 - Search / Select
- Search box: company name
- Results list (if ambiguous): name + city/state + total reviews + last review month
- “Select” action -> go to business view

### Screen 2 - Business view
**Top row:**
- Name, location, status, last review month, total reviews
- Risk badge: bucket + risk_score

**Risk panel:**
- sparkline / small chart of `recent_windows.p_closed` over time

**Problems panel:**
- Themes chips (top 3)
- Keyword string (scrollable)
- Evidence reviews list (3-5):
  - star rating, date, snippet, neg_prob

**Recommendations panel:**
- Top 3 actionable items (bullet list)
- Optional “needs_review” note

---

## 7.6 Runtime requirements (local dashboard use)

### Models you already have locally
1) **Closure risk model (GRU)**
- File: `models/model_gru.keras`
- Purpose: window-level probability of closure within horizon -> business risk score

2) **Keyword/topic components**
- Files: `models/vec.joblib`, `models/nmf.joblib` (and metadata/manifest)
- Purpose: convert review text into keyword/topic signal (problem_keywords + themes)

3) **Sentiment transformer (fine-tuned DistilBERT)**
- Folder: `transformer_sentiment_distilbert/`
  - includes `config.json`, `tokenizer.json`, `model.safetensors`, etc.
- Purpose: per-review sentiment probabilities used for:
  - monthly sentiment aggregates (tx_sent_mean, tx_neg_share, etc.)
  - picking evidence reviews
  - improving problem extraction focus

### Key conventions you must keep consistent
- **Label mapping for transformer**:
  - You validated: test POS sentences produce higher probability on one label consistently.
  - Whatever that label is becomes POS, the other becomes NEG.  Store this mapping once in code.
- **Window settings**:
  - SEQ_LEN=12, H=6, zombie cutoff, right-censor exclusion
- **Feature definitions**:
  - monthly features must match what the GRU was trained on

---

## 7.7 Known challenges we already solved (and why they mattered)

1) **Zombie Open businesses**
- Problem: Open businesses with long inactivity aren’t truly “safe” negatives.
- Fix: mark as Unknown and exclude from training/scoring windows.

2) **Right-censoring**
- Problem: If you can’t observe the full horizon after a window end, labels are unreliable.
- Fix: exclude Open windows where `window_end + H > global_last_month`.

3) **Leakage control**
- Problem: splitting windows randomly leaks business history across train/val.
- Fix: split at the business level (stratified by whether the business has any positive windows).

4) **Imbalance**
- Problem: positives are ~3-4% of windows.
- Fix: balanced-batch tf.data sampling + bias init to base rate + PR-AUC monitoring.

---

## 7.8 Current findings (what the model is doing right now)

- Dataset after censoring:
  - Windows: ~18,942
  - Businesses: ~1,135
  - Positive windows: 720 (~3.7-3.8%)
- Risk triage works as a prioritization tool:
  - Top slices increase precision vs baseline (baseline ~3.7%).
  - Not perfect - but it’s producing a usable ranked worklist and business-level buckets.
- Problems + recommendations are now populating in final output for many businesses:
  - Themes like `food_quality`, `service_speed`, `order_accuracy`, `pricing_value`, `cleanliness`, `staff_attitude`, `menu_ops`
  - Recommendations align to those themes

---

## 7.9 What’s left (dashboard engineering tasks, not modeling)

1) **Business lookup layer**
- Implement fuzzy match + disambiguation list in the UI.

2) **Review retrieval layer**
- Efficiently load all reviews for a selected business_id.
- Cache sentiment outputs so repeat loads are fast.

3) **Realtime feature builder**
- Convert reviews -> monthly table -> windows on demand.

4) **Inference service**
- Wrap models in a single `score_business(company_name, city=None, state=None)` function that returns the JSON contract.

5) **UI**
- Build the two screens (search/select, business view) and wire to the scoring endpoint.

---

## 7.10 Implementation note (what to hand to an IDE agent)

Build one callable that the dashboard uses:

### `score_business(company_name, city=None, state=None) -> dict`
- returns the JSON described in 7.3
- internally executes the full pipeline from 7.4 using the local models:
  - GRU risk model (`model_gru.keras`)
  - sentiment transformer folder (`transformer_sentiment_distilbert/`)
  - NMF/vectorizer (`nmf.joblib`, `vec.joblib`) if used for problems/themes
- never depends on precomputed CSVs for runtime (CSV outputs were notebook artifacts only)