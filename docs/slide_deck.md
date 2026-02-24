# SmallBizPulse Dashboard Slide Deck (Current)

## Slide 1 - Title
- **SmallBizPulse Dashboard (v2 runtime)**
- Restaurant closure-risk scoring + diagnostics
- Workflow: **Search -> Select -> Score -> Diagnose -> Act**

## Slide 2 - Architecture Snapshot
- **UI:** Streamlit (`dashboard/streamlit_app.py`)
- **Service layer:** `SmallBizPulseService` (`dashboard/app/service.py`)
- **Model/data backbone:** v2 bundle + live Yelp files
- **Two pages:** `Live scoring` and `Artifact explorer`

---

## Section A - Runtime + Health (2 slides)

## Slide 3 - Header and Workflow Context
- Runtime chip explains current mode intent per page.
- Live page intent: live source -> live score.
- Artifact page intent: browse scored artifact rows.

## Slide 4 - Sidebar Dependency Indicators
- Show this first in demo.
- Covers:
  - TensorFlow package/runtime
  - `vaderSentiment`, NLTK
  - v2 runtime + GRU model file
  - Yelp source file presence
  - Component 2/3/4 table presence
- If runtime is bad, explain before scoring attempts.

---

## Section B - Live Scoring Page (2 slides)

## Slide 5 - Live Search Inputs
- Widgets:
  - Company name, city, state
  - Include unscorable matches
  - `Min active months in 12-month window` slider
  - `Min reviews in 12-month window` slider
- Purpose:
  - Control pre-gating strictness for live window viability.

## Slide 6 - Live Candidate Table + Action
- Flat selectable table (single row).
- Action button: **Score selected**.
- Tech:
  - Search from live Yelp business dataset (restaurants only).
  - Ranking by exact/prefix/contains/token/fuzzy + location bonuses.
  - Score call is `score_business_live(...)` (explicit live path).

---

## Section C - Artifact Explorer Page (2 slides)

## Slide 7 - Artifact Search Inputs
- Widgets:
  - Name/city/state
  - Risk band (`Any`, low, medium, elevated, high, very_high)
  - Max rows
- Purpose:
  - Explore pre-scored rows and filter by bucket.

## Slide 8 - Artifact Rows Table + Action
- Flat selectable table (single row).
- Action button: **View selected**.
- Tech:
  - Search from `_scored_df` (v2 artifact-backed score table).
  - For narrowed filters, missing bucket rows can be runtime-filled before risk-band filtering.

---

## Section D - Business Scoring Output (2 slides)

## Slide 9 - Risk Summary Block
- Business identity + risk score + bucket + mode.
- Explain `scoring_mode` first (`live`, `v2_runtime:*`, etc.).
- This is the top triage signal.

## Slide 10 - Summary Widgets
- Recent windows (`month`, `p_closed`)
- Themes
- Problem keywords
- Top recommendations (+ notes)
- Explain fallback behavior:
  - `needs_review`
  - generic recommendation fallback note

---

## Section E - Trend Views (2 slides)

## Slide 11 - Ratings and Risk Over Time
- Ratings by month
- Rating buckets by month (1-5 star lines)
- Predicted close by month
- Explain: sparse points usually means few windows survived gating.

## Slide 12 - Topic Trend Visuals
- Topics per class (top 6)
- Topics over time (top 6)
- Explain dependency on topic outputs and available business signal.

---

## Section F - Component Diagnostics (2 slides)

## Slide 13 - Component 2/3 Topic Diagnostics
- Negative-review count used by topic logic
- Terminal topics
- Recovery divergence by topic
- Explain why these can be empty (no rows for business, strict filters, missing optional table).

## Slide 14 - Component 4 Resilience Context
- City context
- Cuisine closure-rate matches
- Check-in floor context
- Recovery pattern
- Explain these as contextual priors, not direct model prediction.

---

## Section G - Evidence + Failure States (2 slides)

## Slide 15 - Evidence Reviews
- Full review table (date, stars, neg_prob, review_text).
- Purpose: auditability and explainability.

## Slide 16 - Not Scored Reasons and Debug Triage
- Common reasons:
  - `yelp_files_missing`
  - `vader_unavailable`
  - `tensorflow_runtime_unavailable`
  - `insufficient_history_for_windows`
- Fast triage order:
  1. Sidebar dependency indicators
  2. Health JSON expander
  3. Verify env vars and data paths
  4. Re-run with relaxed slider thresholds

---

## Appendix Slide - Environment Notes
- Launch command: `scripts/run_dashboard.sh`
- Install dashboard deps: `pip install -r dashboard/requirements.txt`
- Keep TensorFlow-compatible NumPy:
  - `numpy<2`
- Live sentiment requirements:
  - `vaderSentiment`
  - `nltk` + `vader_lexicon`

