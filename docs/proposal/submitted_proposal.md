# SmallBizPulse: An Early Warning & Intervention System for Restaurant Survival

**Team ID:** AP (Armadillo Pulse) - Team # 7  
**Team Members:** Lafi Alanazi, Afan Jeelani, Chandra Carr, Joshua Dollison, Sonia Parikh  

## Executive Summary

SmallBizPulse is an analytics-driven early warning system that predicts restaurant survival and provides actionable intervention recommendations for local economic development agencies, chambers of commerce, and community organizations.  By leveraging sentiment analysis and topic modeling on Yelp review data, our tool identifies at-risk restaurants before they close, enabling timely support that protects local jobs, preserves neighborhood vitality, and strengthens community economic resilience.  Unlike traditional prediction models that simply forecast outcomes, SmallBizPulse diagnoses the underlying issues driving decline and recommends targeted interventions, transforming data insights into community impact.

## Project Background

### The Problem:
Non-chain restaurants are the backbone of local economies, yet they face significant failure rates.  Research from Cornell Hospitality Quarterly indicates that 26% of independent restaurants fail in their first year, while the National Restaurant Association reports an industry-wide failure rate of approximately 30%.  Our own analysis of the Yelp dataset confirms this vulnerability: 37.7% of restaurants in the dataset have already closed, compared to just 25.7% of all businesses.  These closures devastate communities: personal investments evaporate, jobs disappear, neighborhoods lose character, and economic ripple effects harm surrounding businesses.  Current intervention efforts are reactive, occurring only after closure is imminent or complete.

### The Opportunity:
Online reviews contain early warning signals, such as declining service quality, food inconsistencies, and management issues, that precede closure by months or even years.  Our analysis of the Yelp dataset reveals that 37.7% of restaurants (1,557 out of 4,131) have closed, making it a rich environment for predictive modeling.  Additionally, review text provides valuable sentiment and thematic signals that can reveal underlying business challenges before ratings drop significantly.

### The Gap:
Existing survival prediction research focuses heavily on numeric indicators such as star ratings, review volume, or check-in counts, but lacks actionable diagnostic capabilities.  Stakeholders need more than prediction-they need to understand why a restaurant is failing and what can be done to help.

### Our Solution:
SmallBizPulse addresses this gap by combining predictive modeling with diagnostic topic analysis and a recommendation engine.  Our goal is to provide economic development agencies, chambers of commerce, and community organizations with a proactive tool to monitor restaurant health, identify at-risk businesses early, and deliver targeted support recommendations based on the nature of customer complaints.

## Business Question

How can we use publicly available online reviews to predict restaurant closure risk and identify actionable factors contributing to restaurant decline?

## Data Sources

### Primary Data Source

**Yelp Academic Dataset (2025 Release):**
- **Business Data:** restaurant attributes, categories, location, operational status (open/closed)
- **Review Data:** text reviews, star ratings, timestamp, user IDs
- **User Data:** reviewer history, review counts, and activity levels

### Data Filtering Criteria

To build a relevant dataset, we will apply the following filters:
1. **Category Filter:** Select only businesses categorized as "Restaurants" or related food-service categories.
2. **Location Scope:** Focus on businesses within major U.S. metro areas to ensure sufficient data density.
3. **Review Volume Threshold:** Include restaurants with a minimum number of reviews (e.g., 20+) to ensure signal reliability.
4. **Time Window:** Use multi-year review histories to capture sentiment trajectories and survival outcomes.

### Optional External Data (If Available)

- Local unemployment rates or economic indicators (for contextual factors)
- Census demographic and neighborhood variables (to study vulnerability patterns)

## Proposed Methods

### 1. Sentiment Trajectory Modeling

We will compute sentiment scores for reviews over time and track trends in sentiment decline.  Candidate approaches:
- Naive Bayes / Logistic Regression sentiment classifier
- Deep Learning sentiment classifier (LSTM / Transformer-based)
- Aggregate sentiment change rates as predictive features

### 2. Topic Modeling for Diagnostic Insights (Required)

We will extract themes from negative reviews to identify drivers of decline.  Candidate approaches:
- LDA (Latent Dirichlet Allocation)
- BERTopic or other transformer-based topic models

Topics such as "slow service," "food quality decline," "rude staff," or "dirty environment" can be mapped to actionable intervention recommendations.

### 3. Closure Risk Prediction (Survival / Classification Models)

We will predict whether a restaurant is likely to close using:
- Supervised classification models (Random Forest, XGBoost, Neural Networks)
- Survival analysis models (Cox Proportional Hazards) if closure timing can be incorporated

Key features:
- Sentiment trends
- Star rating trajectories
- Review frequency and velocity
- Topic prevalence over time
- Business attributes (price range, category, city)

### 4. Recommendation Mapping Engine

We will link diagnostic topics to intervention strategies for stakeholders.  Examples:
- "Slow service" - staff training, operations review
- "Food quality decline" - supplier evaluation, kitchen oversight
- "Rude staff" - customer service coaching

## Expected Outcomes and Impact

- Early warning system for at-risk restaurants
- Diagnostic insights into why restaurants decline
- Actionable recommendations for intervention
- Improved local economic resilience through proactive support
- Comparative insights into what differentiates survivors from closures

## Deliverables

Our final deliverables will include:
- A predictive model that estimates closure risk
- A dashboard summarizing key restaurant risk indicators
- Topic-based diagnostics for declining businesses
- A recommendation engine providing intervention suggestions
- Visualizations and summaries for stakeholder decision-making

## References

National Restaurant Association.  (2024).  Restaurant Industry Facts at a Glance.  
https://restaurant.org/research-and-media/research/industry-statistics/

Parsa, H. G., Self, J. T., Njite, D., & King, T.  (2005).  Why Restaurants Fail.  Cornell Hotel and Restaurant Administration Quarterly, 46(3), 304-322.

Yelp Inc.  (2025).  Yelp Academic Dataset.  https://www.yelp.com/dataset

Zhang, X., & Luo, L.  (2023).  Can Consumer-Posted Photos Serve as a Leading Indicator of Restaurant Survival?  Management Science, 69(1), 539-555.