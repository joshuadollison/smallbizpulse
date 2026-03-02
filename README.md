# SmallBizPulse: Early Warning & Intervention System for Restaurant Survival

**Team:** AP (Armadillo Pulse) — Team #7  
**Course:** CIS 509 — Analytics Unstructured Data  

## Executive Summary
SmallBizPulse is an analytics-driven early warning system designed to predict restaurant survival and provide actionable intervention recommendations for local economic development agencies, chambers of commerce, and community organizations. As MS-AIB students, our approach bridges the gap between complex unstructured data and practical business execution. We leverage natural language processing and longitudinal engagement metrics to detect the "slow burn" of restaurant failure months or years before it happens.

## The Business Problem
Restaurant closures do not happen overnight; the failure rate in our analyzed market sits at a staggering 37.7% (1,557 out of 4,132 locations). Local stakeholders are currently dealing with this attrition reactively because they lack the consistent tooling to detect early warning signals. By capturing the decline in both sentiment and foot traffic early, interventions can be targeted and effective rather than performing a post-mortem on a closed business.

## Methods: Unstructured Data Analysis
Transforming raw text into business intelligence requires a multi-signal approach. Our methodology utilized the Yelp Academic Dataset (spanning 13 years and 72,124 restaurant reviews) to extract robust, predictive indicators:

*   **Sentiment Trajectory Tracking:** Using VADER sentiment analysis to construct a longitudinal view of customer satisfaction.
*   **Diagnostic Topic Modeling:** Employing BERTopic on 7.1 million tokens across a vocabulary of 163,953 words to dig into "why" the sentiment dropped (e.g., service slowing down, food quality changes).
*   **Behavioral Proxy Metrics:** Analyzing check-in frequency as a direct digital proxy for physical foot traffic and customer velocity.

## Results & Key Findings
Our Exploratory Data Analysis (EDA) revealed that warning signals are real and visible well before a closure event:

1.  **Engagement is the Strongest Early Indicator:** Open restaurants averaged 256 check-ins, whereas closed restaurants averaged only 108 (a significant 2.4x difference). Foot traffic tells the truth before reviews do.
2.  **Sentiment is a "Slow Burn":** The sentiment gap between open and closed businesses is statistically real (p < 0.001). More importantly, the decline in sentiment across the 36 months before a restaurant's last review is gradual and consistent. This provides a practical window for intervention.
3.  **Single Metrics Fail in Isolation:** Star ratings separate open from closed restaurants by a mere 0.03 points (3.46 vs. 3.43). Relying strictly on ratings is a flawed business strategy. Predicting closure demands a combined risk score of check-in velocity, sentiment trend, and review volume.

## Proposed Solution & Next Steps
Nobody is watching these combined signals at scale for small businesses. SmallBizPulse fills this market gap by offering not just a survival prediction, but a diagnosis. 

**Component 1:** Train and evaluate classifiers to generate automated Risk Scores.  
**Component 2:** Run continuous BERTopic modeling on negative reviews to surface the specific complaints driving the decline.  
**Component 3:** Map the discovered diagnostic topics to targeted, actionable interventions for localized economic development.  

This is not a thin dataset stretched to fit a hypothesis. The foundation for a generalizable, deployable, and highly relevant business system is here.