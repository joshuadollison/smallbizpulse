COMPONENT 1: MULTI-SIGNAL SURVIVAL PREDICTION MODEL
-----------------------------------------------------
Why it matters:
  In this dataset, 37.7% of restaurants are closed. That is not an
  outlier problem — it is a baseline failure rate. Chambers of commerce,
  economic development agencies, and small business lenders have no
  scalable tool to see which businesses are slipping before it is too
  late. This model is that tool.

Signal priority based on EDA results:
  1. Check-in frequency and velocity
       Open avg: 256 check-ins | Closed avg: 108 (2.4x difference)
       This is the primary trigger. It reflects actual foot traffic,
       not just customer opinion, and showed the largest separation
       between open and closed restaurants in the entire dataset.

  2. Review sentiment trend (VADER compound score over time)
       Open avg: 0.7129 | Closed avg: 0.6774
       Significant at p < 0.001 but Cohen's d = 0.07, so small in
       isolation. The useful signal is not the absolute score but
       whether it is declining over consecutive quarters. A sustained
       downward trend across 3+ quarters, combined with check-in drop,
       is what triggers a risk alert.

  3. Star ratings, review volume, and tip activity
       The star gap between open and closed restaurants is 0.03 (3.46
       vs. 3.43). Ratings are included as a supporting input but are
       not a reliable standalone trigger.

Modeling approach:
  - Logistic Regression and SVM as interpretable baselines
  - Gradient Boosting to handle signals with unequal predictive weight
  - RNN/LSTM for restaurants with more than 10 reviews, where a
    meaningful time sequence actually exists
  - Rule-based risk scoring for restaurants with 5 or fewer reviews,
    which account for 44.2% of the dataset. These businesses cannot
    support a trajectory model, so check-in count and star rating
    carry the classification for this segment.

Threshold:
  The trajectory analysis found sentiment declining at 0.0014 compound
  score per month across the 36 months before closure. That defines the
  monitoring window. ROC analysis in the modeling phase will set the
  operating threshold, balancing false alarms against missed closures.
  For the business case, missing a closure is the more costly error.


COMPONENT 2: DIAGNOSTIC TOPIC MODELING
---------------------------------------
Why it matters:
  A risk score alone does not help a chamber of commerce program
  officer. They need to know what is going wrong so they can send the
  business to the right resource. "Your sentiment is declining" is not
  actionable. "Your reviews are dominated by service complaints over the
  last six months" is something a staff trainer can work with.

Method: BERTopic applied to negative reviews of flagged restaurants
  - Negative is defined as stars <= 2 AND VADER compound <= -0.05.
    Both conditions required, not just one.
  - Terminal analysis looks at the final 10 reviews before a
    restaurant's last recorded date to identify what complaint pattern
    was present right before closure.
  - Recovery comparison: restaurants that went through a negative period
    and stayed open vs. those that closed. What is different about the
    topics, and at what point does the pattern diverge?

  The specific failure categories (service, food, pricing, management)
  are working hypotheses. BERTopic will confirm or replace them based
  on what the text actually contains.


COMPONENT 3: INTERVENTION RECOMMENDATION ENGINE
-------------------------------------------------
Why it matters:
  SmallBizPulse is not a research tool. It is an operational tool for
  organizations that need to direct limited intervention resources to
  the right businesses at the right time. The recommendation engine
  connects a diagnosed problem to a specific program or partner.

How it works:
  Each topic cluster from Component 2 maps to an intervention type:
    Operational breakdown  ->  Small Business Development Center referral
    Food or supply issues  ->  Culinary consulting or supplier network
    Staff and service      ->  Workforce training program connection
    Financial distress     ->  CDFI loan program or grant eligibility

  The system gives the case manager a starting point. The human still
  makes the decision. The tool just makes sure they are not starting
  from scratch every time.


COMPONENT 4: RESILIENCE AND VULNERABILITY ANALYSIS
----------------------------------------------------
Why it matters:
  Not every restaurant faces the same risk level. Intervention budgets
  are not unlimited. This component helps stakeholders understand which
  segments, cuisines, and geographies have the highest failure rates so
  resources go where they matter most.

Questions grounded in EDA findings:
  - City-level closure rates already vary across the 12 cities in this
    dataset. Which local conditions correlate with above-average failure
    rates? What can a city do differently based on that?
  - Do certain cuisine subcategories fail at higher rates regardless of
    their sentiment scores? Category risk is a useful baseline feature.
  - Is there a check-in floor below which the closure probability jumps
    significantly? The 2.4x gap in the data suggests there is a
    meaningful threshold somewhere in that range.
  - Among restaurants that survived a rough stretch, what do their
    recovery patterns look like? Those patterns define what intervention
    success should look like.