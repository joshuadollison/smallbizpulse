"""Model builder v2 aligned to `docs/methodology.md`."""

from .features import (
    MonthlyFeatureConfig,
    MonthlySignalArtifacts,
    SequenceWindowConfig,
    SequenceWindows,
    build_monthly_signal_panel,
    build_sequence_windows,
    compute_business_snapshot_features,
)
from .io import (
    DatasetLoadError,
    RestaurantTables,
    YelpDatasetPaths,
    YelpTables,
    filter_restaurants,
    load_restaurant_tables,
    load_yelp_tables,
    summarize_tables,
)
from .inference import InferenceDependencyError, SurvivalRuntime, SurvivalRuntimeArtifacts
from .pipeline import ModelBuilderV2, ModelBuilderV2Artifacts, ModelBuilderV2Config
from .recommendation_engine import (
    DEFAULT_RULES,
    InterventionRule,
    RecommendationArtifacts,
    build_recommendation_artifacts,
    map_topic_terms_to_recommendations,
)
from .resilience import (
    ResilienceArtifacts,
    build_resilience_artifacts,
    checkin_floor_analysis,
    city_level_closure_rates,
    cuisine_level_closure_rates,
    recovery_pattern_analysis,
)
from .sentiment import SentimentDependencyError, VaderSentimentScorer, add_vader_compound
from .survival import (
    BaselineModelResult,
    GruModelResult,
    RuleBasedRiskModel,
    SurvivalTrainingArtifacts,
    SurvivalTrainingConfig,
    train_survival_models,
)
from .topic_modeling import (
    TopicModelArtifacts,
    TopicModelConfig,
    TopicModelDependencyError,
    filter_negative_reviews,
    train_diagnostic_topic_model,
)

__all__ = [
    "DEFAULT_RULES",
    "BaselineModelResult",
    "DatasetLoadError",
    "InferenceDependencyError",
    "GruModelResult",
    "InterventionRule",
    "ModelBuilderV2",
    "ModelBuilderV2Artifacts",
    "ModelBuilderV2Config",
    "MonthlyFeatureConfig",
    "MonthlySignalArtifacts",
    "RecommendationArtifacts",
    "ResilienceArtifacts",
    "RestaurantTables",
    "RuleBasedRiskModel",
    "SurvivalRuntime",
    "SurvivalRuntimeArtifacts",
    "SentimentDependencyError",
    "SequenceWindowConfig",
    "SequenceWindows",
    "SurvivalTrainingArtifacts",
    "SurvivalTrainingConfig",
    "TopicModelArtifacts",
    "TopicModelConfig",
    "TopicModelDependencyError",
    "VaderSentimentScorer",
    "YelpDatasetPaths",
    "YelpTables",
    "add_vader_compound",
    "build_monthly_signal_panel",
    "build_recommendation_artifacts",
    "build_resilience_artifacts",
    "build_sequence_windows",
    "checkin_floor_analysis",
    "city_level_closure_rates",
    "compute_business_snapshot_features",
    "cuisine_level_closure_rates",
    "filter_negative_reviews",
    "filter_restaurants",
    "load_restaurant_tables",
    "load_yelp_tables",
    "map_topic_terms_to_recommendations",
    "recovery_pattern_analysis",
    "summarize_tables",
    "train_diagnostic_topic_model",
    "train_survival_models",
]
