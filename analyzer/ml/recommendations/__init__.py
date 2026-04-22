"""
Recommendation Engines for QueryGrade ML

This package contains intelligent recommendation systems:

- contextual_engine: Context-aware recommendations
- personalization_engine: User-personalized feedback
- learning_paths: Educational learning path generation
- natural_language: Natural language feedback generation

All recommendation engines are production-ready.
"""

from .contextual_engine import (
    ContextualRecommendationsEngine,
    ImplementationComplexity,
    Recommendation,
    RecommendationContext,
    RecommendationPriority,
    RecommendationSet,
    RecommendationType,
)
from .learning_paths import (
    LearningFormat,
    LearningModule,
    LearningPathGenerator,
    LearningResource,
    PersonalizedLearningPath,
    SkillLevel,
    TopicCategory,
)
from .natural_language import (
    ComprehensiveFeedback,
    FeedbackCategory,
    FeedbackLevel,
    FeedbackMessage,
    FeedbackTone,
    NaturalLanguageFeedbackGenerator,
)
from .personalization_engine import (
    FeedbackPersonalizationEngine,
    FeedbackStyle,
    LearningStyle,
    PersonalizedFeedback,
    UserPersonality,
    UserProfile,
)

__all__ = [
    # Contextual Engine
    "ContextualRecommendationsEngine",
    "RecommendationContext",
    "Recommendation",
    "RecommendationSet",
    "RecommendationType",
    "RecommendationPriority",
    "ImplementationComplexity",
    # Personalization Engine
    "FeedbackPersonalizationEngine",
    "UserProfile",
    "PersonalizedFeedback",
    "LearningStyle",
    "FeedbackStyle",
    "UserPersonality",
    # Learning Paths
    "LearningPathGenerator",
    "PersonalizedLearningPath",
    "LearningModule",
    "LearningResource",
    "SkillLevel",
    "TopicCategory",
    "LearningFormat",
    # Natural Language
    "NaturalLanguageFeedbackGenerator",
    "ComprehensiveFeedback",
    "FeedbackMessage",
    "FeedbackTone",
    "FeedbackLevel",
    "FeedbackCategory",
]
