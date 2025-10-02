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
    RecommendationContext,
    Recommendation,
    RecommendationSet,
    RecommendationType,
    RecommendationPriority,
    ImplementationComplexity
)

from .personalization_engine import (
    FeedbackPersonalizationEngine,
    UserProfile,
    PersonalizedFeedback,
    LearningStyle,
    FeedbackStyle,
    UserPersonality
)

from .learning_paths import (
    LearningPathGenerator,
    PersonalizedLearningPath,
    LearningModule,
    LearningResource,
    SkillLevel,
    TopicCategory,
    LearningFormat
)

from .natural_language import (
    NaturalLanguageFeedbackGenerator,
    ComprehensiveFeedback,
    FeedbackMessage,
    FeedbackTone,
    FeedbackLevel,
    FeedbackCategory
)

__all__ = [
    # Contextual Engine
    'ContextualRecommendationsEngine',
    'RecommendationContext',
    'Recommendation',
    'RecommendationSet',
    'RecommendationType',
    'RecommendationPriority',
    'ImplementationComplexity',

    # Personalization Engine
    'FeedbackPersonalizationEngine',
    'UserProfile',
    'PersonalizedFeedback',
    'LearningStyle',
    'FeedbackStyle',
    'UserPersonality',

    # Learning Paths
    'LearningPathGenerator',
    'PersonalizedLearningPath',
    'LearningModule',
    'LearningResource',
    'SkillLevel',
    'TopicCategory',
    'LearningFormat',

    # Natural Language
    'NaturalLanguageFeedbackGenerator',
    'ComprehensiveFeedback',
    'FeedbackMessage',
    'FeedbackTone',
    'FeedbackLevel',
    'FeedbackCategory',
]
