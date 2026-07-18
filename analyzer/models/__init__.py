"""
QueryGrade models package.

This package provides Django ORM models for the QueryGrade application
organized by domain:
- query_models: Core SQL query storage and analysis
- user_models: User interaction and feedback tracking
- ml_models: Machine learning system metadata and training

All models are exported at the package level for backward compatibility
with existing migrations and imports.

IMPORTANT: This package maintains the same import paths as the original
monolithic models.py to ensure Django migrations continue to work correctly.
"""

# Machine learning models
from .ml_alert_models import MLAlert
from .ml_models import (
    FeedbackLearning,
    LearningMetrics,
    MLModel,
    MLModelArtifact,
    TrainingData,
)

# Core query models
from .query_models import Query, QueryAnalysis

# User interaction models
from .user_models import QueryFeedback, UserQueryHistory

# Live database connection profiles — must stay last: connection_models imports
# analyzer.services which re-imports from this module; the names above must
# already be bound before that chain resolves.
from .connection_models import UserDatabaseConnection  # isort: skip

__all__ = [
    # Query models
    "Query",
    "QueryAnalysis",
    # User models
    "UserQueryHistory",
    "QueryFeedback",
    # Connection models
    "UserDatabaseConnection",
    # ML models
    "MLModel",
    "MLModelArtifact",
    "TrainingData",
    "LearningMetrics",
    "FeedbackLearning",
    "MLAlert",
]
