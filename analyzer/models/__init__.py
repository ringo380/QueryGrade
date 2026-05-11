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

# Live database connection profiles
from .connection_models import UserDatabaseConnection
# Machine learning models
from .ml_models import FeedbackLearning, LearningMetrics, MLModel, TrainingData
# Core query models
from .query_models import Query, QueryAnalysis
# User interaction models
from .user_models import QueryFeedback, UserQueryHistory

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
    "TrainingData",
    "LearningMetrics",
    "FeedbackLearning",
]
