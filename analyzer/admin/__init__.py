"""
QueryGrade Django admin configuration package.

This package provides admin interfaces for all QueryGrade models organized by domain:
- query_admin: Core query models (Query)
- user_admin: User interaction models (UserQueryHistory, QueryFeedback)
- ml_admin: Machine learning models (MLModel, TrainingData, LearningMetrics, FeedbackLearning)

All admin classes are registered via @admin.register decorators in their respective modules.
This __init__.py file imports all modules to ensure registration occurs.
"""

from django.contrib import admin

# Import all admin modules to trigger @admin.register decorators
from . import ml_admin, query_admin, user_admin

# Custom admin site configuration
admin.site.site_header = "QueryGrade ML Administration"
admin.site.site_title = "QueryGrade ML Admin"
admin.site.index_title = "Machine Learning Dashboard"

from .ml_admin import (FeedbackLearningAdmin, LearningMetricsAdmin,
                       MLModelAdmin, TrainingDataAdmin)
# Export admin classes for explicit imports if needed
from .query_admin import QueryAdmin
from .user_admin import QueryFeedbackAdmin, UserQueryHistoryAdmin

__all__ = [
    "QueryAdmin",
    "UserQueryHistoryAdmin",
    "QueryFeedbackAdmin",
    "MLModelAdmin",
    "TrainingDataAdmin",
    "LearningMetricsAdmin",
    "FeedbackLearningAdmin",
]
