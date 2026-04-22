"""
QueryGrade serializers package.

This package provides serializers for REST API operations organized by purpose:
- model_serializers: ORM model to JSON conversion
- request_serializers: API input validation
- response_serializers: API output formatting

All serializers are exported at the package level for backward compatibility
with existing imports.
"""

# Model serializers
from .model_serializers import (
    QueryAnalysisSerializer,
    QueryFeedbackSerializer,
    QueryHistoryListSerializer,
    QuerySerializer,
    UserQueryHistorySerializer,
    UserSerializer,
)

# Request serializers
from .request_serializers import (
    BatchQueryRequestSerializer,
    QueryGradeRequestSerializer,
)

# Response serializers
from .response_serializers import (
    BatchQueryResponseSerializer,
    QueryGradeResponseSerializer,
)

__all__ = [
    # Model serializers
    "QuerySerializer",
    "QueryAnalysisSerializer",
    "UserSerializer",
    "UserQueryHistorySerializer",
    "QueryFeedbackSerializer",
    "QueryHistoryListSerializer",
    # Request serializers
    "QueryGradeRequestSerializer",
    "BatchQueryRequestSerializer",
    # Response serializers
    "QueryGradeResponseSerializer",
    "BatchQueryResponseSerializer",
]
