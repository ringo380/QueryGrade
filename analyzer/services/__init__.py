"""
Service Layer for QueryGrade

This package contains business logic extracted from views, following the
Service Layer pattern. Services handle:
- Business rules and validation
- Domain logic orchestration
- Data transformation
- External service integration

Services are thin wrappers around domain logic that:
1. Accept primitive types or DTOs
2. Coordinate multiple operations
3. Return results or raise domain exceptions
4. Are transaction-aware

Usage:
    from analyzer.services import QueryAnalysisService

    service = QueryAnalysisService()
    result = service.analyze_query(sql_text, database_type, user_id)
"""

from .database_introspection_service import DatabaseIntrospectionService
from .feedback_service import FeedbackService
from .query_analysis_service import QueryAnalysisService

__all__ = [
    "QueryAnalysisService",
    "FeedbackService",
    "DatabaseIntrospectionService",
]

__version__ = "1.0.0"
