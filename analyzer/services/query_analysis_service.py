"""
Query Analysis Service

Handles all business logic related to SQL query analysis including:
- Traditional rule-based analysis
- ML-enhanced analysis
- User history tracking
- Error handling and validation
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from django.contrib.auth.models import User

from ..ml.analysis.unified_analyzer import AnalysisRequest, UnifiedQueryAnalyzer
from ..models import Query, QueryAnalysis, UserQueryHistory
from ..query_analyzer import analyze_query

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysisRequest:
    """DTO for query analysis requests"""

    sql_query: str
    user: User
    database_type: str = ""
    database_version: str = ""
    use_case_notes: str = ""
    ip_address: str = ""
    user_agent: str = ""
    enable_ml: bool = True


@dataclass
class QueryAnalysisResult:
    """DTO for query analysis results"""

    query: Query
    analysis: QueryAnalysis
    user_history: UserQueryHistory
    ml_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class QueryAnalysisService:
    """
    Service for handling SQL query analysis operations.

    This service encapsulates the business logic for:
    - Analyzing SQL queries with rule-based and ML methods
    - Creating user history records
    - Handling errors gracefully
    - Orchestrating ML analysis
    """

    def __init__(self):
        """Initialize the query analysis service."""
        self.unified_analyzer = None

    def analyze_query_for_user(
        self, request: QueryAnalysisRequest
    ) -> QueryAnalysisResult:
        """
        Analyze a SQL query for a specific user.

        Args:
            request: QueryAnalysisRequest containing query details

        Returns:
            QueryAnalysisResult containing analysis results

        Raises:
            ValueError: If query has syntax errors
            Exception: For unexpected errors
        """
        try:
            # Perform traditional rule-based analysis
            query, analysis = analyze_query(request.sql_query, request.database_type)

            # Perform ML-enhanced analysis if enabled
            ml_analysis = None
            if request.enable_ml:
                ml_analysis = self._perform_ml_analysis(
                    request.sql_query,
                    request.user.id,
                    request.database_type,
                    request.database_version,
                    request.use_case_notes,
                    request.user_agent,
                    request.ip_address,
                )

            # Create user history record
            user_history = self._create_user_history(
                request.user,
                query,
                request.ip_address,
                request.user_agent,
                request.database_type,
                request.database_version,
                request.use_case_notes,
            )

            return QueryAnalysisResult(
                query=query,
                analysis=analysis,
                user_history=user_history,
                ml_analysis=ml_analysis,
            )

        except ValueError as e:
            # Handle SQL syntax errors
            error_msg = self._format_syntax_error(str(e))
            raise ValueError(error_msg) from e

        except Exception as e:
            logger.error(
                f"Unexpected error analyzing query for user {request.user.username}: {e}"
            )
            raise

    def _perform_ml_analysis(
        self,
        sql_query: str,
        user_id: int,
        database_type: str,
        database_version: str,
        use_case_notes: str,
        user_agent: str,
        ip_address: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Perform ML-enhanced query analysis.

        Args:
            sql_query: The SQL query to analyze
            user_id: ID of the user performing the analysis
            database_type: Target database type
            database_version: Database version
            use_case_notes: User's use case description
            user_agent: User agent string
            ip_address: Client IP address

        Returns:
            Dictionary containing ML analysis results, or None if analysis fails
        """
        try:
            if self.unified_analyzer is None:
                self.unified_analyzer = UnifiedQueryAnalyzer()

            analysis_request = AnalysisRequest(
                query=sql_query,
                user_id=str(user_id),
                database_type=database_type,
                database_version=database_version,
                context={
                    "use_case": use_case_notes,
                    "user_agent": user_agent,
                    "ip_address": ip_address,
                },
            )

            # Run async analysis in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ml_result = loop.run_until_complete(
                    self.unified_analyzer.analyze_query(analysis_request)
                )

                # Convert to dictionary for session storage
                return {
                    "semantic_metrics": ml_result.semantic_metrics,
                    "performance_prediction": ml_result.performance_prediction,
                    "feedback": ml_result.feedback,
                    "recommendations": ml_result.recommendations,
                    "personalized_feedback": ml_result.personalized_feedback,
                    "rewrite_suggestions": ml_result.rewrite_suggestions,
                }

            finally:
                loop.close()

        except Exception as ml_error:
            logger.warning(f"ML analysis failed for user {user_id}: {ml_error}")
            return None

    def _create_user_history(
        self,
        user: User,
        query: Query,
        ip_address: str,
        user_agent: str,
        database_type: str,
        database_version: str,
        use_case_notes: str,
    ) -> UserQueryHistory:
        """
        Create a user history record for the query analysis.

        Args:
            user: User performing the analysis
            query: Query object
            ip_address: Client IP address
            user_agent: User agent string
            database_type: Target database type
            database_version: Database version
            use_case_notes: User's use case description

        Returns:
            Created UserQueryHistory object
        """
        return UserQueryHistory.objects.create(
            user=user,
            query=query,
            ip_address=ip_address,
            user_agent=user_agent[:255],  # Truncate to field length
            database_type=database_type,
            database_version=database_version,
            use_case_notes=use_case_notes,
        )

    def _format_syntax_error(self, error_msg: str) -> str:
        """
        Format SQL syntax error messages for user display.

        Args:
            error_msg: Raw error message

        Returns:
            User-friendly error message
        """
        if "typos in keywords" in error_msg:
            return "SQL syntax error: Your query contains apparent typos in SQL keywords. Please check your spelling."
        elif "Unable to parse" in error_msg:
            return "SQL parsing error: We couldn't parse your SQL query. Please check the syntax and try again."
        elif "No SQL keywords found" in error_msg:
            return "Invalid input: No SQL keywords detected. Please enter a valid SQL query."
        else:
            return f"SQL error: {error_msg}"

    def get_analysis_by_id(self, analysis_id: int) -> Optional[QueryAnalysis]:
        """
        Retrieve a query analysis by ID.

        Args:
            analysis_id: ID of the QueryAnalysis object

        Returns:
            QueryAnalysis object or None if not found
        """
        try:
            return QueryAnalysis.objects.get(id=analysis_id)
        except QueryAnalysis.DoesNotExist:
            return None
