import logging

from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ..exceptions import QueryAnalysisError
from ..models import Query, QueryAnalysis, UserQueryHistory
from ..query_analyzer import analyze_query
from ..serializers import (BatchQueryRequestSerializer,
                           QueryGradeRequestSerializer,
                           QueryGradeResponseSerializer)

logger = logging.getLogger(__name__)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def grade_query_api(request):
    """
    Grade a single SQL query via API.

    POST /api/grade-query/
    {
        "sql_text": "SELECT * FROM users WHERE id = 1",
        "database_type": "mysql",
        "database_version": "8.0",
        "use_case_notes": "User lookup query"
    }
    """
    serializer = QueryGradeRequestSerializer(data=request.data)

    if not serializer.is_valid():
        return Response(
            {"error": "Invalid request data", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        # Analyze the query
        query, analysis = analyze_query(
            serializer.validated_data["sql_text"],
            serializer.validated_data["database_type"],
        )

        # Create user history record
        user_history = UserQueryHistory.objects.create(
            user=request.user,
            query=query,
            database_type=serializer.validated_data["database_type"],
            database_version=serializer.validated_data["database_version"],
            use_case_notes=serializer.validated_data["use_case_notes"],
            ip_address=request.META.get("REMOTE_ADDR"),
            user_agent=request.META.get("HTTP_USER_AGENT", "")[:255],
        )

        # Prepare response data
        response_data = {
            "query_id": query.id,
            "analysis_id": analysis.id,
            "grade": analysis.grade,
            "score": analysis.score,
            "query_type": query.query_type,
            "complexity": query.estimated_complexity,
            "issues_found": analysis.issues_found,
            "recommendations": analysis.recommendations,
            "performance_notes": analysis.performance_notes,
            "execution_time_ms": analysis.execution_time_ms,
            "created_at": analysis.created_at,
        }

        response_serializer = QueryGradeResponseSerializer(response_data)

        logger.info(
            f"API query graded for user {request.user.username}: {analysis.grade} ({analysis.score:.1f})"
        )

        return Response(response_serializer.data, status=status.HTTP_201_CREATED)

    except QueryAnalysisError as e:
        logger.warning(f"Query analysis error for user {request.user.username}: {e}")
        return Response(
            {"error": "Query analysis failed", "message": str(e)},
            status=status.HTTP_400_BAD_REQUEST,
        )

    except Exception as e:
        logger.error(
            f"Unexpected error in API query grading for user {request.user.username}: {e}"
        )
        return Response(
            {
                "error": "Internal server error",
                "message": "An unexpected error occurred while analyzing the query",
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def batch_analysis_api(request):
    """
    Analyze multiple SQL queries in a batch via API.

    POST /api/batch-analysis/
    {
        "queries": ["SELECT * FROM users", "SELECT COUNT(*) FROM orders"],
        "database_type": "postgresql",
        "database_version": "13.0",
        "analysis_notes": "Performance review queries"
    }
    """
    serializer = BatchQueryRequestSerializer(data=request.data)

    if not serializer.is_valid():
        return Response(
            {"error": "Invalid request data", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    queries = serializer.validated_data["queries"]
    database_type = serializer.validated_data["database_type"]
    database_version = serializer.validated_data["database_version"]
    analysis_notes = serializer.validated_data["analysis_notes"]

    successful_results = []
    failed_results = []
    total_score = 0

    for i, sql_text in enumerate(queries, 1):
        try:
            # Analyze each query
            query, analysis = analyze_query(sql_text, database_type)

            # Create user history record
            user_history = UserQueryHistory.objects.create(
                user=request.user,
                query=query,
                database_type=database_type,
                database_version=database_version,
                use_case_notes=f"Batch analysis {i}/{len(queries)}: {analysis_notes}",
                ip_address=request.META.get("REMOTE_ADDR"),
                user_agent=request.META.get("HTTP_USER_AGENT", "")[:255],
            )

            result_data = {
                "query_id": query.id,
                "analysis_id": analysis.id,
                "grade": analysis.grade,
                "score": analysis.score,
                "query_type": query.query_type,
                "complexity": query.estimated_complexity,
                "issues_found": analysis.issues_found,
                "recommendations": analysis.recommendations,
                "performance_notes": analysis.performance_notes,
                "execution_time_ms": analysis.execution_time_ms,
                "created_at": analysis.created_at,
            }

            successful_results.append(result_data)
            total_score += analysis.score

        except Exception as e:
            logger.warning(
                f"Failed to analyze query {i} for user {request.user.username}: {e}"
            )
            failed_results.append(
                {
                    "query_index": i,
                    "query": (
                        sql_text[:100] + "..." if len(sql_text) > 100 else sql_text
                    ),
                    "error": str(e),
                }
            )

    # Calculate averages
    successful_count = len(successful_results)
    failed_count = len(failed_results)

    if successful_count > 0:
        average_score = total_score / successful_count
        grade_counts = {}
        for result in successful_results:
            grade = result["grade"]
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        # Find most common grade
        average_grade = max(grade_counts.items(), key=lambda x: x[1])[0]
    else:
        average_score = 0
        average_grade = "N/A"

    # Generate summary
    summary_lines = [
        f"Batch analysis completed: {successful_count} successful, {failed_count} failed",
        (
            f"Average score: {average_score:.1f}/100"
            if successful_count > 0
            else "No successful analyses"
        ),
    ]

    if successful_count > 0:
        summary_lines.append(f"Most common grade: {average_grade}")

    response_data = {
        "total_queries": len(queries),
        "successful_analyses": successful_count,
        "failed_analyses": failed_count,
        "average_grade": average_grade,
        "average_score": average_score,
        "results": successful_results,
        "summary": "\n".join(summary_lines),
        "created_at": timezone.now(),
    }

    logger.info(
        f"API batch analysis completed for user {request.user.username}: {successful_count}/{len(queries)} successful"
    )

    return Response(response_data, status=status.HTTP_201_CREATED)
