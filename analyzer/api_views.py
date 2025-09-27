from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.pagination import PageNumberPagination
from django.utils import timezone
from django.db.models import Q, Avg, Count
from django.contrib.auth.models import User
import logging

from .models import Query, QueryAnalysis, UserQueryHistory, QueryFeedback
from .serializers import (
    QuerySerializer, QueryAnalysisSerializer, UserQueryHistorySerializer,
    QueryFeedbackSerializer, QueryGradeRequestSerializer, QueryGradeResponseSerializer,
    BatchQueryRequestSerializer, BatchQueryResponseSerializer, QueryHistoryListSerializer
)
from .query_analyzer import analyze_query
from .exceptions import QueryAnalysisError

logger = logging.getLogger(__name__)


class QueryGradingPagination(PageNumberPagination):
    """Custom pagination for query grading API."""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


@api_view(['POST'])
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
        return Response({
            'error': 'Invalid request data',
            'details': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Analyze the query
        query, analysis = analyze_query(
            serializer.validated_data['sql_text'],
            serializer.validated_data['database_type']
        )

        # Create user history record
        user_history = UserQueryHistory.objects.create(
            user=request.user,
            query=query,
            database_type=serializer.validated_data['database_type'],
            database_version=serializer.validated_data['database_version'],
            use_case_notes=serializer.validated_data['use_case_notes'],
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT', '')[:255]
        )

        # Prepare response data
        response_data = {
            'query_id': query.id,
            'analysis_id': analysis.id,
            'grade': analysis.grade,
            'score': analysis.score,
            'query_type': query.query_type,
            'complexity': query.estimated_complexity,
            'issues_found': analysis.issues_found,
            'recommendations': analysis.recommendations,
            'performance_notes': analysis.performance_notes,
            'execution_time_ms': analysis.execution_time_ms,
            'created_at': analysis.created_at
        }

        response_serializer = QueryGradeResponseSerializer(response_data)

        logger.info(f"API query graded for user {request.user.username}: {analysis.grade} ({analysis.score:.1f})")

        return Response(response_serializer.data, status=status.HTTP_201_CREATED)

    except QueryAnalysisError as e:
        logger.warning(f"Query analysis error for user {request.user.username}: {e}")
        return Response({
            'error': 'Query analysis failed',
            'message': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        logger.error(f"Unexpected error in API query grading for user {request.user.username}: {e}")
        return Response({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred while analyzing the query'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
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
        return Response({
            'error': 'Invalid request data',
            'details': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    queries = serializer.validated_data['queries']
    database_type = serializer.validated_data['database_type']
    database_version = serializer.validated_data['database_version']
    analysis_notes = serializer.validated_data['analysis_notes']

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
                ip_address=request.META.get('REMOTE_ADDR'),
                user_agent=request.META.get('HTTP_USER_AGENT', '')[:255]
            )

            result_data = {
                'query_id': query.id,
                'analysis_id': analysis.id,
                'grade': analysis.grade,
                'score': analysis.score,
                'query_type': query.query_type,
                'complexity': query.estimated_complexity,
                'issues_found': analysis.issues_found,
                'recommendations': analysis.recommendations,
                'performance_notes': analysis.performance_notes,
                'execution_time_ms': analysis.execution_time_ms,
                'created_at': analysis.created_at
            }

            successful_results.append(result_data)
            total_score += analysis.score

        except Exception as e:
            logger.warning(f"Failed to analyze query {i} for user {request.user.username}: {e}")
            failed_results.append({
                'query_index': i,
                'query': sql_text[:100] + "..." if len(sql_text) > 100 else sql_text,
                'error': str(e)
            })

    # Calculate averages
    successful_count = len(successful_results)
    failed_count = len(failed_results)

    if successful_count > 0:
        average_score = total_score / successful_count
        grade_counts = {}
        for result in successful_results:
            grade = result['grade']
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        # Find most common grade
        average_grade = max(grade_counts.items(), key=lambda x: x[1])[0]
    else:
        average_score = 0
        average_grade = 'N/A'

    # Generate summary
    summary_lines = [
        f"Batch analysis completed: {successful_count} successful, {failed_count} failed",
        f"Average score: {average_score:.1f}/100" if successful_count > 0 else "No successful analyses"
    ]

    if successful_count > 0:
        summary_lines.append(f"Most common grade: {average_grade}")

    response_data = {
        'total_queries': len(queries),
        'successful_analyses': successful_count,
        'failed_analyses': failed_count,
        'average_grade': average_grade,
        'average_score': average_score,
        'results': successful_results,
        'summary': '\n'.join(summary_lines),
        'created_at': timezone.now()
    }

    logger.info(f"API batch analysis completed for user {request.user.username}: {successful_count}/{len(queries)} successful")

    return Response(response_data, status=status.HTTP_201_CREATED)


class QueryHistoryListAPIView(generics.ListAPIView):
    """
    List user's query history via API.

    GET /api/query-history/
    """
    serializer_class = QueryHistoryListSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = QueryGradingPagination

    def get_queryset(self):
        """Get query history for the authenticated user."""
        return UserQueryHistory.objects.filter(
            user=self.request.user
        ).select_related('query', 'query__analysis').order_by('-submitted_at')


class QueryAnalysisDetailAPIView(generics.RetrieveAPIView):
    """
    Get detailed analysis results via API.

    GET /api/analysis/{analysis_id}/
    """
    serializer_class = QueryAnalysisSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Ensure users can only access their own analyses."""
        return QueryAnalysis.objects.filter(
            query__userqueryhistory__user=self.request.user
        ).select_related('query')


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def submit_feedback_api(request, analysis_id):
    """
    Submit feedback for a query analysis via API.

    POST /api/feedback/{analysis_id}/
    {
        "accuracy_rating": 4,
        "usefulness_rating": 5,
        "clarity_rating": 4,
        "suggestions": "Great analysis!",
        "would_recommend": true
    }
    """
    try:
        # Verify user has access to this analysis
        user_history = UserQueryHistory.objects.get(
            query__analysis__id=analysis_id,
            user=request.user
        )
    except UserQueryHistory.DoesNotExist:
        return Response({
            'error': 'Analysis not found or access denied'
        }, status=status.HTTP_404_NOT_FOUND)

    # Check if feedback already exists
    existing_feedback = QueryFeedback.objects.filter(user_history=user_history).first()

    if existing_feedback:
        # Update existing feedback
        serializer = QueryFeedbackSerializer(existing_feedback, data=request.data, partial=True)
    else:
        # Create new feedback
        serializer = QueryFeedbackSerializer(data=request.data)

    if not serializer.is_valid():
        return Response({
            'error': 'Invalid feedback data',
            'details': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    try:
        if existing_feedback:
            # Update existing
            for field, value in serializer.validated_data.items():
                setattr(existing_feedback, field, value)
            existing_feedback.save()
            feedback = existing_feedback
            action = 'updated'
        else:
            # Create new
            feedback = QueryFeedback.objects.create(
                user_history=user_history,
                **serializer.validated_data
            )
            action = 'created'

        # Update user history
        user_history.was_helpful = True
        user_history.feedback_comments = feedback.suggestions
        user_history.save()

        logger.info(f"Feedback {action} via API for user {request.user.username}, analysis {analysis_id}")

        return Response({
            'message': f'Feedback {action} successfully',
            'feedback_id': feedback.id
        }, status=status.HTTP_201_CREATED if action == 'created' else status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error saving feedback via API for user {request.user.username}: {e}")
        return Response({
            'error': 'Failed to save feedback',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_stats_api(request):
    """
    Get user statistics via API.

    GET /api/user-stats/
    """
    user_history = UserQueryHistory.objects.filter(user=request.user)

    if not user_history.exists():
        return Response({
            'total_queries': 0,
            'message': 'No query history found'
        })

    # Calculate statistics
    total_queries = user_history.count()

    # Grade distribution
    grade_distribution = {}
    for grade_choice in QueryAnalysis.GRADE_CHOICES:
        grade = grade_choice[0]
        count = user_history.filter(query__analysis__grade=grade).count()
        grade_distribution[grade] = count

    # Average score
    avg_score = user_history.aggregate(
        avg_score=Avg('query__analysis__score')
    )['avg_score'] or 0

    # Feedback statistics
    feedback_count = QueryFeedback.objects.filter(
        user_history__user=request.user
    ).count()

    # Recent activity (last 30 days)
    from datetime import timedelta
    thirty_days_ago = timezone.now() - timedelta(days=30)
    recent_queries = user_history.filter(submitted_at__gte=thirty_days_ago).count()

    stats = {
        'total_queries': total_queries,
        'average_score': round(avg_score, 2),
        'grade_distribution': grade_distribution,
        'feedback_submitted': feedback_count,
        'recent_activity': {
            'queries_last_30_days': recent_queries
        },
        'most_common_database': user_history.values('database_type').annotate(
            count=Count('database_type')
        ).order_by('-count').first()
    }

    return Response(stats)


@api_view(['GET'])
@permission_classes([AllowAny])
def api_health(request):
    """
    API health check endpoint.

    GET /api/health/
    """
    return Response({
        'status': 'healthy',
        'timestamp': timezone.now(),
        'version': '1.0',
        'features': {
            'query_grading': True,
            'batch_analysis': True,
            'feedback_system': True,
            'user_statistics': True
        }
    })