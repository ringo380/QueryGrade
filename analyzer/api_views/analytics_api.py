from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from django.db.models import Avg, Count
from datetime import timedelta

from ..models import QueryAnalysis, QueryFeedback, UserQueryHistory


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
