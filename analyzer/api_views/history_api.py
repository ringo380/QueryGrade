from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from ..models import QueryAnalysis, UserQueryHistory
from ..serializers import QueryHistoryListSerializer, QueryAnalysisSerializer
from .pagination import QueryGradingPagination


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
