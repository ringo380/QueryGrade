from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.db import transaction

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


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_query_history(request):
    """
    Delete query history entries for the authenticated user.

    DELETE /api/query-history/delete/
    Body: {"query_ids": [1, 2, 3]}

    Returns:
        200: {"deleted": 3, "message": "Successfully deleted 3 queries"}
        400: {"error": "query_ids is required and must be a list"}
        403: {"error": "You can only delete your own query history"}
    """
    query_ids = request.data.get('query_ids', [])

    # Validate input
    if not isinstance(query_ids, list) or not query_ids:
        return Response(
            {'error': 'query_ids is required and must be a non-empty list'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Validate all IDs are integers
    try:
        query_ids = [int(qid) for qid in query_ids]
    except (ValueError, TypeError):
        return Response(
            {'error': 'All query_ids must be valid integers'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Delete only the user's own history entries
    with transaction.atomic():
        deleted_count, _ = UserQueryHistory.objects.filter(
            id__in=query_ids,
            user=request.user
        ).delete()

    if deleted_count == 0:
        return Response(
            {'error': 'No matching query history found or you do not have permission to delete these queries'},
            status=status.HTTP_404_NOT_FOUND
        )

    return Response({
        'deleted': deleted_count,
        'message': f'Successfully deleted {deleted_count} quer{"y" if deleted_count == 1 else "ies"}'
    }, status=status.HTTP_200_OK)
