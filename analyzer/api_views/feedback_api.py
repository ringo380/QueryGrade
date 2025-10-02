from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
import logging

from ..models import QueryFeedback, UserQueryHistory
from ..serializers import QueryFeedbackSerializer

logger = logging.getLogger(__name__)


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
