from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.utils import timezone


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
