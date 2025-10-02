"""
Custom error view handlers for QueryGrade application.
Provides user-friendly 404 and 500 error pages with helpful navigation.
"""
from django.shortcuts import render


def custom_404(request, exception=None):
    """
    Custom 404 error handler with helpful navigation and suggestions.

    Args:
        request: The HTTP request object
        exception: The exception that triggered the 404 (optional)

    Returns:
        Rendered 404 error page with 404 status code
    """
    return render(request, '404.html', status=404)


def custom_500(request):
    """
    Custom 500 error handler with troubleshooting guidance.

    Args:
        request: The HTTP request object

    Returns:
        Rendered 500 error page with 500 status code
    """
    return render(request, '500.html', status=500)
