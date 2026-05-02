"""
Utility functions shared across view modules.
"""

from django.conf import settings
from django.shortcuts import render

from .constants import ANON_TRIAL_COUNT_KEY


def anon_trial_state(request):
    """Return (cap, count, remaining) for the anonymous trial system."""
    cap = getattr(settings, "ANON_TRIAL_CAP", 3)
    count = request.session.get(ANON_TRIAL_COUNT_KEY, 0)
    remaining = max(0, cap - count)
    return cap, count, remaining


def get_client_ip(request):
    """
    Get the client's IP address from the request.

    Args:
        request: The HTTP request object.

    Returns:
        str: The client's IP address.
    """
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        ip = x_forwarded_for.split(",")[0]
    else:
        ip = request.META.get("REMOTE_ADDR")
    return ip


def csrf_failure(request, reason=""):
    """
    Custom CSRF failure view.

    Args:
        request: The HTTP request object.
        reason: The reason for the CSRF failure.

    Returns:
        HttpResponse: A rendered template with the CSRF error message.
    """
    return render(request, "403_csrf.html", {"reason": reason}, status=403)
