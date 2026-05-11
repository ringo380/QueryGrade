"""
Utility functions shared across view modules.
"""

from django.conf import settings
from django.http import HttpResponse, JsonResponse
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

    Returns JSON for AJAX/fetch callers and a minimal HTML body otherwise.
    Previously rendered a ``403_csrf.html`` template that doesn't exist in
    this repo, so every CSRF rejection bubbled up as a 500.
    """
    wants_json = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
        or request.path.endswith("/ajax/")
    )
    if wants_json:
        return JsonResponse(
            {
                "status": "csrf_failure",
                "reason": reason or "CSRF verification failed.",
                "message": "Your session expired. Please refresh the page and try again.",
            },
            status=403,
        )
    body = (
        "<!doctype html><meta charset=utf-8><title>403 Forbidden</title>"
        "<h1>403 Forbidden</h1>"
        "<p>CSRF verification failed. Refresh the page and try again.</p>"
    )
    return HttpResponse(body, status=403, content_type="text/html; charset=utf-8")
