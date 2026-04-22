"""
Custom security middleware for enhanced CSRF and XSS protection.
"""

import logging
import re

from django.core.exceptions import SuspiciousOperation
from django.http import HttpResponseForbidden, JsonResponse
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)


class EnhancedSecurityMiddleware(MiddlewareMixin):
    """
    Enhanced security middleware providing additional XSS and CSRF protection
    beyond Django's built-in security features.
    """

    # Common XSS attack patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript\s*:",
        r"vbscript\s*:",
        r"on\w+\s*=",
        r"expression\s*\(",
        r"url\s*\(",
        r"@import",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"<form[^>]*>",
        r"document\.(write|writeln|cookie)",
        r"window\.(location|open)",
        r"eval\s*\(",
        r"setTimeout\s*\(",
        r"setInterval\s*\(",
    ]

    # Suspicious patterns that might indicate attack attempts
    SUSPICIOUS_PATTERNS = [
        r"\.\./",  # Path traversal
        r"union\s+select",  # SQL injection
        r"exec\s*\(",  # Code execution
        r"system\s*\(",  # System commands
        r"<\?php",  # PHP injection
        r"<%.*?%>",  # JSP/ASP injection
        r"\$\{.*?\}",  # Expression language injection
    ]

    def __init__(self, get_response):
        self.get_response = get_response
        super().__init__(get_response)

    def process_request(self, request):
        """
        Process incoming requests for security threats.
        """
        # Skip security checks for certain paths (admin, static files, etc.)
        skip_paths = ["/admin/", "/static/", "/media/", "/__debug__/"]
        if any(request.path.startswith(path) for path in skip_paths):
            return None

        # Check for XSS patterns in GET parameters
        if request.GET:
            for key, value in request.GET.items():
                if self._contains_xss_patterns(value):
                    logger.warning(
                        f"XSS attempt detected in GET parameter '{key}': {value[:100]}"
                    )
                    self._log_security_event(request, "XSS_GET", f"Parameter: {key}")
                    return HttpResponseForbidden("Invalid request parameters detected.")

        # Check for suspicious patterns
        if request.GET:
            for key, value in request.GET.items():
                if self._contains_suspicious_patterns(value):
                    logger.warning(
                        f"Suspicious pattern in GET parameter '{key}': {value[:100]}"
                    )
                    self._log_security_event(
                        request, "SUSPICIOUS_GET", f"Parameter: {key}"
                    )
                    return HttpResponseForbidden("Suspicious request detected.")

        return None

    def process_view(self, request, view_func, view_args, view_kwargs):
        """
        Process view-level security checks.
        """
        # Enhanced CSRF validation for AJAX requests
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            if request.method == "POST" and not self._validate_ajax_csrf(request):
                logger.warning(
                    f"AJAX CSRF validation failed for user: {getattr(request.user, 'username', 'anonymous')}"
                )
                self._log_security_event(
                    request, "CSRF_AJAX", "AJAX request without proper CSRF"
                )
                return JsonResponse({"error": "CSRF validation failed"}, status=403)

        return None

    def process_response(self, request, response):
        """
        Add additional security headers to responses.
        """
        # Add additional XSS protection headers
        response["X-XSS-Protection"] = "1; mode=block"
        response["X-Content-Type-Options"] = "nosniff"
        response["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add CSP nonce for inline scripts/styles if CSP is enabled
        if hasattr(response, "csp_nonce"):
            response[
                "Content-Security-Policy"
            ] += f"; script-src 'nonce-{response.csp_nonce}'"

        # Remove potentially sensitive headers
        headers_to_remove = ["Server", "X-Powered-By"]
        for header in headers_to_remove:
            if header in response:
                del response[header]

        return response

    def _contains_xss_patterns(self, value):
        """
        Check if a value contains XSS attack patterns.
        """
        value_lower = value.lower()
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE | re.DOTALL):
                return True
        return False

    def _contains_suspicious_patterns(self, value):
        """
        Check if a value contains suspicious patterns.
        """
        value_lower = value.lower()
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE | re.DOTALL):
                return True
        return False

    def _validate_ajax_csrf(self, request):
        """
        Enhanced CSRF validation for AJAX requests.
        """
        # Check for CSRF token in headers
        csrf_token = request.META.get("HTTP_X_CSRFTOKEN")
        if not csrf_token:
            # Also check in POST data
            csrf_token = request.POST.get("csrfmiddlewaretoken")

        return bool(csrf_token)

    def _log_security_event(self, request, event_type, details):
        """
        Log security events for monitoring and analysis.
        """
        logger.warning(
            f"Security Event: {event_type} | "
            f"IP: {self._get_client_ip(request)} | "
            f"User: {getattr(request.user, 'username', 'anonymous')} | "
            f"Path: {request.path} | "
            f"Details: {details}"
        )

    def _get_client_ip(self, request):
        """
        Get the client's IP address from the request.
        """
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip = x_forwarded_for.split(",")[0]
        else:
            ip = request.META.get("REMOTE_ADDR")
        return ip


class CSRFFailureMiddleware(MiddlewareMixin):
    """
    Enhanced CSRF failure handling middleware.
    """

    def process_exception(self, request, exception):
        """
        Handle CSRF failures with enhanced logging and response.
        """
        if isinstance(exception, SuspiciousOperation) and "CSRF" in str(exception):
            logger.warning(
                f"CSRF failure: {exception} | "
                f"IP: {self._get_client_ip(request)} | "
                f"User: {getattr(request.user, 'username', 'anonymous')} | "
                f"Path: {request.path} | "
                f"Referer: {request.META.get('HTTP_REFERER', 'None')}"
            )

            # Return appropriate response based on request type
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse(
                    {
                        "error": "CSRF verification failed",
                        "message": "Please refresh the page and try again.",
                    },
                    status=403,
                )

        return None

    def _get_client_ip(self, request):
        """Get the client's IP address from the request."""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip = x_forwarded_for.split(",")[0]
        else:
            ip = request.META.get("REMOTE_ADDR")
        return ip
