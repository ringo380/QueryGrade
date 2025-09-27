"""
Custom template filters for enhanced XSS protection and secure output rendering.
"""

from django import template
from django.utils.html import escape, strip_tags
from django.utils.safestring import mark_safe
import html
import re

register = template.Library()


@register.filter
def secure_html_escape(value):
    """
    Enhanced HTML escaping filter that provides additional XSS protection
    beyond Django's default escaping.
    """
    if value is None:
        return ""

    # Convert to string if not already
    value = str(value)

    # Strip any existing HTML tags
    value = strip_tags(value)

    # Escape HTML characters
    value = html.escape(value, quote=True)

    # Additional escaping for potential XSS vectors
    xss_patterns = {
        'javascript:': '&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;&#58;',
        'vbscript:': '&#118;&#98;&#115;&#99;&#114;&#105;&#112;&#116;&#58;',
        'data:': '&#100;&#97;&#116;&#97;&#58;',
        'onload': '&#111;&#110;&#108;&#111;&#97;&#100;',
        'onerror': '&#111;&#110;&#101;&#114;&#114;&#111;&#114;',
        'onclick': '&#111;&#110;&#99;&#108;&#105;&#99;&#107;',
    }

    for pattern, replacement in xss_patterns.items():
        value = re.sub(re.escape(pattern), replacement, value, flags=re.IGNORECASE)

    return value


@register.filter
def sanitize_sql_output(value):
    """
    Sanitize SQL query output for safe display in templates.
    Prevents SQL injection in displayed results.
    """
    if value is None:
        return ""

    value = str(value)

    # Remove potentially dangerous SQL keywords when displaying
    dangerous_sql = [
        'EXEC', 'EXECUTE', 'SP_EXECUTESQL', 'XP_CMDSHELL',
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE'
    ]

    for keyword in dangerous_sql:
        # Replace with escaped version for display
        value = re.sub(
            rf'\b{re.escape(keyword)}\b',
            f'[{keyword}]',
            value,
            flags=re.IGNORECASE
        )

    # Escape HTML
    value = html.escape(value, quote=True)

    return value


@register.filter
def secure_url(value):
    """
    Validate and sanitize URLs to prevent XSS through href attributes.
    """
    if not value:
        return ""

    value = str(value).strip()

    # Allow only safe URL schemes
    safe_schemes = ['http', 'https', 'mailto', 'tel']

    # Check if URL has a scheme
    if '://' in value:
        scheme = value.split('://')[0].lower()
        if scheme not in safe_schemes:
            return "#"  # Replace with safe placeholder
    elif value.startswith('//'):
        # Protocol-relative URLs - make them https
        value = 'https:' + value
    elif not value.startswith('/') and not value.startswith('#'):
        # Relative URL without leading slash - make it relative
        value = '/' + value

    # Remove javascript: and other dangerous schemes
    dangerous_schemes = ['javascript:', 'vbscript:', 'data:', 'file:']
    for scheme in dangerous_schemes:
        if value.lower().startswith(scheme):
            return "#"

    # HTML escape the URL
    return html.escape(value, quote=True)


@register.filter
def strip_script_tags(value):
    """
    Remove all script tags and their content from text.
    """
    if not value:
        return ""

    value = str(value)

    # Remove script tags and their content
    value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.DOTALL | re.IGNORECASE)

    # Remove any remaining script-related attributes
    value = re.sub(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', '', value, flags=re.IGNORECASE)

    return value


@register.filter
def filename_safe(value):
    """
    Make a filename safe for display by removing potentially dangerous characters.
    """
    if not value:
        return ""

    value = str(value)

    # Remove path traversal attempts
    value = value.replace('..', '').replace('/', '').replace('\\', '')

    # Remove dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
    for char in dangerous_chars:
        value = value.replace(char, '_')

    # Limit length
    if len(value) > 255:
        value = value[:255]

    # HTML escape
    return html.escape(value, quote=True)


@register.filter
def safe_json(value):
    """
    Safely output JSON data in templates to prevent XSS.
    """
    import json

    if value is None:
        return "{}"

    try:
        # Ensure it's properly serialized JSON
        json_str = json.dumps(value, ensure_ascii=True, separators=(',', ':'))

        # Additional escaping for HTML context
        json_str = json_str.replace('<', '\\u003c')
        json_str = json_str.replace('>', '\\u003e')
        json_str = json_str.replace('&', '\\u0026')

        return mark_safe(json_str)
    except (TypeError, ValueError):
        return "{}"