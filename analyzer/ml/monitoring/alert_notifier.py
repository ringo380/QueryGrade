"""
Email delivery for MLAlert rows raised by the monitor_models task.

Single entry point: `send_alert_email(alert)`. Throttled at 4h per
(recipient, model, alert_type) tuple via the default Django cache so a
flapping detector doesn't pager-bomb operators. Throttling is a soft
gate — failures (cache miss, mail backend down) fall through to a sync
send rather than silently dropping the alert.

Templates live in `analyzer/templates/analyzer/emails/ml_alert.{txt,html}`.
"""

from __future__ import annotations

import logging
from typing import List

from django.conf import settings
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string

from analyzer.models import MLAlert

logger = logging.getLogger(__name__)


THROTTLE_SECONDS = 4 * 60 * 60  # 4 hours per (recipient, model, alert_type)

SEVERITY_SUBJECT_PREFIX = {
    "CRITICAL": "[CRITICAL]",
    "HIGH": "[HIGH]",
    "MEDIUM": "[MEDIUM]",
    "LOW": "[LOW]",
}


def _recipients() -> List[str]:
    configured = list(getattr(settings, "ML_ALERT_RECIPIENTS", []) or [])
    if configured:
        return [r.strip() for r in configured if r.strip()]
    fallback = getattr(settings, "DEFAULT_FROM_EMAIL", None)
    return [fallback] if fallback else []


def _throttle_key(recipient: str, alert: MLAlert) -> str:
    return f"ml_alert_throttle:{recipient}:{alert.model_id}:{alert.alert_type}"


def _is_throttled(recipient: str, alert: MLAlert) -> bool:
    return cache.get(_throttle_key(recipient, alert)) is not None


def _mark_throttled(recipient: str, alert: MLAlert) -> None:
    cache.set(_throttle_key(recipient, alert), True, timeout=THROTTLE_SECONDS)


def _alert_url(alert: MLAlert) -> str:
    """Build a best-effort link to the dashboard alerts panel. The panel
    lands in PR 5 of this stack; the URL is stable now so emails sent
    in the meantime point at the right place once it ships."""
    base = getattr(settings, "SITE_URL", "https://querygrade.com").rstrip("/")
    return f"{base}/ml/alerts/{alert.pk}/"


def send_alert_email(alert: MLAlert) -> int:
    """Send `alert` to every configured recipient that isn't currently throttled.

    Returns the number of emails actually sent (0–N). Designed to be called
    from the monitor_models task immediately after MLAlert.save() — the
    sync send keeps the issue-#5 alert-response SLA honest.
    """
    recipients = _recipients()
    if not recipients:
        logger.warning(
            "MLAlert id=%s created but no recipients configured (ML_ALERT_RECIPIENTS empty + no DEFAULT_FROM_EMAIL); skipping email.",
            alert.pk,
        )
        return 0

    sendable = [r for r in recipients if not _is_throttled(r, alert)]
    if not sendable:
        logger.info(
            "All recipients throttled for alert id=%s model=%s type=%s; skipping email.",
            alert.pk,
            alert.model_id,
            alert.alert_type,
        )
        return 0

    subject = (
        f"{SEVERITY_SUBJECT_PREFIX.get(alert.severity, '[ALERT]')} "
        f"{alert.get_alert_type_display()} — {alert.model.name}"
    )

    ctx = {
        "alert": alert,
        "model": alert.model,
        "alert_url": _alert_url(alert),
        "alert_type_display": alert.get_alert_type_display(),
        "severity_display": alert.get_severity_display(),
    }
    text_body = render_to_string("analyzer/emails/ml_alert.txt", ctx)
    html_body = render_to_string("analyzer/emails/ml_alert.html", ctx)

    from_email = getattr(settings, "DEFAULT_FROM_EMAIL", None)
    sent = 0
    for recipient in sendable:
        try:
            msg = EmailMultiAlternatives(
                subject=subject,
                body=text_body,
                from_email=from_email,
                to=[recipient],
            )
            msg.attach_alternative(html_body, "text/html")
            msg.send(fail_silently=False)
            _mark_throttled(recipient, alert)
            sent += 1
        except Exception as exc:
            # One bad recipient shouldn't poison the rest. The MLAlert row
            # is already persisted, so this email failure is recoverable —
            # the dashboard surface still shows the alert.
            logger.exception(
                "Failed to send MLAlert email to %s for alert id=%s: %s",
                recipient,
                alert.pk,
                exc,
            )

    return sent
