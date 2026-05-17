"""
Views for ML alert triage and manual rollback (issue #5).

POST-only mutating endpoints, staff-only. The list view (GET /ml/alerts/)
ships in PR 5 — this PR adds the action endpoints so the dashboard panel
in PR 5 has working buttons from day one.
"""

from __future__ import annotations

import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from analyzer.ml.monitoring.rollback import RollbackError, perform_rollback
from analyzer.models import MLAlert, MLModel

logger = logging.getLogger(__name__)


def _is_staff_or_superuser(user) -> bool:
    return user.is_staff or user.is_superuser


def _redirect_back(request) -> HttpResponseRedirect:
    """Send the operator back where they came from, or the dashboard."""
    referrer = request.META.get("HTTP_REFERER")
    if referrer:
        return HttpResponseRedirect(referrer)
    # ml_alerts route lands in PR 5; until then fall back to the dashboard
    try:
        return redirect("ml_alerts")
    except Exception:
        return redirect("ml_dashboard")


def _set_status(
    alert: MLAlert,
    new_status: str,
    request,
    *,
    resolution: str = "",
) -> None:
    alert.status = new_status
    alert.acknowledged_at = timezone.now()
    alert.acknowledged_by = request.user
    if resolution:
        alert.resolution = resolution
    alert.save(
        update_fields=["status", "acknowledged_at", "acknowledged_by", "resolution"]
    )


@login_required
@user_passes_test(_is_staff_or_superuser)
@require_POST
def ack_alert(request, alert_id: int):
    """Acknowledge an OPEN alert — operator has seen it but it's not yet
    resolved. Idempotent for already-ACKNOWLEDGED alerts."""
    alert = get_object_or_404(MLAlert, pk=alert_id)
    if alert.status == "OPEN":
        _set_status(alert, "ACKNOWLEDGED", request)
        messages.success(request, f"Acknowledged alert #{alert.pk}.")
    else:
        messages.info(
            request, f"Alert #{alert.pk} is already {alert.get_status_display()}."
        )
    return _redirect_back(request)


@login_required
@user_passes_test(_is_staff_or_superuser)
@require_POST
def dismiss_alert(request, alert_id: int):
    """Resolve an alert — operator confirms it's been handled. Captures
    optional `resolution` text from the form for audit."""
    alert = get_object_or_404(MLAlert, pk=alert_id)
    if alert.is_terminal:
        messages.info(request, f"Alert #{alert.pk} already terminal.")
        return _redirect_back(request)
    resolution = (request.POST.get("resolution") or "").strip()
    _set_status(alert, "RESOLVED", request, resolution=resolution)
    messages.success(request, f"Resolved alert #{alert.pk}.")
    return _redirect_back(request)


@login_required
@user_passes_test(_is_staff_or_superuser)
@require_POST
def mark_false_positive(request, alert_id: int):
    """Mark an alert as a false positive. Feeds the rolling FPR widget
    (PR 5) so the issue #5 <5% success metric is measurable."""
    alert = get_object_or_404(MLAlert, pk=alert_id)
    if alert.is_terminal:
        messages.info(request, f"Alert #{alert.pk} already terminal.")
        return _redirect_back(request)
    resolution = (request.POST.get("resolution") or "").strip()
    _set_status(alert, "FALSE_POSITIVE", request, resolution=resolution)
    messages.success(request, f"Marked alert #{alert.pk} as false positive.")
    return _redirect_back(request)


@login_required
@user_passes_test(_is_staff_or_superuser)
@require_POST
def rollback_model(request, model_id: int):
    """Roll a model back to its previous DEPRECATED version. Guards in
    `analyzer.ml.monitoring.rollback.perform_rollback` enforce: target
    must be ACTIVE, prior must exist, no other rollback in last 24 h."""
    target = get_object_or_404(MLModel, pk=model_id)
    try:
        audit = perform_rollback(target, performed_by=request.user)
    except RollbackError as exc:
        logger.warning("Rollback refused for model id=%s: %s", model_id, exc)
        messages.error(request, str(exc))
        return _redirect_back(request)

    messages.success(
        request,
        f"Rolled back {target.name} v{target.version}. Audit alert #{audit.pk} created.",
    )
    return _redirect_back(request)
