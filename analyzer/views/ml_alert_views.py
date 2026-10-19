"""
Views for ML alert triage and manual rollback (issue #5).

POST-only mutating endpoints, staff-only. GET /ml/alerts/ renders the
triage list with filters + rolling FPR widget.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from analyzer.ml.monitoring.rollback import (
    RollbackError,
    can_rollback,
    perform_rollback,
)
from analyzer.models import MLAlert, MLModel

logger = logging.getLogger(__name__)


FPR_WINDOW = timedelta(days=7)


def _compute_fpr(window: timedelta = FPR_WINDOW) -> dict:
    """Rolling false-positive rate across the resolved population.

    Returns {fpr_pct, fp_count, resolved_count, total_terminal, window_days}.
    `fpr_pct` is None when the resolved population is empty (avoid divide by
    zero and avoid implying a meaningful rate from no data).
    """
    cutoff = timezone.now() - window
    counts = MLAlert.objects.filter(
        created_at__gte=cutoff,
        status__in=("RESOLVED", "FALSE_POSITIVE"),
    ).aggregate(
        fp=Count("id", filter=Q(status="FALSE_POSITIVE")),
        resolved=Count("id", filter=Q(status="RESOLVED")),
    )
    fp = counts["fp"] or 0
    resolved = counts["resolved"] or 0
    total = fp + resolved
    fpr_pct = (100.0 * fp / total) if total else None
    return {
        "fpr_pct": fpr_pct,
        "fp_count": fp,
        "resolved_count": resolved,
        "total_terminal": total,
        "window_days": window.days,
    }


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


@login_required
@user_passes_test(_is_staff_or_superuser)
def ml_alerts_list(request):
    """Paginated triage list with severity/status filters + FPR widget.

    Query params:
      - status: comma-separated filter (default: OPEN,ACKNOWLEDGED)
      - severity: comma-separated filter (default: all)
      - model: model id filter
      - page: pagination cursor
    """
    qs = MLAlert.objects.select_related("model", "acknowledged_by")

    raw_status = request.GET.get("status", "OPEN,ACKNOWLEDGED")
    status_filter = [s for s in (s.strip() for s in raw_status.split(",")) if s]
    if status_filter and "all" not in status_filter:
        qs = qs.filter(status__in=status_filter)

    raw_severity = request.GET.get("severity", "")
    severity_filter = [s for s in (s.strip() for s in raw_severity.split(",")) if s]
    if severity_filter:
        qs = qs.filter(severity__in=severity_filter)

    model_filter = request.GET.get("model")
    if model_filter:
        try:
            qs = qs.filter(model_id=int(model_filter))
        except (TypeError, ValueError):
            pass

    paginator = Paginator(qs, 25)
    page_obj = paginator.get_page(request.GET.get("page"))

    # Decorate each alert with whether the underlying model is rollback-eligible
    # (the dashboard renders a rollback button per row only when this is True).
    page_alerts = list(page_obj.object_list)
    seen_models = {}
    for alert in page_alerts:
        model = alert.model
        if model.pk not in seen_models:
            seen_models[model.pk] = can_rollback(model)
        alert.model_can_rollback = seen_models[model.pk]

    open_count = MLAlert.objects.filter(status="OPEN").count()
    active_models = MLModel.objects.filter(status="ACTIVE")

    context = {
        "page_obj": page_obj,
        "alerts": page_alerts,
        "open_count": open_count,
        "fpr": _compute_fpr(),
        "active_models": active_models,
        "status_choices": MLAlert.STATUS_CHOICES,
        "severity_choices": MLAlert.SEVERITY_CHOICES,
        "current_status": raw_status,
        "current_severity": raw_severity,
        "current_model": model_filter or "",
    }
    return render(request, "analyzer/ml_alerts.html", context)
