"""
Glue between the existing ML detectors and the MLAlert model.

The existing `ConfidenceBasedRetrainingSystem.evaluate_retraining_need()`
returns a list of `RetrainingTrigger` dataclass instances. This module
translates them into persisted `MLAlert` rows, deduped against any open
alert of the same (model, alert_type) raised within the last hour.

Keeping the translation in one place means the management command, the
Celery task, and any future re-use (e.g., a manual "run evaluation now"
button) all see the same alert semantics.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from typing import List, Optional, Tuple

from django.utils import timezone

from analyzer.ml.monitoring.retraining_system import (
    ConfidenceBasedRetrainingSystem,
    RetrainingTrigger,
    TriggerReason,
    TriggerUrgency,
)
from analyzer.models import MLAlert, MLModel

logger = logging.getLogger(__name__)


REASON_TO_ALERT_TYPE = {
    TriggerReason.LOW_CONFIDENCE: "CONFIDENCE",
    TriggerReason.PERFORMANCE_DEGRADATION: "PERFORMANCE",
    TriggerReason.DATA_DRIFT: "DATA_DRIFT",
    TriggerReason.FEEDBACK_DIVERGENCE: "USER_AGREEMENT",
    TriggerReason.TIME_BASED: "TIME_BASED",
    TriggerReason.EMERGENCY: "PERFORMANCE",
    TriggerReason.MANUAL_OVERRIDE: "TIME_BASED",
}

URGENCY_TO_SEVERITY = {
    TriggerUrgency.LOW: "LOW",
    TriggerUrgency.MEDIUM: "MEDIUM",
    TriggerUrgency.HIGH: "HIGH",
    TriggerUrgency.CRITICAL: "CRITICAL",
}

UNRESOLVED_STATUSES = ("OPEN", "ACKNOWLEDGED")
# After an alert is RESOLVED/FALSE_POSITIVE, wait this long before re-alerting
# the same (model, alert_type) — so resolving a persistent-condition alert
# doesn't respawn it on the very next monitoring tick.
RESOLVE_COOLDOWN = timedelta(hours=6)


def _serialize_evidence(evidence: dict) -> dict:
    """RetrainingTrigger.evidence can contain dataclass values — coerce to JSON-safe."""
    safe = {}
    for key, value in (evidence or {}).items():
        if is_dataclass(value):
            safe[key] = asdict(value)
        elif isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _active_target_model() -> Optional[MLModel]:
    """The model these alerts get attached to.

    Per CLAUDE.md, the production grading path uses an ACTIVE HYBRID_SCORER.
    If none is active yet (fresh deploy), fall back to the most recently
    deployed ACTIVE model of any type so alerts still attach to something
    real.
    """
    target = (
        MLModel.objects.filter(model_type="HYBRID_SCORER", status="ACTIVE")
        .order_by("-deployed_at", "-created_at")
        .first()
    )
    if target is None:
        target = (
            MLModel.objects.filter(status="ACTIVE")
            .order_by("-deployed_at", "-created_at")
            .first()
        )
    return target


def _should_suppress_alert(model: MLModel, alert_type: str) -> bool:
    """Whether to skip creating a new alert for this (model, alert_type).

    Caps pile-up at one live alert per type: an unresolved alert (OPEN or
    ACKNOWLEDGED) suppresses new ones **regardless of age** — it already
    represents the live condition until an operator triages it. After the
    alert is RESOLVED/FALSE_POSITIVE, a RESOLVE_COOLDOWN keeps a persistent
    condition from immediately respawning on the next tick (uses
    `acknowledged_at`, set on terminal transition by the triage views and the
    bulk-resolve op).
    """
    if MLAlert.objects.filter(
        model=model, alert_type=alert_type, status__in=UNRESOLVED_STATUSES
    ).exists():
        return True

    cutoff = timezone.now() - RESOLVE_COOLDOWN
    return MLAlert.objects.filter(
        model=model,
        alert_type=alert_type,
        status__in=("RESOLVED", "FALSE_POSITIVE"),
        acknowledged_at__gte=cutoff,
    ).exists()


def trigger_to_alert(trigger: RetrainingTrigger, model: MLModel) -> Optional[MLAlert]:
    """Persist a single RetrainingTrigger as an MLAlert. Returns the new alert
    or None if a recent open alert already covers the same model+type."""
    alert_type = REASON_TO_ALERT_TYPE.get(trigger.reason)
    severity = URGENCY_TO_SEVERITY.get(trigger.urgency, "MEDIUM")

    if alert_type is None:
        logger.warning(
            "Unmapped trigger reason %s — skipping alert creation", trigger.reason
        )
        return None

    if _should_suppress_alert(model, alert_type):
        logger.debug(
            "Skipping duplicate alert: model=%s type=%s (unresolved exists or within resolve cooldown)",
            model.pk,
            alert_type,
        )
        return None

    payload = {
        "trigger_id": trigger.trigger_id,
        "reason": trigger.reason.value,
        "urgency": trigger.urgency.name,
        "confidence_score": trigger.confidence_score,
        "estimated_improvement": trigger.estimated_improvement,
        "evidence": _serialize_evidence(trigger.evidence),
        "cost_estimate": trigger.cost_estimate or {},
    }

    return MLAlert.objects.create(
        model=model,
        alert_type=alert_type,
        severity=severity,
        message=trigger.recommendation[:500],
        payload=payload,
    )


def run_evaluation() -> Tuple[List[MLAlert], int]:
    """Run a full monitoring evaluation, persist new alerts, send emails,
    return the new alerts.

    Returns (created_alerts, skipped_count). The skipped count tracks how
    many triggers were deduped against a recent open alert — useful for
    monitoring task logging without leaking warnings on every healthy run.

    Email delivery is best-effort: the MLAlert row is persisted first, then
    the notifier is called. Mail failures are logged but don't fail the
    evaluation — the alert is still visible in the dashboard / admin.
    """
    # Import here so settings/email backend init happens at call time, not
    # at module import (matters for test isolation + the worker startup path).
    from analyzer.ml.monitoring.alert_notifier import send_alert_email

    target = _active_target_model()
    if target is None:
        logger.info("No ACTIVE MLModel found; skipping evaluation.")
        return [], 0

    system = ConfidenceBasedRetrainingSystem()
    triggers = system.evaluate_retraining_need()

    created: List[MLAlert] = []
    skipped = 0
    for trigger in triggers:
        alert = trigger_to_alert(trigger, target)
        if alert is None:
            skipped += 1
        else:
            created.append(alert)
            try:
                send_alert_email(alert)
            except Exception as exc:
                logger.exception(
                    "send_alert_email failed for alert id=%s: %s", alert.pk, exc
                )

    logger.info(
        "Evaluation complete: triggers=%d, created=%d, skipped=%d, target_model=%s",
        len(triggers),
        len(created),
        skipped,
        target,
    )
    return created, skipped
