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

DEDUPE_WINDOW = timedelta(hours=1)


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


def _recent_open_alert_exists(model: MLModel, alert_type: str) -> bool:
    cutoff = timezone.now() - DEDUPE_WINDOW
    return MLAlert.objects.filter(
        model=model,
        alert_type=alert_type,
        status="OPEN",
        created_at__gte=cutoff,
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

    if _recent_open_alert_exists(model, alert_type):
        logger.debug(
            "Skipping duplicate alert: model=%s type=%s (dedupe window %s)",
            model.pk,
            alert_type,
            DEDUPE_WINDOW,
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
    """Run a full monitoring evaluation, persist new alerts, return them.

    Returns (created_alerts, skipped_count). The skipped count tracks how
    many triggers were deduped against a recent open alert — useful for
    monitoring task logging without leaking warnings on every healthy run.
    """
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

    logger.info(
        "Evaluation complete: triggers=%d, created=%d, skipped=%d, target_model=%s",
        len(triggers),
        len(created),
        skipped,
        target,
    )
    return created, skipped
