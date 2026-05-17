"""
Manual rollback to a previous ML model version, with safety guards.

Issue #5 explicitly scopes this as **operator-initiated**, not automatic —
auto-rollback graduates only after the false-positive rate is proven <5%
over a 30-day window. For now: the dashboard offers a button per active
model, this helper validates + executes the swap.

Guards (in order — first failure short-circuits):

  1. Target model must be currently ACTIVE.
  2. A prior version must exist (same model_type, lower version,
     status DEPRECATED). If never had a prior ACTIVE version, refuse.
  3. No other rollback for the same model_type within the last 24 h
     (prevents thrash if the prior model is also bad).

On success: atomic swap. Target → DEPRECATED, prior → ACTIVE.
Audit row created as MLAlert(alert_type='ROLLBACK_PERFORMED',
severity='HIGH', status='RESOLVED') attached to the rolled-back model.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone

from analyzer.models import MLAlert, MLModel

logger = logging.getLogger(__name__)

ROLLBACK_COOLDOWN = timedelta(hours=24)


class RollbackError(Exception):
    """Raised when a rollback request fails a guard."""


def _previous_deprecated(model: MLModel) -> Optional[MLModel]:
    """The most recent DEPRECATED model of the same type, prior to `model`."""
    return (
        MLModel.objects.filter(model_type=model.model_type, status="DEPRECATED")
        .exclude(pk=model.pk)
        .order_by("-deployed_at", "-created_at")
        .first()
    )


def _recent_rollback_exists(model_type: str) -> bool:
    cutoff = timezone.now() - ROLLBACK_COOLDOWN
    return MLAlert.objects.filter(
        alert_type="ROLLBACK_PERFORMED",
        model__model_type=model_type,
        created_at__gte=cutoff,
    ).exists()


def perform_rollback(target: MLModel, performed_by: Optional[User] = None) -> MLAlert:
    """Roll `target` back to its previous DEPRECATED version of the same type.

    Returns the audit MLAlert row. Raises RollbackError on any guard failure.
    """
    if target.status != "ACTIVE":
        raise RollbackError(
            f"Cannot roll back: target model {target} is not ACTIVE "
            f"(status={target.status})."
        )

    prior = _previous_deprecated(target)
    if prior is None:
        raise RollbackError(
            f"Cannot roll back: no prior DEPRECATED {target.model_type} model exists."
        )

    if _recent_rollback_exists(target.model_type):
        raise RollbackError(
            f"Cannot roll back: another rollback for model_type "
            f"{target.model_type} happened in the last 24 h. "
            f"Manual intervention required."
        )

    now = timezone.now()
    resolution = (
        f"Manual rollback: {target.name} v{target.version} → "
        f"{prior.name} v{prior.version}"
        + (f" by {performed_by.username}" if performed_by else "")
        + f" at {now.isoformat()}"
    )

    with transaction.atomic():
        target.status = "DEPRECATED"
        target.save(update_fields=["status"])

        prior.status = "ACTIVE"
        prior.deployed_at = now
        prior.save(update_fields=["status", "deployed_at"])

        audit = MLAlert.objects.create(
            model=target,
            alert_type="ROLLBACK_PERFORMED",
            severity="HIGH",
            status="RESOLVED",
            message=f"Rolled back to {prior.name} v{prior.version}.",
            resolution=resolution,
            acknowledged_by=performed_by,
            acknowledged_at=now,
            payload={
                "rolled_back_to": {
                    "id": prior.id,
                    "name": prior.name,
                    "version": prior.version,
                },
                "rolled_back_from": {
                    "id": target.id,
                    "name": target.name,
                    "version": target.version,
                },
            },
        )

    logger.warning(
        "Rollback performed: %s v%s → %s v%s (by %s)",
        target.name,
        target.version,
        prior.name,
        prior.version,
        performed_by.username if performed_by else "system",
    )
    return audit


def can_rollback(target: MLModel) -> bool:
    """Cheap pre-check used by the dashboard to decide whether to render
    the rollback button. Mirrors perform_rollback's guards without
    mutating any state."""
    if target.status != "ACTIVE":
        return False
    if _previous_deprecated(target) is None:
        return False
    if _recent_rollback_exists(target.model_type):
        return False
    return True
