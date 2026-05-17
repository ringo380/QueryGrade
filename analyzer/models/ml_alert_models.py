"""
ML monitoring alerts.

MLAlert persists signals raised by the periodic monitoring task
(see analyzer/management/commands/monitor_models.py, added in a
later PR in this stack). Each alert points at the MLModel the
signal concerns and captures enough payload to reconstruct the
detector reading for audit and false-positive review.

The model is intentionally simple — heavy detector state stays in
the detector classes; this table only stores what's needed for the
ack/dismiss/false-positive flow and the rolling FPR widget.
"""

from django.contrib.auth.models import User
from django.db import models

from .ml_models import MLModel


class MLAlert(models.Model):
    """A monitoring signal raised against an active ML model."""

    SEVERITY_CHOICES = [
        ("LOW", "Low"),
        ("MEDIUM", "Medium"),
        ("HIGH", "High"),
        ("CRITICAL", "Critical"),
    ]

    STATUS_CHOICES = [
        ("OPEN", "Open"),
        ("ACKNOWLEDGED", "Acknowledged"),
        ("RESOLVED", "Resolved"),
        ("FALSE_POSITIVE", "False positive"),
    ]

    ALERT_TYPE_CHOICES = [
        ("DRIFT", "Concept drift"),
        ("DATA_DRIFT", "Data drift"),
        ("PERFORMANCE", "Performance degradation"),
        ("CONFIDENCE", "Low confidence"),
        ("USER_AGREEMENT", "User-agreement drop"),
        ("TIME_BASED", "Time-based retrain"),
        ("ROLLBACK_PERFORMED", "Rollback performed"),
    ]

    model = models.ForeignKey(
        MLModel,
        on_delete=models.CASCADE,
        related_name="alerts",
    )
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPE_CHOICES)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="OPEN")

    message = models.CharField(max_length=500)
    payload = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    acknowledged_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    resolution = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["model", "-created_at"]),
            models.Index(fields=["status", "severity"]),
            models.Index(fields=["-created_at"]),
            models.Index(fields=["alert_type", "status"]),
        ]

    def __str__(self):
        return f"[{self.severity}] {self.get_alert_type_display()} on {self.model} ({self.status})"

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN"

    @property
    def is_terminal(self) -> bool:
        return self.status in ("RESOLVED", "FALSE_POSITIVE")
