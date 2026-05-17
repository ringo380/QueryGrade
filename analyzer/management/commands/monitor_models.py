"""
Periodic ML monitoring entrypoint.

Runs all existing detectors (concept drift, data drift, performance
degradation, low confidence, time-based) against the currently-ACTIVE
HYBRID_SCORER and persists each surfaced trigger as an MLAlert row.

Intended to be invoked every ~15 min via Celery Beat (see
`querygrade/celery.py:beat_schedule`). Safe to run by hand at any
time — the alert_evaluator dedupes against open alerts in the same
(model, alert_type) raised in the last hour, so manual runs alongside
the scheduled task won't double up.
"""

from django.core.management.base import BaseCommand

from analyzer.ml.monitoring.alert_evaluator import run_evaluation


class Command(BaseCommand):
    help = (
        "Run ML monitoring detectors and persist any surfaced triggers as MLAlert rows."
    )

    def handle(self, *args, **options):
        created, skipped = run_evaluation()

        self.stdout.write(
            self.style.SUCCESS(
                f"Monitoring run complete: created {len(created)} alert(s), "
                f"skipped {skipped} duplicate(s)."
            )
        )
        for alert in created:
            self.stdout.write(
                f"  [{alert.severity}] {alert.get_alert_type_display()} — {alert.message[:80]}"
            )
