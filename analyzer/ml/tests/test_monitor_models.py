"""
Tests for the periodic ML monitoring entrypoint.

Covers the alert_evaluator (trigger → MLAlert translation + dedupe)
and the `monitor_models` management command. The underlying
ConfidenceBasedRetrainingSystem is mocked to keep these tests fast
and deterministic; that class has its own coverage elsewhere.
"""

from datetime import timedelta
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from analyzer.ml.monitoring import alert_evaluator
from analyzer.ml.monitoring.retraining_system import (
    RetrainingTrigger,
    TriggerReason,
    TriggerUrgency,
)
from analyzer.models import MLAlert, MLModel


def _make_trigger(
    reason: TriggerReason = TriggerReason.PERFORMANCE_DEGRADATION,
    urgency: TriggerUrgency = TriggerUrgency.HIGH,
    recommendation: str = "Retrain recommended due to performance drop.",
):
    return RetrainingTrigger(
        trigger_id=f"trig_{reason.value}_{urgency.name}",
        reason=reason,
        urgency=urgency,
        confidence_score=0.42,
        evidence={"baseline_accuracy": 0.91, "current_accuracy": 0.74},
        recommendation=recommendation,
        estimated_improvement=0.12,
        cost_estimate={"compute_minutes": 30},
        timestamp=timezone.now(),
    )


class AlertEvaluatorTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.model = MLModel.objects.create(
            name="HybridScorer",
            model_type="HYBRID_SCORER",
            version="1.0.0",
            status="ACTIVE",
            file_path="/tmp/dummy.joblib",
            checksum="0" * 64,
        )

    def test_no_active_model_returns_empty(self):
        # Deactivate the only model
        MLModel.objects.update(status="DEPRECATED")
        with patch.object(
            alert_evaluator.ConfidenceBasedRetrainingSystem,
            "evaluate_retraining_need",
            return_value=[_make_trigger()],
        ):
            created, skipped = alert_evaluator.run_evaluation()
        self.assertEqual(created, [])
        self.assertEqual(skipped, 0)
        self.assertEqual(MLAlert.objects.count(), 0)

    def test_creates_alert_from_trigger(self):
        with patch.object(
            alert_evaluator.ConfidenceBasedRetrainingSystem,
            "evaluate_retraining_need",
            return_value=[_make_trigger()],
        ):
            created, skipped = alert_evaluator.run_evaluation()

        self.assertEqual(len(created), 1)
        self.assertEqual(skipped, 0)
        alert = created[0]
        self.assertEqual(alert.alert_type, "PERFORMANCE")
        self.assertEqual(alert.severity, "HIGH")
        self.assertEqual(alert.model, self.model)
        self.assertIn("Retrain recommended", alert.message)
        self.assertEqual(alert.payload["reason"], "performance_degradation")
        self.assertEqual(alert.payload["urgency"], "HIGH")

    def test_dedupes_within_window(self):
        triggers = [_make_trigger(), _make_trigger()]
        with patch.object(
            alert_evaluator.ConfidenceBasedRetrainingSystem,
            "evaluate_retraining_need",
            return_value=triggers,
        ):
            created, skipped = alert_evaluator.run_evaluation()

        # First trigger creates an alert, second is deduped against it
        self.assertEqual(len(created), 1)
        self.assertEqual(skipped, 1)
        self.assertEqual(MLAlert.objects.count(), 1)

    def test_dedupe_only_applies_to_open_alerts(self):
        # Pre-existing RESOLVED alert of same type shouldn't block a new OPEN one
        MLAlert.objects.create(
            model=self.model,
            alert_type="PERFORMANCE",
            severity="MEDIUM",
            status="RESOLVED",
            message="Previously resolved.",
        )
        with patch.object(
            alert_evaluator.ConfidenceBasedRetrainingSystem,
            "evaluate_retraining_need",
            return_value=[_make_trigger()],
        ):
            created, skipped = alert_evaluator.run_evaluation()
        self.assertEqual(len(created), 1)
        self.assertEqual(skipped, 0)

    def test_dedupe_window_expires_after_an_hour(self):
        # Open alert older than 1h should not block a new one
        stale = MLAlert.objects.create(
            model=self.model,
            alert_type="PERFORMANCE",
            severity="HIGH",
            status="OPEN",
            message="Stale open alert.",
        )
        # Bypass auto_now_add by direct update
        MLAlert.objects.filter(pk=stale.pk).update(
            created_at=timezone.now() - timedelta(hours=2)
        )

        with patch.object(
            alert_evaluator.ConfidenceBasedRetrainingSystem,
            "evaluate_retraining_need",
            return_value=[_make_trigger()],
        ):
            created, skipped = alert_evaluator.run_evaluation()
        self.assertEqual(len(created), 1)
        self.assertEqual(skipped, 0)

    def test_each_reason_maps_to_alert_type(self):
        cases = [
            (TriggerReason.LOW_CONFIDENCE, "CONFIDENCE"),
            (TriggerReason.PERFORMANCE_DEGRADATION, "PERFORMANCE"),
            (TriggerReason.DATA_DRIFT, "DATA_DRIFT"),
            (TriggerReason.FEEDBACK_DIVERGENCE, "USER_AGREEMENT"),
            (TriggerReason.TIME_BASED, "TIME_BASED"),
        ]
        for reason, expected_type in cases:
            MLAlert.objects.all().delete()  # clear dedupe state per case
            with patch.object(
                alert_evaluator.ConfidenceBasedRetrainingSystem,
                "evaluate_retraining_need",
                return_value=[_make_trigger(reason=reason)],
            ):
                created, _ = alert_evaluator.run_evaluation()
            self.assertEqual(len(created), 1, f"reason={reason}")
            self.assertEqual(created[0].alert_type, expected_type, f"reason={reason}")

    def test_urgency_maps_to_severity(self):
        cases = [
            (TriggerUrgency.LOW, "LOW"),
            (TriggerUrgency.MEDIUM, "MEDIUM"),
            (TriggerUrgency.HIGH, "HIGH"),
            (TriggerUrgency.CRITICAL, "CRITICAL"),
        ]
        for urgency, expected_severity in cases:
            MLAlert.objects.all().delete()
            with patch.object(
                alert_evaluator.ConfidenceBasedRetrainingSystem,
                "evaluate_retraining_need",
                return_value=[_make_trigger(urgency=urgency)],
            ):
                created, _ = alert_evaluator.run_evaluation()
            self.assertEqual(created[0].severity, expected_severity)

    def test_recommendation_truncated_to_500_chars(self):
        long_rec = "X" * 800
        with patch.object(
            alert_evaluator.ConfidenceBasedRetrainingSystem,
            "evaluate_retraining_need",
            return_value=[_make_trigger(recommendation=long_rec)],
        ):
            created, _ = alert_evaluator.run_evaluation()
        self.assertEqual(len(created[0].message), 500)


class MonitorModelsCommandTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.model = MLModel.objects.create(
            name="HybridScorer",
            model_type="HYBRID_SCORER",
            version="1.0.0",
            status="ACTIVE",
            file_path="/tmp/dummy.joblib",
            checksum="0" * 64,
        )

    def test_command_runs_clean_with_no_triggers(self):
        out = StringIO()
        with patch.object(
            alert_evaluator.ConfidenceBasedRetrainingSystem,
            "evaluate_retraining_need",
            return_value=[],
        ):
            call_command("monitor_models", stdout=out)
        self.assertIn("created 0 alert(s)", out.getvalue())
        self.assertEqual(MLAlert.objects.count(), 0)

    def test_command_creates_alerts(self):
        out = StringIO()
        triggers = [
            _make_trigger(
                reason=TriggerReason.LOW_CONFIDENCE, urgency=TriggerUrgency.MEDIUM
            ),
            _make_trigger(
                reason=TriggerReason.DATA_DRIFT, urgency=TriggerUrgency.CRITICAL
            ),
        ]
        with patch.object(
            alert_evaluator.ConfidenceBasedRetrainingSystem,
            "evaluate_retraining_need",
            return_value=triggers,
        ):
            call_command("monitor_models", stdout=out)
        self.assertEqual(MLAlert.objects.count(), 2)
        self.assertIn("created 2 alert(s)", out.getvalue())

    def test_celery_task_wrapper(self):
        from analyzer.tasks.monitoring_tasks import monitor_ml_models

        with patch.object(
            alert_evaluator.ConfidenceBasedRetrainingSystem,
            "evaluate_retraining_need",
            return_value=[_make_trigger()],
        ):
            result = monitor_ml_models()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["created"], 1)
        self.assertEqual(result["skipped"], 0)

    def test_celery_task_handles_exception(self):
        from analyzer.tasks.monitoring_tasks import monitor_ml_models

        with patch(
            "analyzer.ml.monitoring.alert_evaluator.run_evaluation",
            side_effect=RuntimeError("boom"),
        ):
            result = monitor_ml_models()
        self.assertEqual(result["status"], "error")
        self.assertIn("boom", result["error"])
