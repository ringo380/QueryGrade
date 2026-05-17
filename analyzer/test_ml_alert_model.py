"""
Tests for the MLAlert model.

Covers the persisted shape only — the alert-creation pipeline
(monitor_models task) and the ack/dismiss views ship in later PRs
in this stack and have their own test files.
"""

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from analyzer.models import MLAlert, MLModel


class MLAlertModelTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            username="opstest", email="ops@example.com", password="opspass123"
        )
        cls.model = MLModel.objects.create(
            name="HybridScorer",
            model_type="HYBRID_SCORER",
            version="1.0.0",
            status="ACTIVE",
            file_path="/tmp/dummy.joblib",
            checksum="0" * 64,
        )

    def test_creates_with_defaults(self):
        alert = MLAlert.objects.create(
            model=self.model,
            alert_type="DRIFT",
            severity="HIGH",
            message="Mean error doubled vs baseline.",
            payload={"baseline_error": 0.1, "current_error": 0.22},
        )
        self.assertEqual(alert.status, "OPEN")
        self.assertTrue(alert.is_open)
        self.assertFalse(alert.is_terminal)
        self.assertIsNotNone(alert.created_at)
        self.assertIsNone(alert.acknowledged_at)
        self.assertIsNone(alert.acknowledged_by)
        self.assertEqual(alert.payload["current_error"], 0.22)

    def test_str_format(self):
        alert = MLAlert.objects.create(
            model=self.model,
            alert_type="PERFORMANCE",
            severity="CRITICAL",
            message="Accuracy collapsed.",
        )
        s = str(alert)
        self.assertIn("CRITICAL", s)
        self.assertIn("Performance degradation", s)
        self.assertIn("OPEN", s)

    def test_acknowledge_flow(self):
        alert = MLAlert.objects.create(
            model=self.model,
            alert_type="DATA_DRIFT",
            severity="MEDIUM",
            message="Feature distribution shifted.",
        )
        alert.status = "ACKNOWLEDGED"
        alert.acknowledged_at = timezone.now()
        alert.acknowledged_by = self.user
        alert.save()

        alert.refresh_from_db()
        self.assertFalse(alert.is_open)
        self.assertFalse(alert.is_terminal)
        self.assertEqual(alert.acknowledged_by, self.user)

    def test_terminal_states(self):
        for terminal_status in ("RESOLVED", "FALSE_POSITIVE"):
            alert = MLAlert.objects.create(
                model=self.model,
                alert_type="CONFIDENCE",
                severity="LOW",
                message="Below confidence threshold.",
                status=terminal_status,
            )
            self.assertTrue(alert.is_terminal)
            self.assertFalse(alert.is_open)

    def test_related_name_on_mlmodel(self):
        MLAlert.objects.create(
            model=self.model,
            alert_type="DRIFT",
            severity="LOW",
            message="One.",
        )
        MLAlert.objects.create(
            model=self.model,
            alert_type="PERFORMANCE",
            severity="HIGH",
            message="Two.",
        )
        # related_name='alerts' on the FK
        self.assertEqual(self.model.alerts.count(), 2)

    def test_cascade_delete(self):
        MLAlert.objects.create(
            model=self.model,
            alert_type="DRIFT",
            severity="LOW",
            message="Will be cascaded.",
        )
        model_id = self.model.id
        self.model.delete()
        self.assertEqual(MLAlert.objects.filter(model_id=model_id).count(), 0)

    def test_ordering_descending_by_created_at(self):
        first = MLAlert.objects.create(
            model=self.model,
            alert_type="DRIFT",
            severity="LOW",
            message="first",
        )
        second = MLAlert.objects.create(
            model=self.model,
            alert_type="DRIFT",
            severity="LOW",
            message="second",
        )
        most_recent = MLAlert.objects.first()
        self.assertEqual(most_recent.id, second.id)
        self.assertNotEqual(most_recent.id, first.id)

    def test_payload_defaults_to_empty_dict(self):
        alert = MLAlert.objects.create(
            model=self.model,
            alert_type="TIME_BASED",
            severity="LOW",
            message="Time-based retrain due.",
        )
        self.assertEqual(alert.payload, {})

    def test_acknowledged_by_set_null_when_user_deleted(self):
        alert = MLAlert.objects.create(
            model=self.model,
            alert_type="DRIFT",
            severity="HIGH",
            message="ack-test",
            acknowledged_by=self.user,
            acknowledged_at=timezone.now(),
            status="ACKNOWLEDGED",
        )
        self.user.delete()
        alert.refresh_from_db()
        self.assertIsNone(alert.acknowledged_by)
        # status remains; only the FK is cleared
        self.assertEqual(alert.status, "ACKNOWLEDGED")
