"""
Tests for ML alert ack/dismiss/false-positive views and the
manual-rollback endpoint (issue #5, PR 4).

Auth: all 4 endpoints require login + staff/superuser. Tests cover
both the success paths and the safety guards on rollback.
"""

from datetime import timedelta

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from analyzer.models import MLAlert, MLModel


class AckDismissFalsePositiveTests(TestCase):
    """Coverage for the three triage endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.staff = User.objects.create_user(
            username="ops", password="opspass123", is_staff=True
        )
        cls.regular = User.objects.create_user(
            username="rando", password="randopass123"
        )
        cls.model = MLModel.objects.create(
            name="HybridScorer",
            model_type="HYBRID_SCORER",
            version="2.0.0",
            status="ACTIVE",
            file_path="/tmp/dummy.joblib",
            checksum="0" * 64,
        )

    def setUp(self):
        self.client = Client()
        self.alert = MLAlert.objects.create(
            model=self.model,
            alert_type="PERFORMANCE",
            severity="HIGH",
            message="Accuracy dropped.",
        )

    # ---- ack ----

    def test_ack_requires_login(self):
        url = reverse("ml_alert_ack", args=[self.alert.pk])
        response = self.client.post(url)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith("/login/"))

    def test_ack_requires_staff(self):
        self.client.login(username="rando", password="randopass123")
        url = reverse("ml_alert_ack", args=[self.alert.pk])
        response = self.client.post(url)
        # @user_passes_test redirects to LOGIN_URL on failure
        self.assertEqual(response.status_code, 302)
        # Status unchanged
        self.alert.refresh_from_db()
        self.assertEqual(self.alert.status, "OPEN")

    def test_ack_get_not_allowed(self):
        self.client.login(username="ops", password="opspass123")
        response = self.client.get(reverse("ml_alert_ack", args=[self.alert.pk]))
        self.assertEqual(response.status_code, 405)

    def test_ack_open_alert_transitions_to_acknowledged(self):
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_alert_ack", args=[self.alert.pk])
        response = self.client.post(url)
        self.assertEqual(response.status_code, 302)
        self.alert.refresh_from_db()
        self.assertEqual(self.alert.status, "ACKNOWLEDGED")
        self.assertEqual(self.alert.acknowledged_by, self.staff)
        self.assertIsNotNone(self.alert.acknowledged_at)

    def test_ack_idempotent_for_already_acknowledged(self):
        self.alert.status = "ACKNOWLEDGED"
        self.alert.acknowledged_by = self.staff
        self.alert.acknowledged_at = timezone.now()
        self.alert.save()
        original_at = self.alert.acknowledged_at

        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_alert_ack", args=[self.alert.pk])
        self.client.post(url)
        self.alert.refresh_from_db()
        # Status unchanged, ack_at unchanged
        self.assertEqual(self.alert.status, "ACKNOWLEDGED")
        self.assertEqual(self.alert.acknowledged_at, original_at)

    # ---- dismiss / resolve ----

    def test_dismiss_resolves_with_optional_resolution(self):
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_alert_dismiss", args=[self.alert.pk])
        response = self.client.post(url, {"resolution": "Hot-patched the index."})
        self.assertEqual(response.status_code, 302)
        self.alert.refresh_from_db()
        self.assertEqual(self.alert.status, "RESOLVED")
        self.assertEqual(self.alert.resolution, "Hot-patched the index.")
        self.assertEqual(self.alert.acknowledged_by, self.staff)

    def test_dismiss_skips_already_terminal(self):
        self.alert.status = "FALSE_POSITIVE"
        self.alert.save()
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_alert_dismiss", args=[self.alert.pk])
        self.client.post(url)
        self.alert.refresh_from_db()
        self.assertEqual(self.alert.status, "FALSE_POSITIVE")  # unchanged

    # ---- false positive ----

    def test_mark_false_positive(self):
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_alert_false_positive", args=[self.alert.pk])
        response = self.client.post(url, {"resolution": "Detector noise."})
        self.assertEqual(response.status_code, 302)
        self.alert.refresh_from_db()
        self.assertEqual(self.alert.status, "FALSE_POSITIVE")
        self.assertEqual(self.alert.resolution, "Detector noise.")

    def test_false_positive_skips_already_terminal(self):
        self.alert.status = "RESOLVED"
        self.alert.save()
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_alert_false_positive", args=[self.alert.pk])
        self.client.post(url)
        self.alert.refresh_from_db()
        self.assertEqual(self.alert.status, "RESOLVED")  # unchanged

    def test_endpoint_404_for_missing_alert(self):
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_alert_ack", args=[999_999])
        response = self.client.post(url)
        self.assertEqual(response.status_code, 404)


class RollbackTests(TestCase):
    """Coverage for the manual rollback endpoint + the safety guards in
    `analyzer.ml.monitoring.rollback.perform_rollback`."""

    @classmethod
    def setUpTestData(cls):
        cls.staff = User.objects.create_user(
            username="ops", password="opspass123", is_staff=True
        )

    def setUp(self):
        self.client = Client()
        # Prior DEPRECATED version
        self.prior = MLModel.objects.create(
            name="HybridScorer",
            model_type="HYBRID_SCORER",
            version="1.0.0",
            status="DEPRECATED",
            file_path="/tmp/dummy-prior.joblib",
            checksum="1" * 64,
            deployed_at=timezone.now() - timedelta(days=30),
        )
        # Current ACTIVE version
        self.current = MLModel.objects.create(
            name="HybridScorer",
            model_type="HYBRID_SCORER",
            version="2.0.0",
            status="ACTIVE",
            file_path="/tmp/dummy-current.joblib",
            checksum="2" * 64,
            deployed_at=timezone.now() - timedelta(days=2),
        )

    def test_requires_staff(self):
        self.client.login(
            username=User.objects.create_user(username="r", password="p").username,
            password="p",
        )
        url = reverse("ml_model_rollback", args=[self.current.pk])
        response = self.client.post(url)
        self.assertEqual(response.status_code, 302)
        self.current.refresh_from_db()
        self.assertEqual(self.current.status, "ACTIVE")  # untouched

    def test_get_not_allowed(self):
        self.client.login(username="ops", password="opspass123")
        response = self.client.get(reverse("ml_model_rollback", args=[self.current.pk]))
        self.assertEqual(response.status_code, 405)

    def test_happy_path_swaps_statuses_and_creates_audit(self):
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_model_rollback", args=[self.current.pk])
        response = self.client.post(url)
        self.assertEqual(response.status_code, 302)

        self.current.refresh_from_db()
        self.prior.refresh_from_db()
        self.assertEqual(self.current.status, "DEPRECATED")
        self.assertEqual(self.prior.status, "ACTIVE")

        audit = MLAlert.objects.filter(alert_type="ROLLBACK_PERFORMED").first()
        self.assertIsNotNone(audit)
        self.assertEqual(audit.model, self.current)
        self.assertEqual(audit.status, "RESOLVED")
        self.assertEqual(audit.severity, "HIGH")
        self.assertEqual(audit.acknowledged_by, self.staff)
        self.assertIn("v1.0.0", audit.message)
        self.assertEqual(audit.payload["rolled_back_to"]["version"], "1.0.0")
        self.assertEqual(audit.payload["rolled_back_from"]["version"], "2.0.0")

    def test_refuses_when_target_not_active(self):
        self.current.status = "DEPRECATED"
        self.current.save()
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_model_rollback", args=[self.current.pk])
        self.client.post(url)
        # No audit row was created
        self.assertFalse(
            MLAlert.objects.filter(alert_type="ROLLBACK_PERFORMED").exists()
        )

    def test_refuses_when_no_prior_version(self):
        self.prior.delete()
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_model_rollback", args=[self.current.pk])
        self.client.post(url)
        self.current.refresh_from_db()
        self.assertEqual(self.current.status, "ACTIVE")  # unchanged
        self.assertFalse(
            MLAlert.objects.filter(alert_type="ROLLBACK_PERFORMED").exists()
        )

    def test_refuses_when_recent_rollback_exists(self):
        # Pre-existing rollback audit within cool-down window
        MLAlert.objects.create(
            model=self.current,
            alert_type="ROLLBACK_PERFORMED",
            severity="HIGH",
            status="RESOLVED",
            message="prior rollback",
        )
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_model_rollback", args=[self.current.pk])
        self.client.post(url)
        self.current.refresh_from_db()
        self.assertEqual(self.current.status, "ACTIVE")  # cool-down blocked

    def test_cooldown_expires_after_24h(self):
        recent = MLAlert.objects.create(
            model=self.current,
            alert_type="ROLLBACK_PERFORMED",
            severity="HIGH",
            status="RESOLVED",
            message="prior rollback",
        )
        # Push it past the 24h cool-down
        MLAlert.objects.filter(pk=recent.pk).update(
            created_at=timezone.now() - timedelta(hours=25)
        )
        self.client.login(username="ops", password="opspass123")
        url = reverse("ml_model_rollback", args=[self.current.pk])
        self.client.post(url)
        self.current.refresh_from_db()
        self.prior.refresh_from_db()
        self.assertEqual(self.current.status, "DEPRECATED")
        self.assertEqual(self.prior.status, "ACTIVE")

    def test_can_rollback_helper(self):
        from analyzer.ml.monitoring.rollback import can_rollback

        self.assertTrue(can_rollback(self.current))

        # No prior
        self.prior.delete()
        self.assertFalse(can_rollback(self.current))
