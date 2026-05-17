"""
Tests for the /ml/alerts/ dashboard panel + FPR widget (issue #5, PR 5).
"""

from datetime import timedelta

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from analyzer.models import MLAlert, MLModel


class FPRComputationTests(TestCase):
    """Direct exercise of _compute_fpr()."""

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

    def _alert(self, status="OPEN", days_ago=0):
        a = MLAlert.objects.create(
            model=self.model,
            alert_type="PERFORMANCE",
            severity="HIGH",
            status=status,
            message="test",
        )
        if days_ago > 0:
            MLAlert.objects.filter(pk=a.pk).update(
                created_at=timezone.now() - timedelta(days=days_ago)
            )
        return a

    def test_empty_dataset_returns_none_pct(self):
        from analyzer.views.ml_alert_views import _compute_fpr

        result = _compute_fpr()
        self.assertIsNone(result["fpr_pct"])
        self.assertEqual(result["fp_count"], 0)
        self.assertEqual(result["resolved_count"], 0)
        self.assertEqual(result["window_days"], 7)

    def test_only_resolved_returns_zero_pct(self):
        from analyzer.views.ml_alert_views import _compute_fpr

        self._alert(status="RESOLVED")
        self._alert(status="RESOLVED")
        result = _compute_fpr()
        self.assertEqual(result["fpr_pct"], 0.0)
        self.assertEqual(result["resolved_count"], 2)

    def test_mixed_yields_correct_pct(self):
        from analyzer.views.ml_alert_views import _compute_fpr

        # 1 FP + 9 RESOLVED → 10% FPR
        self._alert(status="FALSE_POSITIVE")
        for _ in range(9):
            self._alert(status="RESOLVED")
        result = _compute_fpr()
        self.assertEqual(result["fp_count"], 1)
        self.assertEqual(result["resolved_count"], 9)
        self.assertEqual(result["total_terminal"], 10)
        self.assertAlmostEqual(result["fpr_pct"], 10.0)

    def test_open_alerts_excluded(self):
        from analyzer.views.ml_alert_views import _compute_fpr

        self._alert(status="OPEN")
        self._alert(status="ACKNOWLEDGED")
        self._alert(status="FALSE_POSITIVE")
        self._alert(status="RESOLVED")
        result = _compute_fpr()
        # Only FALSE_POSITIVE + RESOLVED counted
        self.assertEqual(result["total_terminal"], 2)
        self.assertEqual(result["fpr_pct"], 50.0)

    def test_outside_window_excluded(self):
        from analyzer.views.ml_alert_views import _compute_fpr

        self._alert(status="FALSE_POSITIVE", days_ago=10)  # outside 7d
        self._alert(status="RESOLVED")  # in window
        result = _compute_fpr()
        self.assertEqual(result["fp_count"], 0)
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["fpr_pct"], 0.0)


class AlertsListViewTests(TestCase):
    """The /ml/alerts/ GET endpoint."""

    @classmethod
    def setUpTestData(cls):
        cls.staff = User.objects.create_user(
            username="ops", password="opspass123", is_staff=True
        )
        cls.regular = User.objects.create_user(
            username="rando", password="randopass123"
        )
        cls.model_a = MLModel.objects.create(
            name="HybridScorer",
            model_type="HYBRID_SCORER",
            version="2.0.0",
            status="ACTIVE",
            file_path="/tmp/a.joblib",
            checksum="0" * 64,
        )
        cls.model_b = MLModel.objects.create(
            name="FeatureExtractor",
            model_type="FEATURE_EXTRACTOR",
            version="1.0.0",
            status="ACTIVE",
            file_path="/tmp/b.joblib",
            checksum="1" * 64,
        )

    def setUp(self):
        self.client = Client()
        # Seed a representative mix of alerts
        self.open_critical = MLAlert.objects.create(
            model=self.model_a,
            alert_type="PERFORMANCE",
            severity="CRITICAL",
            status="OPEN",
            message="Accuracy collapsed.",
        )
        self.open_high = MLAlert.objects.create(
            model=self.model_b,
            alert_type="DATA_DRIFT",
            severity="HIGH",
            status="OPEN",
            message="KS-stat spiked.",
        )
        self.ack = MLAlert.objects.create(
            model=self.model_a,
            alert_type="CONFIDENCE",
            severity="MEDIUM",
            status="ACKNOWLEDGED",
            message="Confidence low.",
        )
        self.resolved = MLAlert.objects.create(
            model=self.model_a,
            alert_type="TIME_BASED",
            severity="LOW",
            status="RESOLVED",
            message="Routine retrain hit.",
        )
        self.fp = MLAlert.objects.create(
            model=self.model_b,
            alert_type="DATA_DRIFT",
            severity="LOW",
            status="FALSE_POSITIVE",
            message="Detector noise.",
        )

    def test_requires_login(self):
        response = self.client.get(reverse("ml_alerts"))
        self.assertEqual(response.status_code, 302)

    def test_requires_staff(self):
        self.client.login(username="rando", password="randopass123")
        response = self.client.get(reverse("ml_alerts"))
        self.assertEqual(response.status_code, 302)  # to LOGIN_URL

    def test_renders_for_staff(self):
        self.client.login(username="ops", password="opspass123")
        response = self.client.get(reverse("ml_alerts"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "ML alerts")
        # FPR widget rendered (1 FP + 1 RESOLVED → 50%)
        self.assertContains(response, "50.0%")

    def test_default_filter_shows_open_and_acked(self):
        self.client.login(username="ops", password="opspass123")
        response = self.client.get(reverse("ml_alerts"))
        self.assertContains(response, "Accuracy collapsed.")
        self.assertContains(response, "KS-stat spiked.")
        self.assertContains(response, "Confidence low.")
        # Resolved + FP not in default view
        self.assertNotContains(response, "Routine retrain hit.")
        self.assertNotContains(response, "Detector noise.")

    def test_all_filter_shows_terminal_too(self):
        self.client.login(username="ops", password="opspass123")
        response = self.client.get(reverse("ml_alerts") + "?status=all")
        self.assertContains(response, "Routine retrain hit.")
        self.assertContains(response, "Detector noise.")

    def test_severity_filter(self):
        self.client.login(username="ops", password="opspass123")
        response = self.client.get(reverse("ml_alerts") + "?severity=CRITICAL")
        self.assertContains(response, "Accuracy collapsed.")
        self.assertNotContains(response, "KS-stat spiked.")

    def test_model_filter(self):
        self.client.login(username="ops", password="opspass123")
        response = self.client.get(reverse("ml_alerts") + f"?model={self.model_b.id}")
        self.assertContains(response, "KS-stat spiked.")
        self.assertNotContains(response, "Accuracy collapsed.")

    def test_open_count_in_header(self):
        self.client.login(username="ops", password="opspass123")
        response = self.client.get(reverse("ml_alerts"))
        # 2 OPEN seeded
        self.assertContains(response, "2")
        self.assertContains(response, "Open")

    def test_action_buttons_render_for_open_alerts(self):
        self.client.login(username="ops", password="opspass123")
        response = self.client.get(reverse("ml_alerts"))
        # Ack/Resolve/FP buttons for OPEN
        self.assertContains(response, "Ack")
        self.assertContains(response, "Resolve")
        # POST URLs present
        self.assertContains(
            response, reverse("ml_alert_ack", args=[self.open_critical.id])
        )
        self.assertContains(
            response, reverse("ml_alert_dismiss", args=[self.open_critical.id])
        )
        self.assertContains(
            response, reverse("ml_alert_false_positive", args=[self.open_critical.id])
        )

    def test_pagination_kicks_in_above_25(self):
        # Create 30 more open alerts to push past page size 25
        for i in range(30):
            MLAlert.objects.create(
                model=self.model_a,
                alert_type="PERFORMANCE",
                severity="LOW",
                status="OPEN",
                message=f"bulk-{i}",
            )
        self.client.login(username="ops", password="opspass123")
        response = self.client.get(reverse("ml_alerts"))
        self.assertContains(response, "Page 1 of")
        self.assertContains(response, "Next →")
