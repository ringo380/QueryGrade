"""
Tests for ML alert email delivery (issue #5, PR 3).

Uses Django's locmem mail backend (the test runner default) and
DummyCache to keep throttling state isolated per test.
"""

from unittest.mock import patch

from django.core import mail
from django.test import TestCase, override_settings

from analyzer.ml.monitoring.alert_notifier import (
    THROTTLE_SECONDS,
    send_alert_email,
)
from analyzer.models import MLAlert, MLModel


@override_settings(
    CACHES={
        "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
    },
    ML_ALERT_RECIPIENTS=["ops@example.com"],
    DEFAULT_FROM_EMAIL="alerts@querygrade.com",
    SITE_URL="https://querygrade.test",
)
class SendAlertEmailTests(TestCase):
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

    def _make_alert(self, alert_type="PERFORMANCE", severity="HIGH"):
        return MLAlert.objects.create(
            model=self.model,
            alert_type=alert_type,
            severity=severity,
            message="Accuracy dropped from 0.91 to 0.74.",
            payload={
                "evidence": {"baseline": 0.91, "current": 0.74},
                "estimated_improvement": 0.12,
            },
        )

    def setUp(self):
        # Reset locmem mail outbox each test
        mail.outbox = []
        # Clear throttle cache
        from django.core.cache import cache

        cache.clear()

    def test_sends_to_configured_recipient(self):
        alert = self._make_alert()
        sent = send_alert_email(alert)
        self.assertEqual(sent, 1)
        self.assertEqual(len(mail.outbox), 1)

        msg = mail.outbox[0]
        self.assertEqual(msg.to, ["ops@example.com"])
        self.assertEqual(msg.from_email, "alerts@querygrade.com")
        self.assertIn("[HIGH]", msg.subject)
        self.assertIn("Performance degradation", msg.subject)
        self.assertIn("HybridScorer", msg.subject)

    def test_text_body_includes_key_fields(self):
        alert = self._make_alert(severity="CRITICAL")
        send_alert_email(alert)
        body = mail.outbox[0].body
        self.assertIn("Critical", body)
        self.assertIn("HybridScorer v1.0.0", body)
        self.assertIn("Accuracy dropped", body)
        self.assertIn("https://querygrade.test/ml/alerts/", body)
        self.assertIn(str(alert.pk), body)

    def test_html_alternative_attached(self):
        alert = self._make_alert()
        send_alert_email(alert)
        msg = mail.outbox[0]
        self.assertTrue(
            any(content_type == "text/html" for _, content_type in msg.alternatives)
        )

    def test_throttle_blocks_repeat_within_window(self):
        alert = self._make_alert()
        first = send_alert_email(alert)
        second = send_alert_email(alert)
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertEqual(len(mail.outbox), 1)

    def test_throttle_is_per_recipient(self):
        with override_settings(
            ML_ALERT_RECIPIENTS=["a@example.com", "b@example.com"],
        ):
            alert = self._make_alert()
            sent = send_alert_email(alert)
            self.assertEqual(sent, 2)
            # Same alert again — both throttled now
            sent_again = send_alert_email(alert)
            self.assertEqual(sent_again, 0)

    def test_different_alert_type_not_throttled(self):
        a = self._make_alert(alert_type="PERFORMANCE")
        b = self._make_alert(alert_type="DATA_DRIFT")
        send_alert_email(a)
        send_alert_email(b)
        self.assertEqual(len(mail.outbox), 2)

    def test_falls_back_to_default_from_email_when_recipients_empty(self):
        with override_settings(ML_ALERT_RECIPIENTS=[]):
            alert = self._make_alert()
            sent = send_alert_email(alert)
            self.assertEqual(sent, 1)
            self.assertEqual(mail.outbox[0].to, ["alerts@querygrade.com"])

    def test_no_recipients_at_all_skips_send(self):
        with override_settings(ML_ALERT_RECIPIENTS=[], DEFAULT_FROM_EMAIL=""):
            alert = self._make_alert()
            sent = send_alert_email(alert)
            self.assertEqual(sent, 0)
            self.assertEqual(len(mail.outbox), 0)

    def test_individual_recipient_failure_doesnt_block_others(self):
        with override_settings(
            ML_ALERT_RECIPIENTS=["bad@example.com", "ok@example.com"],
        ):
            alert = self._make_alert()
            real_send = mail.EmailMultiAlternatives.send

            def selective_send(self_, *args, **kwargs):
                if "bad@example.com" in self_.to:
                    raise RuntimeError("SMTP rejected")
                return real_send(self_, *args, **kwargs)

            with patch.object(mail.EmailMultiAlternatives, "send", selective_send):
                sent = send_alert_email(alert)
            self.assertEqual(sent, 1)  # only ok@ succeeded

    def test_subject_prefix_per_severity(self):
        cases = {
            "LOW": "[LOW]",
            "MEDIUM": "[MEDIUM]",
            "HIGH": "[HIGH]",
            "CRITICAL": "[CRITICAL]",
        }
        for severity, prefix in cases.items():
            mail.outbox = []
            from django.core.cache import cache

            cache.clear()
            alert = self._make_alert(severity=severity)
            send_alert_email(alert)
            self.assertEqual(len(mail.outbox), 1)
            self.assertTrue(mail.outbox[0].subject.startswith(prefix))


@override_settings(
    CACHES={
        "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
    },
    ML_ALERT_RECIPIENTS=["ops@example.com"],
)
class EvaluatorEmailIntegrationTests(TestCase):
    """run_evaluation() should send mail for each newly-created alert."""

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

    def setUp(self):
        mail.outbox = []
        from django.core.cache import cache

        cache.clear()

    def test_evaluation_sends_one_email_per_new_alert(self):
        from django.utils import timezone as dj_timezone

        from analyzer.ml.monitoring import alert_evaluator
        from analyzer.ml.monitoring.retraining_system import (
            RetrainingTrigger,
            TriggerReason,
            TriggerUrgency,
        )

        triggers = [
            RetrainingTrigger(
                trigger_id="t1",
                reason=TriggerReason.PERFORMANCE_DEGRADATION,
                urgency=TriggerUrgency.HIGH,
                confidence_score=0.4,
                evidence={"baseline": 0.9, "current": 0.7},
                recommendation="Retrain due to perf drop.",
                estimated_improvement=0.15,
                cost_estimate={},
                timestamp=dj_timezone.now(),
            ),
            RetrainingTrigger(
                trigger_id="t2",
                reason=TriggerReason.DATA_DRIFT,
                urgency=TriggerUrgency.CRITICAL,
                confidence_score=0.3,
                evidence={"ks_stat": 0.42},
                recommendation="Feature distribution shifted.",
                estimated_improvement=0.1,
                cost_estimate={},
                timestamp=dj_timezone.now(),
            ),
        ]
        with patch.object(
            alert_evaluator.ConfidenceBasedRetrainingSystem,
            "evaluate_retraining_need",
            return_value=triggers,
        ):
            created, skipped = alert_evaluator.run_evaluation()
        self.assertEqual(len(created), 2)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(mail.outbox), 2)
