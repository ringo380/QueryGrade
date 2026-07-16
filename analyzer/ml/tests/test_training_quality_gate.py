"""The deploy quality gate.

The gate was `validation_accuracy >= performance_threshold` and nothing
else. The synthetic bootstrap cleared it at val=0.712 against a 0.7 bar
while scoring test=0.095, so a model with no predictive power became the
ACTIVE one (issue #92). Validation alone cannot catch that; the held-out
test score is the whole point of holding it out.
"""

from django.test import SimpleTestCase, override_settings

from analyzer.ml.core.training_pipeline import TrainingConfig

# The real numbers from the ACTIVE bootstrap, query_grader 20260521_021306.
BOOTSTRAP_VAL = 0.712
BOOTSTRAP_TEST = 0.095


class QualityGateTests(SimpleTestCase):
    def test_rejects_the_actual_bootstrap_model(self):
        """The regression, in the exact numbers that caused it."""
        config = TrainingConfig()

        ok, reason = config.meets_quality_gate(BOOTSTRAP_VAL, BOOTSTRAP_TEST)

        self.assertFalse(ok)
        self.assertIn("test accuracy", reason)

    def test_old_gate_would_have_accepted_it(self):
        """Guards the test above: if the bootstrap's validation score did not
        actually clear the old bar, rejecting it now would prove nothing."""
        config = TrainingConfig()

        self.assertGreaterEqual(BOOTSTRAP_VAL, config.performance_threshold)

    def test_accepts_a_genuinely_good_model(self):
        config = TrainingConfig()

        ok, reason = config.meets_quality_gate(0.85, 0.82)

        self.assertTrue(ok, reason)

    def test_rejects_low_validation_accuracy(self):
        config = TrainingConfig()

        ok, reason = config.meets_quality_gate(0.5, 0.9)

        self.assertFalse(ok)
        self.assertIn("validation accuracy", reason)

    def test_rejects_a_model_that_does_not_generalize(self):
        """Both scores can clear their own bars while the gap still shows the
        model memorised the training set."""
        config = TrainingConfig(
            performance_threshold=0.7,
            test_performance_threshold=0.7,
            max_validation_test_gap=0.15,
        )

        ok, reason = config.meets_quality_gate(0.95, 0.72)

        self.assertFalse(ok)
        self.assertIn("gap", reason)

    def test_treats_missing_scores_as_failing(self):
        """A model that never reported a test score must not deploy by
        default -- None is not a passing grade."""
        config = TrainingConfig()

        ok, _ = config.meets_quality_gate(0.9, None)

        self.assertFalse(ok)

    @override_settings(
        ML_PERFORMANCE_THRESHOLD=0.9,
        ML_TEST_PERFORMANCE_THRESHOLD=0.9,
        ML_MIN_TRAINING_SAMPLES=123,
    )
    def test_thresholds_come_from_settings(self):
        """These settings existed but were read by nothing, so tuning them
        did nothing at all."""
        config = TrainingConfig()

        self.assertEqual(config.performance_threshold, 0.9)
        self.assertEqual(config.test_performance_threshold, 0.9)
        self.assertEqual(config.min_training_samples, 123)
        # A model that used to pass at 0.85 must now fail.
        ok, _ = config.meets_quality_gate(0.85, 0.85)
        self.assertFalse(ok)
