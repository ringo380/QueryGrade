"""Tests for the real-feedback retrain gate (issue #92).

The gate exists so the pipeline does not "retrain" on the synthetic bootstrap
rows and ship another non-predictive model. It must count only real feedback,
ignore the synthetic seed no matter how many seed rows exist, and let a forced
run through.
"""

from unittest.mock import patch

from django.test import TestCase, override_settings

from analyzer.ml.core import training_gates
from analyzer.ml.core.training_pipeline import (
    TrainingConfig,
    TrainingPipelineManager,
)
from analyzer.models import Query, TrainingData


def _query(hash_):
    return Query.objects.create(
        sql_text="SELECT 1",
        query_type="SELECT",
        query_hash=hash_,
    )


def _training_row(hash_, source):
    return TrainingData.objects.create(
        query=_query(hash_),
        user_grade_avg=4.0,
        system_grade="B",
        system_score=80.0,
        target_score=75.0,
        validation_source=source,
    )


@override_settings(ML_MIN_REAL_FEEDBACK_SAMPLES=3)
class RealFeedbackGateTests(TestCase):
    def test_synthetic_rows_do_not_count_as_real(self):
        for i in range(10):
            _training_row(f"syn{i}", training_gates.SYNTHETIC_SEED_SOURCE)

        self.assertEqual(training_gates.real_training_sample_count(), 0)
        ok, real, threshold = training_gates.real_feedback_gate()
        self.assertFalse(ok)
        self.assertEqual((real, threshold), (0, 3))

    def test_real_rows_count_and_open_the_gate_at_threshold(self):
        # Two real rows: still below the threshold of 3.
        for i in range(2):
            _training_row(f"real{i}", "user_feedback")
        ok, real, _ = training_gates.real_feedback_gate()
        self.assertFalse(ok)
        self.assertEqual(real, 2)

        # A third real row crosses it.
        _training_row("real2", "user_feedback")
        ok, real, _ = training_gates.real_feedback_gate()
        self.assertTrue(ok)
        self.assertEqual(real, 3)

    def test_real_count_ignores_seed_even_when_mixed(self):
        for i in range(5):
            _training_row(f"syn{i}", training_gates.SYNTHETIC_SEED_SOURCE)
        for i in range(3):
            _training_row(f"real{i}", "user_feedback")
        self.assertEqual(training_gates.real_training_sample_count(), 3)
        self.assertTrue(training_gates.real_feedback_gate()[0])


@override_settings(ML_MIN_REAL_FEEDBACK_SAMPLES=3, ML_MIN_TRAINING_SAMPLES=1)
class TrainingPipelineGateTests(TestCase):
    """The pipeline must refuse synthetic-only data even when the plain
    sample-count gate (ML_MIN_TRAINING_SAMPLES) would pass."""

    def test_pipeline_refuses_when_only_synthetic_data(self):
        for i in range(20):
            _training_row(f"syn{i}", training_gates.SYNTHETIC_SEED_SOURCE)

        manager = TrainingPipelineManager(TrainingConfig())
        result = manager.run_training_pipeline(force_retrain=False)

        self.assertFalse(result.success)
        self.assertIn("real feedback", result.error_message.lower())

    def test_force_retrain_bypasses_the_gate(self):
        # No real feedback, but a forced run must get past the gate (it may
        # still fail later for lack of usable features - it just must not be
        # stopped by the real-feedback gate itself).
        for i in range(20):
            _training_row(f"syn{i}", training_gates.SYNTHETIC_SEED_SOURCE)

        manager = TrainingPipelineManager(TrainingConfig())
        result = manager.run_training_pipeline(force_retrain=True)

        # Whatever the outcome, it is not the real-feedback refusal.
        if not result.success:
            self.assertNotIn("real feedback", result.error_message.lower())


class HybridGraderGateTests(TestCase):
    """The hybrid grader's train_model must also honor the real-feedback gate,
    short-circuiting before it collects any training data."""

    def test_train_model_short_circuits_on_gate(self):
        from analyzer.ml.core.hybrid_grader import HybridQueryGrader

        grader = HybridQueryGrader()
        with patch.object(
            grader.feedback_collector, "get_training_dataset"
        ) as get_data, patch(
            "analyzer.ml.core.training_gates.real_feedback_gate",
            return_value=(False, 0, 25),
        ):
            result = grader.train_model(force_retrain=False)

        self.assertIsNone(result)
        # The gate fired before any data was collected.
        get_data.assert_not_called()

    def test_forced_run_passes_the_gate_to_data_collection(self):
        from analyzer.ml.core.hybrid_grader import HybridQueryGrader

        grader = HybridQueryGrader()
        # Force bypasses the gate; train_model then reaches data collection and
        # stops there (empty dataset), proving the gate did not block it.
        with patch.object(
            grader.feedback_collector, "get_training_dataset", return_value=[]
        ) as get_data:
            result = grader.train_model(force_retrain=True)

        self.assertIsNone(result)
        get_data.assert_called_once()
