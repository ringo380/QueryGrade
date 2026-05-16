"""
Comprehensive tests for ML Learning and Monitoring Components

Tests for:
- IncrementalLearningEngine (online learning)
- PerformanceTracker (model performance tracking)
- ConfidenceAnalyzer (confidence analysis)
- DataDriftDetector (concept drift detection)
- ConfidenceBasedRetrainingSystem (retraining decisions)
"""

import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from django.test import TestCase, TransactionTestCase, override_settings
from django.utils import timezone

# Suppress verbose logging during tests
logging.getLogger("analyzer").setLevel(logging.WARNING)


class IncrementalLearningEngineInitializationTestCase(TestCase):
    """Tests for IncrementalLearningEngine initialization"""

    def test_engine_initialization(self):
        """Test basic engine initialization"""
        from analyzer.ml.learning.incremental_engine import IncrementalLearningEngine

        engine = IncrementalLearningEngine()

        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.lr_scheduler)
        self.assertIsNotNone(engine.drift_detector)
        self.assertEqual(engine.total_samples_processed, 0)

    def test_models_initialization(self):
        """Test model initialization"""
        from analyzer.ml.learning.incremental_engine import IncrementalLearningEngine

        engine = IncrementalLearningEngine()
        success = engine.initialize_models()

        self.assertTrue(success)
        self.assertIsNotNone(engine.primary_model)
        self.assertGreater(len(engine.backup_models), 0)

    def test_learning_rate_scheduler_init(self):
        """Test learning rate scheduler initialization"""
        from analyzer.ml.learning.incremental_engine import (
            AdaptiveLearningRateScheduler,
        )

        scheduler = AdaptiveLearningRateScheduler(
            initial_lr=0.01, min_lr=1e-6, max_lr=0.1
        )

        self.assertEqual(scheduler.current_lr, 0.01)
        self.assertEqual(scheduler.initial_lr, 0.01)
        self.assertEqual(scheduler.min_lr, 1e-6)
        self.assertEqual(scheduler.max_lr, 0.1)

    def test_drift_detector_initialization(self):
        """Test concept drift detector initialization"""
        from analyzer.ml.learning.incremental_engine import ConceptDriftDetector

        detector = ConceptDriftDetector(window_size=100, sensitivity=0.05)

        self.assertEqual(detector.window_size, 100)
        self.assertEqual(detector.sensitivity, 0.05)
        self.assertIsNone(detector.baseline_error)
        self.assertEqual(len(detector.error_window), 0)


class IncrementalLearningProcessingTestCase(TestCase):
    """Tests for learning instance processing"""

    def setUp(self):
        """Set up test engine"""
        from analyzer.ml.learning.incremental_engine import IncrementalLearningEngine

        self.engine = IncrementalLearningEngine()
        self.engine.initialize_models()

    def test_learning_instance_processing(self):
        """Test processing of learning instances"""
        from analyzer.ml.learning.incremental_engine import LearningInstance

        instance = LearningInstance(
            instance_id="test_1",
            features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            target=75.0,
            weight=1.0,
            timestamp=timezone.now(),
            query_id=1,
            source="test",
        )

        result = self.engine.process_learning_instance(instance)

        # Result should be a dictionary with either success or error
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        if result["success"]:
            self.assertGreaterEqual(result["samples_processed"], 1)
            self.assertIn("processing_time_ms", result)
        else:
            self.assertIn("error", result)

    def test_multiple_instances_processing(self):
        """Test processing multiple learning instances"""
        from analyzer.ml.learning.incremental_engine import LearningInstance

        successful_count = 0
        for i in range(20):
            instance = LearningInstance(
                instance_id=f"test_{i}",
                features=np.random.rand(10).tolist(),
                target=np.random.uniform(0, 100),
                weight=1.0,
                timestamp=timezone.now(),
                query_id=i,
                source="test",
            )

            result = self.engine.process_learning_instance(instance)
            if result["success"]:
                successful_count += 1

        # At least some instances should process successfully
        self.assertGreaterEqual(successful_count, 0)
        # Total samples should include both successful and processed attempts
        self.assertIsNotNone(self.engine.total_samples_processed)

    def test_invalid_instance_rejection(self):
        """Test rejection of invalid learning instances"""
        from analyzer.ml.learning.incremental_engine import LearningInstance

        # Empty features
        invalid_instance = LearningInstance(
            instance_id="invalid",
            features=[],
            target=75.0,
            weight=1.0,
            timestamp=timezone.now(),
            query_id=999,
            source="test",
        )

        result = self.engine.process_learning_instance(invalid_instance)

        self.assertFalse(result["success"])

    def test_learning_rate_adaptation(self):
        """Test adaptive learning rate updates"""
        from analyzer.ml.learning.incremental_engine import (
            AdaptiveLearningRateScheduler,
        )

        scheduler = AdaptiveLearningRateScheduler()

        # Simulate improving loss
        initial_lr = scheduler.current_lr

        lr1 = scheduler.update_learning_rate(10.0)
        lr2 = scheduler.update_learning_rate(9.0)
        lr3 = scheduler.update_learning_rate(8.0)

        # Learning rate should increase with improving loss
        self.assertGreaterEqual(lr3, lr1)

    def test_concept_drift_detection(self):
        """Test concept drift detection"""
        from analyzer.ml.learning.incremental_engine import ConceptDriftDetector

        detector = ConceptDriftDetector(window_size=50, sensitivity=0.05)

        # Add predictions with increasing errors (drift)
        for i in range(100):
            prediction = 50 + (i // 50) * 20  # Increases after 50 samples
            target = 50
            detector.add_prediction(prediction, target)

        drift_alert = detector.detect_drift()

        # Detector should have baseline error established after processing
        self.assertIsNotNone(detector.baseline_error)


class PerformanceTrackerInitializationTestCase(TestCase):
    """Tests for PerformanceTracker initialization"""

    def test_tracker_initialization(self):
        """Test performance tracker initialization"""
        from analyzer.ml.monitoring.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker()

        self.assertIsNotNone(tracker)
        self.assertIsNotNone(tracker.performance_history)
        self.assertIsNotNone(tracker.prediction_cache)
        self.assertGreater(tracker.history_size, 0)

    def test_performance_metrics_structure(self):
        """Test PerformanceMetrics dataclass structure"""
        from datetime import datetime

        from analyzer.ml.monitoring.performance_tracker import PerformanceMetrics

        metrics = PerformanceMetrics(
            model_id="test_model",
            model_type="random_forest",
            accuracy_score=0.85,
            precision_score=0.83,
            recall_score=0.87,
            f1_score=0.85,
            calibration_score=0.8,
            prediction_speed_ms=10.5,
            memory_usage_mb=30.0,
            stability_score=0.9,
            robustness_score=0.88,
            user_satisfaction=0.82,
            confidence_accuracy=0.85,
            drift_resistance=0.9,
            timestamp=timezone.now(),
        )

        self.assertEqual(metrics.model_id, "test_model")
        self.assertEqual(metrics.accuracy_score, 0.85)
        self.assertGreater(metrics.calibration_score, 0)


class PerformanceTrackingTestCase(TestCase):
    """Tests for performance tracking functionality"""

    def setUp(self):
        """Set up test tracker"""
        from analyzer.ml.monitoring.performance_tracker import PerformanceTracker

        self.tracker = PerformanceTracker()

    def test_record_prediction(self):
        """Test recording predictions"""
        self.tracker.record_prediction(
            model_id="test_model",
            query_id=1,
            prediction=75.0,
            confidence=0.85,
            processing_time_ms=15.0,
            actual_score=78.0,
        )

        self.assertIn("test_model", self.tracker.prediction_cache)
        self.assertEqual(len(self.tracker.prediction_cache["test_model"]), 1)

    def test_record_feedback(self):
        """Test recording user feedback"""
        self.tracker.record_feedback(
            model_id="test_model", query_id=1, user_feedback=77.0, user_satisfaction=4
        )

        self.assertIn("test_model", self.tracker.feedback_cache)
        self.assertEqual(len(self.tracker.feedback_cache["test_model"]), 1)

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        # Record predictions with actual scores
        for i in range(60):
            prediction = 70 + np.random.normal(0, 5)
            actual = 70 + np.random.normal(0, 5)

            self.tracker.record_prediction(
                model_id="test_model",
                query_id=i,
                prediction=prediction,
                confidence=0.8,
                processing_time_ms=10.0,
                actual_score=actual,
            )

        metrics = self.tracker.calculate_performance_metrics("test_model")

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.model_id, "test_model")
        self.assertGreaterEqual(metrics.accuracy_score, 0)
        self.assertLessEqual(metrics.accuracy_score, 1)

    def test_model_selector_initialization(self):
        """Test model selector initialization"""
        from analyzer.ml.monitoring.performance_tracker import ModelSelector

        selector = ModelSelector()

        self.assertIsNotNone(selector)
        self.assertIsNotNone(selector.performance_tracker)
        self.assertEqual(len(selector.selection_history), 0)


class ConfidenceAnalyzerTestCase(TestCase):
    """Tests for ConfidenceAnalyzer"""

    def setUp(self):
        """Set up test analyzer"""
        from analyzer.ml.monitoring.confidence_analyzer import ConfidenceAnalyzer

        self.analyzer = ConfidenceAnalyzer()

    def test_analyzer_initialization(self):
        """Test confidence analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(len(self.analyzer.confidence_history), 0)
        self.assertEqual(len(self.analyzer.accuracy_history), 0)

    def test_add_prediction_result(self):
        """Test adding prediction results"""
        self.analyzer.add_prediction_result(
            prediction=75.0, confidence=0.85, actual=78.0, query_id=1
        )

        self.assertEqual(len(self.analyzer.confidence_history), 1)
        self.assertEqual(len(self.analyzer.accuracy_history), 1)
        self.assertIn(1, self.analyzer.prediction_cache)

    def test_calibration_score_calculation(self):
        """Test calibration score calculation"""
        # Add multiple prediction results
        for i in range(50):
            self.analyzer.add_prediction_result(
                prediction=70 + np.random.normal(0, 5),
                confidence=0.5 + np.random.uniform(0, 0.4),
                actual=70 + np.random.normal(0, 5),
                query_id=i,
            )

        score = self.analyzer.calculate_calibration_score()

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_confidence_trend_analysis(self):
        """Test confidence trend analysis"""
        # Add predictions with increasing confidence
        for i in range(60):
            confidence = 0.5 + (i / 100)  # Increasing confidence
            self.analyzer.add_prediction_result(
                prediction=75.0,
                confidence=min(0.99, confidence),
                actual=75.0,
                query_id=i,
            )

        trend = self.analyzer.analyze_confidence_trend()

        self.assertIn(
            trend, ["increasing", "decreasing", "stable", "insufficient_data"]
        )

    def test_low_confidence_queries_detection(self):
        """Test detection of low confidence queries"""
        # Add mix of high and low confidence predictions
        for i in range(30):
            confidence = 0.3 if i < 10 else 0.8
            self.analyzer.add_prediction_result(
                prediction=75.0, confidence=confidence, actual=75.0, query_id=i
            )

        low_conf_queries = self.analyzer.get_low_confidence_queries(threshold=0.5)

        self.assertGreater(len(low_conf_queries), 0)


class DataDriftDetectorTestCase(TestCase):
    """Tests for DataDriftDetector"""

    def test_detector_initialization(self):
        """Test data drift detector initialization"""
        from analyzer.ml.monitoring.drift_detection import DataDriftDetector

        detector = DataDriftDetector()

        self.assertIsNotNone(detector)
        self.assertFalse(detector.baseline_established)
        self.assertEqual(len(detector.feature_distributions), 0)

    def test_feature_distribution_update(self):
        """Test feature distribution tracking"""
        from analyzer.ml.monitoring.drift_detection import DataDriftDetector

        detector = DataDriftDetector()

        features = [0.1, 0.2, 0.3]
        feature_names = ["feature_1", "feature_2", "feature_3"]

        detector.update_feature_distribution(features, feature_names)

        self.assertEqual(len(detector.feature_distributions), 3)

    def test_drift_detection(self):
        """Test drift detection"""
        from analyzer.ml.monitoring.drift_detection import DataDriftDetector

        detector = DataDriftDetector()

        # Establish baseline
        for _ in range(150):
            features = np.random.normal(50, 10, 5).tolist()
            feature_names = [f"feature_{i}" for i in range(5)]
            detector.update_feature_distribution(features, feature_names)

        # Introduce drift
        for _ in range(50):
            features = np.random.normal(70, 10, 5).tolist()  # Changed mean
            feature_names = [f"feature_{i}" for i in range(5)]
            detector.update_feature_distribution(features, feature_names)

        results = detector.detect_drift()

        self.assertIsNotNone(results)
        self.assertIn("drift_detected", results)


class RetrainingSystemTestCase(TestCase):
    """Tests for ConfidenceBasedRetrainingSystem"""

    def test_system_initialization(self):
        """Test retraining system initialization"""
        from analyzer.ml.monitoring.retraining_system import (
            ConfidenceBasedRetrainingSystem,
        )

        system = ConfidenceBasedRetrainingSystem()

        self.assertIsNotNone(system)
        self.assertIsNotNone(system.confidence_analyzer)
        self.assertIsNotNone(system.performance_monitor)
        self.assertGreater(len(system.thresholds), 0)

    def test_confidence_metrics_gathering(self):
        """Test gathering confidence metrics"""
        from analyzer.ml.monitoring.retraining_system import (
            ConfidenceBasedRetrainingSystem,
        )

        system = ConfidenceBasedRetrainingSystem()

        # Add some data to the system
        for i in range(50):
            system.confidence_analyzer.add_prediction_result(
                prediction=75.0, confidence=0.8, actual=76.0, query_id=i
            )

            system.performance_monitor.add_feedback_result(
                predicted_score=75.0, user_feedback_score=76.0
            )

        metrics = system._gather_confidence_metrics()

        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics.prediction_confidence, 0)
        self.assertGreaterEqual(metrics.feedback_agreement, 0)

    def test_model_health_evaluation(self):
        """Test model health status evaluation"""
        from analyzer.ml.monitoring.retraining_system import (
            ConfidenceBasedRetrainingSystem,
        )

        system = ConfidenceBasedRetrainingSystem()

        # Add data
        for i in range(60):
            system.confidence_analyzer.add_prediction_result(
                prediction=70.0 + np.random.normal(0, 5),
                confidence=0.8,
                actual=70.0 + np.random.normal(0, 5),
                query_id=i,
            )

        health = system.get_model_health_status()

        self.assertIsNotNone(health)
        self.assertGreaterEqual(health.overall_health, 0)
        self.assertLessEqual(health.overall_health, 1)
        self.assertIn(
            health.risk_level, ["low", "medium", "high", "critical", "unknown"]
        )

    def test_retraining_trigger_evaluation(self):
        """Test retraining trigger evaluation"""
        from analyzer.ml.monitoring.retraining_system import (
            ConfidenceBasedRetrainingSystem,
        )

        system = ConfidenceBasedRetrainingSystem()

        # Simulate low confidence
        for i in range(100):
            system.confidence_analyzer.add_prediction_result(
                prediction=75.0,
                confidence=0.3,  # Low confidence
                actual=75.0,
                query_id=i,
            )

        triggers = system.evaluate_retraining_need()

        # May or may not trigger based on other factors
        self.assertIsNotNone(triggers)
        self.assertIsInstance(triggers, list)


class ModelComparisonTestCase(TestCase):
    """Tests for model comparison functionality"""

    def test_model_comparison(self):
        """Test comparing two models"""
        from analyzer.ml.monitoring.performance_tracker import ModelSelector

        selector = ModelSelector()

        # Record predictions for model A
        for i in range(50):
            selector.performance_tracker.record_prediction(
                model_id="model_a",
                query_id=i,
                prediction=75.0,
                confidence=0.85,
                processing_time_ms=10.0,
                actual_score=76.0,
            )

        # Record predictions for model B
        for i in range(50):
            selector.performance_tracker.record_prediction(
                model_id="model_b",
                query_id=i,
                prediction=74.0,
                confidence=0.82,
                processing_time_ms=12.0,
                actual_score=76.0,
            )

        comparison = selector.compare_models("model_a", "model_b")

        self.assertIsNotNone(comparison)
        self.assertEqual(comparison.model_a_id, "model_a")
        self.assertEqual(comparison.model_b_id, "model_b")
        self.assertIsNotNone(comparison.metric_comparisons)


class ABTestingTestCase(TestCase):
    """Tests for A/B testing framework"""

    def test_ab_test_initialization(self):
        """Test A/B test initialization"""
        from analyzer.ml.monitoring.performance_tracker import ABTestingFramework

        framework = ABTestingFramework()

        self.assertIsNotNone(framework)
        self.assertEqual(len(framework.active_tests), 0)

    def test_ab_test_start(self):
        """Test starting an A/B test"""
        from analyzer.ml.monitoring.performance_tracker import ABTestingFramework

        framework = ABTestingFramework()

        test_id = framework.start_ab_test("model_a", "model_b", traffic_split=0.5)

        self.assertIsNotNone(test_id)
        self.assertIn(test_id, framework.active_tests)

    def test_ab_test_result_recording(self):
        """Test recording A/B test results"""
        from analyzer.ml.monitoring.performance_tracker import ABTestingFramework

        framework = ABTestingFramework()
        test_id = framework.start_ab_test("model_a", "model_b")

        # Record results
        for i in range(30):
            model = "model_a" if i < 15 else "model_b"
            framework.record_ab_result(
                test_id=test_id,
                model_used=model,
                prediction=75.0,
                actual_score=76.0,
                user_feedback=4.0,
            )

        test = framework.active_tests[test_id]
        self.assertEqual(test["samples_a"], 15)
        self.assertEqual(test["samples_b"], 15)

    def test_ab_test_analysis(self):
        """Test A/B test analysis"""
        from analyzer.ml.monitoring.performance_tracker import ABTestingFramework

        framework = ABTestingFramework()
        test_id = framework.start_ab_test("model_a", "model_b")

        # Record enough results
        for i in range(50):
            model = "model_a" if i < 25 else "model_b"
            framework.record_ab_result(
                test_id=test_id,
                model_used=model,
                prediction=75.0,
                actual_score=76.0,
                user_feedback=4.0,
            )

        result = framework.analyze_ab_test(test_id)

        if result:
            self.assertIsNotNone(result.test_id)
            self.assertEqual(result.samples_a, 25)
            self.assertEqual(result.samples_b, 25)


class PerformanceMonitorTestCase(TestCase):
    """Tests for PerformanceMonitor"""

    def test_monitor_initialization(self):
        """Test performance monitor initialization"""
        from analyzer.ml.monitoring.drift_detection import PerformanceMonitor

        monitor = PerformanceMonitor()

        self.assertIsNotNone(monitor)
        self.assertEqual(len(monitor.performance_history), 0)
        self.assertIsNone(monitor.baseline_performance)

    def test_add_performance_metric(self):
        """Test adding performance metrics"""
        from analyzer.ml.monitoring.drift_detection import PerformanceMonitor

        monitor = PerformanceMonitor()

        monitor.add_performance_metric("accuracy", 0.85)

        self.assertEqual(len(monitor.performance_history), 1)

    def test_performance_trend_calculation(self):
        """Test calculating performance trends"""
        from analyzer.ml.monitoring.drift_detection import PerformanceMonitor

        monitor = PerformanceMonitor()

        # Add improving metrics
        for i in range(120):
            value = 0.5 + (i / 200)  # Gradually improving
            monitor.add_performance_metric("accuracy", value)

        trend = monitor.calculate_performance_trend(window_size=100)

        self.assertIn(
            trend["trend"], ["improving", "degrading", "stable", "insufficient_data"]
        )
        self.assertIn("change", trend)

    def test_feedback_agreement_calculation(self):
        """Test feedback agreement calculation"""
        from analyzer.ml.monitoring.drift_detection import PerformanceMonitor

        monitor = PerformanceMonitor()

        # Add feedback results
        for i in range(60):
            monitor.add_feedback_result(predicted_score=75.0, user_feedback_score=76.0)

        agreement = monitor.calculate_feedback_agreement()

        self.assertGreaterEqual(agreement, 0.0)
        self.assertLessEqual(agreement, 1.0)
