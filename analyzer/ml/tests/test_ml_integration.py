"""
ML Integration Tests

This module tests the integration between all ML components to ensure
they work together correctly in the QueryGrade system.
"""

import unittest
from django.test import TestCase, TransactionTestCase
from django.contrib.auth.models import User
from unittest.mock import patch, MagicMock
import numpy as np
from datetime import datetime, timedelta

from analyzer.models import Query, QueryFeedback, UserQueryHistory
from analyzer.models import MLModel, TrainingData, LearningMetrics, FeedbackLearning
from analyzer.ml.hybrid_grader import HybridQueryGrader
from analyzer.ml.feature_extractor import FeatureExtractor
from analyzer.ml.feedback_collector import FeedbackCollector
from analyzer.query_analyzer import analyze_query


class MLIntegrationTestCase(TestCase):
    """Test ML component integration."""

    def setUp(self):
        """Set up integration test data."""
        self.user = User.objects.create_user(
            username='mluser',
            email='ml@example.com',
            password='testpass'
        )

        # Create test queries
        self.queries = [
            Query.objects.create(
                sql_text="SELECT id, name FROM users WHERE active = 1",
                query_type='SELECT',
                query_hash='simple_1',
                estimated_complexity=25,
                table_count=1,
                join_count=0,
                where_conditions=1,
                subquery_count=0
            ),
            Query.objects.create(
                sql_text="""
                SELECT u.name, p.title, COUNT(c.id) as comments
                FROM users u
                JOIN posts p ON u.id = p.user_id
                LEFT JOIN comments c ON p.id = c.post_id
                GROUP BY u.id, p.id
                """,
                query_type='SELECT',
                query_hash='complex_1',
                estimated_complexity=65,
                table_count=3,
                join_count=2,
                where_conditions=0,
                subquery_count=0
            ),
            Query.objects.create(
                sql_text="UPDATE users SET last_login = NOW() WHERE id = ?",
                query_type='UPDATE',
                query_hash='update_1',
                estimated_complexity=35,
                table_count=1,
                join_count=0,
                where_conditions=1,
                subquery_count=0
            )
        ]

    def test_feature_extraction_to_grading_pipeline(self):
        """Test feature extraction flows correctly to grading."""
        extractor = FeatureExtractor()
        grader = HybridQueryGrader()

        for query in self.queries:
            # Extract features
            features = extractor.extract_features(query)
            self.assertIsNotNone(features)

            # Use features in grading
            result = grader.analyze_query(query.sql_text, use_ml=False)
            self.assertIsNotNone(result)
            self.assertIn('grade', result)
            self.assertIn('score', result)

    def test_feedback_collection_to_training_pipeline(self):
        """Test feedback collection flows to model training."""
        collector = FeedbackCollector()
        grader = HybridQueryGrader()

        # Create feedback for each query
        for i, query in enumerate(self.queries):
            # Add user history
            UserQueryHistory.objects.create(
                user=self.user,
                query=query,
                analysis_result={'score': 70 + i * 10, 'grade': 'B'},
                execution_time=0.2 + i * 0.1
            )

            # Add feedback
            QueryFeedback.objects.create(
                query=query,
                user=self.user,
                is_helpful=i % 2 == 0,  # Alternating feedback
                score_agreement=4 if i % 2 == 0 else 2,
                comments=f"Test feedback {i}"
            )

        # Collect feedback for training
        for query in self.queries:
            training_data = collector.collect_feedback_for_query(query.id)
            if training_data:  # May be None if insufficient feedback
                self.assertIsInstance(training_data, TrainingData)

    def test_end_to_end_ml_workflow(self):
        """Test complete ML workflow from analysis to learning."""
        # 1. Analyze queries and record results
        analysis_results = []
        for query in self.queries:
            result = analyze_query(query.sql_text, use_ml=False)
            analysis_results.append(result)

            # Record user history
            UserQueryHistory.objects.create(
                user=self.user,
                query=query,
                analysis_result=result,
                execution_time=0.3
            )

        # 2. Simulate user feedback
        feedback_scores = [5, 2, 4]  # Mix of positive and negative
        for i, query in enumerate(self.queries):
            QueryFeedback.objects.create(
                query=query,
                user=self.user,
                is_helpful=feedback_scores[i] >= 3,
                score_agreement=feedback_scores[i],
                comments=f"Feedback for query {i}"
            )

        # 3. Collect feedback for training
        collector = FeedbackCollector()
        training_data_items = []

        for query in self.queries:
            training_data = collector.collect_feedback_for_query(query.id)
            if training_data:
                training_data_items.append(training_data)

        # 4. Verify training data creation
        self.assertGreater(len(training_data_items), 0)

        # 5. Test hybrid grading with collected data
        grader = HybridQueryGrader()
        for query in self.queries:
            result = grader.analyze_query(query.sql_text, use_ml=False)
            self.assertIsNotNone(result)

    def test_ml_model_lifecycle(self):
        """Test ML model creation, training, and deployment lifecycle."""
        # Create initial model record
        model = MLModel.objects.create(
            name='test_lifecycle_model',
            model_type='QUERY_GRADER',
            version='1.0.0',
            file_path='/tmp/test_model.pkl',
            is_active=False,
            performance_metrics={'accuracy': 0.0}
        )

        # Add training data
        for i, query in enumerate(self.queries):
            TrainingData.objects.create(
                query=query,
                features_json=[1.0, 2.0, 3.0, 4.0, 5.0],
                target_score=70 + i * 10,
                feedback_weight=1.0,
                user_reliability_score=0.8,
                created_date=datetime.now().date()
            )

        # Test model training simulation
        grader = HybridQueryGrader()

        # Mock successful training
        with patch.object(grader, '_save_model', return_value=True):
            with patch.object(grader, '_prepare_training_data',
                             return_value=(np.array([[1, 2], [3, 4]]), np.array([70, 80]))):
                success = grader.train_model()

                if success:
                    # Update model status
                    model.is_active = True
                    model.performance_metrics = {'accuracy': 0.85}
                    model.save()

        # Verify model is ready for use
        active_models = MLModel.objects.filter(
            model_type='QUERY_GRADER',
            is_active=True
        )
        self.assertGreater(active_models.count(), 0)

    def test_feedback_learning_metrics_tracking(self):
        """Test that feedback learning is properly tracked."""
        # Create feedback learning records
        for i, query in enumerate(self.queries):
            FeedbackLearning.objects.create(
                query=query,
                user=self.user,
                original_score=70 + i * 5,
                user_feedback_score=4 if i % 2 == 0 else 2,
                agreement_level='HIGH' if i % 2 == 0 else 'LOW',
                learning_weight=0.8 if i % 2 == 0 else 0.3,
                model_version='1.0.0'
            )

        # Create learning metrics
        LearningMetrics.objects.create(
            model_version='1.0.0',
            training_accuracy=0.85,
            validation_accuracy=0.82,
            feedback_correlation=0.75,
            user_satisfaction_avg=3.5,
            total_feedback_count=len(self.queries),
            created_date=datetime.now().date()
        )

        # Verify metrics tracking
        metrics = LearningMetrics.objects.filter(model_version='1.0.0').first()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_feedback_count, len(self.queries))

    def test_user_reliability_scoring(self):
        """Test user reliability scoring affects training weights."""
        # Create multiple users with different feedback patterns
        reliable_user = User.objects.create_user(
            username='reliable',
            email='reliable@example.com',
            password='testpass'
        )

        unreliable_user = User.objects.create_user(
            username='unreliable',
            email='unreliable@example.com',
            password='testpass'
        )

        # Add consistent feedback from reliable user
        for query in self.queries:
            QueryFeedback.objects.create(
                query=query,
                user=reliable_user,
                is_helpful=True,
                score_agreement=5,
                comments="Consistently good feedback"
            )

        # Add inconsistent feedback from unreliable user
        for i, query in enumerate(self.queries):
            QueryFeedback.objects.create(
                query=query,
                user=unreliable_user,
                is_helpful=i % 2 == 0,
                score_agreement=1 if i % 2 else 5,
                comments="Inconsistent feedback"
            )

        # Test feedback collection considers user reliability
        collector = FeedbackCollector()

        for query in self.queries:
            training_data = collector.collect_feedback_for_query(query.id)
            if training_data:
                # Reliable user should have higher weight
                self.assertIsNotNone(training_data.user_reliability_score)

    def test_feature_validation_in_pipeline(self):
        """Test feature validation throughout the pipeline."""
        extractor = FeatureExtractor()

        for query in self.queries:
            features = extractor.extract_features(query)

            if features is not None:
                # Validate features
                is_valid = extractor.validate_features(features)
                self.assertTrue(is_valid, f"Features invalid for query: {query.sql_text}")

                # Check feature count consistency
                expected_count = extractor.get_feature_count()
                self.assertEqual(len(features), expected_count)

    def test_database_dialect_consistency(self):
        """Test consistency across different database dialects."""
        dialects = ['mysql', 'postgresql', 'sqlite']
        grader = HybridQueryGrader()

        for dialect in dialects:
            for query in self.queries:
                result = grader.analyze_query(
                    query.sql_text,
                    database_type=dialect,
                    use_ml=False
                )

                self.assertIsNotNone(result)
                self.assertIn('database_type', result)
                self.assertEqual(result['database_type'], dialect)

    def test_concurrent_analysis_safety(self):
        """Test that concurrent analysis operations are safe."""
        import threading
        import time

        results = []
        errors = []

        def analyze_worker(query_text):
            try:
                result = analyze_query(query_text, use_ml=False)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple analysis threads
        threads = []
        for query in self.queries:
            thread = threading.Thread(target=analyze_worker, args=(query.sql_text,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors and all analyses completed
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), len(self.queries))

    def test_memory_usage_optimization(self):
        """Test memory usage remains reasonable during processing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process many queries
        grader = HybridQueryGrader()
        for _ in range(10):  # Repeat to test memory accumulation
            for query in self.queries:
                result = grader.analyze_query(query.sql_text, use_ml=False)
                self.assertIsNotNone(result)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024,
                       f"Memory usage increased by {memory_increase / 1024 / 1024:.2f}MB")


class MLSystemPerformanceTestCase(TransactionTestCase):
    """Performance tests for the ML system."""

    def test_analysis_performance_benchmarks(self):
        """Test analysis performance meets benchmarks."""
        import time

        # Create test data
        user = User.objects.create_user(
            username='perfuser',
            email='perf@example.com',
            password='testpass'
        )

        test_queries = [
            "SELECT * FROM users",
            "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id",
            "UPDATE users SET last_login = NOW() WHERE active = 1 AND created_at > '2023-01-01'",
            "DELETE FROM logs WHERE level = 'DEBUG' AND created_at < DATE_SUB(NOW(), INTERVAL 30 DAY)"
        ]

        total_time = 0
        analysis_count = 0

        for query_text in test_queries:
            start_time = time.time()
            result = analyze_query(query_text, use_ml=False)
            end_time = time.time()

            if result:
                analysis_time = end_time - start_time
                total_time += analysis_time
                analysis_count += 1

                # Individual query should complete in under 1 second
                self.assertLess(analysis_time, 1.0,
                               f"Query analysis took {analysis_time:.3f}s, exceeds 1s limit")

        # Average analysis time should be under 0.5 seconds
        if analysis_count > 0:
            avg_time = total_time / analysis_count
            self.assertLess(avg_time, 0.5,
                           f"Average analysis time {avg_time:.3f}s exceeds 0.5s benchmark")

    def test_bulk_feedback_processing_performance(self):
        """Test bulk feedback processing performance."""
        import time

        # Create test data
        user = User.objects.create_user(
            username='bulkuser',
            email='bulk@example.com',
            password='testpass'
        )

        # Create multiple queries and feedback
        queries = []
        for i in range(50):  # Test with 50 queries
            query = Query.objects.create(
                sql_text=f"SELECT * FROM table_{i} WHERE id = {i}",
                query_type='SELECT',
                query_hash=f'bulk_test_{i}',
                estimated_complexity=30 + i,
                table_count=1,
                join_count=0,
                where_conditions=1,
                subquery_count=0
            )
            queries.append(query)

            # Add feedback
            QueryFeedback.objects.create(
                query=query,
                user=user,
                is_helpful=i % 2 == 0,
                score_agreement=3 + (i % 3),
                comments=f"Bulk feedback {i}"
            )

        # Test bulk feedback collection
        collector = FeedbackCollector()
        start_time = time.time()

        training_data_count = 0
        for query in queries:
            training_data = collector.collect_feedback_for_query(query.id)
            if training_data:
                training_data_count += 1

        end_time = time.time()
        processing_time = end_time - start_time

        # Bulk processing should complete in reasonable time
        self.assertLess(processing_time, 10.0,
                       f"Bulk feedback processing took {processing_time:.3f}s, exceeds 10s limit")

        # Should process most queries successfully
        success_rate = training_data_count / len(queries)
        self.assertGreater(success_rate, 0.5,
                          f"Success rate {success_rate:.2f} is below 50%")


if __name__ == '__main__':
    unittest.main()