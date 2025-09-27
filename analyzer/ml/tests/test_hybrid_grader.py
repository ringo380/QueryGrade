"""
Tests for the ML Hybrid Query Grader

This module tests the hybrid grading system that combines rule-based
analysis with machine learning predictions to provide accurate query scores.
"""

import unittest
from django.test import TestCase
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime, timedelta

from analyzer.models import Query, QueryFeedback, UserQueryHistory
from analyzer.models import MLModel, TrainingData, LearningMetrics, FeedbackLearning
from analyzer.ml.hybrid_grader import HybridQueryGrader
from django.contrib.auth.models import User


class HybridQueryGraderTestCase(TestCase):
    """Test cases for the HybridQueryGrader class."""

    def setUp(self):
        """Set up test data."""
        self.grader = HybridQueryGrader()

        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass'
        )

        # Create test queries with different complexity levels
        self.simple_query = Query.objects.create(
            sql_text="SELECT id, name FROM users WHERE id = 1",
            query_type='SELECT',
            query_hash='simple_test',
            estimated_complexity=25,
            table_count=1,
            join_count=0,
            where_conditions=1,
            subquery_count=0
        )

        self.complex_query = Query.objects.create(
            sql_text="""
            SELECT u.id, u.name, p.title, COUNT(c.id) as comment_count
            FROM users u
            INNER JOIN posts p ON u.id = p.user_id
            LEFT JOIN comments c ON p.id = c.post_id
            WHERE u.active = 1 AND p.published_at > '2023-01-01'
            GROUP BY u.id, u.name, p.title
            HAVING COUNT(c.id) > 5
            ORDER BY comment_count DESC
            LIMIT 10
            """,
            query_type='SELECT',
            query_hash='complex_test',
            estimated_complexity=75,
            table_count=3,
            join_count=2,
            where_conditions=2,
            subquery_count=0
        )

    def test_initialization(self):
        """Test grader initialization and model loading."""
        self.assertIsNotNone(self.grader.feature_extractor)
        self.assertIsNotNone(self.grader.feedback_collector)
        self.assertIsInstance(self.grader.ml_weight, float)
        self.assertBetween(self.grader.ml_weight, 0.0, 1.0)

    def test_analyze_query_rule_based_only(self):
        """Test query analysis using only rule-based grading."""
        result = self.grader.analyze_query(
            self.simple_query.sql_text,
            use_ml=False
        )

        self.assertIsInstance(result, dict)
        self.assertIn('grade', result)
        self.assertIn('score', result)
        self.assertIn('feedback', result)
        self.assertIn('method_used', result)
        self.assertEqual(result['method_used'], 'rule_based')
        self.assertBetween(result['score'], 0, 100)

    @patch('analyzer.ml.hybrid_grader.HybridQueryGrader._get_ml_prediction')
    def test_analyze_query_with_ml(self, mock_ml_prediction):
        """Test query analysis using hybrid approach with ML."""
        # Mock ML prediction
        mock_ml_prediction.return_value = {
            'score': 85,
            'confidence': 0.8,
            'features': np.array([1.0, 2.0, 3.0])
        }

        result = self.grader.analyze_query(
            self.simple_query.sql_text,
            use_ml=True
        )

        self.assertIsInstance(result, dict)
        self.assertIn('grade', result)
        self.assertIn('score', result)
        self.assertIn('feedback', result)
        self.assertIn('method_used', result)
        self.assertIn('ml_confidence', result)
        self.assertEqual(result['method_used'], 'hybrid')
        self.assertBetween(result['score'], 0, 100)

    @patch('analyzer.ml.hybrid_grader.HybridQueryGrader._get_ml_prediction')
    def test_low_confidence_ml_prediction(self, mock_ml_prediction):
        """Test behavior when ML prediction has low confidence."""
        # Mock low confidence ML prediction
        mock_ml_prediction.return_value = {
            'score': 60,
            'confidence': 0.2,  # Low confidence
            'features': np.array([1.0, 2.0, 3.0])
        }

        result = self.grader.analyze_query(
            self.simple_query.sql_text,
            use_ml=True
        )

        # Should rely more heavily on rule-based score
        self.assertIn('ml_confidence', result)
        self.assertEqual(result['ml_confidence'], 0.2)
        # Method should still be hybrid but weighted toward rules
        self.assertEqual(result['method_used'], 'hybrid')

    def test_score_to_grade_conversion(self):
        """Test score to letter grade conversion."""
        test_cases = [
            (95, 'A'),
            (85, 'B'),
            (75, 'C'),
            (65, 'D'),
            (55, 'F'),
            (100, 'A'),
            (0, 'F')
        ]

        for score, expected_grade in test_cases:
            grade = self.grader._score_to_grade(score)
            self.assertEqual(grade, expected_grade,
                           f"Score {score} should map to grade {expected_grade}")

    def test_model_confidence_calculation(self):
        """Test ML model confidence calculation."""
        # Test with different prediction scenarios
        test_cases = [
            ([0.9, 0.1, 0.0], 0.9),  # High confidence
            ([0.4, 0.3, 0.3], 0.4),  # Low confidence
            ([0.6, 0.4], 0.6),       # Medium confidence
            ([1.0], 1.0),            # Single prediction
        ]

        for probabilities, expected_confidence in test_cases:
            confidence = self.grader._calculate_confidence(np.array(probabilities))
            self.assertAlmostEqual(confidence, expected_confidence, places=2)

    @patch('analyzer.ml.hybrid_grader.joblib.load')
    def test_model_loading(self, mock_joblib_load):
        """Test ML model loading functionality."""
        # Create a mock model in database
        ml_model = MLModel.objects.create(
            name='test_grader',
            model_type='QUERY_GRADER',
            version='1.0.0',
            file_path='/tmp/test_model.pkl',
            is_active=True,
            performance_metrics={'accuracy': 0.85}
        )

        # Mock joblib.load to return a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([75])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_joblib_load.return_value = mock_model

        # Load model
        loaded_model = self.grader._load_model()

        self.assertIsNotNone(loaded_model)
        mock_joblib_load.assert_called_once()

    def test_training_data_preparation(self):
        """Test training data preparation from feedback."""
        # Create feedback data
        QueryFeedback.objects.create(
            query=self.simple_query,
            user=self.user,
            is_helpful=True,
            score_agreement=5,
            comments="Good analysis"
        )

        QueryFeedback.objects.create(
            query=self.complex_query,
            user=self.user,
            is_helpful=False,
            score_agreement=2,
            comments="Score too high"
        )

        # Prepare training data
        X, y = self.grader._prepare_training_data()

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(X), len(y))

    @patch('analyzer.ml.hybrid_grader.RandomForestRegressor')
    def test_model_training(self, mock_rf):
        """Test model training process."""
        # Mock training data
        mock_X = np.array([[1, 2, 3], [4, 5, 6]])
        mock_y = np.array([80, 60])

        with patch.object(self.grader, '_prepare_training_data',
                         return_value=(mock_X, mock_y)):

            # Mock the model
            mock_model = MagicMock()
            mock_rf.return_value = mock_model

            # Train model
            success = self.grader.train_model()

            self.assertTrue(success)
            mock_model.fit.assert_called_once_with(mock_X, mock_y)

    def test_feedback_integration(self):
        """Test integration with feedback collection system."""
        # Create some query history
        history = UserQueryHistory.objects.create(
            user=self.user,
            query=self.simple_query,
            analysis_result={'score': 80, 'grade': 'B'},
            execution_time=0.5
        )

        # Add feedback
        QueryFeedback.objects.create(
            query=self.simple_query,
            user=self.user,
            is_helpful=True,
            score_agreement=4,
            comments="Good score"
        )

        # Test that grader can access this feedback
        feedback_data = self.grader.feedback_collector.collect_feedback_for_query(
            self.simple_query.id
        )

        self.assertIsNotNone(feedback_data)

    def test_database_type_handling(self):
        """Test handling of different database types."""
        database_types = ['mysql', 'postgresql', 'sqlite', 'oracle']

        for db_type in database_types:
            result = self.grader.analyze_query(
                self.simple_query.sql_text,
                database_type=db_type,
                use_ml=False
            )

            self.assertIsInstance(result, dict)
            self.assertIn('grade', result)
            self.assertIn('database_type', result)
            self.assertEqual(result['database_type'], db_type)

    def test_error_handling(self):
        """Test error handling in analysis."""
        # Test with empty query
        result = self.grader.analyze_query("", use_ml=False)
        self.assertIsNone(result)

        # Test with malformed SQL
        result = self.grader.analyze_query(
            "SELECT FROM WHERE;",
            use_ml=False
        )
        self.assertIsInstance(result, dict)
        self.assertIn('grade', result)

    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        with patch('analyzer.ml.hybrid_grader.time.time') as mock_time:
            mock_time.side_effect = [1000.0, 1000.5]  # 0.5 second execution

            result = self.grader.analyze_query(
                self.simple_query.sql_text,
                use_ml=False
            )

            self.assertIn('execution_time', result)
            self.assertGreater(result['execution_time'], 0)

    def test_model_versioning(self):
        """Test model versioning and updates."""
        # Create multiple model versions
        old_model = MLModel.objects.create(
            name='grader_v1',
            model_type='QUERY_GRADER',
            version='1.0.0',
            file_path='/tmp/old_model.pkl',
            is_active=False,
            performance_metrics={'accuracy': 0.75}
        )

        new_model = MLModel.objects.create(
            name='grader_v2',
            model_type='QUERY_GRADER',
            version='2.0.0',
            file_path='/tmp/new_model.pkl',
            is_active=True,
            performance_metrics={'accuracy': 0.85}
        )

        # Should load the active model
        active_model = MLModel.objects.filter(
            model_type='QUERY_GRADER',
            is_active=True
        ).first()

        self.assertEqual(active_model.version, '2.0.0')

    def assertBetween(self, value, min_val, max_val, msg=None):
        """Custom assertion to check if value is between min and max."""
        if not (min_val <= value <= max_val):
            msg = msg or f"{value} is not between {min_val} and {max_val}"
            raise AssertionError(msg)


class HybridQueryGraderIntegrationTestCase(TestCase):
    """Integration tests for the hybrid grader with real data."""

    def setUp(self):
        """Set up integration test data."""
        self.grader = HybridQueryGrader()

        # Create test user
        self.user = User.objects.create_user(
            username='integrationuser',
            email='integration@example.com',
            password='testpass'
        )

    def test_end_to_end_grading_workflow(self):
        """Test complete grading workflow from query to feedback."""
        sql_text = """
        SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at > '2023-01-01'
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 5
        ORDER BY total_spent DESC
        """

        # 1. Analyze query
        result = self.grader.analyze_query(sql_text, use_ml=False)
        self.assertIsNotNone(result)

        # 2. Create query record
        query = Query.objects.create(
            sql_text=sql_text,
            query_type='SELECT',
            query_hash='integration_test',
            estimated_complexity=result.get('score', 50),
            table_count=2,
            join_count=1,
            where_conditions=1,
            subquery_count=0
        )

        # 3. Record user history
        history = UserQueryHistory.objects.create(
            user=self.user,
            query=query,
            analysis_result=result,
            execution_time=0.3
        )

        # 4. Add feedback
        feedback = QueryFeedback.objects.create(
            query=query,
            user=self.user,
            is_helpful=True,
            score_agreement=4,
            comments="Accurate analysis"
        )

        # 5. Verify feedback collection
        collected_feedback = self.grader.feedback_collector.collect_feedback_for_query(
            query.id
        )

        self.assertIsNotNone(collected_feedback)

    def test_batch_query_analysis(self):
        """Test analyzing multiple queries in batch."""
        queries = [
            "SELECT * FROM users",
            "SELECT id, name FROM users WHERE active = 1",
            "UPDATE users SET last_login = NOW() WHERE id = 1",
            "DELETE FROM logs WHERE created_at < '2023-01-01'"
        ]

        results = []
        for sql_text in queries:
            result = self.grader.analyze_query(sql_text, use_ml=False)
            results.append(result)

        self.assertEqual(len(results), len(queries))
        for result in results:
            self.assertIsNotNone(result)
            self.assertIn('grade', result)
            self.assertIn('score', result)

    def test_feedback_aggregation_impact(self):
        """Test how feedback aggregation affects future predictions."""
        # Create a query with consistent feedback
        query = Query.objects.create(
            sql_text="SELECT COUNT(*) FROM users",
            query_type='SELECT',
            query_hash='feedback_test',
            estimated_complexity=30,
            table_count=1,
            join_count=0,
            where_conditions=0,
            subquery_count=0
        )

        # Add multiple positive feedback entries
        for i in range(5):
            user = User.objects.create_user(
                username=f'feedbackuser{i}',
                email=f'feedback{i}@example.com',
                password='testpass'
            )

            QueryFeedback.objects.create(
                query=query,
                user=user,
                is_helpful=True,
                score_agreement=5,
                comments="Perfect score"
            )

        # Collect and verify aggregated feedback
        training_data = self.grader.feedback_collector.collect_feedback_for_query(
            query.id
        )

        self.assertIsNotNone(training_data)
        self.assertGreater(training_data.positive_feedback_count, 0)
        self.assertEqual(training_data.negative_feedback_count, 0)


if __name__ == '__main__':
    unittest.main()