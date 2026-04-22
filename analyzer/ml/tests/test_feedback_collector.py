"""
Tests for the ML Feedback Collector

This module tests the feedback collection and aggregation functionality
for training machine learning models from user feedback.
"""

import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from django.contrib.auth.models import User
from django.test import TestCase

from analyzer.ml.core.feedback_collector import FeedbackCollector
from analyzer.models import (
    FeedbackLearning,
    Query,
    QueryAnalysis,
    QueryFeedback,
    TrainingData,
    UserQueryHistory,
)


class FeedbackCollectorTestCase(TestCase):
    """Test cases for the FeedbackCollector class."""

    def setUp(self):
        """Set up test data."""
        self.collector = FeedbackCollector()

        # Create test users
        self.user1 = User.objects.create_user(
            username="testuser1", email="test1@example.com", password="testpass"
        )
        self.user2 = User.objects.create_user(
            username="testuser2", email="test2@example.com", password="testpass"
        )
        self.user3 = User.objects.create_user(
            username="testuser3", email="test3@example.com", password="testpass"
        )

        # Create test query and analysis
        self.query = Query.objects.create(
            sql_text="SELECT * FROM users WHERE id = 1",
            query_type="SELECT",
            query_hash="test_hash",
            estimated_complexity=25,
            table_count=1,
            join_count=0,
            where_conditions=1,
            subquery_count=0,
        )

        self.analysis = QueryAnalysis.objects.create(
            query=self.query,
            grade="C",
            score=75.0,
            issues_found=[{"type": "SELECT_STAR", "severity": "medium"}],
            recommendations=[{"type": "SELECT_SPECIFIC", "priority": "medium"}],
            performance_notes="Query uses SELECT *",
        )

    def test_extract_feedback_from_history_detailed(self):
        """Test extracting feedback from detailed feedback."""
        # Create user history
        history = UserQueryHistory.objects.create(
            user=self.user1, query=self.query, database_type="MySQL"
        )

        # Create detailed feedback
        detailed_feedback = QueryFeedback.objects.create(
            user_history=history,
            accuracy_rating=4,
            usefulness_rating=5,
            clarity_rating=3,
            suggestions="Good analysis overall",
            would_recommend=True,
        )

        feedback_data = self.collector._extract_feedback_from_history(history)

        self.assertIsNotNone(feedback_data)
        self.assertEqual(feedback_data["user"], self.user1)
        self.assertEqual(feedback_data["accuracy_rating"], 4)
        self.assertEqual(feedback_data["usefulness_rating"], 5)
        self.assertEqual(feedback_data["clarity_rating"], 3)
        self.assertTrue(feedback_data["would_recommend"])
        self.assertGreater(feedback_data["confidence"], 0.5)

    def test_extract_feedback_from_history_simple(self):
        """Test extracting feedback from simple thumbs up/down."""
        # Create user history with simple feedback
        history = UserQueryHistory.objects.create(
            user=self.user1, query=self.query, database_type="MySQL", was_helpful=True
        )

        feedback_data = self.collector._extract_feedback_from_history(history)

        self.assertIsNotNone(feedback_data)
        self.assertEqual(feedback_data["user"], self.user1)
        self.assertEqual(feedback_data["user_grade"], 4.0)  # True -> 4.0
        self.assertEqual(
            feedback_data["confidence"], 0.7
        )  # Lower confidence for simple feedback
        self.assertTrue(feedback_data["would_recommend"])

    def test_extract_feedback_no_feedback(self):
        """Test extracting feedback when no feedback exists."""
        history = UserQueryHistory.objects.create(
            user=self.user1, query=self.query, database_type="MySQL"
        )

        feedback_data = self.collector._extract_feedback_from_history(history)
        self.assertIsNone(feedback_data)

    def test_convert_ratings_to_grade(self):
        """Test conversion of ratings to single grade value."""
        # Test with all ratings
        grade = self.collector._convert_ratings_to_grade(4, 5, 3)
        expected = 4 * 0.3 + 5 * 0.5 + 3 * 0.2  # Weighted average
        self.assertAlmostEqual(grade, expected, places=2)

        # Test with missing ratings
        grade = self.collector._convert_ratings_to_grade(4, None, 3)
        expected = (4 * 0.3 + 3 * 0.2) / (0.3 + 0.2)  # Only accuracy and clarity
        self.assertAlmostEqual(grade, expected, places=2)

        # Test with no ratings
        grade = self.collector._convert_ratings_to_grade(None, None, None)
        self.assertEqual(grade, 3.0)  # Default neutral

    def test_calculate_feedback_confidence(self):
        """Test feedback confidence calculation."""
        # Create feedback with all ratings
        history = UserQueryHistory.objects.create(user=self.user1, query=self.query)
        feedback = QueryFeedback.objects.create(
            user_history=history,
            accuracy_rating=4,
            usefulness_rating=4,
            clarity_rating=4,
            suggestions="Detailed suggestion with good length",
            would_recommend=True,
        )

        confidence = self.collector._calculate_feedback_confidence(feedback)
        self.assertGreater(confidence, 0.5)  # Should be fairly confident

        # Test with inconsistent ratings
        feedback.accuracy_rating = 1
        feedback.usefulness_rating = 5
        feedback.clarity_rating = 3
        feedback.save()

        confidence = self.collector._calculate_feedback_confidence(feedback)
        # Should be lower due to inconsistency

    def test_aggregate_feedback(self):
        """Test aggregation of multiple feedback instances."""
        feedback_data = [
            {
                "user_grade": 4.0,
                "confidence": 0.8,
                "accuracy_rating": 4,
                "usefulness_rating": 4,
                "clarity_rating": 4,
                "user": self.user1,
                "timestamp": None,
                "would_recommend": True,
            },
            {
                "user_grade": 3.0,
                "confidence": 0.7,
                "accuracy_rating": 3,
                "usefulness_rating": 3,
                "clarity_rating": 3,
                "user": self.user2,
                "timestamp": None,
                "would_recommend": False,
            },
            {
                "user_grade": 5.0,
                "confidence": 0.9,
                "accuracy_rating": 5,
                "usefulness_rating": 5,
                "clarity_rating": 5,
                "user": self.user3,
                "timestamp": None,
                "would_recommend": True,
            },
        ]

        with patch.object(self.collector, "_get_user_reliability", return_value=0.8):
            aggregated = self.collector._aggregate_feedback(feedback_data)

        self.assertIn("user_grade_avg", aggregated)
        self.assertIn("user_grade_count", aggregated)
        self.assertIn("user_grade_stddev", aggregated)
        self.assertEqual(aggregated["user_grade_count"], 3)
        self.assertGreater(aggregated["user_grade_avg"], 3.0)
        self.assertLess(aggregated["user_grade_avg"], 5.0)

    def test_get_user_reliability(self):
        """Test user reliability scoring."""
        # New user with no feedback history
        reliability = self.collector._get_user_reliability(self.user1)
        self.assertEqual(reliability, 0.5)  # Default for new users

        # Create some feedback history for user
        for i in range(5):
            query = Query.objects.create(
                sql_text=f"SELECT * FROM table_{i}",  # nosec
                query_type="SELECT",
                query_hash=f"test_hash_{i}",
            )
            analysis = QueryAnalysis.objects.create(query=query, grade="B", score=80.0)
            history = UserQueryHistory.objects.create(user=self.user1, query=query)
            QueryFeedback.objects.create(
                user_history=history,
                accuracy_rating=4,
                usefulness_rating=4,
                clarity_rating=4,
            )

        reliability = self.collector._get_user_reliability(self.user1)
        self.assertGreater(reliability, 0.5)  # Should increase with more feedback

    def test_collect_feedback_for_query_success(self):
        """Test successful feedback collection for a query."""
        # Create multiple user histories with feedback
        histories = []
        for user in [self.user1, self.user2, self.user3]:
            history = UserQueryHistory.objects.create(
                user=user, query=self.query, database_type="MySQL"
            )
            QueryFeedback.objects.create(
                user_history=history,
                accuracy_rating=4,
                usefulness_rating=4,
                clarity_rating=4,
                would_recommend=True,
            )
            histories.append(history)

        training_data = self.collector.collect_feedback_for_query(self.query.id)

        self.assertIsNotNone(training_data)
        self.assertIsInstance(training_data, TrainingData)
        self.assertEqual(training_data.query, self.query)
        self.assertEqual(training_data.user_grade_count, 3)
        self.assertGreater(training_data.user_grade_avg, 0)

        # Check that learning records were created
        learning_records = FeedbackLearning.objects.filter(user_history__in=histories)
        self.assertEqual(learning_records.count(), 3)

    def test_collect_feedback_insufficient_data(self):
        """Test feedback collection with insufficient data."""
        # Create only one feedback (below threshold)
        history = UserQueryHistory.objects.create(
            user=self.user1, query=self.query, database_type="MySQL"
        )
        QueryFeedback.objects.create(
            user_history=history,
            accuracy_rating=4,
            usefulness_rating=4,
            clarity_rating=4,
        )

        training_data = self.collector.collect_feedback_for_query(self.query.id)
        self.assertIsNone(
            training_data
        )  # Should return None due to insufficient feedback

    def test_collect_feedback_no_histories(self):
        """Test feedback collection with no user histories."""
        training_data = self.collector.collect_feedback_for_query(self.query.id)
        self.assertIsNone(training_data)

    def test_create_training_data(self):
        """Test creation of training data record."""
        aggregated_data = {
            "user_grade_avg": 4.2,
            "user_grade_count": 5,
            "user_grade_stddev": 0.8,
            "accuracy_rating_avg": 4.0,
            "usefulness_rating_avg": 4.5,
            "clarity_rating_avg": 4.2,
            "total_weight": 3.5,
        }

        training_data = self.collector._create_training_data(
            self.query, self.analysis, aggregated_data
        )

        self.assertIsNotNone(training_data)
        self.assertEqual(training_data.query, self.query)
        self.assertEqual(training_data.user_grade_avg, 4.2)
        self.assertEqual(training_data.user_grade_count, 5)
        self.assertEqual(training_data.system_grade, "C")
        self.assertEqual(training_data.system_score, 75.0)
        self.assertEqual(training_data.query_complexity, 25)

    def test_create_training_data_update_existing(self):
        """Test updating existing training data."""
        # Create initial training data
        initial_data = TrainingData.objects.create(
            query=self.query,
            user_grade_avg=3.0,
            user_grade_count=2,
            user_grade_stddev=0.5,
            system_grade="C",
            system_score=75.0,
            query_complexity=25,
        )

        aggregated_data = {
            "user_grade_avg": 4.0,
            "user_grade_count": 4,
            "user_grade_stddev": 0.8,
            "accuracy_rating_avg": 4.0,
            "usefulness_rating_avg": 4.0,
            "clarity_rating_avg": 4.0,
        }

        updated_data = self.collector._create_training_data(
            self.query, self.analysis, aggregated_data
        )

        # Should be the same object, updated
        self.assertEqual(updated_data.id, initial_data.id)
        self.assertEqual(updated_data.user_grade_avg, 4.0)
        self.assertEqual(updated_data.user_grade_count, 4)

    def test_batch_collect_feedback(self):
        """Test batch feedback collection."""
        # Create multiple queries with feedback
        queries = []
        for i in range(3):
            query = Query.objects.create(
                sql_text=f"SELECT * FROM table_{i}",  # nosec
                query_type="SELECT",
                query_hash=f"batch_test_{i}",
                estimated_complexity=30 + i * 10,
            )
            analysis = QueryAnalysis.objects.create(
                query=query, grade="B", score=80.0 + i * 5
            )

            # Add feedback for each query
            for user in [self.user1, self.user2, self.user3]:
                history = UserQueryHistory.objects.create(
                    user=user, query=query, database_type="MySQL"
                )
                QueryFeedback.objects.create(
                    user_history=history,
                    accuracy_rating=4,
                    usefulness_rating=4,
                    clarity_rating=4,
                )

            queries.append(query)

        # Collect feedback for all queries
        training_data_list = self.collector.batch_collect_feedback(days_back=1)

        self.assertEqual(len(training_data_list), 3)
        for training_data in training_data_list:
            self.assertIsInstance(training_data, TrainingData)
            self.assertEqual(training_data.user_grade_count, 3)

    def test_get_training_dataset(self):
        """Test getting training dataset for model training."""
        # Create training data
        for i in range(5):
            query = Query.objects.create(
                sql_text=f"SELECT * FROM table_{i}",  # nosec
                query_type="SELECT",
                query_hash=f"dataset_test_{i}",
            )
            TrainingData.objects.create(
                query=query,
                user_grade_avg=3.5 + i * 0.3,
                user_grade_count=3 + i,
                user_grade_stddev=0.5,
                system_grade="B",
                system_score=75.0,
                query_complexity=30,
                is_validated=(i < 3),  # First 3 are validated
            )

        # Get all training data
        dataset = self.collector.get_training_dataset(min_feedback_count=3)
        self.assertEqual(len(dataset), 5)

        # Get only validated data
        validated_dataset = self.collector.get_training_dataset(
            min_feedback_count=3, include_validated_only=True
        )
        self.assertEqual(len(validated_dataset), 3)

    def test_feedback_weight_threshold(self):
        """Test feedback weight threshold filtering."""
        # Create feedback with very low reliability
        with patch.object(self.collector, "_get_user_reliability", return_value=0.05):
            feedback_data = [
                {
                    "user_grade": 4.0,
                    "confidence": 0.1,  # Very low confidence
                    "accuracy_rating": 4,
                    "usefulness_rating": 4,
                    "clarity_rating": 4,
                    "user": self.user1,
                    "timestamp": None,
                    "would_recommend": True,
                }
            ]

            aggregated = self.collector._aggregate_feedback(feedback_data)
            self.assertEqual(aggregated, {})  # Should be empty due to low weight


class FeedbackCollectorIntegrationTestCase(TestCase):
    """Integration tests for feedback collector with realistic scenarios."""

    def setUp(self):
        """Set up integration test data."""
        self.collector = FeedbackCollector()

    def test_end_to_end_feedback_pipeline(self):
        """Test complete feedback collection pipeline."""
        # Create a realistic scenario with multiple users providing feedback
        users = [
            User.objects.create_user(f"user{i}", f"user{i}@test.com", "pass")
            for i in range(10)
        ]

        query = Query.objects.create(
            sql_text="SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id",
            query_type="SELECT",
            query_hash="integration_test",
            estimated_complexity=60,
            table_count=2,
            join_count=1,
            where_conditions=0,
            subquery_count=0,
        )

        analysis = QueryAnalysis.objects.create(
            query=query,
            grade="B",
            score=82.0,
            issues_found=[],
            recommendations=[{"type": "add_index", "priority": "medium"}],
        )

        # Simulate varied user feedback
        ratings = [
            (4, 5, 4),
            (3, 4, 3),
            (5, 5, 5),
            (4, 4, 4),
            (3, 3, 4),
            (4, 5, 3),
            (5, 4, 5),
            (3, 4, 4),
            (4, 4, 4),
            (5, 5, 4),
        ]

        for user, (acc, use, cla) in zip(users, ratings):
            history = UserQueryHistory.objects.create(
                user=user,
                query=query,
                database_type="PostgreSQL",
                database_version="13.0",
            )

            QueryFeedback.objects.create(
                user_history=history,
                accuracy_rating=acc,
                usefulness_rating=use,
                clarity_rating=cla,
                suggestions=f"Feedback from {user.username}",
                would_recommend=(acc + use + cla) > 9,
            )

        # Collect feedback
        training_data = self.collector.collect_feedback_for_query(query.id)

        self.assertIsNotNone(training_data)
        self.assertEqual(training_data.user_grade_count, 10)
        self.assertGreater(training_data.user_grade_avg, 3.0)
        self.assertLess(training_data.user_grade_avg, 5.0)

        # Verify learning records were created
        learning_records = FeedbackLearning.objects.filter(user_history__query=query)
        self.assertEqual(learning_records.count(), 10)

        for record in learning_records:
            self.assertIsNotNone(record.feedback_grade_equivalent)
            self.assertIsNotNone(record.grade_difference)
            self.assertGreater(record.feedback_weight, 0)


if __name__ == "__main__":
    unittest.main()
