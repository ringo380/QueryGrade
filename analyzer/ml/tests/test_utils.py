"""
ML Test Utilities

This module provides utilities and helpers for ML testing across
the QueryGrade application.
"""

import numpy as np
from django.contrib.auth.models import User
from django.utils import timezone

from analyzer.models import (
    FeedbackLearning,
    LearningMetrics,
    MLModel,
    Query,
    QueryFeedback,
    TrainingData,
    UserQueryHistory,
)


class MLTestDataFactory:
    """Factory for creating ML test data."""

    @staticmethod
    def create_test_user(username="testuser", email="test@example.com"):
        """Create a test user."""
        return User.objects.create_user(
            username=username, email=email, password="testpass123"
        )

    @staticmethod
    def create_test_query(
        sql_text="SELECT id, name FROM users WHERE active = 1",
        query_type="SELECT",
        complexity=25,
        table_count=1,
        join_count=0,
        where_conditions=1,
        subquery_count=0,
        query_hash=None,
    ):
        """Create a test query."""
        if query_hash is None:
            import hashlib

            query_hash = hashlib.md5(
                sql_text.encode(), usedforsecurity=False
            ).hexdigest()[:10]

        return Query.objects.create(
            sql_text=sql_text,
            query_type=query_type,
            query_hash=query_hash,
            estimated_complexity=complexity,
            table_count=table_count,
            join_count=join_count,
            where_conditions=where_conditions,
            subquery_count=subquery_count,
        )

    @staticmethod
    def create_query_feedback(
        query, user, is_helpful=True, score_agreement=4, comments="Test feedback"
    ):
        """Create test query feedback."""
        return QueryFeedback.objects.create(
            query=query,
            user=user,
            is_helpful=is_helpful,
            score_agreement=score_agreement,
            comments=comments,
        )

    @staticmethod
    def create_user_history(user, query, analysis_result=None, execution_time=0.3):
        """Create test user query history."""
        if analysis_result is None:
            analysis_result = {
                "score": 75,
                "grade": "B",
                "feedback": ["Good query structure"],
            }

        return UserQueryHistory.objects.create(
            user=user,
            query=query,
            analysis_result=analysis_result,
            execution_time=execution_time,
        )

    @staticmethod
    def create_ml_model(
        name="test_model",
        model_type="HYBRID_SCORER",
        version="1.0.0",
        is_active=True,
        training_accuracy=0.85,
        validation_accuracy=0.82,
        training_samples=100,
    ):
        """Create test ML model."""
        return MLModel.objects.create(
            name=name,
            model_type=model_type,
            version=version,
            file_path=f"/tmp/{name}.pkl",
            status="ACTIVE" if is_active else "TRAINING",
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            training_samples=training_samples,
        )

    @staticmethod
    def create_training_data(
        query,
        features=None,
        target_score=75,
        feedback_weight=1.0,
        user_reliability_score=0.8,
    ):
        """Create test training data."""
        if features is None:
            features = [1.0, 2.0, 3.0, 4.0, 5.0]

        return TrainingData.objects.create(
            query=query,
            features_json=features,
            target_score=target_score,
            feedback_weight=feedback_weight,
            user_reliability_score=user_reliability_score,
        )

    @staticmethod
    def create_learning_metrics(
        model=None,
        accuracy=0.85,
        precision=0.82,
        recall=0.88,
        f1_score=0.85,
        user_agreement_rate=0.75,
        avg_user_rating=3.5,
        avg_prediction_time_ms=10.0,
    ):
        """Create test learning metrics."""
        if model is None:
            model = MLTestDataFactory.create_ml_model()

        now = timezone.now()
        return LearningMetrics.objects.create(
            model=model,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            user_agreement_rate=user_agreement_rate,
            avg_user_rating=avg_user_rating,
            avg_prediction_time_ms=avg_prediction_time_ms,
            measurement_period_start=now,
            measurement_period_end=now,
        )

    @staticmethod
    def create_feedback_learning(
        query,
        user,
        original_score=75,
        user_feedback_score=4,
        agreement_level="HIGH",
        learning_weight=0.8,
    ):
        """Create test feedback learning record."""
        return FeedbackLearning.objects.create(
            query=query,
            user=user,
            original_score=original_score,
            user_feedback_score=user_feedback_score,
            agreement_level=agreement_level,
            learning_weight=learning_weight,
        )


class MLTestAssertions:
    """Custom assertions for ML testing."""

    @staticmethod
    def assert_valid_features(test_case, features, expected_count=None):
        """Assert that features are valid."""
        test_case.assertIsNotNone(features)
        test_case.assertIsInstance(features, (list, np.ndarray))

        if expected_count:
            test_case.assertEqual(len(features), expected_count)

        # Check for NaN values
        if isinstance(features, np.ndarray):
            test_case.assertFalse(
                np.isnan(features).any(), "Features contain NaN values"
            )
        else:
            for feature in features:
                test_case.assertFalse(np.isnan(feature), "Features contain NaN values")

    @staticmethod
    def assert_valid_grade_result(test_case, result):
        """Assert that a grading result is valid."""
        test_case.assertIsInstance(result, dict)
        test_case.assertIn("grade", result)
        test_case.assertIn("score", result)
        test_case.assertIn("feedback", result)

        # Check grade is valid letter
        valid_grades = ["A", "B", "C", "D", "F"]
        test_case.assertIn(result["grade"], valid_grades)

        # Check score is in valid range
        test_case.assertGreaterEqual(result["score"], 0)
        test_case.assertLessEqual(result["score"], 100)

        # Check feedback is a list
        test_case.assertIsInstance(result["feedback"], list)

    @staticmethod
    def assert_score_in_range(test_case, score, min_score=0, max_score=100):
        """Assert that a score is in the expected range."""
        test_case.assertGreaterEqual(
            score, min_score, f"Score {score} below minimum {min_score}"
        )
        test_case.assertLessEqual(
            score, max_score, f"Score {score} above maximum {max_score}"
        )

    @staticmethod
    def assert_model_performance_metrics(test_case, metrics):
        """Assert that model performance metrics are valid."""
        test_case.assertIsInstance(metrics, dict)

        # Check common metrics
        for metric in ["accuracy", "precision", "recall"]:
            if metric in metrics:
                test_case.assertGreaterEqual(metrics[metric], 0.0)
                test_case.assertLessEqual(metrics[metric], 1.0)

    @staticmethod
    def assert_feedback_weight_valid(test_case, weight):
        """Assert that a feedback weight is valid."""
        test_case.assertIsInstance(weight, (int, float))
        test_case.assertGreaterEqual(weight, 0.0)
        test_case.assertLessEqual(weight, 1.0)


class MockMLComponents:
    """Mock components for ML testing."""

    @staticmethod
    def create_mock_model():
        """Create a mock ML model."""
        from unittest.mock import MagicMock

        mock_model = MagicMock()

        # Mock predict method
        def mock_predict(X):
            return np.array([75.0] * len(X))

        def mock_predict_proba(X):
            return np.array([[0.2, 0.8]] * len(X))

        mock_model.predict = mock_predict
        mock_model.predict_proba = mock_predict_proba

        return mock_model

    @staticmethod
    def create_mock_features(count=41):
        """Create mock feature array."""
        return np.random.rand(count).astype(np.float32)

    @staticmethod
    def create_mock_training_data(num_samples=100, num_features=41):
        """Create mock training data."""
        X = np.random.rand(num_samples, num_features).astype(np.float32)
        y = np.random.randint(0, 101, num_samples).astype(np.float32)
        return X, y


class TestQuerySamples:
    """Sample SQL queries for testing."""

    SIMPLE_QUERIES = [
        "SELECT * FROM users",
        "SELECT id, name FROM users WHERE active = 1",
        "INSERT INTO users (name, email) VALUES ('John', 'john@example.com')",
        "UPDATE users SET last_login = NOW() WHERE id = 1",
        "DELETE FROM logs WHERE created_at < '2023-01-01'",
    ]

    COMPLEX_QUERIES = [
        """
        SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at > '2023-01-01'
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 5
        ORDER BY total_spent DESC
        LIMIT 10
        """,
        """
        WITH RECURSIVE category_tree AS (
            SELECT id, name, parent_id, 1 as level
            FROM categories WHERE parent_id IS NULL
            UNION ALL
            SELECT c.id, c.name, c.parent_id, ct.level + 1
            FROM categories c
            JOIN category_tree ct ON c.parent_id = ct.id
        )
        SELECT ct.name, COUNT(p.id) as product_count
        FROM category_tree ct
        LEFT JOIN products p ON ct.id = p.category_id
        WHERE ct.level <= 3
        GROUP BY ct.id, ct.name
        HAVING COUNT(p.id) > 0
        """,
        """
        SELECT DISTINCT
            u.id,
            u.name,
            (SELECT COUNT(*) FROM orders WHERE user_id = u.id) as order_count,
            (SELECT AVG(total) FROM orders WHERE user_id = u.id) as avg_order_value
        FROM users u
        WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.total > 100)
        ORDER BY order_count DESC, avg_order_value DESC
        """,
    ]

    PROBLEMATIC_QUERIES = [
        "SELECT * FROM users WHERE 1=1; DROP TABLE users; --",
        "SELECT * FROM large_table",  # No WHERE clause
        "SELECT u.* FROM users u JOIN posts p JOIN comments c",  # Missing ON clauses
        "SELECT COUNT(*) FROM (SELECT * FROM users) t1, (SELECT * FROM orders) t2",  # Cartesian product
        "",  # Empty query
        "SELCT * FROM users",  # Syntax error
    ]

    DATABASE_SPECIFIC_QUERIES = {
        "mysql": [
            "SELECT * FROM users LIMIT 10",
            "SELECT * FROM users WHERE name REGEXP '^[A-Z]'",
            "INSERT INTO users (name) VALUES ('John') ON DUPLICATE KEY UPDATE name = VALUES(name)",
        ],
        "postgresql": [
            "SELECT * FROM users LIMIT 10 OFFSET 5",
            "SELECT name FROM users WHERE name ~ '^[A-Z]'",
            "INSERT INTO users (name) VALUES ('John') ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name",
        ],
        "sqlite": [
            "SELECT * FROM users LIMIT 10 OFFSET 5",
            "SELECT name FROM users WHERE name GLOB '[A-Z]*'",
            "INSERT OR REPLACE INTO users (name) VALUES ('John')",
        ],
    }


class PerformanceTestUtils:
    """Utilities for performance testing."""

    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """Measure function execution time."""
        import time

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    @staticmethod
    def measure_memory_usage():
        """Measure current memory usage."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    @staticmethod
    def assert_performance_benchmark(
        test_case, execution_time, max_time, operation_name
    ):
        """Assert that execution time meets performance benchmark."""
        test_case.assertLess(
            execution_time,
            max_time,
            f"{operation_name} took {execution_time:.3f}s, exceeds {max_time}s benchmark",
        )

    @staticmethod
    def generate_load_test_data(num_queries=100):
        """Generate data for load testing."""
        import random

        queries = []

        for i in range(num_queries):
            query_type = random.choice(["SELECT", "INSERT", "UPDATE", "DELETE"])
            complexity = random.randint(10, 90)

            query = MLTestDataFactory.create_test_query(
                sql_text=f"SELECT * FROM table_{i} WHERE id = {i}",
                query_type=query_type,
                complexity=complexity,
                query_hash=f"load_test_{i}",
            )
            queries.append(query)

        return queries


class ValidationUtils:
    """Utilities for data validation in tests."""

    @staticmethod
    def validate_sql_query(sql_text):
        """Basic SQL query validation."""
        if not sql_text or not sql_text.strip():
            return False, "Query is empty"

        sql_text = sql_text.strip().upper()
        valid_starters = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "WITH",
            "CREATE",
            "ALTER",
            "DROP",
        ]

        if not any(sql_text.startswith(starter) for starter in valid_starters):
            return False, "Query doesn't start with valid SQL keyword"

        return True, "Valid"

    @staticmethod
    def validate_feature_vector(features, expected_count=None):
        """Validate a feature vector."""
        if features is None:
            return False, "Features are None"

        if not isinstance(features, (list, np.ndarray)):
            return False, "Features must be list or numpy array"

        if len(features) == 0:
            return False, "Features array is empty"

        if expected_count and len(features) != expected_count:
            return False, f"Expected {expected_count} features, got {len(features)}"

        # Check for invalid values
        for i, feature in enumerate(features):
            if np.isnan(feature) or np.isinf(feature):
                return False, f"Feature {i} has invalid value: {feature}"

        return True, "Valid"

    @staticmethod
    def validate_training_data_consistency(X, y):
        """Validate training data consistency."""
        if X is None or y is None:
            return False, "Training data is None"

        if len(X) != len(y):
            return (
                False,
                f"Feature and target lengths don't match: {len(X)} != {len(y)}",
            )

        if len(X) == 0:
            return False, "Training data is empty"

        return True, "Valid"
