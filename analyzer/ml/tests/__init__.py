"""
ML Tests Package

Comprehensive test suite for the QueryGrade machine learning components.

This package includes tests for:
- Feature extraction from SQL queries
- Feedback collection and aggregation
- Hybrid grading system (rule-based + ML)
- ML model training and deployment
- End-to-end ML workflow integration
- Performance and reliability testing

Test Modules:
- test_feature_extractor: Tests for SQL query feature extraction
- test_feedback_collector: Tests for user feedback collection and processing
- test_hybrid_grader: Tests for the hybrid grading system
- test_ml_integration: Integration tests for ML components
- test_utils: Utilities and helpers for ML testing

Usage:
    python manage.py test analyzer.ml.tests
    python manage.py test analyzer.ml.tests.test_feature_extractor
    python manage.py test analyzer.ml.tests.test_feedback_collector
    python manage.py test analyzer.ml.tests.test_hybrid_grader
    python manage.py test analyzer.ml.tests.test_ml_integration
"""

# Test discovery imports
from .test_feature_extractor import *
from .test_feedback_collector import *
from .test_hybrid_grader import *
from .test_ml_integration import *
from .test_utils import *

__all__ = [
    "FeatureExtractorTestCase",
    "FeatureExtractorIntegrationTestCase",
    "FeedbackCollectorTestCase",
    "FeedbackCollectorIntegrationTestCase",
    "HybridQueryGraderTestCase",
    "HybridQueryGraderIntegrationTestCase",
    "MLIntegrationTestCase",
    "MLSystemPerformanceTestCase",
    "MLTestDataFactory",
    "MLTestAssertions",
    "MockMLComponents",
    "TestQuerySamples",
    "PerformanceTestUtils",
    "ValidationUtils",
]
