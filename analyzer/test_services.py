"""
Service Layer tests using TransactionTestCase for ATOMIC_REQUESTS compatibility.

Key Changes from Original:
1. Changed from TestCase to TransactionTestCase (required for ATOMIC_REQUESTS=True)
2. Added all 4 DummyCache backends to @override_settings
3. Added cache reinitialization in setUp()
4. Added proper tearDown() with manual cleanup
5. Wrapped object creation in transaction.atomic() where needed

Related Documentation:
- TESTING.md - Comprehensive testing guide
- test_integration_refactored.py - Similar pattern with detailed documentation
"""
from django.test import TransactionTestCase, override_settings
from django.contrib.auth.models import User
from django.db import transaction

from .models import Query, QueryAnalysis, UserQueryHistory, QueryFeedback
from .services import QueryAnalysisService, FeedbackService
from .services.query_analysis_service import QueryAnalysisRequest
from .services.feedback_service import FeedbackSubmission


@override_settings(
    RATELIMIT_ENABLE=False,
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'query_analysis_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'process_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'template_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        }
    }
)
class QueryAnalysisServiceTests(TransactionTestCase):
    """Tests for QueryAnalysisService"""

    def setUp(self):
        """Set up test fixtures."""
        # Reinitialize cache to use test cache backend
        from analyzer.performance import query_cache
        from django.core.cache import caches

        # Force query_cache to use test cache backend
        query_cache.cache = caches['query_analysis_cache']

        # Clear all caches
        for cache_name in ['default', 'query_analysis_cache', 'process_cache', 'template_cache']:
            try:
                caches[cache_name].clear()
            except:
                pass

        with transaction.atomic():
            self.user = User.objects.create_user(
                username='testuser',
                email='test@example.com',
                password='testpass123'
            )
        self.service = QueryAnalysisService()

    def tearDown(self):
        """Clean up test data."""
        # Manual cleanup required for TransactionTestCase
        QueryFeedback.objects.all().delete()
        UserQueryHistory.objects.all().delete()
        QueryAnalysis.objects.all().delete()
        Query.objects.all().delete()
        User.objects.all().delete()

    def test_analyze_simple_query(self):
        """Test analyzing a simple SELECT query."""
        request = QueryAnalysisRequest(
            sql_query="SELECT * FROM users",
            user=self.user,
            database_type='mysql',
            ip_address='127.0.0.1',
            user_agent='Test',
            enable_ml=False  # Disable ML for basic test
        )

        result = self.service.analyze_query_for_user(request)

        self.assertIsNotNone(result.query)
        self.assertIsNotNone(result.analysis)
        self.assertIsNotNone(result.user_history)
        self.assertIsNone(result.error)
        self.assertEqual(result.user_history.user, self.user)

    def test_analyze_invalid_query(self):
        """Test analyzing an invalid query raises ValueError."""
        request = QueryAnalysisRequest(
            sql_query="SELCT * FROM users",  # Typo
            user=self.user,
            database_type='mysql',
            ip_address='127.0.0.1',
            user_agent='Test',
            enable_ml=False
        )

        with self.assertRaises(ValueError):
            self.service.analyze_query_for_user(request)

    def test_get_analysis_by_id(self):
        """Test retrieving analysis by ID."""
        # Create an analysis first
        request = QueryAnalysisRequest(
            sql_query="SELECT id, name FROM products",
            user=self.user,
            database_type='postgresql',
            ip_address='127.0.0.1',
            user_agent='Test',
            enable_ml=False
        )

        result = self.service.analyze_query_for_user(request)
        analysis_id = result.analysis.id

        # Retrieve it
        retrieved = self.service.get_analysis_by_id(analysis_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, analysis_id)

    def test_get_nonexistent_analysis(self):
        """Test retrieving non-existent analysis returns None."""
        result = self.service.get_analysis_by_id(999999)
        self.assertIsNone(result)


@override_settings(
    RATELIMIT_ENABLE=False,
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'query_analysis_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'process_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'template_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        }
    }
)
class FeedbackServiceTests(TransactionTestCase):
    """Tests for FeedbackService"""

    def setUp(self):
        """Set up test fixtures."""
        # Reinitialize cache to use test cache backend
        from analyzer.performance import query_cache
        from django.core.cache import caches

        # Force query_cache to use test cache backend
        query_cache.cache = caches['query_analysis_cache']

        # Clear all caches
        for cache_name in ['default', 'query_analysis_cache', 'process_cache', 'template_cache']:
            try:
                caches[cache_name].clear()
            except:
                pass

        with transaction.atomic():
            self.user = User.objects.create_user(
                username='testuser',
                email='test@example.com',
                password='testpass123'
            )
        self.service = FeedbackService()

        # Create a query and analysis for testing
        with transaction.atomic():
            self.query = Query.objects.create(
                sql_text="SELECT * FROM users",
                query_hash="test_hash_12345"
            )
            self.analysis = QueryAnalysis.objects.create(
                query=self.query,
                grade='B',
                score=75.0
            )
            self.user_history = UserQueryHistory.objects.create(
                user=self.user,
                query=self.query,
                ip_address='127.0.0.1',
                user_agent='Test'
            )

    def tearDown(self):
        """Clean up test data."""
        # Manual cleanup required for TransactionTestCase
        QueryFeedback.objects.all().delete()
        UserQueryHistory.objects.all().delete()
        QueryAnalysis.objects.all().delete()
        Query.objects.all().delete()
        User.objects.all().delete()

    def test_submit_new_detailed_feedback(self):
        """Test submitting new detailed feedback."""
        submission = FeedbackSubmission(
            user=self.user,
            analysis_id=self.analysis.id,
            accuracy_rating=5,
            usefulness_rating=4,
            clarity_rating=5,
            suggestions='Great analysis!',
            would_recommend=True
        )

        result = self.service.submit_detailed_feedback(submission)

        self.assertTrue(result.success)
        self.assertTrue(result.created)
        self.assertIsNotNone(result.feedback)
        self.assertEqual(result.feedback.accuracy_rating, 5)
        self.assertEqual(result.feedback.usefulness_rating, 4)

    def test_update_existing_feedback(self):
        """Test updating existing feedback."""
        # Create initial feedback
        with transaction.atomic():
            QueryFeedback.objects.create(
                user_history=self.user_history,
                accuracy_rating=3,
                usefulness_rating=3,
                clarity_rating=3
            )

        # Update it
        submission = FeedbackSubmission(
            user=self.user,
            analysis_id=self.analysis.id,
            accuracy_rating=5,
            usefulness_rating=5,
            clarity_rating=5,
            suggestions='Updated!',
            would_recommend=True
        )

        result = self.service.submit_detailed_feedback(submission)

        self.assertTrue(result.success)
        self.assertFalse(result.created)  # Was updated, not created
        self.assertEqual(result.feedback.accuracy_rating, 5)

    def test_submit_quick_feedback(self):
        """Test submitting quick thumbs up/down feedback."""
        submission = FeedbackSubmission(
            user=self.user,
            analysis_id=self.analysis.id,
            is_helpful=True
        )

        result = self.service.submit_quick_feedback(submission)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.feedback)
        self.assertTrue(result.user_history.was_helpful)

    def test_get_feedback_for_analysis(self):
        """Test retrieving feedback for an analysis."""
        # Create feedback
        with transaction.atomic():
            QueryFeedback.objects.create(
                user_history=self.user_history,
                accuracy_rating=4,
                usefulness_rating=4,
                clarity_rating=4
            )

        # Retrieve it
        feedback = self.service.get_feedback_for_analysis(
            self.user,
            self.analysis.id
        )

        self.assertIsNotNone(feedback)
        self.assertEqual(feedback.accuracy_rating, 4)

    def test_feedback_statistics(self):
        """Test feedback statistics calculation."""
        # Create some feedback
        with transaction.atomic():
            for i in range(3):
                user_history = UserQueryHistory.objects.create(
                    user=self.user,
                    query=self.query,
                    ip_address='127.0.0.1'
                )
                QueryFeedback.objects.create(
                    user_history=user_history,
                    accuracy_rating=4,
                    usefulness_rating=5,
                    clarity_rating=4,
                    would_recommend=True
                )

        stats = self.service.get_feedback_statistics()

        self.assertGreater(stats['total_feedback'], 0)
        self.assertGreater(stats['average_accuracy'], 0)
        self.assertGreater(stats['recommendation_rate'], 0)
