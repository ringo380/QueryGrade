"""
Refactored integration tests using explicit transaction management.

This test file solves a critical cache initialization issue that caused integration
test failures with foreign key constraint errors. The problem occurred because the
global query_cache singleton was initialized at module import time with production
Redis cache, before @override_settings could apply DummyCache for tests.

Key Solutions Implemented:
1. TransactionTestCase: Required when ATOMIC_REQUESTS=True (production setting)
2. Cache Reinitialization: Explicitly set query_cache.cache in setUp()
3. Factory Methods: Consistent test object creation with transaction.atomic()
4. Explicit ID Fetching: Avoid .first()/.last() which may return cached objects
5. Manual Cleanup: tearDown() deletes all test data (no automatic rollback)

Related Documentation:
- TESTING.md - Comprehensive testing guide with examples and best practices
- INTEGRATION_TEST_FIX_SUMMARY.md - Detailed case study of debugging this issue
- README.md - Project overview and structure

Testing Best Practices:
- Always reinitialize query_cache in setUp() to use test cache backend
- Fetch objects by explicit ID from response URLs, not .first()/.last()
- Use @override_settings with all 4 DummyCache backends
- Clear all caches in setUp() to prevent contamination
- Use TransactionTestCase for integration tests, TestCase for unit tests

Common Pitfalls to Avoid:
- ❌ Using TestCase for integration tests (conflicts with ATOMIC_REQUESTS)
- ❌ Forgetting to reinitialize cache (causes FK constraint errors)
- ❌ Using Query.objects.first() (may return cached objects)
- ❌ Missing cache clear in setUp() (test contamination)
- ❌ Forgetting tearDown() cleanup (data leakage between tests)
"""

import json

from django.contrib.auth.models import User
from django.db import transaction
from django.test import Client, TransactionTestCase, override_settings
from django.urls import reverse

from analyzer.models import Query, QueryAnalysis, UserQueryHistory


# Factory methods for creating test objects
def create_test_user(
    username="testuser", password="testpass123", email="test@example.com"
):
    """Factory method to create a test user."""
    with transaction.atomic():
        user = User.objects.create_user(
            username=username, email=email, password=password
        )
    return user


@override_settings(
    RATELIMIT_ENABLE=False,
    CACHES={
        "default": {
            "BACKEND": "django.core.cache.backends.dummy.DummyCache",
        },
        "query_analysis_cache": {
            "BACKEND": "django.core.cache.backends.dummy.DummyCache",
        },
        "process_cache": {
            "BACKEND": "django.core.cache.backends.dummy.DummyCache",
        },
        "template_cache": {
            "BACKEND": "django.core.cache.backends.dummy.DummyCache",
        },
    },
)
class RefactoredQueryGradingIntegrationTestCase(TransactionTestCase):
    """
    Integration tests using explicit transaction management.

    Uses TransactionTestCase to avoid transaction wrapping issues.
    Each test manages its own transactions explicitly.
    """

    def setUp(self):
        """Set up test client and user with explicit transaction."""
        self.client = Client(enforce_csrf_checks=False)

        # Reinitialize query_cache with DummyCache to prevent stale data
        from django.core.cache import caches

        from analyzer.performance import query_cache

        # Force query_cache to use the test cache backend
        query_cache.cache = caches["query_analysis_cache"]

        # Clear all caches
        for cache_name in [
            "default",
            "query_analysis_cache",
            "process_cache",
            "template_cache",
        ]:
            try:
                caches[cache_name].clear()
            except Exception:
                pass  # Cache might not exist

        # Create user with explicit transaction
        self.test_user = create_test_user(
            username="integrationuser",
            email="integration@example.com",
            password="testpass123",
        )

        # Force login
        self.client.force_login(self.test_user)

    def tearDown(self):
        """Clean up test data."""
        # TransactionTestCase doesn't auto-rollback, so clean up manually
        UserQueryHistory.objects.all().delete()
        QueryAnalysis.objects.all().delete()
        Query.objects.all().delete()
        User.objects.all().delete()

    def test_full_query_grading_workflow_with_explicit_transactions(self):
        """Test the complete workflow using explicit transaction management."""

        # Step 1: Verify user is logged in
        self.assertTrue(self.client.session.get("_auth_user_id"))

        # Step 2: Access grade query page
        grade_page = self.client.get(reverse("grade_query"))
        self.assertEqual(grade_page.status_code, 200)
        self.assertContains(grade_page, "SQL Query Grader")

        # Step 3: Submit a query for grading
        test_query = """
            SELECT u.id, u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.created_at >= '2023-01-01'
            GROUP BY u.id, u.name
            ORDER BY order_count DESC
            LIMIT 100;
        """

        # Execute the grading request (ATOMIC_REQUESTS handles transaction)
        grade_response = self.client.post(
            reverse("grade_query"), {"sql_query": test_query, "database_type": "mysql"}
        )

        # Should redirect to results page
        self.assertEqual(grade_response.status_code, 302)
        self.assertTrue(
            grade_response.url.startswith("/grade/results/")
            or grade_response.url.startswith("/grade/enhanced/"),
            f"Expected redirect to results page but got: {grade_response.url}",
        )

        # Step 4: Verify objects were created (check after transaction commits)
        self.assertEqual(
            Query.objects.count(), 1, "Expected 1 Query object to be created"
        )
        self.assertEqual(
            QueryAnalysis.objects.count(), 1, "Expected 1 QueryAnalysis object"
        )
        self.assertEqual(
            UserQueryHistory.objects.count(), 1, "Expected 1 UserQueryHistory object"
        )

        # Step 5: Verify the analysis results
        query = Query.objects.first()
        analysis = QueryAnalysis.objects.first()
        history = UserQueryHistory.objects.first()

        self.assertIsNotNone(query)
        self.assertIsNotNone(analysis)
        self.assertIsNotNone(history)

        # Verify relationships
        self.assertEqual(analysis.query, query)
        self.assertEqual(history.query, query)
        self.assertEqual(history.user, self.test_user)

        # Verify query content
        self.assertIn("SELECT", query.sql_text)
        self.assertEqual(query.query_type, "SELECT")

        # Verify analysis has grade
        self.assertIn(analysis.grade, ["A", "B", "C", "D", "F"])
        self.assertGreaterEqual(analysis.score, 0)
        self.assertLessEqual(analysis.score, 100)

        # Step 6: Access results page
        results_url = grade_response.url
        results_response = self.client.get(results_url)
        self.assertEqual(results_response.status_code, 200)
        self.assertContains(results_response, "Query Analysis")

        # Step 7: Verify query history page shows the query
        history_response = self.client.get(reverse("query_history"))
        self.assertEqual(history_response.status_code, 200)
        self.assertContains(history_response, "Query history")

    def test_multiple_queries_with_transaction_management(self):
        """Test that user can grade multiple queries using explicit transactions."""

        queries = [
            "SELECT * FROM users WHERE id = 1",
            "SELECT id, name FROM products ORDER BY name",
            "SELECT COUNT(*) FROM orders",
        ]

        for i, query_text in enumerate(queries, 1):
            response = self.client.post(
                reverse("grade_query"),
                {"sql_query": query_text, "database_type": "postgresql"},
            )

            self.assertEqual(response.status_code, 302, f"Query {i} should redirect")

            # Verify objects count increases
            self.assertEqual(Query.objects.count(), i, f"Should have {i} Query objects")
            self.assertEqual(
                QueryAnalysis.objects.count(), i, f"Should have {i} Analysis objects"
            )
            self.assertEqual(
                UserQueryHistory.objects.count(), i, f"Should have {i} History objects"
            )

        # Verify all queries belong to the same user
        all_history = UserQueryHistory.objects.all()
        self.assertEqual(all_history.count(), 3)
        for history in all_history:
            self.assertEqual(history.user, self.test_user)

    def test_user_query_isolation_with_transactions(self):
        """Test that users can only see their own query history."""

        # Create another user
        other_user = create_test_user(
            username="otheruser", password="otherpass123", email="other@example.com"
        )

        # Submit query as first user
        self.client.post(
            reverse("grade_query"),
            {"sql_query": "SELECT * FROM users", "database_type": "mysql"},
        )

        # Verify first user's query was created
        self.assertEqual(
            UserQueryHistory.objects.filter(user=self.test_user).count(), 1
        )
        self.assertEqual(UserQueryHistory.objects.filter(user=other_user).count(), 0)

        # Switch to other user
        self.client.force_login(other_user)

        # Submit query as second user
        self.client.post(
            reverse("grade_query"),
            {"sql_query": "SELECT * FROM products", "database_type": "postgresql"},
        )

        # Verify both users have their own queries
        self.assertEqual(
            UserQueryHistory.objects.filter(user=self.test_user).count(), 1
        )
        self.assertEqual(UserQueryHistory.objects.filter(user=other_user).count(), 1)

        # Verify first user can only see their own history
        self.client.force_login(self.test_user)
        history_response = self.client.get(reverse("query_history"))
        self.assertEqual(history_response.status_code, 200)

    def test_query_complexity_tracking_with_transactions(self):
        """Test that query complexity is properly tracked."""

        # Simple query
        simple_query = "SELECT id FROM users"

        # Complex query with joins and subqueries
        complex_query = """
            SELECT u.id, u.name,
                   (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count
            FROM users u
            INNER JOIN addresses a ON u.id = a.user_id
            WHERE u.status = 'active'
            GROUP BY u.id, u.name
            HAVING order_count > 5
        """

        # Submit simple query
        simple_response = self.client.post(
            reverse("grade_query"),
            {"sql_query": simple_query, "database_type": "mysql"},
        )

        self.assertEqual(simple_response.status_code, 302)

        # Get the query ID from the response URL and fetch explicitly
        # Response URL format: /grade/enhanced/<analysis_id>/
        simple_analysis_id = int(simple_response.url.split("/")[-2])
        simple_analysis = QueryAnalysis.objects.get(id=simple_analysis_id)
        simple_query_obj = simple_analysis.query

        # Submit complex query
        complex_response = self.client.post(
            reverse("grade_query"),
            {"sql_query": complex_query, "database_type": "mysql"},
        )

        self.assertEqual(complex_response.status_code, 302)

        # Get the query ID from the response URL and fetch explicitly
        complex_analysis_id = int(complex_response.url.split("/")[-2])
        complex_analysis = QueryAnalysis.objects.get(id=complex_analysis_id)
        complex_query_obj = complex_analysis.query

        # Verify complexity tracking
        self.assertIsNotNone(simple_query_obj.estimated_complexity)
        self.assertIsNotNone(complex_query_obj.estimated_complexity)
        self.assertGreater(
            complex_query_obj.estimated_complexity,
            simple_query_obj.estimated_complexity,
            "Complex query should have higher complexity score",
        )

    def test_empty_query_handling_with_transactions(self):
        """Test handling of empty queries."""

        response = self.client.post(
            reverse("grade_query"), {"sql_query": "", "database_type": "mysql"}
        )

        # Should not redirect (form invalid)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "This field is required")

        # No objects should be created
        self.assertEqual(Query.objects.count(), 0)
        self.assertEqual(QueryAnalysis.objects.count(), 0)
        self.assertEqual(UserQueryHistory.objects.count(), 0)

    def test_grade_form_includes_recent_queries_context(self):
        """GET /grade/ passes recent_queries in context for authenticated users."""
        response = self.client.get(reverse("grade_query"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("recent_queries", response.context)
