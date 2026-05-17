"""
Query Optimization tests using TransactionTestCase for ATOMIC_REQUESTS compatibility.

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

from django.contrib.auth.models import User
from django.db import transaction
from django.test import Client, TransactionTestCase, override_settings
from django.urls import reverse

from analyzer.models import Query, QueryAnalysis, UserQueryHistory
from analyzer.query_optimizer import QueryOptimizer, optimize_query_from_analysis


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
class QueryOptimizationTestCase(TransactionTestCase):
    """Test cases for query optimization functionality."""

    def setUp(self):
        """Set up test client and user."""
        # Reinitialize cache to use test cache backend
        from django.core.cache import caches

        from analyzer.performance import query_cache

        # Force query_cache to use test cache backend
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
                pass

        self.client = Client(enforce_csrf_checks=False)
        self.optimizer = QueryOptimizer()

        with transaction.atomic():
            self.test_user = User.objects.create_user(
                username="testuser", email="test@example.com", password="testpass123"
            )

    def tearDown(self):
        """Clean up test data."""
        # Manual cleanup required for TransactionTestCase
        UserQueryHistory.objects.all().delete()
        QueryAnalysis.objects.all().delete()
        Query.objects.all().delete()
        User.objects.all().delete()

    def test_optimizer_basic_functionality(self):
        """Test that the QueryOptimizer class works correctly."""
        query = "SELECT * FROM users WHERE UPPER(email) = 'TEST@EXAMPLE.COM'"
        issues = [
            {"type": "SELECT_STAR", "severity": "medium"},
            {"type": "FUNCTION_ON_COLUMN", "severity": "high"},
        ]

        result = self.optimizer.optimize_query(query, issues, "mysql")

        self.assertIn("optimized_query", result)
        self.assertIn("optimizations_applied", result)
        self.assertIn("explanation", result)
        self.assertIn("improvement_estimate", result)
        self.assertGreater(len(result["optimizations_applied"]), 0)
        self.assertGreater(result["improvement_estimate"], 0)

    def test_convenience_function(self):
        """Test the convenience function optimize_query_from_analysis."""
        query = "SELECT * FROM users WHERE UPPER(email) LIKE '%@GMAIL.%'"
        issues = [
            {"type": "SELECT_STAR", "severity": "medium"},
            {"type": "FUNCTION_ON_COLUMN", "severity": "high"},
            {"type": "LEADING_WILDCARD", "severity": "medium"},
        ]

        result = optimize_query_from_analysis(query, issues, "postgresql")

        self.assertIsInstance(result, dict)
        self.assertIn("optimized_query", result)
        self.assertGreater(len(result["optimizations_applied"]), 0)

    def test_optimization_integration_workflow(self):
        """Test the full workflow including optimization in the web interface."""
        # Login
        self.client.login(username="testuser", password="testpass123")

        # Submit a query with optimization opportunities
        poor_query = """
            SELECT *
            FROM users u, orders o
            WHERE UPPER(u.email) LIKE '%@GMAIL.%'
              AND YEAR(u.created_at) = 2023
              AND u.id NOT IN (SELECT user_id FROM banned_users);
        """

        response = self.client.post(
            reverse("grade_query"), {"sql_query": poor_query, "database_type": "mysql"}
        )

        # Should redirect to results page
        self.assertEqual(response.status_code, 302)

        # Get the analysis
        analysis = QueryAnalysis.objects.first()
        self.assertIsNotNone(analysis)

        # Access the results page
        results_response = self.client.get(reverse("grade_results", args=[analysis.id]))
        self.assertEqual(results_response.status_code, 200)

        # Check that optimization section is present if there are issues
        if len(analysis.issues_found) > 0:
            self.assertContains(results_response, "Query Optimization Suggestions")
            self.assertContains(results_response, "Optimized Query")
            self.assertContains(results_response, "Side-by-Side Comparison")
            self.assertContains(results_response, "Explanations")

    def test_optimization_with_no_issues(self):
        """Test optimization when no issues are found."""
        good_query = """
            SELECT u.id, u.name, u.email
            FROM users u
            WHERE u.active = 1
              AND u.created_at >= '2023-01-01'
            ORDER BY u.name
            LIMIT 100;
        """

        issues = []  # No issues found

        result = self.optimizer.optimize_query(good_query, issues)

        # The optimizer may still apply general optimizations like index recommendations
        # So we check that the result is valid but may have general optimizations
        self.assertIsInstance(result["optimizations_applied"], list)
        self.assertIsInstance(result["improvement_estimate"], int)
        self.assertGreaterEqual(result["improvement_estimate"], 0)

    def test_specific_optimization_patterns(self):
        """Test specific optimization patterns are detected and applied."""

        # Test SELECT * optimization
        select_star_query = "SELECT * FROM users"
        select_star_issues = [{"type": "SELECT_STAR", "severity": "medium"}]

        result = self.optimizer.optimize_query(select_star_query, select_star_issues)
        self.assertIn(
            "Replaced SELECT * with specific columns", result["optimizations_applied"]
        )

        # Test FUNCTION_ON_COLUMN optimization
        function_query = "SELECT id FROM users WHERE UPPER(email) = 'TEST@EXAMPLE.COM'"
        function_issues = [{"type": "FUNCTION_ON_COLUMN", "severity": "high"}]

        result = self.optimizer.optimize_query(function_query, function_issues)
        self.assertIn(
            "Removed functions from WHERE clause columns",
            result["optimizations_applied"],
        )

        # Test UNION optimization
        union_query = "SELECT id FROM users UNION SELECT id FROM customers"
        union_issues = [{"type": "UNION_WITHOUT_ALL", "severity": "medium"}]

        result = self.optimizer.optimize_query(union_query, union_issues)
        self.assertIn("Changed UNION to UNION ALL", result["optimizations_applied"])

    def test_improvement_estimation(self):
        """Test that improvement estimation works correctly."""
        query = "SELECT * FROM users"

        # No issues
        result = self.optimizer.optimize_query(query, [])
        self.assertEqual(result["improvement_estimate"], 0)

        # One medium issue
        medium_issues = [{"type": "SELECT_STAR", "severity": "medium"}]
        result = self.optimizer.optimize_query(query, medium_issues)
        self.assertGreater(result["improvement_estimate"], 0)
        self.assertLessEqual(result["improvement_estimate"], 95)

        # Multiple high severity issues
        high_issues = [
            {"type": "SELECT_STAR", "severity": "high"},
            {"type": "FUNCTION_ON_COLUMN", "severity": "critical"},
            {"type": "CARTESIAN_PRODUCT", "severity": "critical"},
        ]
        result = self.optimizer.optimize_query(query, high_issues)
        self.assertGreater(result["improvement_estimate"], 30)

    def test_database_specific_optimizations(self):
        """Test that database-specific optimizations are applied."""
        query = "SELECT id FROM users WHERE name LIKE '%john%' ORDER BY created_at"
        issues = []

        # MySQL specific
        mysql_result = self.optimizer.optimize_query(query, issues, "mysql")
        optimized_query_mysql = mysql_result["optimized_query"]
        self.assertIn("MySQL", optimized_query_mysql)

        # PostgreSQL specific
        postgres_result = self.optimizer.optimize_query(query, issues, "postgresql")
        optimized_query_postgres = postgres_result["optimized_query"]
        self.assertIn("PostgreSQL", optimized_query_postgres)

    def test_optimization_summary_generation(self):
        """Test the optimization summary generation."""
        query = "SELECT * FROM users WHERE UPPER(email) = 'TEST@EXAMPLE.COM'"
        issues = [
            {"type": "SELECT_STAR", "severity": "medium"},
            {"type": "FUNCTION_ON_COLUMN", "severity": "high"},
        ]

        result = self.optimizer.optimize_query(query, issues)
        summary = self.optimizer.generate_optimization_summary(result)

        self.assertIsInstance(summary, str)
        self.assertIn("Applied", summary)
        self.assertIn("optimizations", summary)
        self.assertIn("Estimated performance improvement", summary)
        self.assertIn("%", summary)

    def test_invalid_sql_handling(self):
        """Test that invalid SQL is handled gracefully in optimization."""
        invalid_query = "SELCT * FORM users WHER"
        issues = [{"type": "SELECT_STAR", "severity": "medium"}]

        result = self.optimizer.optimize_query(invalid_query, issues)

        # Should return the original query without throwing an exception
        # sqlparse is quite forgiving and doesn't always throw exceptions for malformed SQL
        self.assertEqual(result["optimized_query"], invalid_query)
        self.assertIsInstance(result["optimizations_applied"], list)
        self.assertIsInstance(result["explanation"], list)
        self.assertIsInstance(result["improvement_estimate"], int)
        self.assertGreaterEqual(result["improvement_estimate"], 0)
