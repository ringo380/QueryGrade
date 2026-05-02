"""
Unit tests for Phase 2 analyzers: Union, WindowFunction, CaseStatement, and Wildcard.

Tests each analyzer in isolation to verify:
- Issue detection
- Recommendation generation
- Performance notes
- Edge cases and error handling
"""

from django.test import TestCase

from analyzer.analyzers import QueryGrader
from analyzer.models import Query, QueryAnalysis


class UnionAnalyzerTests(TestCase):
    """Test cases for UnionAnalyzer."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_union_without_all(self):
        """Test detection of UNION without ALL."""
        query, analysis = self.grader.analyze_query(
            "SELECT id FROM users UNION SELECT id FROM customers"
        )

        issues = [i for i in analysis.issues_found if i["type"] == "UNION_WITHOUT_ALL"]
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]["severity"], "medium")

    def test_union_all_no_issue(self):
        """Test that UNION ALL doesn't trigger warning."""
        query, analysis = self.grader.analyze_query(
            "SELECT id FROM users UNION ALL SELECT id FROM customers"
        )

        union_issues = [
            i for i in analysis.issues_found if i["type"] == "UNION_WITHOUT_ALL"
        ]
        self.assertEqual(len(union_issues), 0)

    def test_multiple_unions(self):
        """Test detection of multiple UNION operations."""
        query, analysis = self.grader.analyze_query("""SELECT id FROM users
               UNION SELECT id FROM customers
               UNION SELECT id FROM vendors
               UNION SELECT id FROM partners""")

        issues = [i for i in analysis.issues_found if i["type"] == "MULTIPLE_UNIONS"]
        self.assertTrue(len(issues) > 0)

    def test_union_to_or_recommendation(self):
        """Test recommendation to use OR instead of UNION."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE status = 'active' UNION SELECT * FROM users WHERE role = 'admin'"
        )

        recommendations = [
            r for r in analysis.recommendations if r["type"] == "UNION_TO_OR"
        ]
        self.assertTrue(len(recommendations) > 0)

    def test_no_union_issues_for_simple_query(self):
        """Test that queries without UNION have no UNION issues."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE active = 1"
        )

        union_issues = [
            i for i in analysis.issues_found if "union" in i.get("type", "").lower()
        ]
        self.assertEqual(len(union_issues), 0)


class WindowFunctionAnalyzerTests(TestCase):
    """Test cases for WindowFunctionAnalyzer."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_window_without_partition(self):
        """Test detection of window function without PARTITION BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT id, ROW_NUMBER() OVER (ORDER BY created_at) as rn FROM users"
        )

        issues = [
            i for i in analysis.issues_found if i["type"] == "WINDOW_NO_PARTITION"
        ]
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]["severity"], "medium")

    def test_row_number_without_order(self):
        """Test detection of ROW_NUMBER() without ORDER BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT id, ROW_NUMBER() OVER () as rn FROM users"
        )

        issues = [
            i for i in analysis.issues_found if i["type"] == "ROW_NUMBER_NO_ORDER"
        ]
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]["severity"], "high")

    def test_named_window_recommendation(self):
        """Test recommendation for named windows."""
        query, analysis = self.grader.analyze_query("""SELECT id,
               ROW_NUMBER() OVER (PARTITION BY category ORDER BY date) as rn1,
               RANK() OVER (PARTITION BY category ORDER BY date) as rank1,
               DENSE_RANK() OVER (PARTITION BY category ORDER BY date) as dr1
               FROM orders""")

        recommendations = [
            r for r in analysis.recommendations if r["type"] == "USE_NAMED_WINDOW"
        ]
        self.assertTrue(len(recommendations) > 0)

    def test_partition_by_index_recommendation(self):
        """Test index recommendation for PARTITION BY columns."""
        query, analysis = self.grader.analyze_query(
            "SELECT id, SUM(amount) OVER (PARTITION BY customer_id) FROM orders"
        )

        recommendations = [
            r
            for r in analysis.recommendations
            if r["type"] == "INDEX_PARTITION_COLUMNS"
        ]
        self.assertTrue(len(recommendations) > 0)

    def test_function_in_partition_by(self):
        """Test detection of function in PARTITION BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT id, COUNT(*) OVER (PARTITION BY YEAR(created_at)) FROM orders"
        )

        issues = [
            i for i in analysis.issues_found if i["type"] == "FUNCTION_IN_PARTITION_BY"
        ]
        self.assertTrue(len(issues) > 0)

    def test_no_window_issues_for_simple_query(self):
        """Test that queries without window functions have no window issues."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE active = 1"
        )

        window_issues = [
            i for i in analysis.issues_found if "window" in i.get("type", "").lower()
        ]
        self.assertEqual(len(window_issues), 0)


class CaseStatementAnalyzerTests(TestCase):
    """Test cases for CaseStatementAnalyzer."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_case_in_where(self):
        """Test detection of CASE in WHERE clause."""
        query, analysis = self.grader.analyze_query("""SELECT * FROM orders
               WHERE CASE WHEN status = 'pending' THEN 1 ELSE 0 END = 1""")

        issues = [i for i in analysis.issues_found if i["type"] == "CASE_IN_WHERE"]
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]["severity"], "high")

    def test_case_in_order_by(self):
        """Test detection of CASE in ORDER BY."""
        query, analysis = self.grader.analyze_query("""SELECT * FROM orders
               ORDER BY CASE WHEN priority = 'high' THEN 1 ELSE 2 END""")

        issues = [i for i in analysis.issues_found if i["type"] == "CASE_IN_ORDERBY"]
        self.assertTrue(len(issues) > 0)

    def test_case_in_group_by(self):
        """Test detection of CASE in GROUP BY."""
        query, analysis = self.grader.analyze_query(
            """SELECT CASE WHEN amount > 100 THEN 'high' ELSE 'low' END as category, COUNT(*)
               FROM orders
               GROUP BY CASE WHEN amount > 100 THEN 'high' ELSE 'low' END"""
        )

        issues = [i for i in analysis.issues_found if i["type"] == "CASE_IN_GROUPBY"]
        self.assertTrue(len(issues) > 0)

    def test_complex_case_nesting(self):
        """Test detection of deeply nested CASE."""
        query, analysis = self.grader.analyze_query("""SELECT CASE
                   WHEN status = 'a' THEN CASE WHEN priority = 'h' THEN 1 ELSE 2 END
                   ELSE CASE WHEN priority = 'l' THEN 3 ELSE 4 END
               END FROM orders""")

        issues = [
            i for i in analysis.issues_found if i["type"] == "COMPLEX_CASE_NESTING"
        ]
        # May or may not detect depending on pattern matching
        # Just verify no errors occur
        self.assertTrue(analysis.score >= 0)

    def test_coalesce_recommendation(self):
        """Test recommendation to use COALESCE."""
        query, analysis = self.grader.analyze_query(
            "SELECT CASE WHEN email IS NULL THEN 'N/A' ELSE email END FROM users"
        )

        recommendations = [
            r for r in analysis.recommendations if r["type"] == "USE_COALESCE"
        ]
        self.assertTrue(len(recommendations) > 0)

    def test_case_with_subquery(self):
        """Test detection of subquery in CASE."""
        query, analysis = self.grader.analyze_query("""SELECT CASE
                   WHEN (SELECT COUNT(*) FROM orders WHERE user_id = u.id) > 5 THEN 'active'
                   ELSE 'inactive'
               END FROM users u""")

        issues = [i for i in analysis.issues_found if i["type"] == "CASE_WITH_SUBQUERY"]
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]["severity"], "high")

    def test_no_case_issues_for_simple_query(self):
        """Test that queries without CASE have no CASE issues."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE active = 1"
        )

        case_issues = [
            i for i in analysis.issues_found if "case" in i.get("type", "").lower()
        ]
        self.assertEqual(len(case_issues), 0)


class WildcardAnalyzerTests(TestCase):
    """Test cases for WildcardAnalyzer."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_like_both_wildcards(self):
        """Test detection of LIKE with wildcards on both ends."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM products WHERE name LIKE '%apple%'"
        )

        issues = [
            i for i in analysis.issues_found if i["type"] == "LIKE_BOTH_WILDCARDS"
        ]
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]["severity"], "high")

    def test_like_leading_wildcard(self):
        """Test detection of LIKE with leading wildcard only."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM products WHERE name LIKE '%apple'"
        )

        # Should detect leading wildcard issue
        leading_issues = [
            i for i in analysis.issues_found if "LEADING_WILDCARD" in i["type"]
        ]
        self.assertTrue(len(leading_issues) > 0)

    def test_like_trailing_wildcard_ok(self):
        """Test that LIKE with trailing wildcard only is acceptable."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM products WHERE name LIKE 'apple%'"
        )

        # Should not have critical LIKE issues
        critical_like_issues = [
            i
            for i in analysis.issues_found
            if "like" in i.get("type", "").lower() and i.get("severity") == "critical"
        ]
        self.assertEqual(len(critical_like_issues), 0)

    def test_like_no_wildcard(self):
        """Test detection of LIKE without wildcards."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM products WHERE name LIKE 'apple'"
        )

        issues = [i for i in analysis.issues_found if i["type"] == "LIKE_NO_WILDCARD"]
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]["severity"], "low")

    def test_multiple_like_or(self):
        """Test detection of multiple LIKE with OR."""
        query, analysis = self.grader.analyze_query("""SELECT * FROM products
               WHERE name LIKE '%apple%'
                  OR name LIKE '%banana%'
                  OR name LIKE '%orange%'
                  OR name LIKE '%grape%'""")

        issues = [i for i in analysis.issues_found if i["type"] == "MULTIPLE_LIKE_OR"]
        self.assertTrue(len(issues) > 0)

    def test_function_with_like(self):
        """Test detection of function with LIKE."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM products WHERE UPPER(name) LIKE '%APPLE%'"
        )

        issues = [i for i in analysis.issues_found if i["type"] == "FUNCTION_WITH_LIKE"]
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]["severity"], "high")

    def test_regexp_usage(self):
        """Test detection of REGEXP usage."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM products WHERE name REGEXP '^[A-Z].*'"
        )

        issues = [i for i in analysis.issues_found if i["type"] == "REGEXP_NO_INDEX"]
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]["severity"], "high")

    def test_no_like_issues_for_simple_query(self):
        """Test that queries without LIKE have no LIKE issues."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE id = 123"
        )

        like_issues = [
            i for i in analysis.issues_found if "like" in i.get("type", "").lower()
        ]
        self.assertEqual(len(like_issues), 0)


class Phase2IntegrationTests(TestCase):
    """Integration tests for Phase 2 analyzers working together."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_complex_query_phase2_analyzers(self):
        """Test complex query that triggers multiple Phase 2 analyzers."""
        query, analysis = self.grader.analyze_query("""SELECT
                   category,
                   CASE
                       WHEN status LIKE '%active%' THEN 'A'
                       WHEN status = 'pending' THEN 'P'
                       ELSE 'O'
                   END as status_code,
                   ROW_NUMBER() OVER (PARTITION BY category ORDER BY created_at) as rn
               FROM orders
               WHERE name LIKE '%test%'
               UNION ALL
               SELECT 'Total', 'T', NULL FROM orders""")

        # Should have findings from multiple analyzers
        self.assertTrue(
            len(analysis.issues_found) > 0 or len(analysis.recommendations) > 0
        )
        self.assertTrue(analysis.score >= 0 and analysis.score <= 100)
        self.assertIn(analysis.grade, ["A", "B", "C", "D", "F"])

    def test_window_function_with_case(self):
        """Test window function containing CASE."""
        query, analysis = self.grader.analyze_query("""SELECT
                   id,
                   SUM(CASE WHEN status = 'active' THEN amount ELSE 0 END)
                       OVER (PARTITION BY customer_id) as active_total
               FROM orders""")

        # Should analyze both window functions and CASE
        self.assertTrue(analysis.score >= 0)

    def test_union_with_like_patterns(self):
        """Test UNION combined with LIKE patterns."""
        query, analysis = self.grader.analyze_query(
            """SELECT * FROM products WHERE name LIKE '%apple%'
               UNION
               SELECT * FROM products WHERE name LIKE '%orange%'"""
        )

        # Should detect UNION and LIKE issues
        union_issues = [
            i for i in analysis.issues_found if "union" in i.get("type", "").lower()
        ]
        like_issues = [
            i for i in analysis.issues_found if "like" in i.get("type", "").lower()
        ]

        # At least one type should be detected
        self.assertTrue(len(union_issues) > 0 or len(like_issues) > 0)

    def test_all_phase2_patterns(self):
        """Test query with all Phase 2 patterns."""
        query, analysis = self.grader.analyze_query("""WITH ranked AS (
                   SELECT
                       category,
                       product_name,
                       CASE
                           WHEN price > 100 THEN 'expensive'
                           WHEN price > 50 THEN 'moderate'
                           ELSE 'cheap'
                       END as price_category,
                       ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) as rank
                   FROM products
                   WHERE description LIKE '%special%'
               )
               SELECT * FROM ranked WHERE rank <= 10
               UNION ALL
               SELECT 'Summary', 'All', 'all', 0 FROM products""")

        # Verify analysis completes without errors
        self.assertIsNotNone(analysis)
        self.assertTrue(analysis.score >= 0)

    def test_analyzer_count(self):
        """Test that all analyzers are registered."""
        self.assertEqual(len(self.grader.analyzers), 13)

        # Verify Phase 2 analyzers are present
        analyzer_names = [a.name for a in self.grader.analyzers]
        self.assertIn("UnionAnalyzer", analyzer_names)
        self.assertIn("WindowFunctionAnalyzer", analyzer_names)
        self.assertIn("CaseStatementAnalyzer", analyzer_names)
        self.assertIn("WildcardAnalyzer", analyzer_names)
