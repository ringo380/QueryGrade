"""
Unit tests for Phase 1 analyzers: Indexing, Subquery, OrderBy, and GroupBy.

Tests each analyzer in isolation to verify:
- Issue detection
- Recommendation generation
- Performance notes
- Edge cases and error handling
"""

from django.test import TestCase
from analyzer.models import Query, QueryAnalysis
from analyzer.analyzers import QueryGrader


class IndexingAnalyzerTests(TestCase):
    """Test cases for IndexingAnalyzer."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_where_equality_index_recommendation(self):
        """Test index recommendation for WHERE equality."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE user_id = 123"
        )

        recommendations = [r for r in analysis.recommendations if r['type'] == 'INDEX_WHERE']
        self.assertTrue(len(recommendations) > 0)
        self.assertIn('index', recommendations[0]['description'].lower())

    def test_where_range_index_recommendation(self):
        """Test index recommendation for range queries."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM orders WHERE order_date > '2024-01-01'"
        )

        recommendations = [r for r in analysis.recommendations if r['type'] == 'INDEX_RANGE']
        self.assertTrue(len(recommendations) > 0)
        self.assertIn('range', recommendations[0]['description'].lower())

    def test_like_leading_wildcard_issue(self):
        """Test detection of LIKE with leading wildcard."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM products WHERE name LIKE '%apple%'"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'LIKE_LEADING_WILDCARD']
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]['severity'], 'high')

    def test_join_index_recommendation(self):
        """Test index recommendation for JOIN conditions."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id"
        )

        recommendations = [r for r in analysis.recommendations if r['type'] == 'INDEX_JOIN']
        self.assertTrue(len(recommendations) > 0)

    def test_function_on_indexed_column(self):
        """Test detection of function on column in WHERE."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE UPPER(email) = 'TEST@EXAMPLE.COM'"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'FUNCTION_ON_INDEXED_COLUMN']
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]['severity'], 'high')

    def test_composite_index_opportunity(self):
        """Test composite index recommendation for multiple AND conditions."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM orders WHERE status = 'active' AND customer_id = 123 AND total > 100"
        )

        recommendations = [r for r in analysis.recommendations if r['type'] == 'COMPOSITE_INDEX']
        self.assertTrue(len(recommendations) > 0)

    def test_order_by_index_recommendation(self):
        """Test index recommendation for ORDER BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM products ORDER BY created_at DESC"
        )

        recommendations = [r for r in analysis.recommendations if r['type'] == 'INDEX_ORDER_BY']
        self.assertTrue(len(recommendations) > 0)

    def test_no_indexing_issues_for_simple_query(self):
        """Test that simple queries without WHERE/JOIN get minimal indexing recommendations."""
        query, analysis = self.grader.analyze_query(
            "SELECT id, name FROM users LIMIT 10"
        )

        # Should have minimal or no critical indexing issues
        critical_issues = [i for i in analysis.issues_found if i.get('severity') == 'critical']
        self.assertEqual(len(critical_issues), 0)


class SubqueryAnalyzerTests(TestCase):
    """Test cases for SubqueryAnalyzer."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_correlated_subquery_detection(self):
        """Test detection of correlated subquery."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE active = 1)"
        )

        # Should have recommendations about IN subquery
        recommendations = [r for r in analysis.recommendations if 'subquery' in r.get('description', '').lower()]
        self.assertTrue(len(recommendations) > 0)

    def test_scalar_subquery_in_select(self):
        """Test detection of scalar subquery in SELECT clause."""
        query, analysis = self.grader.analyze_query(
            "SELECT id, (SELECT COUNT(*) FROM orders WHERE customer_id = c.id) as order_count FROM customers c"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'SCALAR_SUBQUERY_IN_SELECT']
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]['severity'], 'high')

    def test_not_in_subquery_issue(self):
        """Test detection of NOT IN subquery."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE id NOT IN (SELECT user_id FROM blocked_users)"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'NOT_IN_SUBQUERY']
        self.assertTrue(len(issues) > 0)

    def test_deep_subquery_nesting(self):
        """Test detection of deeply nested subqueries."""
        query, analysis = self.grader.analyze_query(
            """SELECT * FROM orders WHERE customer_id IN
               (SELECT id FROM customers WHERE region_id IN
                (SELECT id FROM regions WHERE country_id IN
                 (SELECT id FROM countries WHERE active = 1)))"""
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'DEEP_SUBQUERY_NESTING']
        self.assertTrue(len(issues) > 0)

    def test_cte_opportunity_recommendation(self):
        """Test CTE recommendation for multiple subqueries."""
        query, analysis = self.grader.analyze_query(
            """SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers)
               AND product_id IN (SELECT id FROM products)"""
        )

        recommendations = [r for r in analysis.recommendations if r['type'] == 'USE_CTE']
        self.assertTrue(len(recommendations) > 0)

    def test_exists_recommendation(self):
        """Test EXISTS recommendation."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM orders WHERE EXISTS (SELECT 1 FROM customers WHERE id = orders.customer_id)"
        )

        recommendations = [r for r in analysis.recommendations if 'EXISTS' in r.get('type', '')]
        self.assertTrue(len(recommendations) > 0)

    def test_no_subquery_issues_for_simple_query(self):
        """Test that simple queries without subqueries have no subquery issues."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE active = 1"
        )

        subquery_issues = [i for i in analysis.issues_found if 'subquery' in i.get('type', '').lower()]
        self.assertEqual(len(subquery_issues), 0)


class OrderByAnalyzerTests(TestCase):
    """Test cases for OrderByAnalyzer."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_order_by_without_limit(self):
        """Test detection of ORDER BY without LIMIT."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users ORDER BY created_at DESC"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'ORDER_BY_NO_LIMIT']
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]['severity'], 'medium')

    def test_function_in_order_by(self):
        """Test detection of function in ORDER BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users ORDER BY UPPER(username)"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'FUNCTION_IN_ORDER_BY']
        self.assertTrue(len(issues) > 0)

    def test_expression_in_order_by(self):
        """Test detection of expression in ORDER BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM products ORDER BY price * quantity DESC"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'EXPRESSION_IN_ORDER_BY']
        self.assertTrue(len(issues) > 0)

    def test_multi_column_order_by(self):
        """Test recommendation for multi-column ORDER BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM orders ORDER BY customer_id, order_date, status"
        )

        recommendations = [r for r in analysis.recommendations if r['type'] == 'MULTI_COLUMN_INDEX']
        self.assertTrue(len(recommendations) > 0)

    def test_mixed_sort_order(self):
        """Test detection of mixed ASC/DESC in ORDER BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users ORDER BY last_name ASC, created_at DESC"
        )

        recommendations = [r for r in analysis.recommendations if r['type'] == 'MIXED_SORT_ORDER']
        self.assertTrue(len(recommendations) > 0)

    def test_order_by_random(self):
        """Test detection of ORDER BY RAND()."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM products ORDER BY RAND() LIMIT 10"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'ORDER_BY_RANDOM']
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]['severity'], 'high')

    def test_order_by_with_limit_no_issue(self):
        """Test that ORDER BY with LIMIT doesn't trigger no-limit issue."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users ORDER BY created_at DESC LIMIT 10"
        )

        no_limit_issues = [i for i in analysis.issues_found if i['type'] == 'ORDER_BY_NO_LIMIT']
        self.assertEqual(len(no_limit_issues), 0)

    def test_no_order_by_issues_for_simple_query(self):
        """Test that queries without ORDER BY have no ORDER BY issues."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE active = 1"
        )

        order_issues = [i for i in analysis.issues_found if 'order' in i.get('type', '').lower()]
        self.assertEqual(len(order_issues), 0)


class GroupByAnalyzerTests(TestCase):
    """Test cases for GroupByAnalyzer."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_group_by_index_recommendation(self):
        """Test index recommendation for GROUP BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id"
        )

        recommendations = [r for r in analysis.recommendations if r['type'] == 'INDEX_GROUP_BY']
        self.assertTrue(len(recommendations) > 0)

    def test_function_in_group_by(self):
        """Test detection of function in GROUP BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT YEAR(order_date), COUNT(*) FROM orders GROUP BY YEAR(order_date)"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'FUNCTION_IN_GROUP_BY']
        self.assertTrue(len(issues) > 0)

    def test_expression_in_group_by(self):
        """Test detection of expression in GROUP BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT price * quantity, COUNT(*) FROM orders GROUP BY price * quantity"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'EXPRESSION_IN_GROUP_BY']
        self.assertTrue(len(issues) > 0)

    def test_many_group_columns(self):
        """Test detection of many columns in GROUP BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT col1, col2, col3, col4, col5, COUNT(*) FROM table1 GROUP BY col1, col2, col3, col4, col5"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'MANY_GROUP_COLUMNS']
        self.assertTrue(len(issues) > 0)

    def test_having_without_aggregate(self):
        """Test detection of HAVING without aggregate function."""
        query, analysis = self.grader.analyze_query(
            "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING customer_id > 100"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'HAVING_WITHOUT_AGGREGATE']
        self.assertTrue(len(issues) > 0)

    def test_distinct_in_aggregate(self):
        """Test detection of DISTINCT in aggregate function."""
        query, analysis = self.grader.analyze_query(
            "SELECT COUNT(DISTINCT customer_id) FROM orders"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'DISTINCT_IN_AGGREGATE']
        self.assertTrue(len(issues) > 0)

    def test_distinct_with_group_by(self):
        """Test detection of DISTINCT with GROUP BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT DISTINCT customer_id FROM orders GROUP BY customer_id"
        )

        issues = [i for i in analysis.issues_found if i['type'] == 'DISTINCT_WITH_GROUP_BY']
        self.assertTrue(len(issues) > 0)

    def test_composite_group_index_recommendation(self):
        """Test composite index recommendation for multi-column GROUP BY."""
        query, analysis = self.grader.analyze_query(
            "SELECT customer_id, product_id, status, COUNT(*) FROM orders GROUP BY customer_id, product_id, status"
        )

        recommendations = [r for r in analysis.recommendations if r['type'] == 'COMPOSITE_GROUP_INDEX']
        self.assertTrue(len(recommendations) > 0)

    def test_no_group_by_issues_for_simple_query(self):
        """Test that queries without GROUP BY have no GROUP BY issues."""
        query, analysis = self.grader.analyze_query(
            "SELECT * FROM users WHERE active = 1"
        )

        group_issues = [i for i in analysis.issues_found if 'group' in i.get('type', '').lower()]
        self.assertEqual(len(group_issues), 0)


class IntegrationTests(TestCase):
    """Integration tests for multiple analyzers working together."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_complex_query_all_analyzers(self):
        """Test complex query that triggers multiple analyzers."""
        query, analysis = self.grader.analyze_query(
            """SELECT customer_id, YEAR(order_date), COUNT(DISTINCT product_id)
               FROM orders
               WHERE status IN (SELECT status FROM valid_statuses)
               GROUP BY customer_id, YEAR(order_date)
               HAVING COUNT(*) > 5
               ORDER BY COUNT(*) DESC"""
        )

        # Should have findings from multiple analyzers
        self.assertTrue(len(analysis.issues_found) > 0 or len(analysis.recommendations) > 0)
        self.assertTrue(analysis.score >= 0 and analysis.score <= 100)
        self.assertIn(analysis.grade, ['A', 'B', 'C', 'D', 'F'])

    def test_grade_calculation_with_multiple_issues(self):
        """Test that grade is calculated correctly with multiple issues."""
        query, analysis = self.grader.analyze_query(
            """SELECT * FROM orders
               WHERE UPPER(status) = 'ACTIVE'
               AND customer_id IN (SELECT id FROM customers WHERE region LIKE '%US%')
               ORDER BY RAND()"""
        )

        # Should have multiple issues of varying severity
        self.assertTrue(len(analysis.issues_found) > 0)
        # Grade should reflect the issues
        self.assertLess(analysis.score, 90)

    def test_optimal_query_good_grade(self):
        """Test that an optimal query gets a good grade."""
        query, analysis = self.grader.analyze_query(
            "SELECT id, name, email FROM users WHERE active = 1 ORDER BY id LIMIT 10"
        )

        # Should have few or no critical issues
        critical_issues = [i for i in analysis.issues_found if i.get('severity') == 'critical']
        self.assertEqual(len(critical_issues), 0)
        # Grade should be good
        self.assertGreaterEqual(analysis.score, 70)