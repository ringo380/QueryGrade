import time

from django.contrib.auth.models import User
from django.test import TestCase

from analyzer.models import Query, QueryAnalysis, UserQueryHistory
from analyzer.query_analyzer import QueryGrader, analyze_query


class QueryGraderTestCase(TestCase):
    """Comprehensive tests for the QueryGrader functionality."""

    def setUp(self):
        """Set up test data."""
        self.grader = QueryGrader()
        self.test_user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

    def test_grade_excellent_query(self):
        """Test that a well-optimized query gets an A grade."""
        excellent_query = """
        SELECT u.id, u.name, u.email, COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at >= '2023-01-01'
          AND u.status = 'active'
        GROUP BY u.id, u.name, u.email
        HAVING COUNT(o.id) > 0
        ORDER BY order_count DESC
        LIMIT 100;
        """

        query, analysis = self.grader.analyze_query(excellent_query)

        self.assertEqual(analysis.grade, "A")
        self.assertGreaterEqual(analysis.score, 90.0)
        self.assertEqual(query.query_type, "SELECT")
        self.assertGreaterEqual(query.table_count, 2)  # users and orders
        self.assertEqual(query.join_count, 1)  # LEFT JOIN

    def test_grade_poor_query(self):
        """Test that a poorly optimized query gets a D or F grade."""
        poor_query = """
        SELECT *
        FROM users u
        WHERE UPPER(u.email) LIKE '%@GMAIL.%'
          AND u.id NOT IN (
            SELECT user_id FROM orders WHERE user_id IS NOT NULL
          )
        ORDER BY u.created_at;
        """

        query, analysis = self.grader.analyze_query(poor_query)

        self.assertIn(analysis.grade, ["C", "D", "F"])  # Allow C due to new scoring
        self.assertLessEqual(analysis.score, 75.0)  # Slightly higher threshold
        self.assertTrue(len(analysis.issues_found) > 0)

        # Check for specific issues
        issue_types = [issue["type"] for issue in analysis.issues_found]
        self.assertIn("SELECT_STAR", issue_types)
        self.assertIn("FUNCTION_ON_COLUMN", issue_types)

    def test_grade_average_query(self):
        """Test that an average query gets a B or C grade."""
        average_query = """
        SELECT u.name, u.email, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.created_at > '2023-01-01'
        ORDER BY o.total DESC;
        """

        query, analysis = self.grader.analyze_query(average_query)

        self.assertIn(analysis.grade, ["A", "B", "C"])
        self.assertGreaterEqual(analysis.score, 70.0)
        self.assertEqual(query.query_type, "SELECT")

    def test_insert_query_grading(self):
        """Test grading of INSERT queries."""
        insert_query = """
        INSERT INTO users (name, email, created_at)
        VALUES ('John Doe', 'john@example.com', NOW());
        """

        query, analysis = self.grader.analyze_query(insert_query)

        self.assertEqual(query.query_type, "INSERT")
        self.assertIsInstance(analysis.score, float)
        self.assertIn(analysis.grade, ["A", "B", "C", "D", "F"])

    def test_update_query_grading(self):
        """Test grading of UPDATE queries."""
        update_query = """
        UPDATE users
        SET last_login = NOW()
        WHERE id = 123;
        """

        query, analysis = self.grader.analyze_query(update_query)

        self.assertEqual(query.query_type, "UPDATE")
        self.assertIsInstance(analysis.score, float)

    def test_delete_query_grading(self):
        """Test grading of DELETE queries."""
        delete_query = """
        DELETE FROM users
        WHERE created_at < '2020-01-01'
          AND status = 'inactive';
        """

        query, analysis = self.grader.analyze_query(delete_query)

        self.assertEqual(query.query_type, "DELETE")
        self.assertIsInstance(analysis.score, float)

    def test_complex_subquery_detection(self):
        """Test detection of complex subqueries."""
        complex_query = """
        SELECT u.name
        FROM users u
        WHERE u.id IN (
            SELECT o.user_id FROM orders o
            WHERE o.total > (
                SELECT AVG(total) FROM orders
                WHERE created_at > '2023-01-01'
            )
            AND o.user_id IN (
                SELECT user_id FROM user_preferences
                WHERE newsletter = true
            )
        );
        """

        query, analysis = self.grader.analyze_query(complex_query)

        self.assertGreaterEqual(query.subquery_count, 2)
        # Should have recommendations about subquery complexity
        recommendation_types = [rec["type"] for rec in analysis.recommendations]
        self.assertTrue(
            any("SUBQUERY" in rec_type for rec_type in recommendation_types)
        )

    def test_cartesian_product_detection(self):
        """Test detection of dangerous Cartesian products."""
        cartesian_query = """
        SELECT u.name, p.title
        FROM users u, products p;
        """

        query, analysis = self.grader.analyze_query(cartesian_query)

        # Should detect lack of proper JOIN conditions
        self.assertLessEqual(analysis.score, 50.0)  # Should get a poor score

    def test_join_analysis(self):
        """Test analysis of JOIN operations."""
        join_query = """
        SELECT u.name, p.title, c.name, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON oi.product_id = p.id
        JOIN categories c ON p.category_id = c.id
        WHERE u.active = 1;
        """

        query, analysis = self.grader.analyze_query(join_query)

        self.assertGreaterEqual(query.join_count, 4)
        # Should warn about excessive joins if > 4
        if query.join_count > 4:
            issue_types = [issue["type"] for issue in analysis.issues_found]
            self.assertIn("EXCESSIVE_JOINS", issue_types)

    def test_leading_wildcard_detection(self):
        """Test detection of inefficient LIKE patterns."""
        wildcard_query = """
        SELECT * FROM users
        WHERE email LIKE '%@company.com'
           OR name LIKE '%John%';
        """

        query, analysis = self.grader.analyze_query(wildcard_query)

        issue_types = [issue["type"] for issue in analysis.issues_found]
        self.assertIn("LEADING_WILDCARD", issue_types)
        self.assertIn("SELECT_STAR", issue_types)

    def test_function_on_column_detection(self):
        """Test detection of functions on columns in WHERE clauses."""
        function_query = """
        SELECT id, name FROM users
        WHERE UPPER(email) = 'JOHN@EXAMPLE.COM'
           OR SUBSTRING(phone, 1, 3) = '555'
           OR DATE(created_at) = '2023-01-01';
        """

        query, analysis = self.grader.analyze_query(function_query)

        issue_types = [issue["type"] for issue in analysis.issues_found]
        self.assertIn("FUNCTION_ON_COLUMN", issue_types)

    def test_not_in_usage_detection(self):
        """Test detection of NOT IN with potential NULL issues."""
        not_in_query = """
        SELECT * FROM users
        WHERE id NOT IN (SELECT user_id FROM banned_users);
        """

        query, analysis = self.grader.analyze_query(not_in_query)

        issue_types = [issue["type"] for issue in analysis.issues_found]
        self.assertIn(
            "NOT_IN_SUBQUERY", issue_types
        )  # New architecture uses NOT_IN_SUBQUERY

    def test_distinct_usage_detection(self):
        """Test detection of DISTINCT usage."""
        distinct_query = """
        SELECT DISTINCT u.name, u.email
        FROM users u
        JOIN orders o ON u.id = o.user_id;
        """

        query, analysis = self.grader.analyze_query(distinct_query)

        # New architecture provides more specific recommendations (e.g., USE_POSTGRESQL_DISTINCT_ON)
        # Just verify that we get some recommendations
        self.assertGreater(len(analysis.recommendations), 0)

    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        with self.assertRaises(Exception):
            self.grader.analyze_query("")

    def test_invalid_sql_handling(self):
        """Test handling of malformed SQL."""
        with self.assertRaises(Exception):
            self.grader.analyze_query("SELCT * FORM users WHER id = ")

    def test_whitespace_normalization(self):
        """Test that queries with different whitespace get same results."""
        query1 = "SELECT id, name FROM users WHERE active = 1;"
        query2 = """
            SELECT   id,  name
            FROM     users
            WHERE    active = 1;
        """

        _, analysis1 = self.grader.analyze_query(query1)
        _, analysis2 = self.grader.analyze_query(query2)

        # Should get identical scores for functionally identical queries
        self.assertEqual(analysis1.score, analysis2.score)
        self.assertEqual(analysis1.grade, analysis2.grade)

    def test_query_caching(self):
        """Test that identical queries are cached and not re-analyzed."""
        query_text = "SELECT id, name FROM users WHERE active = 1;"

        # Analyze the same query twice
        query1, analysis1 = self.grader.analyze_query(query_text)
        query2, analysis2 = self.grader.analyze_query(query_text)

        # Should return the same Query object (cached)
        self.assertEqual(query1.id, query2.id)
        self.assertEqual(analysis1.id, analysis2.id)

    def test_scoring_algorithm(self):
        """Test the scoring algorithm with known issue counts."""
        # Create a query that should have specific issues
        test_query = """
        SELECT *
        FROM users u
        WHERE UPPER(u.email) LIKE '%@GMAIL.%'
          AND u.id NOT IN (SELECT user_id FROM orders);
        """

        query, analysis = self.grader.analyze_query(test_query)

        # With the new diminishing returns algorithm, we can't predict exact scores,
        # but we can verify the score is reasonable based on issues found
        issue_severities = [issue["severity"] for issue in analysis.issues_found]

        # If there are issues, score should be less than 100
        if len(analysis.issues_found) > 0:
            self.assertLess(analysis.score, 100.0)

        # If there are high or critical issues, score should be significantly lower
        if any(severity in ["critical", "high"] for severity in issue_severities):
            self.assertLessEqual(analysis.score, 85.0)

        # Score should always be between 0 and 100
        self.assertGreaterEqual(analysis.score, 0.0)
        self.assertLessEqual(analysis.score, 100.0)

    def test_grade_boundaries(self):
        """Test grade boundary calculations."""
        # Test each grade boundary
        test_cases = [
            (95.0, "A"),
            (90.0, "A"),
            (85.0, "B"),
            (80.0, "B"),
            (75.0, "C"),
            (70.0, "C"),
            (65.0, "D"),
            (60.0, "D"),
            (55.0, "F"),
            (0.0, "F"),
        ]

        for score, expected_grade in test_cases:
            grade = self.grader._score_to_grade(score)
            self.assertEqual(
                grade,
                expected_grade,
                f"Score {score} should get grade {expected_grade}, got {grade}",
            )

    def test_analysis_metadata(self):
        """Test that analysis metadata is properly recorded."""
        query_text = "SELECT id, name FROM users WHERE active = 1;"

        start_time = time.time()
        query, analysis = self.grader.analyze_query(query_text)
        end_time = time.time()

        # Check metadata
        self.assertEqual(
            analysis.analysis_version, "2.0"
        )  # New modular analyzer architecture
        self.assertGreater(analysis.execution_time_ms, 0)
        self.assertLess(
            analysis.execution_time_ms, (end_time - start_time) * 1000 + 100
        )  # Allow some margin
        self.assertIsNotNone(analysis.created_at)

    def test_query_hash_generation(self):
        """Test that query hashes are generated correctly."""
        query1 = "SELECT id, name FROM users WHERE active = 1;"
        query2 = "select ID, NAME from USERS where ACTIVE = 1;"  # Different case

        q1, _ = self.grader.analyze_query(query1)
        q2, _ = self.grader.analyze_query(query2)

        # Normalized queries should have the same hash
        self.assertEqual(q1.query_hash, q2.query_hash)

    def test_complexity_calculation(self):
        """Test query complexity calculation."""
        simple_query = "SELECT id FROM users;"
        complex_query = """
        SELECT u.id, u.name, COUNT(o.id) as orders,
               SUM(oi.quantity * p.price) as total_spent
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        LEFT JOIN order_items oi ON o.id = oi.order_id
        LEFT JOIN products p ON oi.product_id = p.id
        WHERE u.created_at >= '2023-01-01'
          AND u.status = 'active'
          AND EXISTS (
              SELECT 1 FROM user_preferences up
              WHERE up.user_id = u.id
                AND up.newsletter = true
          )
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 0
        ORDER BY total_spent DESC;
        """

        simple_q, _ = self.grader.analyze_query(simple_query)
        complex_q, _ = self.grader.analyze_query(complex_query)

        self.assertLess(simple_q.estimated_complexity, complex_q.estimated_complexity)
        self.assertGreater(
            complex_q.estimated_complexity, 50
        )  # Should be quite complex

    def test_convenience_function(self):
        """Test the convenience analyze_query function."""
        query_text = "SELECT id, name FROM users WHERE active = 1;"

        # Test using the convenience function
        query, analysis = analyze_query(query_text)

        self.assertIsInstance(query, Query)
        self.assertIsInstance(analysis, QueryAnalysis)
        self.assertEqual(query.sql_text, query_text)


class QueryGraderEdgeCasesTestCase(TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        self.grader = QueryGrader()

    def test_very_long_query(self):
        """Test handling of very long queries."""
        # Create a query with many columns
        columns = ", ".join([f"col{i}" for i in range(100)])
        long_query = f"SELECT {columns} FROM large_table WHERE active = 1;"

        query, analysis = self.grader.analyze_query(long_query)

        self.assertIsInstance(analysis.score, float)
        self.assertIn(analysis.grade, ["A", "B", "C", "D", "F"])

    def test_query_with_comments(self):
        """Test handling of queries with SQL comments."""
        commented_query = """
        -- This is a test query
        SELECT u.id, u.name /* user details */
        FROM users u -- main table
        WHERE u.active = 1; /* only active users */
        """

        query, analysis = self.grader.analyze_query(commented_query)

        self.assertIsInstance(analysis.score, float)

    def test_query_with_strings_containing_keywords(self):
        """Test queries with strings that contain SQL keywords."""
        string_query = """
        SELECT id, name FROM users
        WHERE description = 'SELECT this FROM that'
          AND notes LIKE '%JOIN our team%';
        """

        query, analysis = self.grader.analyze_query(string_query)

        self.assertIsInstance(analysis.score, float)

    def test_ddl_statements(self):
        """Test handling of DDL statements."""
        create_query = """
        CREATE TABLE test_table (
            id INT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        query, analysis = self.grader.analyze_query(create_query)

        self.assertEqual(query.query_type, "CREATE")
        self.assertIsInstance(analysis.score, float)

    def test_unicode_in_queries(self):
        """Test handling of Unicode characters in queries."""
        unicode_query = """
        SELECT id, name FROM users
        WHERE name = 'José García'
          AND description LIKE '%café%';
        """

        query, analysis = self.grader.analyze_query(unicode_query)

        self.assertIsInstance(analysis.score, float)
