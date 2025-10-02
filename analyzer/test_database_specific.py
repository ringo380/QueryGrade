from django.test import TestCase
from unittest import skip
from analyzer.query_analyzer import QueryGrader, analyze_query


class DatabaseSpecificTestCase(TestCase):
    """Test database-specific SQL analysis and recommendations."""

    def setUp(self):
        """Set up test data."""
        self.grader = QueryGrader()

    def test_mysql_syntax_detection(self):
        """Test MySQL-specific syntax issue detection."""
        # Test SQL Server TOP syntax in MySQL context
        mysql_query_with_top = """
            SELECT TOP 10 u.name, u.email
            FROM users u
            ORDER BY u.created_at DESC;
        """

        query, analysis = analyze_query(mysql_query_with_top, 'mysql')

        # Should detect MySQL syntax error
        issue_types = [issue['type'] for issue in analysis.issues_found]
        self.assertIn('MYSQL_SYNTAX_ERROR', issue_types)

        # Should provide MySQL-specific recommendation
        recommendation_types = [rec['type'] for rec in analysis.recommendations]
        self.assertIn('USE_MYSQL_LIMIT', recommendation_types)

    @skip("Oracle analyzer not yet implemented - marked as Phase 3-4 future work")
    def test_oracle_syntax_detection(self):
        """Test Oracle-specific syntax issue detection."""
        # Test LIMIT syntax in Oracle context
        oracle_query_with_limit = """
            SELECT u.name, u.email
            FROM users u
            ORDER BY u.created_at DESC
            LIMIT 10;
        """

        query, analysis = analyze_query(oracle_query_with_limit, 'oracle')

        # Should detect Oracle syntax error
        issue_types = [issue['type'] for issue in analysis.issues_found]
        self.assertIn('ORACLE_SYNTAX_ERROR', issue_types)

        # Should provide Oracle-specific recommendation
        recommendation_types = [rec['type'] for rec in analysis.recommendations]
        self.assertIn('USE_ORACLE_ROWNUM', recommendation_types)

        # Grade should be lower due to syntax error
        self.assertLess(analysis.score, 90.0)

    @skip("SQL Server analyzer not yet implemented - marked as Phase 3-4 future work")
    def test_sqlserver_syntax_detection(self):
        """Test SQL Server-specific syntax issue detection."""
        # Test LIMIT syntax in SQL Server context
        sqlserver_query_with_limit = """
            SELECT u.name, u.email
            FROM users u
            ORDER BY u.created_at DESC
            LIMIT 10;
        """

        query, analysis = analyze_query(sqlserver_query_with_limit, 'sqlserver')

        # Should detect SQL Server syntax error
        issue_types = [issue['type'] for issue in analysis.issues_found]
        self.assertIn('SQLSERVER_SYNTAX_ERROR', issue_types)

        # Should provide SQL Server-specific recommendation
        recommendation_types = [rec['type'] for rec in analysis.recommendations]
        self.assertIn('USE_SQLSERVER_TOP', recommendation_types)

    def test_postgresql_recommendations(self):
        """Test PostgreSQL-specific recommendations."""
        postgresql_query = """
            SELECT DISTINCT u.name, u.email
            FROM users u
            JOIN orders o ON u.id = o.user_id
            WHERE u.description LIKE '%developer%'
            ORDER BY u.name;
        """

        query, analysis = analyze_query(postgresql_query, 'postgresql')

        # Should provide PostgreSQL-specific recommendations
        recommendation_types = [rec['type'] for rec in analysis.recommendations]
        self.assertIn('USE_POSTGRESQL_DISTINCT_ON', recommendation_types)
        self.assertIn('POSTGRESQL_TEXT_SEARCH', recommendation_types)

    @skip("SQLite analyzer not yet implemented - marked as Phase 3-4 future work")
    def test_sqlite_unsupported_features(self):
        """Test SQLite unsupported feature detection."""
        # Test RIGHT JOIN (unsupported in SQLite)
        sqlite_query_with_right_join = """
            SELECT u.name, p.title
            FROM users u
            RIGHT JOIN products p ON u.id = p.created_by
            WHERE u.active = 1;
        """

        query, analysis = analyze_query(sqlite_query_with_right_join, 'sqlite')

        # Should detect unsupported feature
        issue_types = [issue['type'] for issue in analysis.issues_found]
        self.assertIn('SQLITE_UNSUPPORTED_FEATURE', issue_types)

        # Should have critical severity
        for issue in analysis.issues_found:
            if issue['type'] == 'SQLITE_UNSUPPORTED_FEATURE':
                self.assertEqual(issue['severity'], 'critical')

        # Grade should be significantly lower
        self.assertLess(analysis.score, 80.0)

    def test_database_specific_recommendations_added(self):
        """Test that database-specific recommendations are added."""
        basic_query = """
            SELECT u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            GROUP BY u.id, u.name
            ORDER BY order_count DESC;
        """

        # Test with only implemented database analyzers (MySQL, PostgreSQL)
        # Note: SQLite, Oracle, SQL Server analyzers are Phase 3-4 future work
        database_types = ['mysql', 'postgresql']

        for db_type in database_types:
            query, analysis = analyze_query(basic_query, db_type)

            # Should have some database-specific recommendations
            db_specific_recs = [
                rec for rec in analysis.recommendations
                if db_type.upper() in rec['type']
            ]

            self.assertGreater(len(db_specific_recs), 0,
                             f"Should have {db_type}-specific recommendations")

    def test_no_database_type_still_works(self):
        """Test that analysis still works without specifying database type."""
        basic_query = """
            SELECT u.name, u.email
            FROM users u
            WHERE u.active = 1
            ORDER BY u.created_at DESC;
        """

        # Test without database_type (default behavior)
        query, analysis = analyze_query(basic_query)

        self.assertIsNotNone(analysis)
        self.assertIsInstance(analysis.score, float)
        self.assertIn(analysis.grade, ['A', 'B', 'C', 'D', 'F'])

        # Test with empty database_type
        query2, analysis2 = analyze_query(basic_query, '')

        self.assertEqual(analysis.score, analysis2.score)
        self.assertEqual(analysis.grade, analysis2.grade)

    def test_mysql_specific_optimizations(self):
        """Test MySQL-specific optimization recommendations."""
        mysql_query_with_limit = """
            SELECT u.name, u.email
            FROM users u
            ORDER BY u.created_at DESC
            LIMIT 100;
        """

        query, analysis = analyze_query(mysql_query_with_limit, 'mysql')

        # Should provide MySQL-specific optimization tips
        recommendation_types = [rec['type'] for rec in analysis.recommendations]
        self.assertIn('MYSQL_INDEX_OPTIMIZATION', recommendation_types)
        self.assertIn('MYSQL_STORAGE_ENGINE', recommendation_types)

    @skip("SQL Server analyzer not yet implemented - marked as Phase 3-4 future work")
    def test_sqlserver_date_functions(self):
        """Test SQL Server date function recommendations."""
        # Query with date functions that aren't optimal for SQL Server
        sqlserver_query = """
            SELECT u.name, u.email
            FROM users u
            WHERE DATEPART(year, u.created_at) = 2023
            ORDER BY u.created_at DESC;
        """

        query, analysis = analyze_query(sqlserver_query, 'sqlserver')

        # Should provide SQL Server-specific recommendations
        recommendation_types = [rec['type'] for rec in analysis.recommendations]
        sqlserver_recs = [rec for rec in recommendation_types if 'SQLSERVER' in rec]
        self.assertGreater(len(sqlserver_recs), 0)

    def test_postgresql_window_functions(self):
        """Test PostgreSQL window function recommendations."""
        postgresql_query = """
            SELECT u.name, COUNT(o.id) as order_count
            FROM users u
            JOIN orders o ON u.id = o.user_id
            GROUP BY u.id, u.name
            ORDER BY order_count DESC;
        """

        query, analysis = analyze_query(postgresql_query, 'postgresql')

        # Should suggest PostgreSQL window functions
        recommendation_types = [rec['type'] for rec in analysis.recommendations]
        self.assertIn('USE_POSTGRESQL_WINDOW_FUNCTIONS', recommendation_types)

    def test_grade_impact_of_database_syntax_errors(self):
        """Test that database syntax errors properly impact grades."""
        # Test queries with syntax errors for implemented databases only
        # Note: Oracle, SQL Server, SQLite analyzers are Phase 3-4 future work
        test_cases = [
            ('mysql', 'SELECT TOP 10 * FROM users;', 'MYSQL_SYNTAX_ERROR'),
        ]

        for db_type, bad_query, expected_error_type in test_cases:
            query, analysis = analyze_query(bad_query, db_type)

            # Should have the expected error
            issue_types = [issue['type'] for issue in analysis.issues_found]
            self.assertIn(expected_error_type, issue_types,
                         f"Should detect {expected_error_type} for {db_type}")

            # Grade should be impacted
            self.assertLess(analysis.score, 100.0,
                           f"Score should be reduced for {db_type} syntax error")

    def test_case_insensitive_detection(self):
        """Test that database-specific detection is case insensitive."""
        # Test with lowercase SQL keywords using MySQL (implemented analyzer)
        mysql_query_lower = """
            select top 10 u.name, u.email
            from users u
            order by u.created_at desc;
        """

        query, analysis = analyze_query(mysql_query_lower, 'mysql')

        # Should still detect the TOP syntax issue even in lowercase
        issue_types = [issue['type'] for issue in analysis.issues_found]
        self.assertIn('MYSQL_SYNTAX_ERROR', issue_types)