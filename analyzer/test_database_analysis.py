import json
from unittest import skip
from unittest.mock import MagicMock, Mock, patch

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from .database_introspector import (ColumnInfo, DatabaseIntrospector,
                                    IndexInfo, TableInfo)
from .forms import DatabaseConnectionForm


class DatabaseConnectionFormTest(TestCase):
    """Test database connection form validation."""

    def test_valid_mysql_connection(self):
        """Test valid MySQL connection form data."""
        form_data = {
            "engine": "mysql",
            "name": "test_db",
            "host": "localhost",
            "port": "3306",
            "user": "testuser",
            "password": "testpass",
            "schema": "",
        }
        form = DatabaseConnectionForm(data=form_data)
        self.assertTrue(form.is_valid())

        config = form.get_connection_config()
        self.assertEqual(config["engine"], "mysql")
        self.assertEqual(config["name"], "test_db")
        self.assertEqual(config["host"], "localhost")

    def test_valid_postgresql_connection(self):
        """Test valid PostgreSQL connection form data."""
        form_data = {
            "engine": "postgresql",
            "name": "test_db",
            "host": "localhost",
            "port": "5432",
            "user": "testuser",
            "password": "testpass",
            "schema": "public",
        }
        form = DatabaseConnectionForm(data=form_data)
        self.assertTrue(form.is_valid())

        config = form.get_connection_config()
        self.assertEqual(config["engine"], "postgresql")
        self.assertEqual(config["schema"], "public")

    def test_valid_sqlite_connection(self):
        """Test valid SQLite connection form data."""
        form_data = {
            "engine": "sqlite",
            "name": "/path/to/database.db",
            "host": "",
            "port": "",
            "user": "",
            "password": "",
            "schema": "",
        }
        form = DatabaseConnectionForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_mysql_without_username_invalid(self):
        """Test MySQL connection without username is invalid."""
        form_data = {
            "engine": "mysql",
            "name": "test_db",
            "host": "localhost",
            "user": "",  # Missing username
            "password": "testpass",
        }
        form = DatabaseConnectionForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("Username is required to connect to MYSQL", str(form.errors))

    def test_sqlite_without_name_invalid(self):
        """Test SQLite connection without database name is invalid."""
        form_data = {
            "engine": "sqlite",
            "name": "",  # Missing database name
        }
        form = DatabaseConnectionForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("Database file path is required for SQLite", str(form.errors))


class DatabaseIntrospectorTest(TestCase):
    """Test database introspector functionality."""

    def setUp(self):
        self.db_config = {
            "engine": "sqlite",
            "name": ":memory:",
            "host": "",
            "port": "",
            "user": "",
            "password": "",
        }
        self.introspector = DatabaseIntrospector(self.db_config)

    def test_create_django_db_config(self):
        """Test Django database configuration creation."""
        config = self.introspector._create_django_db_config()
        self.assertEqual(config["ENGINE"], "django.db.backends.sqlite3")
        self.assertEqual(config["NAME"], ":memory:")

    def test_mysql_db_config(self):
        """Test MySQL database configuration."""
        mysql_config = {
            "engine": "mysql",
            "name": "test_db",
            "host": "localhost",
            "port": "3306",
            "user": "testuser",
            "password": "testpass",
        }
        introspector = DatabaseIntrospector(mysql_config)
        config = introspector._create_django_db_config()

        self.assertEqual(config["ENGINE"], "django.db.backends.mysql")
        self.assertEqual(config["NAME"], "test_db")
        self.assertEqual(config["HOST"], "localhost")
        self.assertEqual(config["PORT"], "3306")
        self.assertEqual(config["USER"], "testuser")

    @patch("analyzer.database_introspector.connections")
    def test_connect_success(self, mock_connections):
        """Test successful database connection."""
        # Mock Django connections
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor_context = Mock()
        mock_cursor_context.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor_context.__exit__ = Mock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor_context
        mock_cursor.execute.return_value = None

        mock_connections.__getitem__.return_value = mock_connection
        mock_connections.databases = {}

        result = self.introspector.connect()
        self.assertTrue(result)

    @patch("analyzer.database_introspector.connections")
    def test_connect_failure(self, mock_connections):
        """Test database connection failure."""
        # Mock connection failure
        mock_connections.__getitem__.side_effect = Exception("Connection failed")
        mock_connections.databases = {}

        result = self.introspector.connect()
        self.assertFalse(result)

    def test_analyze_query_context_mock(self):
        """Test query context analysis with mocked data."""
        # Mock a simple query context analysis
        sql_query = "SELECT * FROM users WHERE active = 1"

        # Mock connection and parsing
        with patch.object(self.introspector, "connection", Mock()):
            with patch("analyzer.database_introspector.sqlparse") as mock_sqlparse:
                # Mock sqlparse parsing
                mock_parsed = Mock()
                mock_sqlparse.parse.return_value = [mock_parsed]

                # Mock table and column extraction
                with patch.object(
                    self.introspector,
                    "_extract_table_references",
                    return_value=["users"],
                ):
                    with patch.object(
                        self.introspector,
                        "_extract_column_references",
                        return_value=["active"],
                    ):
                        with patch.object(
                            self.introspector,
                            "_tables_cache",
                            {"users": self._create_mock_table()},
                        ):
                            context = self.introspector.analyze_query_context(sql_query)

                            self.assertIn("tables_referenced", context)
                            self.assertIn("columns_referenced", context)
                            self.assertIn("schema_analysis", context)
                            self.assertIn("recommendations", context)

    def _create_mock_table(self):
        """Create a mock table for testing."""
        table = TableInfo(name="users", row_count=1000, size_mb=10.5)
        table.columns = [
            {"name": "id", "type": "integer"},
            {"name": "active", "type": "boolean"},
            {"name": "name", "type": "varchar"},
        ]
        table.indexes = [
            {"name": "primary", "columns": ["id"], "primary": True, "unique": True}
        ]
        return table


class DatabaseAnalysisViewsTest(TestCase):
    """Test database analysis views."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )
        self.client.login(username="testuser", password="testpass123")

    def test_database_analyze_get(self):
        """Test database analyze view GET request."""
        response = self.client.get(reverse("database_analyze"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Database Architecture Analysis")
        self.assertContains(response, "Database Engine")

    def test_database_analyze_requires_login(self):
        """Test database analyze view requires authentication."""
        self.client.logout()
        response = self.client.get(reverse("database_analyze"))
        self.assertRedirects(response, "/login/?next=/database/")

    @skip(
        "Database introspector views not yet implemented - marked as Phase 3-4 future work"
    )
    def test_database_analyze_post_success(self):
        """Test successful database connection."""
        with patch("analyzer.views.DatabaseIntrospector") as mock_introspector_class:
            # Mock introspector
            mock_introspector = Mock()
            mock_introspector.connect.return_value = True
            mock_introspector_class.return_value = mock_introspector

            form_data = {
                "engine": "sqlite",
                "name": ":memory:",
                "host": "",
                "port": "",
                "user": "",
                "password": "",
                "schema": "",
            }

            response = self.client.post(reverse("database_analyze"), data=form_data)
            self.assertRedirects(response, reverse("database_schema"))

            # Check that connection config is stored in session
            session = self.client.session
            self.assertIn("db_config", session)
            self.assertEqual(session["db_config"]["engine"], "sqlite")

    @skip(
        "Database introspector views not yet implemented - marked as Phase 3-4 future work"
    )
    def test_database_analyze_post_failure(self):
        """Test failed database connection."""
        with patch("analyzer.views.DatabaseIntrospector") as mock_introspector_class:
            # Mock introspector connection failure
            mock_introspector = Mock()
            mock_introspector.connect.return_value = False
            mock_introspector_class.return_value = mock_introspector

            form_data = {
                "engine": "mysql",
                "name": "test_db",
                "host": "localhost",
                "user": "testuser",
                "password": "wrongpass",
            }

            response = self.client.post(reverse("database_analyze"), data=form_data)
            self.assertEqual(response.status_code, 200)

            # Check error message
            messages = list(response.context["messages"])
            self.assertTrue(any("Failed to connect" in str(msg) for msg in messages))

    def test_database_schema_no_connection(self):
        """Test database schema view without connection."""
        response = self.client.get(reverse("database_schema"))
        self.assertRedirects(response, reverse("database_analyze"))

    @skip(
        "Database introspector views not yet implemented - marked as Phase 3-4 future work"
    )
    def test_database_schema_with_connection(self):
        """Test database schema view with valid connection."""
        with patch("analyzer.views.DatabaseIntrospector") as mock_introspector_class:
            # Set up session with connection config
            session = self.client.session
            session["db_config"] = {
                "engine": "sqlite",
                "name": ":memory:",
                "host": "",
                "port": "",
                "user": "",
                "password": "",
            }
            session.save()

            # Mock introspector
            mock_introspector = Mock()
            mock_introspector.connect.return_value = True
            mock_introspector.get_tables.return_value = [
                self._create_test_table("users"),
                self._create_test_table("orders"),
            ]
            mock_introspector_class.return_value = mock_introspector

            # Mock schema analysis function
            with patch("analyzer.views.analyze_database_schema") as mock_analyze:
                mock_analyze.return_value = {
                    "recommendations": ["Add indexes on foreign keys"],
                    "issues": ["Table users lacks primary key"],
                    "performance_notes": ["Table orders has many rows"],
                    "statistics": {},
                }

                response = self.client.get(reverse("database_schema"))
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "Schema Analysis Results")
                self.assertContains(response, "users")
                self.assertContains(response, "orders")

    def test_query_with_context_no_connection(self):
        """Test context query view without database connection."""
        response = self.client.get(reverse("query_with_context"))
        self.assertRedirects(response, reverse("database_analyze"))

    @skip(
        "Database introspector views not yet implemented - marked as Phase 3-4 future work"
    )
    def test_query_with_context_post(self):
        """Test context-aware query analysis."""
        with patch("analyzer.views.DatabaseIntrospector") as mock_introspector_class:
            with patch("analyzer.views.analyze_query") as mock_analyze_query:
                # Set up session with connection
                session = self.client.session
                session["db_config"] = {
                    "engine": "sqlite",
                    "name": ":memory:",
                    "host": "",
                    "port": "",
                    "user": "",
                    "password": "",
                }
                session.save()

                # Mock introspector
                mock_introspector = Mock()
                mock_introspector.connect.return_value = True
                mock_introspector.analyze_query_context.return_value = {
                    "tables_referenced": ["users"],
                    "columns_referenced": ["id", "name"],
                    "recommendations": ["Add index on name column"],
                }
                mock_introspector.get_execution_plan.return_value = {
                    "plan": "Seq Scan on users",
                    "format": "text",
                }
                mock_introspector_class.return_value = mock_introspector

                # Mock query analysis
                from .models import Query, QueryAnalysis

                mock_query = Mock(spec=Query)
                mock_query.id = 1
                mock_analysis = Mock(spec=QueryAnalysis)
                mock_analysis.id = 1
                mock_analyze_query.return_value = (mock_query, mock_analysis)

                form_data = {
                    "sql_query": 'SELECT * FROM users WHERE name = "test"',
                    "database_version": "5.7",
                    "use_case_notes": "Test query",
                }

                response = self.client.post(
                    reverse("query_with_context"), data=form_data
                )
                self.assertRedirects(
                    response, reverse("contextualized_results", args=[1])
                )

                # Check that context analysis is stored in session
                session = self.client.session
                self.assertIn("context_analysis", session)
                self.assertIn("execution_plan", session)

    def test_contextualized_results_no_analysis(self):
        """Test contextualized results view with non-existent analysis."""
        response = self.client.get(reverse("contextualized_results", args=[999]))
        self.assertRedirects(response, reverse("query_history"))

    def _create_test_table(self, name):
        """Create a test table for mocking."""
        table = TableInfo(name=name, row_count=100, size_mb=1.5)
        table.columns = [
            {"name": "id", "type": "integer"},
            {"name": "name", "type": "varchar"},
        ]
        table.indexes = [{"name": "primary", "columns": ["id"], "primary": True}]
        table.foreign_keys = []
        return table


class DatabaseSchemaAnalysisTest(TestCase):
    """Test database schema analysis functions."""

    @skip(
        "Database schema analysis not yet implemented - marked as Phase 3-4 future work"
    )
    def test_analyze_database_schema_no_issues(self):
        """Test schema analysis with well-structured tables."""
        from .views import analyze_database_schema

        tables = [self._create_good_table()]
        db_config = {"engine": "mysql"}

        analysis = analyze_database_schema(tables, db_config)

        self.assertIn("recommendations", analysis)
        self.assertIn("issues", analysis)
        self.assertIn("performance_notes", analysis)
        self.assertIn("statistics", analysis)

    @skip(
        "Database schema analysis not yet implemented - marked as Phase 3-4 future work"
    )
    def test_analyze_database_schema_with_issues(self):
        """Test schema analysis with problematic tables."""
        from .views import analyze_database_schema

        tables = [
            self._create_table_without_primary_key(),
            self._create_large_table(),
            self._create_table_with_many_columns(),
        ]
        db_config = {"engine": "postgresql"}

        analysis = analyze_database_schema(tables, db_config)

        # Should find issues
        self.assertTrue(len(analysis["issues"]) > 0)
        self.assertTrue(len(analysis["recommendations"]) > 0)
        self.assertIn("lacks a primary key", " ".join(analysis["issues"]))

    def _create_good_table(self):
        """Create a well-structured table for testing."""
        table = TableInfo(name="users", row_count=1000, size_mb=5.0)
        table.columns = [
            {"name": "id", "type": "integer"},
            {"name": "email", "type": "varchar"},
            {"name": "name", "type": "varchar"},
        ]
        table.indexes = [
            {"name": "primary", "columns": ["id"], "primary": True, "unique": True},
            {
                "name": "idx_email",
                "columns": ["email"],
                "primary": False,
                "unique": True,
            },
        ]
        table.foreign_keys = []
        return table

    def _create_table_without_primary_key(self):
        """Create a table without primary key for testing."""
        table = TableInfo(name="logs", row_count=10000, size_mb=50.0)
        table.columns = [
            {"name": "timestamp", "type": "datetime"},
            {"name": "message", "type": "text"},
        ]
        table.indexes = []  # No primary key
        table.foreign_keys = []
        return table

    def _create_large_table(self):
        """Create a large table for testing."""
        table = TableInfo(name="events", row_count=5000000, size_mb=1500.0)
        table.columns = [
            {"name": "id", "type": "integer"},
            {"name": "event_data", "type": "json"},
        ]
        table.indexes = [{"name": "primary", "columns": ["id"], "primary": True}]
        table.foreign_keys = []
        return table

    def _create_table_with_many_columns(self):
        """Create a table with many columns for testing."""
        table = TableInfo(name="wide_table", row_count=1000, size_mb=10.0)
        # Create 25 columns (more than the 20 column threshold)
        table.columns = [{"name": f"col_{i}", "type": "varchar"} for i in range(25)]
        table.indexes = [{"name": "primary", "columns": ["col_0"], "primary": True}]
        table.foreign_keys = []
        return table
