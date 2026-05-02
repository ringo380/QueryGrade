from django.core.exceptions import ValidationError
from django.test import TestCase

from analyzer.forms.validators import validate_sql_query


class ValidateSqlQueryTestCase(TestCase):
    """Unit tests for the validate_sql_query form validator."""

    # --- Statements that must pass ---

    def test_select_passes(self):
        validate_sql_query("SELECT id, name FROM users WHERE active = 1")

    def test_select_with_cte_passes(self):
        validate_sql_query("WITH cte AS (SELECT id FROM users) SELECT * FROM cte")

    def test_select_with_union_passes(self):
        validate_sql_query(
            "SELECT id FROM users UNION ALL SELECT id FROM archived_users"
        )

    def test_select_with_comments_passes(self):
        validate_sql_query("SELECT id FROM users -- filter by status\nWHERE active = 1")

    def test_select_with_block_comment_passes(self):
        validate_sql_query("SELECT /* all cols */ id, name FROM users")

    def test_insert_passes(self):
        validate_sql_query("INSERT INTO orders (user_id, total) VALUES (1, 99.99)")

    def test_update_passes(self):
        validate_sql_query("UPDATE users SET email = 'x@x.com' WHERE id = 1")

    def test_delete_passes(self):
        validate_sql_query("DELETE FROM sessions WHERE expires_at < NOW()")

    def test_create_table_passes(self):
        validate_sql_query("CREATE TABLE logs (id INT PRIMARY KEY, msg TEXT)")

    def test_alter_table_passes(self):
        validate_sql_query("ALTER TABLE users ADD COLUMN bio TEXT")

    def test_drop_table_passes(self):
        validate_sql_query("DROP TABLE IF EXISTS temp_logs")

    def test_truncate_passes(self):
        validate_sql_query("TRUNCATE TABLE sessions")

    def test_select_with_information_schema_passes(self):
        validate_sql_query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )

    def test_select_with_string_functions_passes(self):
        validate_sql_query("SELECT SUBSTRING(name, 1, 10), CHAR(65) FROM users")

    def test_select_with_sleep_passes(self):
        # Grading tools should be able to analyze queries containing sleep() calls
        validate_sql_query("SELECT sleep(1)")

    def test_trailing_semicolon_passes(self):
        validate_sql_query("SELECT id FROM users;")

    # --- Statements that must still be rejected ---

    def test_empty_query_rejected(self):
        with self.assertRaises(ValidationError):
            validate_sql_query("")

    def test_whitespace_only_rejected(self):
        with self.assertRaises(ValidationError):
            validate_sql_query("   \n\t  ")

    def test_query_over_length_limit_rejected(self):
        with self.assertRaises(ValidationError):
            validate_sql_query("SELECT " + "a, " * 4000 + "1 FROM t")

    def test_multiple_statements_rejected(self):
        with self.assertRaises(ValidationError):
            validate_sql_query("SELECT 1; SELECT 2;")

    def test_two_semicolons_rejected(self):
        with self.assertRaises(ValidationError):
            validate_sql_query("SELECT 1; DROP TABLE users;")
