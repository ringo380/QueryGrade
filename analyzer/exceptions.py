"""
Custom exceptions for SQL query analysis and grading.
"""


class QueryAnalysisError(Exception):
    """Base class for query analysis errors."""

    def __init__(self, message, error_type=None, suggestions=None):
        self.message = message
        self.error_type = error_type or "general"
        self.suggestions = suggestions or []
        super().__init__(self.message)


class EmptyQueryError(QueryAnalysisError):
    """Raised when query is empty or contains only whitespace."""

    def __init__(self):
        super().__init__(
            message="Your SQL query appears to be empty.",
            error_type="empty_query",
            suggestions=[
                "Enter a valid SQL statement (SELECT, INSERT, UPDATE, DELETE, etc.)",
                "Make sure your query contains actual SQL commands",
                "Try our example: SELECT * FROM users WHERE active = 1;",
            ],
        )


class SyntaxError(QueryAnalysisError):
    """Raised when query contains syntax errors."""

    def __init__(self, details=None):
        message = "Your SQL query contains syntax errors."
        suggestions = [
            "Check for missing keywords (SELECT, FROM, WHERE, etc.)",
            "Verify parentheses and quotes are properly closed",
            "Ensure table and column names are spelled correctly",
            "Use a SQL formatter to help identify structure issues",
        ]

        if details:
            message += f" {details}"

        super().__init__(
            message=message, error_type="syntax_error", suggestions=suggestions
        )


class TypoError(QueryAnalysisError):
    """Raised when query contains apparent typos in SQL keywords."""

    def __init__(self, detected_typos=None):
        message = "Your SQL query appears to contain typos in SQL keywords."
        suggestions = [
            "Common typos: 'SELCT' should be 'SELECT'",
            "Common typos: 'FORM' should be 'FROM'",
            "Common typos: 'WHER' should be 'WHERE'",
            "Use an SQL editor with syntax highlighting to catch typos",
            "Double-check spelling of all SQL keywords",
        ]

        if detected_typos:
            message += f" Possible issues detected: {', '.join(detected_typos)}"

        super().__init__(
            message=message, error_type="typo_error", suggestions=suggestions
        )


class IncompleteQueryError(QueryAnalysisError):
    """Raised when query appears incomplete or truncated."""

    def __init__(self, missing_part=None):
        message = "Your SQL query appears to be incomplete."
        suggestions = [
            "Make sure your query has all required parts (SELECT...FROM...)",
            "Check that your query ends properly (with semicolon if needed)",
            "Ensure WHERE clauses have complete conditions",
            "Verify JOIN statements have ON conditions",
        ]

        if missing_part:
            message += f" Missing: {missing_part}"

        super().__init__(
            message=message, error_type="incomplete_query", suggestions=suggestions
        )


class UnsupportedQueryError(QueryAnalysisError):
    """Raised when query type is not supported for analysis."""

    def __init__(self, query_type=None):
        message = "This type of SQL query is not currently supported for analysis."
        suggestions = [
            "Try SELECT, INSERT, UPDATE, or DELETE statements",
            "DDL statements (CREATE, ALTER, DROP) are not yet supported",
            "Administrative commands are not supported",
            "Break complex queries into simpler parts for analysis",
        ]

        if query_type:
            message += f" Query type detected: {query_type}"

        super().__init__(
            message=message, error_type="unsupported_query", suggestions=suggestions
        )


class DatabaseSpecificError(QueryAnalysisError):
    """Raised when query contains database-specific syntax issues."""

    def __init__(self, database_type, issue_description, correct_syntax=None):
        message = f"Your query contains syntax that is not compatible with {database_type.title()}."
        if issue_description:
            message += f" {issue_description}"

        suggestions = [
            f"Review {database_type.title()}-specific SQL syntax documentation",
            "Consider using standard SQL for better portability",
            "Check if your database supports the features you're using",
        ]

        if correct_syntax:
            suggestions.insert(0, f"Try using: {correct_syntax}")

        super().__init__(
            message=message,
            error_type="database_specific_error",
            suggestions=suggestions,
        )
