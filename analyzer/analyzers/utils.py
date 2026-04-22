"""
Utility functions for SQL query analysis.

This module provides helper functions for extracting metrics and metadata
from parsed SQL queries.
"""

import re

from sqlparse import tokens
from sqlparse.sql import Statement


def get_query_type(parsed: Statement) -> str:
    """
    Determine the type of SQL query (SELECT, INSERT, UPDATE, etc.).

    Args:
        parsed: Parsed SQL statement

    Returns:
        str: Query type in uppercase (e.g., 'SELECT', 'INSERT', 'UPDATE')
    """
    for token in parsed.flatten():
        if token.ttype is tokens.Keyword.DML:
            return token.value.upper()
        elif token.ttype is tokens.Keyword.DDL:
            return token.value.upper()
    return "UNKNOWN"


def count_tables(parsed: Statement) -> int:
    """
    Count the number of tables referenced in the query.

    Examines FROM, JOIN, INSERT, UPDATE, and DELETE clauses to identify
    unique table references.

    Args:
        parsed: Parsed SQL statement

    Returns:
        int: Number of unique tables referenced
    """
    sql_text = str(parsed).upper()
    tables = set()

    # Look for FROM and JOIN patterns to identify tables
    from_pattern = (
        r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:[a-zA-Z_][a-zA-Z0-9_]*)?"
    )
    matches = re.findall(from_pattern, sql_text, re.IGNORECASE)

    for match in matches:
        if match and match.upper() not in [
            "SELECT",
            "WHERE",
            "GROUP",
            "ORDER",
            "HAVING",
        ]:
            tables.add(match.lower())

    # Also check for table references in INSERT, UPDATE, DELETE
    insert_pattern = r"\bINSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    update_pattern = r"\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    delete_pattern = r"\bDELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)"

    for pattern in [insert_pattern, update_pattern, delete_pattern]:
        matches = re.findall(pattern, sql_text, re.IGNORECASE)
        for match in matches:
            if match:
                tables.add(match.lower())

    return len(tables)


def count_joins(parsed: Statement) -> int:
    """
    Count the number of JOIN operations in the query.

    Handles INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, CROSS JOIN,
    and their OUTER variants. Processes patterns in order of specificity
    to avoid double-counting.

    Args:
        parsed: Parsed SQL statement

    Returns:
        int: Number of JOIN operations
    """
    sql_text = str(parsed).upper()

    # More precise pattern to avoid double-counting
    join_patterns = [
        r"\bINNER\s+JOIN\b",
        r"\bLEFT\s+OUTER\s+JOIN\b",
        r"\bRIGHT\s+OUTER\s+JOIN\b",
        r"\bFULL\s+OUTER\s+JOIN\b",
        r"\bLEFT\s+JOIN\b",
        r"\bRIGHT\s+JOIN\b",
        r"\bFULL\s+JOIN\b",
        r"\bCROSS\s+JOIN\b",
        r"\bJOIN\b",  # This should be last to avoid double counting
    ]

    join_count = 0
    remaining_text = sql_text

    # Process patterns in order of specificity
    for pattern in join_patterns:
        matches = re.findall(pattern, remaining_text)
        join_count += len(matches)
        # Remove found matches to prevent double counting
        remaining_text = re.sub(pattern, "", remaining_text)

    return join_count


def count_where_conditions(parsed: Statement) -> int:
    """
    Count the number of conditions in WHERE clauses.

    Counts AND/OR operators as indicators of multiple conditions.
    If there's a WHERE clause but no AND/OR, returns 1.

    Args:
        parsed: Parsed SQL statement

    Returns:
        int: Number of WHERE conditions
    """
    condition_count = 0
    sql_text = str(parsed).upper()

    # Count AND/OR operators as indicators of multiple conditions
    condition_count += sql_text.count(" AND ")
    condition_count += sql_text.count(" OR ")

    # If there's a WHERE clause but no AND/OR, there's at least one condition
    if "WHERE" in sql_text and condition_count == 0:
        condition_count = 1

    return condition_count


def count_subqueries(parsed: Statement) -> int:
    """
    Count the number of subqueries in the query.

    Identifies subqueries by looking for SELECT, INSERT, UPDATE, or DELETE
    keywords within parentheses.

    Args:
        parsed: Parsed SQL statement

    Returns:
        int: Number of subqueries
    """
    sql_text = str(parsed)
    # Simple approach: count opening parentheses that likely indicate subqueries
    subquery_indicators = ["SELECT", "INSERT", "UPDATE", "DELETE"]
    subquery_count = 0

    in_parentheses = 0
    i = 0
    while i < len(sql_text):
        if sql_text[i] == "(":
            in_parentheses += 1
            # Look ahead to see if this contains a query keyword
            remaining = sql_text[i : i + 50].upper()
            if any(keyword in remaining for keyword in subquery_indicators):
                subquery_count += 1
        elif sql_text[i] == ")":
            in_parentheses -= 1
        i += 1

    return subquery_count
