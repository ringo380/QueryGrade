"""
ORDER BY analyzer for detecting sorting inefficiencies.

This module analyzes queries to identify:
- ORDER BY on unindexed columns
- Sorting large result sets without LIMIT
- ORDER BY with expressions/functions
- Multiple column sorting order
- DISTINCT with ORDER BY
"""

import re

from .base import AnalysisContext, BaseAnalyzer


class OrderByAnalyzer(BaseAnalyzer):
    """
    Analyzer for ORDER BY clause optimization.

    Detects issues such as:
    - ORDER BY without LIMIT (sorting entire result set)
    - ORDER BY with functions (prevents index usage)
    - ORDER BY on expressions
    - Mixed ASC/DESC in multi-column ORDER BY
    - ORDER BY with DISTINCT
    """

    @property
    def name(self) -> str:
        return "OrderByAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze ORDER BY clause for optimization opportunities.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()

        # Only analyze if ORDER BY is present
        if "ORDER BY" not in sql_text:
            return

        self._check_orderby_without_limit(sql_text, context)
        self._check_orderby_with_functions(sql_text, context)
        self._check_orderby_expression(sql_text, context)
        self._check_multi_column_ordering(sql_text, context)
        self._check_orderby_with_distinct(sql_text, context)
        self._check_orderby_in_subquery(sql_text, context)
        self._check_random_ordering(sql_text, context)

    def _check_orderby_without_limit(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for ORDER BY without LIMIT - sorts entire result set."""
        has_order_by = "ORDER BY" in sql_text
        has_limit = (
            "LIMIT" in sql_text or "TOP" in sql_text or "FETCH FIRST" in sql_text
        )

        if has_order_by and not has_limit:
            # Check if it's a subquery (less critical there)
            if sql_text.count("SELECT") == 1:  # Main query only
                context.issues.append(
                    {
                        "type": "ORDER_BY_NO_LIMIT",
                        "severity": "medium",
                        "description": "ORDER BY without LIMIT sorts the entire result set - consider adding LIMIT if you only need top N rows",  # noqa: E501
                    }
                )
                context.recommendations.append(
                    {
                        "type": "ADD_LIMIT",
                        "priority": "medium",
                        "description": "If you only need the first N results, add LIMIT to reduce sorting cost",
                        "example": "SELECT ... ORDER BY column LIMIT 10",
                    }
                )

    def _check_orderby_with_functions(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for functions in ORDER BY clause."""
        # Common functions in ORDER BY
        function_patterns = [
            (r"ORDER BY[^(]*UPPER\(", "UPPER"),
            (r"ORDER BY[^(]*LOWER\(", "LOWER"),
            (r"ORDER BY[^(]*DATE\(", "DATE"),
            (r"ORDER BY[^(]*YEAR\(", "YEAR"),
            (r"ORDER BY[^(]*SUBSTRING\(", "SUBSTRING"),
            (r"ORDER BY[^(]*CONCAT\(", "CONCAT"),
            (r"ORDER BY[^(]*ABS\(", "ABS"),
            (r"ORDER BY[^(]*LENGTH\(", "LENGTH"),
        ]

        for pattern, func_name in function_patterns:
            if re.search(pattern, sql_text):
                context.issues.append(
                    {
                        "type": "FUNCTION_IN_ORDER_BY",
                        "severity": "medium",
                        "description": f"Using {func_name}() in ORDER BY prevents index usage and requires computing the function for all rows",  # noqa: E501
                    }
                )
                context.recommendations.append(
                    {
                        "type": "PRECOMPUTE_ORDER_COLUMN",
                        "priority": "medium",
                        "description": f"Consider storing {func_name}(column) as a computed/generated column with an index",  # noqa: E501
                        "example": "ALTER TABLE table_name ADD COLUMN computed_col GENERATED ALWAYS AS (function(col)) STORED; CREATE INDEX idx_computed ON table_name(computed_col);",  # noqa: E501
                    }
                )
                break  # Only report once

    def _check_orderby_expression(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for complex expressions in ORDER BY."""
        # Look for ORDER BY with arithmetic, CASE, or other expressions
        expression_patterns = [
            r"ORDER BY[^(]*[\+\-\*/]",  # Arithmetic
            r"ORDER BY[^(]*CASE\s+WHEN",  # CASE expressions
        ]

        for pattern in expression_patterns:
            if re.search(pattern, sql_text):
                context.issues.append(
                    {
                        "type": "EXPRESSION_IN_ORDER_BY",
                        "severity": "medium",
                        "description": "Complex expression in ORDER BY cannot use indexes and must be computed for all rows",  # noqa: E501
                    }
                )
                context.recommendations.append(
                    {
                        "type": "COMPUTED_COLUMN_FOR_ORDER",
                        "priority": "medium",
                        "description": "Consider creating a computed/generated column for complex ORDER BY expressions",
                        "example": "ALTER TABLE table_name ADD COLUMN order_value AS (expression) STORED; CREATE INDEX ON table_name(order_value);",  # noqa: E501
                    }
                )
                break

    def _check_multi_column_ordering(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for multi-column ORDER BY."""
        # Count commas in ORDER BY clause
        order_by_match = re.search(r"ORDER BY([^;]*?)(?:LIMIT|OFFSET|$|;)", sql_text)
        if order_by_match:
            order_clause = order_by_match.group(1)
            comma_count = order_clause.count(",")

            if comma_count >= 2:
                context.recommendations.append(
                    {
                        "type": "MULTI_COLUMN_INDEX",
                        "priority": "high",
                        "description": f"ORDER BY uses {comma_count + 1} columns - composite index with columns in the same order can improve performance",  # noqa: E501
                        "example": "CREATE INDEX idx_multi_order ON table_name(col1, col2, col3) -- in ORDER BY order",
                    }
                )

            # Check for mixed ASC/DESC
            if "ASC" in order_clause and "DESC" in order_clause:
                context.recommendations.append(
                    {
                        "type": "MIXED_SORT_ORDER",
                        "priority": "medium",
                        "description": "Mixed ASC/DESC in ORDER BY may prevent full index usage on some databases",
                        "example": "Some databases support mixed-order indexes: CREATE INDEX idx_mixed ON table(col1 ASC, col2 DESC)",  # noqa: E501
                    }
                )

    def _check_orderby_with_distinct(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for ORDER BY with DISTINCT."""
        if "DISTINCT" in sql_text and "ORDER BY" in sql_text:
            context.performance_notes.append(
                "DISTINCT with ORDER BY requires sorting after duplicate removal. "
                "Ensure ORDER BY columns are in SELECT list or use GROUP BY instead."
            )

            # Check if ORDER BY column might not be in SELECT
            # This is a simplified check
            context.recommendations.append(
                {
                    "type": "DISTINCT_ORDER_BY",
                    "priority": "low",
                    "description": "ORDER BY columns must be in SELECT list when using DISTINCT",
                    "example": "SELECT DISTINCT col1, col2, order_col FROM table ORDER BY order_col",
                }
            )

    def _check_orderby_in_subquery(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for ORDER BY in subquery."""
        # This is a simplified check for ORDER BY in subqueries
        if sql_text.count("SELECT") > 1 and "ORDER BY" in sql_text:
            # Check if ORDER BY might be in subquery (rough approximation)
            if re.search(r"\([^)]*ORDER BY[^)]*\)", sql_text):
                context.issues.append(
                    {
                        "type": "ORDER_BY_IN_SUBQUERY",
                        "severity": "low",
                        "description": "ORDER BY in subquery without LIMIT is usually ignored and wastes resources",
                    }
                )
                context.recommendations.append(
                    {
                        "type": "REMOVE_SUBQUERY_ORDER",
                        "priority": "low",
                        "description": "Remove ORDER BY from subquery unless combined with LIMIT/TOP",
                        "example": "Subquery ORDER BY is only useful with LIMIT: (SELECT ... ORDER BY col LIMIT 10)",
                    }
                )

    def _check_random_ordering(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for RAND()/RANDOM() ordering."""
        random_patterns = ["RAND()", "RANDOM()", "NEWID()"]

        for pattern in random_patterns:
            if pattern in sql_text and "ORDER BY" in sql_text:
                context.issues.append(
                    {
                        "type": "ORDER_BY_RANDOM",
                        "severity": "high",
                        "description": f"ORDER BY {pattern} requires sorting the entire table - very expensive on large tables",  # noqa: E501
                    }
                )
                context.recommendations.append(
                    {
                        "type": "AVOID_RANDOM_ORDERING",
                        "priority": "high",
                        "description": "For random sampling, use alternative methods like sampling algorithms or pre-randomized data",  # noqa: E501
                        "example": "Use: WHERE id >= (SELECT FLOOR(RAND() * MAX(id)) FROM table) LIMIT 1 for single random row",  # noqa: E501
                    }
                )
                break
