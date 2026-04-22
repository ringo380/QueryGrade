"""
Indexing analyzer for detecting missing indexes and index optimization opportunities.

This module analyzes queries to identify:
- WHERE clauses without indexes
- ORDER BY on unindexed columns
- JOIN conditions without indexes
- Function usage on indexed columns (breaks index)
- Index hints and usage
"""

import re

from .base import AnalysisContext, BaseAnalyzer


class IndexingAnalyzer(BaseAnalyzer):
    """
    Analyzer for index optimization opportunities.

    Detects issues such as:
    - WHERE clauses that could benefit from indexes
    - ORDER BY without supporting indexes
    - JOIN conditions missing indexes
    - Functions applied to indexed columns
    - Missing composite indexes
    """

    @property
    def name(self) -> str:
        return "IndexingAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze query for indexing opportunities.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()

        self._check_where_indexing(sql_text, context)
        self._check_join_indexing(sql_text, context)
        self._check_orderby_indexing(sql_text, context)
        self._check_function_on_columns(sql_text, context)
        self._check_composite_index_opportunities(sql_text, context)
        self._check_index_hints(sql_text, context)

    def _check_where_indexing(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for WHERE clauses that need indexes."""
        if "WHERE" in sql_text:
            # Check for common patterns that benefit from indexes
            if re.search(r"WHERE\s+\w+\s*=", sql_text):
                context.recommendations.append(
                    {
                        "type": "INDEX_WHERE",
                        "priority": "high",
                        "description": "Consider adding indexes on columns used in WHERE clause equality conditions",
                        "example": "CREATE INDEX idx_column_name ON table_name(column_name)",
                    }
                )

            # Check for range queries
            if re.search(r"WHERE\s+\w+\s*[<>]", sql_text) or "BETWEEN" in sql_text:
                context.recommendations.append(
                    {
                        "type": "INDEX_RANGE",
                        "priority": "high",
                        "description": "Range queries (>, <, BETWEEN) benefit from B-tree indexes",
                        "example": "CREATE INDEX idx_date_range ON table_name(date_column)",
                    }
                )

            # Check for LIKE with leading wildcard
            if re.search(r"LIKE\s+['\"]%", sql_text):
                context.issues.append(
                    {
                        "type": "LIKE_LEADING_WILDCARD",
                        "severity": "high",
                        "description": "LIKE with leading wildcard (LIKE '%...') cannot use indexes effectively",
                    }
                )
                context.recommendations.append(
                    {
                        "type": "FULL_TEXT_SEARCH",
                        "priority": "medium",
                        "description": "Consider using full-text search indexes for LIKE queries with leading wildcards",  # noqa: E501
                        "example": "CREATE FULLTEXT INDEX idx_text_search ON table_name(text_column)",
                    }
                )

    def _check_join_indexing(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for JOIN conditions that need indexes."""
        if "JOIN" in sql_text:
            # Check for JOIN with ON clause
            if "ON" in sql_text:
                context.recommendations.append(
                    {
                        "type": "INDEX_JOIN",
                        "priority": "high",
                        "description": "Ensure both columns in JOIN condition are indexed for optimal performance",
                        "example": "CREATE INDEX idx_foreign_key ON table_name(foreign_key_column)",
                    }
                )

            # Check for multiple JOINs
            join_count = sql_text.count("JOIN")
            if join_count >= 3:
                context.recommendations.append(
                    {
                        "type": "COMPOSITE_JOIN_INDEX",
                        "priority": "high",
                        "description": f"Query has {join_count} JOINs - consider composite indexes for frequently joined columns",  # noqa: E501
                        "example": "CREATE INDEX idx_composite ON table_name(col1, col2, col3)",
                    }
                )

    def _check_orderby_indexing(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for ORDER BY without supporting indexes."""
        if "ORDER BY" in sql_text:
            context.recommendations.append(
                {
                    "type": "INDEX_ORDER_BY",
                    "priority": "medium",
                    "description": "ORDER BY can benefit significantly from indexes on the sorted columns",
                    "example": "CREATE INDEX idx_sorted_column ON table_name(sort_column)",
                }
            )

            # Check for ORDER BY with multiple columns
            if re.search(r"ORDER BY[^)]*,", sql_text):
                context.recommendations.append(
                    {
                        "type": "COMPOSITE_ORDER_INDEX",
                        "priority": "medium",
                        "description": "Multi-column ORDER BY benefits from composite index in the same order",
                        "example": "CREATE INDEX idx_sort_composite ON table_name(col1, col2, col3)",
                    }
                )

            # Check for ORDER BY DESC
            if "DESC" in sql_text and "ORDER BY" in sql_text:
                context.recommendations.append(
                    {
                        "type": "INDEX_DESC_ORDER",
                        "priority": "low",
                        "description": "Consider creating descending indexes for DESC order queries on some databases",
                        "example": "CREATE INDEX idx_desc ON table_name(column DESC)",
                    }
                )

    def _check_function_on_columns(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for functions applied to columns in WHERE clause."""
        # Common functions that break indexes
        function_patterns = [
            (r"WHERE\s+UPPER\(", "UPPER"),
            (r"WHERE\s+LOWER\(", "LOWER"),
            (r"WHERE\s+DATE\(", "DATE"),
            (r"WHERE\s+YEAR\(", "YEAR"),
            (r"WHERE\s+MONTH\(", "MONTH"),
            (r"WHERE\s+SUBSTRING\(", "SUBSTRING"),
            (r"WHERE\s+CONCAT\(", "CONCAT"),
            (r"WHERE\s+TRIM\(", "TRIM"),
        ]

        for pattern, func_name in function_patterns:
            if re.search(pattern, sql_text):
                context.issues.append(
                    {
                        "type": "FUNCTION_ON_INDEXED_COLUMN",
                        "severity": "high",
                        "description": f"Using {func_name}() on column in WHERE clause prevents index usage",
                    }
                )
                context.recommendations.append(
                    {
                        "type": "AVOID_FUNCTION_IN_WHERE",
                        "priority": "high",
                        "description": f"Avoid applying {func_name}() to columns in WHERE clause",
                        "example": f"Instead of WHERE {func_name}(col)=value, normalize data or use computed columns",
                    }
                )
                break  # Only report once

    def _check_composite_index_opportunities(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for opportunities to use composite indexes."""
        # Check for WHERE with multiple conditions
        where_and_count = sql_text.count(" AND ")
        if where_and_count >= 2:
            context.recommendations.append(
                {
                    "type": "COMPOSITE_INDEX",
                    "priority": "high",
                    "description": f"Query has {where_and_count + 1} AND conditions - composite index may improve performance",  # noqa: E501
                    "example": "CREATE INDEX idx_composite ON table_name(col1, col2, col3) -- ordered by selectivity",
                }
            )

        # Check for WHERE + ORDER BY combination
        if "WHERE" in sql_text and "ORDER BY" in sql_text:
            context.recommendations.append(
                {
                    "type": "COVERING_INDEX",
                    "priority": "medium",
                    "description": "Consider covering index including WHERE columns + ORDER BY columns + SELECT columns",  # noqa: E501
                    "example": "CREATE INDEX idx_covering ON table_name(where_col, order_col) INCLUDE (select_col)",
                }
            )

    def _check_index_hints(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for explicit index hints."""
        hint_patterns = [
            "USE INDEX",
            "FORCE INDEX",
            "IGNORE INDEX",
            "WITH (INDEX",
        ]

        for hint in hint_patterns:
            if hint in sql_text:
                context.performance_notes.append(
                    f"Query uses explicit index hint ({hint}). Verify this is still optimal as data distribution changes."  # noqa: E501
                )
                context.recommendations.append(
                    {
                        "type": "INDEX_HINT_REVIEW",
                        "priority": "low",
                        "description": "Explicit index hints can become suboptimal over time - review periodically",
                        "example": "Let query optimizer choose index unless proven necessary",
                    }
                )
                break
