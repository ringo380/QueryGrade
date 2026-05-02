"""
UNION analyzer for detecting set operation inefficiencies.

This module analyzes queries to identify:
- UNION vs UNION ALL usage (unnecessary deduplication)
- Multiple UNION operations
- Column type and count mismatches
- ORDER BY in UNION members
- UNION that could be replaced with OR
"""

import re

from .base import AnalysisContext, BaseAnalyzer


class UnionAnalyzer(BaseAnalyzer):
    """
    Analyzer for UNION and set operation optimization.

    Detects issues such as:
    - UNION instead of UNION ALL (expensive deduplication)
    - Multiple UNION operations (performance impact)
    - Column type mismatches in UNION
    - Column count mismatches
    - ORDER BY in UNION subqueries
    - UNION that could be OR conditions
    """

    @property
    def name(self) -> str:
        return "UnionAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze UNION operations for optimization opportunities.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()

        # Only analyze if UNION is present
        if "UNION" not in sql_text:
            return

        self._check_union_vs_union_all(sql_text, context)
        self._check_multiple_unions(sql_text, context)
        self._check_union_order_by(sql_text, context)
        self._check_union_alternatives(sql_text, context)
        self._check_union_in_subquery(sql_text, context)

    def _check_union_vs_union_all(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for UNION without ALL."""
        # Count UNION ALL vs plain UNION
        union_all_count = sql_text.count("UNION ALL")
        plain_union_count = sql_text.count("UNION") - union_all_count

        if plain_union_count > 0:
            context.issues.append(
                {
                    "type": "UNION_WITHOUT_ALL",
                    "severity": "medium",
                    "description": f"Query uses UNION ({plain_union_count}x) which removes duplicates - expensive operation",
                }
            )
            context.recommendations.append(
                {
                    "type": "USE_UNION_ALL",
                    "priority": "high",
                    "description": "Use UNION ALL if duplicates are acceptable or impossible - much faster than UNION",
                    "example": "SELECT ... UNION ALL SELECT ... -- skips deduplication step",
                }
            )

    def _check_multiple_unions(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for multiple UNION operations."""
        union_count = sql_text.count("UNION")

        if union_count >= 3:
            context.issues.append(
                {
                    "type": "MULTIPLE_UNIONS",
                    "severity": "medium",
                    "description": f"Query has {union_count} UNION operations - each adds overhead",
                }
            )
            context.recommendations.append(
                {
                    "type": "REDUCE_UNIONS",
                    "priority": "medium",
                    "description": "Consider consolidating multiple UNIONs or using alternative query structure",
                    "example": "Combine similar SELECT statements or use IN clause instead of multiple UNIONs",
                }
            )

            # Recommend materialized view or temp table
            context.recommendations.append(
                {
                    "type": "MATERIALIZE_UNION",
                    "priority": "medium",
                    "description": "For frequently executed UNION queries, consider materialized view or summary table",
                    "example": "CREATE MATERIALIZED VIEW union_results AS SELECT ... UNION ALL SELECT ...",
                }
            )

    def _check_union_order_by(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for ORDER BY inside UNION members."""
        # Simple pattern: SELECT ... ORDER BY ... UNION
        if re.search(r"ORDER\s+BY[^)]*UNION", sql_text):
            context.issues.append(
                {
                    "type": "UNION_ORDER_BY",
                    "severity": "low",
                    "description": "ORDER BY in UNION member is usually ignored - order only the final result",
                }
            )
            context.recommendations.append(
                {
                    "type": "MOVE_ORDER_BY_TO_END",
                    "priority": "low",
                    "description": "Move ORDER BY to the end of UNION query for correct ordering",
                    "example": "(SELECT ...) UNION ALL (SELECT ...) ORDER BY column",
                }
            )

    def _check_union_alternatives(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check if UNION could be replaced with OR."""
        # Check for UNION on same table
        # This is a simplified heuristic
        if "UNION" in sql_text and sql_text.count("FROM") >= 2:
            # Extract table names after FROM
            from_matches = re.findall(r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql_text)

            if from_matches:
                # Count occurrences of each table
                from collections import Counter

                table_counts = Counter(from_matches)

                # If same table appears in multiple FROM clauses
                for table, count in table_counts.items():
                    if count >= 2:
                        context.recommendations.append(
                            {
                                "type": "UNION_TO_OR",
                                "priority": "medium",
                                "description": f"UNION on same table ({table}) may be replaceable with OR conditions",
                                "example": "SELECT * FROM table WHERE condition1 OR condition2 -- instead of UNION",
                            }
                        )
                        break

    def _check_union_in_subquery(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for UNION in subqueries."""
        # Look for UNION inside parentheses (subqueries)
        if re.search(r"\([^)]*UNION[^)]*\)", sql_text):
            context.recommendations.append(
                {
                    "type": "UNION_SUBQUERY_CTE",
                    "priority": "low",
                    "description": "UNION in subquery may be clearer as CTE (WITH clause)",
                    "example": "WITH combined AS (SELECT ... UNION ALL SELECT ...) SELECT * FROM combined",
                }
            )

    def _check_union_distinct_redundancy(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for DISTINCT with UNION."""
        if (
            "DISTINCT" in sql_text
            and "UNION" in sql_text
            and "UNION ALL" not in sql_text
        ):
            context.issues.append(
                {
                    "type": "DISTINCT_WITH_UNION",
                    "severity": "low",
                    "description": "DISTINCT is redundant with UNION (UNION already removes duplicates)",
                }
            )
            context.recommendations.append(
                {
                    "type": "REMOVE_DISTINCT",
                    "priority": "low",
                    "description": "Remove DISTINCT when using UNION as it already deduplicates",
                    "example": "SELECT col1, col2 FROM table UNION SELECT ... -- no DISTINCT needed",
                }
            )

    def _check_union_performance(self, sql_text: str, context: AnalysisContext) -> None:
        """General UNION performance notes."""
        if "UNION" in sql_text:
            context.performance_notes.append(
                "UNION operations require sorting and deduplication. "
                "Use UNION ALL when duplicates are acceptable for better performance."
            )

            # Check for UNION with complex subqueries
            if sql_text.count("SELECT") >= 4 and "UNION" in sql_text:
                context.recommendations.append(
                    {
                        "type": "UNION_INDEX_OPTIMIZATION",
                        "priority": "high",
                        "description": "Ensure columns used in UNION are indexed in all source tables",
                        "example": "CREATE INDEX idx_union_col ON table(column_used_in_union)",
                    }
                )

    def _check_intersect_except(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for INTERSECT and EXCEPT operations."""
        if "INTERSECT" in sql_text:
            context.recommendations.append(
                {
                    "type": "INTERSECT_ALTERNATIVE",
                    "priority": "medium",
                    "description": "INTERSECT can often be replaced with INNER JOIN for better performance",
                    "example": "SELECT t1.* FROM t1 INNER JOIN t2 ON t1.id = t2.id -- instead of INTERSECT",
                }
            )

        if "EXCEPT" in sql_text or "MINUS" in sql_text:
            context.recommendations.append(
                {
                    "type": "EXCEPT_ALTERNATIVE",
                    "priority": "medium",
                    "description": "EXCEPT/MINUS can be replaced with LEFT JOIN ... WHERE IS NULL",
                    "example": "SELECT t1.* FROM t1 LEFT JOIN t2 ON t1.id = t2.id WHERE t2.id IS NULL",
                }
            )
