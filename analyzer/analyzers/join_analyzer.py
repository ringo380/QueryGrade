"""
JOIN analyzer.

This module analyzes JOIN patterns and efficiency, detecting issues
such as excessive joins, Cartesian products, and missing join conditions.
"""

import re

from .base import AnalysisContext, BaseAnalyzer


class JoinAnalyzer(BaseAnalyzer):
    """
    Analyzer for JOIN efficiency and patterns.

    Detects issues such as:
    - Excessive number of joins
    - Cartesian products (comma-separated tables)
    - Joins without proper conditions
    - Join order optimization opportunities
    """

    @property
    def name(self) -> str:
        return "JoinAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze JOIN patterns and update context with findings.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()
        join_count = context.query.join_count

        self._check_excessive_joins(join_count, context)
        self._check_cartesian_products(sql_text, context)
        self._check_join_conditions(sql_text, context)

    def _check_excessive_joins(self, join_count: int, context: AnalysisContext) -> None:
        """Check for excessive number of joins."""
        if join_count > 4:
            context.issues.append(
                {
                    "type": "EXCESSIVE_JOINS",
                    "severity": "high",
                    "description": f"Query has {join_count} joins, which may impact performance",
                }
            )
            context.recommendations.append(
                {
                    "type": "REDUCE_JOINS",
                    "priority": "high",
                    "description": "Consider denormalizing data or using materialized views for complex joins",
                }
            )

    def _check_cartesian_products(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for Cartesian products (comma-separated tables without proper join conditions)."""
        # Check for implicit joins (comma-separated tables)
        comma_join_pattern = r"\bFROM\s+[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*|FROM\s+[a-zA-Z_][a-zA-Z0-9_]*\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*"  # noqa: E501
        if re.search(comma_join_pattern, sql_text):
            context.issues.append(
                {
                    "type": "CARTESIAN_PRODUCT",
                    "severity": "catastrophic",
                    "description": "Comma-separated tables without JOIN conditions create Cartesian product",
                }
            )
            context.recommendations.append(
                {
                    "type": "USE_EXPLICIT_JOINS",
                    "priority": "critical",
                    "description": "Use explicit JOIN syntax with proper ON conditions instead of comma-separated tables",  # noqa: E501
                    "example": 'Replace "FROM table1, table2" with "FROM table1 INNER JOIN table2 ON table1.id = table2.id"',  # noqa: E501
                }
            )

    def _check_join_conditions(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for joins without proper conditions."""
        # Skip CROSS JOINs (intentional Cartesian products)
        if "CROSS JOIN" in sql_text:
            return

        # Find all JOINs
        join_pattern = r"\b(LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|FULL\s+)?JOIN\s+"
        joins = list(re.finditer(join_pattern, sql_text))

        # For each JOIN, check if there's an ON or USING before the next major clause
        for join_match in joins:
            start_pos = join_match.end()
            # Find the next major SQL keyword
            next_clause_pattern = (
                r"\b(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION|JOIN|;)\b"
            )
            next_clause = re.search(next_clause_pattern, sql_text[start_pos:])

            # Check for ON or USING between this JOIN and the next clause
            if next_clause:
                segment = sql_text[start_pos : start_pos + next_clause.start()]
            else:
                segment = sql_text[start_pos:]

            # Check if ON or USING appears in this segment
            if not re.search(r"\b(ON|USING)\b", segment):
                context.issues.append(
                    {
                        "type": "JOIN_WITHOUT_CONDITION",
                        "severity": "catastrophic",
                        "description": "JOIN without proper conditions may create Cartesian product",
                    }
                )
                context.recommendations.append(
                    {
                        "type": "ADD_JOIN_CONDITIONS",
                        "priority": "critical",
                        "description": "Always specify JOIN conditions using ON or USING clauses",
                    }
                )
                break  # Only report once even if multiple JOINs lack conditions
