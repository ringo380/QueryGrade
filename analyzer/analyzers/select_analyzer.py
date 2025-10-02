"""
SELECT clause analyzer.

This module analyzes SELECT clause efficiency and identifies common
performance issues related to column selection.
"""

import re
from .base import BaseAnalyzer, AnalysisContext


class SelectAnalyzer(BaseAnalyzer):
    """
    Analyzer for SELECT clause efficiency.

    Detects issues such as:
    - SELECT * usage
    - Unnecessary DISTINCT
    - Scalar subqueries in SELECT
    - COUNT(*) optimization opportunities
    """

    @property
    def name(self) -> str:
        return "SelectAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze SELECT clause and update context with findings.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()

        self._check_select_star(sql_text, context)
        self._check_unnecessary_distinct(sql_text, context)
        self._check_count_optimization(sql_text, context)
        self._check_scalar_subqueries(sql_text, context)

    def _check_select_star(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for SELECT * usage."""
        if 'SELECT *' in sql_text:
            context.issues.append({
                'type': 'SELECT_STAR',
                'severity': 'medium',
                'description': 'Using SELECT * retrieves all columns, which may be inefficient'
            })
            context.recommendations.append({
                'type': 'SELECT_SPECIFIC',
                'priority': 'medium',
                'description': 'Specify only the columns you need instead of using SELECT *',
                'example': 'Replace "SELECT *" with "SELECT column1, column2, ..."'
            })

    def _check_unnecessary_distinct(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for unnecessary DISTINCT usage."""
        if 'SELECT DISTINCT' in sql_text and 'JOIN' not in sql_text:
            context.recommendations.append({
                'type': 'UNNECESSARY_DISTINCT',
                'priority': 'low',
                'description': 'DISTINCT may be unnecessary without JOINs; verify if duplicates are actually possible'
            })

    def _check_count_optimization(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for COUNT(*) optimization opportunities."""
        if 'COUNT(*)' in sql_text:
            context.recommendations.append({
                'type': 'COUNT_OPTIMIZATION',
                'priority': 'low',
                'description': 'Consider using COUNT(primary_key) instead of COUNT(*) for better performance on some databases'
            })

    def _check_scalar_subqueries(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for scalar subqueries in SELECT clause."""
        # Check for SELECT inside SELECT (nested subqueries in SELECT)
        if sql_text.count('SELECT') > 1 and '(' in sql_text:
            subquery_select_pattern = r'SELECT[^(]*\([^)]*SELECT'
            if re.search(subquery_select_pattern, sql_text):
                context.issues.append({
                    'type': 'SCALAR_SUBQUERY',
                    'severity': 'medium',
                    'description': 'Scalar subqueries in SELECT clause can be performance bottlenecks'
                })
                context.recommendations.append({
                    'type': 'AVOID_SCALAR_SUBQUERIES',
                    'priority': 'medium',
                    'description': 'Consider using JOINs or window functions instead of scalar subqueries'
                })