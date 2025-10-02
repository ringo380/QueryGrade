"""
WHERE clause analyzer.

This module analyzes WHERE clause efficiency and identifies performance
issues related to filtering conditions.
"""

import re
from .base import BaseAnalyzer, AnalysisContext


class WhereAnalyzer(BaseAnalyzer):
    """
    Analyzer for WHERE clause efficiency.

    Detects issues such as:
    - Functions on columns in WHERE (prevents index usage)
    - Leading wildcards in LIKE patterns
    - Inequality operators that may limit index effectiveness
    - WHERE clauses comparing constants
    - Multiple OR conditions that might benefit from UNION
    """

    @property
    def name(self) -> str:
        return "WhereAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze WHERE clause and update context with findings.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()

        self._check_functions_on_columns(sql_text, context)
        self._check_leading_wildcards(sql_text, context)
        self._check_multiple_or_conditions(sql_text, context)
        self._check_inequality_operators(sql_text, context)
        self._check_constant_comparisons(sql_text, context)
        self._check_date_range_queries(sql_text, context)

    def _check_functions_on_columns(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for functions on columns in WHERE clause."""
        function_patterns = [
            r'UPPER\([^)]+\)', r'LOWER\([^)]+\)', r'SUBSTRING\([^)]+\)',
            r'CONCAT\([^)]+\)', r'DATE\([^)]+\)', r'YEAR\([^)]+\)', r'MONTH\([^)]+\)'
        ]

        for pattern in function_patterns:
            if re.search(pattern, sql_text):
                context.issues.append({
                    'type': 'FUNCTION_ON_COLUMN',
                    'severity': 'medium',
                    'description': 'Using functions on columns in WHERE clause prevents index usage'
                })
                context.recommendations.append({
                    'type': 'AVOID_FUNCTIONS_ON_COLUMNS',
                    'priority': 'medium',
                    'description': 'Avoid using functions on columns in WHERE conditions',
                    'example': 'Instead of WHERE YEAR(date_col) = 2024, use WHERE date_col >= \'2024-01-01\' AND date_col < \'2025-01-01\''
                })
                break

    def _check_leading_wildcards(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for LIKE with leading wildcard."""
        if re.search(r'LIKE\s+["\'][%]', sql_text):
            context.issues.append({
                'type': 'LEADING_WILDCARD',
                'severity': 'medium',
                'description': 'LIKE with leading wildcard (%) prevents index usage'
            })
            context.recommendations.append({
                'type': 'AVOID_LEADING_WILDCARDS',
                'priority': 'medium',
                'description': 'Avoid starting LIKE patterns with % when possible'
            })

    def _check_multiple_or_conditions(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for multiple OR conditions that might benefit from UNION."""
        if sql_text.count(' OR ') > 2:
            context.recommendations.append({
                'type': 'CONSIDER_UNION_FOR_OR',
                'priority': 'low',
                'description': 'Multiple OR conditions might perform better as UNION queries with proper indexing'
            })

    def _check_inequality_operators(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for inequality operators that might prevent index usage."""
        if re.search(r'!= |<> ', sql_text):
            context.recommendations.append({
                'type': 'INEQUALITY_INDEX_IMPACT',
                'priority': 'low',
                'description': 'Inequality operators (!= or <>) may limit index effectiveness'
            })

    def _check_constant_comparisons(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for WHERE clause with only constants (likely a mistake)."""
        if 'WHERE' in sql_text and re.search(r'WHERE\s+[\'"]\w+[\'"]\s*=\s*[\'"]\w+[\'"]', sql_text):
            context.issues.append({
                'type': 'WHERE_CONSTANT_COMPARISON',
                'severity': 'high',
                'description': 'WHERE clause appears to compare constants instead of columns'
            })

    def _check_date_range_queries(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for date range queries and provide optimization tips."""
        if re.search(r'BETWEEN.*AND.*', sql_text) and 'DATE' in sql_text:
            context.recommendations.append({
                'type': 'DATE_RANGE_OPTIMIZATION',
                'priority': 'low',
                'description': 'For date ranges, ensure proper indexing on date columns for optimal performance'
            })