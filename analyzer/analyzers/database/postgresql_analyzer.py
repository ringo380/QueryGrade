"""
PostgreSQL-specific analyzer.

This module analyzes PostgreSQL-specific syntax, features, and optimization
patterns.
"""

from ..base import BaseAnalyzer, AnalysisContext


class PostgreSQLAnalyzer(BaseAnalyzer):
    """
    Analyzer for PostgreSQL-specific patterns and optimizations.

    Detects issues such as:
    - Incorrect syntax (TOP instead of LIMIT)
    - Opportunities for DISTINCT ON
    - Full-text search opportunities
    - Window function recommendations
    """

    @property
    def name(self) -> str:
        return "PostgreSQLAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze PostgreSQL-specific patterns and update context with findings.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        # Only analyze if database type is PostgreSQL
        if context.database_type.lower() not in ['postgresql', 'postgres', '']:
            return

        sql_text = context.sql_text.upper()

        self._check_syntax_errors(sql_text, context)
        self._check_distinct_on_opportunities(sql_text, context)
        self._check_text_search_opportunities(sql_text, context)
        self._check_window_function_opportunities(sql_text, context)
        self._check_general_optimizations(sql_text, context)

    def _check_syntax_errors(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for SQL Server syntax that doesn't work in PostgreSQL."""
        if 'TOP ' in sql_text:
            context.issues.append({
                'type': 'POSTGRESQL_SYNTAX_ERROR',
                'severity': 'high',
                'description': 'PostgreSQL uses LIMIT, not TOP for limiting results'
            })

    def _check_distinct_on_opportunities(self, sql_text: str, context: AnalysisContext) -> None:
        """Suggest PostgreSQL DISTINCT ON for efficient distinct queries."""
        if 'DISTINCT' in sql_text:
            context.recommendations.append({
                'type': 'USE_POSTGRESQL_DISTINCT_ON',
                'priority': 'low',
                'description': 'Consider using DISTINCT ON for more efficient distinct queries in PostgreSQL'
            })

    def _check_text_search_opportunities(self, sql_text: str, context: AnalysisContext) -> None:
        """Recommend PostgreSQL full-text search for complex text queries."""
        if 'LIKE' in sql_text and "'%" in sql_text:
            context.recommendations.append({
                'type': 'POSTGRESQL_TEXT_SEARCH',
                'priority': 'medium',
                'description': 'Consider PostgreSQL full-text search for complex text queries'
            })

    def _check_window_function_opportunities(self, sql_text: str, context: AnalysisContext) -> None:
        """Suggest window functions for complex aggregations."""
        if 'ORDER BY' in sql_text and 'GROUP BY' in sql_text:
            context.recommendations.append({
                'type': 'USE_POSTGRESQL_WINDOW_FUNCTIONS',
                'priority': 'low',
                'description': 'PostgreSQL window functions can often replace complex GROUP BY queries'
            })

    def _check_general_optimizations(self, sql_text: str, context: AnalysisContext) -> None:
        """Provide general PostgreSQL optimization recommendations."""
        if 'SELECT' in sql_text:
            context.recommendations.append({
                'type': 'POSTGRESQL_PERFORMANCE_OPTIMIZATION',
                'priority': 'low',
                'description': 'Consider PostgreSQL-specific optimizations like proper indexing and query planning'
            })