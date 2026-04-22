"""
MySQL-specific analyzer.

This module analyzes MySQL-specific syntax, features, and optimization
patterns.
"""

from ..base import AnalysisContext, BaseAnalyzer


class MySQLAnalyzer(BaseAnalyzer):
    """
    Analyzer for MySQL-specific patterns and optimizations.

    Detects issues such as:
    - Incorrect syntax (TOP instead of LIMIT)
    - Inefficient date functions
    - Storage engine recommendations
    - MySQL-specific optimization opportunities
    """

    @property
    def name(self) -> str:
        return "MySQLAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze MySQL-specific patterns and update context with findings.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        # Only analyze if database type is MySQL
        if context.database_type.lower() not in ["mysql", "mariadb", ""]:
            return

        sql_text = context.sql_text.upper()

        self._check_syntax_errors(sql_text, context)
        self._check_date_functions(sql_text, context)
        self._check_storage_engine(context)
        self._check_index_optimization(sql_text, context)
        self._check_general_optimizations(sql_text, context)

    def _check_syntax_errors(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for SQL Server syntax that doesn't work in MySQL."""
        if "TOP " in sql_text:
            context.issues.append(
                {
                    "type": "MYSQL_SYNTAX_ERROR",
                    "severity": "high",
                    "description": "MySQL uses LIMIT, not TOP for limiting results",
                }
            )
            context.recommendations.append(
                {
                    "type": "USE_MYSQL_LIMIT",
                    "priority": "high",
                    "description": "Replace TOP with LIMIT for MySQL compatibility",
                }
            )

    def _check_date_functions(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for inefficient date functions."""
        if "DATEPART(" in sql_text or "DATEDIFF(" in sql_text:
            context.recommendations.append(
                {
                    "type": "USE_MYSQL_DATE_FUNCTIONS",
                    "priority": "medium",
                    "description": "Use MySQL date functions like YEAR(), MONTH(), DATE_SUB() for better performance",
                }
            )

    def _check_storage_engine(self, context: AnalysisContext) -> None:
        """Provide storage engine recommendations."""
        context.recommendations.append(
            {
                "type": "MYSQL_STORAGE_ENGINE",
                "priority": "low",
                "description": "Consider using InnoDB for ACID compliance and row-level locking",
            }
        )

    def _check_index_optimization(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Check for MySQL-specific indexing opportunities."""
        if "ORDER BY" in sql_text and "LIMIT" in sql_text:
            context.recommendations.append(
                {
                    "type": "MYSQL_INDEX_OPTIMIZATION",
                    "priority": "medium",
                    "description": "Ensure proper indexes exist for ORDER BY columns when using LIMIT",
                }
            )

    def _check_general_optimizations(
        self, sql_text: str, context: AnalysisContext
    ) -> None:
        """Provide general MySQL optimization recommendations."""
        if "SELECT" in sql_text:
            context.recommendations.append(
                {
                    "type": "MYSQL_PERFORMANCE_OPTIMIZATION",
                    "priority": "low",
                    "description": "Consider MySQL-specific optimizations like proper storage engine selection and query cache usage",  # noqa: E501
                }
            )
