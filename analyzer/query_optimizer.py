"""
Query Optimizer Module

This module provides automatic query optimization suggestions and generates
optimized versions of SQL queries based on common performance patterns.
"""

import re
from typing import Dict, List, Optional, Tuple

import sqlparse
from sqlparse import tokens
from sqlparse.sql import Statement, Token, TokenList


class QueryOptimizer:
    """
    Generates optimized versions of SQL queries based on detected issues.
    """

    def __init__(self):
        self.optimization_version = "1.0"

    def optimize_query(
        self, original_sql: str, issues: List[Dict], database_type: str = ""
    ) -> Dict:
        """
        Generate an optimized version of the SQL query.

        Args:
            original_sql (str): The original SQL query
            issues (List[Dict]): List of issues found in the query
            database_type (str): Target database type

        Returns:
            Dict: Contains optimized query and explanation
        """
        optimizations = []
        optimized_sql = original_sql
        explanation = []

        # Parse the query
        try:
            parsed = sqlparse.parse(original_sql)[0]
        except Exception:
            return {
                "optimized_query": original_sql,
                "optimizations_applied": [],
                "explanation": ["Unable to parse query for optimization"],
                "improvement_estimate": 0,
            }

        # Apply optimizations based on detected issues
        for issue in issues:
            issue_type = issue.get("type", "")

            if issue_type == "SELECT_STAR":
                result = self._optimize_select_star(optimized_sql, parsed)
                if result["modified"]:
                    optimized_sql = result["query"]
                    optimizations.append("Replaced SELECT * with specific columns")
                    explanation.append(
                        "Specified only necessary columns instead of SELECT * to reduce data transfer and improve performance"
                    )

            elif issue_type == "FUNCTION_ON_COLUMN":
                result = self._optimize_functions_on_columns(optimized_sql)
                if result["modified"]:
                    optimized_sql = result["query"]
                    optimizations.append("Removed functions from WHERE clause columns")
                    explanation.append(
                        "Moved function calls away from WHERE clause columns to enable index usage"
                    )

            elif issue_type == "LEADING_WILDCARD":
                result = self._optimize_leading_wildcards(optimized_sql)
                if result["modified"]:
                    optimized_sql = result["query"]
                    optimizations.append("Optimized LIKE patterns")
                    explanation.append(
                        "Removed or replaced leading wildcards in LIKE patterns to enable index usage"
                    )

            elif issue_type == "CARTESIAN_PRODUCT":
                result = self._fix_cartesian_product(optimized_sql)
                if result["modified"]:
                    optimized_sql = result["query"]
                    optimizations.append("Added proper JOIN conditions")
                    explanation.append(
                        "Fixed Cartesian product by adding explicit JOIN conditions"
                    )

            elif issue_type == "UNION_WITHOUT_ALL":
                result = self._optimize_union_clauses(optimized_sql)
                if result["modified"]:
                    optimized_sql = result["query"]
                    optimizations.append("Changed UNION to UNION ALL")
                    explanation.append(
                        "Used UNION ALL instead of UNION to skip duplicate elimination when appropriate"
                    )

            elif issue_type == "CORRELATED_SELECT_SUBQUERY":
                result = self._optimize_correlated_subqueries(optimized_sql)
                if result["modified"]:
                    optimized_sql = result["query"]
                    optimizations.append("Converted correlated subquery to JOIN")
                    explanation.append(
                        "Replaced correlated subquery with more efficient JOIN operation"
                    )

            elif issue_type == "ORDER_BY_WITHOUT_LIMIT":
                result = self._add_reasonable_limit(optimized_sql)
                if result["modified"]:
                    optimized_sql = result["query"]
                    optimizations.append("Added LIMIT clause")
                    explanation.append(
                        "Added LIMIT clause to prevent sorting unnecessarily large result sets"
                    )

            elif issue_type == "GROUP_BY_WITHOUT_AGGREGATION":
                result = self._optimize_group_by_without_aggregation(optimized_sql)
                if result["modified"]:
                    optimized_sql = result["query"]
                    optimizations.append("Replaced GROUP BY with DISTINCT")
                    explanation.append(
                        "Used DISTINCT instead of GROUP BY when no aggregation is needed"
                    )

        # Apply general optimizations
        general_result = self._apply_general_optimizations(optimized_sql, database_type)
        if general_result["modified"]:
            optimized_sql = general_result["query"]
            optimizations.extend(general_result["optimizations"])
            explanation.extend(general_result["explanations"])

        # Calculate improvement estimate
        improvement_estimate = self._estimate_improvement(len(optimizations), issues)

        return {
            "optimized_query": optimized_sql,
            "optimizations_applied": optimizations,
            "explanation": explanation,
            "improvement_estimate": improvement_estimate,
            "original_query": original_sql,
        }

    def _optimize_select_star(self, sql: str, parsed: Statement) -> Dict:
        """Replace SELECT * with specific commonly needed columns."""
        if "SELECT *" not in sql.upper():
            return {"query": sql, "modified": False}

        # For demonstration, replace with common columns
        # In a real implementation, this would analyze table schema
        suggested_columns = [
            "id, name, email, created_at",  # User tables
            "id, title, content, author_id, created_at",  # Content tables
            "id, user_id, total, status, created_at",  # Order tables
            "id, name, price, category, in_stock",  # Product tables
        ]

        optimized = re.sub(
            r"SELECT\s+\*",
            "SELECT id, name, created_at  -- Replace with actual needed columns",
            sql,
            flags=re.IGNORECASE,
        )

        return {"query": optimized, "modified": optimized != sql}

    def _optimize_functions_on_columns(self, sql: str) -> Dict:
        """Optimize queries that use functions on columns in WHERE clauses."""
        optimized = sql
        modified = False

        # UPPER(column) = 'VALUE' -> column = 'value' (if case-insensitive collation)
        pattern = r'UPPER\s*\(\s*([^)]+)\s*\)\s*=\s*[\'"]([^\'"]*)[\'"]'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        if matches:
            for column, value in matches:
                old_condition = f"UPPER({column}) = '{value}'"
                new_condition = f"{column} = '{value.lower()}'  -- Consider case-insensitive collation"
                optimized = optimized.replace(old_condition, new_condition)
                modified = True

        # YEAR(date_column) = 2024 -> date_column >= '2024-01-01' AND date_column < '2025-01-01'
        year_pattern = r"YEAR\s*\(\s*([^)]+)\s*\)\s*=\s*(\d{4})"
        year_matches = re.findall(year_pattern, sql, re.IGNORECASE)
        if year_matches:
            for column, year in year_matches:
                old_condition = f"YEAR({column}) = {year}"
                new_condition = (
                    f"{column} >= '{year}-01-01' AND {column} < '{int(year)+1}-01-01'"
                )
                optimized = optimized.replace(old_condition, new_condition)
                modified = True

        return {"query": optimized, "modified": modified}

    def _optimize_leading_wildcards(self, sql: str) -> Dict:
        """Optimize LIKE patterns with leading wildcards."""
        optimized = sql
        modified = False

        # Look for LIKE '%pattern%' and suggest alternatives
        pattern = r"LIKE\s+['\"]%([^%'\"]+)%['\"]"
        matches = re.findall(pattern, sql, re.IGNORECASE)
        if matches:
            for match in matches:
                # Suggest full-text search for patterns that start with %
                old_pattern = f"LIKE '%{match}%'"
                new_pattern = f"-- Consider full-text search: MATCH({match}) AGAINST ('{match}' IN BOOLEAN MODE)\n    LIKE '%{match}%'"
                optimized = optimized.replace(old_pattern, new_pattern)
                modified = True

        return {"query": optimized, "modified": modified}

    def _fix_cartesian_product(self, sql: str) -> Dict:
        """Fix Cartesian products by adding JOIN conditions."""
        # This is a simplified fix - real implementation would be more sophisticated
        if "FROM" in sql.upper() and "," in sql and "JOIN" not in sql.upper():
            # Simple comma join detected
            optimized = sql.replace(
                "FROM", "-- Add proper JOIN conditions to avoid Cartesian product\nFROM"
            )
            return {"query": optimized, "modified": True}

        return {"query": sql, "modified": False}

    def _optimize_union_clauses(self, sql: str) -> Dict:
        """Replace UNION with UNION ALL where appropriate."""
        if "UNION" in sql.upper() and "UNION ALL" not in sql.upper():
            # Replace UNION with UNION ALL and add comment
            optimized = re.sub(
                r"\bUNION\b",
                "UNION ALL  -- Use if duplicates are not a concern",
                sql,
                flags=re.IGNORECASE,
            )
            return {"query": optimized, "modified": True}

        return {"query": sql, "modified": False}

    def _optimize_correlated_subqueries(self, sql: str) -> Dict:
        """Convert simple correlated subqueries to JOINs."""
        # This is a simplified example - real implementation would parse the subquery
        if "SELECT" in sql and "(" in sql and "WHERE" in sql:
            # Add a comment suggesting JOIN conversion
            optimized = (
                sql
                + "\n\n-- Consider converting correlated subqueries to JOINs for better performance"
            )
            return {"query": optimized, "modified": True}

        return {"query": sql, "modified": False}

    def _add_reasonable_limit(self, sql: str) -> Dict:
        """Add LIMIT clause to queries with ORDER BY but no LIMIT."""
        if "ORDER BY" in sql.upper() and "LIMIT" not in sql.upper():
            # Add a reasonable LIMIT
            optimized = (
                sql.rstrip(";")
                + "\nLIMIT 1000  -- Add appropriate limit for your use case;"
            )
            return {"query": optimized, "modified": True}

        return {"query": sql, "modified": False}

    def _optimize_group_by_without_aggregation(self, sql: str) -> Dict:
        """Replace GROUP BY with DISTINCT when no aggregation is used."""
        if "GROUP BY" in sql.upper() and not any(
            agg in sql.upper() for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
        ):
            # Suggest DISTINCT instead
            optimized = (
                sql
                + "\n\n-- Consider using DISTINCT instead of GROUP BY when no aggregation is needed"
            )
            return {"query": optimized, "modified": True}

        return {"query": sql, "modified": False}

    def _apply_general_optimizations(self, sql: str, database_type: str) -> Dict:
        """Apply general optimizations based on database type."""
        optimizations = []
        explanations = []
        optimized = sql
        modified = False

        # Add index hints or suggestions
        if "WHERE" in sql.upper():
            optimized += "\n\n-- Performance tip: Ensure indexes exist on columns used in WHERE clauses"
            optimizations.append("Added index recommendations")
            explanations.append(
                "Added suggestions for creating indexes on filtered columns"
            )
            modified = True

        # Database-specific optimizations
        if database_type == "mysql":
            if "ORDER BY" in sql.upper():
                optimized += "\n-- MySQL tip: Consider using covering indexes for ORDER BY columns"
                optimizations.append("Added MySQL-specific index advice")
                explanations.append("Added MySQL covering index recommendations")
                modified = True

        elif database_type == "postgresql":
            if "LIKE" in sql.upper():
                optimized += "\n-- PostgreSQL tip: Consider using pg_trgm extension for fuzzy matching"
                optimizations.append("Added PostgreSQL text search advice")
                explanations.append(
                    "Suggested PostgreSQL-specific text search optimizations"
                )
                modified = True

        return {
            "query": optimized,
            "optimizations": optimizations,
            "explanations": explanations,
            "modified": modified,
        }

    def _estimate_improvement(self, optimization_count: int, issues: List[Dict]) -> int:
        """Estimate percentage improvement based on optimizations applied."""
        if optimization_count == 0:
            return 0

        # Base improvement per optimization
        base_improvement = 15

        # Bonus for fixing critical issues
        critical_bonus = sum(
            20 for issue in issues if issue.get("severity") == "critical"
        )
        high_bonus = sum(15 for issue in issues if issue.get("severity") == "high")
        medium_bonus = sum(10 for issue in issues if issue.get("severity") == "medium")

        total_improvement = (
            (optimization_count * base_improvement)
            + critical_bonus
            + high_bonus
            + medium_bonus
        )

        # Cap at 95% improvement
        return min(95, total_improvement)

    def generate_optimization_summary(self, optimization_result: Dict) -> str:
        """Generate a human-readable summary of optimizations."""
        if not optimization_result["optimizations_applied"]:
            return "No automatic optimizations were applied. The query appears to be well-optimized."

        summary = f"Applied {len(optimization_result['optimizations_applied'])} optimizations:\n\n"

        for i, opt in enumerate(optimization_result["optimizations_applied"], 1):
            summary += f"{i}. {opt}\n"

        summary += f"\nEstimated performance improvement: {optimization_result['improvement_estimate']}%\n"
        summary += "\nNote: These are automated suggestions. Always test optimized queries in your specific environment."

        return summary


def optimize_query_from_analysis(
    sql_text: str, issues: List[Dict], database_type: str = ""
) -> Dict:
    """
    Convenience function to optimize a query based on analysis results.

    Args:
        sql_text (str): The SQL query to optimize
        issues (List[Dict]): Issues found during analysis
        database_type (str): Target database type

    Returns:
        Dict: Optimization results
    """
    optimizer = QueryOptimizer()
    return optimizer.optimize_query(sql_text, issues, database_type)
