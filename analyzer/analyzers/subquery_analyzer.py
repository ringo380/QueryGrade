"""
Subquery analyzer for detecting inefficient subquery patterns.

This module analyzes queries to identify:
- Correlated subqueries that can be rewritten as JOINs
- Subqueries in SELECT clause (scalar subqueries)
- IN/EXISTS subquery optimization opportunities
- Common Table Expressions (CTE) opportunities
- Nested subquery depth
"""

import re
from .base import BaseAnalyzer, AnalysisContext


class SubqueryAnalyzer(BaseAnalyzer):
    """
    Analyzer for subquery optimization opportunities.

    Detects issues such as:
    - Correlated subqueries (dependent subqueries)
    - Scalar subqueries in SELECT
    - IN vs EXISTS optimization
    - Deep nesting of subqueries
    - CTE (WITH clause) opportunities
    """

    @property
    def name(self) -> str:
        return "SubqueryAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze query for subquery optimization opportunities.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()

        # Check if query has subqueries
        if '(' not in sql_text or 'SELECT' not in sql_text:
            return

        subquery_count = context.query.subquery_count
        if subquery_count > 0:
            self._check_correlated_subqueries(sql_text, context)
            self._check_scalar_subqueries(sql_text, context)
            self._check_in_vs_exists(sql_text, context)
            self._check_subquery_depth(sql_text, context, subquery_count)
            self._check_cte_opportunities(sql_text, context, subquery_count)
            self._check_not_in_with_nulls(sql_text, context)

    def _check_correlated_subqueries(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for correlated subqueries that reference outer query."""
        # Look for common correlated subquery patterns
        # This is a simplified check - real correlation is complex to detect
        if re.search(r'WHERE\s+\w+\s+IN\s*\(\s*SELECT', sql_text):
            # Check if subquery likely references outer table
            if sql_text.count('SELECT') > 1 and sql_text.count('WHERE') > 1:
                context.issues.append({
                    'type': 'CORRELATED_SUBQUERY',
                    'severity': 'high',
                    'description': 'Correlated subquery detected - runs once for each outer row'
                })
                context.recommendations.append({
                    'type': 'REWRITE_AS_JOIN',
                    'priority': 'high',
                    'description': 'Consider rewriting correlated subquery as JOIN for better performance',
                    'example': 'Replace WHERE col IN (SELECT...) with INNER JOIN or LEFT JOIN'
                })

        # Check for EXISTS with correlated condition
        if 'EXISTS' in sql_text and sql_text.count('SELECT') > 1:
            context.recommendations.append({
                'type': 'EXISTS_OPTIMIZATION',
                'priority': 'medium',
                'description': 'EXISTS is generally more efficient than IN for correlated subqueries',
                'example': 'Use WHERE EXISTS (SELECT 1 FROM...) instead of WHERE col IN (SELECT col FROM...)'
            })

    def _check_scalar_subqueries(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for scalar subqueries in SELECT clause."""
        # Detect SELECT with subquery in select list
        if re.search(r'SELECT[^(]*\([^)]*SELECT', sql_text):
            context.issues.append({
                'type': 'SCALAR_SUBQUERY_IN_SELECT',
                'severity': 'high',
                'description': 'Scalar subquery in SELECT clause executes for each row returned'
            })
            context.recommendations.append({
                'type': 'ELIMINATE_SCALAR_SUBQUERY',
                'priority': 'high',
                'description': 'Replace scalar subquery with LEFT JOIN or window function',
                'example': 'SELECT t1.*, t2.value FROM t1 LEFT JOIN t2 ON t1.id = t2.id'
            })

    def _check_in_vs_exists(self, sql_text: str, context: AnalysisContext) -> None:
        """Check IN vs EXISTS usage and suggest optimization."""
        has_in_subquery = bool(re.search(r'IN\s*\(\s*SELECT', sql_text))
        has_exists = 'EXISTS' in sql_text

        if has_in_subquery and not has_exists:
            context.recommendations.append({
                'type': 'IN_VS_EXISTS',
                'priority': 'medium',
                'description': 'Consider using EXISTS instead of IN for better performance with large subquery results',
                'example': 'WHERE EXISTS (SELECT 1 FROM table WHERE condition) instead of WHERE col IN (SELECT col FROM table)'
            })

        # Check for NOT IN
        if 'NOT IN' in sql_text:
            context.issues.append({
                'type': 'NOT_IN_SUBQUERY',
                'severity': 'medium',
                'description': 'NOT IN can be slow and behaves unexpectedly with NULL values'
            })
            context.recommendations.append({
                'type': 'USE_NOT_EXISTS',
                'priority': 'high',
                'description': 'Use NOT EXISTS or LEFT JOIN with IS NULL instead of NOT IN',
                'example': 'WHERE NOT EXISTS (SELECT 1 FROM table WHERE condition)'
            })

    def _check_subquery_depth(self, sql_text: str, context: AnalysisContext, subquery_count: int) -> None:
        """Check for deeply nested subqueries."""
        if subquery_count >= 3:
            context.issues.append({
                'type': 'DEEP_SUBQUERY_NESTING',
                'severity': 'medium',
                'description': f'Query has {subquery_count} levels of subquery nesting - difficult to optimize and maintain'
            })
            context.recommendations.append({
                'type': 'SIMPLIFY_SUBQUERIES',
                'priority': 'medium',
                'description': 'Break complex nested subqueries into CTEs (WITH clause) for readability and optimization',
                'example': 'WITH cte1 AS (...), cte2 AS (...) SELECT ... FROM cte1 JOIN cte2'
            })

    def _check_cte_opportunities(self, sql_text: str, context: AnalysisContext, subquery_count: int) -> None:
        """Check for opportunities to use Common Table Expressions."""
        has_with = 'WITH' in sql_text and 'AS' in sql_text

        if subquery_count >= 2 and not has_with:
            context.recommendations.append({
                'type': 'USE_CTE',
                'priority': 'medium',
                'description': 'Multiple subqueries detected - consider using CTEs (WITH clause) for better readability and potential optimization',
                'example': 'WITH subquery_name AS (SELECT ...) SELECT ... FROM subquery_name'
            })

        # Check for repeated subqueries (same subquery used multiple times)
        if sql_text.count('SELECT') >= 4:
            context.recommendations.append({
                'type': 'CTE_FOR_REUSE',
                'priority': 'medium',
                'description': 'If the same subquery is used multiple times, extract it to a CTE to avoid duplicate computation',
                'example': 'WITH repeated_logic AS (SELECT ...) SELECT ... FROM repeated_logic JOIN repeated_logic'
            })

    def _check_not_in_with_nulls(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for NOT IN which can have unexpected behavior with NULLs."""
        if 'NOT IN' in sql_text:
            context.performance_notes.append(
                'NOT IN returns no results if the subquery contains any NULL values. '
                'Use NOT EXISTS or LEFT JOIN ... WHERE ... IS NULL instead.'
            )
            context.recommendations.append({
                'type': 'NOT_IN_NULL_HANDLING',
                'priority': 'high',
                'description': 'NOT IN with NULLs can produce unexpected results',
                'example': 'Use: WHERE NOT EXISTS (SELECT 1 FROM table WHERE condition) OR WHERE t1.id NOT IN (SELECT id FROM t2 WHERE id IS NOT NULL)'
            })

    def _check_subquery_in_from(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for subqueries in FROM clause (derived tables)."""
        if re.search(r'FROM\s*\(\s*SELECT', sql_text):
            context.recommendations.append({
                'type': 'DERIVED_TABLE',
                'priority': 'low',
                'description': 'Derived table (subquery in FROM) detected - ensure it\'s necessary and consider CTE for clarity',
                'example': 'WITH derived AS (SELECT ...) SELECT ... FROM derived'
            })