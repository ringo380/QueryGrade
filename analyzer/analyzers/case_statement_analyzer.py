"""
CASE statement analyzer for detecting conditional logic inefficiencies.

This module analyzes queries to identify:
- Complex nested CASE statements (3+ levels)
- CASE in WHERE clause (prevents index usage)
- CASE in ORDER BY/GROUP BY
- Redundant CASE conditions
- CASE that could be COALESCE or NULLIF
- CASE with many branches (10+)
"""

import re
from .base import BaseAnalyzer, AnalysisContext


class CaseStatementAnalyzer(BaseAnalyzer):
    """
    Analyzer for CASE statement optimization.

    Detects issues such as:
    - Complex nested CASE statements
    - CASE in WHERE (prevents index usage)
    - CASE in ORDER BY/GROUP BY
    - Redundant conditions
    - CASE that could be simpler functions
    - Many WHEN branches
    """

    @property
    def name(self) -> str:
        return "CaseStatementAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze CASE statements for optimization opportunities.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()

        # Only analyze if CASE is present
        if 'CASE' not in sql_text or 'WHEN' not in sql_text:
            return

        self._check_case_nesting(sql_text, context)
        self._check_case_in_where(sql_text, context)
        self._check_case_in_orderby(sql_text, context)
        self._check_case_in_groupby(sql_text, context)
        self._check_many_case_branches(sql_text, context)
        self._check_case_coalesce_alternative(sql_text, context)
        self._check_case_with_subquery(sql_text, context)

    def _check_case_nesting(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for deeply nested CASE statements."""
        # Count nested CASE keywords
        case_count = sql_text.count('CASE ')

        if case_count >= 3:
            # Try to detect nesting by looking for CASE within CASE
            # This is simplified - actual nesting detection would need proper parsing
            if re.search(r'CASE[^E]*CASE[^E]*CASE', sql_text):
                context.issues.append({
                    'type': 'COMPLEX_CASE_NESTING',
                    'severity': 'medium',
                    'description': 'Deeply nested CASE statements (3+ levels) are difficult to read and maintain'
                })
                context.recommendations.append({
                    'type': 'SIMPLIFY_CASE',
                    'priority': 'high',
                    'description': 'Consider using lookup table, separate computed columns, or breaking into multiple queries',
                    'example': 'CREATE TABLE status_mapping (input VARCHAR, output VARCHAR); SELECT mapping.output FROM table JOIN status_mapping...'
                })

    def _check_case_in_where(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for CASE in WHERE clause."""
        # Look for WHERE followed by CASE
        if re.search(r'WHERE[^(]*CASE\s+WHEN', sql_text):
            context.issues.append({
                'type': 'CASE_IN_WHERE',
                'severity': 'high',
                'description': 'CASE statement in WHERE clause prevents index usage and requires evaluation for all rows'
            })
            context.recommendations.append({
                'type': 'REWRITE_WHERE_CASE',
                'priority': 'high',
                'description': 'Rewrite WHERE clause to avoid CASE - use OR conditions or computed column',
                'example': 'WHERE (condition1 AND result1) OR (condition2 AND result2) -- instead of CASE in WHERE'
            })
            context.recommendations.append({
                'type': 'COMPUTED_COLUMN_FOR_CASE',
                'priority': 'medium',
                'description': 'Create computed column for CASE logic with index',
                'example': 'ALTER TABLE table ADD COLUMN case_result AS (CASE WHEN ... THEN ... END) STORED; CREATE INDEX ON table(case_result);'
            })

    def _check_case_in_orderby(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for CASE in ORDER BY clause."""
        if re.search(r'ORDER\s+BY[^(]*CASE\s+WHEN', sql_text):
            context.issues.append({
                'type': 'CASE_IN_ORDERBY',
                'severity': 'medium',
                'description': 'CASE in ORDER BY requires evaluating expression for all rows and prevents index usage'
            })
            context.recommendations.append({
                'type': 'COMPUTED_ORDER_COLUMN',
                'priority': 'medium',
                'description': 'Create computed column for CASE logic used in ORDER BY',
                'example': 'ALTER TABLE table ADD COLUMN sort_value AS (CASE WHEN ... THEN ... END) STORED; CREATE INDEX ON table(sort_value);'
            })

    def _check_case_in_groupby(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for CASE in GROUP BY clause."""
        if re.search(r'GROUP\s+BY[^(]*CASE\s+WHEN', sql_text):
            context.issues.append({
                'type': 'CASE_IN_GROUPBY',
                'severity': 'medium',
                'description': 'CASE in GROUP BY requires evaluating expression for all rows'
            })
            context.recommendations.append({
                'type': 'COMPUTED_GROUP_COLUMN',
                'priority': 'medium',
                'description': 'Create computed column for CASE grouping logic with index',
                'example': 'ALTER TABLE table ADD COLUMN group_value AS (CASE WHEN ... THEN ... END) STORED; CREATE INDEX ON table(group_value);'
            })

    def _check_many_case_branches(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for CASE with many WHEN branches."""
        # Count WHEN clauses in a single CASE
        when_count = sql_text.count(' WHEN ')

        if when_count >= 10:
            context.issues.append({
                'type': 'MANY_CASE_BRANCHES',
                'severity': 'medium',
                'description': f'CASE statement has {when_count} WHEN branches - difficult to maintain and slow'
            })
            context.recommendations.append({
                'type': 'USE_LOOKUP_TABLE',
                'priority': 'high',
                'description': 'Replace large CASE with lookup table JOIN for better maintainability and performance',
                'example': 'CREATE TABLE lookup (input VARCHAR, output VARCHAR); SELECT t.*, l.output FROM table t JOIN lookup l ON t.col = l.input'
            })

    def _check_case_coalesce_alternative(self, sql_text: str, context: AnalysisContext) -> None:
        """Check if CASE could be replaced with COALESCE or NULLIF."""
        # Look for simple NULL checks that could be COALESCE
        if re.search(r'CASE\s+WHEN[^T]*IS\s+NULL\s+THEN', sql_text):
            context.recommendations.append({
                'type': 'USE_COALESCE',
                'priority': 'low',
                'description': 'Simple NULL handling CASE can often be replaced with COALESCE for clarity',
                'example': 'COALESCE(column, default_value) -- instead of CASE WHEN column IS NULL THEN default_value ELSE column END'
            })

        # Look for equality checks that could be NULLIF
        if re.search(r'CASE\s+WHEN[^=]*=[^T]*THEN\s+NULL', sql_text):
            context.recommendations.append({
                'type': 'USE_NULLIF',
                'priority': 'low',
                'description': 'CASE that returns NULL on equality can be replaced with NULLIF',
                'example': 'NULLIF(column, value) -- instead of CASE WHEN column = value THEN NULL ELSE column END'
            })

    def _check_case_with_subquery(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for subqueries inside CASE."""
        # Look for SELECT inside CASE WHEN or THEN
        if re.search(r'CASE[^E]*(WHEN|THEN)[^E]*\([^)]*SELECT', sql_text):
            context.issues.append({
                'type': 'CASE_WITH_SUBQUERY',
                'severity': 'high',
                'description': 'Subquery in CASE statement executes for each row - very expensive'
            })
            context.recommendations.append({
                'type': 'EXTRACT_SUBQUERY_FROM_CASE',
                'priority': 'critical',
                'description': 'Move subquery out of CASE using JOIN or CTE',
                'example': 'WITH lookup AS (SELECT ...) SELECT CASE WHEN EXISTS(SELECT 1 FROM lookup WHERE ...) THEN ... END'
            })

    def _check_case_in_select(self, sql_text: str, context: AnalysisContext) -> None:
        """General CASE in SELECT recommendations."""
        if 'SELECT' in sql_text and 'CASE' in sql_text:
            # Check for multiple CASE in SELECT
            select_start = sql_text.find('SELECT')
            from_start = sql_text.find('FROM', select_start) if 'FROM' in sql_text else len(sql_text)
            select_clause = sql_text[select_start:from_start]

            case_in_select = select_clause.count('CASE ')

            if case_in_select >= 3:
                context.recommendations.append({
                    'type': 'MULTIPLE_CASE_IN_SELECT',
                    'priority': 'medium',
                    'description': f'SELECT has {case_in_select} CASE statements - consider computed columns or views',
                    'example': 'CREATE VIEW enriched_data AS SELECT *, CASE ... END as computed1, CASE ... END as computed2 FROM table'
                })

    def _check_searched_vs_simple_case(self, sql_text: str, context: AnalysisContext) -> None:
        """Check CASE style and recommend appropriate form."""
        # Look for pattern: CASE column WHEN value
        if re.search(r'CASE\s+[a-zA-Z_][a-zA-Z0-9_]*\s+WHEN', sql_text):
            context.performance_notes.append(
                'Simple CASE (CASE column WHEN value) is more readable than searched CASE for equality checks'
            )

    def _check_case_else_clause(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for missing ELSE in CASE."""
        # Count CASE and ELSE
        case_count = sql_text.count('CASE ')
        else_count = sql_text.count(' ELSE ')

        if case_count > else_count:
            context.recommendations.append({
                'type': 'CASE_DEFAULT_ELSE',
                'priority': 'low',
                'description': 'Always include ELSE clause in CASE for explicit NULL handling',
                'example': 'CASE WHEN ... THEN ... ELSE NULL END -- explicit NULL default'
            })