"""
Window function analyzer for detecting analytical query inefficiencies.

This module analyzes queries to identify:
- Window functions without PARTITION BY (processes entire table)
- Redundant window definitions
- Window functions that could be simple aggregates
- ROW_NUMBER() without ORDER BY (non-deterministic)
- Inefficient PARTITION BY columns
- Multiple window functions with same partition
"""

import re
from .base import BaseAnalyzer, AnalysisContext


class WindowFunctionAnalyzer(BaseAnalyzer):
    """
    Analyzer for window function optimization.

    Detects issues such as:
    - Window functions without PARTITION BY
    - Redundant window definitions
    - ROW_NUMBER() without ORDER BY
    - Window functions that could be GROUP BY
    - Inefficient PARTITION BY columns
    - Named window opportunities
    """

    @property
    def name(self) -> str:
        return "WindowFunctionAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze window functions for optimization opportunities.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()

        # Check for window functions
        has_window_function = any(func in sql_text for func in [
            'ROW_NUMBER()', 'RANK()', 'DENSE_RANK()', 'NTILE(',
            'LAG(', 'LEAD(', 'FIRST_VALUE(', 'LAST_VALUE(',
            ' OVER('  # Generic OVER clause without space
        ])

        # Also check for OVER with space before parenthesis
        if not has_window_function:
            has_window_function = ' OVER (' in sql_text

        if not has_window_function:
            return

        self._check_window_without_partition(sql_text, context)
        self._check_row_number_without_order(sql_text, context)
        self._check_redundant_window_definitions(sql_text, context)
        self._check_window_vs_aggregate(sql_text, context)
        self._check_named_window_opportunity(sql_text, context)
        self._check_partition_by_efficiency(sql_text, context)

    def _check_window_without_partition(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for window functions without PARTITION BY."""
        # Look for OVER() without PARTITION BY
        if re.search(r'OVER\s*\(\s*ORDER\s+BY', sql_text):
            # Has ORDER BY but check if PARTITION BY is missing
            if not re.search(r'PARTITION\s+BY', sql_text):
                context.issues.append({
                    'type': 'WINDOW_NO_PARTITION',
                    'severity': 'medium',
                    'description': 'Window function without PARTITION BY processes entire table - may be slow on large datasets'
                })
                context.recommendations.append({
                    'type': 'ADD_PARTITION_BY',
                    'priority': 'high',
                    'description': 'Add PARTITION BY to limit window function scope to logical groups',
                    'example': 'ROW_NUMBER() OVER (PARTITION BY category ORDER BY date) -- limits scope'
                })

    def _check_row_number_without_order(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for ROW_NUMBER() without ORDER BY."""
        if 'ROW_NUMBER()' in sql_text:
            # Check if ROW_NUMBER has ORDER BY
            if not re.search(r'ROW_NUMBER\(\)\s*OVER\s*\([^)]*ORDER\s+BY', sql_text):
                context.issues.append({
                    'type': 'ROW_NUMBER_NO_ORDER',
                    'severity': 'high',
                    'description': 'ROW_NUMBER() without ORDER BY produces non-deterministic results'
                })
                context.recommendations.append({
                    'type': 'ADD_ORDER_BY_TO_ROW_NUMBER',
                    'priority': 'critical',
                    'description': 'Always specify ORDER BY with ROW_NUMBER() for deterministic results',
                    'example': 'ROW_NUMBER() OVER (PARTITION BY category ORDER BY date DESC)'
                })

    def _check_redundant_window_definitions(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for repeated window definitions."""
        # Count OVER clauses - match both "OVER(" and "OVER ("
        over_count = sql_text.count(' OVER(') + sql_text.count(' OVER (')

        if over_count >= 3:
            # Check if WINDOW clause exists
            if 'WINDOW' not in sql_text:
                context.recommendations.append({
                    'type': 'USE_NAMED_WINDOW',
                    'priority': 'medium',
                    'description': f'Query has {over_count} window functions - consider using named windows (WINDOW clause)',
                    'example': 'WINDOW w AS (PARTITION BY category ORDER BY date) then use: SUM(amount) OVER w'
                })

    def _check_window_vs_aggregate(self, sql_text: str, context: AnalysisContext) -> None:
        """Check if window function could be replaced with simple aggregate."""
        # Look for window function with aggregates but no ranking/offset functions
        has_window_aggregate = any(func in sql_text for func in [
            'SUM(', 'AVG(', 'COUNT(', 'MAX(', 'MIN('
        ]) and ' OVER(' in sql_text

        has_ranking = any(func in sql_text for func in [
            'ROW_NUMBER()', 'RANK()', 'DENSE_RANK()', 'LAG(', 'LEAD('
        ])

        if has_window_aggregate and not has_ranking:
            # Check if there's no non-window columns (which would require window function)
            context.recommendations.append({
                'type': 'WINDOW_VS_GROUP_BY',
                'priority': 'low',
                'description': 'If you only need aggregates without row-level detail, consider GROUP BY instead of window functions',
                'example': 'SELECT category, SUM(amount) FROM table GROUP BY category -- simpler than window function'
            })

    def _check_named_window_opportunity(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for opportunities to use WINDOW clause."""
        # Find OVER clauses and check for similar patterns
        over_clauses = re.findall(r'OVER\s*\([^)]+\)', sql_text)

        if len(over_clauses) >= 2:
            # Check if any are identical
            from collections import Counter
            clause_counts = Counter(over_clauses)

            for clause, count in clause_counts.items():
                if count >= 2:
                    context.recommendations.append({
                        'type': 'DEFINE_NAMED_WINDOW',
                        'priority': 'medium',
                        'description': f'Same window definition used {count}x - define as named window',
                        'example': f'WINDOW w AS {clause} then use: function() OVER w'
                    })
                    break

    def _check_partition_by_efficiency(self, sql_text: str, context: AnalysisContext) -> None:
        """Check PARTITION BY column efficiency."""
        if 'PARTITION BY' in sql_text:
            context.recommendations.append({
                'type': 'INDEX_PARTITION_COLUMNS',
                'priority': 'high',
                'description': 'Ensure PARTITION BY columns are indexed for efficient window function execution',
                'example': 'CREATE INDEX idx_partition ON table(partition_column, order_column)'
            })

            # Check for PARTITION BY with functions - look for function name followed by (
            # Match patterns like YEAR(col), DATE(col), UPPER(col), etc.
            if re.search(r'PARTITION\s+BY\s+[A-Z_]+\s*\(', sql_text):
                context.issues.append({
                    'type': 'FUNCTION_IN_PARTITION_BY',
                    'severity': 'medium',
                    'description': 'Function in PARTITION BY prevents index usage'
                })
                context.recommendations.append({
                    'type': 'COMPUTED_PARTITION_COLUMN',
                    'priority': 'medium',
                    'description': 'Create computed/generated column for function used in PARTITION BY',
                    'example': 'ALTER TABLE table ADD COLUMN partition_value AS (function(col)) STORED; CREATE INDEX ON table(partition_value);'
                })

    def _check_frame_clause(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for frame clause usage."""
        # Check for ROWS/RANGE frame clauses
        if 'ROWS BETWEEN' in sql_text or 'RANGE BETWEEN' in sql_text:
            context.performance_notes.append(
                'Window frame clause (ROWS/RANGE BETWEEN) requires additional processing. '
                'Ensure it\'s necessary for your calculation.'
            )

            # Check for unbounded frame
            if 'UNBOUNDED' in sql_text:
                context.recommendations.append({
                    'type': 'LIMIT_WINDOW_FRAME',
                    'priority': 'low',
                    'description': 'Unbounded window frames process many rows - consider limiting frame size if possible',
                    'example': 'ROWS BETWEEN 10 PRECEDING AND CURRENT ROW -- instead of UNBOUNDED PRECEDING'
                })

    def _check_window_function_types(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for specific window function patterns."""
        # Check for NTILE
        if 'NTILE(' in sql_text:
            context.recommendations.append({
                'type': 'NTILE_USAGE',
                'priority': 'low',
                'description': 'NTILE creates equal-sized buckets - ensure PARTITION BY for accurate distribution',
                'example': 'NTILE(4) OVER (PARTITION BY category ORDER BY amount) -- quartiles per category'
            })

        # Check for LAG/LEAD
        if 'LAG(' in sql_text or 'LEAD(' in sql_text:
            context.recommendations.append({
                'type': 'LAG_LEAD_INDEX',
                'priority': 'medium',
                'description': 'LAG/LEAD requires sorted data - index ORDER BY columns',
                'example': 'CREATE INDEX idx_lag ON table(partition_col, order_col)'
            })

        # Check for FIRST_VALUE/LAST_VALUE
        if 'FIRST_VALUE(' in sql_text or 'LAST_VALUE(' in sql_text:
            if 'RANGE BETWEEN' not in sql_text and 'ROWS BETWEEN' not in sql_text:
                context.recommendations.append({
                    'type': 'SPECIFY_FRAME_CLAUSE',
                    'priority': 'medium',
                    'description': 'FIRST_VALUE/LAST_VALUE may need explicit frame clause for expected results',
                    'example': 'LAST_VALUE(col) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)'
                })

    def _check_window_with_distinct(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for DISTINCT with window functions."""
        if 'DISTINCT' in sql_text and ' OVER(' in sql_text:
            context.performance_notes.append(
                'DISTINCT with window functions may not produce expected results. '
                'Window functions are applied before DISTINCT elimination.'
            )