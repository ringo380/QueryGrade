"""
GROUP BY analyzer for detecting aggregation inefficiencies.

This module analyzes queries to identify:
- Missing indexes on GROUP BY columns
- HAVING clause that could be WHERE
- Aggregate functions without GROUP BY
- GROUP BY on expressions
- Unnecessary columns in GROUP BY
"""

import re
from .base import BaseAnalyzer, AnalysisContext


class GroupByAnalyzer(BaseAnalyzer):
    """
    Analyzer for GROUP BY and aggregation optimization.

    Detects issues such as:
    - GROUP BY without indexes
    - HAVING that should be WHERE
    - GROUP BY on expressions
    - Missing columns in SELECT vs GROUP BY
    - COUNT(*) vs COUNT(column)
    """

    @property
    def name(self) -> str:
        return "GroupByAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze GROUP BY clause for optimization opportunities.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()

        # Only analyze if GROUP BY or aggregate functions are present
        has_group_by = 'GROUP BY' in sql_text
        has_aggregates = any(func in sql_text for func in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('])

        if not (has_group_by or has_aggregates):
            return

        if has_group_by:
            self._check_groupby_indexing(sql_text, context)
            self._check_groupby_expressions(sql_text, context)
            self._check_groupby_column_count(sql_text, context)
            self._check_having_vs_where(sql_text, context)
            self._check_groupby_order(sql_text, context)
            self._check_distinct_with_groupby(sql_text, context)

        if has_aggregates:
            self._check_aggregate_functions(sql_text, context)

    def _check_groupby_indexing(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for indexes on GROUP BY columns."""
        context.recommendations.append({
            'type': 'INDEX_GROUP_BY',
            'priority': 'high',
            'description': 'GROUP BY benefits from indexes on grouped columns',
            'example': 'CREATE INDEX idx_grouped_cols ON table_name(group_col1, group_col2)'
        })

        # Check for GROUP BY with multiple columns
        group_by_match = re.search(r'GROUP BY\s+(.+?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|;|$)', sql_text)
        if group_by_match:
            group_clause = group_by_match.group(1)
            comma_count = group_clause.count(',')

            if comma_count >= 2:
                context.recommendations.append({
                    'type': 'COMPOSITE_GROUP_INDEX',
                    'priority': 'high',
                    'description': f'GROUP BY uses {comma_count + 1} columns - composite index can improve performance',
                    'example': 'CREATE INDEX idx_composite_group ON table_name(col1, col2, col3)'
                })

    def _check_groupby_expressions(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for expressions in GROUP BY clause."""
        # Look for GROUP BY with functions or expressions
        expression_patterns = [
            (r'GROUP BY[^(]*DATE\(', 'DATE'),
            (r'GROUP BY[^(]*YEAR\(', 'YEAR'),
            (r'GROUP BY[^(]*MONTH\(', 'MONTH'),
            (r'GROUP BY[^(]*DAY\(', 'DAY'),
            (r'GROUP BY[^(]*UPPER\(', 'UPPER'),
            (r'GROUP BY[^(]*LOWER\(', 'LOWER'),
            (r'GROUP BY[^(]*SUBSTRING\(', 'SUBSTRING'),
            (r'GROUP BY[^(]*CONCAT\(', 'CONCAT'),
        ]

        for pattern, func_name in expression_patterns:
            if re.search(pattern, sql_text):
                context.issues.append({
                    'type': 'FUNCTION_IN_GROUP_BY',
                    'severity': 'medium',
                    'description': f'Using {func_name}() in GROUP BY prevents index usage and requires computing function for all rows'
                })
                context.recommendations.append({
                    'type': 'COMPUTED_COLUMN_FOR_GROUP',
                    'priority': 'medium',
                    'description': f'Consider creating a computed/generated column for {func_name}(column) with an index',
                    'example': 'ALTER TABLE table_name ADD COLUMN grouped_value AS (function(col)) STORED; CREATE INDEX ON table_name(grouped_value);'
                })
                break

        # Check for arithmetic expressions
        if re.search(r'GROUP BY[^(]*[\+\-\*/]', sql_text):
            context.issues.append({
                'type': 'EXPRESSION_IN_GROUP_BY',
                'severity': 'medium',
                'description': 'Arithmetic expression in GROUP BY cannot use indexes'
            })
            context.recommendations.append({
                'type': 'PRECOMPUTE_GROUP_EXPRESSION',
                'priority': 'medium',
                'description': 'Store computed group expression as a column with an index',
                'example': 'ALTER TABLE table_name ADD COLUMN computed_group AS (expression) STORED;'
            })

    def _check_groupby_column_count(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for excessive columns in GROUP BY."""
        group_by_match = re.search(r'GROUP BY\s+(.+?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|;|$)', sql_text)
        if group_by_match:
            group_clause = group_by_match.group(1)
            column_count = group_clause.count(',') + 1

            if column_count >= 5:
                context.issues.append({
                    'type': 'MANY_GROUP_COLUMNS',
                    'severity': 'low',
                    'description': f'GROUP BY has {column_count} columns - may indicate design issue or denormalization opportunity'
                })
                context.recommendations.append({
                    'type': 'REVIEW_GROUP_DESIGN',
                    'priority': 'low',
                    'description': 'Consider whether all GROUP BY columns are necessary or if data model could be optimized',
                    'example': 'Review if grouping requirements indicate missing aggregation table or summary table'
                })

    def _check_having_vs_where(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for HAVING clause that should be WHERE."""
        if 'HAVING' in sql_text:
            # Check if HAVING contains non-aggregate conditions
            having_match = re.search(r'HAVING\s+(.+?)(?:\s+ORDER|\s+LIMIT|;|$)', sql_text)
            if having_match:
                having_clause = having_match.group(1)

                # Simple heuristic: if no aggregate functions in HAVING, it should be WHERE
                has_aggregate_in_having = any(func in having_clause for func in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('])

                if not has_aggregate_in_having:
                    context.issues.append({
                        'type': 'HAVING_WITHOUT_AGGREGATE',
                        'severity': 'medium',
                        'description': 'HAVING clause filters non-aggregated columns - should use WHERE instead for better performance'
                    })
                    context.recommendations.append({
                        'type': 'MOVE_TO_WHERE',
                        'priority': 'high',
                        'description': 'Move non-aggregate filters from HAVING to WHERE clause to filter before grouping',
                        'example': 'WHERE condition AND ... GROUP BY ... HAVING aggregate_condition'
                    })

            context.performance_notes.append(
                'HAVING filters after grouping. Use WHERE to filter before grouping when possible.'
            )

    def _check_groupby_order(self, sql_text: str, context: AnalysisContext) -> None:
        """Check GROUP BY column order for optimization."""
        if 'GROUP BY' in sql_text and 'WHERE' in sql_text:
            context.recommendations.append({
                'type': 'GROUP_COLUMN_ORDER',
                'priority': 'medium',
                'description': 'Order GROUP BY columns by selectivity (most selective first) for better index usage',
                'example': 'GROUP BY high_cardinality_col, low_cardinality_col -- most selective first'
            })

    def _check_aggregate_functions(self, sql_text: str, context: AnalysisContext) -> None:
        """Check aggregate function usage."""
        # Check for COUNT(*) vs COUNT(column)
        if 'COUNT(*)' in sql_text:
            context.recommendations.append({
                'type': 'COUNT_OPTIMIZATION',
                'priority': 'low',
                'description': 'COUNT(*) vs COUNT(column): COUNT(*) includes NULLs, COUNT(column) excludes them',
                'example': 'Use COUNT(*) for total rows, COUNT(column) to count non-NULL values'
            })

        # Check for multiple aggregates
        aggregate_count = sum(sql_text.count(func) for func in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('])
        if aggregate_count >= 5:
            context.recommendations.append({
                'type': 'MANY_AGGREGATES',
                'priority': 'low',
                'description': f'Query computes {aggregate_count} aggregates - consider if all are necessary',
                'example': 'Multiple aggregates can be expensive on large datasets'
            })

        # Check for AVG which can be derived from SUM/COUNT
        if 'AVG(' in sql_text:
            context.recommendations.append({
                'type': 'AVG_ALTERNATIVE',
                'priority': 'low',
                'description': 'AVG can be computed as SUM/COUNT if you need both values',
                'example': 'SELECT SUM(col)/COUNT(col) instead of AVG(col) when you need both sum and count'
            })

        # Check for DISTINCT in aggregate
        if re.search(r'(COUNT|SUM|AVG)\s*\(\s*DISTINCT', sql_text):
            context.issues.append({
                'type': 'DISTINCT_IN_AGGREGATE',
                'severity': 'medium',
                'description': 'DISTINCT inside aggregate function requires deduplication before aggregation'
            })
            context.recommendations.append({
                'type': 'OPTIMIZE_DISTINCT_AGGREGATE',
                'priority': 'medium',
                'description': 'DISTINCT in aggregate functions can be expensive - consider preprocessing or CTEs',
                'example': 'WITH distinct_values AS (SELECT DISTINCT col FROM table) SELECT COUNT(*) FROM distinct_values'
            })

    def _check_distinct_with_groupby(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for DISTINCT with GROUP BY."""
        if 'DISTINCT' in sql_text and 'GROUP BY' in sql_text:
            context.issues.append({
                'type': 'DISTINCT_WITH_GROUP_BY',
                'severity': 'low',
                'description': 'DISTINCT is usually unnecessary when using GROUP BY - they both perform deduplication'
            })
            context.recommendations.append({
                'type': 'REMOVE_DISTINCT',
                'priority': 'low',
                'description': 'Remove DISTINCT when using GROUP BY as they serve the same purpose',
                'example': 'SELECT col1, col2 FROM table GROUP BY col1, col2 -- no DISTINCT needed'
            })