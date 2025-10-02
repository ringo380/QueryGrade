"""
Wildcard analyzer for detecting LIKE pattern inefficiencies.

This module analyzes queries to identify:
- LIKE '%pattern%' (leading wildcard - cannot use index)
- LIKE '%pattern' (trailing wildcard only)
- Multiple LIKE in OR conditions
- LIKE with no wildcards (should use =)
- LIKE in correlated subqueries
- REGEXP/RLIKE without index support
"""

import re
from .base import BaseAnalyzer, AnalysisContext


class WildcardAnalyzer(BaseAnalyzer):
    """
    Analyzer for LIKE pattern and wildcard optimization.

    Detects issues such as:
    - LIKE with leading wildcards
    - LIKE with both wildcards
    - Multiple LIKE patterns
    - LIKE without wildcards
    - REGEXP/RLIKE usage
    - LIKE in loops/subqueries
    """

    @property
    def name(self) -> str:
        return "WildcardAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze LIKE patterns for optimization opportunities.

        Args:
            context: AnalysisContext object containing parsed query and results
        """
        sql_text = context.sql_text.upper()

        # Check for LIKE or REGEXP usage
        has_like = 'LIKE' in sql_text
        has_regexp = any(pattern in sql_text for pattern in ['REGEXP', 'RLIKE', 'REGEX'])

        if not (has_like or has_regexp):
            return

        if has_like:
            self._check_like_both_wildcards(sql_text, context)
            self._check_like_no_wildcard(sql_text, context)
            self._check_multiple_like_or(sql_text, context)
            self._check_like_in_loop(sql_text, context)
            self._check_case_sensitivity(sql_text, context)

        if has_regexp:
            self._check_regexp_usage(sql_text, context)

    def _check_like_both_wildcards(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for LIKE with wildcards on both ends."""
        # Pattern: LIKE '%...%'
        if re.search(r"LIKE\s+['\"]%[^'\"]+%['\"]", sql_text):
            context.issues.append({
                'type': 'LIKE_BOTH_WILDCARDS',
                'severity': 'high',
                'description': 'LIKE with wildcards on both ends (LIKE \'%pattern%\') cannot use indexes - requires full table scan'
            })
            context.recommendations.append({
                'type': 'USE_FULLTEXT_SEARCH',
                'priority': 'critical',
                'description': 'Replace LIKE \'%pattern%\' with full-text search for better performance',
                'example': 'CREATE FULLTEXT INDEX idx_text ON table(column); SELECT * FROM table WHERE MATCH(column) AGAINST(\'pattern\')'
            })
            context.recommendations.append({
                'type': 'CONSIDER_SEARCH_ENGINE',
                'priority': 'medium',
                'description': 'For complex text search, consider external search engine (Elasticsearch, Solr)',
                'example': 'Index data in Elasticsearch and search there instead of SQL LIKE'
            })

        # Pattern: LIKE '%...' (leading wildcard only)
        elif re.search(r"LIKE\s+['\"]%[^'\"]+['\"]", sql_text):
            # Check if it's not a trailing wildcard case
            if not re.search(r"LIKE\s+['\"][^'\"]+%['\"]", sql_text):
                context.issues.append({
                    'type': 'LIKE_LEADING_WILDCARD',
                    'severity': 'high',
                    'description': 'LIKE with leading wildcard (LIKE \'%pattern\') cannot use indexes'
                })
                context.recommendations.append({
                    'type': 'FULLTEXT_FOR_LEADING_WILDCARD',
                    'priority': 'high',
                    'description': 'Use full-text search for patterns with leading wildcards',
                    'example': 'CREATE FULLTEXT INDEX idx_text ON table(column); WHERE MATCH(column) AGAINST(\'pattern\')'
                })

        # Pattern: LIKE 'pattern%' (trailing wildcard - can use index)
        if re.search(r"LIKE\s+['\"][^%'][^'\"]*%['\"]", sql_text):
            context.performance_notes.append(
                'LIKE with trailing wildcard only (LIKE \'pattern%\') can use indexes if the column is indexed'
            )

    def _check_like_no_wildcard(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for LIKE without wildcards."""
        # Pattern: LIKE 'value' (no % or _)
        if re.search(r"LIKE\s+['\"][^%_'\"]+['\"]", sql_text):
            context.issues.append({
                'type': 'LIKE_NO_WILDCARD',
                'severity': 'low',
                'description': 'LIKE without wildcards should be replaced with = for better performance and clarity'
            })
            context.recommendations.append({
                'type': 'USE_EQUALITY',
                'priority': 'low',
                'description': 'Replace LIKE with = when no wildcards are present',
                'example': 'WHERE column = \'value\' -- instead of WHERE column LIKE \'value\''
            })

    def _check_multiple_like_or(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for multiple LIKE patterns with OR."""
        # Count LIKE in combination with OR
        like_count = sql_text.count(' LIKE ')
        or_count = sql_text.count(' OR ')

        if like_count >= 3 and or_count >= 2:
            context.issues.append({
                'type': 'MULTIPLE_LIKE_OR',
                'severity': 'medium',
                'description': f'Query has {like_count} LIKE patterns with OR - each requires separate pattern match'
            })
            context.recommendations.append({
                'type': 'FULLTEXT_BOOLEAN_MODE',
                'priority': 'high',
                'description': 'Use full-text search in boolean mode for multiple pattern matching',
                'example': 'WHERE MATCH(column) AGAINST(\'+pattern1 +pattern2 +pattern3\' IN BOOLEAN MODE)'
            })
            context.recommendations.append({
                'type': 'NORMALIZE_SEARCH_DATA',
                'priority': 'medium',
                'description': 'Consider normalizing search data into separate searchable columns or table',
                'example': 'CREATE TABLE search_terms (id INT, term VARCHAR); SELECT ... WHERE id IN (SELECT id FROM search_terms WHERE term IN (...))'
            })

    def _check_like_in_loop(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for LIKE in correlated subquery."""
        # Look for LIKE in subquery with WHERE
        if sql_text.count('SELECT') > 1 and 'LIKE' in sql_text:
            # Simple heuristic: if there's a subquery and LIKE
            if re.search(r'\([^)]*SELECT[^)]*LIKE', sql_text):
                context.issues.append({
                    'type': 'LIKE_IN_LOOP',
                    'severity': 'high',
                    'description': 'LIKE in correlated subquery executes pattern matching for each outer row'
                })
                context.recommendations.append({
                    'type': 'EXTRACT_LIKE_FROM_SUBQUERY',
                    'priority': 'high',
                    'description': 'Move LIKE pattern matching out of subquery using JOIN or CTE',
                    'example': 'WITH matched AS (SELECT id FROM table WHERE column LIKE \'pattern%\') SELECT ... FROM outer JOIN matched ON outer.id = matched.id'
                })

    def _check_case_sensitivity(self, sql_text: str, context: AnalysisContext) -> None:
        """Check LIKE case sensitivity."""
        # Check for UPPER/LOWER with LIKE
        if re.search(r'(UPPER|LOWER)\([^)]+\)\s+LIKE', sql_text):
            context.issues.append({
                'type': 'FUNCTION_WITH_LIKE',
                'severity': 'high',
                'description': 'Using UPPER/LOWER with LIKE prevents index usage'
            })
            context.recommendations.append({
                'type': 'CASE_INSENSITIVE_LIKE',
                'priority': 'high',
                'description': 'Use case-insensitive LIKE (ILIKE in PostgreSQL) or case-insensitive collation',
                'example': 'WHERE column ILIKE \'pattern%\' -- PostgreSQL case-insensitive LIKE'
            })
            context.recommendations.append({
                'type': 'CASE_INSENSITIVE_INDEX',
                'priority': 'high',
                'description': 'Create case-insensitive index or computed column',
                'example': 'CREATE INDEX idx_lower ON table(LOWER(column)); WHERE LOWER(column) LIKE LOWER(\'pattern%\')'
            })

    def _check_regexp_usage(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for REGEXP/RLIKE usage."""
        context.issues.append({
            'type': 'REGEXP_NO_INDEX',
            'severity': 'high',
            'description': 'Regular expressions (REGEXP/RLIKE) cannot use indexes and require full table scan'
        })
        context.recommendations.append({
            'type': 'AVOID_REGEXP',
            'priority': 'critical',
            'description': 'Avoid REGEXP in WHERE clause on large tables - use simpler patterns or full-text search',
            'example': 'Use LIKE for simple patterns or full-text search for complex patterns'
        })

        # Check for complex regex patterns
        if re.search(r'REGEXP\s+[\'"][^\'"]{20,}', sql_text):
            context.recommendations.append({
                'type': 'PREPROCESS_REGEXP',
                'priority': 'high',
                'description': 'For complex regex patterns, preprocess data into searchable format',
                'example': 'Extract relevant fields into separate columns and index them'
            })

    def _check_underscore_wildcard(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for underscore wildcard usage."""
        # Pattern: LIKE '_pattern' or 'pattern_'
        if re.search(r"LIKE\s+['\"]_[^'\"]*['\"]", sql_text):
            context.performance_notes.append(
                'Underscore (_) wildcard in LIKE matches single character. '
                'Like % wildcard, leading _ prevents index usage.'
            )

    def _check_escape_characters(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for ESCAPE clause usage."""
        if 'ESCAPE' in sql_text:
            context.performance_notes.append(
                'ESCAPE clause in LIKE requires additional processing. '
                'Consider if the escape character is necessary.'
            )

    def _check_not_like(self, sql_text: str, context: AnalysisContext) -> None:
        """Check for NOT LIKE usage."""
        if 'NOT LIKE' in sql_text:
            context.issues.append({
                'type': 'NOT_LIKE_PERFORMANCE',
                'severity': 'medium',
                'description': 'NOT LIKE requires checking all rows that don\'t match - can be slow'
            })
            context.recommendations.append({
                'type': 'NOT_LIKE_ALTERNATIVE',
                'priority': 'medium',
                'description': 'Consider alternative approaches for NOT LIKE queries',
                'example': 'Use positive matching with exclusion: WHERE column LIKE \'valid%\' AND column NOT IN (SELECT ...)'
            })