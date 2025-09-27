"""
Feature Extractor for QueryGrade ML System

This module extracts numerical features from SQL queries for machine learning models.
Features include structural, complexity, and semantic characteristics of queries.
"""

import re
import logging
import hashlib
from typing import List, Dict, Optional, Any
import sqlparse
from sqlparse import tokens
from sqlparse.sql import Statement, Token, TokenList

from ..models import Query

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts numerical features from SQL queries for ML training and prediction."""

    def __init__(self):
        self.feature_names = [
            # Basic structure features
            'query_length',
            'token_count',
            'keyword_count',
            'identifier_count',

            # Complexity features
            'table_count',
            'join_count',
            'where_conditions',
            'subquery_count',
            'aggregate_function_count',
            'window_function_count',

            # Query type features (one-hot encoded)
            'is_select',
            'is_insert',
            'is_update',
            'is_delete',
            'is_create',
            'is_alter',

            # Advanced features
            'nesting_level',
            'function_call_count',
            'case_statement_count',
            'union_count',
            'having_clause_present',
            'order_by_present',
            'group_by_present',
            'distinct_present',
            'limit_present',

            # Performance indicators
            'select_star_present',
            'cross_join_present',
            'cartesian_product_risk',
            'missing_where_clause',
            'potential_index_usage',

            # Semantic features
            'avg_identifier_length',
            'keyword_diversity',
            'operator_count',
            'comparison_operator_count',
            'logical_operator_count',

            # Database-specific features
            'mysql_specific_syntax',
            'postgresql_specific_syntax',
            'sqlite_specific_syntax',
            'oracle_specific_syntax',
            'sqlserver_specific_syntax',

            # Pattern features
            'has_correlated_subquery',
            'has_exists_clause',
            'has_in_clause',
            'has_like_pattern',
            'has_null_check',
        ]

    def extract_features(self, query: Query, database_type: str = '') -> Optional[List[float]]:
        """
        Extract numerical features from a query object.

        Args:
            query: Query object to extract features from
            database_type: Database type for database-specific features

        Returns:
            List of numerical features or None if extraction fails
        """
        try:
            sql_text = query.sql_text
            parsed_statements = sqlparse.parse(sql_text)

            if not parsed_statements:
                logger.warning(f"Could not parse query {query.id}")
                return None

            parsed = parsed_statements[0]
            features = []

            # Extract all feature categories
            features.extend(self._extract_basic_features(sql_text, parsed))
            features.extend(self._extract_complexity_features(query, parsed))
            features.extend(self._extract_query_type_features(parsed))
            features.extend(self._extract_advanced_features(sql_text, parsed))
            features.extend(self._extract_performance_features(sql_text, parsed))
            features.extend(self._extract_semantic_features(sql_text, parsed))
            features.extend(self._extract_database_specific_features(sql_text, database_type))
            features.extend(self._extract_pattern_features(sql_text, parsed))

            # Validate feature count
            if len(features) != len(self.feature_names):
                logger.error(f"Feature count mismatch: {len(features)} != {len(self.feature_names)}")
                return None

            return features

        except Exception as e:
            logger.error(f"Error extracting features from query {query.id}: {str(e)}")
            return None

    def _extract_basic_features(self, sql_text: str, parsed: Statement) -> List[float]:
        """Extract basic structural features."""
        tokens = list(parsed.flatten())

        # Count different token types
        keyword_count = sum(1 for token in tokens if token.ttype in tokens.Keyword)
        identifier_count = sum(1 for token in tokens if token.ttype in (tokens.Name, tokens.Name.Builtin))

        return [
            float(len(sql_text)),  # query_length
            float(len(tokens)),    # token_count
            float(keyword_count),  # keyword_count
            float(identifier_count),  # identifier_count
        ]

    def _extract_complexity_features(self, query: Query, parsed: Statement) -> List[float]:
        """Extract complexity-related features."""
        sql_text = str(parsed).upper()

        # Count aggregate functions
        aggregate_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT', 'STRING_AGG']
        aggregate_count = sum(sql_text.count(f' {func}(') for func in aggregate_functions)

        # Count window functions
        window_functions = ['ROW_NUMBER', 'RANK', 'DENSE_RANK', 'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE']
        window_count = sum(sql_text.count(f' {func}(') for func in window_functions)

        return [
            float(query.table_count),      # table_count
            float(query.join_count),       # join_count
            float(query.where_conditions), # where_conditions
            float(query.subquery_count),   # subquery_count
            float(aggregate_count),        # aggregate_function_count
            float(window_count),          # window_function_count
        ]

    def _extract_query_type_features(self, parsed: Statement) -> List[float]:
        """Extract one-hot encoded query type features."""
        query_type = 'UNKNOWN'

        # Determine query type
        for token in parsed.flatten():
            if token.ttype in tokens.Keyword.DML:
                query_type = token.value.upper()
                break
            elif token.ttype in tokens.Keyword.DDL:
                query_type = token.value.upper()
                break

        return [
            float(query_type == 'SELECT'),  # is_select
            float(query_type == 'INSERT'),  # is_insert
            float(query_type == 'UPDATE'),  # is_update
            float(query_type == 'DELETE'),  # is_delete
            float(query_type == 'CREATE'),  # is_create
            float(query_type == 'ALTER'),   # is_alter
        ]

    def _extract_advanced_features(self, sql_text: str, parsed: Statement) -> List[float]:
        """Extract advanced structural features."""
        sql_upper = sql_text.upper()

        # Calculate nesting level (count of parentheses depth)
        nesting_level = 0
        max_nesting = 0
        for char in sql_text:
            if char == '(':
                nesting_level += 1
                max_nesting = max(max_nesting, nesting_level)
            elif char == ')':
                nesting_level -= 1

        # Count various SQL constructs
        function_calls = len(re.findall(r'\w+\s*\(', sql_text))
        case_statements = sql_upper.count('CASE ')
        union_count = sql_upper.count('UNION')

        # Check for clause presence
        having_present = 'HAVING' in sql_upper
        order_by_present = 'ORDER BY' in sql_upper
        group_by_present = 'GROUP BY' in sql_upper
        distinct_present = 'DISTINCT' in sql_upper
        limit_present = any(keyword in sql_upper for keyword in ['LIMIT', 'TOP', 'FETCH'])

        return [
            float(max_nesting),           # nesting_level
            float(function_calls),        # function_call_count
            float(case_statements),       # case_statement_count
            float(union_count),           # union_count
            float(having_present),        # having_clause_present
            float(order_by_present),      # order_by_present
            float(group_by_present),      # group_by_present
            float(distinct_present),      # distinct_present
            float(limit_present),         # limit_present
        ]

    def _extract_performance_features(self, sql_text: str, parsed: Statement) -> List[float]:
        """Extract performance-related features."""
        sql_upper = sql_text.upper()

        # Performance anti-patterns
        select_star = 'SELECT *' in sql_upper
        cross_join = 'CROSS JOIN' in sql_upper

        # Cartesian product risk (JOIN without ON clause in some contexts)
        cartesian_risk = (sql_upper.count('JOIN') > 0 and
                         sql_upper.count('ON ') < sql_upper.count('JOIN'))

        # Missing WHERE clause in UPDATE/DELETE
        missing_where = (any(keyword in sql_upper for keyword in ['UPDATE ', 'DELETE ']) and
                        'WHERE' not in sql_upper)

        # Potential index usage (presence of WHERE with column references)
        where_with_columns = ('WHERE' in sql_upper and
                             len(re.findall(r'WHERE\s+\w+\s*[=<>]', sql_upper)) > 0)

        return [
            float(select_star),           # select_star_present
            float(cross_join),            # cross_join_present
            float(cartesian_risk),        # cartesian_product_risk
            float(missing_where),         # missing_where_clause
            float(where_with_columns),    # potential_index_usage
        ]

    def _extract_semantic_features(self, sql_text: str, parsed: Statement) -> List[float]:
        """Extract semantic and linguistic features."""
        tokens = list(parsed.flatten())

        # Calculate average identifier length
        identifiers = [token.value for token in tokens
                      if token.ttype in (tokens.Name, tokens.Name.Builtin)
                      and len(token.value.strip()) > 0]
        avg_identifier_length = (sum(len(ident) for ident in identifiers) / len(identifiers)
                               if identifiers else 0)

        # Keyword diversity (unique keywords / total keywords)
        keywords = [token.value.upper() for token in tokens if token.ttype in tokens.Keyword]
        keyword_diversity = len(set(keywords)) / len(keywords) if keywords else 0

        # Count different types of operators
        operators = [token.value for token in tokens if token.ttype in tokens.Operator]
        comparison_ops = [op for op in operators if op in ['=', '!=', '<>', '<', '>', '<=', '>=']]
        logical_ops = [token.value.upper() for token in tokens
                      if token.ttype in tokens.Keyword and token.value.upper() in ['AND', 'OR', 'NOT']]

        return [
            float(avg_identifier_length),    # avg_identifier_length
            float(keyword_diversity),        # keyword_diversity
            float(len(operators)),           # operator_count
            float(len(comparison_ops)),      # comparison_operator_count
            float(len(logical_ops)),         # logical_operator_count
        ]

    def _extract_database_specific_features(self, sql_text: str, database_type: str) -> List[float]:
        """Extract database-specific syntax features."""
        sql_upper = sql_text.upper()

        # MySQL-specific features
        mysql_features = any(keyword in sql_upper for keyword in [
            'LIMIT', 'AUTO_INCREMENT', 'ENGINE=', 'CHARSET=', 'COLLATE=',
            'ON DUPLICATE KEY', 'REPLACE INTO', 'INSERT IGNORE'
        ])

        # PostgreSQL-specific features
        postgresql_features = any(keyword in sql_upper for keyword in [
            'SERIAL', 'BIGSERIAL', 'RETURNING', 'ILIKE', 'OFFSET',
            'WINDOW', 'OVER(', 'ARRAY[', 'JSONB', '::'
        ])

        # SQLite-specific features
        sqlite_features = any(keyword in sql_upper for keyword in [
            'AUTOINCREMENT', 'WITHOUT ROWID', 'PRAGMA', 'ATTACH', 'DETACH'
        ])

        # Oracle-specific features
        oracle_features = any(keyword in sql_upper for keyword in [
            'ROWNUM', 'DUAL', 'SYSDATE', 'CONNECT BY', 'START WITH',
            'DECODE', 'NVL', 'SEQUENCE'
        ])

        # SQL Server-specific features
        sqlserver_features = any(keyword in sql_upper for keyword in [
            'IDENTITY', 'GETDATE()', 'DATEPART', 'ISNULL', 'CHARINDEX',
            'LEN(', 'SCOPE_IDENTITY', 'TOP '
        ])

        return [
            float(mysql_features),        # mysql_specific_syntax
            float(postgresql_features),   # postgresql_specific_syntax
            float(sqlite_features),       # sqlite_specific_syntax
            float(oracle_features),       # oracle_specific_syntax
            float(sqlserver_features),    # sqlserver_specific_syntax
        ]

    def _extract_pattern_features(self, sql_text: str, parsed: Statement) -> List[float]:
        """Extract SQL pattern-based features."""
        sql_upper = sql_text.upper()

        # Check for specific SQL patterns
        correlated_subquery = self._has_correlated_subquery(sql_text)
        has_exists = 'EXISTS' in sql_upper
        has_in = ' IN (' in sql_upper
        has_like = 'LIKE' in sql_upper
        has_null_check = any(pattern in sql_upper for pattern in [
            'IS NULL', 'IS NOT NULL', 'ISNULL', 'COALESCE', 'NVL'
        ])

        return [
            float(correlated_subquery),   # has_correlated_subquery
            float(has_exists),            # has_exists_clause
            float(has_in),                # has_in_clause
            float(has_like),              # has_like_pattern
            float(has_null_check),        # has_null_check
        ]

    def _has_correlated_subquery(self, sql_text: str) -> bool:
        """Check if query contains correlated subqueries."""
        # This is a simplified check - could be more sophisticated
        sql_upper = sql_text.upper()

        # Look for subqueries that reference outer query tables
        # This is a basic heuristic - could be improved with better parsing
        if 'SELECT' in sql_upper and '(' in sql_text:
            # Count nested SELECT statements
            select_count = sql_upper.count('SELECT')
            if select_count > 1:
                # Basic check for correlation (table aliases referenced in subqueries)
                # This is simplified - real implementation would need more sophisticated parsing
                return True

        return False

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()

    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.feature_names)

    def validate_features(self, features: List[float]) -> bool:
        """Validate that feature vector has correct length and valid values."""
        if len(features) != len(self.feature_names):
            return False

        # Check for invalid values (NaN, infinite)
        for feature in features:
            if not isinstance(feature, (int, float)) or feature != feature:  # NaN check
                return False

        return True

    def get_feature_description(self, feature_index: int) -> str:
        """Get description of a specific feature."""
        if 0 <= feature_index < len(self.feature_names):
            feature_name = self.feature_names[feature_index]

            descriptions = {
                'query_length': 'Total character length of the SQL query',
                'token_count': 'Number of SQL tokens in the query',
                'keyword_count': 'Number of SQL keywords',
                'identifier_count': 'Number of identifiers (table/column names)',
                'table_count': 'Number of tables referenced in the query',
                'join_count': 'Number of JOIN operations',
                'where_conditions': 'Number of conditions in WHERE clause',
                'subquery_count': 'Number of subqueries',
                'aggregate_function_count': 'Number of aggregate functions (COUNT, SUM, etc.)',
                'window_function_count': 'Number of window functions',
                'is_select': 'Whether this is a SELECT statement',
                'is_insert': 'Whether this is an INSERT statement',
                'is_update': 'Whether this is an UPDATE statement',
                'is_delete': 'Whether this is a DELETE statement',
                'is_create': 'Whether this is a CREATE statement',
                'is_alter': 'Whether this is an ALTER statement',
                'nesting_level': 'Maximum nesting level of parentheses',
                'function_call_count': 'Number of function calls',
                'case_statement_count': 'Number of CASE statements',
                'union_count': 'Number of UNION operations',
                'having_clause_present': 'Whether query has HAVING clause',
                'order_by_present': 'Whether query has ORDER BY clause',
                'group_by_present': 'Whether query has GROUP BY clause',
                'distinct_present': 'Whether query uses DISTINCT',
                'limit_present': 'Whether query has LIMIT/TOP clause',
                'select_star_present': 'Whether query uses SELECT *',
                'cross_join_present': 'Whether query has CROSS JOIN',
                'cartesian_product_risk': 'Risk of cartesian product',
                'missing_where_clause': 'Whether UPDATE/DELETE lacks WHERE',
                'potential_index_usage': 'Potential for index utilization',
                'avg_identifier_length': 'Average length of identifiers',
                'keyword_diversity': 'Diversity of keywords used',
                'operator_count': 'Total number of operators',
                'comparison_operator_count': 'Number of comparison operators',
                'logical_operator_count': 'Number of logical operators',
                'mysql_specific_syntax': 'Uses MySQL-specific syntax',
                'postgresql_specific_syntax': 'Uses PostgreSQL-specific syntax',
                'sqlite_specific_syntax': 'Uses SQLite-specific syntax',
                'oracle_specific_syntax': 'Uses Oracle-specific syntax',
                'sqlserver_specific_syntax': 'Uses SQL Server-specific syntax',
                'has_correlated_subquery': 'Contains correlated subqueries',
                'has_exists_clause': 'Uses EXISTS clause',
                'has_in_clause': 'Uses IN clause',
                'has_like_pattern': 'Uses LIKE pattern matching',
                'has_null_check': 'Contains NULL checks',
            }

            return descriptions.get(feature_name, f'Feature: {feature_name}')

        return 'Invalid feature index'