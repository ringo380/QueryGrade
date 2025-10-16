"""
Advanced Semantic Feature Extraction for SQL Queries

This module implements sophisticated semantic analysis of SQL queries, extracting
features that capture the deeper meaning and structure beyond basic syntax.
"""

import re
import logging
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import sqlparse
from sqlparse.sql import Statement, Token, TokenList
from sqlparse.tokens import Keyword, Name, Punctuation, Number, String
import numpy as np
from collections import defaultdict, Counter

# Import semantic analyzers
from .nested_subquery_analyzer import NestedSubqueryAnalyzer
from .join_semantic_analyzer import JoinSemanticAnalyzer


class QueryIntent(Enum):
    """Classification of query intent/purpose"""
    ANALYTICAL = "analytical"  # Complex aggregations, reporting
    TRANSACTIONAL = "transactional"  # Simple CRUD operations
    EXPLORATORY = "exploratory"  # Ad-hoc data exploration
    MAINTENANCE = "maintenance"  # Schema changes, admin tasks
    HYBRID = "hybrid"  # Mixed purposes


class AccessPattern(Enum):
    """Common data access patterns"""
    SEQUENTIAL_SCAN = "sequential_scan"
    INDEX_LOOKUP = "index_lookup"
    RANGE_SCAN = "range_scan"
    JOIN_HEAVY = "join_heavy"
    AGGREGATION_HEAVY = "aggregation_heavy"
    SUBQUERY_HEAVY = "subquery_heavy"


@dataclass
class SemanticMetrics:
    """Comprehensive semantic metrics for a query"""
    # Query Intent Analysis
    primary_intent: QueryIntent = QueryIntent.TRANSACTIONAL
    intent_confidence: float = 0.0
    secondary_intents: List[QueryIntent] = field(default_factory=list)

    # Relational Algebra Operations
    projection_complexity: float = 0.0  # SELECT clause complexity
    selection_complexity: float = 0.0   # WHERE clause complexity
    join_complexity: float = 0.0        # JOIN operations complexity
    aggregation_complexity: float = 0.0 # GROUP BY/aggregate functions
    sorting_complexity: float = 0.0     # ORDER BY complexity

    # Data Flow Analysis
    data_sources: Set[str] = field(default_factory=set)
    data_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    output_cardinality_estimate: str = "unknown"  # low/medium/high/unknown

    # Access Pattern Analysis
    primary_access_pattern: AccessPattern = AccessPattern.SEQUENTIAL_SCAN
    access_pattern_confidence: float = 0.0
    secondary_patterns: List[AccessPattern] = field(default_factory=list)

    # Semantic Complexity Indicators
    conceptual_complexity: float = 0.0    # How complex is the query conceptually
    cognitive_load: float = 0.0          # Mental effort required to understand
    maintenance_difficulty: float = 0.0   # How hard to modify/debug

    # Advanced Features
    temporal_complexity: float = 0.0     # Time-based operations complexity
    hierarchical_complexity: float = 0.0 # Recursive/tree operations
    set_operation_complexity: float = 0.0 # UNION, INTERSECT, EXCEPT
    window_function_complexity: float = 0.0 # Analytical window functions

    # Performance Predictors
    estimated_selectivity: float = 0.0   # How much data will be filtered
    join_selectivity: float = 0.0       # JOIN filtering effectiveness
    index_usage_probability: float = 0.0 # Likelihood of efficient index usage
    parallel_execution_potential: float = 0.0 # Parallelization opportunities

    # Nested Subquery Analysis (Phase 1)
    nesting_depth: int = 0  # Maximum nesting level
    subquery_count: int = 0  # Total number of subqueries
    correlated_subquery_count: int = 0  # Correlated subqueries (performance risk)
    derived_table_count: int = 0  # Derived tables in FROM clause
    nesting_complexity_score: float = 0.0  # Complexity from nesting (0-1)
    subquery_types_distribution: Dict[str, int] = field(default_factory=dict)  # Type distribution
    subquery_performance_risk: str = "low"  # low/medium/high/critical

    # JOIN Semantic Analysis (Phase 2)
    join_count: int = 0  # Total number of JOINs
    inner_join_count: int = 0  # INNER JOIN count
    outer_join_count: int = 0  # LEFT/RIGHT/FULL JOIN count
    cross_join_count: int = 0  # CROSS JOIN count
    implicit_join_count: int = 0  # Joins in WHERE clause
    join_complexity_score: float = 0.0  # Overall JOIN complexity (0-1)
    join_types_distribution: Dict[str, int] = field(default_factory=dict)  # Type breakdown
    join_impacts_distribution: Dict[str, int] = field(default_factory=dict)  # Impact breakdown
    result_cardinality_impact: str = "unknown"  # reducing/preserving/expanding/unknown
    avg_join_condition_complexity: float = 0.0  # Average condition complexity (0-1)
    has_implicit_joins: bool = False  # Whether query has implicit joins
    redundant_join_count: int = 0  # Redundant JOINs

    @property
    def overall_score(self) -> float:
        """Calculate overall semantic score (0-100)"""
        # Weight different aspects of semantic complexity
        complexity_score = (
            self.conceptual_complexity * 0.3 +
            self.cognitive_load * 0.2 +
            (1.0 - self.maintenance_difficulty) * 0.2 +  # Lower difficulty = higher score
            self.index_usage_probability * 0.15 +
            self.parallel_execution_potential * 0.15
        )
        return min(100.0, max(0.0, complexity_score * 100))

    @property
    def complexity_level(self) -> str:
        """Get human-readable complexity level"""
        avg_complexity = (
            self.conceptual_complexity +
            self.cognitive_load +
            self.maintenance_difficulty
        ) / 3.0

        if avg_complexity < 0.3:
            return "Simple"
        elif avg_complexity < 0.6:
            return "Moderate"
        elif avg_complexity < 0.8:
            return "Complex"
        else:
            return "Very Complex"

    @property
    def query_intent(self) -> str:
        """Get primary intent as string"""
        return self.primary_intent.value if self.primary_intent else "unknown"


class SemanticFeatureExtractor:
    """Advanced semantic feature extraction for SQL queries"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()
        # Initialize semantic analyzers
        self.subquery_analyzer = NestedSubqueryAnalyzer()
        self.join_analyzer = JoinSemanticAnalyzer()

    def _compile_patterns(self):
        """Compile regex patterns for semantic analysis"""
        self.patterns = {
            # Temporal patterns
            'temporal_functions': re.compile(
                r'\b(NOW|CURRENT_TIMESTAMP|DATE|DATEADD|DATEDIFF|EXTRACT|YEAR|MONTH|DAY)\b',
                re.IGNORECASE
            ),
            'temporal_comparisons': re.compile(
                r'\b(BETWEEN.*AND|>=|<=|>|<)\s+[\'"]*\d{4}-\d{2}-\d{2}',
                re.IGNORECASE
            ),

            # Analytical patterns
            'window_functions': re.compile(
                r'\b(ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|FIRST_VALUE|LAST_VALUE|NTILE)\s*\(',
                re.IGNORECASE
            ),
            'advanced_aggregates': re.compile(
                r'\b(STDDEV|VARIANCE|PERCENTILE|MEDIAN|MODE)\s*\(',
                re.IGNORECASE
            ),

            # Set operations
            'set_operations': re.compile(
                r'\b(UNION|INTERSECT|EXCEPT|MINUS)\b',
                re.IGNORECASE
            ),

            # Hierarchical patterns
            'recursive_cte': re.compile(
                r'\bWITH\s+RECURSIVE\b',
                re.IGNORECASE
            ),
            'hierarchical_functions': re.compile(
                r'\b(CONNECT\s+BY|START\s+WITH|LEVEL)\b',
                re.IGNORECASE
            ),

            # Performance indicators
            'force_index': re.compile(
                r'\b(FORCE\s+INDEX|USE\s+INDEX|IGNORE\s+INDEX)\b',
                re.IGNORECASE
            ),
            'hints': re.compile(
                r'/\*\+.*?\*/',
                re.DOTALL
            )
        }

    def extract_semantic_features(self, query: str, database_type: str = 'generic') -> SemanticMetrics:
        """Extract comprehensive semantic features from a SQL query"""
        try:
            # Parse the query
            parsed = sqlparse.parse(query)[0]

            # Initialize metrics
            metrics = SemanticMetrics(
                primary_intent=QueryIntent.TRANSACTIONAL,
                intent_confidence=0.5,
                primary_access_pattern=AccessPattern.SEQUENTIAL_SCAN,
                access_pattern_confidence=0.5,
                projection_complexity=0.0,
                selection_complexity=0.0,
                join_complexity=0.0,
                aggregation_complexity=0.0,
                sorting_complexity=0.0,
                conceptual_complexity=0.0,
                cognitive_load=0.0,
                maintenance_difficulty=0.0,
                temporal_complexity=0.0,
                hierarchical_complexity=0.0,
                set_operation_complexity=0.0,
                window_function_complexity=0.0,
                estimated_selectivity=0.5,
                join_selectivity=0.5,
                index_usage_probability=0.5,
                parallel_execution_potential=0.5
            )

            # Analyze different aspects
            self._analyze_query_intent(query, parsed, metrics)
            self._analyze_relational_operations(query, parsed, metrics)
            self._analyze_data_flow(query, parsed, metrics)
            self._analyze_access_patterns(query, parsed, metrics)
            self._analyze_complexity_indicators(query, parsed, metrics)
            self._analyze_advanced_features(query, parsed, metrics)
            self._analyze_nested_subqueries(query, parsed, metrics)  # Phase 1
            self._analyze_joins(query, parsed, metrics)  # Phase 2
            self._predict_performance_characteristics(query, parsed, metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error extracting semantic features: {e}")
            # Return default metrics on error
            return SemanticMetrics(
                primary_intent=QueryIntent.TRANSACTIONAL,
                intent_confidence=0.0,
                primary_access_pattern=AccessPattern.SEQUENTIAL_SCAN,
                access_pattern_confidence=0.0,
                projection_complexity=0.0,
                selection_complexity=0.0,
                join_complexity=0.0,
                aggregation_complexity=0.0,
                sorting_complexity=0.0,
                conceptual_complexity=0.0,
                cognitive_load=0.0,
                maintenance_difficulty=0.0,
                temporal_complexity=0.0,
                hierarchical_complexity=0.0,
                set_operation_complexity=0.0,
                window_function_complexity=0.0,
                estimated_selectivity=0.0,
                join_selectivity=0.0,
                index_usage_probability=0.0,
                parallel_execution_potential=0.0
            )

    def _analyze_nested_subqueries(self, query: str, parsed: Statement, metrics: SemanticMetrics):
        """Analyze nested subqueries (Phase 1 enhancement)"""
        try:
            analysis = self.subquery_analyzer.analyze_nested_subqueries(query)

            # Update metrics with nested subquery analysis
            metrics.nesting_depth = analysis.max_nesting_depth
            metrics.subquery_count = analysis.total_subquery_count
            metrics.correlated_subquery_count = analysis.correlated_count
            metrics.derived_table_count = analysis.derived_table_count
            metrics.nesting_complexity_score = analysis.complexity_score
            metrics.subquery_types_distribution = analysis.subquery_types
            metrics.subquery_performance_risk = analysis.performance_risk_level

            # Adjust conceptual complexity based on nesting
            if analysis.max_nesting_depth >= 3:
                metrics.conceptual_complexity = min(1.0, metrics.conceptual_complexity + 0.3)
            elif analysis.max_nesting_depth >= 2:
                metrics.conceptual_complexity = min(1.0, metrics.conceptual_complexity + 0.15)

            # Adjust maintenance difficulty based on subquery complexity
            if analysis.correlated_count > 0:
                metrics.maintenance_difficulty = min(1.0, metrics.maintenance_difficulty + 0.25)

            if analysis.total_subquery_count > 5:
                metrics.maintenance_difficulty = min(1.0, metrics.maintenance_difficulty + 0.15)

        except Exception as e:
            self.logger.warning(f"Error analyzing nested subqueries: {e}")

    def _analyze_joins(self, query: str, parsed: Statement, metrics: SemanticMetrics):
        """Analyze JOIN semantics (Phase 2 enhancement)"""
        try:
            analysis = self.join_analyzer.analyze_joins(query)

            # Update metrics with JOIN analysis
            metrics.join_count = analysis.total_join_count
            metrics.inner_join_count = analysis.inner_join_count
            metrics.outer_join_count = analysis.outer_join_count
            metrics.cross_join_count = analysis.cross_join_count
            metrics.implicit_join_count = analysis.implicit_join_count
            metrics.join_complexity_score = analysis.overall_complexity_score
            metrics.join_types_distribution = analysis.join_types
            metrics.join_impacts_distribution = analysis.join_impacts
            metrics.result_cardinality_impact = analysis.result_cardinality_impact
            metrics.avg_join_condition_complexity = analysis.avg_condition_complexity
            metrics.has_implicit_joins = analysis.has_implicit_joins
            metrics.redundant_join_count = analysis.redundant_join_count

            # Adjust complexity scores based on JOINs
            if analysis.total_join_count > 0:
                metrics.join_complexity_score = analysis.overall_complexity_score
                metrics.conceptual_complexity = min(1.0, metrics.conceptual_complexity + analysis.overall_complexity_score * 0.2)

            # Implicit JOINs are harder to understand
            if analysis.has_implicit_joins:
                metrics.cognitive_load = min(1.0, metrics.cognitive_load + 0.15)
                metrics.maintenance_difficulty = min(1.0, metrics.maintenance_difficulty + 0.2)

            # CROSS JOINs indicate high risk
            if analysis.cross_join_count > 0:
                metrics.conceptual_complexity = min(1.0, metrics.conceptual_complexity + 0.25)
                metrics.maintenance_difficulty = min(1.0, metrics.maintenance_difficulty + 0.3)

        except Exception as e:
            self.logger.warning(f"Error analyzing JOINs: {e}")

    def _analyze_query_intent(self, query: str, parsed: Statement, metrics: SemanticMetrics):
        """Analyze the primary intent/purpose of the query"""
        query_upper = query.upper()

        # Check for analytical indicators
        analytical_score = 0
        if any(func in query_upper for func in ['GROUP BY', 'HAVING', 'PARTITION BY']):
            analytical_score += 2
        if any(func in query_upper for func in ['SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN(']):
            analytical_score += 1
        if len(re.findall(r'\bJOIN\b', query_upper)) >= 2:
            analytical_score += 1
        if self.patterns['window_functions'].search(query):
            analytical_score += 3
        if self.patterns['advanced_aggregates'].search(query):
            analytical_score += 2

        # Check for transactional indicators
        transactional_score = 0
        if any(op in query_upper for op in ['INSERT', 'UPDATE', 'DELETE']):
            transactional_score += 3
        if 'WHERE' in query_upper and 'GROUP BY' not in query_upper:
            transactional_score += 1
        if 'LIMIT' in query_upper or 'TOP' in query_upper:
            transactional_score += 1

        # Check for exploratory indicators
        exploratory_score = 0
        if 'SELECT *' in query_upper:
            exploratory_score += 2
        if 'DESCRIBE' in query_upper or 'SHOW' in query_upper:
            exploratory_score += 3
        if 'ORDER BY' in query_upper and 'GROUP BY' not in query_upper:
            exploratory_score += 1

        # Check for maintenance indicators
        maintenance_score = 0
        if any(op in query_upper for op in ['CREATE', 'DROP', 'ALTER', 'TRUNCATE']):
            maintenance_score += 3
        if any(op in query_upper for op in ['GRANT', 'REVOKE', 'VACUUM', 'ANALYZE']):
            maintenance_score += 2

        # Determine primary intent
        scores = {
            QueryIntent.ANALYTICAL: analytical_score,
            QueryIntent.TRANSACTIONAL: transactional_score,
            QueryIntent.EXPLORATORY: exploratory_score,
            QueryIntent.MAINTENANCE: maintenance_score
        }

        if max(scores.values()) == 0:
            metrics.primary_intent = QueryIntent.TRANSACTIONAL
            metrics.intent_confidence = 0.3
        else:
            metrics.primary_intent = max(scores, key=scores.get)
            total_score = sum(scores.values())
            metrics.intent_confidence = scores[metrics.primary_intent] / total_score

            # Add secondary intents
            sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for intent, score in sorted_intents[1:]:
                if score > 0 and score >= total_score * 0.2:  # At least 20% of total score
                    metrics.secondary_intents.append(intent)

        # Check for hybrid queries
        if len(metrics.secondary_intents) >= 2:
            metrics.primary_intent = QueryIntent.HYBRID

    def _analyze_relational_operations(self, query: str, parsed: Statement, metrics: SemanticMetrics):
        """Analyze relational algebra operations complexity"""
        query_upper = query.upper()

        # Projection complexity (SELECT clause)
        select_items = len(re.findall(r',', query.split('FROM')[0])) + 1
        distinct_used = 'DISTINCT' in query_upper
        case_statements = len(re.findall(r'\bCASE\b', query_upper))
        subqueries_in_select = len(re.findall(r'SELECT.*?\(.*?SELECT', query_upper))

        metrics.projection_complexity = min(1.0, (
            select_items * 0.1 +
            (0.2 if distinct_used else 0) +
            case_statements * 0.15 +
            subqueries_in_select * 0.3
        ))

        # Selection complexity (WHERE clause)
        if 'WHERE' in query_upper:
            where_clause = query.split('WHERE', 1)[1].split('GROUP BY')[0] if 'GROUP BY' in query else query.split('WHERE', 1)[1]
            logical_operators = len(re.findall(r'\b(AND|OR)\b', where_clause, re.IGNORECASE))
            comparison_operators = len(re.findall(r'[<>=!]', where_clause))
            in_operators = len(re.findall(r'\bIN\s*\(', where_clause, re.IGNORECASE))
            like_operators = len(re.findall(r'\bLIKE\b', where_clause, re.IGNORECASE))
            exists_operators = len(re.findall(r'\bEXISTS\b', where_clause, re.IGNORECASE))

            metrics.selection_complexity = min(1.0, (
                logical_operators * 0.1 +
                comparison_operators * 0.05 +
                in_operators * 0.15 +
                like_operators * 0.1 +
                exists_operators * 0.25
            ))

        # Join complexity
        join_count = len(re.findall(r'\bJOIN\b', query_upper))
        left_joins = len(re.findall(r'\bLEFT\s+JOIN\b', query_upper))
        right_joins = len(re.findall(r'\bRIGHT\s+JOIN\b', query_upper))
        full_joins = len(re.findall(r'\bFULL\s+JOIN\b', query_upper))
        cross_joins = len(re.findall(r'\bCROSS\s+JOIN\b', query_upper))

        metrics.join_complexity = min(1.0, (
            join_count * 0.2 +
            left_joins * 0.05 +
            right_joins * 0.1 +
            full_joins * 0.15 +
            cross_joins * 0.3
        ))

        # Aggregation complexity
        group_by_used = 'GROUP BY' in query_upper
        having_used = 'HAVING' in query_upper
        aggregate_functions = len(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN|STDDEV|VARIANCE)\s*\(', query_upper))

        metrics.aggregation_complexity = min(1.0, (
            (0.3 if group_by_used else 0) +
            (0.2 if having_used else 0) +
            aggregate_functions * 0.1
        ))

        # Sorting complexity
        order_by_used = 'ORDER BY' in query_upper
        if order_by_used:
            order_by_columns = len(re.findall(r',', query.split('ORDER BY')[1])) + 1
            desc_used = 'DESC' in query_upper
            metrics.sorting_complexity = min(1.0, 0.2 + order_by_columns * 0.1 + (0.1 if desc_used else 0))

    def _analyze_data_flow(self, query: str, parsed: Statement, metrics: SemanticMetrics):
        """Analyze data sources and dependencies"""
        query_upper = query.upper()

        # Extract table names (simplified approach)
        from_match = re.search(r'\bFROM\s+([^WHERE^GROUP^ORDER^LIMIT^UNION^INTERSECT^EXCEPT]+)', query_upper)
        if from_match:
            from_clause = from_match.group(1)
            # Extract table names (basic pattern)
            tables = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', from_clause)
            metrics.data_sources = set(tables)

        # Estimate output cardinality
        if 'GROUP BY' in query_upper:
            metrics.output_cardinality_estimate = "medium"
        elif 'DISTINCT' in query_upper:
            metrics.output_cardinality_estimate = "medium"
        elif any(agg in query_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
            metrics.output_cardinality_estimate = "low"
        elif 'LIMIT' in query_upper or 'TOP' in query_upper:
            metrics.output_cardinality_estimate = "low"
        else:
            metrics.output_cardinality_estimate = "high"

    def _analyze_access_patterns(self, query: str, parsed: Statement, metrics: SemanticMetrics):
        """Analyze expected data access patterns"""
        query_upper = query.upper()

        pattern_scores = {
            AccessPattern.SEQUENTIAL_SCAN: 0,
            AccessPattern.INDEX_LOOKUP: 0,
            AccessPattern.RANGE_SCAN: 0,
            AccessPattern.JOIN_HEAVY: 0,
            AccessPattern.AGGREGATION_HEAVY: 0,
            AccessPattern.SUBQUERY_HEAVY: 0
        }

        # Sequential scan indicators
        if 'SELECT *' in query_upper:
            pattern_scores[AccessPattern.SEQUENTIAL_SCAN] += 2
        if 'WHERE' not in query_upper:
            pattern_scores[AccessPattern.SEQUENTIAL_SCAN] += 3

        # Index lookup indicators
        if 'WHERE' in query_upper and '=' in query:
            pattern_scores[AccessPattern.INDEX_LOOKUP] += 2
        if 'PRIMARY KEY' in query_upper or 'UNIQUE' in query_upper:
            pattern_scores[AccessPattern.INDEX_LOOKUP] += 1

        # Range scan indicators
        if any(op in query_upper for op in ['BETWEEN', '>', '<', '>=']):
            pattern_scores[AccessPattern.RANGE_SCAN] += 2
        if 'LIKE' in query_upper:
            pattern_scores[AccessPattern.RANGE_SCAN] += 1

        # Join heavy indicators
        join_count = len(re.findall(r'\bJOIN\b', query_upper))
        if join_count >= 2:
            pattern_scores[AccessPattern.JOIN_HEAVY] += join_count

        # Aggregation heavy indicators
        if 'GROUP BY' in query_upper:
            pattern_scores[AccessPattern.AGGREGATION_HEAVY] += 2
        agg_count = len(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', query_upper))
        pattern_scores[AccessPattern.AGGREGATION_HEAVY] += agg_count

        # Subquery heavy indicators
        subquery_count = query_upper.count('SELECT') - 1  # Subtract main query
        if subquery_count > 0:
            pattern_scores[AccessPattern.SUBQUERY_HEAVY] += subquery_count * 2

        # Determine primary access pattern
        if max(pattern_scores.values()) == 0:
            metrics.primary_access_pattern = AccessPattern.SEQUENTIAL_SCAN
            metrics.access_pattern_confidence = 0.3
        else:
            metrics.primary_access_pattern = max(pattern_scores, key=pattern_scores.get)
            total_score = sum(pattern_scores.values())
            metrics.access_pattern_confidence = pattern_scores[metrics.primary_access_pattern] / total_score

    def _analyze_complexity_indicators(self, query: str, parsed: Statement, metrics: SemanticMetrics):
        """Analyze various complexity indicators"""
        query_upper = query.upper()

        # Conceptual complexity - how hard is it to understand the query's purpose
        conceptual_factors = [
            len(re.findall(r'\bJOIN\b', query_upper)) * 0.1,  # Joins add conceptual complexity
            (1 if 'GROUP BY' in query_upper else 0) * 0.2,    # Aggregation adds complexity
            len(re.findall(r'\bSELECT\b', query_upper)) * 0.15,  # Subqueries add complexity
            len(re.findall(r'\bCASE\b', query_upper)) * 0.1,  # Conditional logic
            (1 if any(op in query_upper for op in ['UNION', 'INTERSECT', 'EXCEPT']) else 0) * 0.3
        ]
        metrics.conceptual_complexity = min(1.0, sum(conceptual_factors))

        # Cognitive load - mental effort required
        cognitive_factors = [
            len(query) / 1000,  # Longer queries are harder to process
            len(re.findall(r'\(', query)) * 0.05,  # Nested structures
            len(re.findall(r',', query)) * 0.02,   # Number of elements to track
            (1 if self.patterns['window_functions'].search(query) else 0) * 0.3
        ]
        metrics.cognitive_load = min(1.0, sum(cognitive_factors))

        # Maintenance difficulty
        maintenance_factors = [
            len(re.findall(r'\bSELECT\b', query_upper)) * 0.1,  # Subqueries make maintenance harder
            (1 if 'HAVING' in query_upper else 0) * 0.2,         # Complex filtering
            len(re.findall(r'\bCASE\b', query_upper)) * 0.15,    # Conditional logic
            (1 if self.patterns['hints'].search(query) else 0) * 0.3  # Database hints
        ]
        metrics.maintenance_difficulty = min(1.0, sum(maintenance_factors))

    def _analyze_advanced_features(self, query: str, parsed: Statement, metrics: SemanticMetrics):
        """Analyze advanced SQL features"""
        # Temporal complexity
        temporal_functions = len(self.patterns['temporal_functions'].findall(query))
        temporal_comparisons = len(self.patterns['temporal_comparisons'].findall(query))
        metrics.temporal_complexity = min(1.0, (temporal_functions + temporal_comparisons) * 0.2)

        # Hierarchical complexity
        recursive_cte = len(self.patterns['recursive_cte'].findall(query))
        hierarchical_functions = len(self.patterns['hierarchical_functions'].findall(query))
        metrics.hierarchical_complexity = min(1.0, (recursive_cte + hierarchical_functions) * 0.5)

        # Set operation complexity
        set_operations = len(self.patterns['set_operations'].findall(query))
        metrics.set_operation_complexity = min(1.0, set_operations * 0.3)

        # Window function complexity
        window_functions = len(self.patterns['window_functions'].findall(query))
        metrics.window_function_complexity = min(1.0, window_functions * 0.25)

    def _predict_performance_characteristics(self, query: str, parsed: Statement, metrics: SemanticMetrics):
        """Predict performance-related characteristics"""
        query_upper = query.upper()

        # Estimated selectivity (how much data will be filtered)
        if 'WHERE' in query_upper:
            where_clause = query.split('WHERE', 1)[1]
            equality_conditions = len(re.findall(r'=', where_clause))
            range_conditions = len(re.findall(r'[<>]', where_clause))

            # More specific conditions = higher selectivity (less data returned)
            selectivity_score = equality_conditions * 0.3 + range_conditions * 0.2
            metrics.estimated_selectivity = min(0.9, max(0.1, 1.0 - selectivity_score))
        else:
            metrics.estimated_selectivity = 0.1  # Low selectivity if no WHERE clause

        # Join selectivity
        join_count = len(re.findall(r'\bJOIN\b', query_upper))
        if join_count > 0:
            # More joins typically mean more filtering
            metrics.join_selectivity = min(0.9, max(0.1, 0.8 - join_count * 0.1))
        else:
            metrics.join_selectivity = 1.0  # No joins = no join filtering

        # Index usage probability
        index_indicators = [
            'WHERE' in query_upper and '=' in query,  # Equality conditions
            'ORDER BY' in query_upper,                # Sorting can use indexes
            'GROUP BY' in query_upper,                # Grouping can use indexes
            not ('SELECT *' in query_upper),          # Specific columns more likely to use indexes
        ]
        metrics.index_usage_probability = sum(index_indicators) / len(index_indicators)

        # Parallel execution potential
        parallel_indicators = [
            'GROUP BY' in query_upper,                # Aggregations can be parallelized
            len(re.findall(r'\bJOIN\b', query_upper)) >= 2,  # Complex joins benefit from parallelism
            not ('ORDER BY' in query_upper),          # Sorting reduces parallelism potential
            'DISTINCT' not in query_upper,            # DISTINCT reduces parallelism
        ]
        metrics.parallel_execution_potential = sum(parallel_indicators) / len(parallel_indicators)


def analyze_query_semantics(query: str, database_type: str = 'generic') -> Dict[str, Any]:
    """
    Convenience function to analyze query semantics and return results as dictionary
    """
    extractor = SemanticFeatureExtractor()
    metrics = extractor.extract_semantic_features(query, database_type)

    return {
        'query_intent': {
            'primary_intent': metrics.primary_intent.value,
            'confidence': metrics.intent_confidence,
            'secondary_intents': [intent.value for intent in metrics.secondary_intents]
        },
        'relational_operations': {
            'projection_complexity': metrics.projection_complexity,
            'selection_complexity': metrics.selection_complexity,
            'join_complexity': metrics.join_complexity,
            'aggregation_complexity': metrics.aggregation_complexity,
            'sorting_complexity': metrics.sorting_complexity
        },
        'data_flow': {
            'data_sources': list(metrics.data_sources),
            'output_cardinality_estimate': metrics.output_cardinality_estimate
        },
        'access_patterns': {
            'primary_pattern': metrics.primary_access_pattern.value,
            'confidence': metrics.access_pattern_confidence,
            'secondary_patterns': [pattern.value for pattern in metrics.secondary_patterns]
        },
        'complexity_indicators': {
            'conceptual_complexity': metrics.conceptual_complexity,
            'cognitive_load': metrics.cognitive_load,
            'maintenance_difficulty': metrics.maintenance_difficulty
        },
        'advanced_features': {
            'temporal_complexity': metrics.temporal_complexity,
            'hierarchical_complexity': metrics.hierarchical_complexity,
            'set_operation_complexity': metrics.set_operation_complexity,
            'window_function_complexity': metrics.window_function_complexity
        },
        'performance_predictors': {
            'estimated_selectivity': metrics.estimated_selectivity,
            'join_selectivity': metrics.join_selectivity,
            'index_usage_probability': metrics.index_usage_probability,
            'parallel_execution_potential': metrics.parallel_execution_potential
        },
        'nested_subqueries': {
            'nesting_depth': metrics.nesting_depth,
            'subquery_count': metrics.subquery_count,
            'correlated_count': metrics.correlated_subquery_count,
            'derived_table_count': metrics.derived_table_count,
            'complexity_score': metrics.nesting_complexity_score,
            'types_distribution': metrics.subquery_types_distribution,
            'performance_risk': metrics.subquery_performance_risk
        },
        'join_semantics': {
            'join_count': metrics.join_count,
            'inner_join_count': metrics.inner_join_count,
            'outer_join_count': metrics.outer_join_count,
            'cross_join_count': metrics.cross_join_count,
            'implicit_join_count': metrics.implicit_join_count,
            'complexity_score': metrics.join_complexity_score,
            'types_distribution': metrics.join_types_distribution,
            'impacts_distribution': metrics.join_impacts_distribution,
            'result_cardinality_impact': metrics.result_cardinality_impact,
            'avg_condition_complexity': metrics.avg_join_condition_complexity,
            'has_implicit_joins': metrics.has_implicit_joins,
            'redundant_join_count': metrics.redundant_join_count
        }
    }


if __name__ == "__main__":
    # Example usage
    test_query = """
    SELECT
        c.customer_name,
        SUM(o.total_amount) as total_spent,
        COUNT(o.order_id) as order_count,
        AVG(o.total_amount) as avg_order_value,
        RANK() OVER (ORDER BY SUM(o.total_amount) DESC) as spending_rank
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_date >= '2023-01-01'
        AND c.customer_type = 'premium'
    GROUP BY c.customer_id, c.customer_name
    HAVING COUNT(o.order_id) > 5
    ORDER BY total_spent DESC
    LIMIT 100
    """

    results = analyze_query_semantics(test_query)

    print("=== Semantic Analysis Results ===")
    for category, metrics in results.items():
        print(f"\n{category.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")