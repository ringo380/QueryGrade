"""
Complexity-Based Benchmark Categorization for QueryGrade ML System

This module provides sophisticated analysis and categorization of SQL query complexity
across multiple dimensions, enabling intelligent benchmark organization and grading.
"""

import logging
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlparse
from sqlparse import keywords, sql, tokens
from sqlparse.sql import (Function, Identifier, IdentifierList, Statement,
                          Token, TokenList, Where)

logger = logging.getLogger(__name__)


class ComplexityDimension(Enum):
    """Different dimensions of query complexity."""

    SYNTACTIC = "syntactic"  # Syntax complexity (keywords, nesting)
    SEMANTIC = "semantic"  # Logical complexity (joins, subqueries)
    COMPUTATIONAL = "computational"  # Execution complexity (performance impact)
    COGNITIVE = "cognitive"  # Human readability complexity
    ARCHITECTURAL = "architectural"  # Database design complexity


class ComplexityLevel(Enum):
    """Complexity levels for categorization."""

    TRIVIAL = 1  # Very simple queries
    SIMPLE = 2  # Basic queries
    MODERATE = 3  # Moderately complex queries
    COMPLEX = 4  # Complex queries
    ADVANCED = 5  # Very complex queries
    EXPERT = 6  # Expert-level queries


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for a query."""

    # Syntactic complexity
    token_count: int = 0
    keyword_count: int = 0
    nesting_depth: int = 0
    line_count: int = 0

    # Semantic complexity
    table_count: int = 0
    join_count: int = 0
    subquery_count: int = 0
    function_count: int = 0
    condition_count: int = 0

    # Computational complexity
    cartesian_product_risk: float = 0.0
    index_usage_score: float = 0.0
    scan_efficiency: float = 0.0

    # Cognitive complexity
    readability_score: float = 0.0
    alias_clarity: float = 0.0
    naming_consistency: float = 0.0

    # Architectural complexity
    schema_dependencies: int = 0
    data_type_complexity: float = 0.0
    constraint_complexity: float = 0.0

    # Overall scores
    overall_complexity: float = 0.0
    complexity_level: ComplexityLevel = ComplexityLevel.SIMPLE
    confidence: float = 0.0


@dataclass
class ComplexityCategory:
    """A category for organizing queries by complexity."""

    name: str
    description: str
    complexity_range: Tuple[float, float]  # Min, max complexity scores
    expected_grade_range: Tuple[str, str]  # Expected grade range
    characteristics: List[str]
    example_patterns: List[str]
    training_weight: float = 1.0  # Weight for ML training


class QueryComplexityAnalyzer:
    """Analyzes SQL query complexity across multiple dimensions."""

    def __init__(self):
        self.complexity_weights = {
            ComplexityDimension.SYNTACTIC: 0.20,
            ComplexityDimension.SEMANTIC: 0.30,
            ComplexityDimension.COMPUTATIONAL: 0.25,
            ComplexityDimension.COGNITIVE: 0.15,
            ComplexityDimension.ARCHITECTURAL: 0.10,
        }

        self.complexity_categories = self._initialize_complexity_categories()
        self.sql_keywords = self._get_sql_keywords()

    def _initialize_complexity_categories(self) -> List[ComplexityCategory]:
        """Initialize predefined complexity categories."""
        return [
            ComplexityCategory(
                name="trivial_lookups",
                description="Very simple single-table lookups with primary key conditions",
                complexity_range=(0.0, 15.0),
                expected_grade_range=("A", "A"),
                characteristics=[
                    "Single table access",
                    "Primary key or unique index lookup",
                    "No joins or subqueries",
                    "Simple WHERE conditions",
                ],
                example_patterns=[
                    "SELECT * FROM table WHERE id = ?",
                    "SELECT column FROM table WHERE unique_key = ?",
                ],
                training_weight=1.2,
            ),
            ComplexityCategory(
                name="simple_queries",
                description="Basic queries with simple conditions and sorting",
                complexity_range=(15.0, 35.0),
                expected_grade_range=("A", "B"),
                characteristics=[
                    "Single table or simple joins",
                    "Basic WHERE conditions",
                    "Simple ORDER BY or GROUP BY",
                    "Standard aggregate functions",
                ],
                example_patterns=[
                    "SELECT * FROM table WHERE condition ORDER BY column",
                    "SELECT COUNT(*) FROM table WHERE condition",
                    "SELECT * FROM table1 JOIN table2 ON simple_condition",
                ],
                training_weight=1.0,
            ),
            ComplexityCategory(
                name="moderate_queries",
                description="Moderately complex queries with multiple tables and conditions",
                complexity_range=(35.0, 55.0),
                expected_grade_range=("B", "C"),
                characteristics=[
                    "Multiple table joins",
                    "Complex WHERE conditions",
                    "Subqueries or CTEs",
                    "Window functions",
                    "Advanced grouping",
                ],
                example_patterns=[
                    "SELECT ... FROM table1 JOIN table2 JOIN table3 WHERE complex_conditions",
                    "SELECT ... FROM table WHERE col IN (SELECT ...)",
                    "SELECT ROW_NUMBER() OVER (...) FROM table",
                ],
                training_weight=1.1,
            ),
            ComplexityCategory(
                name="complex_queries",
                description="Complex analytical queries with advanced features",
                complexity_range=(55.0, 75.0),
                expected_grade_range=("C", "D"),
                characteristics=[
                    "Multiple nested subqueries",
                    "Complex joins (CROSS, FULL OUTER)",
                    "Recursive CTEs",
                    "Advanced window functions",
                    "Multiple UNION operations",
                ],
                example_patterns=[
                    "WITH RECURSIVE cte AS (...) SELECT ...",
                    "SELECT ... FROM (SELECT ... FROM (SELECT ...))",
                    "Complex analytical queries with multiple CTEs",
                ],
                training_weight=1.3,
            ),
            ComplexityCategory(
                name="advanced_queries",
                description="Very complex queries with performance implications",
                complexity_range=(75.0, 90.0),
                expected_grade_range=("D", "F"),
                characteristics=[
                    "Deep nesting (4+ levels)",
                    "Cartesian products",
                    "Complex correlated subqueries",
                    "Advanced analytical functions",
                    "Dynamic SQL patterns",
                ],
                example_patterns=[
                    "Queries with CROSS JOINs",
                    "Multiple correlated subqueries",
                    "Complex pivot operations",
                ],
                training_weight=1.5,
            ),
            ComplexityCategory(
                name="expert_queries",
                description="Expert-level queries requiring deep optimization knowledge",
                complexity_range=(90.0, 100.0),
                expected_grade_range=("F", "F"),
                characteristics=[
                    "Extreme nesting or complexity",
                    "Anti-patterns that need rewriting",
                    "Queries requiring deep database knowledge",
                    "Performance-critical optimizations",
                ],
                example_patterns=[
                    "Extremely complex analytical queries",
                    "Queries with multiple anti-patterns",
                    "Resource-intensive operations",
                ],
                training_weight=2.0,
            ),
        ]

    def _get_sql_keywords(self) -> Set[str]:
        """Get set of SQL keywords for analysis."""
        return {
            "SELECT",
            "FROM",
            "WHERE",
            "JOIN",
            "INNER",
            "LEFT",
            "RIGHT",
            "FULL",
            "OUTER",
            "ON",
            "GROUP",
            "BY",
            "HAVING",
            "ORDER",
            "LIMIT",
            "OFFSET",
            "UNION",
            "ALL",
            "DISTINCT",
            "AS",
            "AND",
            "OR",
            "NOT",
            "IN",
            "EXISTS",
            "LIKE",
            "BETWEEN",
            "IS",
            "NULL",
            "CASE",
            "WHEN",
            "THEN",
            "ELSE",
            "END",
            "WITH",
            "RECURSIVE",
            "OVER",
            "PARTITION",
            "WINDOW",
            "ROW_NUMBER",
            "RANK",
            "DENSE_RANK",
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
            "CROSS",
            "NATURAL",
        }

    def analyze_complexity(
        self, query: str, database_type: str = "generic"
    ) -> ComplexityMetrics:
        """
        Perform comprehensive complexity analysis of a SQL query.

        Args:
            query: SQL query string
            database_type: Target database type

        Returns:
            ComplexityMetrics object with detailed analysis
        """
        try:
            # Parse the query
            parsed = sqlparse.parse(query)[0]

            # Initialize metrics
            metrics = ComplexityMetrics()

            # Analyze different complexity dimensions
            self._analyze_syntactic_complexity(query, parsed, metrics)
            self._analyze_semantic_complexity(query, parsed, metrics)
            self._analyze_computational_complexity(query, parsed, metrics)
            self._analyze_cognitive_complexity(query, parsed, metrics)
            self._analyze_architectural_complexity(
                query, parsed, metrics, database_type
            )

            # Calculate overall complexity
            self._calculate_overall_complexity(metrics)

            # Determine complexity level and confidence
            metrics.complexity_level = self._determine_complexity_level(
                metrics.overall_complexity
            )
            metrics.confidence = self._calculate_confidence(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing query complexity: {e}")
            return ComplexityMetrics()

    def _analyze_syntactic_complexity(
        self, query: str, parsed: Statement, metrics: ComplexityMetrics
    ):
        """Analyze syntactic complexity of the query."""
        # Token and keyword counts
        tokens = list(parsed.flatten())
        metrics.token_count = len(tokens)

        keyword_count = 0
        for token in tokens:
            if (
                token.ttype in keywords.Keyword
                or token.value.upper() in self.sql_keywords
            ):
                keyword_count += 1
        metrics.keyword_count = keyword_count

        # Nesting depth (parentheses levels)
        nesting_depth = 0
        max_depth = 0
        for char in query:
            if char == "(":
                nesting_depth += 1
                max_depth = max(max_depth, nesting_depth)
            elif char == ")":
                nesting_depth -= 1
        metrics.nesting_depth = max_depth

        # Line count
        metrics.line_count = len(query.split("\n"))

    def _analyze_semantic_complexity(
        self, query: str, parsed: Statement, metrics: ComplexityMetrics
    ):
        """Analyze semantic complexity of the query."""
        query_upper = query.upper()

        # Table count (FROM and JOIN patterns)
        table_patterns = [
            r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        ]
        tables = set()
        for pattern in table_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            tables.update(matches)
        metrics.table_count = len(tables)

        # Join count
        join_types = [
            "JOIN",
            "INNER JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "FULL JOIN",
            "CROSS JOIN",
        ]
        metrics.join_count = sum(
            query_upper.count(join_type) for join_type in join_types
        )

        # Subquery count
        # Count SELECT statements (excluding the main one)
        select_count = query_upper.count("SELECT")
        metrics.subquery_count = max(0, select_count - 1)

        # Function count
        function_patterns = [
            r"COUNT\s*\(",
            r"SUM\s*\(",
            r"AVG\s*\(",
            r"MIN\s*\(",
            r"MAX\s*\(",
            r"ROW_NUMBER\s*\(",
            r"RANK\s*\(",
            r"DENSE_RANK\s*\(",
            r"LAG\s*\(",
            r"LEAD\s*\(",
        ]
        function_count = 0
        for pattern in function_patterns:
            function_count += len(re.findall(pattern, query, re.IGNORECASE))
        metrics.function_count = function_count

        # Condition count
        condition_keywords = ["WHERE", "HAVING", "ON"]
        condition_count = 0
        for keyword in condition_keywords:
            if keyword in query_upper:
                # Count AND/OR operators after the keyword
                keyword_pos = query_upper.find(keyword)
                remaining = query_upper[keyword_pos:]
                condition_count += remaining.count("AND") + remaining.count("OR") + 1
        metrics.condition_count = condition_count

    def _analyze_computational_complexity(
        self, query: str, parsed: Statement, metrics: ComplexityMetrics
    ):
        """Analyze computational complexity and performance implications."""
        query_upper = query.upper()

        # Cartesian product risk
        has_cross_join = "CROSS JOIN" in query_upper
        join_count = metrics.join_count if hasattr(metrics, "join_count") else 0
        on_count = query_upper.count(" ON ")

        if has_cross_join:
            metrics.cartesian_product_risk = 1.0
        elif join_count > on_count:
            metrics.cartesian_product_risk = 0.8
        else:
            metrics.cartesian_product_risk = 0.0

        # Index usage score
        index_indicators = [
            ("WHERE.*=", 0.3),  # Equality conditions
            (r"WHERE.*ID\s*=", 0.5),  # ID equality (likely indexed)
            ("ORDER BY.*ID", 0.2),  # Ordering by ID
            ("GROUP BY", 0.1),  # Grouping operations
        ]

        index_score = 0.0
        for pattern, score in index_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                index_score += score

        # Negative indicators
        if "SELECT *" in query_upper:
            index_score -= 0.2
        if re.search(r"WHERE.*LIKE.*%.*%", query, re.IGNORECASE):
            index_score -= 0.3  # LIKE with leading wildcard

        metrics.index_usage_score = max(0.0, min(1.0, index_score))

        # Scan efficiency
        scan_efficiency = 1.0
        if "SELECT *" in query_upper:
            scan_efficiency -= 0.3
        if metrics.cartesian_product_risk > 0.5:
            scan_efficiency -= 0.4
        if metrics.subquery_count > 2:
            scan_efficiency -= 0.2

        metrics.scan_efficiency = max(0.0, scan_efficiency)

    def _analyze_cognitive_complexity(
        self, query: str, parsed: Statement, metrics: ComplexityMetrics
    ):
        """Analyze cognitive complexity (readability and maintainability)."""
        # Readability score based on various factors
        readability = 1.0

        # Penalize very long queries
        if len(query) > 1000:
            readability -= 0.3
        elif len(query) > 500:
            readability -= 0.1

        # Penalize deep nesting
        if metrics.nesting_depth > 5:
            readability -= 0.4
        elif metrics.nesting_depth > 3:
            readability -= 0.2

        # Penalize many subqueries
        if metrics.subquery_count > 3:
            readability -= 0.3
        elif metrics.subquery_count > 1:
            readability -= 0.1

        # Bonus for good formatting (multiple lines, proper indentation)
        if metrics.line_count > 1 and "\n" in query:
            readability += 0.1

        metrics.readability_score = max(0.0, readability)

        # Alias clarity (simplified analysis)
        alias_patterns = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        aliases = re.findall(alias_patterns, query, re.IGNORECASE)

        clear_aliases = 0
        for original, alias in aliases:
            if len(alias) > 1 and alias.lower() != original.lower():
                clear_aliases += 1

        metrics.alias_clarity = clear_aliases / max(1, len(aliases)) if aliases else 1.0

        # Naming consistency (simplified)
        identifiers = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query)
        snake_case = sum(1 for id in identifiers if "_" in id)
        camel_case = sum(1 for id in identifiers if re.match(r"[a-z]+[A-Z]", id))

        if len(identifiers) > 0:
            consistency = max(snake_case, camel_case) / len(identifiers)
        else:
            consistency = 1.0

        metrics.naming_consistency = consistency

    def _analyze_architectural_complexity(
        self,
        query: str,
        parsed: Statement,
        metrics: ComplexityMetrics,
        database_type: str,
    ):
        """Analyze architectural complexity and database-specific features."""
        query_upper = query.upper()

        # Schema dependencies (rough estimate based on table count and relationships)
        metrics.schema_dependencies = metrics.table_count * (1 + metrics.join_count)

        # Data type complexity (simplified)
        complex_types = ["JSON", "XML", "GEOMETRY", "ARRAY", "JSONB"]
        type_complexity = 0.0
        for data_type in complex_types:
            if data_type in query_upper:
                type_complexity += 0.2
        metrics.data_type_complexity = min(1.0, type_complexity)

        # Constraint complexity (based on conditions and functions)
        constraint_complexity = 0.0

        # Advanced functions indicate complex constraints
        advanced_functions = ["REGEXP", "EXTRACT", "CAST", "CONVERT", "COALESCE"]
        for func in advanced_functions:
            if func in query_upper:
                constraint_complexity += 0.1

        # Complex conditions
        if re.search(r"CASE\s+WHEN", query, re.IGNORECASE):
            constraint_complexity += 0.2

        metrics.constraint_complexity = min(1.0, constraint_complexity)

    def _calculate_overall_complexity(self, metrics: ComplexityMetrics):
        """Calculate the overall complexity score."""
        # Normalize individual scores to 0-100 range
        syntactic_score = min(
            100,
            (
                (metrics.token_count / 10)
                + (metrics.keyword_count * 2)
                + (metrics.nesting_depth * 10)
                + (metrics.line_count * 2)
            ),
        )

        semantic_score = min(
            100,
            (
                (metrics.table_count * 10)
                + (metrics.join_count * 15)
                + (metrics.subquery_count * 20)
                + (metrics.function_count * 5)
                + (metrics.condition_count * 3)
            ),
        )

        computational_score = (
            (1 - metrics.index_usage_score) * 40
            + metrics.cartesian_product_risk * 40
            + (1 - metrics.scan_efficiency) * 20
        )

        cognitive_score = (
            (1 - metrics.readability_score) * 40
            + (1 - metrics.alias_clarity) * 30
            + (1 - metrics.naming_consistency) * 30
        )

        architectural_score = min(
            100,
            (
                (metrics.schema_dependencies * 5)
                + (metrics.data_type_complexity * 30)
                + (metrics.constraint_complexity * 30)
            ),
        )

        # Weighted combination
        overall = (
            syntactic_score * self.complexity_weights[ComplexityDimension.SYNTACTIC]
            + semantic_score * self.complexity_weights[ComplexityDimension.SEMANTIC]
            + computational_score
            * self.complexity_weights[ComplexityDimension.COMPUTATIONAL]
            + cognitive_score * self.complexity_weights[ComplexityDimension.COGNITIVE]
            + architectural_score
            * self.complexity_weights[ComplexityDimension.ARCHITECTURAL]
        )

        metrics.overall_complexity = min(100.0, overall)

    def _determine_complexity_level(self, complexity_score: float) -> ComplexityLevel:
        """Determine complexity level from score."""
        if complexity_score < 15:
            return ComplexityLevel.TRIVIAL
        elif complexity_score < 35:
            return ComplexityLevel.SIMPLE
        elif complexity_score < 55:
            return ComplexityLevel.MODERATE
        elif complexity_score < 75:
            return ComplexityLevel.COMPLEX
        elif complexity_score < 90:
            return ComplexityLevel.ADVANCED
        else:
            return ComplexityLevel.EXPERT

    def _calculate_confidence(self, metrics: ComplexityMetrics) -> float:
        """Calculate confidence in the complexity analysis."""
        confidence = 0.8  # Base confidence

        # Higher confidence for longer queries (more data to analyze)
        if metrics.token_count > 20:
            confidence += 0.1

        # Lower confidence for very short queries
        if metrics.token_count < 5:
            confidence -= 0.2

        # Higher confidence when multiple complexity indicators agree
        complexity_indicators = [
            metrics.nesting_depth > 2,
            metrics.join_count > 1,
            metrics.subquery_count > 0,
            metrics.function_count > 2,
        ]

        indicator_count = sum(complexity_indicators)
        if indicator_count >= 3:
            confidence += 0.1
        elif indicator_count == 0:
            confidence -= 0.1

        return max(0.1, min(1.0, confidence))

    def categorize_query(
        self, query: str, database_type: str = "generic"
    ) -> Tuple[ComplexityCategory, ComplexityMetrics]:
        """
        Categorize a query based on its complexity analysis.

        Returns:
            Tuple of (category, metrics)
        """
        metrics = self.analyze_complexity(query, database_type)

        # Find the best matching category
        best_category = None
        best_score = float("inf")

        for category in self.complexity_categories:
            min_range, max_range = category.complexity_range

            if min_range <= metrics.overall_complexity <= max_range:
                # Direct match
                best_category = category
                break
            else:
                # Find closest category
                if metrics.overall_complexity < min_range:
                    distance = min_range - metrics.overall_complexity
                else:
                    distance = metrics.overall_complexity - max_range

                if distance < best_score:
                    best_score = distance
                    best_category = category

        # Fallback to simple category if no match found
        if best_category is None:
            best_category = self.complexity_categories[1]  # simple_queries

        return best_category, metrics

    def get_complexity_distribution(self, queries: List[str]) -> Dict[str, int]:
        """Get distribution of queries across complexity categories."""
        distribution = {category.name: 0 for category in self.complexity_categories}

        for query in queries:
            category, _ = self.categorize_query(query)
            distribution[category.name] += 1

        return distribution

    def suggest_training_weights(self, queries: List[str]) -> Dict[str, float]:
        """Suggest training weights based on query distribution."""
        distribution = self.get_complexity_distribution(queries)
        total_queries = len(queries)

        if total_queries == 0:
            return {}

        weights = {}
        for category in self.complexity_categories:
            count = distribution[category.name]
            if count > 0:
                # Inverse frequency weighting with category bias
                frequency = count / total_queries
                base_weight = 1.0 / frequency if frequency > 0 else 1.0
                adjusted_weight = base_weight * category.training_weight
                weights[category.name] = adjusted_weight
            else:
                weights[category.name] = category.training_weight

        return weights


# Usage examples and utilities
def analyze_query_complexity(
    query: str, database_type: str = "generic"
) -> Dict[str, Any]:
    """Convenience function to analyze a single query."""
    analyzer = QueryComplexityAnalyzer()
    category, metrics = analyzer.categorize_query(query, database_type)

    return {
        "query": query,
        "category": {
            "name": category.name,
            "description": category.description,
            "expected_grade_range": category.expected_grade_range,
        },
        "metrics": asdict(metrics),
        "recommendations": _generate_complexity_recommendations(metrics, category),
    }


def _generate_complexity_recommendations(
    metrics: ComplexityMetrics, category: ComplexityCategory
) -> List[str]:
    """Generate recommendations based on complexity analysis."""
    recommendations = []

    if metrics.cartesian_product_risk > 0.5:
        recommendations.append(
            "Consider adding proper JOIN conditions to avoid cartesian products"
        )

    if metrics.index_usage_score < 0.3:
        recommendations.append("Review WHERE conditions for better index utilization")

    if metrics.readability_score < 0.5:
        recommendations.append(
            "Consider breaking down complex query into smaller parts or adding comments"
        )

    if metrics.nesting_depth > 4:
        recommendations.append(
            "High nesting depth may impact readability and performance"
        )

    if metrics.subquery_count > 3:
        recommendations.append("Multiple subqueries may benefit from CTE refactoring")

    return recommendations


if __name__ == "__main__":
    # Test the complexity analyzer
    test_queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id",
        "WITH RECURSIVE dept_tree AS (SELECT id, name, parent_id FROM departments WHERE parent_id IS NULL UNION ALL SELECT d.id, d.name, d.parent_id FROM departments d JOIN dept_tree dt ON d.parent_id = dt.id) SELECT * FROM dept_tree",
    ]

    analyzer = QueryComplexityAnalyzer()

    for query in test_queries:
        result = analyze_query_complexity(query)
        print(f"\nQuery: {query}")
        print(f"Category: {result['category']['name']}")
        print(f"Complexity Score: {result['metrics']['overall_complexity']:.1f}")
        print(f"Level: {result['metrics']['complexity_level']}")
        if result["recommendations"]:
            print("Recommendations:")
            for rec in result["recommendations"]:
                print(f"  - {rec}")
