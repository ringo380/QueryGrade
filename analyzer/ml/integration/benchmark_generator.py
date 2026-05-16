"""
Automated Benchmark Generator for QueryGrade ML System

This module automatically generates SQL query benchmarks from documentation sources,
creating a comprehensive dataset for training and validation.
"""

import hashlib
import json
import logging
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import sqlparse
from bs4 import BeautifulSoup
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from sqlparse import sql, tokens

from ...models import Query, QueryAnalysis, TrainingData
from .documentation_loader import (
    BenchmarkResult,
    DocumentationLoader,
    DocumentationRule,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuery:
    """A generated benchmark query with expected grade."""

    query_text: str
    expected_score: float
    expected_grade: str
    complexity_level: str  # 'simple', 'medium', 'complex', 'advanced'
    category: str  # 'performance', 'syntax', 'best_practice', 'anti_pattern'
    database_type: str
    explanation: str
    source_rule: str
    variations: List[str]  # Query variations
    confidence: float = 1.0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = timezone.now()


@dataclass
class BenchmarkSet:
    """A set of related benchmark queries."""

    name: str
    description: str
    database_type: str
    benchmarks: List[BenchmarkQuery]
    source: str
    version: str = "1.0"
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = timezone.now()


class QueryPatternGenerator:
    """Generates SQL query patterns for benchmark creation."""

    def __init__(self):
        self.patterns = {
            "simple_select": [
                "SELECT {columns} FROM {table}",
                "SELECT {columns} FROM {table} WHERE {condition}",
                "SELECT {columns} FROM {table} ORDER BY {column}",
                "SELECT {columns} FROM {table} LIMIT {limit}",
            ],
            "medium_select": [
                "SELECT {columns} FROM {table} WHERE {condition} AND {condition2}",
                "SELECT {columns} FROM {table} GROUP BY {group_column} HAVING {having_condition}",
                "SELECT {columns} FROM {table1} INNER JOIN {table2} ON {join_condition}",
                "SELECT {columns} FROM {table} WHERE {column} IN (SELECT {column} FROM {table2})",
            ],
            "complex_select": [
                "SELECT {columns} FROM {table1} t1 LEFT JOIN {table2} t2 ON {join_condition} WHERE {complex_condition}",
                "SELECT {columns} FROM {table} WHERE EXISTS (SELECT 1 FROM {table2} WHERE {correlated_condition})",
                "WITH cte AS (SELECT {columns} FROM {table} WHERE {condition}) SELECT {columns} FROM cte",
                "SELECT {columns} FROM {table1} UNION ALL SELECT {columns} FROM {table2}",
            ],
            "advanced_select": [
                "SELECT {window_function} OVER (PARTITION BY {partition_column} ORDER BY {order_column}) FROM {table}",
                "SELECT {columns} FROM {table1} t1 CROSS JOIN {table2} t2 WHERE {cartesian_condition}",
                "SELECT CASE WHEN {condition} THEN {value1} ELSE {value2} END FROM {table}",
            ],
        }

        self.anti_patterns = {
            "performance": [
                "SELECT * FROM {large_table}",  # Select *
                "SELECT {columns} FROM {table1} CROSS JOIN {table2}",  # Cartesian product
                "SELECT {columns} FROM {table} WHERE {function}({column}) = {value}",  # Function in WHERE
                "SELECT {columns} FROM {table} ORDER BY RAND()",  # Random ordering
            ],
            "syntax": [
                "SELECT {columns} FROM {table} WHERE {column} = NULL",  # Wrong NULL comparison
                "SELECT {columns} FROM {table} GROUP BY {column} ORDER BY {other_column}",  # Invalid GROUP BY
                "SELECT COUNT(*) FROM {table} WHERE 1=1",  # Redundant condition
            ],
        }

        # Sample data for pattern generation
        self.sample_tables = [
            "users",
            "orders",
            "products",
            "customers",
            "employees",
            "departments",
        ]
        self.sample_columns = [
            "id",
            "name",
            "email",
            "created_at",
            "status",
            "price",
            "quantity",
        ]
        self.sample_conditions = [
            "id > 100",
            'status = "active"',
            'created_at > "2023-01-01"',
        ]

    def generate_pattern_queries(self, pattern_type: str, count: int = 10) -> List[str]:
        """Generate queries based on pattern type."""
        if pattern_type not in self.patterns:
            return []

        queries = []
        patterns = self.patterns[pattern_type]

        for _ in range(count):
            pattern = random.choice(patterns)
            query = self._substitute_pattern(pattern)
            queries.append(query)

        return queries

    def generate_anti_pattern_queries(
        self, anti_pattern_type: str, count: int = 5
    ) -> List[str]:
        """Generate anti-pattern queries for negative examples."""
        if anti_pattern_type not in self.anti_patterns:
            return []

        queries = []
        patterns = self.anti_patterns[anti_pattern_type]

        for _ in range(count):
            pattern = random.choice(patterns)
            query = self._substitute_pattern(pattern)
            queries.append(query)

        return queries

    def _substitute_pattern(self, pattern: str) -> str:
        """Substitute placeholders in pattern with sample data."""
        substitutions = {
            "{columns}": random.choice(
                ["*", "id", "name, email", "COUNT(*)", "id, name"]
            ),
            "{table}": random.choice(self.sample_tables),
            "{table1}": random.choice(self.sample_tables),
            "{table2}": random.choice([t for t in self.sample_tables if t != "users"]),
            "{column}": random.choice(self.sample_columns),
            "{condition}": random.choice(self.sample_conditions),
            "{condition2}": random.choice(
                ["price > 50", "quantity < 100", 'name LIKE "%test%"']
            ),
            "{join_condition}": "users.id = orders.user_id",
            "{complex_condition}": 'users.status = "active" AND orders.total > 100',
            "{correlated_condition}": "table2.user_id = table.id",
            "{limit}": str(random.randint(10, 1000)),
            "{group_column}": random.choice(["status", "category", "department"]),
            "{having_condition}": "COUNT(*) > 5",
            "{large_table}": "large_transactions",
            "{function}": random.choice(["UPPER", "LOWER", "SUBSTRING"]),
            "{value}": '"test"',
            "{window_function}": random.choice(["ROW_NUMBER()", "RANK()", "COUNT(*)"]),
            "{partition_column}": random.choice(["department", "category", "status"]),
            "{order_column}": random.choice(["created_at", "id", "name"]),
            "{cartesian_condition}": "t1.id != t2.id",
            "{value1}": '"High"',
            "{value2}": '"Low"',
        }

        for placeholder, value in substitutions.items():
            pattern = pattern.replace(placeholder, value)

        return pattern


class BenchmarkGenerator:
    """Main benchmark generator class."""

    def __init__(self):
        self.documentation_loader = DocumentationLoader()
        self.pattern_generator = QueryPatternGenerator()
        self.benchmark_dir = os.path.join(settings.BASE_DIR, "ml_benchmarks")
        os.makedirs(self.benchmark_dir, exist_ok=True)

        # Scoring rules for different query categories
        self.scoring_rules = {
            "simple": {"base_score": 85, "variance": 10},
            "medium": {"base_score": 75, "variance": 15},
            "complex": {"base_score": 65, "variance": 20},
            "advanced": {"base_score": 55, "variance": 25},
            "anti_pattern": {"base_score": 25, "variance": 15},
        }

    def generate_comprehensive_benchmarks(
        self, database_types: List[str] = None
    ) -> List[BenchmarkSet]:
        """Generate comprehensive benchmark sets for training."""
        if database_types is None:
            database_types = ["mysql", "postgresql", "sqlite", "generic"]

        benchmark_sets = []

        for db_type in database_types:
            logger.info(f"Generating benchmarks for {db_type}")

            # Generate pattern-based benchmarks
            pattern_benchmarks = self._generate_pattern_benchmarks(db_type)

            # Generate documentation-based benchmarks
            doc_benchmarks = self._generate_documentation_benchmarks(db_type)

            # Generate performance benchmarks
            perf_benchmarks = self._generate_performance_benchmarks(db_type)

            # Combine all benchmarks
            all_benchmarks = pattern_benchmarks + doc_benchmarks + perf_benchmarks

            if all_benchmarks:
                benchmark_set = BenchmarkSet(
                    name=f"{db_type}_comprehensive_benchmarks",
                    description=f"Comprehensive benchmark set for {db_type} database",
                    database_type=db_type,
                    benchmarks=all_benchmarks,
                    source="automated_generation",
                    version="1.0",
                )
                benchmark_sets.append(benchmark_set)

        return benchmark_sets

    def _generate_pattern_benchmarks(self, database_type: str) -> List[BenchmarkQuery]:
        """Generate benchmarks based on query patterns."""
        benchmarks = []

        for complexity in ["simple", "medium", "complex", "advanced"]:
            queries = self.pattern_generator.generate_pattern_queries(
                complexity, count=10
            )

            for query_text in queries:
                score = self._calculate_pattern_score(query_text, complexity)
                grade = self._score_to_grade(score)

                benchmark = BenchmarkQuery(
                    query_text=query_text,
                    expected_score=score,
                    expected_grade=grade,
                    complexity_level=complexity,
                    category="pattern_based",
                    database_type=database_type,
                    explanation=f"Pattern-based {complexity} query with expected score {score:.1f}",
                    source_rule=f"pattern_{complexity}",
                    variations=self._generate_query_variations(query_text),
                )
                benchmarks.append(benchmark)

        # Add anti-pattern examples
        for anti_type in ["performance", "syntax"]:
            queries = self.pattern_generator.generate_anti_pattern_queries(
                anti_type, count=5
            )

            for query_text in queries:
                score = self._calculate_anti_pattern_score(query_text, anti_type)
                grade = self._score_to_grade(score)

                benchmark = BenchmarkQuery(
                    query_text=query_text,
                    expected_score=score,
                    expected_grade=grade,
                    complexity_level="anti_pattern",
                    category=f"anti_pattern_{anti_type}",
                    database_type=database_type,
                    explanation=f"Anti-pattern example for {anti_type} with low score {score:.1f}",
                    source_rule=f"anti_pattern_{anti_type}",
                    variations=[],
                )
                benchmarks.append(benchmark)

        return benchmarks

    def _generate_documentation_benchmarks(
        self, database_type: str
    ) -> List[BenchmarkQuery]:
        """Generate benchmarks from documentation sources."""
        benchmarks = []

        try:
            # Load documentation rules
            rules = self.documentation_loader.load_all_rules(database_type)

            for rule in rules:
                if rule.example_good:
                    # Create positive example
                    score = 90 - (rule.score_impact if rule.score_impact < 0 else 0)
                    grade = self._score_to_grade(score)

                    benchmark = BenchmarkQuery(
                        query_text=rule.example_good,
                        expected_score=score,
                        expected_grade=grade,
                        complexity_level=self._infer_complexity(rule.example_good),
                        category=rule.rule_type,
                        database_type=database_type,
                        explanation=f"Positive example: {rule.description}",
                        source_rule=rule.rule_id,
                        variations=self._generate_query_variations(rule.example_good),
                        confidence=rule.confidence,
                    )
                    benchmarks.append(benchmark)

                if rule.example_bad:
                    # Create negative example
                    score = 40 + (rule.score_impact if rule.score_impact < 0 else -20)
                    grade = self._score_to_grade(score)

                    benchmark = BenchmarkQuery(
                        query_text=rule.example_bad,
                        expected_score=score,
                        expected_grade=grade,
                        complexity_level=self._infer_complexity(rule.example_bad),
                        category=f"{rule.rule_type}_negative",
                        database_type=database_type,
                        explanation=f"Negative example: {rule.description}",
                        source_rule=rule.rule_id,
                        variations=[],
                        confidence=rule.confidence,
                    )
                    benchmarks.append(benchmark)

        except Exception as e:
            logger.warning(
                f"Error generating documentation benchmarks for {database_type}: {e}"
            )

        return benchmarks

    def _generate_performance_benchmarks(
        self, database_type: str
    ) -> List[BenchmarkQuery]:
        """Generate performance-focused benchmarks."""
        benchmarks = []

        # Define performance scenarios
        performance_scenarios = [
            {
                "query": "SELECT * FROM users WHERE id = 1",
                "score": 85,
                "explanation": "Simple primary key lookup - excellent performance",
                "category": "performance_good",
            },
            {
                "query": 'SELECT * FROM users WHERE UPPER(name) = "JOHN"',
                "score": 35,
                "explanation": "Function in WHERE clause prevents index usage",
                "category": "performance_bad",
            },
            {
                "query": "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id",
                "score": 78,
                "explanation": "Efficient aggregation with proper JOIN",
                "category": "performance_good",
            },
            {
                "query": "SELECT * FROM users u CROSS JOIN orders o WHERE u.id = o.user_id",
                "score": 25,
                "explanation": "Cartesian product - very poor performance",
                "category": "performance_bad",
            },
        ]

        for scenario in performance_scenarios:
            benchmark = BenchmarkQuery(
                query_text=scenario["query"],
                expected_score=scenario["score"],
                expected_grade=self._score_to_grade(scenario["score"]),
                complexity_level=self._infer_complexity(scenario["query"]),
                category=scenario["category"],
                database_type=database_type,
                explanation=scenario["explanation"],
                source_rule="performance_benchmark",
                variations=self._generate_query_variations(scenario["query"]),
            )
            benchmarks.append(benchmark)

        return benchmarks

    def _calculate_pattern_score(self, query_text: str, complexity: str) -> float:
        """Calculate expected score for pattern-based query."""
        base_score = self.scoring_rules[complexity]["base_score"]
        variance = self.scoring_rules[complexity]["variance"]

        # Add some randomness for realistic scoring
        score = base_score + random.uniform(-variance / 2, variance / 2)

        # Apply query-specific adjustments
        query_upper = query_text.upper()

        # Positive adjustments
        if "WHERE" in query_upper and "id =" in query_upper:
            score += 5  # Primary key usage
        if "LIMIT" in query_upper:
            score += 3  # Result limiting

        # Negative adjustments
        if "SELECT *" in query_upper:
            score -= 10  # Select all
        if "CROSS JOIN" in query_upper:
            score -= 15  # Cartesian product

        return max(0, min(100, score))

    def _calculate_anti_pattern_score(self, query_text: str, anti_type: str) -> float:
        """Calculate score for anti-pattern queries."""
        base_score = self.scoring_rules["anti_pattern"]["base_score"]
        variance = self.scoring_rules["anti_pattern"]["variance"]

        score = base_score + random.uniform(-variance / 2, variance / 2)

        # Apply specific penalties
        query_upper = query_text.upper()

        if anti_type == "performance":
            if "SELECT *" in query_upper:
                score -= 10
            if "CROSS JOIN" in query_upper:
                score -= 20
            if "ORDER BY RAND()" in query_upper:
                score -= 15

        return max(0, min(100, score))

    def _generate_query_variations(self, base_query: str, count: int = 3) -> List[str]:
        """Generate variations of a base query."""
        variations = []

        try:
            parsed = sqlparse.parse(base_query)[0]
            query_text = str(parsed).strip()

            # Simple variations
            variations.extend(
                [
                    query_text.replace("SELECT", "select"),  # Case variation
                    f"{query_text.rstrip(';')};",  # Semicolon variation
                    re.sub(r"\s+", " ", query_text),  # Whitespace normalization
                ]
            )

            # More complex variations could be added here
            # e.g., alias variations, equivalent reformulations

        except Exception as e:
            logger.warning(f"Error generating variations for query: {e}")

        return variations[:count]

    def _infer_complexity(self, query_text: str) -> str:
        """Infer complexity level from query text."""
        query_upper = query_text.upper()

        complexity_indicators = {
            "simple": ["SELECT", "FROM", "WHERE"],
            "medium": ["JOIN", "GROUP BY", "SUBQUERY", "IN ("],
            "complex": ["LEFT JOIN", "RIGHT JOIN", "EXISTS", "UNION"],
            "advanced": ["WINDOW", "CTE", "RECURSIVE", "CROSS JOIN"],
        }

        for level, indicators in complexity_indicators.items():
            if any(indicator in query_upper for indicator in indicators):
                return level

        return "simple"

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def save_benchmark_set(self, benchmark_set: BenchmarkSet) -> str:
        """Save benchmark set to file."""
        filename = f"{benchmark_set.name}_{benchmark_set.version}.json"
        filepath = os.path.join(self.benchmark_dir, filename)

        # Convert to serializable format
        data = {
            "metadata": {
                "name": benchmark_set.name,
                "description": benchmark_set.description,
                "database_type": benchmark_set.database_type,
                "source": benchmark_set.source,
                "version": benchmark_set.version,
                "created_at": benchmark_set.created_at.isoformat(),
                "count": len(benchmark_set.benchmarks),
            },
            "benchmarks": [asdict(benchmark) for benchmark in benchmark_set.benchmarks],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(
            f"Saved benchmark set with {len(benchmark_set.benchmarks)} benchmarks to {filepath}"
        )
        return filepath

    def load_benchmark_set(self, filepath: str) -> Optional[BenchmarkSet]:
        """Load benchmark set from file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            metadata = data["metadata"]
            benchmarks = [
                BenchmarkQuery(**benchmark_data)
                for benchmark_data in data["benchmarks"]
            ]

            return BenchmarkSet(
                name=metadata["name"],
                description=metadata["description"],
                database_type=metadata["database_type"],
                benchmarks=benchmarks,
                source=metadata["source"],
                version=metadata["version"],
            )

        except Exception as e:
            logger.error(f"Error loading benchmark set from {filepath}: {e}")
            return None

    def create_training_data_from_benchmarks(
        self, benchmark_sets: List[BenchmarkSet]
    ) -> int:
        """Create TrainingData objects from benchmark sets."""
        created_count = 0

        with transaction.atomic():
            for benchmark_set in benchmark_sets:
                for benchmark in benchmark_set.benchmarks:
                    try:
                        # Create or get Query object
                        query_hash = hashlib.md5(
                            f"{benchmark.query_text.lower()}|{benchmark.database_type}".encode(),
                            usedforsecurity=False,
                        ).hexdigest()

                        query, created = Query.objects.get_or_create(
                            query_hash=query_hash,
                            defaults={
                                "sql_text": benchmark.query_text,
                                "query_type": self._get_query_type(
                                    benchmark.query_text
                                ),
                                "estimated_complexity": self._estimate_complexity(
                                    benchmark.query_text
                                ),
                            },
                        )

                        # Create TrainingData
                        training_data, created = TrainingData.objects.get_or_create(
                            query=query,
                            defaults={
                                "user_grade_avg": (benchmark.expected_score / 100)
                                * 5,  # Convert to 1-5 scale
                                "user_grade_count": 1,
                                "user_grade_stddev": 0.1,
                                "system_grade": benchmark.expected_grade,
                                "system_score": benchmark.expected_score,
                                "query_complexity": self._estimate_complexity(
                                    benchmark.query_text
                                ),
                                "is_validated": True,  # Mark as validated benchmark
                            },
                        )

                        if created:
                            created_count += 1

                    except Exception as e:
                        logger.warning(
                            f"Error creating training data for benchmark query: {e}"
                        )

        logger.info(f"Created {created_count} training data records from benchmarks")
        return created_count

    def _get_query_type(self, query_text: str) -> str:
        """Extract query type from SQL text."""
        try:
            parsed = sqlparse.parse(query_text)[0]
            for token in parsed.flatten():
                if token.ttype in (tokens.Keyword.DML, tokens.Keyword.DDL):
                    return token.value.upper()
        except Exception:
            pass
        return "UNKNOWN"

    def _estimate_complexity(self, query_text: str) -> int:
        """Estimate query complexity (0-100)."""
        query_upper = query_text.upper()

        complexity = 0

        # Basic complexity indicators
        complexity += query_upper.count("JOIN") * 10
        complexity += query_upper.count("SUBQUERY") * 15
        complexity += query_upper.count("UNION") * 12
        complexity += query_upper.count("CASE") * 8
        complexity += query_upper.count("GROUP BY") * 6
        complexity += query_upper.count("ORDER BY") * 3
        complexity += query_upper.count("HAVING") * 8

        # Length-based complexity
        complexity += len(query_text) // 50

        return min(100, complexity)


# Usage example and main entry point
def generate_default_benchmarks():
    """Generate default benchmark sets for all database types."""
    generator = BenchmarkGenerator()
    benchmark_sets = generator.generate_comprehensive_benchmarks()

    saved_files = []
    for benchmark_set in benchmark_sets:
        filepath = generator.save_benchmark_set(benchmark_set)
        saved_files.append(filepath)

    # Create training data
    training_count = generator.create_training_data_from_benchmarks(benchmark_sets)

    logger.info(
        f"Generated {len(benchmark_sets)} benchmark sets with {training_count} training examples"
    )
    return saved_files, training_count


if __name__ == "__main__":
    generate_default_benchmarks()
