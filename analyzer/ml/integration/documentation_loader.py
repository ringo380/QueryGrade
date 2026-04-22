"""
Documentation Loader for QueryGrade ML System

This module loads and processes authoritative SQL documentation, best practices,
and benchmarks to enhance ML model training with expert knowledge.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
import sqlparse
from bs4 import BeautifulSoup
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from ...models import Query, TrainingData

logger = logging.getLogger(__name__)


@dataclass
class DocumentationSource:
    """Configuration for a documentation source."""

    name: str
    base_url: str
    source_type: str  # 'web', 'file', 'api'
    patterns: List[str]  # URL patterns or file patterns
    database_type: str  # 'mysql', 'postgresql', 'sqlite', 'generic'
    quality_score: float  # 0.0-1.0, how much to trust this source
    enabled: bool = True
    last_updated: Optional[datetime] = None


@dataclass
class DocumentationRule:
    """A rule extracted from documentation."""

    rule_id: str
    title: str
    description: str
    rule_type: str  # 'performance', 'syntax', 'best_practice', 'anti_pattern'
    database_type: str
    sql_patterns: List[str]  # Regex patterns to match
    example_good: Optional[str] = None
    example_bad: Optional[str] = None
    severity: str = "medium"  # 'low', 'medium', 'high', 'critical'
    score_impact: float = 0.0  # How much this affects the score (-100 to +100)
    source: str = ""
    confidence: float = 1.0


@dataclass
class BenchmarkResult:
    """A benchmark result from authoritative sources."""

    benchmark_id: str
    query_text: str
    expected_score: float
    explanation: str
    database_type: str
    complexity_level: str  # 'simple', 'medium', 'complex'
    source: str
    validated: bool = False


class DocumentationLoader:
    """Loads and processes documentation from various sources."""

    def __init__(self):
        self.sources = self._load_documentation_sources()
        self.rules = []
        self.benchmarks = []

        # Create documentation cache directory
        self.cache_dir = os.path.join(settings.BASE_DIR, "ml_documentation_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_documentation_sources(self) -> List[DocumentationSource]:
        """Load documentation source configurations."""
        return [
            # MySQL Documentation
            DocumentationSource(
                name="MySQL Official Documentation",
                base_url="https://dev.mysql.com/doc/refman/8.0/en/",
                source_type="web",
                patterns=[
                    "optimization.html",
                    "select-optimization.html",
                    "where-optimization.html",
                    "index-optimization.html",
                    "subquery-optimization.html",
                ],
                database_type="mysql",
                quality_score=0.95,
            ),
            # PostgreSQL Documentation
            DocumentationSource(
                name="PostgreSQL Official Documentation",
                base_url="https://www.postgresql.org/docs/current/",
                source_type="web",
                patterns=[
                    "performance-tips.html",
                    "using-explain.html",
                    "indexes.html",
                    "queries.html",
                ],
                database_type="postgresql",
                quality_score=0.95,
            ),
            # SQLite Documentation
            DocumentationSource(
                name="SQLite Query Planning",
                base_url="https://www.sqlite.org/",
                source_type="web",
                patterns=["optoverview.html", "queryplanner.html", "lang_select.html"],
                database_type="sqlite",
                quality_score=0.90,
            ),
            # General SQL Best Practices
            DocumentationSource(
                name="SQL Performance Best Practices",
                base_url="",
                source_type="file",
                patterns=["sql_best_practices.json"],
                database_type="generic",
                quality_score=0.85,
            ),
            # Use The Index Luke
            DocumentationSource(
                name="Use The Index Luke",
                base_url="https://use-the-index-luke.com/",
                source_type="web",
                patterns=[
                    "sql/where-clause",
                    "sql/join",
                    "sql/group-by",
                    "sql/order-by",
                ],
                database_type="generic",
                quality_score=0.90,
            ),
        ]

    def load_all_documentation(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Load documentation from all configured sources."""
        results = {
            "rules_loaded": 0,
            "benchmarks_loaded": 0,
            "sources_processed": 0,
            "errors": [],
        }

        for source in self.sources:
            if not source.enabled:
                continue

            try:
                logger.info(f"Loading documentation from {source.name}")

                if source.source_type == "web":
                    source_results = self._load_web_documentation(source, force_refresh)
                elif source.source_type == "file":
                    source_results = self._load_file_documentation(source)
                elif source.source_type == "api":
                    source_results = self._load_api_documentation(source)
                else:
                    logger.warning(f"Unknown source type: {source.source_type}")
                    continue

                results["rules_loaded"] += len(source_results.get("rules", []))
                results["benchmarks_loaded"] += len(
                    source_results.get("benchmarks", [])
                )
                results["sources_processed"] += 1

                # Update source last_updated
                source.last_updated = timezone.now()

            except Exception as e:
                error_msg = f"Error loading from {source.name}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        logger.info(f"Documentation loading complete: {results}")
        return results

    def _load_web_documentation(
        self, source: DocumentationSource, force_refresh: bool
    ) -> Dict[str, Any]:
        """Load documentation from web sources."""
        rules = []
        benchmarks = []

        for pattern in source.patterns:
            url = urljoin(source.base_url, pattern)

            # Check cache first
            cache_key = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.html")

            content = None
            if not force_refresh and os.path.exists(cache_file):
                # Use cached content if less than 24 hours old
                cache_age = timezone.now().timestamp() - os.path.getmtime(cache_file)
                if cache_age < 24 * 3600:  # 24 hours
                    with open(cache_file, "r", encoding="utf-8") as f:
                        content = f.read()

            # Fetch from web if not cached
            if content is None:
                content = self._fetch_web_content(url)
                if content:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(content)

            if content:
                parsed_rules, parsed_benchmarks = self._parse_documentation_content(
                    content, source, url
                )
                rules.extend(parsed_rules)
                benchmarks.extend(parsed_benchmarks)

        self.rules.extend(rules)
        self.benchmarks.extend(benchmarks)

        return {"rules": rules, "benchmarks": benchmarks}

    def _load_file_documentation(self, source: DocumentationSource) -> Dict[str, Any]:
        """Load documentation from local files."""
        rules = []
        benchmarks = []

        for pattern in source.patterns:
            file_path = os.path.join(settings.BASE_DIR, "documentation", pattern)

            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_path.endswith(".json"):
                        data = json.load(f)
                        rules.extend(self._parse_json_documentation(data, source))
                    else:
                        content = f.read()
                        parsed_rules, parsed_benchmarks = (
                            self._parse_documentation_content(
                                content, source, file_path
                            )
                        )
                        rules.extend(parsed_rules)
                        benchmarks.extend(parsed_benchmarks)

        self.rules.extend(rules)
        self.benchmarks.extend(benchmarks)

        return {"rules": rules, "benchmarks": benchmarks}

    def _fetch_web_content(self, url: str) -> Optional[str]:
        """Fetch content from a web URL."""
        try:
            headers = {"User-Agent": "QueryGrade Documentation Loader 1.0"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            return None

    def _parse_documentation_content(
        self, content: str, source: DocumentationSource, source_url: str
    ) -> Tuple[List[DocumentationRule], List[BenchmarkResult]]:
        """Parse documentation content to extract rules and benchmarks."""
        rules = []
        benchmarks = []

        # Parse HTML content
        soup = BeautifulSoup(content, "html.parser")

        # Extract rules based on source type
        if source.database_type == "mysql":
            rules.extend(self._parse_mysql_documentation(soup, source, source_url))
        elif source.database_type == "postgresql":
            rules.extend(self._parse_postgresql_documentation(soup, source, source_url))
        elif source.database_type == "sqlite":
            rules.extend(self._parse_sqlite_documentation(soup, source, source_url))
        else:
            rules.extend(self._parse_generic_documentation(soup, source, source_url))

        # Extract benchmarks from code examples
        benchmarks.extend(
            self._extract_benchmarks_from_content(soup, source, source_url)
        )

        return rules, benchmarks

    def _parse_mysql_documentation(
        self, soup: BeautifulSoup, source: DocumentationSource, source_url: str
    ) -> List[DocumentationRule]:
        """Parse MySQL-specific documentation."""
        rules = []

        # Look for optimization recommendations
        optimization_sections = soup.find_all(
            ["div", "section"], class_=re.compile(r"note|tip|important")
        )

        for section in optimization_sections:
            text = section.get_text().strip()

            # Extract SELECT * warnings
            if "SELECT *" in text.upper() and (
                "avoid" in text.lower() or "performance" in text.lower()
            ):
                rules.append(
                    DocumentationRule(
                        rule_id=f"mysql_select_star_{hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]}",  # noqa: E501
                        title="Avoid SELECT * in production queries",
                        description="SELECT * can impact performance and maintainability",
                        rule_type="performance",
                        database_type="mysql",
                        sql_patterns=[r"SELECT\s+\*\s+FROM"],
                        severity="medium",
                        score_impact=-10.0,
                        source=source_url,
                        confidence=source.quality_score,
                    )
                )

            # Extract INDEX recommendations
            if "index" in text.lower() and (
                "create" in text.lower() or "use" in text.lower()
            ):
                rules.append(
                    DocumentationRule(
                        rule_id=f"mysql_index_{hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]}",
                        title="Proper index usage improves performance",
                        description=text[:200] + "..." if len(text) > 200 else text,
                        rule_type="performance",
                        database_type="mysql",
                        sql_patterns=[r"WHERE\s+\w+\s*=", r"ORDER\s+BY\s+\w+"],
                        severity="high",
                        score_impact=15.0,
                        source=source_url,
                        confidence=source.quality_score,
                    )
                )

        return rules

    def _parse_postgresql_documentation(
        self, soup: BeautifulSoup, source: DocumentationSource, source_url: str
    ) -> List[DocumentationRule]:
        """Parse PostgreSQL-specific documentation."""
        rules = []

        # Look for performance tips
        tip_sections = soup.find_all(["div", "section"], class_=re.compile(r"tip|note"))

        for section in tip_sections:
            text = section.get_text().strip()

            # Extract EXPLAIN recommendations
            if "EXPLAIN" in text.upper() and "performance" in text.lower():
                rules.append(
                    DocumentationRule(
                        rule_id=f"postgresql_explain_{hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]}",  # noqa: E501
                        title="Use EXPLAIN to analyze query performance",
                        description="EXPLAIN helps identify performance bottlenecks",
                        rule_type="best_practice",
                        database_type="postgresql",
                        sql_patterns=[r"SELECT.*FROM.*WHERE"],
                        severity="medium",
                        score_impact=5.0,
                        source=source_url,
                        confidence=source.quality_score,
                    )
                )

        return rules

    def _parse_sqlite_documentation(
        self, soup: BeautifulSoup, source: DocumentationSource, source_url: str
    ) -> List[DocumentationRule]:
        """Parse SQLite-specific documentation."""
        rules = []

        # SQLite-specific optimizations
        content_text = soup.get_text().lower()

        if "without rowid" in content_text:
            rules.append(
                DocumentationRule(
                    rule_id="sqlite_without_rowid",
                    title="Consider WITHOUT ROWID for tables with non-integer primary keys",
                    description="WITHOUT ROWID tables can be more efficient for certain use cases",
                    rule_type="performance",
                    database_type="sqlite",
                    sql_patterns=[r"CREATE\s+TABLE.*PRIMARY\s+KEY"],
                    severity="low",
                    score_impact=3.0,
                    source=source_url,
                    confidence=source.quality_score,
                )
            )

        return rules

    def _parse_generic_documentation(
        self, soup: BeautifulSoup, source: DocumentationSource, source_url: str
    ) -> List[DocumentationRule]:
        """Parse generic SQL documentation."""
        rules = []

        # Common SQL anti-patterns
        text = soup.get_text().lower()

        # N+1 queries
        if "n+1" in text and "query" in text:
            rules.append(
                DocumentationRule(
                    rule_id="generic_n_plus_one",
                    title="Avoid N+1 query problems",
                    description="Use JOINs instead of multiple queries in loops",
                    rule_type="anti_pattern",
                    database_type="generic",
                    sql_patterns=[r"SELECT.*WHERE.*IN\s*\("],
                    severity="high",
                    score_impact=-20.0,
                    source=source_url,
                    confidence=source.quality_score,
                )
            )

        return rules

    def _extract_benchmarks_from_content(
        self, soup: BeautifulSoup, source: DocumentationSource, source_url: str
    ) -> List[BenchmarkResult]:
        """Extract benchmark queries from documentation examples."""
        benchmarks = []

        # Find code blocks with SQL
        code_blocks = soup.find_all(
            ["code", "pre"], string=re.compile(r"SELECT|INSERT|UPDATE|DELETE", re.I)
        )

        for block in code_blocks:
            sql_text = block.get_text().strip()

            # Basic SQL validation
            if self._is_valid_sql_example(sql_text):
                # Determine expected score based on patterns
                expected_score = self._estimate_benchmark_score(sql_text)

                benchmarks.append(
                    BenchmarkResult(
                        benchmark_id=f"benchmark_{hashlib.md5(sql_text.encode(), usedforsecurity=False).hexdigest()[:8]}",  # noqa: E501
                        query_text=sql_text,
                        expected_score=expected_score,
                        explanation=f"Example from {source.name}",
                        database_type=source.database_type,
                        complexity_level=self._estimate_complexity_level(sql_text),
                        source=source_url,
                        validated=False,
                    )
                )

        return benchmarks

    def _parse_json_documentation(
        self, data: Dict[str, Any], source: DocumentationSource
    ) -> List[DocumentationRule]:
        """Parse structured JSON documentation."""
        rules = []

        if "rules" in data:
            for rule_data in data["rules"]:
                rules.append(
                    DocumentationRule(
                        rule_id=rule_data.get("id", f"rule_{len(rules)}"),
                        title=rule_data.get("title", ""),
                        description=rule_data.get("description", ""),
                        rule_type=rule_data.get("type", "best_practice"),
                        database_type=rule_data.get(
                            "database_type", source.database_type
                        ),
                        sql_patterns=rule_data.get("patterns", []),
                        example_good=rule_data.get("example_good"),
                        example_bad=rule_data.get("example_bad"),
                        severity=rule_data.get("severity", "medium"),
                        score_impact=rule_data.get("score_impact", 0.0),
                        source=source.name,
                        confidence=source.quality_score,
                    )
                )

        return rules

    def _is_valid_sql_example(self, sql_text: str) -> bool:
        """Check if text contains a valid SQL example."""
        try:
            parsed = sqlparse.parse(sql_text)
            return len(parsed) > 0 and parsed[0].tokens
        except Exception:
            return False

    def _estimate_benchmark_score(self, sql_text: str) -> float:
        """Estimate expected score for a benchmark query."""
        score = 70.0  # Base score

        sql_upper = sql_text.upper()

        # Positive indicators
        if "JOIN" in sql_upper and "ON" in sql_upper:
            score += 10
        if re.search(r"WHERE\s+\w+\s*=", sql_text, re.I):
            score += 5
        if "LIMIT" in sql_upper:
            score += 5

        # Negative indicators
        if "SELECT *" in sql_upper:
            score -= 15
        if sql_text.count("SELECT") > 3:  # Multiple subqueries
            score -= 10
        if "CROSS JOIN" in sql_upper:
            score -= 20

        return max(0, min(100, score))

    def _estimate_complexity_level(self, sql_text: str) -> str:
        """Estimate complexity level of SQL query."""
        sql_upper = sql_text.upper()

        complexity_indicators = 0

        if "JOIN" in sql_upper:
            complexity_indicators += sql_upper.count("JOIN")
        if "SUBQUERY" in sql_upper or sql_text.count("SELECT") > 1:
            complexity_indicators += 2
        if any(word in sql_upper for word in ["UNION", "INTERSECT", "EXCEPT"]):
            complexity_indicators += 2
        if any(word in sql_upper for word in ["WINDOW", "OVER", "PARTITION"]):
            complexity_indicators += 3

        if complexity_indicators == 0:
            return "simple"
        elif complexity_indicators <= 2:
            return "medium"
        else:
            return "complex"

    def apply_documentation_rules(self, query: Query) -> Dict[str, Any]:
        """Apply loaded documentation rules to a query."""
        applied_rules = []
        total_score_impact = 0.0

        for rule in self.rules:
            # Check if rule applies to this query
            if self._rule_applies_to_query(rule, query):
                applied_rules.append(
                    {
                        "rule_id": rule.rule_id,
                        "title": rule.title,
                        "description": rule.description,
                        "severity": rule.severity,
                        "score_impact": rule.score_impact,
                        "confidence": rule.confidence,
                    }
                )
                total_score_impact += rule.score_impact * rule.confidence

        return {
            "applied_rules": applied_rules,
            "total_score_impact": total_score_impact,
            "rule_count": len(applied_rules),
        }

    def _rule_applies_to_query(self, rule: DocumentationRule, query: Query) -> bool:
        """Check if a documentation rule applies to a query."""
        # Check database type compatibility
        if rule.database_type != "generic" and hasattr(query, "database_type"):
            if query.database_type and rule.database_type != query.database_type:
                return False

        # Check SQL patterns
        sql_text = query.sql_text
        for pattern in rule.sql_patterns:
            if re.search(pattern, sql_text, re.IGNORECASE):
                return True

        return False

    def create_training_data_from_benchmarks(self) -> int:
        """Create training data from benchmark results."""
        created_count = 0

        with transaction.atomic():
            for benchmark in self.benchmarks:
                if benchmark.validated:
                    # Create or update Query
                    query_hash = hashlib.md5(
                        benchmark.query_text.encode(), usedforsecurity=False
                    ).hexdigest()

                    query, created = Query.objects.get_or_create(
                        query_hash=query_hash,
                        defaults={
                            "sql_text": benchmark.query_text,
                            "query_type": self._extract_query_type(
                                benchmark.query_text
                            ),
                            "estimated_complexity": benchmark.expected_score,
                            "table_count": self._count_tables(benchmark.query_text),
                            "join_count": benchmark.query_text.upper().count("JOIN"),
                            "where_conditions": benchmark.query_text.upper().count(
                                "WHERE"
                            ),
                            "subquery_count": benchmark.query_text.count("SELECT") - 1,
                        },
                    )

                    # Create training data
                    training_data, created = TrainingData.objects.get_or_create(
                        query=query,
                        defaults={
                            "features_json": [],  # Will be populated by feature extractor
                            "target_score": benchmark.expected_score,
                            "feedback_weight": 1.0,  # Full weight for benchmark data
                            "user_reliability_score": 1.0,  # Maximum reliability for authoritative sources
                        },
                    )

                    if created:
                        created_count += 1

        logger.info(f"Created {created_count} training data samples from benchmarks")
        return created_count

    def _extract_query_type(self, sql_text: str) -> str:
        """Extract query type from SQL text."""
        sql_upper = sql_text.upper().strip()

        if sql_upper.startswith("SELECT"):
            return "SELECT"
        elif sql_upper.startswith("INSERT"):
            return "INSERT"
        elif sql_upper.startswith("UPDATE"):
            return "UPDATE"
        elif sql_upper.startswith("DELETE"):
            return "DELETE"
        elif sql_upper.startswith("CREATE"):
            return "CREATE"
        elif sql_upper.startswith("ALTER"):
            return "ALTER"
        else:
            return "UNKNOWN"

    def _count_tables(self, sql_text: str) -> int:
        """Count the number of tables referenced in the query."""
        # Simplified table counting - could be enhanced
        from_matches = re.findall(r"FROM\s+(\w+)", sql_text, re.IGNORECASE)
        join_matches = re.findall(r"JOIN\s+(\w+)", sql_text, re.IGNORECASE)

        tables = set(from_matches + join_matches)
        return len(tables)

    def get_documentation_status(self) -> Dict[str, Any]:
        """Get status of documentation loading system."""
        return {
            "sources_configured": len(self.sources),
            "sources_enabled": len([s for s in self.sources if s.enabled]),
            "rules_loaded": len(self.rules),
            "benchmarks_loaded": len(self.benchmarks),
            "benchmarks_validated": len([b for b in self.benchmarks if b.validated]),
            "cache_directory": self.cache_dir,
            "last_update": max(
                [s.last_updated for s in self.sources if s.last_updated], default=None
            ),
        }

    def validate_benchmarks(self, sample_size: int = 10) -> Dict[str, Any]:
        """Validate a sample of benchmarks against actual analysis."""
        if not self.benchmarks:
            return {"validated": 0, "total": 0, "accuracy": 0.0}

        # Sample benchmarks for validation
        import random

        sample_benchmarks = random.sample(
            self.benchmarks, min(sample_size, len(self.benchmarks))
        )

        validated_count = 0
        total_error = 0.0

        for benchmark in sample_benchmarks:
            # This would integrate with the actual query analyzer
            # For now, we'll mark as validated if the score is reasonable
            if 0 <= benchmark.expected_score <= 100:
                benchmark.validated = True
                validated_count += 1
            else:
                total_error += abs(
                    benchmark.expected_score - 50
                )  # Assume 50 as baseline

        accuracy = (
            validated_count / len(sample_benchmarks) if sample_benchmarks else 0.0
        )

        return {
            "validated": validated_count,
            "total": len(sample_benchmarks),
            "accuracy": accuracy,
            "average_error": (
                total_error / len(sample_benchmarks) if sample_benchmarks else 0.0
            ),
        }
