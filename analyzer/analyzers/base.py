"""
Base analyzer classes and orchestration logic.

This module provides:
- BaseAnalyzer: Abstract base class for all query analyzers
- QueryGrader: Main orchestrator that coordinates all analyzers
- Convenience functions for backward compatibility
"""

import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sqlparse
from django.core.cache import caches
from sqlparse import tokens
from sqlparse.sql import Statement

from ..exceptions import EmptyQueryError
from ..models import Query, QueryAnalysis
from ..performance import PerformanceMonitor, query_cache

logger = logging.getLogger(__name__)


@dataclass
class AnalysisContext:
    """Context object passed between analyzers containing shared state."""

    parsed: Statement
    sql_text: str
    normalized_sql: str
    database_type: str
    query: Query
    issues: List[Dict] = field(default_factory=list)
    recommendations: List[Dict] = field(default_factory=list)
    performance_notes: List[str] = field(default_factory=list)
    # Optional live schema snapshot. When present, schema-aware analyzers
    # (e.g. IndexingAnalyzer, IndexRecommender) use real table/index/column
    # metadata instead of relying on query text alone.
    live_schema: Optional[object] = (
        None  # analyzer.services.live_schema_context.LiveSchemaContext
    )


class BaseAnalyzer(ABC):
    """
    Abstract base class for all query analyzers.

    Each analyzer focuses on a specific aspect of query analysis and implements
    the Strategy pattern for extensibility.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> None:
        """
        Analyze the query and update the context with findings.

        Args:
            context: AnalysisContext object containing parsed query and results

        Note:
            Implementations should modify context.issues, context.recommendations,
            and context.performance_notes in place.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the analyzer name for logging and debugging."""
        pass


class QueryGrader:
    """
    Main orchestrator for SQL query analysis.

    Coordinates multiple specialized analyzers using the Chain of Responsibility pattern.
    Maintains backward compatibility with the original QueryGrader interface.
    """

    def __init__(self):
        self.analysis_version = "2.0"
        self.analyzers: List[BaseAnalyzer] = []
        self._initialize_analyzers()

    def _initialize_analyzers(self):
        """Initialize and register all analyzers in the analysis pipeline."""
        from .case_statement_analyzer import CaseStatementAnalyzer
        from .database import MySQLAnalyzer, PostgreSQLAnalyzer
        from .groupby_analyzer import GroupByAnalyzer
        from .indexing_analyzer import IndexingAnalyzer
        from .join_analyzer import JoinAnalyzer
        from .orderby_analyzer import OrderByAnalyzer
        from .select_analyzer import SelectAnalyzer
        from .subquery_analyzer import SubqueryAnalyzer
        from .union_analyzer import UnionAnalyzer
        from .where_analyzer import WhereAnalyzer
        from .wildcard_analyzer import WildcardAnalyzer
        from .window_function_analyzer import WindowFunctionAnalyzer

        self.analyzers = [
            # Core clause analyzers (always run)
            SelectAnalyzer(),
            JoinAnalyzer(),
            WhereAnalyzer(),
            OrderByAnalyzer(),
            GroupByAnalyzer(),
            # Performance and optimization analyzers (Phase 1)
            IndexingAnalyzer(),
            SubqueryAnalyzer(),
            # Advanced pattern analyzers (Phase 2)
            UnionAnalyzer(),
            WindowFunctionAnalyzer(),
            CaseStatementAnalyzer(),
            WildcardAnalyzer(),
            # Database-specific analyzers (conditionally run based on database_type)
            MySQLAnalyzer(),
            PostgreSQLAnalyzer(),
            # Future analyzers to be implemented (Phase 3-4):
            # SecurityAnalyzer(),
            # DataTypeAnalyzer(),
            # NullHandlingAnalyzer(),
            # SQLiteAnalyzer(),
            # OracleAnalyzer(),
            # SQLServerAnalyzer(),
        ]

    @PerformanceMonitor.time_function("query_analysis")
    def analyze_query(
        self, sql_text: str, database_type: str = ""
    ) -> Tuple[Query, QueryAnalysis]:
        """
        Analyze a SQL query and return Query and QueryAnalysis objects.

        This is the main entry point for query analysis, maintaining backward
        compatibility with the original API.

        Args:
            sql_text (str): The SQL query to analyze
            database_type (str): The target database type (mysql, postgresql, etc.)

        Returns:
            Tuple[Query, QueryAnalysis]: The created Query and QueryAnalysis objects

        Raises:
            EmptyQueryError: If the query is empty or whitespace
            ValueError: If the query cannot be parsed
        """
        start_time = time.time()

        # Validate input
        if not sql_text or not sql_text.strip():
            raise EmptyQueryError()

        # Clean and normalize query
        normalized_sql = self._normalize_query(sql_text)
        query_hash = self._generate_query_hash(normalized_sql, database_type)

        # Check cache first for performance
        cached_result = query_cache.get_analysis(normalized_sql, database_type)
        if cached_result:
            return cached_result

        # Check if we've already analyzed this exact query
        try:
            existing_query = Query.objects.get(query_hash=query_hash)
            if hasattr(existing_query, "analysis"):
                result = (existing_query, existing_query.analysis)
                query_cache.set_analysis(normalized_sql, result, database_type)
                return result
        except Query.DoesNotExist:
            pass

        # Parse the query and validate
        try:
            parsed_statements = sqlparse.parse(sql_text)
            if not parsed_statements:
                raise ValueError("Unable to parse SQL query")
            parsed = parsed_statements[0]

            # Validate for common typos
            self._validate_query_syntax(sql_text)

        except Exception as e:
            raise ValueError(f"Invalid SQL query: {str(e)}")

        analysis_start = time.time()

        # Create Query object
        query = self._create_query_object(sql_text, normalized_sql, query_hash, parsed)

        # Create analysis context
        context = AnalysisContext(
            parsed=parsed,
            sql_text=sql_text,
            normalized_sql=normalized_sql,
            database_type=database_type,
            query=query,
        )

        # For non-SELECT statements the analyzer pipeline is SELECT-optimized;
        # flag this so results pages can surface a "limited analysis" notice.
        if query.query_type not in ("SELECT", "UNKNOWN"):
            context.performance_notes.append(
                f"Analysis is optimized for SELECT queries. {query.query_type} statements "
                f"receive structural checks only — DML/DDL-specific recommendations are limited."
            )

        # Run all analyzers in sequence
        for analyzer in self.analyzers:
            try:
                analyzer.analyze(context)
            except Exception as e:
                logger.error(f"Analyzer {analyzer.name} failed: {e}")
                # Continue with other analyzers even if one fails

        # Calculate score and grade
        score = self._calculate_score(context)
        grade = self._score_to_grade(score)

        # Calculate actual analysis time
        analysis_time_ms = max(1, int((time.time() - analysis_start) * 1000))

        # Create QueryAnalysis object
        analysis = QueryAnalysis.objects.create(
            query=query,
            grade=grade,
            score=score,
            issues_found=context.issues,
            recommendations=context.recommendations,
            performance_notes=(
                "\n".join(context.performance_notes)
                if context.performance_notes
                else ""
            ),
            analysis_version=self.analysis_version,
            execution_time_ms=analysis_time_ms,
        )

        # Cache the result
        result = (query, analysis)
        query_cache.set_analysis(normalized_sql, result, database_type)

        return result

    def _normalize_query(self, sql_text: str) -> str:
        """Normalize SQL query for consistent analysis."""
        normalized = re.sub(r"\s+", " ", sql_text.strip())
        return normalized.lower()

    def _generate_query_hash(self, normalized_sql: str, database_type: str = "") -> str:
        """Generate MD5 hash for normalized query including database type."""
        hash_input = f"{normalized_sql}|{database_type}"
        return hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()

    def _validate_query_syntax(self, sql_text: str):
        """Validate query for common syntax errors and typos."""
        sql_upper = sql_text.upper().strip()
        malformed_patterns = [
            r"\bSELCT\b",
            r"\bFORM\b(?!\s+WHERE)",
            r"\bWHER\b(?!\s)",
            r"(?<!\w)SELCT(?!\w)",
            r"(?<!\w)WHER(?=\s+\w+\s*=)",
        ]

        for pattern in malformed_patterns:
            if re.search(pattern, sql_upper):
                raise ValueError("SQL contains apparent typos in keywords")

        # Check for basic structure - at least one keyword
        parsed = sqlparse.parse(sql_text)[0]
        has_keyword = any(token.ttype in tokens.Keyword for token in parsed.flatten())
        if not has_keyword:
            raise ValueError("No SQL keywords found in query")

    def _create_query_object(
        self, original_sql: str, normalized_sql: str, query_hash: str, parsed: Statement
    ) -> Query:
        """Create and save Query object with basic metrics."""
        from .utils import (
            count_joins,
            count_subqueries,
            count_tables,
            count_where_conditions,
            get_query_type,
        )

        # Determine query type and calculate metrics
        query_type = get_query_type(parsed)
        table_count = count_tables(parsed)
        join_count = count_joins(parsed)
        where_conditions = count_where_conditions(parsed)
        subquery_count = count_subqueries(parsed)

        # Estimate complexity
        complexity = min(
            100,
            (table_count * 10)
            + (join_count * 15)
            + (where_conditions * 5)
            + (subquery_count * 20),
        )

        query = Query.objects.create(
            sql_text=original_sql,
            query_type=query_type,
            query_hash=query_hash,
            estimated_complexity=complexity,
            table_count=table_count,
            join_count=join_count,
            where_conditions=where_conditions,
            subquery_count=subquery_count,
        )

        return query

    def _calculate_score(self, context: AnalysisContext) -> float:
        """
        Calculate numeric score (0-100) based on analysis results.

        Scoring rubric:
        - Start at 100
        - Deduct points for issues based on severity
        - Add bonus points for positive patterns
        """
        base_score = 100.0

        # Deduct points for issues
        for issue in context.issues:
            severity = issue.get("severity", "medium")
            if severity == "catastrophic":
                base_score -= 50  # Catastrophic issues like Cartesian products
            elif severity == "critical":
                base_score -= 25
            elif severity == "high":
                base_score -= 15
            elif severity == "medium":
                base_score -= 10
            elif severity == "low":
                base_score -= 5

        # Add bonus for good practices (max 5 points)
        positive_recommendations = [
            r for r in context.recommendations if r.get("priority") == "positive"
        ]
        if positive_recommendations:
            base_score += min(5, len(positive_recommendations))

        return max(0.0, min(100.0, base_score))

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


# Convenience functions for backward compatibility
def analyze_query(
    sql_text: str, database_type: str = "", use_ml: Optional[bool] = None
) -> Tuple[Query, QueryAnalysis]:
    """
    Convenience function to analyze a SQL query with optional ML integration.

    Maintains backward compatibility with the original API.

    Args:
        sql_text (str): The SQL query to analyze
        database_type (str): The target database type
        use_ml (bool, optional): Whether to use ML-enhanced grading

    Returns:
        Tuple[Query, QueryAnalysis]: The created Query and QueryAnalysis objects
    """
    from django.conf import settings

    # Determine whether to use ML
    should_use_ml = (
        use_ml if use_ml is not None else getattr(settings, "ML_HYBRID_GRADING", False)
    )

    if should_use_ml and getattr(settings, "ML_ENABLED", False):
        try:
            from ..ml.core.hybrid_grader import HybridQueryGrader

            hybrid_grader = HybridQueryGrader()
            return hybrid_grader.analyze_query(sql_text, database_type, use_ml=True)
        except ImportError:
            # A missing module is a packaging bug, not a runtime condition.
            # Folding it into the warning below let this path import a module
            # that does not exist and fall back on every single call for as
            # long as nobody read the logs.
            raise
        except Exception as e:
            logger.warning(f"ML grading failed, falling back to rule-based: {str(e)}")

    # Use rule-based grading
    grader = QueryGrader()
    return grader.analyze_query(sql_text, database_type)


def grade_single_query(
    sql_text: str,
    database_type: str = "",
    database_version: str = "",
    use_ml: Optional[bool] = None,
) -> QueryAnalysis:
    """
    Convenience function to grade a single SQL query and return just the analysis.

    Args:
        sql_text (str): The SQL query to analyze
        database_type (str): The target database type
        database_version (str): The database version
        use_ml (bool, optional): Whether to use ML-enhanced grading

    Returns:
        QueryAnalysis: The analysis object with grade and recommendations
    """
    query, analysis = analyze_query(sql_text, database_type, use_ml)
    return analysis
