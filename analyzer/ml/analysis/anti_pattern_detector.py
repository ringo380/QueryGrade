"""
SQL Anti-Pattern Detection System

This module detects common SQL anti-patterns and provides recommendations
for fixing them to improve query performance and maintainability.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlparse
from sqlparse.sql import Statement, Token, TokenList
from sqlparse.tokens import Keyword, Name


class AntiPatternSeverity(Enum):
    """Severity levels for anti-patterns"""

    CRITICAL = "critical"  # Will cause errors or severe performance issues
    HIGH = "high"  # Significant performance impact
    MEDIUM = "medium"  # Moderate performance impact
    LOW = "low"  # Minor issues or style concerns


class AntiPatternCategory(Enum):
    """Categories of SQL anti-patterns"""

    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    SCALABILITY = "scalability"
    PORTABILITY = "portability"


@dataclass
class AntiPattern:
    """Represents a detected anti-pattern"""

    pattern_id: str
    name: str
    category: AntiPatternCategory
    severity: AntiPatternSeverity
    description: str
    location: str  # Where in the query
    impact: str  # What impact it has
    recommendation: str  # How to fix it
    example_fix: Optional[str] = None
    estimated_performance_impact: float = 0.0  # 0-1 scale
    references: List[str] = field(default_factory=list)


@dataclass
class AntiPatternReport:
    """Complete anti-pattern analysis report"""

    query: str
    anti_patterns: List[AntiPattern]
    overall_score: float  # 0-100, higher is better
    performance_risk: str  # low, medium, high
    maintainability_risk: str
    security_risk: str
    suggested_rewrite: Optional[str] = None
    learning_resources: List[str] = field(default_factory=list)


class AntiPatternDetector:
    """Detects SQL anti-patterns and provides recommendations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for anti-pattern detection"""
        self.patterns = {
            # Performance anti-patterns
            "select_star": re.compile(r"SELECT\s+\*\s+FROM", re.IGNORECASE),
            "implicit_cross_join": re.compile(
                r"FROM\s+(\w+)\s*,\s*(\w+)", re.IGNORECASE
            ),
            "or_in_join": re.compile(r"ON\s+.+\s+OR\s+", re.IGNORECASE),
            "functions_in_where": re.compile(
                r"WHERE\s+.*(YEAR|MONTH|DAY|LOWER|UPPER|SUBSTRING)\s*\([^)]*\w+\.\w+[^)]*\)",
                re.IGNORECASE,
            ),
            "not_equals_join": re.compile(r"ON\s+.+(<>|!=)", re.IGNORECASE),
            "wildcard_prefix": re.compile(r"LIKE\s+['\"]%\w+", re.IGNORECASE),
            "nested_not_in": re.compile(r"NOT\s+IN\s*\(\s*SELECT", re.IGNORECASE),
            "cursor_usage": re.compile(r"\b(CURSOR|FETCH|DEALLOCATE)\b", re.IGNORECASE),
            "row_by_row": re.compile(
                r"WHILE|LOOP|FOR\s+.+\s+IN\s+SELECT", re.IGNORECASE
            ),
            # Correctness anti-patterns
            "null_comparison": re.compile(
                r"WHERE\s+\w+\s*=\s*NULL|WHERE\s+\w+\s*!=\s*NULL", re.IGNORECASE
            ),
            "double_negative": re.compile(
                r"NOT\s+.+\s+NOT\s+(IN|EXISTS|LIKE)", re.IGNORECASE
            ),
            "ambiguous_columns": re.compile(
                r"SELECT\s+(?!.*\.)id|name|status|type(?!\w)", re.IGNORECASE
            ),
            # Security anti-patterns
            "dynamic_sql": re.compile(r"EXEC\s*\(|EXECUTE\s+IMMEDIATE", re.IGNORECASE),
            "string_concat_where": re.compile(
                r"WHERE\s+.+\+|WHERE\s+.+\|\||WHERE\s+.+CONCAT", re.IGNORECASE
            ),
            # Maintainability anti-patterns
            "magic_numbers": re.compile(
                r"WHERE\s+.+\s*=\s*\d{3,}(?!\d)", re.IGNORECASE
            ),
            "no_alias": re.compile(
                r"FROM\s+\w{15,}(?!\s+(?:AS\s+)?\w{1,5}\s)", re.IGNORECASE
            ),
            "inconsistent_case": re.compile(r"(?:select|SELECT|Select)", re.MULTILINE),
            # Scalability anti-patterns
            "offset_pagination": re.compile(r"OFFSET\s+(\d{4,}|\w+)", re.IGNORECASE),
            "union_instead_union_all": re.compile(
                r"\bUNION\b(?!\s+ALL)", re.IGNORECASE
            ),
            "distinct_misuse": re.compile(
                r"SELECT\s+DISTINCT\s+.+\s+FROM\s+.+\s+JOIN", re.IGNORECASE
            ),
        }

    def detect_anti_patterns(self, query: str) -> AntiPatternReport:
        """Detect all anti-patterns in a SQL query"""
        anti_patterns = []

        # Check each anti-pattern
        anti_patterns.extend(self._check_performance_antipatterns(query))
        anti_patterns.extend(self._check_correctness_antipatterns(query))
        anti_patterns.extend(self._check_security_antipatterns(query))
        anti_patterns.extend(self._check_maintainability_antipatterns(query))
        anti_patterns.extend(self._check_scalability_antipatterns(query))

        # Advanced pattern detection using sqlparse
        try:
            parsed = sqlparse.parse(query)[0]
            anti_patterns.extend(self._check_parsed_antipatterns(parsed))
        except Exception as e:
            self.logger.debug(f"Error parsing query: {e}")

        # Calculate scores
        overall_score = self._calculate_overall_score(anti_patterns)
        performance_risk = self._assess_risk_level(
            anti_patterns, AntiPatternCategory.PERFORMANCE
        )
        maintainability_risk = self._assess_risk_level(
            anti_patterns, AntiPatternCategory.MAINTAINABILITY
        )
        security_risk = self._assess_risk_level(
            anti_patterns, AntiPatternCategory.SECURITY
        )

        # Generate suggested rewrite if critical issues found
        suggested_rewrite = None
        if any(ap.severity == AntiPatternSeverity.CRITICAL for ap in anti_patterns):
            suggested_rewrite = self._suggest_rewrite(query, anti_patterns)

        # Add learning resources
        learning_resources = self._get_learning_resources(anti_patterns)

        return AntiPatternReport(
            query=query,
            anti_patterns=anti_patterns,
            overall_score=overall_score,
            performance_risk=performance_risk,
            maintainability_risk=maintainability_risk,
            security_risk=security_risk,
            suggested_rewrite=suggested_rewrite,
            learning_resources=learning_resources,
        )

    def _check_performance_antipatterns(self, query: str) -> List[AntiPattern]:
        """Check for performance-related anti-patterns"""
        anti_patterns = []

        # SELECT * anti-pattern
        if self.patterns["select_star"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="PERF001",
                    name="SELECT * Usage",
                    category=AntiPatternCategory.PERFORMANCE,
                    severity=AntiPatternSeverity.MEDIUM,
                    description="Using SELECT * retrieves all columns",
                    location="SELECT clause",
                    impact="Fetches unnecessary data, increases network traffic and memory usage",
                    recommendation="Specify only required columns explicitly",
                    example_fix="SELECT id, name, email FROM users",
                    estimated_performance_impact=0.3,
                )
            )

        # Implicit cross join
        if self.patterns["implicit_cross_join"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="PERF002",
                    name="Implicit Cross Join",
                    category=AntiPatternCategory.PERFORMANCE,
                    severity=AntiPatternSeverity.HIGH,
                    description="Using comma-separated tables without proper join conditions",
                    location="FROM clause",
                    impact="May result in Cartesian product, exponentially increasing result set",
                    recommendation="Use explicit JOIN syntax with proper conditions",
                    example_fix="FROM table1 JOIN table2 ON table1.id = table2.table1_id",
                    estimated_performance_impact=0.8,
                )
            )

        # OR in JOIN condition
        if self.patterns["or_in_join"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="PERF003",
                    name="OR in JOIN Condition",
                    category=AntiPatternCategory.PERFORMANCE,
                    severity=AntiPatternSeverity.HIGH,
                    description="Using OR in JOIN conditions",
                    location="JOIN clause",
                    impact="Prevents efficient use of indexes, forces nested loop joins",
                    recommendation="Rewrite using UNION or separate queries",
                    example_fix="Use UNION to combine results from separate joins",
                    estimated_performance_impact=0.6,
                )
            )

        # Functions on indexed columns
        if self.patterns["functions_in_where"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="PERF004",
                    name="Functions on Indexed Columns",
                    category=AntiPatternCategory.PERFORMANCE,
                    severity=AntiPatternSeverity.HIGH,
                    description="Applying functions to columns in WHERE clause",
                    location="WHERE clause",
                    impact="Prevents index usage, forces full table scan",
                    recommendation="Rewrite to apply functions to constants instead",
                    example_fix="WHERE date_column >= '2023-01-01' AND date_column < '2024-01-01'",
                    estimated_performance_impact=0.7,
                )
            )

        # Leading wildcard
        if self.patterns["wildcard_prefix"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="PERF005",
                    name="Leading Wildcard in LIKE",
                    category=AntiPatternCategory.PERFORMANCE,
                    severity=AntiPatternSeverity.MEDIUM,
                    description="LIKE pattern starts with wildcard",
                    location="WHERE clause",
                    impact="Cannot use index, requires full table scan",
                    recommendation="Consider full-text search or redesign query",
                    example_fix="Use full-text index or reverse the pattern if possible",
                    estimated_performance_impact=0.5,
                )
            )

        # NOT IN with subquery
        if self.patterns["nested_not_in"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="PERF006",
                    name="NOT IN with Subquery",
                    category=AntiPatternCategory.PERFORMANCE,
                    severity=AntiPatternSeverity.HIGH,
                    description="Using NOT IN with a subquery",
                    location="WHERE clause",
                    impact="Poor performance with NULLs, inefficient execution plan",
                    recommendation="Use NOT EXISTS or LEFT JOIN with NULL check",
                    example_fix="WHERE NOT EXISTS (SELECT 1 FROM other_table WHERE ...)",
                    estimated_performance_impact=0.6,
                )
            )

        # Cursor usage
        if self.patterns["cursor_usage"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="PERF007",
                    name="Cursor Usage",
                    category=AntiPatternCategory.PERFORMANCE,
                    severity=AntiPatternSeverity.CRITICAL,
                    description="Using cursors for row-by-row processing",
                    location="Procedural code",
                    impact="Extremely slow, processes one row at a time",
                    recommendation="Rewrite using set-based operations",
                    example_fix="Use JOIN, UPDATE, or MERGE statements",
                    estimated_performance_impact=0.9,
                )
            )

        # Row-by-row processing
        if self.patterns["row_by_row"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="PERF008",
                    name="Row-by-Row Processing",
                    category=AntiPatternCategory.PERFORMANCE,
                    severity=AntiPatternSeverity.CRITICAL,
                    description="Using loops for row-by-row processing",
                    location="Procedural code",
                    impact="Extremely inefficient compared to set operations",
                    recommendation="Use set-based SQL operations",
                    estimated_performance_impact=0.9,
                )
            )

        return anti_patterns

    def _check_correctness_antipatterns(self, query: str) -> List[AntiPattern]:
        """Check for correctness-related anti-patterns"""
        anti_patterns = []

        # NULL comparison with = or !=
        if self.patterns["null_comparison"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="CORR001",
                    name="Incorrect NULL Comparison",
                    category=AntiPatternCategory.CORRECTNESS,
                    severity=AntiPatternSeverity.CRITICAL,
                    description="Using = or != to compare with NULL",
                    location="WHERE clause",
                    impact="Always returns unknown/false, query won't work as expected",
                    recommendation="Use IS NULL or IS NOT NULL",
                    example_fix="WHERE column IS NULL or WHERE column IS NOT NULL",
                    estimated_performance_impact=0.0,
                )
            )

        # Double negatives
        if self.patterns["double_negative"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="CORR002",
                    name="Double Negative Logic",
                    category=AntiPatternCategory.CORRECTNESS,
                    severity=AntiPatternSeverity.LOW,
                    description="Using double negatives in conditions",
                    location="WHERE clause",
                    impact="Confusing logic, harder to understand and maintain",
                    recommendation="Rewrite with positive logic",
                    example_fix="Use positive conditions where possible",
                    estimated_performance_impact=0.1,
                )
            )

        # Ambiguous column names
        if self.patterns["ambiguous_columns"].search(query) and "JOIN" in query.upper():
            anti_patterns.append(
                AntiPattern(
                    pattern_id="CORR003",
                    name="Ambiguous Column References",
                    category=AntiPatternCategory.CORRECTNESS,
                    severity=AntiPatternSeverity.MEDIUM,
                    description="Column names without table aliases in multi-table query",
                    location="SELECT/WHERE clause",
                    impact="May cause errors or unexpected results with schema changes",
                    recommendation="Always use table aliases for column references",
                    example_fix="SELECT t1.id, t2.name FROM table1 t1 JOIN table2 t2",
                    estimated_performance_impact=0.0,
                )
            )

        return anti_patterns

    def _check_security_antipatterns(self, query: str) -> List[AntiPattern]:
        """Check for security-related anti-patterns"""
        anti_patterns = []

        # Dynamic SQL execution
        if self.patterns["dynamic_sql"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="SEC001",
                    name="Dynamic SQL Execution",
                    category=AntiPatternCategory.SECURITY,
                    severity=AntiPatternSeverity.CRITICAL,
                    description="Using dynamic SQL execution",
                    location="Query construction",
                    impact="Vulnerable to SQL injection attacks",
                    recommendation="Use parameterized queries or stored procedures",
                    example_fix="Use prepared statements with parameters",
                    estimated_performance_impact=0.0,
                    references=["OWASP SQL Injection Prevention"],
                )
            )

        # String concatenation in WHERE
        if self.patterns["string_concat_where"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="SEC002",
                    name="String Concatenation in WHERE",
                    category=AntiPatternCategory.SECURITY,
                    severity=AntiPatternSeverity.HIGH,
                    description="Concatenating strings in WHERE clause",
                    location="WHERE clause",
                    impact="Potential SQL injection vulnerability",
                    recommendation="Use parameterized queries",
                    example_fix="WHERE column = ? (use parameter binding)",
                    estimated_performance_impact=0.0,
                )
            )

        return anti_patterns

    def _check_maintainability_antipatterns(self, query: str) -> List[AntiPattern]:
        """Check for maintainability-related anti-patterns"""
        anti_patterns = []

        # Magic numbers
        if self.patterns["magic_numbers"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="MAINT001",
                    name="Magic Numbers",
                    category=AntiPatternCategory.MAINTAINABILITY,
                    severity=AntiPatternSeverity.LOW,
                    description="Hard-coded numeric values without explanation",
                    location="WHERE/HAVING clause",
                    impact="Unclear business logic, hard to maintain",
                    recommendation="Use named constants or comments to explain values",
                    example_fix="WHERE status = 1 -- Active status",
                    estimated_performance_impact=0.0,
                )
            )

        # Long table names without alias
        if self.patterns["no_alias"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="MAINT002",
                    name="Missing Table Alias",
                    category=AntiPatternCategory.MAINTAINABILITY,
                    severity=AntiPatternSeverity.LOW,
                    description="Long table names without aliases",
                    location="FROM clause",
                    impact="Reduces readability, makes query harder to maintain",
                    recommendation="Use short, meaningful aliases",
                    example_fix="FROM very_long_table_name vlt",
                    estimated_performance_impact=0.0,
                )
            )

        # Inconsistent case
        case_matches = self.patterns["inconsistent_case"].findall(query)
        if len(set(case_matches)) > 1:
            anti_patterns.append(
                AntiPattern(
                    pattern_id="MAINT003",
                    name="Inconsistent Keyword Case",
                    category=AntiPatternCategory.MAINTAINABILITY,
                    severity=AntiPatternSeverity.LOW,
                    description="Mixed case for SQL keywords",
                    location="Throughout query",
                    impact="Reduces readability and consistency",
                    recommendation="Use consistent case (preferably UPPERCASE) for keywords",
                    example_fix="SELECT, FROM, WHERE (all uppercase)",
                    estimated_performance_impact=0.0,
                )
            )

        return anti_patterns

    def _check_scalability_antipatterns(self, query: str) -> List[AntiPattern]:
        """Check for scalability-related anti-patterns"""
        anti_patterns = []

        # Large OFFSET pagination
        offset_match = self.patterns["offset_pagination"].search(query)
        if offset_match:
            try:
                offset_value = offset_match.group(1)
                if offset_value.isdigit() and int(offset_value) > 1000:
                    anti_patterns.append(
                        AntiPattern(
                            pattern_id="SCALE001",
                            name="Large OFFSET Pagination",
                            category=AntiPatternCategory.SCALABILITY,
                            severity=AntiPatternSeverity.HIGH,
                            description="Using large OFFSET values for pagination",
                            location="LIMIT/OFFSET clause",
                            impact="Performance degrades linearly with offset value",
                            recommendation="Use keyset pagination (WHERE id > last_id)",
                            example_fix="WHERE id > ? ORDER BY id LIMIT 100",
                            estimated_performance_impact=0.6,
                        )
                    )
            except Exception:
                pass

        # UNION instead of UNION ALL
        if self.patterns["union_instead_union_all"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="SCALE002",
                    name="Unnecessary UNION",
                    category=AntiPatternCategory.SCALABILITY,
                    severity=AntiPatternSeverity.MEDIUM,
                    description="Using UNION when UNION ALL would suffice",
                    location="Set operation",
                    impact="UNION removes duplicates, requiring sort and additional processing",
                    recommendation="Use UNION ALL if duplicates are acceptable",
                    example_fix="... UNION ALL ...",
                    estimated_performance_impact=0.3,
                )
            )

        # DISTINCT with JOINs
        if self.patterns["distinct_misuse"].search(query):
            anti_patterns.append(
                AntiPattern(
                    pattern_id="SCALE003",
                    name="DISTINCT with JOINs",
                    category=AntiPatternCategory.SCALABILITY,
                    severity=AntiPatternSeverity.MEDIUM,
                    description="Using DISTINCT to eliminate JOIN duplicates",
                    location="SELECT clause",
                    impact="Masks incorrect JOIN conditions, requires additional processing",
                    recommendation="Fix JOIN conditions instead of using DISTINCT",
                    example_fix="Ensure proper JOIN conditions to avoid duplicates",
                    estimated_performance_impact=0.4,
                )
            )

        return anti_patterns

    def _check_parsed_antipatterns(self, parsed: Statement) -> List[AntiPattern]:
        """Check for anti-patterns using parsed SQL structure"""
        anti_patterns = []

        # Check for multiple subqueries that could be CTEs
        subquery_count = str(parsed).upper().count("(SELECT")
        if subquery_count >= 3:
            anti_patterns.append(
                AntiPattern(
                    pattern_id="STRUCT001",
                    name="Excessive Nested Subqueries",
                    category=AntiPatternCategory.MAINTAINABILITY,
                    severity=AntiPatternSeverity.MEDIUM,
                    description="Multiple nested subqueries",
                    location="Throughout query",
                    impact="Hard to read and maintain, potential performance issues",
                    recommendation="Use Common Table Expressions (CTEs) for clarity",
                    example_fix="WITH cte_name AS (SELECT ...) SELECT ... FROM cte_name",
                    estimated_performance_impact=0.2,
                )
            )

        # Check for missing WHERE clause in UPDATE/DELETE
        query_type = str(parsed.get_type()).upper()
        if query_type in ["UPDATE", "DELETE"]:
            if "WHERE" not in str(parsed).upper():
                anti_patterns.append(
                    AntiPattern(
                        pattern_id="DANGER001",
                        name="UPDATE/DELETE without WHERE",
                        category=AntiPatternCategory.CORRECTNESS,
                        severity=AntiPatternSeverity.CRITICAL,
                        description=f"{query_type} statement without WHERE clause",
                        location="Statement",
                        impact="Will affect ALL rows in the table",
                        recommendation="Add WHERE clause to limit affected rows",
                        example_fix=f"{query_type} ... WHERE id = ?",
                        estimated_performance_impact=0.0,
                    )
                )

        return anti_patterns

    def _calculate_overall_score(self, anti_patterns: List[AntiPattern]) -> float:
        """Calculate overall query score (0-100, higher is better)"""
        if not anti_patterns:
            return 100.0

        severity_penalties = {
            AntiPatternSeverity.CRITICAL: 25,
            AntiPatternSeverity.HIGH: 15,
            AntiPatternSeverity.MEDIUM: 8,
            AntiPatternSeverity.LOW: 3,
        }

        total_penalty = sum(
            severity_penalties.get(ap.severity, 0) for ap in anti_patterns
        )

        # Cap penalty at 100
        total_penalty = min(100, total_penalty)

        return max(0, 100 - total_penalty)

    def _assess_risk_level(
        self, anti_patterns: List[AntiPattern], category: AntiPatternCategory
    ) -> str:
        """Assess risk level for a specific category"""
        category_patterns = [ap for ap in anti_patterns if ap.category == category]

        if not category_patterns:
            return "low"

        # Check for critical issues
        if any(ap.severity == AntiPatternSeverity.CRITICAL for ap in category_patterns):
            return "high"

        # Count high severity issues
        high_count = sum(
            1 for ap in category_patterns if ap.severity == AntiPatternSeverity.HIGH
        )
        if high_count >= 2:
            return "high"
        elif high_count == 1:
            return "medium"

        # Count medium severity issues
        medium_count = sum(
            1 for ap in category_patterns if ap.severity == AntiPatternSeverity.MEDIUM
        )
        if medium_count >= 3:
            return "medium"

        return "low"

    def _suggest_rewrite(
        self, query: str, anti_patterns: List[AntiPattern]
    ) -> Optional[str]:
        """Suggest a rewritten query fixing critical issues"""
        rewritten = query

        # Apply fixes for critical anti-patterns
        for ap in anti_patterns:
            if ap.severity == AntiPatternSeverity.CRITICAL:
                if ap.pattern_id == "CORR001":  # NULL comparison
                    rewritten = re.sub(
                        r"(\w+)\s*=\s*NULL",
                        r"\1 IS NULL",
                        rewritten,
                        flags=re.IGNORECASE,
                    )
                    rewritten = re.sub(
                        r"(\w+)\s*!=\s*NULL",
                        r"\1 IS NOT NULL",
                        rewritten,
                        flags=re.IGNORECASE,
                    )

                elif ap.pattern_id == "DANGER001":  # Missing WHERE
                    # Can't auto-fix this safely
                    rewritten = f"{rewritten}\n-- WARNING: Add WHERE clause!"

        return rewritten if rewritten != query else None

    def _get_learning_resources(self, anti_patterns: List[AntiPattern]) -> List[str]:
        """Get relevant learning resources based on detected anti-patterns"""
        resources = []

        categories = set(ap.category for ap in anti_patterns)

        resource_map = {
            AntiPatternCategory.PERFORMANCE: "SQL Performance Tuning Guide",
            AntiPatternCategory.SECURITY: "SQL Injection Prevention Best Practices",
            AntiPatternCategory.CORRECTNESS: "SQL Logic and NULL Handling",
            AntiPatternCategory.MAINTAINABILITY: "SQL Code Style and Best Practices",
            AntiPatternCategory.SCALABILITY: "Scaling SQL Queries for Large Datasets",
        }

        for category in categories:
            if category in resource_map:
                resources.append(resource_map[category])

        return resources


def analyze_query_antipatterns(query: str) -> Dict[str, Any]:
    """
    Convenience function to analyze query for anti-patterns
    """
    detector = AntiPatternDetector()
    report = detector.detect_anti_patterns(query)

    return {
        "overall_score": report.overall_score,
        "anti_patterns": [
            {
                "name": ap.name,
                "severity": ap.severity.value,
                "category": ap.category.value,
                "impact": ap.impact,
                "recommendation": ap.recommendation,
            }
            for ap in report.anti_patterns
        ],
        "risk_assessment": {
            "performance": report.performance_risk,
            "maintainability": report.maintainability_risk,
            "security": report.security_risk,
        },
        "suggested_rewrite": report.suggested_rewrite,
        "learning_resources": report.learning_resources,
    }


if __name__ == "__main__":
    # Example usage with various anti-patterns
    test_query = """
    SELECT * FROM users, orders
    WHERE users.id = orders.user_id
      AND YEAR(orders.created_at) = 2023
      AND status != NULL
      AND name LIKE '%smith'
      AND user_id NOT IN (SELECT user_id FROM blacklist)
    ORDER BY orders.created_at
    OFFSET 10000 LIMIT 100
    """

    print("=== Anti-Pattern Analysis ===\n")

    detector = AntiPatternDetector()
    report = detector.detect_anti_patterns(test_query)

    print(f"Overall Score: {report.overall_score:.1f}/100")
    print(f"Performance Risk: {report.performance_risk}")
    print(f"Maintainability Risk: {report.maintainability_risk}")
    print(f"Security Risk: {report.security_risk}")

    if report.anti_patterns:
        print(f"\nDetected {len(report.anti_patterns)} anti-patterns:")
        for ap in report.anti_patterns:
            print(f"\n[{ap.severity.value.upper()}] {ap.name}")
            print(f"  Category: {ap.category.value}")
            print(f"  Impact: {ap.impact}")
            print(f"  Fix: {ap.recommendation}")
            if ap.example_fix:
                print(f"  Example: {ap.example_fix}")

    if report.suggested_rewrite:
        print("\nSuggested Rewrite:")
        print(report.suggested_rewrite)

    if report.learning_resources:
        print("\nRecommended Learning Resources:")
        for resource in report.learning_resources:
            print(f"  - {resource}")
