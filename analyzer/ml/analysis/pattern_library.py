"""
Query Pattern Library

This module maintains a comprehensive library of SQL query patterns for
pattern matching, template generation, and best practice recommendations.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class PatternCategory(Enum):
    """Categories of query patterns"""

    BASIC_CRUD = "basic_crud"
    JOIN_PATTERNS = "join_patterns"
    AGGREGATION = "aggregation"
    SUBQUERY = "subquery"
    WINDOW_FUNCTION = "window_function"
    RECURSIVE = "recursive"
    PIVOT = "pivot"
    TEMPORAL = "temporal"
    FULL_TEXT_SEARCH = "full_text_search"
    OPTIMIZATION = "optimization"


class PatternQuality(Enum):
    """Quality rating for patterns"""

    BEST_PRACTICE = "best_practice"
    ACCEPTABLE = "acceptable"
    SUBOPTIMAL = "suboptimal"
    ANTI_PATTERN = "anti_pattern"


@dataclass
class QueryPattern:
    """Represents a query pattern in the library"""

    pattern_id: str
    name: str
    category: PatternCategory
    quality: PatternQuality
    description: str
    pattern_regex: str
    template: str
    example_query: str
    use_cases: List[str]
    performance_notes: str
    alternatives: List[str] = field(default_factory=list)
    database_specific: Dict[str, str] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    complexity_score: float = 0.5
    common_mistakes: List[str] = field(default_factory=list)


class QueryPatternLibrary:
    """Manages a library of SQL query patterns"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patterns: Dict[str, QueryPattern] = {}
        self.pattern_index: Dict[PatternCategory, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)

        # Initialize with common patterns
        self._initialize_default_patterns()

    def _initialize_default_patterns(self):
        """Initialize library with common SQL patterns"""

        # Basic CRUD patterns
        self.add_pattern(
            QueryPattern(
                pattern_id="crud_001",
                name="Simple SELECT with WHERE",
                category=PatternCategory.BASIC_CRUD,
                quality=PatternQuality.BEST_PRACTICE,
                description="Basic filtered selection from a single table",
                pattern_regex=r"SELECT\s+.+\s+FROM\s+\w+\s+WHERE\s+\w+\s*=\s*",
                template="SELECT {columns} FROM {table} WHERE {condition}",
                example_query="SELECT id, name, email FROM users WHERE status = 'active'",
                use_cases=["Retrieving specific records", "Filtering data by criteria"],
                performance_notes="Use indexes on WHERE clause columns for better performance",
                alternatives=["Using IN clause for multiple values"],
                tags={"basic", "filtering", "select"},
            )
        )

        self.add_pattern(
            QueryPattern(
                pattern_id="crud_002",
                name="INSERT with VALUES",
                category=PatternCategory.BASIC_CRUD,
                quality=PatternQuality.BEST_PRACTICE,
                description="Standard single-row insert",
                pattern_regex=r"INSERT\s+INTO\s+\w+\s*\([^)]+\)\s*VALUES\s*\([^)]+\)",
                template="INSERT INTO {table} ({columns}) VALUES ({values})",
                example_query="INSERT INTO users (name, email, created_at) VALUES ('John', 'john@example.com', NOW())",
                use_cases=["Adding new records", "Single row insertion"],
                performance_notes="Use batch inserts for multiple rows",
                alternatives=["INSERT ... SELECT", "Bulk insert"],
                tags={"basic", "insert", "dml"},
            )
        )

        # Join patterns
        self.add_pattern(
            QueryPattern(
                pattern_id="join_001",
                name="INNER JOIN with ON clause",
                category=PatternCategory.JOIN_PATTERNS,
                quality=PatternQuality.BEST_PRACTICE,
                description="Standard inner join between two tables",
                pattern_regex=r"SELECT\s+.+\s+FROM\s+\w+\s+(?:INNER\s+)?JOIN\s+\w+\s+ON\s+",
                template="SELECT {columns} FROM {table1} JOIN {table2} ON {join_condition}",
                example_query="SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id",
                use_cases=[
                    "Combining related data",
                    "Fetching data from multiple tables",
                ],
                performance_notes="Ensure join columns are indexed",
                alternatives=["LEFT JOIN for optional relationships"],
                database_specific={"mysql": "Use STRAIGHT_JOIN for join order hints"},
                tags={"join", "relationship", "multi-table"},
            )
        )

        self.add_pattern(
            QueryPattern(
                pattern_id="join_002",
                name="LEFT JOIN for optional data",
                category=PatternCategory.JOIN_PATTERNS,
                quality=PatternQuality.BEST_PRACTICE,
                description="Left outer join to include all records from left table",
                pattern_regex=r"SELECT\s+.+\s+FROM\s+\w+\s+LEFT\s+(?:OUTER\s+)?JOIN\s+\w+\s+ON\s+",
                template="SELECT {columns} FROM {table1} LEFT JOIN {table2} ON {join_condition}",
                example_query="SELECT u.name, p.phone FROM users u LEFT JOIN phones p ON u.id = p.user_id",
                use_cases=[
                    "Optional relationships",
                    "Including records without matches",
                ],
                performance_notes="LEFT JOINs can be slower than INNER JOINs",
                alternatives=["INNER JOIN if relationship is required"],
                tags={"join", "optional", "outer"},
            )
        )

        # Aggregation patterns
        self.add_pattern(
            QueryPattern(
                pattern_id="agg_001",
                name="GROUP BY with aggregates",
                category=PatternCategory.AGGREGATION,
                quality=PatternQuality.BEST_PRACTICE,
                description="Grouping data with aggregate functions",
                pattern_regex=r"SELECT\s+.+(?:COUNT|SUM|AVG|MAX|MIN)\s*\(.+\).+GROUP\s+BY\s+",
                template="SELECT {group_columns}, {aggregate_functions} FROM {table} GROUP BY {group_columns}",
                example_query="SELECT department, COUNT(*) as count, AVG(salary) as avg_salary FROM employees GROUP BY department",  # noqa: E501
                use_cases=["Summarizing data", "Statistical analysis", "Reporting"],
                performance_notes="Indexes on GROUP BY columns improve performance",
                alternatives=["Window functions for running totals"],
                tags={"aggregation", "grouping", "statistics"},
            )
        )

        self.add_pattern(
            QueryPattern(
                pattern_id="agg_002",
                name="HAVING clause filtering",
                category=PatternCategory.AGGREGATION,
                quality=PatternQuality.BEST_PRACTICE,
                description="Filtering grouped results using HAVING",
                pattern_regex=r"GROUP\s+BY\s+.+\s+HAVING\s+",
                template="SELECT {columns} FROM {table} GROUP BY {group_columns} HAVING {aggregate_condition}",
                example_query="SELECT department, COUNT(*) as cnt FROM employees GROUP BY department HAVING COUNT(*) > 10",  # noqa: E501
                use_cases=[
                    "Filtering aggregated results",
                    "Finding groups meeting criteria",
                ],
                performance_notes="HAVING is applied after grouping, use WHERE for pre-grouping filters",
                alternatives=["Subquery with WHERE clause"],
                tags={"aggregation", "filtering", "having"},
            )
        )

        # Subquery patterns
        self.add_pattern(
            QueryPattern(
                pattern_id="sub_001",
                name="Correlated subquery in WHERE",
                category=PatternCategory.SUBQUERY,
                quality=PatternQuality.ACCEPTABLE,
                description="Subquery that references outer query",
                pattern_regex=r"WHERE\s+.+\s+(?:IN|EXISTS|=|>|<)\s*\(\s*SELECT\s+",
                template="SELECT {columns} FROM {table1} WHERE {column} IN (SELECT {column2} FROM {table2} WHERE {correlation})",  # noqa: E501
                example_query="SELECT name FROM employees e WHERE salary > (SELECT AVG(salary) FROM employees WHERE department = e.department)",  # noqa: E501
                use_cases=["Complex filtering", "Comparing against aggregates"],
                performance_notes="Can be slow for large datasets, consider JOINs",
                alternatives=["JOIN with derived table", "CTE"],
                common_mistakes=["Missing correlation", "N+1 query problem"],
                tags={"subquery", "correlated", "nested"},
            )
        )

        self.add_pattern(
            QueryPattern(
                pattern_id="sub_002",
                name="Scalar subquery in SELECT",
                category=PatternCategory.SUBQUERY,
                quality=PatternQuality.ACCEPTABLE,
                description="Subquery returning single value in SELECT clause",
                pattern_regex=r"SELECT\s+.+\(\s*SELECT\s+.+\)\s+",
                template="SELECT {columns}, (SELECT {aggregate} FROM {table2} WHERE {condition}) as {alias}",
                example_query="SELECT name, (SELECT COUNT(*) FROM orders WHERE customer_id = c.id) as order_count FROM customers c",  # noqa: E501
                use_cases=["Adding calculated columns", "Inline aggregations"],
                performance_notes="Executed once per row, can be inefficient",
                alternatives=["LEFT JOIN with GROUP BY"],
                tags={"subquery", "scalar", "select"},
            )
        )

        # Window function patterns
        self.add_pattern(
            QueryPattern(
                pattern_id="win_001",
                name="ROW_NUMBER for ranking",
                category=PatternCategory.WINDOW_FUNCTION,
                quality=PatternQuality.BEST_PRACTICE,
                description="Assigning row numbers within partitions",
                pattern_regex=r"ROW_NUMBER\s*\(\s*\)\s*OVER\s*\(",
                template="SELECT {columns}, ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY {order}) as rn",
                example_query="SELECT name, department, salary, ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank FROM employees",  # noqa: E501
                use_cases=["Ranking within groups", "Pagination", "Deduplication"],
                performance_notes="Efficient for ranking operations",
                database_specific={
                    "mysql": "Available from MySQL 8.0+",
                    "postgresql": "Fully supported",
                    "sqlite": "Available from SQLite 3.25+",
                },
                tags={"window", "ranking", "analytical"},
            )
        )

        self.add_pattern(
            QueryPattern(
                pattern_id="win_002",
                name="Running total with SUM OVER",
                category=PatternCategory.WINDOW_FUNCTION,
                quality=PatternQuality.BEST_PRACTICE,
                description="Calculating running totals using window functions",
                pattern_regex=r"SUM\s*\([^)]+\)\s*OVER\s*\(",
                template="SELECT {columns}, SUM({column}) OVER (ORDER BY {order_column} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total",  # noqa: E501
                example_query="SELECT date, amount, SUM(amount) OVER (ORDER BY date) as running_total FROM transactions",  # noqa: E501
                use_cases=["Running totals", "Cumulative calculations"],
                performance_notes="More efficient than self-joins for running calculations",
                alternatives=["Recursive CTE", "Application-level calculation"],
                tags={"window", "aggregation", "running"},
            )
        )

        # Recursive patterns
        self.add_pattern(
            QueryPattern(
                pattern_id="rec_001",
                name="Recursive CTE for hierarchies",
                category=PatternCategory.RECURSIVE,
                quality=PatternQuality.BEST_PRACTICE,
                description="Traversing hierarchical data with recursive CTE",
                pattern_regex=r"WITH\s+RECURSIVE\s+\w+\s+",
                template="WITH RECURSIVE {cte_name} AS (SELECT {base_case} UNION ALL SELECT {recursive_case})",
                example_query="""WITH RECURSIVE hierarchy AS (
                SELECT id, name, parent_id, 0 as level FROM categories WHERE parent_id IS NULL
                UNION ALL
                SELECT c.id, c.name, c.parent_id, h.level + 1
                FROM categories c JOIN hierarchy h ON c.parent_id = h.id
            ) SELECT * FROM hierarchy""",
                use_cases=["Tree structures", "Organizational charts", "Path finding"],
                performance_notes="Can be resource-intensive for deep hierarchies",
                database_specific={
                    "mysql": "Supported from MySQL 8.0+",
                    "postgresql": "Fully supported",
                    "sqlite": "Supported",
                },
                tags={"recursive", "cte", "hierarchy"},
            )
        )

        # Optimization patterns
        self.add_pattern(
            QueryPattern(
                pattern_id="opt_001",
                name="EXISTS instead of IN",
                category=PatternCategory.OPTIMIZATION,
                quality=PatternQuality.BEST_PRACTICE,
                description="Using EXISTS for better performance than IN with subquery",
                pattern_regex=r"WHERE\s+EXISTS\s*\(",
                template="SELECT {columns} FROM {table1} WHERE EXISTS (SELECT 1 FROM {table2} WHERE {condition})",
                example_query="SELECT * FROM orders o WHERE EXISTS (SELECT 1 FROM customers c WHERE c.id = o.customer_id AND c.status = 'active')",  # noqa: E501
                use_cases=["Checking existence", "Optimized filtering"],
                performance_notes="EXISTS stops at first match, more efficient than IN for large datasets",
                alternatives=["IN clause for small lists", "JOIN"],
                tags={"optimization", "exists", "performance"},
            )
        )

        self.add_pattern(
            QueryPattern(
                pattern_id="opt_002",
                name="LIMIT with ORDER BY",
                category=PatternCategory.OPTIMIZATION,
                quality=PatternQuality.BEST_PRACTICE,
                description="Limiting results with proper ordering",
                pattern_regex=r"ORDER\s+BY\s+.+\s+LIMIT\s+\d+",
                template="SELECT {columns} FROM {table} ORDER BY {order_column} LIMIT {n}",
                example_query="SELECT * FROM products ORDER BY created_at DESC LIMIT 10",
                use_cases=["Pagination", "Top-N queries", "Recent records"],
                performance_notes="Index on ORDER BY column crucial for performance",
                database_specific={
                    "mysql": "LIMIT offset, count",
                    "postgresql": "LIMIT count OFFSET offset",
                    "mssql": "TOP n or OFFSET FETCH",
                },
                tags={"optimization", "pagination", "limiting"},
            )
        )

        # Temporal patterns
        self.add_pattern(
            QueryPattern(
                pattern_id="temp_001",
                name="Date range filtering",
                category=PatternCategory.TEMPORAL,
                quality=PatternQuality.BEST_PRACTICE,
                description="Filtering records within a date range",
                pattern_regex=r"WHERE\s+.+\s+BETWEEN\s+.+\s+AND\s+",
                template="SELECT {columns} FROM {table} WHERE {date_column} BETWEEN {start_date} AND {end_date}",
                example_query="SELECT * FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31'",
                use_cases=["Time-based reporting", "Historical data analysis"],
                performance_notes="Index on date column essential",
                alternatives=["Using >= and <= operators"],
                tags={"temporal", "date", "filtering"},
            )
        )

    def add_pattern(self, pattern: QueryPattern) -> bool:
        """Add a new pattern to the library"""
        try:
            self.patterns[pattern.pattern_id] = pattern

            # Update indexes
            self.pattern_index[pattern.category].append(pattern.pattern_id)

            for tag in pattern.tags:
                self.tag_index[tag].append(pattern.pattern_id)

            self.logger.info(f"Added pattern: {pattern.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding pattern: {e}")
            return False

    def find_matching_patterns(self, query: str) -> List[QueryPattern]:
        """Find patterns that match a given query"""
        matching_patterns = []

        for pattern in self.patterns.values():
            try:
                if re.search(pattern.pattern_regex, query, re.IGNORECASE):
                    matching_patterns.append(pattern)
            except Exception as e:
                self.logger.debug(f"Error matching pattern {pattern.pattern_id}: {e}")

        # Sort by quality (best practices first)
        quality_order = {
            PatternQuality.BEST_PRACTICE: 0,
            PatternQuality.ACCEPTABLE: 1,
            PatternQuality.SUBOPTIMAL: 2,
            PatternQuality.ANTI_PATTERN: 3,
        }

        matching_patterns.sort(key=lambda p: quality_order.get(p.quality, 99))

        return matching_patterns

    def get_patterns_by_category(self, category: PatternCategory) -> List[QueryPattern]:
        """Get all patterns in a specific category"""
        pattern_ids = self.pattern_index.get(category, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]

    def get_patterns_by_tag(self, tag: str) -> List[QueryPattern]:
        """Get all patterns with a specific tag"""
        pattern_ids = self.tag_index.get(tag, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]

    def suggest_improvements(self, query: str) -> List[Dict[str, Any]]:
        """Suggest improvements for a given query"""
        suggestions = []
        matching_patterns = self.find_matching_patterns(query)

        for pattern in matching_patterns:
            if pattern.quality in [
                PatternQuality.SUBOPTIMAL,
                PatternQuality.ANTI_PATTERN,
            ]:
                # Find better alternatives
                for alt_id in pattern.alternatives:
                    if alt_id in self.patterns:
                        alt_pattern = self.patterns[alt_id]
                        suggestions.append(
                            {
                                "current_pattern": pattern.name,
                                "issue": f"Using {pattern.quality.value} pattern",
                                "suggestion": alt_pattern.name,
                                "example": alt_pattern.example_query,
                                "benefit": alt_pattern.performance_notes,
                            }
                        )

            # Check for common mistakes
            for mistake in pattern.common_mistakes:
                if self._check_for_mistake(query, mistake):
                    suggestions.append(
                        {
                            "current_pattern": pattern.name,
                            "issue": mistake,
                            "suggestion": f"Review {pattern.name} best practices",
                            "example": pattern.example_query,
                            "benefit": pattern.performance_notes,
                        }
                    )

        return suggestions

    def _check_for_mistake(self, query: str, mistake: str) -> bool:
        """Check if a query contains a common mistake"""
        mistake_patterns = {
            "Missing correlation": r"WHERE\s+.+\s+IN\s*\(\s*SELECT\s+.+\)(?!.*WHERE)",
            "N+1 query problem": r"SELECT.*FROM.*WHERE.*IN\s*\([^)]{1000,}\)",  # Long IN list
            "SELECT *": r"SELECT\s+\*\s+FROM",
            "Missing index hint": r"WHERE\s+LOWER\s*\(|WHERE\s+UPPER\s*\(",
            "Implicit conversion": r"WHERE\s+\w+\s*=\s*'\d+'",  # String comparison with number
        }

        pattern = mistake_patterns.get(mistake)
        if pattern:
            return bool(re.search(pattern, query, re.IGNORECASE))

        return False

    def generate_template(self, category: PatternCategory, **kwargs) -> Optional[str]:
        """Generate a query template based on category and parameters"""
        patterns = self.get_patterns_by_category(category)

        if not patterns:
            return None

        # Select best practice pattern
        best_pattern = next(
            (p for p in patterns if p.quality == PatternQuality.BEST_PRACTICE),
            patterns[0],
        )

        # Format template with provided parameters
        try:
            return best_pattern.template.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing template parameter: {e}")
            return best_pattern.template

    def export_library(self, output_path: str, format: str = "json"):
        """Export the pattern library"""
        try:
            if format == "json":
                export_data = {
                    "patterns": [
                        {
                            "pattern_id": p.pattern_id,
                            "name": p.name,
                            "category": p.category.value,
                            "quality": p.quality.value,
                            "description": p.description,
                            "template": p.template,
                            "example": p.example_query,
                            "tags": list(p.tags),
                        }
                        for p in self.patterns.values()
                    ]
                }

                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2)

            self.logger.info(f"Exported {len(self.patterns)} patterns to {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting library: {e}")

    def import_patterns(self, input_path: str, format: str = "json"):
        """Import patterns from a file"""
        try:
            if format == "json":
                with open(input_path, "r") as f:
                    data = json.load(f)

                for pattern_data in data.get("patterns", []):
                    pattern = QueryPattern(
                        pattern_id=pattern_data["pattern_id"],
                        name=pattern_data["name"],
                        category=PatternCategory(pattern_data["category"]),
                        quality=PatternQuality(pattern_data["quality"]),
                        description=pattern_data["description"],
                        pattern_regex=pattern_data.get("pattern_regex", ""),
                        template=pattern_data["template"],
                        example_query=pattern_data["example"],
                        use_cases=pattern_data.get("use_cases", []),
                        performance_notes=pattern_data.get("performance_notes", ""),
                        tags=set(pattern_data.get("tags", [])),
                    )
                    self.add_pattern(pattern)

            self.logger.info(f"Imported patterns from {input_path}")

        except Exception as e:
            self.logger.error(f"Error importing patterns: {e}")


def analyze_query_patterns(query: str) -> Dict[str, Any]:
    """
    Convenience function to analyze query patterns
    """
    library = QueryPatternLibrary()

    matching_patterns = library.find_matching_patterns(query)
    suggestions = library.suggest_improvements(query)

    return {
        "matched_patterns": [
            {
                "name": p.name,
                "category": p.category.value,
                "quality": p.quality.value,
                "description": p.description,
            }
            for p in matching_patterns
        ],
        "improvements": suggestions,
        "pattern_score": _calculate_pattern_score(matching_patterns),
    }


def _calculate_pattern_score(patterns: List[QueryPattern]) -> float:
    """Calculate overall score based on matched patterns"""
    if not patterns:
        return 0.5

    quality_scores = {
        PatternQuality.BEST_PRACTICE: 1.0,
        PatternQuality.ACCEPTABLE: 0.7,
        PatternQuality.SUBOPTIMAL: 0.4,
        PatternQuality.ANTI_PATTERN: 0.1,
    }

    scores = [quality_scores.get(p.quality, 0.5) for p in patterns]
    return sum(scores) / len(scores)


if __name__ == "__main__":
    # Example usage
    library = QueryPatternLibrary()

    # Test query
    test_query = """
    SELECT c.name, COUNT(o.id) as order_count
    FROM customers c
    LEFT JOIN orders o ON c.id = o.customer_id
    WHERE c.created_at >= '2023-01-01'
    GROUP BY c.id, c.name
    HAVING COUNT(o.id) > 5
    ORDER BY order_count DESC
    LIMIT 10
    """

    print("=== Query Pattern Analysis ===")

    # Find matching patterns
    matches = library.find_matching_patterns(test_query)
    print(f"\nMatched {len(matches)} patterns:")
    for pattern in matches:
        print(f"  - {pattern.name} ({pattern.quality.value})")

    # Get suggestions
    suggestions = library.suggest_improvements(test_query)
    if suggestions:
        print("\nImprovement Suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion['issue']}: {suggestion['suggestion']}")

    # Generate template
    template = library.generate_template(
        PatternCategory.JOIN_PATTERNS,
        columns="*",
        table1="users",
        table2="posts",
        join_condition="users.id = posts.user_id",
    )
    print(f"\nGenerated Template: {template}")
