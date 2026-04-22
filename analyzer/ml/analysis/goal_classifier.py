"""
Query Goal Classifier

Classifies SQL queries into goal categories:
- Reporting: Aggregations, summaries, formatted data retrieval
- Transactional: CRUD operations with simple predicates
- Analytical: Complex joins, aggregations, window functions, temporal analysis
- Maintenance: Schema changes, administrative tasks

Provides confidence scores and secondary goal identification.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class QueryGoal(Enum):
    """Query goal classifications"""

    REPORTING = "reporting"  # Aggregations, summaries, formatting
    TRANSACTIONAL = "transactional"  # CRUD operations
    ANALYTICAL = "analytical"  # Complex analysis, multi-table
    MAINTENANCE = "maintenance"  # Schema changes, admin
    EXPLORATORY = "exploratory"  # Ad-hoc data discovery


@dataclass
class GoalAnalysis:
    """Complete goal analysis for a query"""

    primary_goal: QueryGoal = QueryGoal.TRANSACTIONAL
    primary_goal_confidence: float = 0.0

    goal_scores: Dict[str, float] = field(default_factory=dict)  # goal -> confidence
    secondary_goals: List[QueryGoal] = field(default_factory=list)

    # Goal characteristics
    is_read_only: bool = True
    has_data_modification: bool = False
    has_schema_change: bool = False

    # Optimization recommendations specific to goal
    optimization_recommendations: List[str] = field(default_factory=list)

    # Goal-specific metrics
    reporting_characteristics: Dict[str, any] = field(default_factory=dict)
    transactional_characteristics: Dict[str, any] = field(default_factory=dict)
    analytical_characteristics: Dict[str, any] = field(default_factory=dict)
    maintenance_characteristics: Dict[str, any] = field(default_factory=dict)


class QueryGoalClassifier:
    """Classifies query goals and provides recommendations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for goal classification"""
        # Data modification
        self.insert_pattern = re.compile(r"^\s*INSERT\b", re.IGNORECASE | re.MULTILINE)
        self.update_pattern = re.compile(r"^\s*UPDATE\b", re.IGNORECASE | re.MULTILINE)
        self.delete_pattern = re.compile(r"^\s*DELETE\b", re.IGNORECASE | re.MULTILINE)
        self.merge_pattern = re.compile(r"^\s*MERGE\b", re.IGNORECASE | re.MULTILINE)
        self.upsert_pattern = re.compile(
            r"ON\s+DUPLICATE\s+KEY\s+UPDATE", re.IGNORECASE
        )

        # Schema operations
        self.create_pattern = re.compile(r"^\s*CREATE\b", re.IGNORECASE | re.MULTILINE)
        self.alter_pattern = re.compile(r"^\s*ALTER\b", re.IGNORECASE | re.MULTILINE)
        self.drop_pattern = re.compile(r"^\s*DROP\b", re.IGNORECASE | re.MULTILINE)
        self.truncate_pattern = re.compile(
            r"^\s*TRUNCATE\b", re.IGNORECASE | re.MULTILINE
        )
        self.grant_pattern = re.compile(
            r"^\s*(GRANT|REVOKE)\b", re.IGNORECASE | re.MULTILINE
        )

        # Reporting indicators
        self.aggregation_functions = re.compile(
            r"\b(SUM|COUNT|AVG|MIN|MAX|STDDEV|VARIANCE|PERCENTILE|MEDIAN)\s*\(",
            re.IGNORECASE,
        )
        self.group_by_pattern = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
        self.having_pattern = re.compile(r"\bHAVING\b", re.IGNORECASE)
        self.rollup_pattern = re.compile(
            r"\b(ROLLUP|CUBE|GROUPING\s+SETS)\b", re.IGNORECASE
        )
        self.case_pattern = re.compile(r"\bCASE\b", re.IGNORECASE)

        # Analytical indicators
        self.window_functions = re.compile(
            r"\b(ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|FIRST_VALUE|LAST_VALUE|NTILE|SUM|AVG|COUNT|MAX|MIN)\s*\(\s*.*?\)\s+OVER\s*\(",  # noqa: E501
            re.IGNORECASE,
        )
        self.cte_pattern = re.compile(r"\bWITH\b", re.IGNORECASE)
        self.recursive_cte_pattern = re.compile(r"\bWITH\s+RECURSIVE\b", re.IGNORECASE)
        self.complex_join_pattern = re.compile(
            r"\b(LEFT|RIGHT|FULL|CROSS)\s+JOIN\b", re.IGNORECASE
        )
        self.union_pattern = re.compile(
            r"\b(UNION|INTERSECT|EXCEPT|MINUS)\b", re.IGNORECASE
        )
        self.temporal_functions = re.compile(
            r"\b(NOW|CURRENT_TIMESTAMP|DATE|DATEADD|DATEDIFF|EXTRACT|YEAR|MONTH|DAY)\b",
            re.IGNORECASE,
        )

        # Transactional indicators
        self.simple_where_pattern = re.compile(
            r'\bWHERE\s+\w+\s*=\s*[\'"]?\w+[\'"]?\b', re.IGNORECASE
        )
        self.limit_pattern = re.compile(r"\b(LIMIT|TOP)\b", re.IGNORECASE)
        self.lock_pattern = re.compile(
            r"\b(FOR\s+UPDATE|LOCK|PRAGMA|NOLOCK|XLOCK|SHARED)\b", re.IGNORECASE
        )
        self.pragma_pattern = re.compile(r"\bPRAGMA\b", re.IGNORECASE)

        # Exploratory indicators
        self.select_star_pattern = re.compile(r"\bSELECT\s+\*\b", re.IGNORECASE)
        self.show_pattern = re.compile(
            r"\b(SHOW|DESCRIBE|DESC|EXPLAIN|ANALYZE)\b", re.IGNORECASE
        )
        self.order_by_pattern = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)

    def classify_goal(self, query: str) -> GoalAnalysis:
        """Classify the goal of a SQL query"""
        try:
            analysis = GoalAnalysis()

            # Determine data modification type
            analysis.has_data_modification = bool(
                self.insert_pattern.search(query)
                or self.update_pattern.search(query)
                or self.delete_pattern.search(query)
                or self.merge_pattern.search(query)
            )

            analysis.has_schema_change = bool(
                self.create_pattern.search(query)
                or self.alter_pattern.search(query)
                or self.drop_pattern.search(query)
                or self.truncate_pattern.search(query)
            )

            analysis.is_read_only = not (
                analysis.has_data_modification or analysis.has_schema_change
            )

            # Calculate goal scores
            goal_scores = {
                QueryGoal.REPORTING: self._score_reporting_goal(query),
                QueryGoal.TRANSACTIONAL: self._score_transactional_goal(query),
                QueryGoal.ANALYTICAL: self._score_analytical_goal(query),
                QueryGoal.MAINTENANCE: self._score_maintenance_goal(query),
                QueryGoal.EXPLORATORY: self._score_exploratory_goal(query),
            }

            # Normalize scores
            total_score = sum(goal_scores.values())
            if total_score > 0:
                goal_scores = {
                    goal: score / total_score for goal, score in goal_scores.items()
                }

            # Set goal scores as strings in analysis
            analysis.goal_scores = {
                goal.value: score for goal, score in goal_scores.items()
            }

            # Determine primary goal
            if max(goal_scores.values()) == 0:
                analysis.primary_goal = QueryGoal.TRANSACTIONAL
                analysis.primary_goal_confidence = 0.3
            else:
                analysis.primary_goal = max(goal_scores, key=goal_scores.get)
                analysis.primary_goal_confidence = goal_scores[analysis.primary_goal]

                # Identify secondary goals (>20% of primary goal confidence)
                primary_score = goal_scores[analysis.primary_goal]
                for goal, score in goal_scores.items():
                    if goal != analysis.primary_goal and score >= primary_score * 0.2:
                        analysis.secondary_goals.append(goal)

            # Extract goal-specific characteristics
            self._extract_reporting_characteristics(query, analysis)
            self._extract_transactional_characteristics(query, analysis)
            self._extract_analytical_characteristics(query, analysis)
            self._extract_maintenance_characteristics(query, analysis)

            # Generate recommendations
            analysis.optimization_recommendations = self._generate_recommendations(
                analysis
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Error classifying query goal: {e}")
            return GoalAnalysis()

    def _score_reporting_goal(self, query: str) -> float:
        """Score likelihood that query is reporting-focused"""
        score = 0.0

        # Aggregations are strong indicator
        agg_count = len(self.aggregation_functions.findall(query))
        if agg_count > 0:
            score += min(2.0, agg_count * 0.5)

        # GROUP BY + aggregations = strong reporting indicator
        if self.group_by_pattern.search(query) and agg_count > 0:
            score += 2.0

        # HAVING clause for filtering aggregations
        if self.having_pattern.search(query):
            score += 1.0

        # ROLLUP/CUBE for multi-level reporting
        if self.rollup_pattern.search(query):
            score += 1.5

        # CASE for conditional reporting
        case_count = len(self.case_pattern.findall(query))
        if case_count > 0:
            score += case_count * 0.3

        # Multiple columns selected (formatting indicator)
        select_clause = query.split("FROM")[0] if "FROM" in query.upper() else query
        column_count = len(re.findall(r",", select_clause))
        if column_count > 5:
            score += 0.5

        # SELECT * is not reporting (lack of specific formatting)
        if self.select_star_pattern.search(query):
            score -= 0.5

        return max(0.0, score)

    def _score_transactional_goal(self, query: str) -> float:
        """Score likelihood that query is transactional"""
        score = 0.0

        # Data modification operations are strong indicators
        if self.insert_pattern.search(query):
            score += 2.0
        if self.update_pattern.search(query):
            score += 2.0
        if self.delete_pattern.search(query):
            score += 2.0
        if self.merge_pattern.search(query):
            score += 1.5
        if self.upsert_pattern.search(query):
            score += 1.5

        # Simple WHERE clauses indicate transactional (not analytical)
        where_matches = len(self.simple_where_pattern.findall(query))
        if where_matches > 0:
            score += where_matches * 0.3

        # LIMIT/TOP for single row operations
        if self.limit_pattern.search(query) and (
            "LIMIT 1" in query.upper() or "TOP 1" in query.upper()
        ):
            score += 0.5

        # Locking indicators
        if self.lock_pattern.search(query):
            score += 0.5

        # Few JOINs (<2) suggests transactional
        join_count = len(re.findall(r"\bJOIN\b", query, re.IGNORECASE))
        if 0 <= join_count < 2:
            score += 0.3

        # No aggregations = transactional
        if not self.aggregation_functions.search(query):
            score += 0.2

        # No GROUP BY
        if not self.group_by_pattern.search(query):
            score += 0.2

        return score

    def _score_analytical_goal(self, query: str) -> float:
        """Score likelihood that query is analytical"""
        score = 0.0

        # Window functions are strong analytical indicators
        if self.window_functions.search(query):
            score += 2.0

        # CTEs for complex query composition
        if self.cte_pattern.search(query):
            score += 1.0

        # Recursive CTEs for hierarchical analysis
        if self.recursive_cte_pattern.search(query):
            score += 2.0

        # Multiple complex JOINs
        complex_join_count = len(self.complex_join_pattern.findall(query))
        if complex_join_count > 0:
            score += min(1.5, complex_join_count * 0.4)

        # Set operations for complex analysis
        if self.union_pattern.search(query):
            score += 1.0

        # Aggregations with GROUP BY
        agg_count = len(self.aggregation_functions.findall(query))
        if agg_count > 0 and self.group_by_pattern.search(query):
            score += 1.0

        # Temporal analysis
        temporal_count = len(self.temporal_functions.findall(query))
        if temporal_count > 0:
            score += min(1.0, temporal_count * 0.3)

        # Multiple tables (joins > 1)
        join_count = len(re.findall(r"\bJOIN\b", query, re.IGNORECASE))
        if join_count >= 2:
            score += 0.5

        # HAVING for complex filtering
        if self.having_pattern.search(query):
            score += 0.3

        return score

    def _score_maintenance_goal(self, query: str) -> float:
        """Score likelihood that query is maintenance/administrative"""
        score = 0.0

        # Schema operations are strong maintenance indicators
        if self.create_pattern.search(query):
            score += 3.0
        if self.alter_pattern.search(query):
            score += 3.0
        if self.drop_pattern.search(query):
            score += 3.0
        if self.truncate_pattern.search(query):
            score += 2.0

        # Permission management
        if self.grant_pattern.search(query):
            score += 2.0

        # Pragma directives
        if self.pragma_pattern.search(query):
            score += 1.5

        # ANALYZE, VACUUM-like operations
        if re.search(
            r"\b(ANALYZE|VACUUM|OPTIMIZE|REINDEX|EXPLAIN)\b", query, re.IGNORECASE
        ):
            score += 1.5

        # Index operations
        if re.search(
            r"\b(CREATE|DROP|ALTER)\s+(INDEX|TABLE|VIEW|DATABASE)\b",
            query,
            re.IGNORECASE,
        ):
            score += 1.0

        return score

    def _score_exploratory_goal(self, query: str) -> float:
        """Score likelihood that query is exploratory/discovery"""
        score = 0.0

        # SELECT * for data discovery
        if self.select_star_pattern.search(query):
            score += 1.0

        # SHOW/DESCRIBE/EXPLAIN commands
        if self.show_pattern.search(query):
            score += 2.0

        # No WHERE clause (browsing)
        if "WHERE" not in query.upper():
            score += 0.5

        # No aggregations (looking at raw data)
        if not self.aggregation_functions.search(query):
            score += 0.2

        # LIMIT without GROUP BY (sampling)
        if self.limit_pattern.search(query) and not self.group_by_pattern.search(query):
            score += 0.3

        # ORDER BY without aggregation (sorting for discovery)
        if self.order_by_pattern.search(
            query
        ) and not self.aggregation_functions.search(query):
            score += 0.2

        return score

    def _extract_reporting_characteristics(self, query: str, analysis: GoalAnalysis):
        """Extract reporting-specific characteristics"""
        characteristics = {}

        agg_count = len(self.aggregation_functions.findall(query))
        characteristics["aggregation_count"] = agg_count

        if self.group_by_pattern.search(query):
            group_by_cols = (
                len(re.findall(r",", query.split("GROUP BY")[1].split("HAVING")[0])) + 1
                if "GROUP BY" in query.upper()
                else 0
            )
            characteristics["group_by_columns"] = group_by_cols

        characteristics["has_having_clause"] = bool(self.having_pattern.search(query))
        characteristics["has_rollup_cube"] = bool(self.rollup_pattern.search(query))

        case_count = len(self.case_pattern.findall(query))
        characteristics["case_statements"] = case_count

        analysis.reporting_characteristics = characteristics

    def _extract_transactional_characteristics(
        self, query: str, analysis: GoalAnalysis
    ):
        """Extract transactional-specific characteristics"""
        characteristics = {}

        characteristics["is_insert"] = bool(self.insert_pattern.search(query))
        characteristics["is_update"] = bool(self.update_pattern.search(query))
        characteristics["is_delete"] = bool(self.delete_pattern.search(query))
        characteristics["is_merge"] = bool(self.merge_pattern.search(query))
        characteristics["is_upsert"] = bool(self.upsert_pattern.search(query))

        # Check for single row operations
        if self.limit_pattern.search(query):
            limit_match = re.search(r"\b(LIMIT|TOP)\s+(\d+)", query, re.IGNORECASE)
            if limit_match:
                characteristics["limit_rows"] = int(limit_match.group(2))

        characteristics["has_lock"] = bool(self.lock_pattern.search(query))

        analysis.transactional_characteristics = characteristics

    def _extract_analytical_characteristics(self, query: str, analysis: GoalAnalysis):
        """Extract analytical-specific characteristics"""
        characteristics = {}

        characteristics["has_window_functions"] = bool(
            self.window_functions.search(query)
        )
        characteristics["has_cte"] = bool(self.cte_pattern.search(query))
        characteristics["has_recursive_cte"] = bool(
            self.recursive_cte_pattern.search(query)
        )

        complex_join_count = len(self.complex_join_pattern.findall(query))
        characteristics["complex_join_count"] = complex_join_count

        characteristics["has_set_operations"] = bool(self.union_pattern.search(query))

        temporal_count = len(self.temporal_functions.findall(query))
        characteristics["temporal_functions"] = temporal_count

        join_count = len(re.findall(r"\bJOIN\b", query, re.IGNORECASE))
        characteristics["total_join_count"] = join_count

        analysis.analytical_characteristics = characteristics

    def _extract_maintenance_characteristics(self, query: str, analysis: GoalAnalysis):
        """Extract maintenance-specific characteristics"""
        characteristics = {}

        characteristics["is_create"] = bool(self.create_pattern.search(query))
        characteristics["is_alter"] = bool(self.alter_pattern.search(query))
        characteristics["is_drop"] = bool(self.drop_pattern.search(query))
        characteristics["is_truncate"] = bool(self.truncate_pattern.search(query))
        characteristics["is_permission_change"] = bool(self.grant_pattern.search(query))

        # Schema object type
        schema_match = re.search(
            r"\b(CREATE|ALTER|DROP)\s+(TABLE|VIEW|INDEX|DATABASE|PROCEDURE|FUNCTION)\b",
            query,
            re.IGNORECASE,
        )
        if schema_match:
            characteristics["schema_object_type"] = schema_match.group(2)

        characteristics["is_analyze_optimize"] = bool(
            re.search(r"\b(ANALYZE|VACUUM|OPTIMIZE|REINDEX)\b", query, re.IGNORECASE)
        )

        analysis.maintenance_characteristics = characteristics

    def _generate_recommendations(self, analysis: GoalAnalysis) -> List[str]:
        """Generate goal-specific optimization recommendations"""
        recommendations = []

        if analysis.primary_goal == QueryGoal.REPORTING:
            if analysis.reporting_characteristics.get("aggregation_count", 0) > 0:
                if not analysis.reporting_characteristics.get("has_having_clause"):
                    recommendations.append(
                        "Consider adding HAVING clause to filter aggregated results"
                    )

            if analysis.reporting_characteristics.get("case_statements", 0) > 0:
                recommendations.append(
                    "Consider using computed columns for CASE statements used in reports"
                )

            if analysis.reporting_characteristics.get("group_by_columns", 0) > 5:
                recommendations.append(
                    "Many GROUP BY columns - verify all are necessary for the report"
                )

            recommendations.append(
                "Add index on GROUP BY columns for better aggregation performance"
            )

        elif analysis.primary_goal == QueryGoal.TRANSACTIONAL:
            if analysis.transactional_characteristics.get(
                "is_update"
            ) or analysis.transactional_characteristics.get("is_delete"):
                recommendations.append(
                    "Use WHERE clause with indexed columns to limit affected rows"
                )

            if analysis.transactional_characteristics.get("has_lock"):
                recommendations.append(
                    "Review locking strategy - consider optimistic locking for high concurrency"
                )

            if not analysis.transactional_characteristics.get("limit_rows"):
                recommendations.append(
                    "Consider LIMIT for transactional deletes to reduce locking duration"
                )

        elif analysis.primary_goal == QueryGoal.ANALYTICAL:
            if analysis.analytical_characteristics.get("complex_join_count", 0) > 0:
                recommendations.append(
                    "Review JOIN order for optimal execution - consider CTEs to organize"
                )

            if analysis.analytical_characteristics.get("has_window_functions"):
                recommendations.append(
                    "Ensure appropriate indexes on PARTITION BY columns for window functions"
                )

            if analysis.analytical_characteristics.get("has_recursive_cte"):
                recommendations.append(
                    "Monitor recursive CTE depth - consider materialization for large hierarchies"
                )

            recommendations.append(
                "Add indexes on frequently filtered columns before aggregation"
            )

        elif analysis.primary_goal == QueryGoal.MAINTENANCE:
            if analysis.maintenance_characteristics.get("is_create"):
                recommendations.append(
                    "Define appropriate indexes on foreign keys and frequently searched columns"
                )

            if analysis.maintenance_characteristics.get("is_alter"):
                recommendations.append(
                    "Review schema change impact on dependent queries and indexes"
                )

            if analysis.maintenance_characteristics.get("is_drop"):
                recommendations.append(
                    "Verify no active queries depend on dropped objects"
                )

        elif analysis.primary_goal == QueryGoal.EXPLORATORY:
            recommendations.append("Add WHERE clause to limit result set size")
            recommendations.append(
                "Use LIMIT for large tables to improve response time"
            )

        # General recommendations for read-only queries
        if analysis.is_read_only and analysis.primary_goal in [
            QueryGoal.REPORTING,
            QueryGoal.ANALYTICAL,
        ]:
            recommendations.append(
                "Consider materialized views for frequently run queries"
            )

        return recommendations
