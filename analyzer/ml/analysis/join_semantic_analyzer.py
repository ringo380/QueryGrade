"""
JOIN Semantic Analyzer

Advanced analysis of SQL JOIN operations with semantic understanding of:
- JOIN type semantics (INNER, LEFT, RIGHT, FULL, CROSS)
- Impact on result cardinality
- Condition complexity and optimization opportunities
- Implicit joins detection
- Redundant join identification
- Join elimination analysis
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class JoinType(Enum):
    """Classification of JOIN types"""

    INNER = "inner"  # Result reducing, only matches
    LEFT = "left"  # Result preserving from left, matches + nulls
    RIGHT = "right"  # Result preserving from right, matches + nulls
    FULL = "full"  # Result preserving from both, matches + nulls
    CROSS = "cross"  # Cartesian product
    NATURAL = "natural"  # Implicit column matching
    IMPLICIT = "implicit"  # Join in WHERE clause


class JoinImpact(Enum):
    """Impact of JOIN on result cardinality"""

    RESULT_REDUCING = "result_reducing"  # Filters rows (INNER, most cases)
    RESULT_PRESERVING = "result_preserving"  # Preserves left rows (LEFT JOIN)
    RESULT_EXPANDING = "result_expanding"  # Increases rows (CROSS JOIN, duplicates)
    RESULT_UNKNOWN = "result_unknown"  # Cannot determine impact


class ConditionType(Enum):
    """Type of JOIN condition"""

    EQUI_JOIN = "equi_join"  # Equality condition (most efficient)
    THETA_JOIN = "theta_join"  # Non-equality comparison
    NATURAL_JOIN = "natural_join"  # Implicit column matching
    CROSS_JOIN = "cross_join"  # No condition


@dataclass
class JoinCondition:
    """Represents a single JOIN condition"""

    left_table: str  # Left table reference
    left_column: str  # Left column
    operator: str  # =, <>, <, >, <=, >=, LIKE, IN, BETWEEN
    right_table: str  # Right table reference
    right_column: str  # Right column
    condition_type: ConditionType = ConditionType.EQUI_JOIN
    complexity_score: float = 0.0  # 0-1, higher = more complex
    selectivity_estimate: float = 0.5  # Estimated filtering power


@dataclass
class JoinNode:
    """Represents a single JOIN operation"""

    left_table: str
    right_table: str
    join_type: JoinType
    join_impact: JoinImpact
    conditions: List[JoinCondition] = field(default_factory=list)
    condition_count: int = 0
    condition_complexity: float = 0.0
    is_implicit: bool = False
    is_redundant: bool = False
    optimization_hints: List[str] = field(default_factory=list)
    estimated_output_rows_multiplier: float = 1.0  # Effect on row count


@dataclass
class JoinAnalysis:
    """Complete analysis of all JOINs in a query"""

    total_join_count: int = 0
    join_types: Dict[str, int] = field(default_factory=dict)  # type -> count
    join_impacts: Dict[str, int] = field(default_factory=dict)  # impact -> count
    inner_join_count: int = 0
    outer_join_count: int = 0  # LEFT/RIGHT/FULL
    cross_join_count: int = 0
    implicit_join_count: int = 0
    join_nodes: List[JoinNode] = field(default_factory=list)
    avg_condition_count: float = 0.0
    avg_condition_complexity: float = 0.0
    overall_complexity_score: float = 0.0
    join_chain_depth: int = 0  # Deepest join nesting
    result_cardinality_impact: str = "unknown"  # reducing/preserving/expanding/unknown
    optimization_opportunities: List[str] = field(default_factory=list)
    redundant_join_count: int = 0
    has_implicit_joins: bool = False


class JoinSemanticAnalyzer:
    """Advanced analyzer for JOIN semantics and optimization"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for JOIN analysis"""
        # Match JOIN statements
        self.join_pattern = re.compile(
            r"\b(INNER\s+)?JOIN\b|\b(LEFT|RIGHT|FULL)(\s+OUTER)?\s+JOIN\b|\bCROSS\s+JOIN\b|\bNATURAL\s+JOIN\b",
            re.IGNORECASE,
        )

        # Match table names with aliases
        self.table_alias_pattern = re.compile(
            r"\b(?:FROM|JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?(?:\s|,|\)|$)",
            re.IGNORECASE,
        )

        # Match ON conditions
        self.on_condition_pattern = re.compile(
            r"\bON\s+(.+?)(?=\bWHERE\b|\bGROUP\b|\bHAVING\b|\bORDER\b|\bLIMIT\b|$|\bAND\b|\bOR\b)",
            re.IGNORECASE | re.DOTALL,
        )

        # Match WHERE clause for implicit joins
        self.where_pattern = re.compile(
            r"\bWHERE\s+(.+?)(?=\bGROUP\b|\bHAVING\b|\bORDER\b|\bLIMIT\b|$)",
            re.IGNORECASE | re.DOTALL,
        )

        # Match specific operators in conditions
        self.operator_pattern = re.compile(
            r"(\w+)\.(\w+)\s*(=|<>|<|>|<=|>=|LIKE|IN|BETWEEN)\s*(\w+)\.(\w+)",
            re.IGNORECASE,
        )

    def analyze_joins(self, query: str) -> JoinAnalysis:
        """Analyze all JOINs in a query"""
        try:
            analysis = JoinAnalysis()

            # Extract JOIN clauses
            join_nodes = self._extract_joins(query)
            analysis.join_nodes = join_nodes
            analysis.total_join_count = len(join_nodes)

            if analysis.total_join_count == 0:
                return analysis

            # Collect statistics
            for node in join_nodes:
                # Type distribution
                type_name = node.join_type.value
                analysis.join_types[type_name] = (
                    analysis.join_types.get(type_name, 0) + 1
                )

                # Impact distribution
                impact_name = node.join_impact.value
                analysis.join_impacts[impact_name] = (
                    analysis.join_impacts.get(impact_name, 0) + 1
                )

                # Count by category
                if node.join_type == JoinType.INNER:
                    analysis.inner_join_count += 1
                elif node.join_type in [JoinType.LEFT, JoinType.RIGHT, JoinType.FULL]:
                    analysis.outer_join_count += 1
                elif node.join_type == JoinType.CROSS:
                    analysis.cross_join_count += 1

                if node.is_implicit:
                    analysis.implicit_join_count += 1
                    analysis.has_implicit_joins = True

                if node.is_redundant:
                    analysis.redundant_join_count += 1

            # Calculate averages
            if analysis.total_join_count > 0:
                analysis.avg_condition_count = (
                    sum(n.condition_count for n in join_nodes)
                    / analysis.total_join_count
                )
                analysis.avg_condition_complexity = (
                    sum(n.condition_complexity for n in join_nodes)
                    / analysis.total_join_count
                )

            # Determine result cardinality impact
            analysis.result_cardinality_impact = self._determine_cardinality_impact(
                join_nodes
            )

            # Calculate overall complexity
            analysis.overall_complexity_score = self._calculate_join_complexity(
                join_nodes
            )

            # Generate optimization opportunities
            analysis.optimization_opportunities = self._generate_recommendations(
                analysis, join_nodes
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing JOINs: {e}")
            return JoinAnalysis()

    def _extract_joins(self, query: str) -> List[JoinNode]:
        """Extract all JOIN operations from query"""
        nodes = []

        # Find all JOIN keywords
        join_matches = list(self.join_pattern.finditer(query))

        if not join_matches:
            return nodes

        # Extract table mapping (FROM and JOIN clauses)
        table_map = self._extract_table_mapping(query)

        # Process each JOIN
        for i, match in enumerate(join_matches):
            # Determine JOIN type
            join_text = match.group(0).upper()
            join_type = self._classify_join_type(join_text)

            # Find the ON clause for this JOIN
            start_pos = match.end()
            on_condition_text = self._extract_on_condition(query, start_pos)
            conditions = self._parse_conditions(on_condition_text, table_map)

            # Determine impact
            impact = self._determine_join_impact(join_type, conditions)

            # Create node
            left_table = table_map.get("current", "unknown")
            right_table = self._extract_right_table(query, start_pos, table_map)

            node = JoinNode(
                left_table=left_table,
                right_table=right_table,
                join_type=join_type,
                join_impact=impact,
                conditions=conditions,
                condition_count=len(conditions),
                condition_complexity=self._calculate_condition_complexity(conditions),
                is_implicit=False,
                is_redundant=False,
            )

            # Calculate output multiplier
            node.estimated_output_rows_multiplier = self._estimate_output_multiplier(
                node
            )

            nodes.append(node)

        # Detect implicit joins in WHERE clause
        implicit_nodes = self._detect_implicit_joins(query, table_map)
        nodes.extend(implicit_nodes)

        return nodes

    def _classify_join_type(self, join_text: str) -> JoinType:
        """Classify the type of JOIN"""
        join_upper = join_text.upper()

        if "CROSS" in join_upper:
            return JoinType.CROSS
        elif "NATURAL" in join_upper:
            return JoinType.NATURAL
        elif "FULL" in join_upper:
            return JoinType.FULL
        elif "LEFT" in join_upper:
            return JoinType.LEFT
        elif "RIGHT" in join_upper:
            return JoinType.RIGHT
        else:
            return JoinType.INNER

    def _determine_join_impact(
        self, join_type: JoinType, conditions: List[JoinCondition]
    ) -> JoinImpact:
        """Determine the impact of JOIN on result cardinality"""
        if join_type == JoinType.CROSS:
            return JoinImpact.RESULT_EXPANDING

        if join_type == JoinType.INNER:
            return JoinImpact.RESULT_REDUCING

        if join_type in [JoinType.LEFT, JoinType.RIGHT]:
            return JoinImpact.RESULT_PRESERVING

        if join_type == JoinType.FULL:
            return JoinImpact.RESULT_PRESERVING

        return JoinImpact.RESULT_UNKNOWN

    def _extract_table_mapping(self, query: str) -> Dict[str, str]:
        """Extract table name to alias mapping"""
        mapping = {}

        for match in self.table_alias_pattern.finditer(query):
            table_name = match.group(1)
            alias = match.group(2) if match.group(2) else table_name
            mapping[alias] = table_name
            mapping["current"] = alias

        return mapping

    def _extract_on_condition(self, query: str, start_pos: int) -> str:
        """Extract the ON clause condition"""
        remaining = query[start_pos:].upper()

        # Find ON keyword
        on_match = re.search(r"\bON\b", remaining, re.IGNORECASE)
        if not on_match:
            return ""

        # Find end of ON clause (next AND, OR, or JOIN/WHERE keyword)
        on_start = on_match.end()
        on_clause = remaining[on_start:]

        # Find where condition ends
        next_keyword = re.search(
            r"\b(AND|OR|JOIN|WHERE|GROUP|HAVING|ORDER|LIMIT|$)\b",
            on_clause,
            re.IGNORECASE,
        )
        if next_keyword:
            return on_clause[: next_keyword.start()]

        return on_clause

    def _parse_conditions(
        self, condition_text: str, table_map: Dict
    ) -> List[JoinCondition]:
        """Parse JOIN conditions"""
        conditions = []

        if not condition_text.strip():
            return conditions

        # Split by AND/OR
        parts = re.split(r"\b(AND|OR)\b", condition_text, flags=re.IGNORECASE)

        for part in parts:
            if part.upper() in ["AND", "OR"]:
                continue

            # Try to parse as comparison
            match = self.operator_pattern.search(part)
            if match:
                condition = JoinCondition(
                    left_table=match.group(1),
                    left_column=match.group(2),
                    operator=match.group(3),
                    right_table=match.group(4),
                    right_column=match.group(5),
                    condition_type=self._classify_condition_type(match.group(3)),
                )

                condition.complexity_score = self._score_condition_complexity(condition)
                conditions.append(condition)

        return conditions

    def _classify_condition_type(self, operator: str) -> ConditionType:
        """Classify the type of condition based on operator"""
        op_upper = operator.upper()

        if op_upper == "=":
            return ConditionType.EQUI_JOIN

        return ConditionType.THETA_JOIN

    def _score_condition_complexity(self, condition: JoinCondition) -> float:
        """Score the complexity of a condition"""
        score = 0.0

        # Equi-joins are simpler
        if condition.condition_type == ConditionType.EQUI_JOIN:
            score = 0.2
        else:
            score = 0.5  # Theta joins more complex

        # Specific operators can increase complexity
        if condition.operator.upper() == "LIKE":
            score += 0.2
        elif condition.operator.upper() == "BETWEEN":
            score += 0.1

        return min(1.0, score)

    def _calculate_condition_complexity(self, conditions: List[JoinCondition]) -> float:
        """Calculate overall complexity of JOIN conditions"""
        if not conditions:
            return 0.0

        avg_complexity = sum(c.complexity_score for c in conditions) / len(conditions)
        return min(1.0, avg_complexity)

    def _extract_right_table(self, query: str, start_pos: int, table_map: Dict) -> str:
        """Extract the right table name from JOIN clause"""
        remaining = query[start_pos:100]  # Look ahead 100 chars

        # Match table name pattern
        match = re.search(
            r"\b(\w+)(?:\s+(?:AS\s+)?(\w+))?(?:\s+ON\b|\s+USING\b|\s+WHERE\b|$)",
            remaining,
        )
        if match:
            return match.group(1)

        return "unknown"

    def _determine_cardinality_impact(self, nodes: List[JoinNode]) -> str:
        """Determine overall impact on result cardinality"""
        if not nodes:
            return "unknown"

        # Check for any expanding operations
        if any(n.join_impact == JoinImpact.RESULT_EXPANDING for n in nodes):
            return "expanding"

        # Check for any outer joins
        if any(n.join_impact == JoinImpact.RESULT_PRESERVING for n in nodes):
            return "preserving"

        # All are result reducing
        if all(n.join_impact == JoinImpact.RESULT_REDUCING for n in nodes):
            return "reducing"

        return "unknown"

    def _calculate_join_complexity(self, nodes: List[JoinNode]) -> float:
        """Calculate overall JOIN complexity"""
        if not nodes:
            return 0.0

        complexity = 0.0

        # Number of joins factor
        complexity += min(0.3, len(nodes) * 0.1)

        # Join type factor
        for node in nodes:
            if node.join_type == JoinType.CROSS:
                complexity += 0.3
            elif node.join_type == JoinType.INNER:
                complexity += 0.1
            elif node.join_type in [JoinType.LEFT, JoinType.RIGHT]:
                complexity += 0.15
            elif node.join_type == JoinType.FULL:
                complexity += 0.2

        # Condition complexity factor
        avg_condition_complexity = (
            sum(n.condition_complexity for n in nodes) / len(nodes) if nodes else 0
        )
        complexity += avg_condition_complexity * 0.2

        # Implicit joins factor
        if any(n.is_implicit for n in nodes):
            complexity += 0.2

        return min(1.0, complexity)

    def _estimate_output_multiplier(self, node: JoinNode) -> float:
        """Estimate how JOIN affects row count"""
        if node.join_type == JoinType.CROSS:
            return 10.0  # Worst case: Cartesian product

        if node.join_type == JoinType.INNER:
            return 0.5  # Reduces rows on average

        if node.join_type in [JoinType.LEFT, JoinType.RIGHT]:
            return 1.0  # Preserves left/right rows

        if node.join_type == JoinType.FULL:
            return 1.5  # May increase due to nulls

        return 1.0

    def _detect_implicit_joins(self, query: str, table_map: Dict) -> List[JoinNode]:
        """Detect implicit JOINs in WHERE clause"""
        nodes = []

        # Find WHERE clause
        where_match = self.where_pattern.search(query)
        if not where_match:
            return nodes

        where_clause = where_match.group(1)

        # Look for table.column = table.column patterns
        matches = list(self.operator_pattern.finditer(where_clause))

        for match in matches:
            left_table = match.group(1)
            right_table = match.group(4)

            # Only if joining different tables
            if left_table != right_table:
                condition = JoinCondition(
                    left_table=left_table,
                    left_column=match.group(2),
                    operator=match.group(3),
                    right_table=right_table,
                    right_column=match.group(5),
                )

                node = JoinNode(
                    left_table=left_table,
                    right_table=right_table,
                    join_type=JoinType.IMPLICIT,
                    join_impact=JoinImpact.RESULT_REDUCING,
                    conditions=[condition],
                    condition_count=1,
                    condition_complexity=self._score_condition_complexity(condition),
                    is_implicit=True,
                )

                nodes.append(node)

        return nodes

    def _generate_recommendations(
        self, analysis: JoinAnalysis, nodes: List[JoinNode]
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # CROSS JOIN warning
        if analysis.cross_join_count > 0:
            recommendations.append(
                "CROSS JOINs detected - verify these are intentional as they create Cartesian products"
            )

        # Implicit joins
        if analysis.has_implicit_joins:
            recommendations.append(
                "Implicit JOINs found in WHERE clause - use explicit JOIN syntax for clarity and optimization"
            )

        # Too many joins
        if analysis.total_join_count >= 5:
            recommendations.append(
                "Query has many JOINs - consider breaking into simpler queries or using CTEs"
            )

        # Complex conditions
        if analysis.avg_condition_complexity > 0.5:
            recommendations.append(
                "Complex JOIN conditions detected - consider indexes on join columns"
            )

        # Mix of INNER and OUTER
        if analysis.inner_join_count > 0 and analysis.outer_join_count > 0:
            recommendations.append(
                "Mix of INNER and OUTER JOINs - verify join order for optimal performance"
            )

        # All outer joins (can affect result cardinality unpredictably)
        if analysis.outer_join_count > 0 and analysis.inner_join_count == 0:
            recommendations.append(
                "All JOINs are outer JOINs - this may increase result rows unexpectedly"
            )

        return recommendations

    def get_join_recommendations(self, analysis: JoinAnalysis) -> List[str]:
        """Get specific recommendations for JOIN optimization"""
        recommendations = []

        for node in analysis.join_nodes:
            if node.is_implicit:
                recommendations.append(
                    f"Convert implicit join between {node.left_table} and {node.right_table} "
                    f"to explicit JOIN for clarity"
                )

            if node.join_type == JoinType.CROSS:
                recommendations.append(
                    f"Review CROSS JOIN between {node.left_table} and {node.right_table}"
                )

            if len(node.conditions) > 3:
                recommendations.append(
                    f"JOIN between {node.left_table} and {node.right_table} has {len(node.conditions)} "
                    f"conditions - consider denormalization or indexing"
                )

        return recommendations
