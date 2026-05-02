"""
Nested Subquery Semantic Analyzer

Advanced analysis of nested SQL subqueries with support for 3+ nesting levels,
dependency graph construction, and performance impact assessment.
"""

import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


class SubqueryType(Enum):
    """Classification of subquery types"""

    SCALAR = "scalar"  # Returns single value (e.g., in SELECT or WHERE)
    CORRELATED = "correlated"  # References outer query columns
    DERIVED_TABLE = "derived_table"  # Used in FROM clause
    INLINE_VIEW = "inline_view"  # Subquery in FROM with alias
    EXISTS = "exists"  # EXISTS/NOT EXISTS subquery
    IN_LIST = "in_list"  # IN/NOT IN subquery
    COMPARISON = "comparison"  # Used with comparison operators
    UNION_COMPONENT = "union_component"  # Part of UNION query


class SubqueryLocation(Enum):
    """Location of subquery in query structure"""

    SELECT_LIST = "select_list"  # In SELECT clause
    FROM_CLAUSE = "from_clause"  # In FROM clause
    WHERE_CLAUSE = "where_clause"  # In WHERE clause
    JOIN_CONDITION = "join_condition"  # In JOIN ON condition
    HAVING_CLAUSE = "having_clause"  # In HAVING clause
    ORDER_BY = "order_by"  # In ORDER BY clause
    UNION = "union"  # Part of UNION/INTERSECT/EXCEPT
    CTE = "cte"  # In WITH clause


@dataclass
class SubqueryNode:
    """Represents a single subquery in the hierarchy"""

    query_text: str
    depth: int
    type: SubqueryType
    location: SubqueryLocation
    nesting_path: List[int]  # Path in the tree (e.g., [0, 1, 2])
    parent_id: Optional[int] = None
    references_outer: bool = False  # Correlated reference to parent
    column_references: List[str] = field(default_factory=list)
    table_references: List[str] = field(default_factory=list)
    aggregate_functions: List[str] = field(default_factory=list)
    has_group_by: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    complexity_score: float = 0.0
    estimated_selectivity: float = 0.5


@dataclass
class NestedSubqueryAnalysis:
    """Complete analysis of nested subqueries in a query"""

    total_subquery_count: int = 0
    max_nesting_depth: int = 0
    nesting_levels: Dict[int, int] = field(default_factory=dict)  # depth -> count
    subquery_types: Dict[str, int] = field(default_factory=dict)  # type -> count
    subquery_locations: Dict[str, int] = field(
        default_factory=dict
    )  # location -> count
    correlated_count: int = 0
    derived_table_count: int = 0
    subquery_nodes: List[SubqueryNode] = field(default_factory=list)
    dependency_graph: Dict[int, List[int]] = field(
        default_factory=dict
    )  # parent -> children
    complexity_score: float = 0.0
    performance_risk_level: str = "low"  # low, medium, high, critical


class NestedSubqueryAnalyzer:
    """Advanced analyzer for nested subqueries"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()
        self.node_counter = 0

    def _compile_patterns(self):
        """Compile regex patterns for subquery analysis"""
        # Match SELECT statements (including nested)
        self.select_pattern = re.compile(r"\bSELECT\s+(?:DISTINCT\s+)?", re.IGNORECASE)

        # Match various subquery contexts
        self.scalar_subquery_pattern = re.compile(
            r"(COUNT|SUM|AVG|MAX|MIN|STDDEV|VARIANCE)\s*\(\s*(?:SELECT|CASE)",
            re.IGNORECASE,
        )

        self.exists_pattern = re.compile(r"\b(EXISTS|NOT\s+EXISTS)\s*\(", re.IGNORECASE)

        self.in_pattern = re.compile(r"\b(IN|NOT\s+IN)\s*\(", re.IGNORECASE)

        self.from_pattern = re.compile(r"\bFROM\s*\(", re.IGNORECASE)

        self.join_pattern = re.compile(
            r"\b(INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+\(", re.IGNORECASE
        )

        # Correlation indicators
        self.correlation_pattern = re.compile(
            r"(\w+)\.(\w+)\s+(?:=|<>|<|>|<=|>=)", re.IGNORECASE
        )

    def analyze_nested_subqueries(self, query: str) -> NestedSubqueryAnalysis:
        """Analyze all nested subqueries in a query"""
        try:
            self.node_counter = 0
            analysis = NestedSubqueryAnalysis()

            # Extract all SELECT statements with positions
            select_positions = self._find_all_selects(query)

            if len(select_positions) <= 1:
                # No nested subqueries
                return analysis

            # Build subquery tree
            subqueries = self._extract_subqueries(query, select_positions)
            analysis.subquery_nodes = subqueries

            # Calculate nesting structure
            analysis.total_subquery_count = len(subqueries)
            analysis.max_nesting_depth = max([s.depth for s in subqueries], default=0)

            # Collect statistics
            for node in subqueries:
                # Nesting level distribution
                analysis.nesting_levels[node.depth] = (
                    analysis.nesting_levels.get(node.depth, 0) + 1
                )

                # Type distribution
                type_name = node.type.value
                analysis.subquery_types[type_name] = (
                    analysis.subquery_types.get(type_name, 0) + 1
                )

                # Location distribution
                location_name = node.location.value
                analysis.subquery_locations[location_name] = (
                    analysis.subquery_locations.get(location_name, 0) + 1
                )

                # Count correlated
                if node.references_outer:
                    analysis.correlated_count += 1

                # Count derived tables
                if node.type == SubqueryType.DERIVED_TABLE:
                    analysis.derived_table_count += 1

            # Build dependency graph
            analysis.dependency_graph = self._build_dependency_graph(subqueries)

            # Calculate complexity score
            analysis.complexity_score = self._calculate_complexity_score(subqueries)

            # Assess performance risk
            analysis.performance_risk_level = self._assess_performance_risk(
                analysis, subqueries
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing nested subqueries: {e}")
            return NestedSubqueryAnalysis()

    def _find_all_selects(self, query: str) -> List[Tuple[int, int]]:
        """Find all SELECT statement positions in query"""
        positions = []
        for match in self.select_pattern.finditer(query):
            # Find matching closing parenthesis or end of statement
            start = match.start()
            end = self._find_select_end(query, start)
            positions.append((start, end))
        return positions

    def _find_select_end(self, query: str, start: int) -> int:
        """Find the end position of a SELECT statement"""
        i = start
        paren_count = 0
        in_string = False
        string_char = None

        while i < len(query):
            char = query[i]

            # Handle string literals
            if char in ('"', "'") and (i == 0 or query[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False

            # Only count parentheses outside strings
            if not in_string:
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count < 0:
                        return i

                # Check for statement terminators
                if char == ";" and paren_count == 0:
                    return i

            i += 1

        return len(query)

    def _extract_subqueries(
        self, query: str, select_positions: List[Tuple[int, int]]
    ) -> List[SubqueryNode]:
        """Extract and classify all subqueries"""
        subqueries = []

        for pos_idx, (start, end) in enumerate(select_positions):
            if pos_idx == 0:
                continue  # Skip main query

            subquery_text = query[start:end].strip()

            # Determine depth (count parent SELECTs)
            depth = sum(
                1
                for p_start, p_end in select_positions[:pos_idx]
                if p_start < start < p_end
            )

            # Classify type and location
            subquery_type = self._classify_subquery_type(query, start, subquery_text)
            location = self._determine_location(query, start)

            # Create node
            node = SubqueryNode(
                query_text=subquery_text,
                depth=depth + 1,  # Main query is depth 0
                type=subquery_type,
                location=location,
                nesting_path=[pos_idx],
                parent_id=pos_idx - 1 if pos_idx > 1 else None,
                references_outer=self._check_correlated(subquery_text, query[:start]),
            )

            # Extract references and functions
            node.column_references = self._extract_column_references(subquery_text)
            node.table_references = self._extract_table_references(subquery_text)
            node.aggregate_functions = self._extract_aggregate_functions(subquery_text)
            node.has_group_by = "GROUP BY" in subquery_text.upper()
            node.has_order_by = "ORDER BY" in subquery_text.upper()
            node.has_limit = (
                "LIMIT" in subquery_text.upper() or "TOP" in subquery_text.upper()
            )

            # Calculate complexity
            node.complexity_score = self._calculate_node_complexity(node)

            subqueries.append(node)

        return subqueries

    def _classify_subquery_type(
        self, full_query: str, position: int, subquery_text: str
    ) -> SubqueryType:
        """Classify the type of subquery"""
        query_upper = full_query.upper()
        before = full_query[:position].upper()
        after = full_query[position + 6 : position + 20].upper()  # After SELECT keyword

        # Check for EXISTS
        if "EXISTS" in before[-10:]:
            return SubqueryType.EXISTS

        # Check for IN/NOT IN
        if "IN" in before[-5:] and "(" in before[-1:]:
            return SubqueryType.IN_LIST

        # Check for comparison operators (=, <, >, etc.)
        if any(op in before[-3:] for op in ["=", "<", ">"]):
            return SubqueryType.COMPARISON

        # Check for FROM clause (derived table)
        if "FROM" in before[-10:]:
            return SubqueryType.DERIVED_TABLE

        # Check for JOIN
        if "JOIN" in before[-10:]:
            return SubqueryType.DERIVED_TABLE

        # Check for aggregate function (scalar subquery)
        if self.scalar_subquery_pattern.search(before[-50:]):
            return SubqueryType.SCALAR

        # Default to scalar if in SELECT list
        if "SELECT" in before[-50:] and "FROM" not in before[-30:]:
            return SubqueryType.SCALAR

        return SubqueryType.SCALAR

    def _determine_location(self, full_query: str, position: int) -> SubqueryLocation:
        """Determine where the subquery is located in the query"""
        before = full_query[:position].upper()

        # Find the last major clause
        select_pos = before.rfind("SELECT")
        from_pos = before.rfind("FROM")
        where_pos = before.rfind("WHERE")
        having_pos = before.rfind("HAVING")
        orderby_pos = before.rfind("ORDER BY")
        join_pos = before.rfind("JOIN")

        positions = {
            "select": select_pos if select_pos > from_pos else -1,
            "from": from_pos if from_pos > where_pos else -1,
            "where": where_pos if where_pos > from_pos else -1,
            "having": having_pos if having_pos > where_pos else -1,
            "orderby": orderby_pos if orderby_pos > having_pos else -1,
            "join": join_pos if join_pos > from_pos else -1,
        }

        # Determine location based on last major clause
        if positions["orderby"] > positions["having"]:
            return SubqueryLocation.ORDER_BY
        elif positions["having"] > positions["where"]:
            return SubqueryLocation.HAVING_CLAUSE
        elif positions["join"] > positions["where"]:
            return SubqueryLocation.JOIN_CONDITION
        elif positions["where"] > positions["from"]:
            return SubqueryLocation.WHERE_CLAUSE
        elif positions["from"] > positions["select"]:
            return SubqueryLocation.FROM_CLAUSE
        else:
            return SubqueryLocation.SELECT_LIST

    def _check_correlated(self, subquery_text: str, before_query: str) -> bool:
        """Check if subquery is correlated (references outer query)"""
        # Look for table aliases with dots (e.g., t1.column)
        table_alias_pattern = re.compile(r"\b(\w+)\.(\w+)\b", re.IGNORECASE)
        references = table_alias_pattern.findall(subquery_text)

        if not references:
            return False

        # Extract table aliases from outer query
        outer_aliases = set()
        for match in re.finditer(
            r"FROM\s+\w+\s+(?:AS\s+)?(\w+)", before_query, re.IGNORECASE
        ):
            outer_aliases.add(match.group(1).lower())

        # Check if any reference uses outer alias
        for table_alias, _ in references:
            if table_alias.lower() in outer_aliases:
                return True

        return False

    def _extract_column_references(self, subquery_text: str) -> List[str]:
        """Extract all column references from subquery"""
        pattern = re.compile(
            r"\b(\w+)\.(\w+)\b|\b(?:SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY)\s+(\w+)",
            re.IGNORECASE,
        )
        references = []

        for match in pattern.finditer(subquery_text):
            if match.group(1) and match.group(2):  # table.column format
                references.append(f"{match.group(1)}.{match.group(2)}")
            elif match.group(3):  # column name
                references.append(match.group(3))

        return list(set(references))

    def _extract_table_references(self, subquery_text: str) -> List[str]:
        """Extract all table references from subquery"""
        pattern = re.compile(r"\bFROM\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?", re.IGNORECASE)
        tables = []

        for match in pattern.finditer(subquery_text):
            tables.append(match.group(1))

        return tables

    def _extract_aggregate_functions(self, subquery_text: str) -> List[str]:
        """Extract aggregate functions from subquery"""
        pattern = re.compile(
            r"\b(COUNT|SUM|AVG|MAX|MIN|STDDEV|VARIANCE|GROUP_CONCAT|STRING_AGG)\s*\(",
            re.IGNORECASE,
        )
        functions = []

        for match in pattern.finditer(subquery_text):
            functions.append(match.group(1).upper())

        return functions

    def _calculate_node_complexity(self, node: SubqueryNode) -> float:
        """Calculate complexity score for a single subquery node"""
        score = 0.0

        # Depth factor (deeper = more complex)
        score += min(0.3, node.depth * 0.1)

        # Type factor
        if node.type == SubqueryType.CORRELATED:
            score += 0.3  # Correlated very expensive
        elif node.type == SubqueryType.SCALAR:
            score += 0.2

        # Features
        if node.references_outer:
            score += 0.25
        if node.has_group_by:
            score += 0.15
        if node.has_order_by:
            score += 0.1
        if len(node.aggregate_functions) > 0:
            score += 0.15
        if len(node.column_references) > 5:
            score += 0.1

        return min(1.0, score)

    def _build_dependency_graph(
        self, subqueries: List[SubqueryNode]
    ) -> Dict[int, List[int]]:
        """Build dependency graph showing parent-child relationships"""
        graph = defaultdict(list)

        for i, node in enumerate(subqueries):
            if node.parent_id is not None:
                graph[node.parent_id].append(i)

        return dict(graph)

    def _calculate_complexity_score(self, subqueries: List[SubqueryNode]) -> float:
        """Calculate overall complexity score"""
        if not subqueries:
            return 0.0

        # Average node complexity weighted by depth
        depth_weighted_score = sum(
            node.complexity_score * (node.depth / 10) for node in subqueries
        ) / len(subqueries)

        # Maximum nesting penalty
        max_depth = max((node.depth for node in subqueries), default=0)
        depth_penalty = min(0.3, max_depth * 0.1)

        # Correlated subquery penalty
        correlated_count = sum(1 for node in subqueries if node.references_outer)
        correlated_penalty = min(0.3, correlated_count * 0.15)

        return min(1.0, depth_weighted_score + depth_penalty + correlated_penalty)

    def _assess_performance_risk(
        self, analysis: NestedSubqueryAnalysis, subqueries: List[SubqueryNode]
    ) -> str:
        """Assess performance risk level"""
        risk_score = 0.0

        # Nesting depth risk
        if analysis.max_nesting_depth >= 5:
            risk_score += 0.4
        elif analysis.max_nesting_depth >= 3:
            risk_score += 0.2
        elif analysis.max_nesting_depth >= 2:
            risk_score += 0.1

        # Correlated subquery risk (very expensive)
        if analysis.correlated_count >= 2:
            risk_score += 0.4
        elif analysis.correlated_count >= 1:
            risk_score += 0.25

        # Subquery count risk
        if analysis.total_subquery_count >= 10:
            risk_score += 0.3
        elif analysis.total_subquery_count >= 5:
            risk_score += 0.15

        # Derived table without indexes risk
        for node in subqueries:
            if node.type == SubqueryType.DERIVED_TABLE and not node.has_limit:
                risk_score += 0.1

        # Determine risk level
        if risk_score >= 0.7:
            return "critical"
        elif risk_score >= 0.5:
            return "high"
        elif risk_score >= 0.25:
            return "medium"
        else:
            return "low"

    def get_performance_recommendations(
        self, analysis: NestedSubqueryAnalysis
    ) -> List[str]:
        """Get performance recommendations based on analysis"""
        recommendations = []

        if analysis.max_nesting_depth >= 3:
            recommendations.append(
                "Consider using CTEs (WITH clause) to flatten deeply nested subqueries"
            )

        if analysis.correlated_count > 0:
            recommendations.append(
                "Correlated subqueries detected - consider using JOINs or window functions"
            )

        if analysis.derived_table_count > 2:
            recommendations.append(
                "Multiple derived tables found - consider consolidating using CTEs"
            )

        if analysis.total_subquery_count >= 10:
            recommendations.append(
                "Query has many subqueries - consider refactoring for readability and performance"
            )

        return recommendations
