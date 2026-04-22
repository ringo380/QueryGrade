"""
CTE Semantic Analyzer

Advanced analysis of Common Table Expressions (CTEs) with semantic understanding of:
- CTE purpose classification (data prep, aggregation, hierarchy, etc.)
- Recursion detection and validation
- Reusability patterns
- Performance implications
- CTE dependency graphs
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class CTEPurpose(Enum):
    """Classification of CTE purposes"""

    DATA_PREPARATION = "data_preparation"  # Filtering, transforming base data
    AGGREGATION = "aggregation"  # Summary statistics, GROUP BY operations
    HIERARCHY = "hierarchy"  # Tree/hierarchy traversal (recursive)
    DEDUPLICATION = "deduplication"  # DISTINCT, removing duplicates
    STAGING = "staging"  # Intermediate step in complex query
    WINDOW_FUNCTION = "window_function"  # Analytics using window functions
    UNION_BASE = "union_base"  # Base for UNION operations
    ITERATION = "iteration"  # Iterative computation (recursive)
    UNKNOWN = "unknown"  # Cannot determine purpose


class CTEComplexity(Enum):
    """CTE complexity levels"""

    SIMPLE = "simple"  # Single table, basic filtering
    MODERATE = "moderate"  # Joins, aggregations
    COMPLEX = "complex"  # Multiple levels, recursion
    VERY_COMPLEX = "very_complex"  # Deep recursion, many JOINs


@dataclass
class CTEDefinition:
    """Represents a single CTE definition"""

    name: str
    query_text: str
    purpose: CTEPurpose = CTEPurpose.UNKNOWN
    complexity: CTEComplexity = CTEComplexity.SIMPLE
    is_recursive: bool = False
    recursion_depth_estimate: int = 0  # Estimated depth for recursive CTEs
    usage_count: int = 0  # How many times used in query
    is_used: bool = False
    references_tables: List[str] = field(default_factory=list)
    references_ctes: List[str] = field(
        default_factory=list
    )  # Other CTEs this references
    join_count: int = 0
    aggregate_count: int = 0
    window_function_count: int = 0
    subquery_count: int = 0
    complexity_score: float = 0.0  # 0-1 scale
    performance_risk_level: str = "low"  # low/medium/high/critical


@dataclass
class CTEAnalysis:
    """Complete analysis of all CTEs in a query"""

    total_cte_count: int = 0
    cte_definitions: List[CTEDefinition] = field(default_factory=list)
    cte_purposes: Dict[str, int] = field(default_factory=dict)  # purpose -> count
    cte_complexity_distribution: Dict[str, int] = field(
        default_factory=dict
    )  # complexity -> count
    recursive_cte_count: int = 0
    max_recursion_depth: int = 0
    unused_cte_count: int = 0  # CTEs defined but not used
    cte_dependency_graph: Dict[str, List[str]] = field(
        default_factory=dict
    )  # cte_name -> [references]
    overall_complexity_score: float = 0.0
    has_recursive_cte: bool = False
    performance_risk_level: str = "low"  # low/medium/high/critical
    optimization_opportunities: List[str] = field(default_factory=list)


class CTESemanticAnalyzer:
    """Advanced analyzer for CTE semantics and patterns"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for CTE analysis"""
        # Match CTE definition
        self.cte_pattern = re.compile(
            r"\bWITH\s+(?:RECURSIVE\s+)?(\w+)\s+(?:AS\s+)?\(\s*SELECT", re.IGNORECASE
        )

        # Match recursive CTE
        self.recursive_pattern = re.compile(r"\bWITH\s+RECURSIVE\b", re.IGNORECASE)

        # Match multiple CTEs
        self.multi_cte_pattern = re.compile(r"(\w+)\s*(?:AS\s*)?\(", re.IGNORECASE)

        # Match CTE usage
        self.usage_pattern = re.compile(
            r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)|\bIN\s*\(\s*SELECT.*?FROM\s+(\w+)",
            re.IGNORECASE | re.DOTALL,
        )

        # Match aggregation functions
        self.aggregate_pattern = re.compile(
            r"\b(COUNT|SUM|AVG|MAX|MIN|STDDEV|VARIANCE|GROUP_CONCAT|STRING_AGG)\s*\(",
            re.IGNORECASE,
        )

        # Match window functions
        self.window_pattern = re.compile(
            r"\b(ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|FIRST_VALUE|LAST_VALUE|NTILE)\s*\(",
            re.IGNORECASE,
        )

        # Match UNION in CTE
        self.union_pattern = re.compile(r"\bUNION(?:\s+ALL)?\b", re.IGNORECASE)

    def analyze_ctes(self, query: str) -> CTEAnalysis:
        """Analyze all CTEs in a query"""
        try:
            analysis = CTEAnalysis()

            # Check if query has CTEs
            if "WITH" not in query.upper():
                return analysis

            # Check for recursive
            analysis.has_recursive_cte = bool(self.recursive_pattern.search(query))

            # Extract CTE definitions
            cte_defs = self._extract_cte_definitions(query)
            analysis.cte_definitions = cte_defs
            analysis.total_cte_count = len(cte_defs)

            if analysis.total_cte_count == 0:
                return analysis

            # Analyze each CTE
            for cte_def in cte_defs:
                # Classify purpose
                cte_def.purpose = self._classify_cte_purpose(cte_def)

                # Determine complexity
                cte_def.complexity = self._determine_complexity(cte_def)
                cte_def.complexity_score = self._calculate_complexity_score(cte_def)

                # Collect statistics
                purpose_name = cte_def.purpose.value
                analysis.cte_purposes[purpose_name] = (
                    analysis.cte_purposes.get(purpose_name, 0) + 1
                )

                complexity_name = cte_def.complexity.value
                analysis.cte_complexity_distribution[complexity_name] = (
                    analysis.cte_complexity_distribution.get(complexity_name, 0) + 1
                )

                if cte_def.is_recursive:
                    analysis.recursive_cte_count += 1

                if not cte_def.is_used:
                    analysis.unused_cte_count += 1

            # Detect usage of CTEs
            self._detect_cte_usage(query, cte_defs)

            # Build dependency graph
            analysis.cte_dependency_graph = self._build_dependency_graph(cte_defs)

            # Calculate overall metrics
            analysis.overall_complexity_score = self._calculate_overall_complexity(
                cte_defs
            )
            analysis.performance_risk_level = self._assess_performance_risk(
                analysis, cte_defs
            )
            analysis.optimization_opportunities = self._generate_recommendations(
                analysis, cte_defs
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing CTEs: {e}")
            return CTEAnalysis()

    def _extract_cte_definitions(self, query: str) -> List[CTEDefinition]:
        """Extract all CTE definitions from query"""
        definitions = []

        # Find WITH clause
        with_match = re.search(r"\bWITH\s+(?:RECURSIVE\s+)?", query, re.IGNORECASE)
        if not with_match:
            return definitions

        # Find the main query start (after CTEs)
        with_start = with_match.end()
        main_select = re.search(r"\bSELECT\b", query[with_start:], re.IGNORECASE)
        if not main_select:
            return definitions

        with_section = query[with_start : with_start + main_select.start()]

        # Split by comma at top level (not in parentheses)
        cte_strings = self._split_cte_definitions(with_section)

        for cte_str in cte_strings:
            # Extract CTE name and query
            match = re.match(
                r"(\w+)\s+(?:AS\s+)?\((.*)\)\s*$",
                cte_str.strip(),
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                name = match.group(1)
                cte_query = match.group(2)

                definition = CTEDefinition(
                    name=name,
                    query_text=cte_query,
                    is_recursive=bool(
                        re.search(r"\bWITH\s+RECURSIVE\b", cte_query, re.IGNORECASE)
                    ),
                )

                # Analyze the CTE query
                definition.references_tables = self._extract_table_references(cte_query)
                definition.references_ctes = self._extract_cte_references(cte_query)
                definition.join_count = len(
                    re.findall(r"\bJOIN\b", cte_query, re.IGNORECASE)
                )
                definition.aggregate_count = len(
                    self.aggregate_pattern.findall(cte_query)
                )
                definition.window_function_count = len(
                    self.window_pattern.findall(cte_query)
                )
                definition.subquery_count = cte_query.upper().count("SELECT") - 1

                definitions.append(definition)

        return definitions

    def _split_cte_definitions(self, cte_section: str) -> List[str]:
        """Split multiple CTE definitions by comma"""
        ctes = []
        current = ""
        paren_depth = 0

        for char in cte_section:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "," and paren_depth == 0:
                ctes.append(current.strip())
                current = ""
                continue

            current += char

        if current.strip():
            ctes.append(current.strip())

        return ctes

    def _classify_cte_purpose(self, cte_def: CTEDefinition) -> CTEPurpose:
        """Classify the purpose of a CTE"""
        query_upper = cte_def.query_text.upper()

        # Recursive CTE - for hierarchy/iteration
        if cte_def.is_recursive:
            if "CONNECT BY" in query_upper or "START WITH" in query_upper:
                return CTEPurpose.HIERARCHY
            return CTEPurpose.ITERATION

        # Aggregation CTE
        if cte_def.aggregate_count > 0 and "GROUP BY" in query_upper:
            return CTEPurpose.AGGREGATION

        # Window function CTE
        if cte_def.window_function_count > 0:
            return CTEPurpose.WINDOW_FUNCTION

        # UNION CTE
        if self.union_pattern.search(cte_def.query_text):
            return CTEPurpose.UNION_BASE

        # Deduplication (DISTINCT)
        if "DISTINCT" in query_upper:
            return CTEPurpose.DEDUPLICATION

        # Data preparation (filtering, basic transformations)
        if "WHERE" in query_upper:
            return CTEPurpose.DATA_PREPARATION

        # Default staging
        return CTEPurpose.STAGING

    def _determine_complexity(self, cte_def: CTEDefinition) -> CTEComplexity:
        """Determine CTE complexity level"""
        complexity_score = 0

        # Join factor
        if cte_def.join_count >= 3:
            complexity_score += 3
        elif cte_def.join_count >= 2:
            complexity_score += 2
        elif cte_def.join_count >= 1:
            complexity_score += 1

        # Aggregation factor
        if cte_def.aggregate_count > 0:
            complexity_score += 1

        # Window function factor
        if cte_def.window_function_count > 0:
            complexity_score += 1

        # Subquery factor
        if cte_def.subquery_count > 0:
            complexity_score += cte_def.subquery_count

        # Recursion factor
        if cte_def.is_recursive:
            complexity_score += 3

        # Determine level
        if complexity_score >= 6:
            return CTEComplexity.VERY_COMPLEX
        elif complexity_score >= 4:
            return CTEComplexity.COMPLEX
        elif complexity_score >= 2:
            return CTEComplexity.MODERATE
        else:
            return CTEComplexity.SIMPLE

    def _calculate_complexity_score(self, cte_def: CTEDefinition) -> float:
        """Calculate complexity score for a CTE (0-1 scale)"""
        score = 0.0

        # Join complexity
        score += min(0.3, cte_def.join_count * 0.1)

        # Aggregation complexity
        if cte_def.aggregate_count > 0:
            score += 0.15

        # Window function complexity
        if cte_def.window_function_count > 0:
            score += 0.2

        # Subquery complexity
        score += min(0.2, cte_def.subquery_count * 0.1)

        # Recursive complexity
        if cte_def.is_recursive:
            score += 0.4

        # CTE references
        if len(cte_def.references_ctes) > 0:
            score += min(0.15, len(cte_def.references_ctes) * 0.05)

        return min(1.0, score)

    def _detect_cte_usage(self, query: str, cte_defs: List[CTEDefinition]):
        """Detect how CTEs are used in the main query"""
        main_select_match = re.search(r"FROM\s+", query, re.IGNORECASE)
        if not main_select_match:
            return

        main_query = query[main_select_match.start() :]

        # Count usage of each CTE
        for cte_def in cte_defs:
            pattern = r"\b" + re.escape(cte_def.name) + r"\b"
            matches = list(re.finditer(pattern, main_query, re.IGNORECASE))
            cte_def.usage_count = len(matches)
            cte_def.is_used = cte_def.usage_count > 0

    def _extract_table_references(self, cte_query: str) -> List[str]:
        """Extract table references from CTE query"""
        tables = []
        pattern = re.compile(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", re.IGNORECASE)

        for match in pattern.finditer(cte_query):
            table = match.group(1) or match.group(2)
            if table:
                tables.append(table)

        return list(set(tables))

    def _extract_cte_references(self, cte_query: str) -> List[str]:
        """Extract references to other CTEs"""
        # This would require knowing other CTE names, which we do at a higher level
        # For now, return empty - this will be populated by the caller
        return []

    def _build_dependency_graph(
        self, cte_defs: List[CTEDefinition]
    ) -> Dict[str, List[str]]:
        """Build dependency graph showing which CTEs reference which"""
        graph = {}

        for cte_def in cte_defs:
            graph[cte_def.name] = cte_def.references_ctes

        return graph

    def _calculate_overall_complexity(self, cte_defs: List[CTEDefinition]) -> float:
        """Calculate overall complexity across all CTEs"""
        if not cte_defs:
            return 0.0

        avg_score = sum(c.complexity_score for c in cte_defs) / len(cte_defs)

        # Penalty for multiple CTEs
        multi_cte_penalty = min(0.2, len(cte_defs) * 0.05) if len(cte_defs) > 1 else 0

        # Bonus for unused CTEs (indicates dead code)
        unused_penalty = min(0.1, sum(1 for c in cte_defs if not c.is_used) * 0.05)

        return min(1.0, avg_score + multi_cte_penalty + unused_penalty)

    def _assess_performance_risk(
        self, analysis: CTEAnalysis, cte_defs: List[CTEDefinition]
    ) -> str:
        """Assess overall performance risk of CTEs"""
        risk_score = 0.0

        # Recursive CTE risk
        if analysis.recursive_cte_count > 0:
            risk_score += 0.3

        # Unused CTE risk (dead code)
        if analysis.unused_cte_count > 0:
            risk_score += 0.15

        # Complex CTE risk
        complex_count = sum(
            1
            for c in cte_defs
            if c.complexity in [CTEComplexity.COMPLEX, CTEComplexity.VERY_COMPLEX]
        )
        risk_score += min(0.3, complex_count * 0.1)

        # Many CTEs risk
        if analysis.total_cte_count >= 5:
            risk_score += 0.2

        # Determine risk level
        if risk_score >= 0.7:
            return "critical"
        elif risk_score >= 0.5:
            return "high"
        elif risk_score >= 0.25:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self, analysis: CTEAnalysis, cte_defs: List[CTEDefinition]
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Unused CTE warning
        if analysis.unused_cte_count > 0:
            recommendations.append("Remove unused CTEs to simplify query")

        # Recursive CTE warning
        if analysis.recursive_cte_count > 0:
            recommendations.append(
                "Verify recursive CTE termination conditions to avoid infinite loops"
            )

        # Too many CTEs
        if analysis.total_cte_count >= 5:
            recommendations.append(
                "Consider consolidating CTEs or breaking query into simpler parts"
            )

        # Complex CTE warning
        complex_ctes = [
            c for c in cte_defs if c.complexity == CTEComplexity.VERY_COMPLEX
        ]
        if complex_ctes:
            names = ", ".join([c.name for c in complex_ctes])
            recommendations.append(
                f"CTEs are very complex: {names} - consider breaking into smaller pieces"
            )

        # CTE with many joins
        high_join_ctes = [c for c in cte_defs if c.join_count >= 3]
        if high_join_ctes:
            recommendations.append(
                "Some CTEs have many JOINs - verify index usage for optimal performance"
            )

        return recommendations

    def get_cte_recommendations(self, analysis: CTEAnalysis) -> List[str]:
        """Get specific recommendations for CTE optimization"""
        recommendations = []

        for cte_def in analysis.cte_definitions:
            if not cte_def.is_used:
                recommendations.append(
                    f"CTE '{cte_def.name}' is defined but never used"
                )

            if cte_def.complexity == CTEComplexity.VERY_COMPLEX:
                recommendations.append(
                    f"CTE '{cte_def.name}' is very complex - consider simplification"
                )

            if cte_def.is_recursive and cte_def.recursion_depth_estimate > 10:
                recommendations.append(
                    f"Recursive CTE '{cte_def.name}' may be expensive with depth {cte_def.recursion_depth_estimate}"
                )

        return recommendations
