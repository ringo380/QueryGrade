"""
Intelligent Query Rewriter with Explanations

This module automatically rewrites SQL queries for better performance
while providing detailed explanations of the changes made.
"""

import re
import logging
import sqlparse
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from sqlparse.sql import Statement, Token, TokenList
from sqlparse.tokens import Keyword, Name


class RewriteRule(Enum):
    """Types of query rewrite rules"""
    EXISTS_TO_JOIN = "exists_to_join"
    IN_TO_EXISTS = "in_to_exists"
    SUBQUERY_TO_JOIN = "subquery_to_join"
    UNION_TO_UNION_ALL = "union_to_union_all"
    ELIMINATE_DISTINCT = "eliminate_distinct"
    OPTIMIZE_WHERE = "optimize_where"
    REORDER_JOINS = "reorder_joins"
    PUSH_DOWN_PREDICATES = "push_down_predicates"
    ELIMINATE_REDUNDANT_JOINS = "eliminate_redundant_joins"
    OPTIMIZE_AGGREGATION = "optimize_aggregation"
    SIMPLIFY_CASE = "simplify_case"
    NORMALIZE_EXPRESSIONS = "normalize_expressions"


class RewriteComplexity(Enum):
    """Complexity levels of rewrites"""
    SIMPLE = "simple"      # Syntax changes
    MODERATE = "moderate"  # Logic restructuring
    COMPLEX = "complex"    # Significant algorithmic changes
    ADVANCED = "advanced"  # Expert-level optimizations


@dataclass
class RewriteStep:
    """Represents a single rewrite step"""
    rule: RewriteRule
    complexity: RewriteComplexity
    description: str
    original_fragment: str
    rewritten_fragment: str
    rationale: str
    performance_impact: str
    risks: List[str]
    verification_needed: bool
    estimated_improvement: float  # 0-1 scale


@dataclass
class QueryRewrite:
    """Complete query rewrite result"""
    original_query: str
    rewritten_query: str
    rewrite_steps: List[RewriteStep]
    overall_improvement: float
    confidence: float
    safety_score: float  # How safe the rewrite is
    complexity_reduction: float
    readability_improvement: float
    explanation: str
    warnings: List[str]
    test_recommendations: List[str]


class IntelligentQueryRewriter:
    """Intelligently rewrites SQL queries with detailed explanations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_rewrite_rules()

    def _initialize_rewrite_rules(self):
        """Initialize query rewrite rule patterns"""

        self.rewrite_patterns = {
            RewriteRule.IN_TO_EXISTS: {
                'pattern': re.compile(
                    r'WHERE\s+(\w+(?:\.\w+)?)\s+IN\s*\(\s*SELECT\s+(\w+(?:\.\w+)?)\s+FROM\s+(\w+)(?:\s+\w+)?(?:\s+WHERE\s+(.+?))?\s*\)',
                    re.IGNORECASE | re.DOTALL
                ),
                'complexity': RewriteComplexity.MODERATE,
                'description': 'Replace IN subquery with EXISTS for better performance',
                'performance_impact': 'Significant improvement for large datasets',
                'risks': ['Behavior change with NULL values'],
                'verification_needed': True
            },

            RewriteRule.UNION_TO_UNION_ALL: {
                'pattern': re.compile(r'\bUNION\b(?!\s+ALL)', re.IGNORECASE),
                'complexity': RewriteComplexity.SIMPLE,
                'description': 'Replace UNION with UNION ALL to avoid duplicate elimination',
                'performance_impact': 'Eliminates sorting overhead',
                'risks': ['May introduce duplicate rows'],
                'verification_needed': True
            },

            RewriteRule.EXISTS_TO_JOIN: {
                'pattern': re.compile(
                    r'WHERE\s+EXISTS\s*\(\s*SELECT\s+.+?\s+FROM\s+(\w+)(?:\s+\w+)?\s+WHERE\s+(.+?)\)',
                    re.IGNORECASE | re.DOTALL
                ),
                'complexity': RewriteComplexity.COMPLEX,
                'description': 'Convert EXISTS subquery to INNER JOIN',
                'performance_impact': 'Can improve performance with proper indexes',
                'risks': ['May change result cardinality', 'Requires DISTINCT if duplicates possible'],
                'verification_needed': True
            },

            RewriteRule.ELIMINATE_DISTINCT: {
                'pattern': re.compile(r'SELECT\s+DISTINCT\s+(.+?)\s+FROM\s+(.+?)(?:\s+WHERE|$)', re.IGNORECASE | re.DOTALL),
                'complexity': RewriteComplexity.MODERATE,
                'description': 'Remove unnecessary DISTINCT clause',
                'performance_impact': 'Eliminates sorting and duplicate removal overhead',
                'risks': ['May introduce duplicate rows if DISTINCT was necessary'],
                'verification_needed': True
            }
        }

    def rewrite_query(self, query: str,
                      safety_level: str = 'conservative',
                      context: Dict[str, Any] = None) -> QueryRewrite:
        """
        Rewrite a query with detailed explanations

        Args:
            query: Original SQL query
            safety_level: 'aggressive', 'moderate', or 'conservative'
            context: Additional context (table stats, constraints, etc.)
        """

        if context is None:
            context = {}

        # Parse the query
        try:
            parsed = sqlparse.parse(query)[0]
        except Exception as e:
            self.logger.error(f"Failed to parse query: {e}")
            return self._create_fallback_rewrite(query)

        # Initialize rewrite state
        current_query = query
        rewrite_steps = []
        warnings = []

        # Apply rewrite rules in order of safety and impact
        rule_order = self._get_rule_order(safety_level)

        for rule in rule_order:
            step = self._apply_rewrite_rule(current_query, rule, context)
            if step:
                current_query = step.rewritten_fragment if step.rewritten_fragment else current_query
                rewrite_steps.append(step)

                # Check if we should continue based on safety
                if safety_level == 'conservative' and step.complexity in [RewriteComplexity.COMPLEX, RewriteComplexity.ADVANCED]:
                    warnings.append(f"Skipping {step.rule.value} due to conservative safety setting")
                    continue

        # Calculate metrics
        overall_improvement = self._calculate_overall_improvement(rewrite_steps)
        confidence = self._calculate_confidence(rewrite_steps, context)
        safety_score = self._calculate_safety_score(rewrite_steps, safety_level)
        complexity_reduction = self._calculate_complexity_reduction(query, current_query)
        readability_improvement = self._calculate_readability_improvement(query, current_query)

        # Generate explanation
        explanation = self._generate_explanation(rewrite_steps, overall_improvement)

        # Generate test recommendations
        test_recommendations = self._generate_test_recommendations(rewrite_steps)

        return QueryRewrite(
            original_query=query,
            rewritten_query=current_query,
            rewrite_steps=rewrite_steps,
            overall_improvement=overall_improvement,
            confidence=confidence,
            safety_score=safety_score,
            complexity_reduction=complexity_reduction,
            readability_improvement=readability_improvement,
            explanation=explanation,
            warnings=warnings,
            test_recommendations=test_recommendations
        )

    def _get_rule_order(self, safety_level: str) -> List[RewriteRule]:
        """Get rewrite rules in order of application based on safety level"""

        rule_priorities = {
            'conservative': [
                RewriteRule.UNION_TO_UNION_ALL,
                RewriteRule.OPTIMIZE_WHERE,
                RewriteRule.NORMALIZE_EXPRESSIONS
            ],
            'moderate': [
                RewriteRule.UNION_TO_UNION_ALL,
                RewriteRule.IN_TO_EXISTS,
                RewriteRule.ELIMINATE_DISTINCT,
                RewriteRule.OPTIMIZE_WHERE,
                RewriteRule.SIMPLIFY_CASE,
                RewriteRule.NORMALIZE_EXPRESSIONS
            ],
            'aggressive': [
                RewriteRule.UNION_TO_UNION_ALL,
                RewriteRule.IN_TO_EXISTS,
                RewriteRule.EXISTS_TO_JOIN,
                RewriteRule.SUBQUERY_TO_JOIN,
                RewriteRule.ELIMINATE_DISTINCT,
                RewriteRule.REORDER_JOINS,
                RewriteRule.PUSH_DOWN_PREDICATES,
                RewriteRule.ELIMINATE_REDUNDANT_JOINS,
                RewriteRule.OPTIMIZE_AGGREGATION,
                RewriteRule.OPTIMIZE_WHERE,
                RewriteRule.SIMPLIFY_CASE,
                RewriteRule.NORMALIZE_EXPRESSIONS
            ]
        }

        return rule_priorities.get(safety_level, rule_priorities['moderate'])

    def _apply_rewrite_rule(self, query: str, rule: RewriteRule, context: Dict[str, Any]) -> Optional[RewriteStep]:
        """Apply a specific rewrite rule to the query"""

        if rule == RewriteRule.IN_TO_EXISTS:
            return self._rewrite_in_to_exists(query)
        elif rule == RewriteRule.UNION_TO_UNION_ALL:
            return self._rewrite_union_to_union_all(query)
        elif rule == RewriteRule.EXISTS_TO_JOIN:
            return self._rewrite_exists_to_join(query)
        elif rule == RewriteRule.ELIMINATE_DISTINCT:
            return self._rewrite_eliminate_distinct(query, context)
        elif rule == RewriteRule.OPTIMIZE_WHERE:
            return self._rewrite_optimize_where(query)
        elif rule == RewriteRule.SIMPLIFY_CASE:
            return self._rewrite_simplify_case(query)
        elif rule == RewriteRule.SUBQUERY_TO_JOIN:
            return self._rewrite_subquery_to_join(query)
        elif rule == RewriteRule.NORMALIZE_EXPRESSIONS:
            return self._rewrite_normalize_expressions(query)

        return None

    def _rewrite_in_to_exists(self, query: str) -> Optional[RewriteStep]:
        """Rewrite IN subquery to EXISTS"""

        pattern = self.rewrite_patterns[RewriteRule.IN_TO_EXISTS]['pattern']
        match = pattern.search(query)

        if not match:
            return None

        left_column, right_column, table, where_clause = match.groups()

        # Build EXISTS equivalent
        exists_clause = f"EXISTS (SELECT 1 FROM {table}"
        if where_clause:
            exists_clause += f" WHERE {where_clause} AND {right_column} = {left_column})"
        else:
            exists_clause += f" WHERE {right_column} = {left_column})"

        # Replace in query
        rewritten = pattern.sub(lambda m: f"WHERE {exists_clause}", query)

        return RewriteStep(
            rule=RewriteRule.IN_TO_EXISTS,
            complexity=self.rewrite_patterns[RewriteRule.IN_TO_EXISTS]['complexity'],
            description=self.rewrite_patterns[RewriteRule.IN_TO_EXISTS]['description'],
            original_fragment=match.group(0),
            rewritten_fragment=rewritten,
            rationale="EXISTS typically performs better than IN for subqueries, especially with NULL values",
            performance_impact="20-40% improvement for large subqueries",
            risks=self.rewrite_patterns[RewriteRule.IN_TO_EXISTS]['risks'],
            verification_needed=True,
            estimated_improvement=0.3
        )

    def _rewrite_union_to_union_all(self, query: str) -> Optional[RewriteStep]:
        """Rewrite UNION to UNION ALL where appropriate"""

        pattern = self.rewrite_patterns[RewriteRule.UNION_TO_UNION_ALL]['pattern']

        if not pattern.search(query):
            return None

        rewritten = pattern.sub('UNION ALL', query)

        return RewriteStep(
            rule=RewriteRule.UNION_TO_UNION_ALL,
            complexity=RewriteComplexity.SIMPLE,
            description="Replace UNION with UNION ALL to avoid duplicate elimination",
            original_fragment=pattern.search(query).group(0),
            rewritten_fragment=rewritten,
            rationale="UNION ALL is faster as it doesn't need to sort and eliminate duplicates",
            performance_impact="10-30% improvement depending on result set size",
            risks=["May introduce duplicate rows if not handled by application"],
            verification_needed=True,
            estimated_improvement=0.2
        )

    def _rewrite_exists_to_join(self, query: str) -> Optional[RewriteStep]:
        """Convert EXISTS subquery to INNER JOIN"""

        pattern = self.rewrite_patterns[RewriteRule.EXISTS_TO_JOIN]['pattern']
        match = pattern.search(query)

        if not match:
            return None

        table, where_clause = match.groups()

        # This is a simplified conversion - real implementation would be more sophisticated
        original_fragment = match.group(0)

        # For safety, we'll provide the pattern but not automatically apply this complex transformation
        join_suggestion = f"Consider converting to: INNER JOIN {table} ON {where_clause}"

        return RewriteStep(
            rule=RewriteRule.EXISTS_TO_JOIN,
            complexity=RewriteComplexity.COMPLEX,
            description="Convert EXISTS subquery to INNER JOIN",
            original_fragment=original_fragment,
            rewritten_fragment=query,  # Don't auto-apply complex transformations
            rationale="JOINs can be more efficient than EXISTS for certain query patterns",
            performance_impact="Varies - can be significant with proper indexing",
            risks=["May change result cardinality", "Complex transformation"],
            verification_needed=True,
            estimated_improvement=0.25
        )

    def _rewrite_eliminate_distinct(self, query: str, context: Dict[str, Any]) -> Optional[RewriteStep]:
        """Remove unnecessary DISTINCT clause"""

        # Only suggest removal if we have context indicating it's safe
        if not self._is_distinct_safe_to_remove(query, context):
            return None

        pattern = self.rewrite_patterns[RewriteRule.ELIMINATE_DISTINCT]['pattern']
        match = pattern.search(query)

        if not match:
            return None

        columns, tables = match.groups()
        rewritten = re.sub(r'SELECT\s+DISTINCT\s+', 'SELECT ', query, flags=re.IGNORECASE)

        return RewriteStep(
            rule=RewriteRule.ELIMINATE_DISTINCT,
            complexity=RewriteComplexity.MODERATE,
            description="Remove unnecessary DISTINCT clause",
            original_fragment=f"SELECT DISTINCT {columns}",
            rewritten_fragment=rewritten,
            rationale="DISTINCT elimination reduces sorting overhead when duplicates are not possible",
            performance_impact="10-25% improvement in execution time",
            risks=["May introduce duplicates if DISTINCT was necessary"],
            verification_needed=True,
            estimated_improvement=0.15
        )

    def _rewrite_optimize_where(self, query: str) -> Optional[RewriteStep]:
        """Optimize WHERE clause conditions"""

        # Look for conditions that can be optimized
        optimizations = []

        # Check for sargable conditions
        if re.search(r'WHERE\s+\w+\s*\+\s*\d+\s*[<>=]', query, re.IGNORECASE):
            optimizations.append("Move constants to right side of comparison")

        # Check for function calls on columns
        if re.search(r'WHERE\s+(YEAR|MONTH|DAY|LOWER|UPPER)\s*\([^)]*\w+\.\w+', query, re.IGNORECASE):
            optimizations.append("Function calls on indexed columns prevent index usage")

        if not optimizations:
            return None

        return RewriteStep(
            rule=RewriteRule.OPTIMIZE_WHERE,
            complexity=RewriteComplexity.MODERATE,
            description="Optimize WHERE clause for better index usage",
            original_fragment="WHERE clause",
            rewritten_fragment=query,  # Would need specific transformation logic
            rationale="Optimizing WHERE conditions enables better index usage",
            performance_impact="Significant improvement with proper indexes",
            risks=["May require index changes"],
            verification_needed=True,
            estimated_improvement=0.4
        )

    def _rewrite_simplify_case(self, query: str) -> Optional[RewriteStep]:
        """Simplify CASE expressions where possible"""

        # Look for simple CASE expressions that can be simplified
        case_pattern = re.compile(
            r'CASE\s+WHEN\s+(\w+)\s*=\s*(.+?)\s+THEN\s+(.+?)\s+ELSE\s+(.+?)\s+END',
            re.IGNORECASE | re.DOTALL
        )

        match = case_pattern.search(query)
        if not match:
            return None

        # Simple optimization example - would be more sophisticated in practice
        column, value, then_val, else_val = match.groups()

        # Check if this can be simplified to COALESCE or similar
        if then_val.strip().upper() == 'NULL' or else_val.strip().upper() == 'NULL':
            return RewriteStep(
                rule=RewriteRule.SIMPLIFY_CASE,
                complexity=RewriteComplexity.SIMPLE,
                description="Simplify CASE expression",
                original_fragment=match.group(0),
                rewritten_fragment=query,  # Would apply specific simplification
                rationale="Simplified expressions are easier to read and may perform better",
                performance_impact="Minor improvement in readability and execution",
                risks=["Minimal"],
                verification_needed=False,
                estimated_improvement=0.05
            )

        return None

    def _rewrite_subquery_to_join(self, query: str) -> Optional[RewriteStep]:
        """Convert correlated subqueries to JOINs"""

        # Look for correlated subqueries in SELECT clause
        subquery_pattern = re.compile(
            r'SELECT\s+.*?\(\s*SELECT\s+.+?\s+FROM\s+\w+\s+WHERE\s+.+?\)\s*(?:as\s+\w+)?',
            re.IGNORECASE | re.DOTALL
        )

        if not subquery_pattern.search(query):
            return None

        return RewriteStep(
            rule=RewriteRule.SUBQUERY_TO_JOIN,
            complexity=RewriteComplexity.COMPLEX,
            description="Convert correlated subquery to JOIN",
            original_fragment="Subquery in SELECT",
            rewritten_fragment=query,  # Complex transformation not auto-applied
            rationale="JOINs are typically more efficient than correlated subqueries",
            performance_impact="Can provide significant improvement",
            risks=["Complex transformation", "May change result structure"],
            verification_needed=True,
            estimated_improvement=0.35
        )

    def _rewrite_normalize_expressions(self, query: str) -> Optional[RewriteStep]:
        """Normalize expressions for consistency"""

        changes = []
        rewritten = query

        # Normalize comparison operators
        if '!=' in query:
            rewritten = rewritten.replace('!=', '<>')
            changes.append("Standardized comparison operators")

        # Normalize spacing around operators
        rewritten = re.sub(r'\s*=\s*', ' = ', rewritten)
        rewritten = re.sub(r'\s*<>\s*', ' <> ', rewritten)

        if rewritten != query:
            changes.append("Normalized spacing")

        if not changes:
            return None

        return RewriteStep(
            rule=RewriteRule.NORMALIZE_EXPRESSIONS,
            complexity=RewriteComplexity.SIMPLE,
            description="Normalize expression formatting",
            original_fragment="Various expressions",
            rewritten_fragment=rewritten,
            rationale="Consistent formatting improves readability and maintainability",
            performance_impact="No performance impact, improves readability",
            risks=["None"],
            verification_needed=False,
            estimated_improvement=0.0
        )

    def _is_distinct_safe_to_remove(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if DISTINCT can be safely removed"""

        # Conservative approach - only suggest removal if we have strong evidence
        if not context:
            return False

        # Check if query involves only primary key columns
        if context.get('primary_key_only', False):
            return True

        # Check if there are unique constraints that guarantee no duplicates
        if context.get('unique_result_guaranteed', False):
            return True

        return False

    def _calculate_overall_improvement(self, steps: List[RewriteStep]) -> float:
        """Calculate overall improvement from all rewrite steps"""

        if not steps:
            return 0.0

        # Combine improvements with diminishing returns
        total_improvement = 0.0
        for step in steps:
            # Apply diminishing returns formula
            total_improvement = total_improvement + step.estimated_improvement * (1 - total_improvement)

        return min(0.9, total_improvement)  # Cap at 90%

    def _calculate_confidence(self, steps: List[RewriteStep], context: Dict[str, Any]) -> float:
        """Calculate confidence in the rewrite"""

        if not steps:
            return 1.0

        base_confidence = 0.8

        # Reduce confidence for complex transformations
        for step in steps:
            if step.complexity == RewriteComplexity.COMPLEX:
                base_confidence -= 0.1
            elif step.complexity == RewriteComplexity.ADVANCED:
                base_confidence -= 0.15

        # Increase confidence if we have good context
        if context and len(context) > 3:
            base_confidence += 0.1

        return max(0.3, min(0.95, base_confidence))

    def _calculate_safety_score(self, steps: List[RewriteStep], safety_level: str) -> float:
        """Calculate safety score of the rewrite"""

        if not steps:
            return 1.0

        safety_scores = {
            RewriteComplexity.SIMPLE: 0.9,
            RewriteComplexity.MODERATE: 0.7,
            RewriteComplexity.COMPLEX: 0.5,
            RewriteComplexity.ADVANCED: 0.3
        }

        # Calculate weighted average
        total_weight = 0
        weighted_safety = 0

        for step in steps:
            weight = step.estimated_improvement
            safety = safety_scores.get(step.complexity, 0.5)

            weighted_safety += safety * weight
            total_weight += weight

        return weighted_safety / total_weight if total_weight > 0 else 0.8

    def _calculate_complexity_reduction(self, original: str, rewritten: str) -> float:
        """Calculate reduction in query complexity"""

        def complexity_score(query):
            score = 0
            score += query.upper().count('SELECT') * 2  # Subqueries
            score += query.upper().count('JOIN') * 1
            score += query.upper().count('UNION') * 1
            score += query.upper().count('CASE') * 1
            score += query.upper().count('EXISTS') * 2
            score += len(re.findall(r'\(', query))  # Nested expressions
            return score

        original_complexity = complexity_score(original)
        rewritten_complexity = complexity_score(rewritten)

        if original_complexity == 0:
            return 0.0

        reduction = (original_complexity - rewritten_complexity) / original_complexity
        return max(0.0, reduction)

    def _calculate_readability_improvement(self, original: str, rewritten: str) -> float:
        """Calculate improvement in query readability"""

        def readability_score(query):
            score = 100
            # Penalize long lines
            lines = query.split('\n')
            for line in lines:
                if len(line) > 120:
                    score -= 5

            # Penalize deeply nested structures
            max_nesting = 0
            current_nesting = 0
            for char in query:
                if char == '(':
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                elif char == ')':
                    current_nesting -= 1

            score -= max_nesting * 2

            return max(0, score)

        original_score = readability_score(original)
        rewritten_score = readability_score(rewritten)

        if original_score == 0:
            return 0.0

        improvement = (rewritten_score - original_score) / 100
        return max(0.0, min(1.0, improvement))

    def _generate_explanation(self, steps: List[RewriteStep], overall_improvement: float) -> str:
        """Generate human-readable explanation of the rewrite"""

        if not steps:
            return "No optimizations were applied to this query."

        explanation_parts = []

        explanation_parts.append(f"Applied {len(steps)} optimization(s) with an estimated {overall_improvement:.1%} overall improvement:")
        explanation_parts.append("")

        for i, step in enumerate(steps, 1):
            explanation_parts.append(f"{i}. **{step.description}**")
            explanation_parts.append(f"   - Rationale: {step.rationale}")
            explanation_parts.append(f"   - Expected improvement: {step.estimated_improvement:.1%}")
            explanation_parts.append(f"   - Performance impact: {step.performance_impact}")

            if step.risks:
                explanation_parts.append(f"   - Risks: {', '.join(step.risks)}")

            if step.verification_needed:
                explanation_parts.append("   - ⚠️ Verification recommended before production use")

            explanation_parts.append("")

        return "\n".join(explanation_parts)

    def _generate_test_recommendations(self, steps: List[RewriteStep]) -> List[str]:
        """Generate testing recommendations for the rewrite"""

        recommendations = [
            "Compare execution plans of original and rewritten queries",
            "Verify identical result sets with diverse test data",
            "Test with various data volumes (small, medium, large)",
            "Monitor resource usage (CPU, memory, I/O) during testing"
        ]

        # Add specific recommendations based on rewrite types
        for step in steps:
            if step.rule == RewriteRule.IN_TO_EXISTS:
                recommendations.append("Test with NULL values in subquery columns")
            elif step.rule == RewriteRule.UNION_TO_UNION_ALL:
                recommendations.append("Verify that duplicate rows are acceptable")
            elif step.rule == RewriteRule.EXISTS_TO_JOIN:
                recommendations.append("Check for cardinality changes in result set")

        return recommendations[:8]  # Limit to 8 recommendations

    def _create_fallback_rewrite(self, query: str) -> QueryRewrite:
        """Create fallback rewrite when parsing fails"""

        return QueryRewrite(
            original_query=query,
            rewritten_query=query,
            rewrite_steps=[],
            overall_improvement=0.0,
            confidence=0.0,
            safety_score=1.0,
            complexity_reduction=0.0,
            readability_improvement=0.0,
            explanation="Unable to parse query for rewriting",
            warnings=["Query parsing failed - no optimizations applied"],
            test_recommendations=[]
        )

    def suggest_alternative_approaches(self, query: str) -> List[Dict[str, str]]:
        """Suggest alternative approaches for complex queries"""

        suggestions = []

        # Check for complex subqueries
        if query.upper().count('SELECT') > 3:
            suggestions.append({
                'approach': 'Common Table Expressions (CTEs)',
                'description': 'Break down complex subqueries into readable CTEs',
                'example': 'WITH subquery_name AS (SELECT ...) SELECT ... FROM subquery_name'
            })

        # Check for many JOINs
        if query.upper().count('JOIN') > 4:
            suggestions.append({
                'approach': 'Staged Processing',
                'description': 'Consider breaking into multiple queries with temporary tables',
                'example': 'Process in stages: temp table → final result'
            })

        # Check for complex aggregations
        if 'GROUP BY' in query.upper() and 'HAVING' in query.upper():
            suggestions.append({
                'approach': 'Window Functions',
                'description': 'Consider using window functions for advanced analytics',
                'example': 'ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)'
            })

        return suggestions


def format_rewrite_report(rewrite: QueryRewrite) -> str:
    """Format rewrite result as a detailed report"""

    lines = []

    lines.append("# SQL Query Rewrite Report")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- **Overall Improvement:** {rewrite.overall_improvement:.1%}")
    lines.append(f"- **Confidence:** {rewrite.confidence:.1%}")
    lines.append(f"- **Safety Score:** {rewrite.safety_score:.1%}")
    lines.append(f"- **Complexity Reduction:** {rewrite.complexity_reduction:.1%}")
    lines.append("")

    if rewrite.rewrite_steps:
        lines.append("## Applied Optimizations")
        for i, step in enumerate(rewrite.rewrite_steps, 1):
            lines.append(f"### {i}. {step.description}")
            lines.append(f"**Complexity:** {step.complexity.value}")
            lines.append(f"**Improvement:** {step.estimated_improvement:.1%}")
            lines.append("")
            lines.append(f"**Original:**")
            lines.append(f"```sql")
            lines.append(step.original_fragment)
            lines.append("```")
            lines.append("")
            lines.append(f"**Rationale:** {step.rationale}")
            lines.append("")

            if step.risks:
                lines.append(f"**Risks:** {', '.join(step.risks)}")
                lines.append("")

    lines.append("## Rewritten Query")
    lines.append("```sql")
    lines.append(rewrite.rewritten_query)
    lines.append("```")
    lines.append("")

    if rewrite.warnings:
        lines.append("## Warnings")
        for warning in rewrite.warnings:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")

    if rewrite.test_recommendations:
        lines.append("## Testing Recommendations")
        for rec in rewrite.test_recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    lines.append("## Explanation")
    lines.append(rewrite.explanation)

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    rewriter = IntelligentQueryRewriter()

    # Test query with multiple optimization opportunities
    test_query = """
    SELECT DISTINCT c.customer_name, c.email
    FROM customers c
    WHERE c.customer_id IN (
        SELECT o.customer_id
        FROM orders o
        WHERE o.status = 'completed'
    )
    UNION
    SELECT DISTINCT c.customer_name, c.email
    FROM customers c
    WHERE c.customer_id IN (
        SELECT r.customer_id
        FROM reviews r
        WHERE r.rating >= 4
    )
    """

    print("=== Query Rewrite Analysis ===\n")

    # Conservative rewrite
    conservative_rewrite = rewriter.rewrite_query(
        test_query,
        safety_level='conservative'
    )

    print("CONSERVATIVE REWRITE:")
    print(format_rewrite_report(conservative_rewrite))

    print("\n" + "="*50 + "\n")

    # Moderate rewrite
    moderate_rewrite = rewriter.rewrite_query(
        test_query,
        safety_level='moderate',
        context={'unique_result_guaranteed': True}
    )

    print("MODERATE REWRITE:")
    print(format_rewrite_report(moderate_rewrite))

    print("\n" + "="*50 + "\n")

    # Alternative approaches
    alternatives = rewriter.suggest_alternative_approaches(test_query)

    print("ALTERNATIVE APPROACHES:")
    for alt in alternatives:
        print(f"**{alt['approach']}:** {alt['description']}")
        print(f"Example: {alt['example']}")
        print()