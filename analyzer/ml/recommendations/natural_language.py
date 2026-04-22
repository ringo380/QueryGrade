"""
Natural Language Feedback Generation System

This module generates human-readable, contextual feedback for SQL queries
using natural language generation techniques.
"""

import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class FeedbackTone(Enum):
    """Tone of the feedback message"""

    ENCOURAGING = "encouraging"
    NEUTRAL = "neutral"
    CONSTRUCTIVE = "constructive"
    URGENT = "urgent"
    EDUCATIONAL = "educational"


class FeedbackLevel(Enum):
    """Technical level of feedback"""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class FeedbackCategory(Enum):
    """Categories of feedback"""

    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"
    STYLE = "style"
    SECURITY = "security"
    BEST_PRACTICE = "best_practice"
    OPTIMIZATION = "optimization"


@dataclass
class FeedbackMessage:
    """Represents a single feedback message"""

    category: FeedbackCategory
    severity: str  # critical, high, medium, low
    title: str
    description: str
    impact: str
    recommendation: str
    example: Optional[str] = None
    learn_more_url: Optional[str] = None
    confidence: float = 0.8
    technical_details: Optional[str] = None


@dataclass
class ComprehensiveFeedback:
    """Complete feedback package for a query"""

    overall_grade: str  # A-F
    overall_score: float  # 0-100
    summary: str
    strengths: List[str]
    improvements: List[FeedbackMessage]
    quick_wins: List[str]  # Easy fixes with high impact
    long_term_suggestions: List[str]
    learning_resources: List[Dict[str, str]]
    encouragement: str
    next_steps: List[str]
    comparative_analysis: Optional[str] = None
    estimated_improvement: Optional[str] = None


class NaturalLanguageFeedbackGenerator:
    """Generates natural language feedback for SQL queries"""

    def __init__(self, user_level: FeedbackLevel = FeedbackLevel.INTERMEDIATE):
        self.user_level = user_level
        self.logger = logging.getLogger(__name__)
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize feedback message templates"""

        self.grade_descriptions = {
            "A": "Excellent query! Well-optimized and following best practices.",
            "B": "Good query with minor room for improvements.",
            "C": "Decent query but has several optimization opportunities.",
            "D": "Query needs significant improvements for better performance.",
            "F": "Query has critical issues that need immediate attention.",
        }

        self.encouragement_templates = {
            "A": [
                "Outstanding work! Your query demonstrates excellent SQL skills.",
                "Fantastic query! You're writing production-ready SQL.",
                "Impressive! This query is highly optimized and maintainable.",
            ],
            "B": [
                "Good job! With a few tweaks, this query could be excellent.",
                "Well done! You're on the right track with this query.",
                "Nice work! Consider the suggestions to make it even better.",
            ],
            "C": [
                "You're making progress! Let's work on optimizing this query.",
                "Good effort! The suggestions below will significantly improve performance.",
                "Keep going! You have the basics right, now let's refine it.",
            ],
            "D": [
                "Don't worry, we all start somewhere! Let's improve this step by step.",
                "Learning opportunity ahead! These changes will make a big difference.",
                "You're learning! Focus on the high-priority improvements first.",
            ],
            "F": [
                "Let's work together to fix the critical issues first.",
                "Don't be discouraged! Every expert started as a beginner.",
                "This is a great learning opportunity. Let's tackle it systematically.",
            ],
        }

        self.strength_templates = {
            "proper_joins": "✓ Properly structured JOIN conditions",
            "indexed_columns": "✓ Using indexed columns in WHERE clause",
            "limit_usage": "✓ Appropriately limiting result set",
            "column_specification": "✓ Explicitly specifying needed columns",
            "consistent_style": "✓ Consistent formatting and naming",
            "parameterization": "✓ Using parameterized queries for security",
            "aggregation_efficiency": "✓ Efficient use of aggregation functions",
            "proper_grouping": "✓ Correct GROUP BY implementation",
        }

        self.impact_templates = {
            "high": "This could improve performance by {estimate}",
            "medium": "This change would moderately improve {aspect}",
            "low": "Minor improvement to {aspect}",
            "critical": "Critical issue affecting {aspect}",
        }

    def generate_feedback(
        self, analysis_results: Dict[str, Any]
    ) -> ComprehensiveFeedback:
        """Generate comprehensive natural language feedback"""

        # Extract key metrics
        overall_score = analysis_results.get("overall_score", 50)
        anti_patterns = analysis_results.get("anti_patterns", [])
        optimizations = analysis_results.get("optimizations", [])

        # Calculate grade
        grade = self._calculate_grade(overall_score)

        # Generate summary
        summary = self._generate_summary(grade, overall_score, analysis_results)

        # Identify strengths
        strengths = self._identify_strengths(analysis_results)

        # Generate improvement messages
        improvements = self._generate_improvements(anti_patterns, optimizations)

        # Identify quick wins
        quick_wins = self._identify_quick_wins(improvements)

        # Generate long-term suggestions
        long_term = self._generate_long_term_suggestions(analysis_results)

        # Select encouragement message
        encouragement = self._select_encouragement(grade)

        # Generate next steps
        next_steps = self._generate_next_steps(improvements, grade)

        # Generate comparative analysis if available
        comparative = self._generate_comparative_analysis(analysis_results)

        # Estimate improvement potential
        improvement_estimate = self._estimate_improvement(improvements)

        # Compile learning resources
        resources = self._compile_learning_resources(improvements, analysis_results)

        return ComprehensiveFeedback(
            overall_grade=grade,
            overall_score=overall_score,
            summary=summary,
            strengths=strengths,
            improvements=improvements[:10],  # Limit to top 10
            quick_wins=quick_wins[:5],  # Top 5 quick wins
            long_term_suggestions=long_term[:3],  # Top 3 long-term
            learning_resources=resources,
            encouragement=encouragement,
            next_steps=next_steps[:5],  # Top 5 next steps
            comparative_analysis=comparative,
            estimated_improvement=improvement_estimate,
        )

    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score"""
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

    def _generate_summary(
        self, grade: str, score: float, analysis: Dict[str, Any]
    ) -> str:
        """Generate executive summary of the analysis"""

        summary_parts = []

        # Grade description
        summary_parts.append(self.grade_descriptions[grade])

        # Key statistics
        if "query_type" in analysis:
            summary_parts.append(
                f"This {analysis['query_type']} query scores {score:.1f}/100."
            )

        # Major issues
        anti_patterns = analysis.get("anti_patterns", [])
        critical_count = sum(
            1 for ap in anti_patterns if ap.get("severity") == "critical"
        )
        high_count = sum(1 for ap in anti_patterns if ap.get("severity") == "high")

        if critical_count > 0:
            summary_parts.append(
                f"Found {critical_count} critical issue(s) requiring immediate attention."
            )
        elif high_count > 0:
            summary_parts.append(
                f"Identified {high_count} high-priority optimization opportunities."
            )

        # Performance estimate
        if "estimated_execution_time" in analysis:
            exec_time = analysis["estimated_execution_time"]
            if exec_time > 1000:
                summary_parts.append(
                    f"Current estimated execution time: {exec_time/1000:.1f} seconds."
                )
            else:
                summary_parts.append(
                    f"Current estimated execution time: {exec_time:.0f}ms."
                )

        return " ".join(summary_parts)

    def _identify_strengths(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify what the query does well"""

        strengths = []

        # Check for good practices
        if analysis.get("uses_indexes", False):
            strengths.append(self.strength_templates["indexed_columns"])

        if analysis.get("has_proper_joins", False):
            strengths.append(self.strength_templates["proper_joins"])

        if analysis.get("has_limit", False):
            strengths.append(self.strength_templates["limit_usage"])

        if not analysis.get("uses_select_star", False):
            strengths.append(self.strength_templates["column_specification"])

        if analysis.get("consistent_style", False):
            strengths.append(self.strength_templates["consistent_style"])

        if analysis.get("uses_parameterization", False):
            strengths.append(self.strength_templates["parameterization"])

        # Add custom strengths based on patterns
        patterns = analysis.get("matched_patterns", [])
        for pattern in patterns:
            if pattern.get("quality") == "best_practice":
                strengths.append(f"✓ {pattern.get('name', 'Good pattern usage')}")

        return strengths[:8]  # Limit to 8 strengths

    def _generate_improvements(
        self, anti_patterns: List[Dict[str, Any]], optimizations: List[Dict[str, Any]]
    ) -> List[FeedbackMessage]:
        """Generate improvement feedback messages"""

        improvements = []

        # Process anti-patterns
        for ap in anti_patterns:
            message = self._create_feedback_message(
                category=self._map_to_category(ap.get("category", "performance")),
                severity=ap.get("severity", "medium"),
                title=ap.get("name", "Issue detected"),
                description=self._elaborate_description(ap),
                impact=ap.get("impact", "May affect performance"),
                recommendation=ap.get(
                    "recommendation", "Consider reviewing this pattern"
                ),
                example=ap.get("example_fix"),
                confidence=ap.get("confidence", 0.8),
            )
            improvements.append(message)

        # Process optimization opportunities
        for opt in optimizations:
            message = self._create_feedback_message(
                category=FeedbackCategory.OPTIMIZATION,
                severity=opt.get("priority", "medium"),
                title=opt.get("title", "Optimization opportunity"),
                description=opt.get("description", ""),
                impact=self._format_impact(opt),
                recommendation=opt.get("suggestion", ""),
                example=opt.get("example"),
                confidence=opt.get("confidence", 0.7),
            )
            improvements.append(message)

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        improvements.sort(key=lambda x: severity_order.get(x.severity, 99))

        return improvements

    def _create_feedback_message(self, **kwargs) -> FeedbackMessage:
        """Create a feedback message with proper formatting"""

        # Adjust language based on user level
        if self.user_level == FeedbackLevel.BEGINNER:
            kwargs["description"] = self._simplify_language(
                kwargs.get("description", "")
            )
            kwargs["technical_details"] = None
        elif self.user_level == FeedbackLevel.EXPERT:
            # Add more technical details for experts
            kwargs["technical_details"] = self._add_technical_details(kwargs)

        return FeedbackMessage(**kwargs)

    def _elaborate_description(self, anti_pattern: Dict[str, Any]) -> str:
        """Elaborate on anti-pattern description based on user level"""

        base_description = anti_pattern.get("description", "")

        if self.user_level == FeedbackLevel.BEGINNER:
            # Add simple explanation
            explanations = {
                "SELECT * Usage": "When you use SELECT *, the database fetches every column even if you only need a few. This wastes resources and slows things down.",  # noqa: E501
                "Missing Index": "Without an index, the database has to check every single row, like searching a phone book without alphabetical order.",  # noqa: E501
                "Nested Subqueries": "Too many queries inside queries make it hard for the database to optimize and for humans to understand.",  # noqa: E501
            }

            pattern_name = anti_pattern.get("name", "")
            if pattern_name in explanations:
                return f"{base_description} {explanations[pattern_name]}"

        elif self.user_level == FeedbackLevel.EXPERT:
            # Add technical details
            return f"{base_description} (Pattern ID: {anti_pattern.get('pattern_id', 'N/A')})"

        return base_description

    def _format_impact(self, optimization: Dict[str, Any]) -> str:
        """Format impact statement"""

        impact_type = optimization.get("impact_type", "performance")
        estimate = optimization.get("estimated_improvement", 0)

        if estimate > 0.5:
            return f"Could improve {impact_type} by {estimate*100:.0f}%"
        elif estimate > 0.2:
            return f"Moderate improvement to {impact_type} expected"
        else:
            return f"Minor {impact_type} improvement"

    def _identify_quick_wins(self, improvements: List[FeedbackMessage]) -> List[str]:
        """Identify easy fixes with high impact"""

        quick_wins = []

        for imp in improvements:
            # High impact, easy to implement
            if imp.severity in ["high", "critical"] and self._is_easy_fix(imp):
                quick_wins.append(f"🎯 {imp.title}: {imp.recommendation}")

        # Add some universal quick wins if applicable
        universal_wins = [
            "🎯 Add LIMIT clause if fetching for display",
            "🎯 Replace SELECT * with specific columns",
            "🎯 Add indexes to frequently filtered columns",
            "🎯 Use JOIN instead of nested subqueries",
            "🎯 Replace NOT IN with NOT EXISTS for better null handling",
        ]

        # Add relevant universal wins
        for win in universal_wins:
            if len(quick_wins) < 5 and self._is_relevant_win(win, improvements):
                quick_wins.append(win)

        return quick_wins

    def _is_easy_fix(self, improvement: FeedbackMessage) -> bool:
        """Determine if a fix is easy to implement"""

        easy_patterns = [
            "SELECT *",
            "Missing LIMIT",
            "Missing alias",
            "Inconsistent case",
            "Magic numbers",
        ]

        return any(pattern in improvement.title for pattern in easy_patterns)

    def _is_relevant_win(self, win: str, improvements: List[FeedbackMessage]) -> bool:
        """Check if a universal win is relevant to current issues"""

        # Simple relevance check based on keywords
        win_lower = win.lower()
        for imp in improvements:
            if any(keyword in win_lower for keyword in imp.title.lower().split()):
                return False  # Already covered

        return True

    def _generate_long_term_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate strategic long-term improvements"""

        suggestions = []

        # Based on overall patterns
        if analysis.get("avg_query_complexity", 0) > 7:
            suggestions.append(
                "📚 Consider breaking complex queries into smaller, maintainable parts using CTEs or views"
            )

        if analysis.get("subquery_count", 0) > 3:
            suggestions.append(
                "🔄 Refactor nested subqueries into JOINs or CTEs for better performance"
            )

        if not analysis.get("uses_modern_syntax", True):
            suggestions.append(
                "🎓 Migrate to modern SQL syntax (e.g., explicit JOINs instead of comma-separated tables)"
            )

        if analysis.get("missing_indexes_count", 0) > 2:
            suggestions.append(
                "📊 Conduct a comprehensive index analysis and create a indexing strategy"
            )

        if analysis.get("scalability_score", 100) < 70:
            suggestions.append(
                "📈 Design queries with future data growth in mind - current approach may not scale"
            )

        # Add learning suggestions based on patterns
        if analysis.get("repeated_mistakes", []):
            suggestions.append(
                "📖 Focus on learning about: "
                + ", ".join(analysis["repeated_mistakes"][:3])
            )

        return suggestions

    def _select_encouragement(self, grade: str) -> str:
        """Select appropriate encouragement message"""

        messages = self.encouragement_templates.get(
            grade, self.encouragement_templates["C"]
        )
        return random.choice(messages)

    def _generate_next_steps(
        self, improvements: List[FeedbackMessage], grade: str
    ) -> List[str]:
        """Generate actionable next steps"""

        next_steps = []

        # Priority 1: Fix critical issues
        critical = [imp for imp in improvements if imp.severity == "critical"]
        if critical:
            for imp in critical[:2]:
                next_steps.append(f"1️⃣ {imp.title}: {imp.recommendation}")

        # Priority 2: High impact improvements
        high_impact = [imp for imp in improvements if imp.severity == "high"]
        if high_impact and len(next_steps) < 3:
            for imp in high_impact[:2]:
                next_steps.append(f"2️⃣ {imp.title}: {imp.recommendation}")

        # Priority 3: Learning recommendations
        if grade in ["D", "F"] and len(next_steps) < 5:
            next_steps.append("3️⃣ Review SQL performance fundamentals")
            next_steps.append("4️⃣ Practice with simpler queries first")
        elif grade in ["B", "C"] and len(next_steps) < 5:
            next_steps.append("3️⃣ Learn about query execution plans")
            next_steps.append("4️⃣ Study advanced optimization techniques")

        # Priority 4: Testing and validation
        if len(next_steps) < 5:
            next_steps.append(
                "5️⃣ Test changes with EXPLAIN PLAN to verify improvements"
            )

        return next_steps

    def _generate_comparative_analysis(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Generate comparative analysis if baseline available"""

        if "baseline_comparison" not in analysis:
            return None

        baseline = analysis["baseline_comparison"]

        comparison_parts = []

        # Performance comparison
        if "performance_delta" in baseline:
            delta = baseline["performance_delta"]
            if delta > 0:
                comparison_parts.append(
                    f"📈 {delta:.0f}% faster than average similar queries"
                )
            else:
                comparison_parts.append(
                    f"📉 {abs(delta):.0f}% slower than average similar queries"
                )

        # Complexity comparison
        if "complexity_percentile" in baseline:
            percentile = baseline["complexity_percentile"]
            if percentile > 75:
                comparison_parts.append(
                    f"More complex than {percentile:.0f}% of similar queries"
                )
            elif percentile < 25:
                comparison_parts.append(
                    f"Simpler than {100-percentile:.0f}% of similar queries"
                )

        # Best practices comparison
        if "best_practice_score" in baseline:
            score = baseline["best_practice_score"]
            comparison_parts.append(f"Best practices score: {score:.0f}/100")

        return " | ".join(comparison_parts) if comparison_parts else None

    def _estimate_improvement(
        self, improvements: List[FeedbackMessage]
    ) -> Optional[str]:
        """Estimate potential improvement from implementing suggestions"""

        if not improvements:
            return None

        # Calculate cumulative improvement estimate
        total_impact = 0.0
        for imp in improvements[:5]:  # Top 5 improvements
            # Estimate based on severity
            severity_impact = {
                "critical": 0.4,
                "high": 0.25,
                "medium": 0.15,
                "low": 0.05,
            }
            total_impact += severity_impact.get(imp.severity, 0.1)

        # Cap at realistic improvement
        total_impact = min(total_impact, 0.8)

        if total_impact > 0.5:
            return f"Implementing suggested improvements could enhance performance by up to {total_impact*100:.0f}%"
        elif total_impact > 0.2:
            return (
                f"Moderate performance improvement of {total_impact*100:.0f}% expected"
            )
        else:
            return f"Minor performance gains of approximately {total_impact*100:.0f}%"

    def _compile_learning_resources(
        self, improvements: List[FeedbackMessage], analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Compile relevant learning resources"""

        resources = []
        topics_covered = set()

        # Map issues to learning topics
        topic_map = {
            "index": {
                "title": "Database Indexing Fundamentals",
                "url": "#indexing",
                "level": "intermediate",
            },
            "join": {
                "title": "SQL JOIN Types and Optimization",
                "url": "#joins",
                "level": "intermediate",
            },
            "subquery": {
                "title": "Subqueries vs JOINs",
                "url": "#subqueries",
                "level": "advanced",
            },
            "performance": {
                "title": "SQL Performance Tuning",
                "url": "#performance",
                "level": "advanced",
            },
            "security": {
                "title": "SQL Injection Prevention",
                "url": "#security",
                "level": "intermediate",
            },
            "null": {
                "title": "Understanding NULL in SQL",
                "url": "#null-handling",
                "level": "beginner",
            },
            "aggregate": {
                "title": "Aggregation Functions and GROUP BY",
                "url": "#aggregation",
                "level": "intermediate",
            },
            "window": {
                "title": "Window Functions",
                "url": "#window-functions",
                "level": "advanced",
            },
        }

        # Add resources based on improvements needed
        for imp in improvements[:5]:
            for keyword, resource in topic_map.items():
                if keyword in imp.title.lower() and keyword not in topics_covered:
                    # Filter by user level
                    if self._is_appropriate_level(resource["level"]):
                        resources.append(
                            {
                                "title": resource["title"],
                                "url": resource["url"],
                                "relevance": (
                                    "High"
                                    if imp.severity in ["critical", "high"]
                                    else "Medium"
                                ),
                            }
                        )
                        topics_covered.add(keyword)

        # Add general resources based on grade
        grade = self._calculate_grade(analysis.get("overall_score", 50))
        if grade in ["D", "F"]:
            resources.append(
                {
                    "title": "SQL Basics Tutorial",
                    "url": "#sql-basics",
                    "relevance": "High",
                }
            )
        elif grade in ["B", "C"]:
            resources.append(
                {
                    "title": "Advanced SQL Optimization",
                    "url": "#advanced-optimization",
                    "relevance": "Medium",
                }
            )

        return resources[:5]  # Limit to 5 resources

    def _is_appropriate_level(self, resource_level: str) -> bool:
        """Check if resource level is appropriate for user"""

        level_map = {
            FeedbackLevel.BEGINNER: ["beginner"],
            FeedbackLevel.INTERMEDIATE: ["beginner", "intermediate"],
            FeedbackLevel.ADVANCED: ["intermediate", "advanced"],
            FeedbackLevel.EXPERT: ["intermediate", "advanced", "expert"],
        }

        return resource_level in level_map.get(self.user_level, ["intermediate"])

    def _simplify_language(self, text: str) -> str:
        """Simplify technical language for beginners"""

        simplifications = {
            "cardinality": "number of unique values",
            "execution plan": "how the database processes your query",
            "index scan": "quick lookup using an index",
            "table scan": "checking every row in the table",
            "hash join": "a way to combine data from two tables",
            "subquery": "a query inside another query",
            "aggregate": "summary calculation like SUM or COUNT",
            "predicate": "condition in your WHERE clause",
            "selectivity": "how many rows match your condition",
        }

        result = text
        for term, simple in simplifications.items():
            result = re.sub(r"\b" + term + r"\b", simple, result, flags=re.IGNORECASE)

        return result

    def _add_technical_details(self, kwargs: Dict[str, Any]) -> str:
        """Add technical details for expert users"""

        details = []

        if "pattern_id" in kwargs:
            details.append(f"Pattern: {kwargs['pattern_id']}")

        if "estimated_cost" in kwargs:
            details.append(f"Estimated cost: {kwargs['estimated_cost']}")

        if "affected_rows" in kwargs:
            details.append(f"Affected rows: ~{kwargs['affected_rows']}")

        if "index_hint" in kwargs:
            details.append(f"Index hint: {kwargs['index_hint']}")

        return " | ".join(details) if details else None

    def _map_to_category(self, category_str: str) -> FeedbackCategory:
        """Map string category to enum"""

        mapping = {
            "performance": FeedbackCategory.PERFORMANCE,
            "correctness": FeedbackCategory.CORRECTNESS,
            "style": FeedbackCategory.STYLE,
            "security": FeedbackCategory.SECURITY,
            "best_practice": FeedbackCategory.BEST_PRACTICE,
            "optimization": FeedbackCategory.OPTIMIZATION,
        }

        return mapping.get(category_str.lower(), FeedbackCategory.PERFORMANCE)


def format_feedback_as_markdown(feedback: ComprehensiveFeedback) -> str:
    """Format comprehensive feedback as Markdown"""

    lines = []

    # Header with grade
    lines.append("# Query Analysis Report")
    lines.append(
        f"## Overall Grade: **{feedback.overall_grade}** ({feedback.overall_score:.1f}/100)"
    )
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append(feedback.summary)
    lines.append("")

    # Encouragement
    lines.append(f"*{feedback.encouragement}*")
    lines.append("")

    # Strengths
    if feedback.strengths:
        lines.append("## What You Did Well")
        for strength in feedback.strengths:
            lines.append(strength)
        lines.append("")

    # Quick Wins
    if feedback.quick_wins:
        lines.append("## Quick Wins")
        lines.append("*Easy improvements with high impact:*")
        for win in feedback.quick_wins:
            lines.append(win)
        lines.append("")

    # Improvements needed
    if feedback.improvements:
        lines.append("## Areas for Improvement")

        # Group by severity
        critical = [imp for imp in feedback.improvements if imp.severity == "critical"]
        high = [imp for imp in feedback.improvements if imp.severity == "high"]
        medium = [imp for imp in feedback.improvements if imp.severity == "medium"]

        if critical:
            lines.append("### 🚨 Critical Issues")
            for imp in critical:
                lines.append(f"**{imp.title}**")
                lines.append(f"- {imp.description}")
                lines.append(f"- *Impact:* {imp.impact}")
                lines.append(f"- *Fix:* {imp.recommendation}")
                if imp.example:
                    lines.append(f"- *Example:* `{imp.example}`")
                lines.append("")

        if high:
            lines.append("### ⚠️ High Priority")
            for imp in high:
                lines.append(f"**{imp.title}**")
                lines.append(f"- {imp.description}")
                lines.append(f"- *Fix:* {imp.recommendation}")
                lines.append("")

        if medium:
            lines.append("### 💡 Suggestions")
            for imp in medium:
                lines.append(f"- **{imp.title}**: {imp.recommendation}")
            lines.append("")

    # Next Steps
    if feedback.next_steps:
        lines.append("## Next Steps")
        for step in feedback.next_steps:
            lines.append(step)
        lines.append("")

    # Long-term suggestions
    if feedback.long_term_suggestions:
        lines.append("## Long-term Improvements")
        for suggestion in feedback.long_term_suggestions:
            lines.append(suggestion)
        lines.append("")

    # Comparative analysis
    if feedback.comparative_analysis:
        lines.append("## Comparative Analysis")
        lines.append(feedback.comparative_analysis)
        lines.append("")

    # Estimated improvement
    if feedback.estimated_improvement:
        lines.append("## Potential Impact")
        lines.append(feedback.estimated_improvement)
        lines.append("")

    # Learning resources
    if feedback.learning_resources:
        lines.append("## Learning Resources")
        for resource in feedback.learning_resources:
            lines.append(
                f"- [{resource['title']}]({resource['url']}) - Relevance: {resource['relevance']}"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    generator = NaturalLanguageFeedbackGenerator(user_level=FeedbackLevel.INTERMEDIATE)

    # Sample analysis results
    analysis_results = {
        "overall_score": 72,
        "query_type": "SELECT",
        "anti_patterns": [
            {
                "name": "SELECT * Usage",
                "category": "performance",
                "severity": "high",
                "impact": "Fetches unnecessary data",
                "recommendation": "Specify only needed columns",
                "example_fix": "SELECT id, name, email FROM users",
                "confidence": 0.9,
            },
            {
                "name": "Missing Index",
                "category": "performance",
                "severity": "medium",
                "impact": "Full table scan required",
                "recommendation": "Add index on filter columns",
                "confidence": 0.7,
            },
        ],
        "uses_indexes": True,
        "has_proper_joins": True,
        "has_limit": False,
        "uses_select_star": True,
        "estimated_execution_time": 250,
        "avg_query_complexity": 5,
    }

    feedback = generator.generate_feedback(analysis_results)

    # Print formatted feedback
    print(format_feedback_as_markdown(feedback))
