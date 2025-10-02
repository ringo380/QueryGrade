"""
Contextual Recommendations Engine

This module provides intelligent, context-aware recommendations for SQL query improvements
based on query patterns, database context, user history, and best practices.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter
import json


class RecommendationType(Enum):
    """Types of recommendations"""
    INDEX_CREATION = "index_creation"
    QUERY_REWRITE = "query_rewrite"
    SCHEMA_CHANGE = "schema_change"
    CONFIGURATION = "configuration"
    BEST_PRACTICE = "best_practice"
    LEARNING = "learning"
    MONITORING = "monitoring"
    ARCHITECTURE = "architecture"


class RecommendationPriority(Enum):
    """Priority levels for recommendations"""
    CRITICAL = 1  # Must implement immediately
    HIGH = 2      # Should implement soon
    MEDIUM = 3    # Consider implementing
    LOW = 4       # Nice to have
    INFORMATIONAL = 5  # FYI only


class ImplementationComplexity(Enum):
    """Complexity of implementing recommendation"""
    TRIVIAL = "trivial"      # < 5 minutes
    EASY = "easy"            # < 30 minutes
    MODERATE = "moderate"    # < 2 hours
    COMPLEX = "complex"      # < 1 day
    VERY_COMPLEX = "very_complex"  # Multiple days


@dataclass
class Recommendation:
    """Represents a contextual recommendation"""
    recommendation_id: str
    type: RecommendationType
    priority: RecommendationPriority
    complexity: ImplementationComplexity
    title: str
    description: str
    rationale: str  # Why this recommendation
    implementation_steps: List[str]
    expected_impact: Dict[str, Any]  # performance, maintainability, etc.
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    code_example: Optional[str] = None
    estimated_time: str = "Unknown"
    confidence: float = 0.8
    relevant_queries: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    documentation_links: List[str] = field(default_factory=list)


@dataclass
class RecommendationContext:
    """Context for generating recommendations"""
    query: str
    query_patterns: List[Dict[str, Any]]
    database_stats: Dict[str, Any]
    user_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    environment: str  # development, staging, production
    constraints: List[str]  # business/technical constraints
    goals: List[str]  # optimization goals


@dataclass
class RecommendationSet:
    """Set of recommendations for a query/context"""
    context_summary: str
    recommendations: List[Recommendation]
    implementation_plan: List[str]  # Ordered implementation steps
    total_estimated_time: str
    expected_overall_impact: Dict[str, float]
    risk_assessment: str
    success_metrics: List[str]  # How to measure success


class ContextualRecommendationsEngine:
    """Generates intelligent, contextual recommendations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recommendation_history = []
        self.pattern_knowledge_base = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize recommendation knowledge base"""
        return {
            'index_patterns': {
                'missing_where_index': {
                    'pattern': r'WHERE\s+(\w+)\s*=',
                    'recommendation': 'Create index on frequently filtered columns',
                    'impact': {'performance': 0.7}
                },
                'missing_join_index': {
                    'pattern': r'JOIN.*ON\s+\w+\.(\w+)\s*=\s*\w+\.(\w+)',
                    'recommendation': 'Create indexes on join columns',
                    'impact': {'performance': 0.6}
                },
                'missing_order_index': {
                    'pattern': r'ORDER\s+BY\s+(\w+)',
                    'recommendation': 'Create index for sorting',
                    'impact': {'performance': 0.4}
                }
            },
            'rewrite_patterns': {
                'exists_vs_in': {
                    'pattern': r'WHERE\s+\w+\s+IN\s*\(\s*SELECT',
                    'recommendation': 'Replace IN with EXISTS for better performance',
                    'impact': {'performance': 0.5}
                },
                'union_all': {
                    'pattern': r'\bUNION\b(?!\s+ALL)',
                    'recommendation': 'Use UNION ALL if duplicates are acceptable',
                    'impact': {'performance': 0.3}
                }
            },
            'best_practices': {
                'select_star': {
                    'pattern': r'SELECT\s+\*',
                    'recommendation': 'Specify required columns explicitly',
                    'impact': {'performance': 0.2, 'maintainability': 0.4}
                },
                'implicit_conversion': {
                    'pattern': r"WHERE\s+\w+\s*=\s*'\d+'",
                    'recommendation': 'Avoid implicit type conversions',
                    'impact': {'performance': 0.3}
                }
            }
        }

    def generate_recommendations(self, context: RecommendationContext) -> RecommendationSet:
        """Generate comprehensive recommendations based on context"""

        recommendations = []

        # Analyze query patterns
        query_recommendations = self._analyze_query_patterns(context)
        recommendations.extend(query_recommendations)

        # Analyze database statistics
        db_recommendations = self._analyze_database_context(context)
        recommendations.extend(db_recommendations)

        # Analyze user history
        history_recommendations = self._analyze_user_history(context)
        recommendations.extend(history_recommendations)

        # Analyze performance metrics
        perf_recommendations = self._analyze_performance_metrics(context)
        recommendations.extend(perf_recommendations)

        # Generate architecture recommendations
        arch_recommendations = self._generate_architecture_recommendations(context)
        recommendations.extend(arch_recommendations)

        # Apply contextual filtering
        filtered_recommendations = self._filter_by_context(recommendations, context)

        # Prioritize and order recommendations
        prioritized = self._prioritize_recommendations(filtered_recommendations, context)

        # Create implementation plan
        implementation_plan = self._create_implementation_plan(prioritized)

        # Calculate overall impact
        overall_impact = self._calculate_overall_impact(prioritized)

        # Assess risks
        risk_assessment = self._assess_risks(prioritized, context)

        # Define success metrics
        success_metrics = self._define_success_metrics(prioritized)

        return RecommendationSet(
            context_summary=self._summarize_context(context),
            recommendations=prioritized[:10],  # Top 10 recommendations
            implementation_plan=implementation_plan,
            total_estimated_time=self._calculate_total_time(prioritized[:10]),
            expected_overall_impact=overall_impact,
            risk_assessment=risk_assessment,
            success_metrics=success_metrics
        )

    def _analyze_query_patterns(self, context: RecommendationContext) -> List[Recommendation]:
        """Analyze query patterns and generate recommendations"""
        recommendations = []
        query = context.query

        # Check for index opportunities
        for pattern_name, pattern_info in self.pattern_knowledge_base['index_patterns'].items():
            if re.search(pattern_info['pattern'], query, re.IGNORECASE):
                recommendations.append(self._create_index_recommendation(
                    pattern_name, pattern_info, query
                ))

        # Check for rewrite opportunities
        for pattern_name, pattern_info in self.pattern_knowledge_base['rewrite_patterns'].items():
            if re.search(pattern_info['pattern'], query, re.IGNORECASE):
                recommendations.append(self._create_rewrite_recommendation(
                    pattern_name, pattern_info, query
                ))

        # Check for best practice violations
        for pattern_name, pattern_info in self.pattern_knowledge_base['best_practices'].items():
            if re.search(pattern_info['pattern'], query, re.IGNORECASE):
                recommendations.append(self._create_best_practice_recommendation(
                    pattern_name, pattern_info, query
                ))

        return recommendations

    def _create_index_recommendation(self, pattern_name: str, pattern_info: Dict[str, Any],
                                    query: str) -> Recommendation:
        """Create index creation recommendation"""

        # Extract column names from pattern match
        match = re.search(pattern_info['pattern'], query, re.IGNORECASE)
        columns = []
        if match:
            columns = [g for g in match.groups() if g]

        return Recommendation(
            recommendation_id=f"IDX_{pattern_name}_{datetime.now().timestamp()}",
            type=RecommendationType.INDEX_CREATION,
            priority=RecommendationPriority.HIGH,
            complexity=ImplementationComplexity.EASY,
            title=f"Create Index: {pattern_info['recommendation']}",
            description=f"Query analysis detected missing index opportunity on columns: {', '.join(columns)}",
            rationale="Indexes significantly improve query performance by allowing rapid data lookup",
            implementation_steps=[
                f"1. Analyze column cardinality and selectivity",
                f"2. Create index: CREATE INDEX idx_name ON table({', '.join(columns)})",
                f"3. Test query performance with EXPLAIN",
                f"4. Monitor index usage statistics"
            ],
            expected_impact=pattern_info['impact'],
            prerequisites=["Verify column data distribution", "Check existing indexes"],
            risks=["Index maintenance overhead", "Additional storage required"],
            code_example=f"CREATE INDEX idx_{columns[0] if columns else 'column'} ON table_name({', '.join(columns)});",
            estimated_time="15 minutes",
            confidence=0.85,
            tags={"index", "performance", "optimization"}
        )

    def _create_rewrite_recommendation(self, pattern_name: str, pattern_info: Dict[str, Any],
                                      query: str) -> Recommendation:
        """Create query rewrite recommendation"""

        rewritten_query = self._generate_rewrite(query, pattern_name)

        return Recommendation(
            recommendation_id=f"RW_{pattern_name}_{datetime.now().timestamp()}",
            type=RecommendationType.QUERY_REWRITE,
            priority=RecommendationPriority.MEDIUM,
            complexity=ImplementationComplexity.MODERATE,
            title=f"Query Rewrite: {pattern_info['recommendation']}",
            description="Optimize query structure for better performance",
            rationale="Query structure can significantly impact execution plan and performance",
            implementation_steps=[
                "1. Review current query logic",
                "2. Apply recommended rewrite pattern",
                "3. Test functionality equivalence",
                "4. Compare execution plans",
                "5. Deploy and monitor"
            ],
            expected_impact=pattern_info['impact'],
            prerequisites=["Understand current query logic", "Have test data available"],
            risks=["Potential logic changes", "Different result ordering"],
            alternatives=["Keep current structure with index optimization"],
            code_example=rewritten_query,
            estimated_time="30 minutes",
            confidence=0.75,
            tags={"rewrite", "optimization", "performance"}
        )

    def _create_best_practice_recommendation(self, pattern_name: str, pattern_info: Dict[str, Any],
                                            query: str) -> Recommendation:
        """Create best practice recommendation"""

        return Recommendation(
            recommendation_id=f"BP_{pattern_name}_{datetime.now().timestamp()}",
            type=RecommendationType.BEST_PRACTICE,
            priority=RecommendationPriority.LOW,
            complexity=ImplementationComplexity.TRIVIAL,
            title=f"Best Practice: {pattern_info['recommendation']}",
            description="Follow SQL best practices for maintainable, efficient code",
            rationale="Best practices improve code quality, performance, and maintainability",
            implementation_steps=[
                "1. Identify anti-pattern in current query",
                "2. Apply best practice pattern",
                "3. Update documentation",
                "4. Review with team"
            ],
            expected_impact=pattern_info['impact'],
            prerequisites=["None"],
            risks=["Minimal"],
            estimated_time="5 minutes",
            confidence=0.9,
            tags={"best-practice", "maintainability", "code-quality"}
        )

    def _analyze_database_context(self, context: RecommendationContext) -> List[Recommendation]:
        """Generate recommendations based on database statistics"""
        recommendations = []

        if not context.database_stats:
            return recommendations

        # Check for table statistics
        if 'table_stats' in context.database_stats:
            for table_name, stats in context.database_stats['table_stats'].items():
                # Large table without proper indexing
                if stats.get('row_count', 0) > 100000 and stats.get('index_count', 0) < 2:
                    recommendations.append(self._create_table_optimization_recommendation(
                        table_name, stats
                    ))

                # High fragmentation
                if stats.get('fragmentation', 0) > 30:
                    recommendations.append(self._create_maintenance_recommendation(
                        table_name, stats
                    ))

        # Check for missing statistics
        if context.database_stats.get('stats_age_days', 0) > 30:
            recommendations.append(self._create_statistics_recommendation())

        return recommendations

    def _create_table_optimization_recommendation(self, table_name: str,
                                                 stats: Dict[str, Any]) -> Recommendation:
        """Create table optimization recommendation"""

        return Recommendation(
            recommendation_id=f"TBL_OPT_{table_name}_{datetime.now().timestamp()}",
            type=RecommendationType.SCHEMA_CHANGE,
            priority=RecommendationPriority.HIGH,
            complexity=ImplementationComplexity.COMPLEX,
            title=f"Optimize Table: {table_name}",
            description=f"Large table ({stats.get('row_count', 0):,} rows) needs optimization",
            rationale="Large tables without proper indexing cause performance issues",
            implementation_steps=[
                "1. Analyze query patterns against this table",
                "2. Identify frequently accessed columns",
                "3. Create appropriate indexes",
                "4. Consider partitioning for very large tables",
                "5. Update statistics"
            ],
            expected_impact={'performance': 0.6, 'scalability': 0.7},
            prerequisites=["Analyze workload patterns", "Plan maintenance window"],
            risks=["Temporary performance impact during implementation"],
            estimated_time="2-4 hours",
            confidence=0.8,
            tags={"schema", "optimization", "indexing"}
        )

    def _create_maintenance_recommendation(self, table_name: str,
                                          stats: Dict[str, Any]) -> Recommendation:
        """Create maintenance recommendation"""

        fragmentation = stats.get('fragmentation', 0)

        return Recommendation(
            recommendation_id=f"MAINT_{table_name}_{datetime.now().timestamp()}",
            type=RecommendationType.CONFIGURATION,
            priority=RecommendationPriority.MEDIUM,
            complexity=ImplementationComplexity.EASY,
            title=f"Defragment Table: {table_name}",
            description=f"Table has {fragmentation:.1f}% fragmentation",
            rationale="High fragmentation reduces query performance",
            implementation_steps=[
                "1. Schedule maintenance window",
                "2. Run OPTIMIZE TABLE or equivalent",
                "3. Update statistics",
                "4. Verify fragmentation reduction"
            ],
            expected_impact={'performance': 0.3},
            prerequisites=["Maintenance window", "Backup current state"],
            risks=["Table lock during operation"],
            code_example=f"OPTIMIZE TABLE {table_name};",
            estimated_time="30-60 minutes",
            confidence=0.9,
            tags={"maintenance", "fragmentation", "performance"}
        )

    def _create_statistics_recommendation(self) -> Recommendation:
        """Create statistics update recommendation"""

        return Recommendation(
            recommendation_id=f"STATS_UPDATE_{datetime.now().timestamp()}",
            type=RecommendationType.CONFIGURATION,
            priority=RecommendationPriority.HIGH,
            complexity=ImplementationComplexity.TRIVIAL,
            title="Update Database Statistics",
            description="Database statistics are outdated",
            rationale="Query optimizer needs current statistics for optimal execution plans",
            implementation_steps=[
                "1. Run UPDATE STATISTICS or equivalent",
                "2. Schedule regular statistics updates",
                "3. Monitor query plan changes"
            ],
            expected_impact={'performance': 0.4},
            prerequisites=["None"],
            risks=["Minimal"],
            code_example="UPDATE STATISTICS;",
            estimated_time="5-15 minutes",
            confidence=0.95,
            tags={"statistics", "maintenance", "optimizer"}
        )

    def _analyze_user_history(self, context: RecommendationContext) -> List[Recommendation]:
        """Generate recommendations based on user history"""
        recommendations = []

        if not context.user_history:
            return recommendations

        # Analyze repeated patterns
        pattern_counts = Counter()
        for history_item in context.user_history:
            if 'pattern' in history_item:
                pattern_counts[history_item['pattern']] += 1

        # Recommend learning for frequently encountered issues
        for pattern, count in pattern_counts.most_common(3):
            if count >= 3:  # Pattern repeated 3+ times
                recommendations.append(self._create_learning_recommendation(pattern, count))

        return recommendations

    def _create_learning_recommendation(self, pattern: str, frequency: int) -> Recommendation:
        """Create learning recommendation"""

        learning_topics = {
            'subquery': "SQL Subqueries and CTEs",
            'join': "Advanced JOIN Techniques",
            'index': "Database Indexing Strategies",
            'performance': "Query Performance Tuning",
            'aggregation': "Aggregation and Window Functions"
        }

        topic = learning_topics.get(pattern, "SQL Best Practices")

        return Recommendation(
            recommendation_id=f"LEARN_{pattern}_{datetime.now().timestamp()}",
            type=RecommendationType.LEARNING,
            priority=RecommendationPriority.LOW,
            complexity=ImplementationComplexity.MODERATE,
            title=f"Learning Opportunity: {topic}",
            description=f"Pattern '{pattern}' appeared {frequency} times in recent queries",
            rationale="Investing in learning prevents repeated issues and improves overall skills",
            implementation_steps=[
                f"1. Review documentation on {topic}",
                "2. Complete hands-on exercises",
                "3. Apply learnings to current queries",
                "4. Share knowledge with team"
            ],
            expected_impact={'skill_improvement': 0.8, 'future_performance': 0.6},
            prerequisites=["Time allocation for learning"],
            risks=["None"],
            documentation_links=[f"#learn-{pattern}"],
            estimated_time="2-4 hours",
            confidence=0.7,
            tags={"learning", "skill-development", pattern}
        )

    def _analyze_performance_metrics(self, context: RecommendationContext) -> List[Recommendation]:
        """Generate recommendations based on performance metrics"""
        recommendations = []

        if not context.performance_metrics:
            return recommendations

        metrics = context.performance_metrics

        # Check for slow queries
        if metrics.get('avg_execution_time_ms', 0) > 1000:
            recommendations.append(self._create_performance_optimization_recommendation(metrics))

        # Check for high resource usage
        if metrics.get('cpu_usage', 0) > 80:
            recommendations.append(self._create_resource_optimization_recommendation('CPU', metrics))

        if metrics.get('memory_usage', 0) > 80:
            recommendations.append(self._create_resource_optimization_recommendation('Memory', metrics))

        # Check for concurrency issues
        if metrics.get('lock_wait_time_ms', 0) > 100:
            recommendations.append(self._create_concurrency_recommendation(metrics))

        return recommendations

    def _create_performance_optimization_recommendation(self,
                                                       metrics: Dict[str, Any]) -> Recommendation:
        """Create performance optimization recommendation"""

        exec_time = metrics.get('avg_execution_time_ms', 0)

        return Recommendation(
            recommendation_id=f"PERF_OPT_{datetime.now().timestamp()}",
            type=RecommendationType.QUERY_REWRITE,
            priority=RecommendationPriority.CRITICAL,
            complexity=ImplementationComplexity.COMPLEX,
            title="Critical Performance Optimization Needed",
            description=f"Query averaging {exec_time}ms execution time",
            rationale="Slow queries impact user experience and system resources",
            implementation_steps=[
                "1. Analyze execution plan",
                "2. Identify bottlenecks",
                "3. Apply optimization techniques",
                "4. Test improvements",
                "5. Deploy optimized version"
            ],
            expected_impact={'performance': 0.7, 'user_experience': 0.8},
            prerequisites=["Query profiling tools", "Test environment"],
            risks=["Functionality changes", "Different results"],
            estimated_time="2-4 hours",
            confidence=0.85,
            tags={"performance", "critical", "optimization"}
        )

    def _create_resource_optimization_recommendation(self, resource_type: str,
                                                    metrics: Dict[str, Any]) -> Recommendation:
        """Create resource optimization recommendation"""

        usage = metrics.get(f'{resource_type.lower()}_usage', 0)

        return Recommendation(
            recommendation_id=f"RES_{resource_type}_{datetime.now().timestamp()}",
            type=RecommendationType.CONFIGURATION,
            priority=RecommendationPriority.HIGH,
            complexity=ImplementationComplexity.MODERATE,
            title=f"Optimize {resource_type} Usage",
            description=f"{resource_type} usage at {usage}%",
            rationale=f"High {resource_type} usage impacts system stability",
            implementation_steps=[
                f"1. Profile {resource_type} usage patterns",
                f"2. Optimize resource-intensive operations",
                f"3. Consider hardware/configuration upgrades",
                f"4. Implement resource limits"
            ],
            expected_impact={'stability': 0.6, 'performance': 0.4},
            prerequisites=["Resource monitoring tools"],
            risks=["Service disruption during changes"],
            estimated_time="1-2 hours",
            confidence=0.8,
            tags={"resources", resource_type.lower(), "optimization"}
        )

    def _create_concurrency_recommendation(self, metrics: Dict[str, Any]) -> Recommendation:
        """Create concurrency optimization recommendation"""

        lock_time = metrics.get('lock_wait_time_ms', 0)

        return Recommendation(
            recommendation_id=f"CONC_OPT_{datetime.now().timestamp()}",
            type=RecommendationType.ARCHITECTURE,
            priority=RecommendationPriority.HIGH,
            complexity=ImplementationComplexity.VERY_COMPLEX,
            title="Resolve Concurrency Issues",
            description=f"High lock wait time: {lock_time}ms average",
            rationale="Lock contention severely impacts concurrent query performance",
            implementation_steps=[
                "1. Identify lock contention sources",
                "2. Optimize transaction scope",
                "3. Consider read replicas",
                "4. Implement optimistic locking",
                "5. Review isolation levels"
            ],
            expected_impact={'concurrency': 0.7, 'throughput': 0.5},
            prerequisites=["Lock monitoring tools", "Understanding of transaction patterns"],
            risks=["Data consistency concerns", "Application changes required"],
            estimated_time="1-2 days",
            confidence=0.75,
            tags={"concurrency", "locking", "architecture"}
        )

    def _generate_architecture_recommendations(self,
                                              context: RecommendationContext) -> List[Recommendation]:
        """Generate architectural recommendations"""
        recommendations = []

        # Check if caching would help
        if self._should_recommend_caching(context):
            recommendations.append(self._create_caching_recommendation())

        # Check if read replicas would help
        if self._should_recommend_read_replicas(context):
            recommendations.append(self._create_read_replica_recommendation())

        # Check if partitioning would help
        if self._should_recommend_partitioning(context):
            recommendations.append(self._create_partitioning_recommendation())

        return recommendations

    def _should_recommend_caching(self, context: RecommendationContext) -> bool:
        """Determine if caching should be recommended"""
        # Simple heuristic: recommend if query is read-heavy and frequently executed
        if context.performance_metrics:
            read_ratio = context.performance_metrics.get('read_write_ratio', 0)
            frequency = context.performance_metrics.get('query_frequency', 0)
            return read_ratio > 10 and frequency > 100

        return False

    def _create_caching_recommendation(self) -> Recommendation:
        """Create caching recommendation"""

        return Recommendation(
            recommendation_id=f"CACHE_{datetime.now().timestamp()}",
            type=RecommendationType.ARCHITECTURE,
            priority=RecommendationPriority.MEDIUM,
            complexity=ImplementationComplexity.COMPLEX,
            title="Implement Query Result Caching",
            description="Frequently accessed read-heavy queries would benefit from caching",
            rationale="Caching reduces database load and improves response times",
            implementation_steps=[
                "1. Identify cacheable queries",
                "2. Choose caching strategy (Redis, Memcached, etc.)",
                "3. Implement cache layer",
                "4. Define cache invalidation strategy",
                "5. Monitor cache hit rates"
            ],
            expected_impact={'performance': 0.8, 'database_load': 0.6},
            prerequisites=["Cache infrastructure", "Application changes"],
            risks=["Stale data", "Cache invalidation complexity"],
            alternatives=["Materialized views", "Query result sets"],
            estimated_time="1-2 days",
            confidence=0.8,
            tags={"caching", "architecture", "performance"}
        )

    def _should_recommend_read_replicas(self, context: RecommendationContext) -> bool:
        """Determine if read replicas should be recommended"""
        if context.performance_metrics:
            return (context.performance_metrics.get('read_write_ratio', 0) > 5 and
                   context.environment == 'production')
        return False

    def _create_read_replica_recommendation(self) -> Recommendation:
        """Create read replica recommendation"""

        return Recommendation(
            recommendation_id=f"REPLICA_{datetime.now().timestamp()}",
            type=RecommendationType.ARCHITECTURE,
            priority=RecommendationPriority.MEDIUM,
            complexity=ImplementationComplexity.VERY_COMPLEX,
            title="Implement Read Replicas",
            description="Distribute read load across multiple database instances",
            rationale="Read replicas improve scalability and reduce primary database load",
            implementation_steps=[
                "1. Set up replica infrastructure",
                "2. Configure replication",
                "3. Implement read/write splitting in application",
                "4. Handle replication lag",
                "5. Monitor replica health"
            ],
            expected_impact={'scalability': 0.9, 'availability': 0.7},
            prerequisites=["Infrastructure budget", "Application architecture support"],
            risks=["Replication lag", "Consistency challenges"],
            estimated_time="3-5 days",
            confidence=0.75,
            tags={"replication", "scalability", "architecture"}
        )

    def _should_recommend_partitioning(self, context: RecommendationContext) -> bool:
        """Determine if partitioning should be recommended"""
        if context.database_stats and 'table_stats' in context.database_stats:
            for table_stats in context.database_stats['table_stats'].values():
                if table_stats.get('row_count', 0) > 10000000:  # 10M+ rows
                    return True
        return False

    def _create_partitioning_recommendation(self) -> Recommendation:
        """Create partitioning recommendation"""

        return Recommendation(
            recommendation_id=f"PART_{datetime.now().timestamp()}",
            type=RecommendationType.SCHEMA_CHANGE,
            priority=RecommendationPriority.MEDIUM,
            complexity=ImplementationComplexity.VERY_COMPLEX,
            title="Implement Table Partitioning",
            description="Large tables would benefit from partitioning",
            rationale="Partitioning improves query performance and maintenance operations",
            implementation_steps=[
                "1. Choose partitioning strategy (range, list, hash)",
                "2. Plan partition boundaries",
                "3. Create partitioned table structure",
                "4. Migrate data to partitions",
                "5. Update application queries"
            ],
            expected_impact={'performance': 0.6, 'maintenance': 0.8},
            prerequisites=["Database support for partitioning", "Maintenance window"],
            risks=["Complex implementation", "Application changes required"],
            estimated_time="1-2 weeks",
            confidence=0.7,
            tags={"partitioning", "schema", "scalability"}
        )

    def _filter_by_context(self, recommendations: List[Recommendation],
                          context: RecommendationContext) -> List[Recommendation]:
        """Filter recommendations based on context constraints"""

        filtered = []

        for rec in recommendations:
            # Check environment constraints
            if context.environment == 'production':
                # Skip risky recommendations in production
                if rec.complexity == ImplementationComplexity.VERY_COMPLEX:
                    continue

            # Check business constraints
            skip = False
            for constraint in context.constraints:
                if any(risk in constraint.lower() for risk in rec.risks):
                    skip = True
                    break

            if not skip:
                filtered.append(rec)

        return filtered

    def _prioritize_recommendations(self, recommendations: List[Recommendation],
                                   context: RecommendationContext) -> List[Recommendation]:
        """Prioritize recommendations based on context and goals"""

        # Score each recommendation
        scored = []
        for rec in recommendations:
            score = self._calculate_recommendation_score(rec, context)
            scored.append((score, rec))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)

        return [rec for _, rec in scored]

    def _calculate_recommendation_score(self, rec: Recommendation,
                                       context: RecommendationContext) -> float:
        """Calculate score for a recommendation"""

        score = 0.0

        # Priority weight
        priority_weights = {
            RecommendationPriority.CRITICAL: 10.0,
            RecommendationPriority.HIGH: 7.0,
            RecommendationPriority.MEDIUM: 4.0,
            RecommendationPriority.LOW: 2.0,
            RecommendationPriority.INFORMATIONAL: 1.0
        }
        score += priority_weights.get(rec.priority, 0)

        # Complexity penalty
        complexity_penalty = {
            ImplementationComplexity.TRIVIAL: 0,
            ImplementationComplexity.EASY: 1,
            ImplementationComplexity.MODERATE: 2,
            ImplementationComplexity.COMPLEX: 3,
            ImplementationComplexity.VERY_COMPLEX: 5
        }
        score -= complexity_penalty.get(rec.complexity, 0)

        # Goal alignment bonus
        for goal in context.goals:
            if goal.lower() in str(rec.expected_impact).lower():
                score += 3

        # Confidence weight
        score *= rec.confidence

        return max(0, score)

    def _create_implementation_plan(self, recommendations: List[Recommendation]) -> List[str]:
        """Create ordered implementation plan"""

        plan = []

        # Group by complexity
        groups = {
            'quick_wins': [],
            'medium_term': [],
            'long_term': []
        }

        for rec in recommendations[:10]:  # Top 10
            if rec.complexity in [ImplementationComplexity.TRIVIAL, ImplementationComplexity.EASY]:
                groups['quick_wins'].append(rec)
            elif rec.complexity == ImplementationComplexity.MODERATE:
                groups['medium_term'].append(rec)
            else:
                groups['long_term'].append(rec)

        # Create phased plan
        if groups['quick_wins']:
            plan.append("=== Phase 1: Quick Wins (This Week) ===")
            for rec in groups['quick_wins']:
                plan.append(f"- {rec.title} ({rec.estimated_time})")

        if groups['medium_term']:
            plan.append("=== Phase 2: Medium-term Improvements (Next 2 Weeks) ===")
            for rec in groups['medium_term']:
                plan.append(f"- {rec.title} ({rec.estimated_time})")

        if groups['long_term']:
            plan.append("=== Phase 3: Long-term Optimizations (Next Month) ===")
            for rec in groups['long_term']:
                plan.append(f"- {rec.title} ({rec.estimated_time})")

        return plan

    def _calculate_overall_impact(self, recommendations: List[Recommendation]) -> Dict[str, float]:
        """Calculate cumulative impact of recommendations"""

        impact = defaultdict(float)

        for rec in recommendations[:10]:  # Top 10
            for metric, value in rec.expected_impact.items():
                # Diminishing returns for multiple recommendations
                current = impact[metric]
                impact[metric] = current + value * (1 - current * 0.1)

        # Cap at realistic values
        for metric in impact:
            impact[metric] = min(impact[metric], 0.9)

        return dict(impact)

    def _assess_risks(self, recommendations: List[Recommendation],
                     context: RecommendationContext) -> str:
        """Assess overall risk of implementing recommendations"""

        risk_score = 0
        risk_factors = []

        for rec in recommendations[:10]:
            # Add risk based on complexity
            if rec.complexity == ImplementationComplexity.VERY_COMPLEX:
                risk_score += 3
                risk_factors.append(f"Complex implementation: {rec.title}")
            elif rec.complexity == ImplementationComplexity.COMPLEX:
                risk_score += 2

            # Add risk for production environment
            if context.environment == 'production':
                if rec.type in [RecommendationType.SCHEMA_CHANGE, RecommendationType.ARCHITECTURE]:
                    risk_score += 2
                    risk_factors.append(f"Production change: {rec.title}")

        # Determine risk level
        if risk_score >= 10:
            assessment = "High Risk: Significant changes requiring careful planning"
        elif risk_score >= 5:
            assessment = "Medium Risk: Some complex changes, proceed with caution"
        else:
            assessment = "Low Risk: Mostly straightforward improvements"

        if risk_factors:
            assessment += f"\nKey risks: {', '.join(risk_factors[:3])}"

        return assessment

    def _define_success_metrics(self, recommendations: List[Recommendation]) -> List[str]:
        """Define metrics to measure success of implementations"""

        metrics = set()

        for rec in recommendations[:10]:
            # Add metrics based on expected impact
            for impact_area in rec.expected_impact:
                if impact_area == 'performance':
                    metrics.add("Query execution time reduction > 30%")
                    metrics.add("Database CPU usage reduction > 20%")
                elif impact_area == 'scalability':
                    metrics.add("Support 2x current transaction volume")
                    metrics.add("Linear performance scaling with data growth")
                elif impact_area == 'maintainability':
                    metrics.add("Code review approval rate > 90%")
                    metrics.add("Bug report reduction > 25%")
                elif impact_area == 'stability':
                    metrics.add("Error rate reduction > 50%")
                    metrics.add("System uptime > 99.9%")

        return list(metrics)[:8]  # Top 8 metrics

    def _summarize_context(self, context: RecommendationContext) -> str:
        """Create summary of analysis context"""

        parts = []

        if context.query:
            query_type = context.query.split()[0].upper() if context.query else "UNKNOWN"
            parts.append(f"Query Type: {query_type}")

        if context.environment:
            parts.append(f"Environment: {context.environment}")

        if context.goals:
            parts.append(f"Goals: {', '.join(context.goals[:3])}")

        if context.constraints:
            parts.append(f"Constraints: {', '.join(context.constraints[:2])}")

        return " | ".join(parts)

    def _calculate_total_time(self, recommendations: List[Recommendation]) -> str:
        """Calculate total implementation time"""

        total_hours = 0

        for rec in recommendations:
            # Parse estimated time
            time_str = rec.estimated_time.lower()
            if 'minute' in time_str:
                # Extract minutes and convert to hours
                minutes = re.search(r'(\d+)', time_str)
                if minutes:
                    total_hours += int(minutes.group(1)) / 60
            elif 'hour' in time_str:
                # Extract hours
                hours = re.search(r'(\d+)', time_str)
                if hours:
                    total_hours += int(hours.group(1))
            elif 'day' in time_str:
                # Extract days and convert to hours
                days = re.search(r'(\d+)', time_str)
                if days:
                    total_hours += int(days.group(1)) * 8  # Assume 8-hour workday

        if total_hours < 1:
            return f"{int(total_hours * 60)} minutes"
        elif total_hours < 8:
            return f"{total_hours:.1f} hours"
        else:
            return f"{total_hours/8:.1f} days"

    def _generate_rewrite(self, query: str, pattern_name: str) -> str:
        """Generate rewritten query based on pattern"""

        # Simple rewrite examples (would be more sophisticated in production)
        if pattern_name == 'exists_vs_in':
            # Replace IN with EXISTS
            rewritten = re.sub(
                r'WHERE\s+(\w+)\s+IN\s*\(\s*SELECT\s+(\w+)\s+FROM\s+(\w+)',
                r'WHERE EXISTS (SELECT 1 FROM \3 WHERE \3.\2 = \1',
                query,
                flags=re.IGNORECASE
            )
            return rewritten

        elif pattern_name == 'union_all':
            # Replace UNION with UNION ALL
            return re.sub(r'\bUNION\b(?!\s+ALL)', 'UNION ALL', query, flags=re.IGNORECASE)

        return query  # Return original if no rewrite available


if __name__ == "__main__":
    # Example usage
    engine = ContextualRecommendationsEngine()

    # Create sample context
    context = RecommendationContext(
        query="""
        SELECT * FROM orders o
        WHERE customer_id IN (SELECT id FROM customers WHERE status = 'active')
        ORDER BY created_at DESC
        """,
        query_patterns=[],
        database_stats={
            'table_stats': {
                'orders': {'row_count': 5000000, 'index_count': 1, 'fragmentation': 35},
                'customers': {'row_count': 100000, 'index_count': 3}
            },
            'stats_age_days': 45
        },
        user_history=[
            {'pattern': 'subquery', 'timestamp': datetime.now()},
            {'pattern': 'subquery', 'timestamp': datetime.now()},
            {'pattern': 'subquery', 'timestamp': datetime.now()}
        ],
        performance_metrics={
            'avg_execution_time_ms': 1500,
            'cpu_usage': 75,
            'read_write_ratio': 20,
            'query_frequency': 500
        },
        environment='production',
        constraints=['No downtime', 'Limited budget'],
        goals=['Improve performance', 'Reduce costs']
    )

    # Generate recommendations
    recommendation_set = engine.generate_recommendations(context)

    # Display results
    print("=== Contextual Recommendations ===\n")
    print(f"Context: {recommendation_set.context_summary}\n")

    print("Top Recommendations:")
    for i, rec in enumerate(recommendation_set.recommendations[:5], 1):
        print(f"\n{i}. {rec.title}")
        print(f"   Priority: {rec.priority.name}")
        print(f"   Complexity: {rec.complexity.value}")
        print(f"   Time: {rec.estimated_time}")
        print(f"   Impact: {rec.expected_impact}")
        print(f"   Rationale: {rec.rationale}")

    print(f"\n{recommendation_set.risk_assessment}")

    print("\nImplementation Plan:")
    for step in recommendation_set.implementation_plan:
        print(step)

    print(f"\nTotal Time: {recommendation_set.total_estimated_time}")
    print(f"Expected Impact: {recommendation_set.expected_overall_impact}")

    print("\nSuccess Metrics:")
    for metric in recommendation_set.success_metrics:
        print(f"- {metric}")