"""
Unified Query Analyzer

This module serves as the main orchestrator for all ML components,
providing a single interface for comprehensive SQL query analysis.
"""

import asyncio
import concurrent.futures
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..integration.database_stats import (DatabaseStatisticsManager,
                                          generate_context_aware_features)
from ..integration.performance_predictor import (PerformanceBaseline,
                                                 PerformanceImpactPredictor)
from ..optimization.plan_predictor import QueryPlanPredictor
from ..optimization.query_rewriter import IntelligentQueryRewriter
from ..recommendations.contextual_engine import (
    ContextualRecommendationsEngine, RecommendationContext)
from ..recommendations.learning_paths import LearningPathGenerator
from ..recommendations.natural_language import (
    FeedbackLevel, NaturalLanguageFeedbackGenerator)
from ..recommendations.personalization_engine import \
    FeedbackPersonalizationEngine
from .anti_pattern_detector import (AntiPatternDetector,
                                    analyze_query_antipatterns)
from .pattern_library import QueryPatternLibrary, analyze_query_patterns
# Import our ML components - updated paths for reorganization
from .semantic_analyzer import (SemanticFeatureExtractor,
                                analyze_query_semantics)
from .workload_patterns import WorkloadPatternRecognizer, analyze_workload


@dataclass
class AnalysisRequest:
    """Request for comprehensive query analysis"""

    query: str
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    analysis_level: str = "comprehensive"  # basic, standard, comprehensive, expert
    personalize: bool = True
    include_rewrite: bool = True
    include_learning_path: bool = False
    safety_level: str = "moderate"  # conservative, moderate, aggressive


@dataclass
class AnalysisResult:
    """Comprehensive analysis result"""

    request_id: str
    query: str
    user_id: Optional[str]

    # Core Analysis Results
    overall_grade: str  # A-F
    overall_score: float  # 0-100
    semantic_analysis: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    anti_pattern_analysis: Dict[str, Any]
    performance_prediction: Optional[Dict[str, Any]]

    # Advanced Features
    natural_language_feedback: Dict[str, Any]
    personalized_recommendations: Optional[Dict[str, Any]]
    query_rewrite: Optional[Dict[str, Any]]
    learning_path: Optional[Dict[str, Any]]

    # Metadata
    analysis_time_ms: float
    confidence_score: float
    components_used: List[str]
    warnings: List[str]

    # Performance Metrics
    execution_time_by_component: Dict[str, float]
    cache_hits: Dict[str, bool]


class UnifiedQueryAnalyzer:
    """
    Main orchestrator for comprehensive SQL query analysis.
    Coordinates all ML components to provide unified, intelligent feedback.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Initialize all ML components
        self._initialize_components()

        # Performance tracking
        self.analysis_cache = {}
        self.performance_metrics = {
            "total_analyses": 0,
            "average_time_ms": 0,
            "cache_hit_rate": 0,
        }

    def _initialize_components(self):
        """Initialize all ML analysis components"""
        try:
            self.semantic_extractor = SemanticFeatureExtractor()
            self.plan_predictor = QueryPlanPredictor()
            self.stats_manager = DatabaseStatisticsManager()
            self.pattern_recognizer = WorkloadPatternRecognizer()
            self.pattern_library = QueryPatternLibrary()
            self.anti_pattern_detector = AntiPatternDetector()
            self.feedback_generator = NaturalLanguageFeedbackGenerator()
            self.recommendations_engine = ContextualRecommendationsEngine()
            self.learning_path_generator = LearningPathGenerator()
            self.impact_predictor = PerformanceImpactPredictor()
            self.query_rewriter = IntelligentQueryRewriter()
            self.personalization_engine = FeedbackPersonalizationEngine()

            self.logger.info("All ML components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing ML components: {e}")
            raise

    async def analyze_query(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform comprehensive query analysis using all available components
        """
        start_time = time.time()
        request_id = f"analysis_{int(start_time * 1000)}"

        self.logger.info(
            f"Starting analysis {request_id} for query: {request.query[:100]}..."
        )

        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_result = self.analysis_cache.get(cache_key)
        if cached_result and not self.config.get("disable_cache", False):
            self.logger.info(f"Cache hit for analysis {request_id}")
            return cached_result

        components_used = []
        execution_times = {}
        warnings = []
        cache_hits = {}

        try:
            # Core Analysis (always performed)
            semantic_analysis = await self._run_semantic_analysis(
                request.query, execution_times
            )
            components_used.append("semantic_analysis")

            pattern_analysis = await self._run_pattern_analysis(
                request.query, execution_times
            )
            components_used.append("pattern_analysis")

            anti_pattern_analysis = await self._run_anti_pattern_analysis(
                request.query, execution_times
            )
            components_used.append("anti_pattern_analysis")

            # Calculate core metrics
            overall_score, overall_grade = self._calculate_overall_metrics(
                semantic_analysis, pattern_analysis, anti_pattern_analysis
            )

            # Advanced Analysis (based on request level)
            performance_prediction = None
            if request.analysis_level in ["comprehensive", "expert"]:
                performance_prediction = await self._run_performance_prediction(
                    request.query, request.context, execution_times
                )
                components_used.append("performance_prediction")

            # Generate Natural Language Feedback
            feedback_data = {
                "overall_score": overall_score,
                "semantic_analysis": semantic_analysis,
                "pattern_analysis": pattern_analysis,
                "anti_pattern_analysis": anti_pattern_analysis,
                "performance_prediction": performance_prediction,
            }

            # Determine user level for feedback
            user_level = self._determine_user_level(request.user_id, request.context)
            feedback_generator = NaturalLanguageFeedbackGenerator(user_level=user_level)

            natural_language_feedback = feedback_generator.generate_feedback(
                feedback_data
            )
            components_used.append("natural_language_feedback")

            # Personalized Recommendations (if requested)
            personalized_recommendations = None
            if request.personalize and request.user_id:
                personalized_recommendations = (
                    await self._generate_personalized_recommendations(
                        request, feedback_data, execution_times
                    )
                )
                components_used.append("personalized_recommendations")

            # Query Rewrite (if requested)
            query_rewrite = None
            if request.include_rewrite:
                query_rewrite = await self._generate_query_rewrite(
                    request.query, request.safety_level, execution_times
                )
                components_used.append("query_rewrite")

            # Learning Path (if requested)
            learning_path = None
            if request.include_learning_path and request.user_id:
                learning_path = await self._generate_learning_path(
                    request.user_id, feedback_data, execution_times
                )
                components_used.append("learning_path")

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                semantic_analysis, pattern_analysis, anti_pattern_analysis
            )

            # Create result
            total_time = (time.time() - start_time) * 1000

            result = AnalysisResult(
                request_id=request_id,
                query=request.query,
                user_id=request.user_id,
                overall_grade=overall_grade,
                overall_score=overall_score,
                semantic_analysis=semantic_analysis,
                pattern_analysis=pattern_analysis,
                anti_pattern_analysis=anti_pattern_analysis,
                performance_prediction=performance_prediction,
                natural_language_feedback=self._serialize_feedback(
                    natural_language_feedback
                ),
                personalized_recommendations=personalized_recommendations,
                query_rewrite=query_rewrite,
                learning_path=learning_path,
                analysis_time_ms=total_time,
                confidence_score=confidence_score,
                components_used=components_used,
                warnings=warnings,
                execution_time_by_component=execution_times,
                cache_hits=cache_hits,
            )

            # Cache result
            self.analysis_cache[cache_key] = result

            # Update performance metrics
            self._update_performance_metrics(total_time)

            self.logger.info(f"Analysis {request_id} completed in {total_time:.1f}ms")
            return result

        except Exception as e:
            self.logger.error(f"Error in analysis {request_id}: {e}")
            # Return minimal result on error
            return self._create_error_result(
                request_id, request, str(e), time.time() - start_time
            )

    @contextmanager
    def _time_component(self, component_name: str, execution_times: Dict[str, float]):
        """Context manager to time component execution"""
        start = time.time()
        try:
            yield
        finally:
            execution_times[component_name] = (time.time() - start) * 1000

    async def _run_semantic_analysis(
        self, query: str, execution_times: Dict[str, float]
    ) -> Dict[str, Any]:
        """Run semantic analysis with timing"""
        with self._time_component("semantic_analysis", execution_times):
            return analyze_query_semantics(query)

    async def _run_pattern_analysis(
        self, query: str, execution_times: Dict[str, float]
    ) -> Dict[str, Any]:
        """Run pattern analysis with timing"""
        with self._time_component("pattern_analysis", execution_times):
            return analyze_query_patterns(query)

    async def _run_anti_pattern_analysis(
        self, query: str, execution_times: Dict[str, float]
    ) -> Dict[str, Any]:
        """Run anti-pattern analysis with timing"""
        with self._time_component("anti_pattern_analysis", execution_times):
            return analyze_query_antipatterns(query)

    async def _run_performance_prediction(
        self, query: str, context: Dict[str, Any], execution_times: Dict[str, float]
    ) -> Dict[str, Any]:
        """Run performance prediction with timing"""
        with self._time_component("performance_prediction", execution_times):
            # Create baseline from context or estimates
            baseline = self._create_performance_baseline(query, context)
            plan_prediction = self.plan_predictor.predict_execution_plan(query)

            return {
                "execution_plan": {
                    "total_cost": plan_prediction.total_cost,
                    "estimated_time_ms": plan_prediction.estimated_time_ms,
                    "memory_usage_mb": plan_prediction.memory_usage_mb,
                    "bottlenecks": [
                        {
                            "node_type": node.node_type.value,
                            "cost": node.estimated_cost,
                            "optimization_hints": node.optimization_hints,
                        }
                        for node in plan_prediction.bottleneck_nodes
                    ],
                },
                "optimization_opportunities": [
                    {
                        "type": opp["type"],
                        "description": opp["description"],
                        "impact": opp["impact"],
                        "estimated_improvement": opp["estimated_improvement"],
                    }
                    for opp in plan_prediction.optimization_opportunities
                ],
            }

    async def _generate_personalized_recommendations(
        self,
        request: AnalysisRequest,
        feedback_data: Dict[str, Any],
        execution_times: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate personalized recommendations"""
        with self._time_component("personalized_recommendations", execution_times):
            # Create recommendation context
            context = RecommendationContext(
                query=request.query,
                query_patterns=feedback_data.get("pattern_analysis", {}).get(
                    "matched_patterns", []
                ),
                database_stats=request.context.get("database_stats", {}),
                user_history=request.context.get("user_history", []),
                performance_metrics=request.context.get("performance_metrics", {}),
                environment=request.context.get("environment", "development"),
                constraints=request.context.get("constraints", []),
                goals=request.context.get("goals", ["Improve performance"]),
            )

            recommendation_set = self.recommendations_engine.generate_recommendations(
                context
            )

            return {
                "context_summary": recommendation_set.context_summary,
                "recommendations": [
                    {
                        "title": rec.title,
                        "type": rec.type.value,
                        "priority": rec.priority.name,
                        "complexity": rec.complexity.value,
                        "description": rec.description,
                        "implementation_steps": rec.implementation_steps,
                        "expected_impact": rec.expected_impact,
                        "estimated_time": rec.estimated_time,
                    }
                    for rec in recommendation_set.recommendations[:5]  # Top 5
                ],
                "implementation_plan": recommendation_set.implementation_plan,
                "risk_assessment": recommendation_set.risk_assessment,
                "expected_roi": recommendation_set.expected_roi,
            }

    async def _generate_query_rewrite(
        self, query: str, safety_level: str, execution_times: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate query rewrite with explanations"""
        with self._time_component("query_rewrite", execution_times):
            rewrite_result = self.query_rewriter.rewrite_query(query, safety_level)

            return {
                "original_query": rewrite_result.original_query,
                "rewritten_query": rewrite_result.rewritten_query,
                "overall_improvement": rewrite_result.overall_improvement,
                "confidence": rewrite_result.confidence,
                "safety_score": rewrite_result.safety_score,
                "rewrite_steps": [
                    {
                        "rule": step.rule.value,
                        "description": step.description,
                        "rationale": step.rationale,
                        "estimated_improvement": step.estimated_improvement,
                        "risks": step.risks,
                    }
                    for step in rewrite_result.rewrite_steps
                ],
                "explanation": rewrite_result.explanation,
                "warnings": rewrite_result.warnings,
                "test_recommendations": rewrite_result.test_recommendations,
            }

    async def _generate_learning_path(
        self,
        user_id: str,
        feedback_data: Dict[str, Any],
        execution_times: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate personalized learning path"""
        with self._time_component("learning_path", execution_times):
            # Extract skill gaps from analysis
            weak_areas = []
            if feedback_data.get("anti_pattern_analysis", {}).get("anti_patterns"):
                weak_areas.extend(
                    [
                        ap["category"]
                        for ap in feedback_data["anti_pattern_analysis"][
                            "anti_patterns"
                        ]
                    ]
                )

            # Create user profile for learning path
            user_profile = {
                "user_id": user_id,
                "current_skills": [
                    "basic_select",
                    "filtering",
                ],  # Would come from user data
                "weak_areas": weak_areas,
                "learning_goals": ["Improve query performance"],
                "hours_per_week": 5,
            }

            learning_path = self.learning_path_generator.generate_learning_path(
                user_profile
            )

            return {
                "current_level": learning_path.current_skill_level.name,
                "target_level": learning_path.target_skill_level.name,
                "estimated_completion_time": learning_path.estimated_completion_time,
                "skill_gaps": learning_path.identified_gaps,
                "recommended_modules": [
                    {
                        "title": module.title,
                        "description": module.description,
                        "duration_hours": module.total_time_hours,
                        "objectives": module.learning_objectives,
                    }
                    for module in learning_path.recommended_modules[:3]  # Top 3
                ],
                "milestones": [
                    {
                        "title": milestone["title"],
                        "target_week": milestone["target_week"],
                        "requirements": milestone["requirements"],
                    }
                    for milestone in learning_path.milestones[:3]  # Next 3
                ],
                "motivational_tips": learning_path.motivational_tips[:5],
            }

    def _calculate_overall_metrics(
        self,
        semantic_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any],
        anti_pattern_analysis: Dict[str, Any],
    ) -> Tuple[float, str]:
        """Calculate overall score and grade"""

        # Base score from anti-pattern analysis
        base_score = anti_pattern_analysis.get("overall_score", 50)

        # Adjust based on pattern quality
        pattern_score = pattern_analysis.get("pattern_score", 0.5) * 100

        # Adjust based on semantic complexity
        semantic_score = 100 - (
            semantic_analysis.get("complexity_indicators", {}).get("cognitive_load", 0)
            * 50
        )

        # Weighted average
        overall_score = base_score * 0.5 + pattern_score * 0.3 + semantic_score * 0.2

        # Convert to grade
        if overall_score >= 90:
            grade = "A"
        elif overall_score >= 80:
            grade = "B"
        elif overall_score >= 70:
            grade = "C"
        elif overall_score >= 60:
            grade = "D"
        else:
            grade = "F"

        return overall_score, grade

    def _calculate_confidence_score(
        self,
        semantic_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any],
        anti_pattern_analysis: Dict[str, Any],
    ) -> float:
        """Calculate confidence in the analysis"""

        # Base confidence
        confidence = 0.7

        # Increase confidence if multiple patterns match
        matched_patterns = len(pattern_analysis.get("matched_patterns", []))
        if matched_patterns > 0:
            confidence += min(0.2, matched_patterns * 0.05)

        # Increase confidence if semantic analysis is comprehensive
        if semantic_analysis.get("query_intent", {}).get("confidence", 0) > 0.8:
            confidence += 0.1

        return min(0.95, confidence)

    def _determine_user_level(
        self, user_id: Optional[str], context: Dict[str, Any]
    ) -> FeedbackLevel:
        """Determine appropriate feedback level for user"""

        # Try to get from context
        skill_level = context.get("skill_level", "intermediate")

        level_mapping = {
            "beginner": FeedbackLevel.BEGINNER,
            "intermediate": FeedbackLevel.INTERMEDIATE,
            "advanced": FeedbackLevel.ADVANCED,
            "expert": FeedbackLevel.EXPERT,
        }

        return level_mapping.get(skill_level, FeedbackLevel.INTERMEDIATE)

    def _serialize_feedback(self, feedback) -> Dict[str, Any]:
        """Serialize feedback object to dictionary"""
        return {
            "overall_grade": feedback.overall_grade,
            "overall_score": feedback.overall_score,
            "summary": feedback.summary,
            "strengths": feedback.strengths,
            "improvements": [
                {
                    "title": imp.title,
                    "category": imp.category.value,
                    "severity": imp.severity,
                    "description": imp.description,
                    "impact": imp.impact,
                    "recommendation": imp.recommendation,
                    "example": imp.example,
                }
                for imp in feedback.improvements
            ],
            "quick_wins": feedback.quick_wins,
            "next_steps": feedback.next_steps,
            "encouragement": feedback.encouragement,
            "estimated_improvement": feedback.estimated_improvement,
        }

    def _create_performance_baseline(
        self, query: str, context: Dict[str, Any]
    ) -> PerformanceBaseline:
        """Create performance baseline from context or estimates"""
        import hashlib

        return PerformanceBaseline(
            query_hash=hashlib.md5(query.encode(), usedforsecurity=False).hexdigest(),
            execution_time_ms=context.get("execution_time_ms", 100),
            cpu_time_ms=context.get("cpu_time_ms", 80),
            memory_mb=context.get("memory_mb", 64),
            io_reads=context.get("io_reads", 1000),
            io_writes=context.get("io_writes", 10),
            rows_examined=context.get("rows_examined", 10000),
            rows_returned=context.get("rows_returned", 100),
            timestamp=datetime.now(),
            database_size_gb=context.get("database_size_gb", 1.0),
            concurrent_queries=context.get("concurrent_queries", 1),
        )

    def _generate_cache_key(self, request: AnalysisRequest) -> str:
        """Generate cache key for request"""
        import hashlib

        key_parts = [
            request.query,
            request.analysis_level,
            str(request.personalize),
            str(request.include_rewrite),
            str(request.include_learning_path),
            request.safety_level,
        ]

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()

    def _update_performance_metrics(self, analysis_time_ms: float):
        """Update performance tracking metrics"""
        self.performance_metrics["total_analyses"] += 1

        # Update rolling average
        current_avg = self.performance_metrics["average_time_ms"]
        total_count = self.performance_metrics["total_analyses"]

        self.performance_metrics["average_time_ms"] = (
            current_avg * (total_count - 1) + analysis_time_ms
        ) / total_count

    def _create_error_result(
        self, request_id: str, request: AnalysisRequest, error: str, elapsed_time: float
    ) -> AnalysisResult:
        """Create minimal result for error cases"""
        return AnalysisResult(
            request_id=request_id,
            query=request.query,
            user_id=request.user_id,
            overall_grade="F",
            overall_score=0.0,
            semantic_analysis={"error": error},
            pattern_analysis={"error": error},
            anti_pattern_analysis={"error": error},
            performance_prediction=None,
            natural_language_feedback={"error": "Analysis failed"},
            personalized_recommendations=None,
            query_rewrite=None,
            learning_path=None,
            analysis_time_ms=elapsed_time * 1000,
            confidence_score=0.0,
            components_used=[],
            warnings=[f"Analysis failed: {error}"],
            execution_time_by_component={},
            cache_hits={},
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            "cache_size": len(self.analysis_cache),
            "components_loaded": len(
                [
                    comp
                    for comp in [
                        self.semantic_extractor,
                        self.plan_predictor,
                        self.stats_manager,
                        self.pattern_recognizer,
                        self.pattern_library,
                        self.anti_pattern_detector,
                        self.feedback_generator,
                        self.recommendations_engine,
                        self.learning_path_generator,
                        self.impact_predictor,
                        self.query_rewriter,
                        self.personalization_engine,
                    ]
                    if comp is not None
                ]
            ),
        }

    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()
        self.logger.info("Analysis cache cleared")


# Global instance for use in Django views
analyzer_instance = None


def get_analyzer() -> UnifiedQueryAnalyzer:
    """Get global analyzer instance"""
    global analyzer_instance
    if analyzer_instance is None:
        analyzer_instance = UnifiedQueryAnalyzer()
    return analyzer_instance


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        analyzer = UnifiedQueryAnalyzer()

        # Test query
        request = AnalysisRequest(
            query="""
            SELECT c.customer_name, COUNT(o.order_id) as order_count
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            WHERE c.created_at >= '2023-01-01'
            GROUP BY c.customer_id, c.customer_name
            HAVING COUNT(o.order_id) > 5
            ORDER BY order_count DESC
            """,
            user_id="test_user",
            analysis_level="comprehensive",
            personalize=True,
            include_rewrite=True,
            include_learning_path=True,
        )

        # Analyze
        result = await analyzer.analyze_query(request)

        print(f"=== Analysis Result {result.request_id} ===")
        print(f"Overall Grade: {result.overall_grade}")
        print(f"Overall Score: {result.overall_score:.1f}")
        print(f"Confidence: {result.confidence_score:.1%}")
        print(f"Analysis Time: {result.analysis_time_ms:.1f}ms")
        print(f"Components Used: {', '.join(result.components_used)}")

        if result.natural_language_feedback:
            print(
                f"\nFeedback Summary: {result.natural_language_feedback.get('summary', 'N/A')}"
            )

        if result.personalized_recommendations:
            print(
                f"\nRecommendations: {len(result.personalized_recommendations.get('recommendations', []))}"
            )

        if result.query_rewrite:
            print(
                f"\nRewrite Improvement: {result.query_rewrite.get('overall_improvement', 0):.1%}"
            )

        # Performance metrics
        print(f"\nPerformance Metrics:")
        metrics = analyzer.get_performance_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    asyncio.run(main())
