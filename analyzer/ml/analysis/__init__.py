"""
Query Analysis Engines for QueryGrade ML

This package contains specialized analyzers for different aspects
of SQL query analysis:

- unified_analyzer: Main orchestrator for comprehensive analysis
- semantic_analyzer: Semantic understanding and intent detection
- complexity_analyzer: Query complexity scoring
- anti_pattern_detector: Identifies SQL anti-patterns
- pattern_library: Common query pattern recognition
- workload_patterns: Workload pattern analysis

All analyzers in this package are production-ready.
"""

from .unified_analyzer import UnifiedQueryAnalyzer, AnalysisRequest, AnalysisResult
from .semantic_analyzer import SemanticFeatureExtractor, analyze_query_semantics
from .complexity_analyzer import QueryComplexityAnalyzer
from .anti_pattern_detector import AntiPatternDetector, analyze_query_antipatterns
from .pattern_library import QueryPatternLibrary, analyze_query_patterns
from .workload_patterns import WorkloadPatternRecognizer, analyze_workload

__all__ = [
    'UnifiedQueryAnalyzer',
    'AnalysisRequest',
    'AnalysisResult',
    'SemanticFeatureExtractor',
    'analyze_query_semantics',
    'QueryComplexityAnalyzer',
    'AntiPatternDetector',
    'analyze_query_antipatterns',
    'QueryPatternLibrary',
    'analyze_query_patterns',
    'WorkloadPatternRecognizer',
    'analyze_workload',
]
