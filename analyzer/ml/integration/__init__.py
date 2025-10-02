"""
External Integrations for QueryGrade ML

This package contains integrations with external systems and data sources:

- database_stats: Database statistics integration
- documentation_loader: SQL documentation loading
- benchmark_generator: Benchmark query generation
- performance_predictor: Performance impact prediction

All integration modules are production-ready.
"""

from .database_stats import (
    DatabaseStatisticsManager,
    TableStatistics,
    IndexStatistics,
    ColumnStatistics,
    RelationshipStatistics,
    WorkloadStatistics,
    StatisticType,
    DataDistribution,
    generate_context_aware_features
)

from .documentation_loader import (
    DocumentationLoader,
    DocumentationSource,
    DocumentationRule,
    BenchmarkResult
)

from .benchmark_generator import (
    BenchmarkGenerator,
    QueryPatternGenerator,
    BenchmarkSet,
    BenchmarkQuery
)

from .performance_predictor import (
    PerformanceImpactPredictor,
    PerformanceBaseline,
    PerformancePrediction,
    OptimizationScenario,
    ComparativeAnalysis,
    ImpactType,
    OptimizationType
)

__all__ = [
    # Database Stats
    'DatabaseStatisticsManager',
    'TableStatistics',
    'IndexStatistics',
    'ColumnStatistics',
    'RelationshipStatistics',
    'WorkloadStatistics',
    'StatisticType',
    'DataDistribution',
    'generate_context_aware_features',

    # Documentation Loader
    'DocumentationLoader',
    'DocumentationSource',
    'DocumentationRule',
    'BenchmarkResult',

    # Benchmark Generator
    'BenchmarkGenerator',
    'QueryPatternGenerator',
    'BenchmarkSet',
    'BenchmarkQuery',

    # Performance Predictor
    'PerformanceImpactPredictor',
    'PerformanceBaseline',
    'PerformancePrediction',
    'OptimizationScenario',
    'ComparativeAnalysis',
    'ImpactType',
    'OptimizationType',
]
