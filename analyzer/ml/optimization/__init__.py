"""
Query Optimization Engines for QueryGrade ML

This package contains query optimization and rewriting tools:

- query_rewriter: Intelligent query rewriting
- query_mutator: Query mutation for testing
- plan_predictor: Execution plan prediction

All optimization modules are production-ready.
"""

from .query_rewriter import (
    IntelligentQueryRewriter,
    QueryRewrite,
    RewriteStep,
    RewriteRule,
    RewriteComplexity
)

from .query_mutator import (
    QueryMutationEngine,
    MutationResult,
    MutationRule,
    QueryAliasGenerator,
    MutationType
)

from .plan_predictor import (
    QueryPlanPredictor,
    ExecutionPlanPrediction,
    PlanNode,
    PlanNodeType,
    CostCategory
)

__all__ = [
    # Query Rewriter
    'IntelligentQueryRewriter',
    'QueryRewrite',
    'RewriteStep',
    'RewriteRule',
    'RewriteComplexity',

    # Query Mutator
    'QueryMutationEngine',
    'MutationResult',
    'MutationRule',
    'QueryAliasGenerator',
    'MutationType',

    # Plan Predictor
    'QueryPlanPredictor',
    'ExecutionPlanPrediction',
    'PlanNode',
    'PlanNodeType',
    'CostCategory',
]
