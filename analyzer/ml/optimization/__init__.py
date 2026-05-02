"""
Query Optimization Engines for QueryGrade ML

This package contains query optimization and rewriting tools:

- query_rewriter: Intelligent query rewriting
- query_mutator: Query mutation for testing
- plan_predictor: Execution plan prediction

All optimization modules are production-ready.
"""

from .plan_predictor import (CostCategory, ExecutionPlanPrediction, PlanNode,
                             PlanNodeType, QueryPlanPredictor)
from .query_mutator import (MutationResult, MutationRule, MutationType,
                            QueryAliasGenerator, QueryMutationEngine)
from .query_rewriter import (IntelligentQueryRewriter, QueryRewrite,
                             RewriteComplexity, RewriteRule, RewriteStep)

__all__ = [
    # Query Rewriter
    "IntelligentQueryRewriter",
    "QueryRewrite",
    "RewriteStep",
    "RewriteRule",
    "RewriteComplexity",
    # Query Mutator
    "QueryMutationEngine",
    "MutationResult",
    "MutationRule",
    "QueryAliasGenerator",
    "MutationType",
    # Plan Predictor
    "QueryPlanPredictor",
    "ExecutionPlanPrediction",
    "PlanNode",
    "PlanNodeType",
    "CostCategory",
]
