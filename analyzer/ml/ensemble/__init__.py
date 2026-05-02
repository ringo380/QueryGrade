"""
Ensemble and Advanced ML for QueryGrade

This package contains ensemble learning and advanced ML features:

- multi_model: Multi-model ensemble management
- voting_system: Ensemble voting mechanisms

All ensemble modules are production-ready.
"""

from .multi_model import (EnsembleResult, ModelConfiguration, ModelPerformance,
                          ModelType, MultiModelEnsemble, NeuralNetworkModel,
                          RandomForestModel, XGBoostModel)
from .voting_system import (AggregationMethod, ConsensusAnalyzer,
                            EnsembleMetrics, EnsembleVotingSystem,
                            ModelPrediction, ModelWeightCalculator,
                            VotingResult, VotingStrategies, VotingStrategy)

__all__ = [
    # Multi-Model Ensemble
    "MultiModelEnsemble",
    "EnsembleResult",
    "ModelPerformance",
    "ModelConfiguration",
    "ModelType",
    "RandomForestModel",
    "XGBoostModel",
    "NeuralNetworkModel",
    # Voting System
    "EnsembleVotingSystem",
    "VotingResult",
    "ModelPrediction",
    "EnsembleMetrics",
    "VotingStrategy",
    "AggregationMethod",
    "ModelWeightCalculator",
    "ConsensusAnalyzer",
    "VotingStrategies",
]
