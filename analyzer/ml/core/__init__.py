"""
Core ML Infrastructure for QueryGrade

This package contains production-ready ML components that are essential
for the query grading system:

- hybrid_grader: Combines rule-based and ML predictions
- feature_extractor: Extracts features from SQL queries
- training_pipeline: Model training and deployment
- feedback_collector: Processes user feedback for learning
- model_manager: Centralized model loading and versioning

All modules in this package are production-ready and actively used.
"""

from .feature_extractor import FeatureExtractor
from .feedback_collector import FeedbackCollector
from .hybrid_grader import HybridQueryGrader
from .model_manager import (
    LoadedModel,
    ModelManager,
    ModelStatus,
    ModelType,
    get_model_manager,
)
from .training_pipeline import TrainingPipelineManager

__all__ = [
    "ModelManager",
    "get_model_manager",
    "LoadedModel",
    "ModelType",
    "ModelStatus",
    "HybridQueryGrader",
    "FeatureExtractor",
    "FeedbackCollector",
    "TrainingPipelineManager",
]
