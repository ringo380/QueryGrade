"""
ML Model Monitoring and Health for QueryGrade

This package contains model monitoring and maintenance tools:

- performance_tracker: Track model performance over time
- realtime_feedback: Real-time feedback processing
- confidence_analyzer: Model confidence analysis
- drift_detection: Performance monitoring and data drift detection
- retraining_system: Confidence-based retraining decision system

All monitoring modules are production-ready.
"""

from .confidence_analyzer import ConfidenceAnalyzer
from .drift_detection import DataDriftDetector, PerformanceMonitor
from .performance_tracker import (ABTestingFramework, ABTestResult,
                                  ModelComparison, ModelPerformanceManager,
                                  ModelSelector, ModelStatus,
                                  PerformanceMetrics, PerformanceTracker,
                                  SelectionCriteria)
from .realtime_feedback import (FeedbackBuffer, FeedbackEvent,
                                ModelUpdateEvent, OnlineLearningEngine,
                                RealTimeFeedbackProcessor)
from .retraining_system import (ConfidenceBasedRetrainingSystem,
                                ConfidenceMetrics, ModelHealthStatus,
                                RetrainingTrigger, TriggerReason,
                                TriggerUrgency, add_prediction_feedback,
                                evaluate_model_retraining_needs,
                                get_model_health_status,
                                update_feature_distribution)

__all__ = [
    # Performance Tracker
    "ModelPerformanceManager",
    "PerformanceTracker",
    "ModelSelector",
    "ABTestingFramework",
    "PerformanceMetrics",
    "ModelComparison",
    "ABTestResult",
    "ModelStatus",
    "SelectionCriteria",
    # Real-time Feedback
    "RealTimeFeedbackProcessor",
    "OnlineLearningEngine",
    "FeedbackBuffer",
    "FeedbackEvent",
    "ModelUpdateEvent",
    # Confidence Analysis
    "ConfidenceAnalyzer",
    # Drift Detection
    "PerformanceMonitor",
    "DataDriftDetector",
    # Retraining System
    "ConfidenceBasedRetrainingSystem",
    "TriggerReason",
    "TriggerUrgency",
    "ConfidenceMetrics",
    "RetrainingTrigger",
    "ModelHealthStatus",
    "evaluate_model_retraining_needs",
    "get_model_health_status",
    "add_prediction_feedback",
    "update_feature_distribution",
]
