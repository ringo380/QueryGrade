"""
Confidence-Based Retraining System

This module implements the main retraining decision system that evaluates
model health and determines when retraining should be triggered.
"""

import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List

import numpy as np
from django.core.cache import caches
from django.utils import timezone

from ...models import (
    MLModel,
)
from .confidence_analyzer import ConfidenceAnalyzer
from .drift_detection import DataDriftDetector, PerformanceMonitor

logger = logging.getLogger(__name__)


class TriggerReason(Enum):
    """Reasons for triggering model retraining."""

    LOW_CONFIDENCE = "low_confidence"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    FEEDBACK_DIVERGENCE = "feedback_divergence"
    TIME_BASED = "time_based"
    MANUAL_OVERRIDE = "manual_override"
    EMERGENCY = "emergency"


class TriggerUrgency(Enum):
    """Urgency levels for retraining triggers."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence metrics for model performance."""

    prediction_confidence: float  # Average confidence in predictions
    feedback_agreement: float  # Agreement between model and user feedback
    performance_stability: float  # Stability of performance over time
    data_coverage: float  # Coverage of feature space
    calibration_score: float  # How well confidence matches actual accuracy
    uncertainty_trend: str  # 'increasing', 'stable', 'decreasing'
    sample_count: int
    time_window_hours: int


@dataclass
class RetrainingTrigger:
    """Represents a retraining trigger event."""

    trigger_id: str
    reason: TriggerReason
    urgency: TriggerUrgency
    confidence_score: float
    evidence: Dict[str, Any]
    recommendation: str
    estimated_improvement: float
    cost_estimate: Dict[str, Any]
    timestamp: datetime
    auto_approved: bool = False
    executed: bool = False


@dataclass
class ModelHealthStatus:
    """Overall health status of the ML model."""

    overall_health: float  # 0-1 score
    confidence_health: float
    performance_health: float
    data_health: float
    feedback_health: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    recommendations: List[str]
    next_check: datetime


class ConfidenceBasedRetrainingSystem:
    """Main system for confidence-based retraining decisions."""

    def __init__(self):
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        self.data_drift_detector = DataDriftDetector()

        # Thresholds for different trigger conditions
        self.thresholds = {
            "confidence_threshold": 0.6,
            "performance_degradation_threshold": 0.15,
            "feedback_agreement_threshold": 0.7,
            "drift_threshold": 0.2,
            "calibration_threshold": 0.5,
        }

        # History and state
        self.trigger_history = deque(maxlen=100)
        self.last_full_evaluation = timezone.now()
        self.auto_approval_enabled = True

        # Cache
        self.cache = caches["default"]

    def evaluate_retraining_need(self) -> List[RetrainingTrigger]:
        """Evaluate if model retraining is needed based on confidence metrics."""
        triggers = []

        try:
            # Gather confidence metrics
            confidence_metrics = self._gather_confidence_metrics()

            # Check various trigger conditions
            triggers.extend(self._check_confidence_triggers(confidence_metrics))
            triggers.extend(self._check_performance_triggers())
            triggers.extend(self._check_drift_triggers())
            triggers.extend(self._check_feedback_triggers())
            triggers.extend(self._check_time_based_triggers())

            # Sort by urgency
            triggers.sort(key=lambda t: t.urgency.value, reverse=True)

            # Cache evaluation results
            self._cache_evaluation_results(confidence_metrics, triggers)

            self.last_full_evaluation = timezone.now()

        except Exception as e:
            logger.error(f"Error in retraining evaluation: {e}")

        return triggers

    def _gather_confidence_metrics(self) -> ConfidenceMetrics:
        """Gather comprehensive confidence metrics."""
        # Calculate prediction confidence
        prediction_confidence = (
            np.mean(self.confidence_analyzer.confidence_history)
            if self.confidence_analyzer.confidence_history
            else 0.5
        )

        # Calculate feedback agreement
        feedback_agreement = self.performance_monitor.calculate_feedback_agreement()

        # Calculate performance stability
        performance_trend = self.performance_monitor.calculate_performance_trend()
        performance_stability = 1.0 - abs(performance_trend.get("change", 0))

        # Calculate data coverage (simplified)
        data_coverage = min(
            1.0, len(self.confidence_analyzer.confidence_history) / 1000
        )

        # Calculate calibration score
        calibration_score = self.confidence_analyzer.calculate_calibration_score()

        # Analyze uncertainty trend
        uncertainty_trend = self.confidence_analyzer.analyze_confidence_trend()

        return ConfidenceMetrics(
            prediction_confidence=prediction_confidence,
            feedback_agreement=feedback_agreement,
            performance_stability=performance_stability,
            data_coverage=data_coverage,
            calibration_score=calibration_score,
            uncertainty_trend=uncertainty_trend,
            sample_count=len(self.confidence_analyzer.confidence_history),
            time_window_hours=24,
        )

    def _check_confidence_triggers(
        self, metrics: ConfidenceMetrics
    ) -> List[RetrainingTrigger]:
        """Check for confidence-based triggers."""
        triggers = []

        # Low overall confidence
        if metrics.prediction_confidence < self.thresholds["confidence_threshold"]:
            urgency = (
                TriggerUrgency.HIGH
                if metrics.prediction_confidence < 0.4
                else TriggerUrgency.MEDIUM
            )

            trigger = RetrainingTrigger(
                trigger_id=f"conf_{int(time.time())}",
                reason=TriggerReason.LOW_CONFIDENCE,
                urgency=urgency,
                confidence_score=metrics.prediction_confidence,
                evidence={
                    "avg_confidence": metrics.prediction_confidence,
                    "threshold": self.thresholds["confidence_threshold"],
                    "sample_count": metrics.sample_count,
                },
                recommendation="Retrain model to improve prediction confidence",
                estimated_improvement=0.3,
                cost_estimate={"time_hours": 2, "compute_cost": 50},
                timestamp=timezone.now(),
                auto_approved=urgency == TriggerUrgency.HIGH,
            )
            triggers.append(trigger)

        # Poor calibration
        if metrics.calibration_score < self.thresholds["calibration_threshold"]:
            trigger = RetrainingTrigger(
                trigger_id=f"calib_{int(time.time())}",
                reason=TriggerReason.LOW_CONFIDENCE,
                urgency=TriggerUrgency.MEDIUM,
                confidence_score=metrics.calibration_score,
                evidence={
                    "calibration_score": metrics.calibration_score,
                    "threshold": self.thresholds["calibration_threshold"],
                },
                recommendation="Retrain with calibration focus to improve confidence reliability",
                estimated_improvement=0.2,
                cost_estimate={"time_hours": 1.5, "compute_cost": 40},
                timestamp=timezone.now(),
            )
            triggers.append(trigger)

        return triggers

    def _check_performance_triggers(self) -> List[RetrainingTrigger]:
        """Check for performance degradation triggers."""
        triggers = []

        performance_trend = self.performance_monitor.calculate_performance_trend()

        if (
            performance_trend["trend"] == "degrading"
            and abs(performance_trend["change"])
            > self.thresholds["performance_degradation_threshold"]
        ):

            urgency = (
                TriggerUrgency.CRITICAL
                if abs(performance_trend["change"]) > 0.3
                else TriggerUrgency.HIGH
            )

            trigger = RetrainingTrigger(
                trigger_id=f"perf_{int(time.time())}",
                reason=TriggerReason.PERFORMANCE_DEGRADATION,
                urgency=urgency,
                confidence_score=1.0 - abs(performance_trend["change"]),
                evidence={
                    "performance_change": performance_trend["change"],
                    "trend": performance_trend["trend"],
                    "threshold": self.thresholds["performance_degradation_threshold"],
                },
                recommendation="Immediate retraining needed due to performance degradation",
                estimated_improvement=0.4,
                cost_estimate={"time_hours": 3, "compute_cost": 75},
                timestamp=timezone.now(),
                auto_approved=urgency == TriggerUrgency.CRITICAL,
            )
            triggers.append(trigger)

        return triggers

    def _check_drift_triggers(self) -> List[RetrainingTrigger]:
        """Check for data drift triggers."""
        triggers = []

        drift_results = self.data_drift_detector.detect_drift()

        if (
            drift_results["drift_detected"]
            and drift_results["overall_drift"] > self.thresholds["drift_threshold"]
        ):

            urgency = (
                TriggerUrgency.HIGH
                if drift_results["overall_drift"] > 0.4
                else TriggerUrgency.MEDIUM
            )

            trigger = RetrainingTrigger(
                trigger_id=f"drift_{int(time.time())}",
                reason=TriggerReason.DATA_DRIFT,
                urgency=urgency,
                confidence_score=1.0 - drift_results["overall_drift"],
                evidence={
                    "overall_drift": drift_results["overall_drift"],
                    "significant_drifts": drift_results["significant_drifts"],
                    "threshold": self.thresholds["drift_threshold"],
                },
                recommendation="Retrain model to adapt to data distribution changes",
                estimated_improvement=0.25,
                cost_estimate={"time_hours": 2.5, "compute_cost": 60},
                timestamp=timezone.now(),
            )
            triggers.append(trigger)

        return triggers

    def _check_feedback_triggers(self) -> List[RetrainingTrigger]:
        """Check for feedback divergence triggers."""
        triggers = []

        feedback_agreement = self.performance_monitor.calculate_feedback_agreement()

        if feedback_agreement < self.thresholds["feedback_agreement_threshold"]:
            urgency = (
                TriggerUrgency.MEDIUM
                if feedback_agreement < 0.5
                else TriggerUrgency.LOW
            )

            trigger = RetrainingTrigger(
                trigger_id=f"feedback_{int(time.time())}",
                reason=TriggerReason.FEEDBACK_DIVERGENCE,
                urgency=urgency,
                confidence_score=feedback_agreement,
                evidence={
                    "feedback_agreement": feedback_agreement,
                    "threshold": self.thresholds["feedback_agreement_threshold"],
                },
                recommendation="Retrain to better align with user feedback patterns",
                estimated_improvement=0.2,
                cost_estimate={"time_hours": 2, "compute_cost": 45},
                timestamp=timezone.now(),
            )
            triggers.append(trigger)

        return triggers

    def _check_time_based_triggers(self) -> List[RetrainingTrigger]:
        """Check for time-based triggers."""
        triggers = []

        # Get last model update time
        try:
            latest_model = MLModel.objects.filter(
                model_type="HYBRID_SCORER", status="ACTIVE"
            ).first()

            if latest_model:
                time_since_update = timezone.now() - latest_model.deployed_at
                max_age = timedelta(days=7)  # Maximum model age

                if time_since_update > max_age:
                    trigger = RetrainingTrigger(
                        trigger_id=f"time_{int(time.time())}",
                        reason=TriggerReason.TIME_BASED,
                        urgency=TriggerUrgency.LOW,
                        confidence_score=0.8,
                        evidence={
                            "days_since_update": time_since_update.days,
                            "max_days": max_age.days,
                        },
                        recommendation="Regular scheduled retraining to maintain model freshness",
                        estimated_improvement=0.1,
                        cost_estimate={"time_hours": 1, "compute_cost": 30},
                        timestamp=timezone.now(),
                        auto_approved=True,
                    )
                    triggers.append(trigger)

        except Exception as e:
            logger.warning(f"Error checking time-based triggers: {e}")

        return triggers

    def _cache_evaluation_results(
        self, metrics: ConfidenceMetrics, triggers: List[RetrainingTrigger]
    ):
        """Cache evaluation results for monitoring."""
        results = {
            "metrics": asdict(metrics),
            "triggers": [asdict(t) for t in triggers],
            "evaluation_time": timezone.now().isoformat(),
            "trigger_count": len(triggers),
            "highest_urgency": (
                max([t.urgency.value for t in triggers]) if triggers else 0
            ),
        }

        self.cache.set("retraining_evaluation_results", results, timeout=3600)

    def get_model_health_status(self) -> ModelHealthStatus:
        """Get overall model health status."""
        try:
            metrics = self._gather_confidence_metrics()
            triggers = self.evaluate_retraining_need()

            # Calculate health scores
            confidence_health = metrics.prediction_confidence
            performance_health = metrics.performance_stability
            data_health = 1.0 - (
                self.data_drift_detector.detect_drift().get("overall_drift", 0)
            )
            feedback_health = metrics.feedback_agreement

            # Overall health (weighted average)
            overall_health = (
                confidence_health * 0.3
                + performance_health * 0.25
                + data_health * 0.25
                + feedback_health * 0.2
            )

            # Determine risk level
            if overall_health > 0.8:
                risk_level = "low"
            elif overall_health > 0.6:
                risk_level = "medium"
            elif overall_health > 0.4:
                risk_level = "high"
            else:
                risk_level = "critical"

            # Generate recommendations
            recommendations = self._generate_health_recommendations(metrics, triggers)

            return ModelHealthStatus(
                overall_health=overall_health,
                confidence_health=confidence_health,
                performance_health=performance_health,
                data_health=data_health,
                feedback_health=feedback_health,
                risk_level=risk_level,
                recommendations=recommendations,
                next_check=timezone.now() + timedelta(hours=6),
            )

        except Exception as e:
            logger.error(f"Error calculating model health: {e}")
            return ModelHealthStatus(
                overall_health=0.5,
                confidence_health=0.5,
                performance_health=0.5,
                data_health=0.5,
                feedback_health=0.5,
                risk_level="unknown",
                recommendations=["Error calculating health status"],
                next_check=timezone.now() + timedelta(hours=1),
            )

    def _generate_health_recommendations(
        self, metrics: ConfidenceMetrics, triggers: List[RetrainingTrigger]
    ) -> List[str]:
        """Generate health recommendations based on metrics and triggers."""
        recommendations = []

        if metrics.prediction_confidence < 0.7:
            recommendations.append("Monitor prediction confidence closely")

        if metrics.calibration_score < 0.6:
            recommendations.append("Consider calibration-focused training")

        if triggers:
            urgent_triggers = [t for t in triggers if t.urgency.value >= 3]
            if urgent_triggers:
                recommendations.append(
                    f"Address {len(urgent_triggers)} urgent retraining triggers"
                )

        if metrics.uncertainty_trend == "increasing":
            recommendations.append("Investigate increasing uncertainty trend")

        if not recommendations:
            recommendations.append("Model health is good, continue monitoring")

        return recommendations


# Global instance for the application
confidence_system = ConfidenceBasedRetrainingSystem()


# Django integration functions
def evaluate_model_retraining_needs() -> List[RetrainingTrigger]:
    """Evaluate if model retraining is needed."""
    return confidence_system.evaluate_retraining_need()


def get_model_health_status() -> ModelHealthStatus:
    """Get current model health status."""
    return confidence_system.get_model_health_status()


def add_prediction_feedback(
    query_id: int,
    prediction: float,
    confidence: float,
    actual_score: float = None,
    user_feedback: float = None,
):
    """Add prediction result for confidence analysis."""
    if actual_score is not None:
        confidence_system.confidence_analyzer.add_prediction_result(
            prediction, confidence, actual_score, query_id
        )

    if user_feedback is not None:
        confidence_system.performance_monitor.add_feedback_result(
            prediction, user_feedback
        )


def update_feature_distribution(features: List[float], feature_names: List[str]):
    """Update feature distribution for drift detection."""
    confidence_system.data_drift_detector.update_feature_distribution(
        features, feature_names
    )


# Usage example and testing
if __name__ == "__main__":
    # Test the confidence system
    system = ConfidenceBasedRetrainingSystem()

    # Simulate some prediction results
    for i in range(100):
        prediction = np.random.normal(70, 15)
        confidence = np.random.uniform(0.3, 0.9)
        actual = prediction + np.random.normal(0, 10)

        system.confidence_analyzer.add_prediction_result(
            prediction, confidence, actual, i
        )

        if i % 10 == 0:
            user_feedback = actual + np.random.normal(0, 5)
            system.performance_monitor.add_feedback_result(prediction, user_feedback)

    # Evaluate retraining needs
    triggers = system.evaluate_retraining_need()
    print(f"Found {len(triggers)} retraining triggers")

    for trigger in triggers:
        print(f"Trigger: {trigger.reason.value} - Urgency: {trigger.urgency.name}")

    # Get health status
    health = system.get_model_health_status()
    print(f"Model health: {health.overall_health:.2f} - Risk: {health.risk_level}")
