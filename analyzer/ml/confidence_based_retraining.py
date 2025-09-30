"""
Confidence-Based Retraining Triggers for QueryGrade ML System

This module implements intelligent retraining triggers based on model confidence,
performance metrics, and data distribution changes to ensure optimal model performance
through automated decision-making about when to retrain models.
"""

import logging
import numpy as np
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import threading
import time

from django.core.cache import caches
from django.utils import timezone
from django.db import transaction
from django.db.models import Avg, Count, Max, Min

try:
    from scipy import stats
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scipy/sklearn not available. Some confidence analysis features will be limited.")

from ..models import (
    Query, QueryAnalysis, UserQueryHistory, MLModel, LearningMetrics,
    TrainingData, FeedbackLearning
)

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
    feedback_agreement: float     # Agreement between model and user feedback
    performance_stability: float  # Stability of performance over time
    data_coverage: float         # Coverage of feature space
    calibration_score: float     # How well confidence matches actual accuracy
    uncertainty_trend: str       # 'increasing', 'stable', 'decreasing'
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


class ConfidenceAnalyzer:
    """Analyzes model confidence and calibration."""

    def __init__(self):
        self.confidence_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=1000)
        self.prediction_cache = {}

    def add_prediction_result(self, prediction: float, confidence: float,
                            actual: float, query_id: int):
        """Add prediction result for confidence analysis."""
        accuracy = 1.0 - abs(prediction - actual) / max(100.0, abs(actual))
        accuracy = max(0.0, accuracy)

        self.confidence_history.append(confidence)
        self.accuracy_history.append(accuracy)

        # Cache for detailed analysis
        self.prediction_cache[query_id] = {
            'prediction': prediction,
            'confidence': confidence,
            'actual': actual,
            'accuracy': accuracy,
            'timestamp': timezone.now()
        }

    def calculate_calibration_score(self) -> float:
        """Calculate how well confidence correlates with actual accuracy."""
        if len(self.confidence_history) < 20:
            return 0.5  # Neutral score for insufficient data

        confidences = list(self.confidence_history)
        accuracies = list(self.accuracy_history)

        if SKLEARN_AVAILABLE:
            try:
                # Calculate correlation between confidence and accuracy
                correlation, _ = stats.pearsonr(confidences, accuracies)
                return max(0.0, correlation)
            except:
                pass

        # Fallback: simple binned analysis
        return self._simple_calibration_analysis(confidences, accuracies)

    def _simple_calibration_analysis(self, confidences: List[float],
                                   accuracies: List[float]) -> float:
        """Simple calibration analysis without scipy."""
        # Bin predictions by confidence level
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        calibration_errors = []

        for low, high in bins:
            bin_confidences = []
            bin_accuracies = []

            for conf, acc in zip(confidences, accuracies):
                if low <= conf < high:
                    bin_confidences.append(conf)
                    bin_accuracies.append(acc)

            if bin_confidences:
                avg_confidence = np.mean(bin_confidences)
                avg_accuracy = np.mean(bin_accuracies)
                error = abs(avg_confidence - avg_accuracy)
                calibration_errors.append(error)

        if calibration_errors:
            # Return inverse of average calibration error
            avg_error = np.mean(calibration_errors)
            return max(0.0, 1.0 - avg_error)
        else:
            return 0.5

    def analyze_confidence_trend(self, window_size: int = 50) -> str:
        """Analyze trend in confidence over recent predictions."""
        if len(self.confidence_history) < window_size:
            return "insufficient_data"

        recent_confidences = list(self.confidence_history)[-window_size:]
        mid_point = len(recent_confidences) // 2

        first_half = np.mean(recent_confidences[:mid_point])
        second_half = np.mean(recent_confidences[mid_point:])

        if second_half < first_half * 0.9:
            return "decreasing"
        elif second_half > first_half * 1.1:
            return "increasing"
        else:
            return "stable"

    def get_low_confidence_queries(self, threshold: float = 0.5) -> List[int]:
        """Get query IDs with low confidence predictions."""
        low_confidence_queries = []

        for query_id, data in self.prediction_cache.items():
            if data['confidence'] < threshold:
                low_confidence_queries.append(query_id)

        return low_confidence_queries


class PerformanceMonitor:
    """Monitors model performance metrics over time."""

    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        self.feedback_history = deque(maxlen=1000)
        self.baseline_performance = None

    def add_performance_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Add a performance metric."""
        if timestamp is None:
            timestamp = timezone.now()

        self.performance_history.append({
            'metric': metric_name,
            'value': value,
            'timestamp': timestamp
        })

        # Update baseline if this is early in the model's life
        if len(self.performance_history) == 100:
            recent_values = [p['value'] for p in list(self.performance_history)[-50:]]
            self.baseline_performance = np.mean(recent_values)

    def add_feedback_result(self, predicted_score: float, user_feedback_score: float):
        """Add feedback comparison result."""
        difference = abs(predicted_score - user_feedback_score)
        agreement = max(0.0, 1.0 - (difference / 100.0))

        self.feedback_history.append({
            'predicted': predicted_score,
            'feedback': user_feedback_score,
            'agreement': agreement,
            'timestamp': timezone.now()
        })

    def calculate_performance_trend(self, window_size: int = 100) -> Dict[str, Any]:
        """Calculate performance trend over recent window."""
        if len(self.performance_history) < window_size:
            return {'trend': 'insufficient_data', 'change': 0.0}

        recent_metrics = list(self.performance_history)[-window_size:]
        values = [m['value'] for m in recent_metrics]

        if len(values) < 10:
            return {'trend': 'insufficient_data', 'change': 0.0}

        # Calculate trend
        mid_point = len(values) // 2
        first_half = np.mean(values[:mid_point])
        second_half = np.mean(values[mid_point:])

        change = (second_half - first_half) / first_half if first_half > 0 else 0

        if change < -0.1:
            trend = 'degrading'
        elif change > 0.1:
            trend = 'improving'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'change': change,
            'current_avg': second_half,
            'baseline_avg': first_half
        }

    def calculate_feedback_agreement(self, window_size: int = 50) -> float:
        """Calculate agreement between model and user feedback."""
        if len(self.feedback_history) < window_size:
            recent_feedback = list(self.feedback_history)
        else:
            recent_feedback = list(self.feedback_history)[-window_size:]

        if not recent_feedback:
            return 0.5  # Neutral agreement

        agreements = [f['agreement'] for f in recent_feedback]
        return np.mean(agreements)


class DataDriftDetector:
    """Detects changes in data distribution that might affect model performance."""

    def __init__(self):
        self.feature_distributions = {}
        self.baseline_established = False
        self.drift_threshold = 0.1

    def update_feature_distribution(self, features: List[float], feature_names: List[str]):
        """Update feature distribution tracking."""
        for i, (feature_value, feature_name) in enumerate(zip(features, feature_names)):
            if feature_name not in self.feature_distributions:
                self.feature_distributions[feature_name] = deque(maxlen=1000)

            self.feature_distributions[feature_name].append(feature_value)

        # Establish baseline after sufficient data
        if not self.baseline_established and all(
            len(dist) >= 100 for dist in self.feature_distributions.values()
        ):
            self.baseline_established = True
            logger.info("Feature distribution baseline established")

    def detect_drift(self) -> Dict[str, Any]:
        """Detect if significant drift has occurred in feature distributions."""
        if not self.baseline_established:
            return {'drift_detected': False, 'reason': 'baseline_not_established'}

        drift_results = {}
        significant_drifts = []

        for feature_name, values in self.feature_distributions.items():
            if len(values) < 200:
                continue

            # Split into baseline and recent data
            all_values = list(values)
            baseline_size = min(100, len(all_values) // 2)
            baseline_data = all_values[:baseline_size]
            recent_data = all_values[-baseline_size:]

            # Calculate drift score
            drift_score = self._calculate_drift_score(baseline_data, recent_data)
            drift_results[feature_name] = drift_score

            if drift_score > self.drift_threshold:
                significant_drifts.append((feature_name, drift_score))

        return {
            'drift_detected': len(significant_drifts) > 0,
            'drift_scores': drift_results,
            'significant_drifts': significant_drifts,
            'overall_drift': np.mean(list(drift_results.values())) if drift_results else 0.0
        }

    def _calculate_drift_score(self, baseline: List[float], recent: List[float]) -> float:
        """Calculate drift score between two distributions."""
        if not baseline or not recent:
            return 0.0

        try:
            # Statistical test for distribution difference
            if SKLEARN_AVAILABLE:
                # Use Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(baseline, recent)
                return statistic
            else:
                # Simple difference in means as fallback
                baseline_mean = np.mean(baseline)
                recent_mean = np.mean(recent)
                baseline_std = np.std(baseline)

                if baseline_std > 0:
                    normalized_diff = abs(recent_mean - baseline_mean) / baseline_std
                    return min(1.0, normalized_diff / 3.0)  # Normalize to 0-1
                else:
                    return 0.0

        except Exception as e:
            logger.warning(f"Error calculating drift score: {e}")
            return 0.0


class ConfidenceBasedRetrainingSystem:
    """Main system for confidence-based retraining decisions."""

    def __init__(self):
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        self.data_drift_detector = DataDriftDetector()

        # Thresholds for different trigger conditions
        self.thresholds = {
            'confidence_threshold': 0.6,
            'performance_degradation_threshold': 0.15,
            'feedback_agreement_threshold': 0.7,
            'drift_threshold': 0.2,
            'calibration_threshold': 0.5
        }

        # History and state
        self.trigger_history = deque(maxlen=100)
        self.last_full_evaluation = timezone.now()
        self.auto_approval_enabled = True

        # Cache
        self.cache = caches['default']

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
        prediction_confidence = np.mean(self.confidence_analyzer.confidence_history) if \
            self.confidence_analyzer.confidence_history else 0.5

        # Calculate feedback agreement
        feedback_agreement = self.performance_monitor.calculate_feedback_agreement()

        # Calculate performance stability
        performance_trend = self.performance_monitor.calculate_performance_trend()
        performance_stability = 1.0 - abs(performance_trend.get('change', 0))

        # Calculate data coverage (simplified)
        data_coverage = min(1.0, len(self.confidence_analyzer.confidence_history) / 1000)

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
            time_window_hours=24
        )

    def _check_confidence_triggers(self, metrics: ConfidenceMetrics) -> List[RetrainingTrigger]:
        """Check for confidence-based triggers."""
        triggers = []

        # Low overall confidence
        if metrics.prediction_confidence < self.thresholds['confidence_threshold']:
            urgency = TriggerUrgency.HIGH if metrics.prediction_confidence < 0.4 else TriggerUrgency.MEDIUM

            trigger = RetrainingTrigger(
                trigger_id=f"conf_{int(time.time())}",
                reason=TriggerReason.LOW_CONFIDENCE,
                urgency=urgency,
                confidence_score=metrics.prediction_confidence,
                evidence={
                    'avg_confidence': metrics.prediction_confidence,
                    'threshold': self.thresholds['confidence_threshold'],
                    'sample_count': metrics.sample_count
                },
                recommendation="Retrain model to improve prediction confidence",
                estimated_improvement=0.3,
                cost_estimate={'time_hours': 2, 'compute_cost': 50},
                timestamp=timezone.now(),
                auto_approved=urgency == TriggerUrgency.HIGH
            )
            triggers.append(trigger)

        # Poor calibration
        if metrics.calibration_score < self.thresholds['calibration_threshold']:
            trigger = RetrainingTrigger(
                trigger_id=f"calib_{int(time.time())}",
                reason=TriggerReason.LOW_CONFIDENCE,
                urgency=TriggerUrgency.MEDIUM,
                confidence_score=metrics.calibration_score,
                evidence={
                    'calibration_score': metrics.calibration_score,
                    'threshold': self.thresholds['calibration_threshold']
                },
                recommendation="Retrain with calibration focus to improve confidence reliability",
                estimated_improvement=0.2,
                cost_estimate={'time_hours': 1.5, 'compute_cost': 40},
                timestamp=timezone.now()
            )
            triggers.append(trigger)

        return triggers

    def _check_performance_triggers(self) -> List[RetrainingTrigger]:
        """Check for performance degradation triggers."""
        triggers = []

        performance_trend = self.performance_monitor.calculate_performance_trend()

        if (performance_trend['trend'] == 'degrading' and
            abs(performance_trend['change']) > self.thresholds['performance_degradation_threshold']):

            urgency = TriggerUrgency.CRITICAL if abs(performance_trend['change']) > 0.3 else TriggerUrgency.HIGH

            trigger = RetrainingTrigger(
                trigger_id=f"perf_{int(time.time())}",
                reason=TriggerReason.PERFORMANCE_DEGRADATION,
                urgency=urgency,
                confidence_score=1.0 - abs(performance_trend['change']),
                evidence={
                    'performance_change': performance_trend['change'],
                    'trend': performance_trend['trend'],
                    'threshold': self.thresholds['performance_degradation_threshold']
                },
                recommendation="Immediate retraining needed due to performance degradation",
                estimated_improvement=0.4,
                cost_estimate={'time_hours': 3, 'compute_cost': 75},
                timestamp=timezone.now(),
                auto_approved=urgency == TriggerUrgency.CRITICAL
            )
            triggers.append(trigger)

        return triggers

    def _check_drift_triggers(self) -> List[RetrainingTrigger]:
        """Check for data drift triggers."""
        triggers = []

        drift_results = self.data_drift_detector.detect_drift()

        if (drift_results['drift_detected'] and
            drift_results['overall_drift'] > self.thresholds['drift_threshold']):

            urgency = TriggerUrgency.HIGH if drift_results['overall_drift'] > 0.4 else TriggerUrgency.MEDIUM

            trigger = RetrainingTrigger(
                trigger_id=f"drift_{int(time.time())}",
                reason=TriggerReason.DATA_DRIFT,
                urgency=urgency,
                confidence_score=1.0 - drift_results['overall_drift'],
                evidence={
                    'overall_drift': drift_results['overall_drift'],
                    'significant_drifts': drift_results['significant_drifts'],
                    'threshold': self.thresholds['drift_threshold']
                },
                recommendation="Retrain model to adapt to data distribution changes",
                estimated_improvement=0.25,
                cost_estimate={'time_hours': 2.5, 'compute_cost': 60},
                timestamp=timezone.now()
            )
            triggers.append(trigger)

        return triggers

    def _check_feedback_triggers(self) -> List[RetrainingTrigger]:
        """Check for feedback divergence triggers."""
        triggers = []

        feedback_agreement = self.performance_monitor.calculate_feedback_agreement()

        if feedback_agreement < self.thresholds['feedback_agreement_threshold']:
            urgency = TriggerUrgency.MEDIUM if feedback_agreement < 0.5 else TriggerUrgency.LOW

            trigger = RetrainingTrigger(
                trigger_id=f"feedback_{int(time.time())}",
                reason=TriggerReason.FEEDBACK_DIVERGENCE,
                urgency=urgency,
                confidence_score=feedback_agreement,
                evidence={
                    'feedback_agreement': feedback_agreement,
                    'threshold': self.thresholds['feedback_agreement_threshold']
                },
                recommendation="Retrain to better align with user feedback patterns",
                estimated_improvement=0.2,
                cost_estimate={'time_hours': 2, 'compute_cost': 45},
                timestamp=timezone.now()
            )
            triggers.append(trigger)

        return triggers

    def _check_time_based_triggers(self) -> List[RetrainingTrigger]:
        """Check for time-based triggers."""
        triggers = []

        # Get last model update time
        try:
            latest_model = MLModel.objects.filter(
                model_type='HYBRID_SCORER',
                status='ACTIVE'
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
                            'days_since_update': time_since_update.days,
                            'max_days': max_age.days
                        },
                        recommendation="Regular scheduled retraining to maintain model freshness",
                        estimated_improvement=0.1,
                        cost_estimate={'time_hours': 1, 'compute_cost': 30},
                        timestamp=timezone.now(),
                        auto_approved=True
                    )
                    triggers.append(trigger)

        except Exception as e:
            logger.warning(f"Error checking time-based triggers: {e}")

        return triggers

    def _cache_evaluation_results(self, metrics: ConfidenceMetrics, triggers: List[RetrainingTrigger]):
        """Cache evaluation results for monitoring."""
        results = {
            'metrics': asdict(metrics),
            'triggers': [asdict(t) for t in triggers],
            'evaluation_time': timezone.now().isoformat(),
            'trigger_count': len(triggers),
            'highest_urgency': max([t.urgency.value for t in triggers]) if triggers else 0
        }

        self.cache.set('retraining_evaluation_results', results, timeout=3600)

    def get_model_health_status(self) -> ModelHealthStatus:
        """Get overall model health status."""
        try:
            metrics = self._gather_confidence_metrics()
            triggers = self.evaluate_retraining_need()

            # Calculate health scores
            confidence_health = metrics.prediction_confidence
            performance_health = metrics.performance_stability
            data_health = 1.0 - (self.data_drift_detector.detect_drift().get('overall_drift', 0))
            feedback_health = metrics.feedback_agreement

            # Overall health (weighted average)
            overall_health = (
                confidence_health * 0.3 +
                performance_health * 0.25 +
                data_health * 0.25 +
                feedback_health * 0.2
            )

            # Determine risk level
            if overall_health > 0.8:
                risk_level = 'low'
            elif overall_health > 0.6:
                risk_level = 'medium'
            elif overall_health > 0.4:
                risk_level = 'high'
            else:
                risk_level = 'critical'

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
                next_check=timezone.now() + timedelta(hours=6)
            )

        except Exception as e:
            logger.error(f"Error calculating model health: {e}")
            return ModelHealthStatus(
                overall_health=0.5,
                confidence_health=0.5,
                performance_health=0.5,
                data_health=0.5,
                feedback_health=0.5,
                risk_level='unknown',
                recommendations=['Error calculating health status'],
                next_check=timezone.now() + timedelta(hours=1)
            )

    def _generate_health_recommendations(self, metrics: ConfidenceMetrics,
                                       triggers: List[RetrainingTrigger]) -> List[str]:
        """Generate health recommendations based on metrics and triggers."""
        recommendations = []

        if metrics.prediction_confidence < 0.7:
            recommendations.append("Monitor prediction confidence closely")

        if metrics.calibration_score < 0.6:
            recommendations.append("Consider calibration-focused training")

        if triggers:
            urgent_triggers = [t for t in triggers if t.urgency.value >= 3]
            if urgent_triggers:
                recommendations.append(f"Address {len(urgent_triggers)} urgent retraining triggers")

        if metrics.uncertainty_trend == 'increasing':
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


def add_prediction_feedback(query_id: int, prediction: float, confidence: float,
                          actual_score: float = None, user_feedback: float = None):
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
    confidence_system.data_drift_detector.update_feature_distribution(features, feature_names)


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
    print(f"Recommendations: {health.recommendations}")