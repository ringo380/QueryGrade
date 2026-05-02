"""
Performance Monitoring and Data Drift Detection

This module monitors model performance over time and detects data distribution
changes that might indicate model degradation or need for retraining.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from django.utils import timezone

try:
    from scipy import stats

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors model performance metrics over time."""

    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        self.feedback_history = deque(maxlen=1000)
        self.baseline_performance = None

    def add_performance_metric(
        self, metric_name: str, value: float, timestamp: datetime = None
    ):
        """Add a performance metric."""
        if timestamp is None:
            timestamp = timezone.now()

        self.performance_history.append(
            {"metric": metric_name, "value": value, "timestamp": timestamp}
        )

        # Update baseline if this is early in the model's life
        if len(self.performance_history) == 100:
            recent_values = [p["value"] for p in list(self.performance_history)[-50:]]
            self.baseline_performance = np.mean(recent_values)

    def add_feedback_result(self, predicted_score: float, user_feedback_score: float):
        """Add feedback comparison result."""
        difference = abs(predicted_score - user_feedback_score)
        agreement = max(0.0, 1.0 - (difference / 100.0))

        self.feedback_history.append(
            {
                "predicted": predicted_score,
                "feedback": user_feedback_score,
                "agreement": agreement,
                "timestamp": timezone.now(),
            }
        )

    def calculate_performance_trend(self, window_size: int = 100) -> Dict[str, Any]:
        """Calculate performance trend over recent window."""
        if len(self.performance_history) < window_size:
            return {"trend": "insufficient_data", "change": 0.0}

        recent_metrics = list(self.performance_history)[-window_size:]
        values = [m["value"] for m in recent_metrics]

        if len(values) < 10:
            return {"trend": "insufficient_data", "change": 0.0}

        # Calculate trend
        mid_point = len(values) // 2
        first_half = np.mean(values[:mid_point])
        second_half = np.mean(values[mid_point:])

        change = (second_half - first_half) / first_half if first_half > 0 else 0

        if change < -0.1:
            trend = "degrading"
        elif change > 0.1:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change": change,
            "current_avg": second_half,
            "baseline_avg": first_half,
        }

    def calculate_feedback_agreement(self, window_size: int = 50) -> float:
        """Calculate agreement between model and user feedback."""
        if len(self.feedback_history) < window_size:
            recent_feedback = list(self.feedback_history)
        else:
            recent_feedback = list(self.feedback_history)[-window_size:]

        if not recent_feedback:
            return 0.5  # Neutral agreement

        agreements = [f["agreement"] for f in recent_feedback]
        return np.mean(agreements)


class DataDriftDetector:
    """Detects changes in data distribution that might affect model performance."""

    def __init__(self):
        self.feature_distributions = {}
        self.baseline_established = False
        self.drift_threshold = 0.1

    def update_feature_distribution(
        self, features: List[float], feature_names: List[str]
    ):
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
            return {"drift_detected": False, "reason": "baseline_not_established"}

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
            "drift_detected": len(significant_drifts) > 0,
            "drift_scores": drift_results,
            "significant_drifts": significant_drifts,
            "overall_drift": (
                np.mean(list(drift_results.values())) if drift_results else 0.0
            ),
        }

    def _calculate_drift_score(
        self, baseline: List[float], recent: List[float]
    ) -> float:
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
