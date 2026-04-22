"""
Confidence Analysis for ML Model Retraining

This module analyzes model confidence and calibration to detect when
model predictions become unreliable and retraining may be needed.
"""

import logging
from collections import deque
from typing import List

import numpy as np
from django.utils import timezone

try:
    from scipy import stats

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConfidenceAnalyzer:
    """Analyzes model confidence and calibration."""

    def __init__(self):
        self.confidence_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=1000)
        self.prediction_cache = {}

    def add_prediction_result(
        self, prediction: float, confidence: float, actual: float, query_id: int
    ):
        """Add prediction result for confidence analysis."""
        accuracy = 1.0 - abs(prediction - actual) / max(100.0, abs(actual))
        accuracy = max(0.0, accuracy)

        self.confidence_history.append(confidence)
        self.accuracy_history.append(accuracy)

        # Cache for detailed analysis
        self.prediction_cache[query_id] = {
            "prediction": prediction,
            "confidence": confidence,
            "actual": actual,
            "accuracy": accuracy,
            "timestamp": timezone.now(),
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

    def _simple_calibration_analysis(
        self, confidences: List[float], accuracies: List[float]
    ) -> float:
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
            if data["confidence"] < threshold:
                low_confidence_queries.append(query_id)

        return low_confidence_queries
