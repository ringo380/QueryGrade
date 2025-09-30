"""
Model Performance Tracking and Automatic Selection for QueryGrade ML System

This module implements comprehensive performance tracking for ML models with
automatic model selection, A/B testing capabilities, and intelligent model
lifecycle management.
"""

import logging
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import threading
import time
import statistics

from django.core.cache import caches
from django.utils import timezone
from django.db import transaction
from django.db.models import Avg, Count, Max, Min

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy import stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn/scipy not available. Some metrics will be limited.")

from ..models import Query, QueryAnalysis, MLModel, LearningMetrics

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of models in the system."""
    ACTIVE = "active"
    CANDIDATE = "candidate"
    DEPRECATED = "deprecated"
    TESTING = "testing"
    FAILED = "failed"
    RETIRED = "retired"


class SelectionCriteria(Enum):
    """Criteria for automatic model selection."""
    ACCURACY = "accuracy"
    SPEED = "speed"
    RELIABILITY = "reliability"
    CALIBRATION = "calibration"
    ROBUSTNESS = "robustness"
    BALANCED = "balanced"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a model."""
    model_id: str
    model_type: str
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    calibration_score: float
    prediction_speed_ms: float
    memory_usage_mb: float
    stability_score: float
    robustness_score: float
    user_satisfaction: float
    confidence_accuracy: float
    drift_resistance: float
    timestamp: datetime


@dataclass
class ModelComparison:
    """Comparison between two models."""
    model_a_id: str
    model_b_id: str
    metric_comparisons: Dict[str, float]  # Difference (A - B)
    statistical_significance: Dict[str, bool]
    winner: Optional[str]
    confidence_level: float
    recommendation: str


@dataclass
class ABTestResult:
    """Results from A/B testing between models."""
    test_id: str
    model_a_id: str
    model_b_id: str
    traffic_split: float  # Percentage to model A
    samples_a: int
    samples_b: int
    performance_a: Dict[str, float]
    performance_b: Dict[str, float]
    statistical_significance: bool
    winner: Optional[str]
    lift: float
    confidence_interval: Tuple[float, float]
    test_duration: timedelta
    status: str


class PerformanceTracker:
    """Tracks and analyzes model performance over time."""

    def __init__(self):
        self.performance_history = defaultdict(deque)
        self.prediction_cache = defaultdict(deque)
        self.feedback_cache = defaultdict(deque)
        self.benchmark_cache = {}

        # Configuration
        self.history_size = 10000
        self.evaluation_window = 1000
        self.min_samples_for_eval = 50

    def record_prediction(self, model_id: str, query_id: int, prediction: float,
                         confidence: float, processing_time_ms: float,
                         actual_score: Optional[float] = None):
        """Record a model prediction for performance tracking."""
        prediction_record = {
            'query_id': query_id,
            'prediction': prediction,
            'confidence': confidence,
            'processing_time_ms': processing_time_ms,
            'actual_score': actual_score,
            'timestamp': timezone.now()
        }

        # Add to cache
        self.prediction_cache[model_id].append(prediction_record)

        # Limit cache size
        if len(self.prediction_cache[model_id]) > self.history_size:
            self.prediction_cache[model_id].popleft()

    def record_feedback(self, model_id: str, query_id: int, user_feedback: float,
                       user_satisfaction: int):
        """Record user feedback for a model prediction."""
        feedback_record = {
            'query_id': query_id,
            'user_feedback': user_feedback,
            'user_satisfaction': user_satisfaction,
            'timestamp': timezone.now()
        }

        self.feedback_cache[model_id].append(feedback_record)

        # Limit cache size
        if len(self.feedback_cache[model_id]) > self.history_size:
            self.feedback_cache[model_id].popleft()

    def calculate_performance_metrics(self, model_id: str) -> Optional[PerformanceMetrics]:
        """Calculate comprehensive performance metrics for a model."""
        try:
            predictions = list(self.prediction_cache[model_id])
            feedback = list(self.feedback_cache[model_id])

            if len(predictions) < self.min_samples_for_eval:
                logger.warning(f"Insufficient predictions for model {model_id}: {len(predictions)}")
                return None

            # Extract data
            pred_values = [p['prediction'] for p in predictions if p['actual_score'] is not None]
            actual_values = [p['actual_score'] for p in predictions if p['actual_score'] is not None]
            confidences = [p['confidence'] for p in predictions]
            processing_times = [p['processing_time_ms'] for p in predictions]

            # Calculate accuracy metrics
            accuracy_score = self._calculate_accuracy(pred_values, actual_values)
            calibration_score = self._calculate_calibration(pred_values, actual_values, confidences)

            # Calculate speed metrics
            avg_speed = np.mean(processing_times) if processing_times else 0.0

            # Calculate stability (variance in performance over time)
            stability_score = self._calculate_stability(predictions)

            # Calculate robustness (performance across different query types)
            robustness_score = self._calculate_robustness(predictions)

            # Calculate user satisfaction
            user_satisfaction = self._calculate_user_satisfaction(feedback)

            # Calculate confidence accuracy
            confidence_accuracy = self._calculate_confidence_accuracy(pred_values, actual_values, confidences)

            return PerformanceMetrics(
                model_id=model_id,
                model_type=self._get_model_type(model_id),
                accuracy_score=accuracy_score,
                precision_score=accuracy_score,  # Simplified for regression
                recall_score=accuracy_score,     # Simplified for regression
                f1_score=accuracy_score,         # Simplified for regression
                calibration_score=calibration_score,
                prediction_speed_ms=avg_speed,
                memory_usage_mb=self._estimate_memory_usage(model_id),
                stability_score=stability_score,
                robustness_score=robustness_score,
                user_satisfaction=user_satisfaction,
                confidence_accuracy=confidence_accuracy,
                drift_resistance=self._calculate_drift_resistance(predictions),
                timestamp=timezone.now()
            )

        except Exception as e:
            logger.error(f"Error calculating performance metrics for {model_id}: {e}")
            return None

    def _calculate_accuracy(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate accuracy score."""
        if not predictions or not actuals or len(predictions) != len(actuals):
            return 0.0

        try:
            if SKLEARN_AVAILABLE:
                r2 = r2_score(actuals, predictions)
                return max(0.0, r2)
            else:
                # Fallback: 1 - normalized RMSE
                mse = np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)])
                rmse = math.sqrt(mse)
                mean_actual = np.mean(actuals)
                return max(0.0, 1.0 - (rmse / max(1.0, mean_actual)))
        except:
            return 0.0

    def _calculate_calibration(self, predictions: List[float], actuals: List[float],
                             confidences: List[float]) -> float:
        """Calculate calibration score (how well confidence matches accuracy)."""
        if len(predictions) != len(actuals) or len(predictions) != len(confidences):
            return 0.5

        try:
            # Bin predictions by confidence level
            bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            calibration_errors = []

            for low, high in bins:
                bin_predictions = []
                bin_actuals = []
                bin_confidences = []

                for pred, actual, conf in zip(predictions, actuals, confidences):
                    if low <= conf < high:
                        bin_predictions.append(pred)
                        bin_actuals.append(actual)
                        bin_confidences.append(conf)

                if bin_predictions:
                    bin_accuracy = self._calculate_accuracy(bin_predictions, bin_actuals)
                    avg_confidence = np.mean(bin_confidences)
                    error = abs(bin_accuracy - avg_confidence)
                    calibration_errors.append(error)

            return 1.0 - np.mean(calibration_errors) if calibration_errors else 0.5

        except Exception as e:
            logger.warning(f"Error calculating calibration: {e}")
            return 0.5

    def _calculate_stability(self, predictions: List[Dict]) -> float:
        """Calculate stability score based on performance variance over time."""
        if len(predictions) < 20:
            return 0.5

        try:
            # Calculate accuracy in sliding windows
            window_size = len(predictions) // 5
            window_accuracies = []

            for i in range(0, len(predictions) - window_size, window_size):
                window = predictions[i:i + window_size]
                window_preds = [p['prediction'] for p in window if p['actual_score'] is not None]
                window_actuals = [p['actual_score'] for p in window if p['actual_score'] is not None]

                if len(window_preds) > 5:
                    accuracy = self._calculate_accuracy(window_preds, window_actuals)
                    window_accuracies.append(accuracy)

            if len(window_accuracies) > 1:
                stability = 1.0 - np.std(window_accuracies)
                return max(0.0, min(1.0, stability))
            else:
                return 0.5

        except Exception as e:
            logger.warning(f"Error calculating stability: {e}")
            return 0.5

    def _calculate_robustness(self, predictions: List[Dict]) -> float:
        """Calculate robustness across different types of queries."""
        # Simplified robustness calculation
        # In practice, this would analyze performance across different query complexities
        if len(predictions) < self.min_samples_for_eval:
            return 0.5

        try:
            # Group predictions by complexity (simplified)
            simple_preds = []
            complex_preds = []

            for pred in predictions:
                # Simplified complexity detection based on prediction value
                if pred['prediction'] < 50:
                    simple_preds.append(pred)
                else:
                    complex_preds.append(pred)

            simple_accuracy = 0.5
            complex_accuracy = 0.5

            if len(simple_preds) > 5:
                simple_pred_vals = [p['prediction'] for p in simple_preds if p['actual_score'] is not None]
                simple_actual_vals = [p['actual_score'] for p in simple_preds if p['actual_score'] is not None]
                if simple_pred_vals:
                    simple_accuracy = self._calculate_accuracy(simple_pred_vals, simple_actual_vals)

            if len(complex_preds) > 5:
                complex_pred_vals = [p['prediction'] for p in complex_preds if p['actual_score'] is not None]
                complex_actual_vals = [p['actual_score'] for p in complex_preds if p['actual_score'] is not None]
                if complex_pred_vals:
                    complex_accuracy = self._calculate_accuracy(complex_pred_vals, complex_actual_vals)

            # Robustness is how consistent performance is across different complexities
            robustness = 1.0 - abs(simple_accuracy - complex_accuracy)
            return max(0.0, min(1.0, robustness))

        except Exception as e:
            logger.warning(f"Error calculating robustness: {e}")
            return 0.5

    def _calculate_user_satisfaction(self, feedback: List[Dict]) -> float:
        """Calculate user satisfaction score."""
        if not feedback:
            return 0.5

        try:
            satisfaction_scores = [f['user_satisfaction'] for f in feedback if 'user_satisfaction' in f]
            if satisfaction_scores:
                # Convert 1-5 scale to 0-1 scale
                avg_satisfaction = np.mean(satisfaction_scores)
                return (avg_satisfaction - 1) / 4
            else:
                return 0.5
        except:
            return 0.5

    def _calculate_confidence_accuracy(self, predictions: List[float], actuals: List[float],
                                     confidences: List[float]) -> float:
        """Calculate how accurate the confidence estimates are."""
        if len(predictions) != len(actuals) or len(predictions) != len(confidences):
            return 0.5

        try:
            # Calculate actual accuracy for each prediction
            actual_accuracies = []
            for pred, actual in zip(predictions, actuals):
                error = abs(pred - actual)
                accuracy = max(0.0, 1.0 - (error / max(100.0, abs(actual))))
                actual_accuracies.append(accuracy)

            # Calculate correlation between confidence and actual accuracy
            if SKLEARN_AVAILABLE and len(confidences) > 10:
                correlation, _ = stats.pearsonr(confidences, actual_accuracies)
                return max(0.0, correlation)
            else:
                # Simple agreement measure
                agreements = []
                for conf, acc in zip(confidences, actual_accuracies):
                    agreement = 1.0 - abs(conf - acc)
                    agreements.append(agreement)
                return np.mean(agreements)

        except Exception as e:
            logger.warning(f"Error calculating confidence accuracy: {e}")
            return 0.5

    def _calculate_drift_resistance(self, predictions: List[Dict]) -> float:
        """Calculate resistance to concept drift."""
        if len(predictions) < 100:
            return 0.5

        try:
            # Compare recent performance to older performance
            recent_preds = predictions[-50:]
            older_preds = predictions[-100:-50]

            recent_pred_vals = [p['prediction'] for p in recent_preds if p['actual_score'] is not None]
            recent_actual_vals = [p['actual_score'] for p in recent_preds if p['actual_score'] is not None]

            older_pred_vals = [p['prediction'] for p in older_preds if p['actual_score'] is not None]
            older_actual_vals = [p['actual_score'] for p in older_preds if p['actual_score'] is not None]

            if len(recent_pred_vals) > 10 and len(older_pred_vals) > 10:
                recent_accuracy = self._calculate_accuracy(recent_pred_vals, recent_actual_vals)
                older_accuracy = self._calculate_accuracy(older_pred_vals, older_actual_vals)

                # Drift resistance is how well accuracy is maintained
                drift_resistance = 1.0 - abs(recent_accuracy - older_accuracy)
                return max(0.0, min(1.0, drift_resistance))
            else:
                return 0.5

        except Exception as e:
            logger.warning(f"Error calculating drift resistance: {e}")
            return 0.5

    def _get_model_type(self, model_id: str) -> str:
        """Get model type from model ID."""
        if 'rf' in model_id.lower() or 'random_forest' in model_id.lower():
            return 'random_forest'
        elif 'xgb' in model_id.lower() or 'xgboost' in model_id.lower():
            return 'xgboost'
        elif 'nn' in model_id.lower() or 'neural' in model_id.lower():
            return 'neural_network'
        else:
            return 'unknown'

    def _estimate_memory_usage(self, model_id: str) -> float:
        """Estimate memory usage for a model."""
        # Simplified estimation
        model_type = self._get_model_type(model_id)
        if model_type == 'neural_network':
            return 50.0  # MB
        elif model_type == 'xgboost':
            return 20.0  # MB
        elif model_type == 'random_forest':
            return 30.0  # MB
        else:
            return 10.0  # MB


class ModelSelector:
    """Automatic model selection based on performance criteria."""

    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.selection_history = deque(maxlen=100)
        self.cache = caches['default']

    def select_best_model(self, available_models: List[str],
                         criteria: SelectionCriteria = SelectionCriteria.BALANCED,
                         context: Dict[str, Any] = None) -> Optional[str]:
        """Select the best model based on specified criteria."""
        try:
            if not available_models:
                return None

            # Calculate performance for all models
            model_performances = {}
            for model_id in available_models:
                metrics = self.performance_tracker.calculate_performance_metrics(model_id)
                if metrics:
                    model_performances[model_id] = metrics

            if not model_performances:
                logger.warning("No performance metrics available for any model")
                return available_models[0]  # Default to first model

            # Select based on criteria
            if criteria == SelectionCriteria.ACCURACY:
                best_model = max(model_performances.items(), key=lambda x: x[1].accuracy_score)[0]
            elif criteria == SelectionCriteria.SPEED:
                best_model = min(model_performances.items(), key=lambda x: x[1].prediction_speed_ms)[0]
            elif criteria == SelectionCriteria.RELIABILITY:
                best_model = max(model_performances.items(), key=lambda x: x[1].stability_score)[0]
            elif criteria == SelectionCriteria.CALIBRATION:
                best_model = max(model_performances.items(), key=lambda x: x[1].calibration_score)[0]
            elif criteria == SelectionCriteria.ROBUSTNESS:
                best_model = max(model_performances.items(), key=lambda x: x[1].robustness_score)[0]
            else:  # BALANCED
                best_model = self._select_balanced_model(model_performances)

            # Record selection
            self.selection_history.append({
                'selected_model': best_model,
                'criteria': criteria.value,
                'alternatives': list(available_models),
                'timestamp': timezone.now()
            })

            logger.info(f"Selected model {best_model} based on {criteria.value} criteria")
            return best_model

        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            return available_models[0] if available_models else None

    def _select_balanced_model(self, model_performances: Dict[str, PerformanceMetrics]) -> str:
        """Select model based on balanced criteria."""
        scores = {}

        for model_id, metrics in model_performances.items():
            # Balanced score combining multiple factors
            score = (
                metrics.accuracy_score * 0.3 +
                metrics.calibration_score * 0.2 +
                metrics.stability_score * 0.2 +
                metrics.robustness_score * 0.15 +
                metrics.user_satisfaction * 0.15
            )

            # Penalty for slow models
            if metrics.prediction_speed_ms > 100:
                score *= 0.9

            scores[model_id] = score

        return max(scores.items(), key=lambda x: x[1])[0]

    def compare_models(self, model_a: str, model_b: str) -> ModelComparison:
        """Compare two models across all metrics."""
        try:
            metrics_a = self.performance_tracker.calculate_performance_metrics(model_a)
            metrics_b = self.performance_tracker.calculate_performance_metrics(model_b)

            if not metrics_a or not metrics_b:
                return ModelComparison(
                    model_a_id=model_a,
                    model_b_id=model_b,
                    metric_comparisons={},
                    statistical_significance={},
                    winner=None,
                    confidence_level=0.0,
                    recommendation="Insufficient data for comparison"
                )

            # Calculate differences
            comparisons = {
                'accuracy': metrics_a.accuracy_score - metrics_b.accuracy_score,
                'speed': metrics_b.prediction_speed_ms - metrics_a.prediction_speed_ms,  # Lower is better
                'calibration': metrics_a.calibration_score - metrics_b.calibration_score,
                'stability': metrics_a.stability_score - metrics_b.stability_score,
                'robustness': metrics_a.robustness_score - metrics_b.robustness_score,
                'user_satisfaction': metrics_a.user_satisfaction - metrics_b.user_satisfaction
            }

            # Determine winner
            positive_diffs = sum(1 for diff in comparisons.values() if diff > 0.05)
            negative_diffs = sum(1 for diff in comparisons.values() if diff < -0.05)

            if positive_diffs > negative_diffs:
                winner = model_a
            elif negative_diffs > positive_diffs:
                winner = model_b
            else:
                winner = None

            # Calculate confidence
            confidence = abs(positive_diffs - negative_diffs) / len(comparisons)

            return ModelComparison(
                model_a_id=model_a,
                model_b_id=model_b,
                metric_comparisons=comparisons,
                statistical_significance={},  # Simplified
                winner=winner,
                confidence_level=confidence,
                recommendation=f"Model {winner} performs better" if winner else "Models perform similarly"
            )

        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return ModelComparison(
                model_a_id=model_a,
                model_b_id=model_b,
                metric_comparisons={},
                statistical_significance={},
                winner=None,
                confidence_level=0.0,
                recommendation="Error in comparison"
            )


class ABTestingFramework:
    """A/B testing framework for model comparison."""

    def __init__(self):
        self.active_tests = {}
        self.test_history = deque(maxlen=100)
        self.cache = caches['default']

    def start_ab_test(self, model_a: str, model_b: str, traffic_split: float = 0.5,
                     duration_hours: int = 24) -> str:
        """Start an A/B test between two models."""
        test_id = f"ab_test_{int(timezone.now().timestamp())}_{model_a}_{model_b}"

        test_config = {
            'test_id': test_id,
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'start_time': timezone.now(),
            'end_time': timezone.now() + timedelta(hours=duration_hours),
            'samples_a': 0,
            'samples_b': 0,
            'results_a': [],
            'results_b': [],
            'status': 'active'
        }

        self.active_tests[test_id] = test_config
        logger.info(f"Started A/B test {test_id}: {model_a} vs {model_b}")

        return test_id

    def record_ab_result(self, test_id: str, model_used: str, prediction: float,
                        actual_score: float, user_feedback: float = None):
        """Record a result for an A/B test."""
        if test_id not in self.active_tests:
            return

        test = self.active_tests[test_id]
        result = {
            'prediction': prediction,
            'actual_score': actual_score,
            'user_feedback': user_feedback,
            'timestamp': timezone.now()
        }

        if model_used == test['model_a']:
            test['results_a'].append(result)
            test['samples_a'] += 1
        elif model_used == test['model_b']:
            test['results_b'].append(result)
            test['samples_b'] += 1

    def analyze_ab_test(self, test_id: str) -> Optional[ABTestResult]:
        """Analyze results of an A/B test."""
        if test_id not in self.active_tests:
            return None

        test = self.active_tests[test_id]

        try:
            # Calculate performance for both models
            performance_a = self._calculate_ab_performance(test['results_a'])
            performance_b = self._calculate_ab_performance(test['results_b'])

            # Determine statistical significance (simplified)
            significance = self._test_statistical_significance(
                test['results_a'], test['results_b']
            )

            # Determine winner
            winner = None
            if performance_a['accuracy'] > performance_b['accuracy'] + 0.05:
                winner = test['model_a']
            elif performance_b['accuracy'] > performance_a['accuracy'] + 0.05:
                winner = test['model_b']

            # Calculate lift
            lift = (performance_a['accuracy'] - performance_b['accuracy']) / performance_b['accuracy'] \
                if performance_b['accuracy'] > 0 else 0

            result = ABTestResult(
                test_id=test_id,
                model_a_id=test['model_a'],
                model_b_id=test['model_b'],
                traffic_split=test['traffic_split'],
                samples_a=test['samples_a'],
                samples_b=test['samples_b'],
                performance_a=performance_a,
                performance_b=performance_b,
                statistical_significance=significance,
                winner=winner,
                lift=lift,
                confidence_interval=(lift - 0.1, lift + 0.1),  # Simplified
                test_duration=timezone.now() - test['start_time'],
                status='completed'
            )

            # Move to history
            self.test_history.append(result)
            test['status'] = 'completed'

            return result

        except Exception as e:
            logger.error(f"Error analyzing A/B test {test_id}: {e}")
            return None

    def _calculate_ab_performance(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics for A/B test results."""
        if not results:
            return {'accuracy': 0.0, 'user_satisfaction': 0.0}

        # Calculate accuracy
        predictions = [r['prediction'] for r in results if r['actual_score'] is not None]
        actuals = [r['actual_score'] for r in results if r['actual_score'] is not None]

        accuracy = 0.0
        if predictions and actuals:
            errors = [abs(p - a) for p, a in zip(predictions, actuals)]
            accuracy = 1.0 - (np.mean(errors) / 100.0)

        # Calculate user satisfaction
        feedbacks = [r['user_feedback'] for r in results if r['user_feedback'] is not None]
        user_satisfaction = np.mean(feedbacks) / 5.0 if feedbacks else 0.5

        return {
            'accuracy': max(0.0, accuracy),
            'user_satisfaction': user_satisfaction
        }

    def _test_statistical_significance(self, results_a: List[Dict],
                                     results_b: List[Dict]) -> bool:
        """Test for statistical significance between A/B test results."""
        # Simplified significance test
        if len(results_a) < 20 or len(results_b) < 20:
            return False

        try:
            # Compare accuracy distributions
            acc_a = [abs(r['prediction'] - r['actual_score']) for r in results_a
                    if r['actual_score'] is not None]
            acc_b = [abs(r['prediction'] - r['actual_score']) for r in results_b
                    if r['actual_score'] is not None]

            if len(acc_a) < 10 or len(acc_b) < 10:
                return False

            # Simple t-test alternative
            mean_a = np.mean(acc_a)
            mean_b = np.mean(acc_b)
            std_a = np.std(acc_a)
            std_b = np.std(acc_b)

            # Calculate effect size
            pooled_std = math.sqrt((std_a ** 2 + std_b ** 2) / 2)
            effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

            # Significance if effect size > 0.5 (medium effect)
            return effect_size > 0.5

        except Exception as e:
            logger.warning(f"Error in significance test: {e}")
            return False


class ModelPerformanceManager:
    """Main manager for model performance tracking and selection."""

    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.model_selector = ModelSelector()
        self.ab_testing = ABTestingFramework()
        self.cache = caches['default']

        # Configuration
        self.auto_selection_enabled = True
        self.performance_check_interval = timedelta(hours=1)
        self.last_performance_check = timezone.now()

    def get_best_model(self, criteria: str = "balanced") -> Optional[str]:
        """Get the best performing model."""
        try:
            # Get all available models
            available_models = self._get_available_models()

            if not available_models:
                logger.warning("No models available for selection")
                return None

            # Convert criteria string to enum
            criteria_enum = SelectionCriteria(criteria)

            # Select best model
            best_model = self.model_selector.select_best_model(available_models, criteria_enum)

            # Cache result
            self.cache.set('best_model_selection', {
                'model_id': best_model,
                'criteria': criteria,
                'timestamp': timezone.now().isoformat(),
                'alternatives': available_models
            }, timeout=3600)

            return best_model

        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            return None

    def _get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            # Get from database
            active_models = MLModel.objects.filter(status='ACTIVE')
            return [model.name for model in active_models]
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models."""
        try:
            available_models = self._get_available_models()
            summary = {
                'model_count': len(available_models),
                'models': {},
                'best_model': None,
                'performance_trends': {},
                'last_updated': timezone.now().isoformat()
            }

            for model_id in available_models:
                metrics = self.performance_tracker.calculate_performance_metrics(model_id)
                if metrics:
                    summary['models'][model_id] = asdict(metrics)

            # Determine best model
            if summary['models']:
                best_model = max(summary['models'].items(),
                               key=lambda x: x[1]['accuracy_score'])
                summary['best_model'] = best_model[0]

            return summary

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}


# Global performance manager instance
performance_manager = ModelPerformanceManager()


# Django integration functions
def record_model_prediction(model_id: str, query_id: int, prediction: float,
                           confidence: float, processing_time_ms: float,
                           actual_score: float = None):
    """Record a model prediction for performance tracking."""
    performance_manager.performance_tracker.record_prediction(
        model_id, query_id, prediction, confidence, processing_time_ms, actual_score
    )


def record_model_feedback(model_id: str, query_id: int, user_feedback: float,
                         user_satisfaction: int):
    """Record user feedback for a model."""
    performance_manager.performance_tracker.record_feedback(
        model_id, query_id, user_feedback, user_satisfaction
    )


def get_best_performing_model(criteria: str = "balanced") -> Optional[str]:
    """Get the best performing model based on criteria."""
    return performance_manager.get_best_model(criteria)


def get_model_performance_summary() -> Dict[str, Any]:
    """Get performance summary for all models."""
    return performance_manager.get_performance_summary()


def compare_two_models(model_a: str, model_b: str) -> Dict[str, Any]:
    """Compare two models."""
    comparison = performance_manager.model_selector.compare_models(model_a, model_b)
    return asdict(comparison)


def start_model_ab_test(model_a: str, model_b: str, traffic_split: float = 0.5) -> str:
    """Start an A/B test between two models."""
    return performance_manager.ab_testing.start_ab_test(model_a, model_b, traffic_split)


# Mark the final task as completed
if __name__ == "__main__":
    print("Model Performance Tracking and Automatic Selection system initialized")
    print("Features:")
    print("- Comprehensive performance tracking")
    print("- Automatic model selection")
    print("- A/B testing framework")
    print("- Statistical significance testing")
    print("- Real-time performance monitoring")