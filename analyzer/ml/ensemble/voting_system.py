"""
Advanced Ensemble Voting System for QueryGrade ML

This module implements sophisticated voting mechanisms for combining predictions
from multiple ML models with dynamic weighting, confidence-based decisions,
and adaptive ensemble strategies.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import math
import statistics
from collections import defaultdict, deque

from django.core.cache import caches
from django.utils import timezone

try:
    from scipy import stats
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available. Some statistical functions will be limited.")

logger = logging.getLogger(__name__)


class VotingStrategy(Enum):
    """Different voting strategies for ensemble decisions."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    DYNAMIC_WEIGHTED = "dynamic_weighted"
    RANK_BASED = "rank_based"
    BAYESIAN_ENSEMBLE = "bayesian_ensemble"
    ADAPTIVE_VOTING = "adaptive_voting"


class AggregationMethod(Enum):
    """Methods for aggregating model outputs."""
    MEAN = "mean"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    HARMONIC_MEAN = "harmonic_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    MODE = "mode"
    PERCENTILE = "percentile"


@dataclass
class ModelPrediction:
    """Represents a prediction from a single model."""
    model_id: str
    model_type: str
    prediction: float
    confidence: float
    processing_time_ms: float
    feature_importance: Optional[List[float]] = None
    uncertainty: Optional[float] = None
    model_version: str = "1.0"


@dataclass
class VotingResult:
    """Result from ensemble voting process."""
    final_prediction: float
    final_confidence: float
    voting_strategy: VotingStrategy
    aggregation_method: AggregationMethod
    model_predictions: List[ModelPrediction]
    model_weights: Dict[str, float]
    consensus_metrics: Dict[str, float]
    quality_score: float
    explanation: str


@dataclass
class EnsembleMetrics:
    """Metrics for ensemble performance evaluation."""
    prediction_variance: float
    model_agreement: float
    confidence_calibration: float
    diversity_score: float
    stability_score: float
    computational_cost: float


class ModelWeightCalculator:
    """Calculates weights for models based on various criteria."""

    def __init__(self):
        self.historical_performance = defaultdict(deque)
        self.confidence_calibration = defaultdict(float)
        self.recent_predictions = defaultdict(deque)

    def calculate_performance_weights(self, model_performances: Dict[str, float]) -> Dict[str, float]:
        """Calculate weights based on historical performance."""
        if not model_performances:
            return {}

        # Normalize performances to 0-1 range
        min_perf = min(model_performances.values())
        max_perf = max(model_performances.values())
        range_perf = max_perf - min_perf

        if range_perf == 0:
            # All models have same performance, use equal weights
            n_models = len(model_performances)
            return {model_id: 1.0/n_models for model_id in model_performances.keys()}

        normalized_perfs = {}
        for model_id, perf in model_performances.items():
            normalized_perfs[model_id] = (perf - min_perf) / range_perf

        # Softmax transformation for smooth weights
        exp_perfs = {model_id: math.exp(perf * 5) for model_id, perf in normalized_perfs.items()}
        sum_exp = sum(exp_perfs.values())

        return {model_id: exp_perf / sum_exp for model_id, exp_perf in exp_perfs.items()}

    def calculate_confidence_weights(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Calculate weights based on model confidence."""
        if not predictions:
            return {}

        # Higher confidence models get higher weights
        confidences = {pred.model_id: pred.confidence for pred in predictions}
        total_confidence = sum(confidences.values())

        if total_confidence == 0:
            n_models = len(predictions)
            return {pred.model_id: 1.0/n_models for pred in predictions}

        return {model_id: conf / total_confidence for model_id, conf in confidences.items()}

    def calculate_diversity_weights(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Calculate weights to promote diversity in ensemble."""
        if len(predictions) < 2:
            return {pred.model_id: 1.0 for pred in predictions}

        pred_values = [pred.prediction for pred in predictions]
        model_ids = [pred.model_id for pred in predictions]

        # Calculate distance from ensemble mean
        ensemble_mean = np.mean(pred_values)
        distances = [abs(pred - ensemble_mean) for pred in pred_values]

        # Models farther from mean get slightly higher weights (diversity bonus)
        diversity_scores = {}
        for model_id, distance in zip(model_ids, distances):
            # Base weight + diversity bonus
            diversity_scores[model_id] = 1.0 + (distance / max(1.0, ensemble_mean)) * 0.1

        # Normalize
        total_score = sum(diversity_scores.values())
        return {model_id: score / total_score for model_id, score in diversity_scores.items()}

    def calculate_dynamic_weights(self, predictions: List[ModelPrediction],
                                query_features: List[float]) -> Dict[str, float]:
        """Calculate dynamic weights based on query characteristics."""
        if not predictions:
            return {}

        # Start with equal weights
        n_models = len(predictions)
        weights = {pred.model_id: 1.0/n_models for pred in predictions}

        # Adjust based on query complexity
        if query_features:
            complexity_score = self._estimate_query_complexity(query_features)

            for pred in predictions:
                # Some models might be better for complex vs simple queries
                if "neural_network" in pred.model_type.lower() and complexity_score > 0.7:
                    weights[pred.model_id] *= 1.2  # NN better for complex queries
                elif "random_forest" in pred.model_type.lower() and complexity_score < 0.3:
                    weights[pred.model_id] *= 1.2  # RF better for simple queries

        # Normalize weights
        total_weight = sum(weights.values())
        return {model_id: weight / total_weight for model_id, weight in weights.items()}

    def _estimate_query_complexity(self, features: List[float]) -> float:
        """Estimate query complexity from features (0-1 scale)."""
        if not features or len(features) < 5:
            return 0.5

        # Simplified complexity estimation
        # This would be more sophisticated in practice
        feature_variance = np.var(features)
        feature_mean = np.mean(features)
        complexity = min(1.0, (feature_variance + abs(feature_mean)) / 100.0)
        return complexity


class ConsensusAnalyzer:
    """Analyzes consensus and agreement among model predictions."""

    def __init__(self):
        pass

    def calculate_consensus_metrics(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Calculate various consensus metrics."""
        if len(predictions) < 2:
            return {
                'agreement_score': 1.0,
                'prediction_variance': 0.0,
                'coefficient_of_variation': 0.0,
                'range_normalized': 0.0,
                'pairwise_agreement': 1.0
            }

        pred_values = [pred.prediction for pred in predictions]

        # Basic statistics
        mean_pred = np.mean(pred_values)
        variance = np.var(pred_values)
        std_dev = np.std(pred_values)
        pred_range = max(pred_values) - min(pred_values)

        # Agreement score (inverse of coefficient of variation)
        if mean_pred > 0:
            cv = std_dev / mean_pred
            agreement_score = max(0.0, 1.0 - cv)
        else:
            agreement_score = 1.0 if variance == 0 else 0.0

        # Normalized range (0-1 scale)
        range_normalized = pred_range / max(100.0, mean_pred)

        # Pairwise agreement
        pairwise_agreement = self._calculate_pairwise_agreement(pred_values)

        return {
            'agreement_score': agreement_score,
            'prediction_variance': variance,
            'coefficient_of_variation': cv if 'cv' in locals() else 0.0,
            'range_normalized': range_normalized,
            'pairwise_agreement': pairwise_agreement
        }

    def _calculate_pairwise_agreement(self, predictions: List[float]) -> float:
        """Calculate average pairwise agreement between predictions."""
        if len(predictions) < 2:
            return 1.0

        agreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                diff = abs(predictions[i] - predictions[j])
                max_val = max(abs(predictions[i]), abs(predictions[j]), 1.0)
                agreement = max(0.0, 1.0 - (diff / max_val))
                agreements.append(agreement)

        return np.mean(agreements) if agreements else 1.0

    def detect_outlier_predictions(self, predictions: List[ModelPrediction],
                                 threshold: float = 2.0) -> List[str]:
        """Detect outlier predictions using statistical methods."""
        if len(predictions) < 3:
            return []

        pred_values = [pred.prediction for pred in predictions]
        model_ids = [pred.model_id for pred in predictions]

        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)

        outliers = []
        for model_id, value in zip(model_ids, pred_values):
            z_score = abs(value - mean_pred) / std_pred if std_pred > 0 else 0
            if z_score > threshold:
                outliers.append(model_id)

        return outliers


class VotingStrategies:
    """Implementation of various voting strategies."""

    def __init__(self):
        self.weight_calculator = ModelWeightCalculator()
        self.consensus_analyzer = ConsensusAnalyzer()

    def simple_average(self, predictions: List[ModelPrediction]) -> VotingResult:
        """Simple average of all predictions."""
        if not predictions:
            return self._empty_result(VotingStrategy.SIMPLE_AVERAGE)

        pred_values = [pred.prediction for pred in predictions]
        final_prediction = np.mean(pred_values)

        # Equal weights
        weights = {pred.model_id: 1.0/len(predictions) for pred in predictions}

        # Simple confidence (average of confidences)
        confidences = [pred.confidence for pred in predictions]
        final_confidence = np.mean(confidences)

        consensus_metrics = self.consensus_analyzer.calculate_consensus_metrics(predictions)

        return VotingResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            voting_strategy=VotingStrategy.SIMPLE_AVERAGE,
            aggregation_method=AggregationMethod.MEAN,
            model_predictions=predictions,
            model_weights=weights,
            consensus_metrics=consensus_metrics,
            quality_score=consensus_metrics['agreement_score'],
            explanation="Simple average of all model predictions"
        )

    def weighted_average(self, predictions: List[ModelPrediction],
                        model_performances: Dict[str, float]) -> VotingResult:
        """Weighted average based on model performance."""
        if not predictions:
            return self._empty_result(VotingStrategy.WEIGHTED_AVERAGE)

        weights = self.weight_calculator.calculate_performance_weights(model_performances)

        # Calculate weighted prediction
        weighted_sum = 0.0
        total_weight = 0.0

        for pred in predictions:
            weight = weights.get(pred.model_id, 0.0)
            weighted_sum += pred.prediction * weight
            total_weight += weight

        final_prediction = weighted_sum / total_weight if total_weight > 0 else np.mean([p.prediction for p in predictions])

        # Weighted confidence
        confidence_sum = 0.0
        for pred in predictions:
            weight = weights.get(pred.model_id, 0.0)
            confidence_sum += pred.confidence * weight

        final_confidence = confidence_sum / total_weight if total_weight > 0 else np.mean([p.confidence for p in predictions])

        consensus_metrics = self.consensus_analyzer.calculate_consensus_metrics(predictions)

        return VotingResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
            aggregation_method=AggregationMethod.MEAN,
            model_predictions=predictions,
            model_weights=weights,
            consensus_metrics=consensus_metrics,
            quality_score=consensus_metrics['agreement_score'] * np.mean(list(weights.values())),
            explanation="Performance-weighted average of model predictions"
        )

    def confidence_weighted(self, predictions: List[ModelPrediction]) -> VotingResult:
        """Confidence-weighted voting."""
        if not predictions:
            return self._empty_result(VotingStrategy.CONFIDENCE_WEIGHTED)

        weights = self.weight_calculator.calculate_confidence_weights(predictions)

        # Calculate weighted prediction
        weighted_sum = 0.0
        total_weight = 0.0

        for pred in predictions:
            weight = weights.get(pred.model_id, 0.0)
            weighted_sum += pred.prediction * weight
            total_weight += weight

        final_prediction = weighted_sum / total_weight if total_weight > 0 else np.mean([p.prediction for p in predictions])

        # Confidence calculation (higher weights contribute more to confidence)
        final_confidence = sum(pred.confidence * weights.get(pred.model_id, 0.0) for pred in predictions)

        consensus_metrics = self.consensus_analyzer.calculate_consensus_metrics(predictions)

        return VotingResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            voting_strategy=VotingStrategy.CONFIDENCE_WEIGHTED,
            aggregation_method=AggregationMethod.MEAN,
            model_predictions=predictions,
            model_weights=weights,
            consensus_metrics=consensus_metrics,
            quality_score=final_confidence * consensus_metrics['agreement_score'],
            explanation="Confidence-weighted ensemble voting"
        )

    def adaptive_voting(self, predictions: List[ModelPrediction],
                       query_features: List[float],
                       context: Dict[str, Any] = None) -> VotingResult:
        """Adaptive voting that chooses strategy based on context."""
        if not predictions:
            return self._empty_result(VotingStrategy.ADAPTIVE_VOTING)

        # Analyze context to choose best strategy
        consensus_metrics = self.consensus_analyzer.calculate_consensus_metrics(predictions)
        agreement_score = consensus_metrics['agreement_score']

        # Choose strategy based on agreement and context
        if agreement_score > 0.9:
            # High agreement - use simple average
            return self.simple_average(predictions)
        elif agreement_score > 0.7:
            # Medium agreement - use confidence weighting
            return self.confidence_weighted(predictions)
        else:
            # Low agreement - use performance weighting if available
            if context and 'model_performances' in context:
                return self.weighted_average(predictions, context['model_performances'])
            else:
                return self.rank_based_voting(predictions)

    def rank_based_voting(self, predictions: List[ModelPrediction]) -> VotingResult:
        """Rank-based voting using Borda count method."""
        if not predictions:
            return self._empty_result(VotingStrategy.RANK_BASED)

        # Sort predictions by value
        sorted_preds = sorted(predictions, key=lambda p: p.prediction)

        # Assign ranks (higher prediction = higher rank)
        ranks = {}
        for i, pred in enumerate(sorted_preds):
            ranks[pred.model_id] = i + 1

        # Calculate weights based on ranks
        max_rank = len(predictions)
        weights = {model_id: rank / max_rank for model_id, rank in ranks.items()}

        # Weighted average using rank weights
        weighted_sum = sum(pred.prediction * weights[pred.model_id] for pred in predictions)
        total_weight = sum(weights.values())

        final_prediction = weighted_sum / total_weight if total_weight > 0 else np.mean([p.prediction for p in predictions])

        # Confidence based on rank consistency
        final_confidence = self._calculate_rank_consistency(predictions)

        consensus_metrics = self.consensus_analyzer.calculate_consensus_metrics(predictions)

        return VotingResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            voting_strategy=VotingStrategy.RANK_BASED,
            aggregation_method=AggregationMethod.MEAN,
            model_predictions=predictions,
            model_weights=weights,
            consensus_metrics=consensus_metrics,
            quality_score=final_confidence,
            explanation="Rank-based voting using Borda count method"
        )

    def _calculate_rank_consistency(self, predictions: List[ModelPrediction]) -> float:
        """Calculate consistency of rankings."""
        if len(predictions) < 2:
            return 1.0

        # Simple rank consistency based on variance
        pred_values = [pred.prediction for pred in predictions]
        normalized_variance = np.var(pred_values) / max(1.0, np.mean(pred_values)**2)
        consistency = max(0.0, 1.0 - normalized_variance)
        return consistency

    def _empty_result(self, strategy: VotingStrategy) -> VotingResult:
        """Return empty result for edge cases."""
        return VotingResult(
            final_prediction=50.0,  # Default score
            final_confidence=0.1,
            voting_strategy=strategy,
            aggregation_method=AggregationMethod.MEAN,
            model_predictions=[],
            model_weights={},
            consensus_metrics={},
            quality_score=0.0,
            explanation="No valid predictions available"
        )


class EnsembleVotingSystem:
    """Main ensemble voting system with strategy selection."""

    def __init__(self):
        self.voting_strategies = VotingStrategies()
        self.strategy_performance = defaultdict(deque)  # Track strategy performance
        self.cache = caches['default']

    def vote(self, predictions: List[ModelPrediction],
             strategy: VotingStrategy = VotingStrategy.ADAPTIVE_VOTING,
             context: Dict[str, Any] = None) -> VotingResult:
        """Perform ensemble voting using specified strategy."""
        try:
            if not predictions:
                logger.warning("No predictions provided for voting")
                return self._get_default_result(strategy)

            # Remove invalid predictions
            valid_predictions = [p for p in predictions if self._is_valid_prediction(p)]

            if not valid_predictions:
                logger.warning("No valid predictions after filtering")
                return self._get_default_result(strategy)

            # Detect and handle outliers
            outliers = self.voting_strategies.consensus_analyzer.detect_outlier_predictions(valid_predictions)
            if outliers:
                logger.info(f"Detected outlier predictions from models: {outliers}")
                # Option: remove outliers or flag them
                # For now, we'll keep them but log the information

            # Execute voting strategy
            if strategy == VotingStrategy.SIMPLE_AVERAGE:
                result = self.voting_strategies.simple_average(valid_predictions)
            elif strategy == VotingStrategy.WEIGHTED_AVERAGE:
                model_perfs = context.get('model_performances', {}) if context else {}
                result = self.voting_strategies.weighted_average(valid_predictions, model_perfs)
            elif strategy == VotingStrategy.CONFIDENCE_WEIGHTED:
                result = self.voting_strategies.confidence_weighted(valid_predictions)
            elif strategy == VotingStrategy.RANK_BASED:
                result = self.voting_strategies.rank_based_voting(valid_predictions)
            elif strategy == VotingStrategy.ADAPTIVE_VOTING:
                query_features = context.get('query_features', []) if context else []
                result = self.voting_strategies.adaptive_voting(valid_predictions, query_features, context)
            else:
                # Default to simple average
                result = self.voting_strategies.simple_average(valid_predictions)

            # Post-process result
            result = self._post_process_result(result)

            # Cache result for analysis
            self._cache_voting_result(result)

            return result

        except Exception as e:
            logger.error(f"Error in ensemble voting: {e}")
            return self._get_default_result(strategy)

    def _is_valid_prediction(self, prediction: ModelPrediction) -> bool:
        """Check if a prediction is valid."""
        return (
            prediction.prediction is not None and
            0 <= prediction.prediction <= 100 and
            0 <= prediction.confidence <= 1 and
            prediction.model_id and
            prediction.model_type
        )

    def _post_process_result(self, result: VotingResult) -> VotingResult:
        """Post-process voting result to ensure validity."""
        # Clamp prediction to valid range
        result.final_prediction = max(0.0, min(100.0, result.final_prediction))

        # Clamp confidence to valid range
        result.final_confidence = max(0.0, min(1.0, result.final_confidence))

        # Ensure quality score is valid
        result.quality_score = max(0.0, min(1.0, result.quality_score))

        return result

    def _cache_voting_result(self, result: VotingResult):
        """Cache voting result for analysis."""
        cache_key = f"voting_result_{int(timezone.now().timestamp())}"
        cache_data = {
            'strategy': result.voting_strategy.value,
            'prediction': result.final_prediction,
            'confidence': result.final_confidence,
            'quality': result.quality_score,
            'model_count': len(result.model_predictions),
            'timestamp': timezone.now().isoformat()
        }
        self.cache.set(cache_key, cache_data, timeout=3600)

    def _get_default_result(self, strategy: VotingStrategy) -> VotingResult:
        """Get default result for error cases."""
        return VotingResult(
            final_prediction=50.0,
            final_confidence=0.1,
            voting_strategy=strategy,
            aggregation_method=AggregationMethod.MEAN,
            model_predictions=[],
            model_weights={},
            consensus_metrics={},
            quality_score=0.0,
            explanation="Default result due to error or no valid predictions"
        )

    def analyze_voting_performance(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze voting performance over recent period.

        Args:
            days_back: Number of days to look back for analysis

        Returns:
            Dictionary containing performance metrics and recommendations
        """
        try:
            analysis = {
                'total_votes': 0,
                'strategy_usage': defaultdict(int),
                'average_confidence': 0.0,
                'average_quality': 0.0,
                'consensus_trends': {},
                'recommendations': [],
                'performance_metrics': {},
                'time_period_days': days_back
            }

            # Calculate cutoff timestamp for filtering results
            cutoff_time = timezone.now() - timedelta(days=days_back)
            cutoff_timestamp = int(cutoff_time.timestamp())

            # Collect voting results from cache
            # Try to get cache keys (works with Redis backend)
            cached_results = []
            try:
                # Get Django cache backend client (Redis)
                cache_client = self.cache._cache.get_client() if hasattr(self.cache, '_cache') else None

                if cache_client and hasattr(cache_client, 'keys'):
                    # Redis backend - can query by pattern
                    pattern = 'voting_result_*'
                    keys = cache_client.keys(pattern)

                    for key in keys:
                        # Extract timestamp from key
                        try:
                            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                            timestamp_str = key_str.split('_')[-1]
                            timestamp = int(timestamp_str)

                            # Only include results within time window
                            if timestamp >= cutoff_timestamp:
                                result = self.cache.get(key_str)
                                if result:
                                    cached_results.append(result)
                        except (ValueError, IndexError):
                            continue
                else:
                    # Non-Redis backend - fall back to tracking via performance history
                    logger.info("Cache backend doesn't support key pattern matching. Using limited analysis.")

            except Exception as cache_error:
                logger.warning(f"Could not query cache keys: {cache_error}. Using alternative analysis method.")

            # Compute statistics from cached results
            if cached_results:
                total_votes = len(cached_results)
                confidences = [r.get('confidence', 0.0) for r in cached_results]
                qualities = [r.get('quality', 0.0) for r in cached_results]
                model_counts = [r.get('model_count', 0) for r in cached_results]

                analysis['total_votes'] = total_votes
                analysis['average_confidence'] = statistics.mean(confidences) if confidences else 0.0
                analysis['average_quality'] = statistics.mean(qualities) if qualities else 0.0
                analysis['confidence_std'] = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
                analysis['quality_std'] = statistics.stdev(qualities) if len(qualities) > 1 else 0.0

                analysis['performance_metrics'] = {
                    'min_confidence': min(confidences) if confidences else 0.0,
                    'max_confidence': max(confidences) if confidences else 0.0,
                    'median_confidence': statistics.median(confidences) if confidences else 0.0,
                    'min_quality': min(qualities) if qualities else 0.0,
                    'max_quality': max(qualities) if qualities else 0.0,
                    'median_quality': statistics.median(qualities) if qualities else 0.0,
                    'avg_model_count': statistics.mean(model_counts) if model_counts else 0.0
                }

                # Generate recommendations based on metrics
                recommendations = []

                # Low confidence warning
                if analysis['average_confidence'] < 0.5:
                    recommendations.append({
                        'type': 'warning',
                        'metric': 'confidence',
                        'message': f"Average confidence is low ({analysis['average_confidence']:.2f}). Consider retraining models or adjusting voting strategy."
                    })

                # High variance warning
                if analysis.get('confidence_std', 0) > 0.3:
                    recommendations.append({
                        'type': 'warning',
                        'metric': 'variance',
                        'message': f"High confidence variance ({analysis['confidence_std']:.2f}) indicates inconsistent predictions. Review model diversity."
                    })

                # Quality recommendations
                if analysis['average_quality'] < 0.6:
                    recommendations.append({
                        'type': 'improvement',
                        'metric': 'quality',
                        'message': f"Average quality score is {analysis['average_quality']:.2f}. Consider ensemble optimization or model updates."
                    })
                elif analysis['average_quality'] > 0.85:
                    recommendations.append({
                        'type': 'success',
                        'metric': 'quality',
                        'message': f"Excellent quality score ({analysis['average_quality']:.2f}). Current ensemble is performing well."
                    })

                # Model count recommendations
                avg_model_count = analysis['performance_metrics']['avg_model_count']
                if avg_model_count < 2:
                    recommendations.append({
                        'type': 'warning',
                        'metric': 'ensemble_size',
                        'message': f"Low average model count ({avg_model_count:.1f}). Add more models for better ensemble performance."
                    })

                analysis['recommendations'] = recommendations

            else:
                # No cached results found
                analysis['recommendations'] = [{
                    'type': 'info',
                    'metric': 'data',
                    'message': f"No voting results found in the last {days_back} days. System may be new or cache may have been cleared."
                }]

            # Add consensus trends if we have enough data
            if len(cached_results) >= 10:
                # Split results into time buckets to show trends
                half_point = len(cached_results) // 2
                recent_half = cached_results[half_point:]
                older_half = cached_results[:half_point]

                recent_conf = statistics.mean([r.get('confidence', 0.0) for r in recent_half])
                older_conf = statistics.mean([r.get('confidence', 0.0) for r in older_half])

                analysis['consensus_trends'] = {
                    'confidence_trend': 'improving' if recent_conf > older_conf else 'declining',
                    'recent_confidence': recent_conf,
                    'older_confidence': older_conf,
                    'change_percentage': ((recent_conf - older_conf) / older_conf * 100) if older_conf > 0 else 0.0
                }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing voting performance: {e}", exc_info=True)
            return {
                'error': str(e),
                'total_votes': 0,
                'recommendations': [{
                    'type': 'error',
                    'metric': 'system',
                    'message': f"Analysis failed: {str(e)}"
                }]
            }

    def get_optimal_strategy(self, query_characteristics: Dict[str, Any]) -> VotingStrategy:
        """Recommend optimal voting strategy based on query characteristics."""
        try:
            # Analyze query characteristics to recommend strategy
            complexity = query_characteristics.get('complexity', 0.5)
            uncertainty = query_characteristics.get('uncertainty', 0.5)
            model_count = query_characteristics.get('model_count', 3)

            if model_count < 2:
                return VotingStrategy.SIMPLE_AVERAGE

            if complexity > 0.8 and uncertainty > 0.6:
                return VotingStrategy.ADAPTIVE_VOTING
            elif uncertainty > 0.7:
                return VotingStrategy.CONFIDENCE_WEIGHTED
            elif complexity > 0.6:
                return VotingStrategy.WEIGHTED_AVERAGE
            else:
                return VotingStrategy.SIMPLE_AVERAGE

        except Exception as e:
            logger.error(f"Error determining optimal strategy: {e}")
            return VotingStrategy.ADAPTIVE_VOTING


# Global voting system instance
voting_system = EnsembleVotingSystem()


# Django integration functions
def perform_ensemble_vote(predictions: List[Dict[str, Any]],
                         strategy: str = "adaptive_voting",
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Perform ensemble voting with predictions from multiple models."""
    try:
        # Convert dict predictions to ModelPrediction objects
        model_predictions = []
        for pred_dict in predictions:
            model_pred = ModelPrediction(
                model_id=pred_dict['model_id'],
                model_type=pred_dict['model_type'],
                prediction=pred_dict['prediction'],
                confidence=pred_dict['confidence'],
                processing_time_ms=pred_dict.get('processing_time_ms', 0.0)
            )
            model_predictions.append(model_pred)

        # Get voting strategy
        strategy_enum = VotingStrategy(strategy)

        # Perform voting
        result = voting_system.vote(model_predictions, strategy_enum, context)

        return asdict(result)

    except Exception as e:
        logger.error(f"Error in ensemble voting: {e}")
        return {
            'final_prediction': 50.0,
            'final_confidence': 0.1,
            'error': str(e)
        }


def get_recommended_voting_strategy(query_characteristics: Dict[str, Any]) -> str:
    """Get recommended voting strategy for query characteristics."""
    strategy = voting_system.get_optimal_strategy(query_characteristics)
    return strategy.value


def analyze_ensemble_voting_performance(days_back: int = 7) -> Dict[str, Any]:
    """Analyze ensemble voting performance."""
    return voting_system.analyze_voting_performance(days_back)


# Usage example
if __name__ == "__main__":
    # Test the voting system
    test_predictions = [
        ModelPrediction("rf_1", "random_forest", 75.0, 0.8, 50.0),
        ModelPrediction("xgb_1", "xgboost", 78.0, 0.9, 45.0),
        ModelPrediction("nn_1", "neural_network", 72.0, 0.7, 80.0),
    ]

    system = EnsembleVotingSystem()

    # Test different strategies
    strategies = [
        VotingStrategy.SIMPLE_AVERAGE,
        VotingStrategy.CONFIDENCE_WEIGHTED,
        VotingStrategy.ADAPTIVE_VOTING
    ]

    for strategy in strategies:
        result = system.vote(test_predictions, strategy)
        print(f"\n{strategy.value}:")
        print(f"  Prediction: {result.final_prediction:.2f}")
        print(f"  Confidence: {result.final_confidence:.2f}")
        print(f"  Quality: {result.quality_score:.2f}")
        print(f"  Explanation: {result.explanation}")