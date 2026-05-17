"""
Advanced Incremental Learning Engine for QueryGrade ML System

This module implements sophisticated online learning algorithms with memory management,
adaptive learning rates, and streaming data processing capabilities for continuous
model improvement without full retraining.
"""

import hashlib
import json
import logging
import math
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from django.core.cache import caches
from django.db import transaction
from django.utils import timezone

try:
    import joblib
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning(
        "scikit-learn not available. Online learning functionality will be limited."
    )

from analyzer.ml.core.feature_extractor import FeatureExtractor
from analyzer.models import LearningMetrics, MLModel, Query, QueryAnalysis, TrainingData

logger = logging.getLogger(__name__)


@dataclass
class LearningInstance:
    """Represents a single learning instance for incremental learning."""

    instance_id: str
    features: List[float]
    target: float
    weight: float
    timestamp: datetime
    query_id: int
    source: str  # 'feedback', 'benchmark', 'validation'
    confidence: float = 1.0
    processed: bool = False


@dataclass
class ConceptDriftAlert:
    """Alert for detected concept drift."""

    alert_id: str
    drift_type: str  # 'gradual', 'sudden', 'recurring'
    magnitude: float
    affected_features: List[int]
    detection_method: str
    timestamp: datetime
    recommended_action: str


@dataclass
class LearningMetrics:
    """Metrics for tracking learning performance."""

    window_size: int
    samples_processed: int
    average_loss: float
    learning_rate: float
    drift_score: float
    memory_usage_mb: float
    processing_time_ms: float
    accuracy_trend: str  # 'improving', 'stable', 'degrading'


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler for online learning."""

    def __init__(
        self, initial_lr: float = 0.01, min_lr: float = 1e-6, max_lr: float = 0.1
    ):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_lr = initial_lr

        # Performance tracking
        self.loss_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        self.last_improvement = 0
        self.patience_counter = 0
        self.patience_threshold = 10

    def update_learning_rate(
        self, current_loss: float, gradient_norm: float = None
    ) -> float:
        """Update learning rate based on loss and gradient information."""
        self.loss_history.append(current_loss)

        if len(self.loss_history) < 5:
            return self.current_lr

        # Calculate recent loss trend
        recent_losses = list(self.loss_history)[-5:]
        if len(recent_losses) >= 2:
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

            # Adjust learning rate based on trend
            if loss_trend < -0.01:  # Loss decreasing (good)
                self.current_lr = min(self.max_lr, self.current_lr * 1.05)
                self.patience_counter = 0
            elif loss_trend > 0.01:  # Loss increasing (bad)
                self.current_lr = max(self.min_lr, self.current_lr * 0.95)
                self.patience_counter += 1
            else:  # Loss stable
                self.patience_counter += 1

        # Reset learning rate if no improvement for too long
        if self.patience_counter > self.patience_threshold:
            self.current_lr = self.initial_lr * 0.5
            self.patience_counter = 0

        # Gradient-based adjustment (if available)
        if gradient_norm is not None:
            if gradient_norm > 10:  # Large gradients
                self.current_lr = max(self.min_lr, self.current_lr * 0.8)
            elif gradient_norm < 0.1:  # Small gradients
                self.current_lr = min(self.max_lr, self.current_lr * 1.1)

        self.lr_history.append(self.current_lr)
        return self.current_lr

    def get_adaptive_lr(self, epoch: int = None) -> float:
        """Get current adaptive learning rate."""
        return self.current_lr


class ConceptDriftDetector:
    """Detects concept drift in streaming data."""

    def __init__(self, window_size: int = 100, sensitivity: float = 0.05):
        self.window_size = window_size
        self.sensitivity = sensitivity

        # Error tracking
        self.error_window = deque(maxlen=window_size)
        self.prediction_window = deque(maxlen=window_size)
        self.target_window = deque(maxlen=window_size)

        # Drift detection state
        self.baseline_error = None
        self.drift_threshold = None
        self.consecutive_alerts = 0

    def add_prediction(self, prediction: float, target: float):
        """Add new prediction and target for drift detection."""
        error = abs(prediction - target)

        self.error_window.append(error)
        self.prediction_window.append(prediction)
        self.target_window.append(target)

        # Initialize baseline if needed
        if self.baseline_error is None and len(self.error_window) >= 20:
            self.baseline_error = np.mean(list(self.error_window)[:20])
            self.drift_threshold = self.baseline_error * (1 + self.sensitivity)

    def detect_drift(self) -> Optional[ConceptDriftAlert]:
        """Detect if concept drift has occurred."""
        if len(self.error_window) < self.window_size or self.drift_threshold is None:
            return None

        errors = list(self.error_window)
        recent_errors = errors[-20:]  # Last 20 samples
        recent_avg_error = np.mean(recent_errors)

        # Check for sudden drift
        if recent_avg_error > self.drift_threshold:
            self.consecutive_alerts += 1

            if self.consecutive_alerts >= 3:  # Confirm drift
                # Determine drift type
                drift_type = self._classify_drift_type(errors)
                magnitude = (
                    recent_avg_error - self.baseline_error
                ) / self.baseline_error

                alert = ConceptDriftAlert(
                    alert_id=f"drift_{int(time.time() * 1000)}",
                    drift_type=drift_type,
                    magnitude=magnitude,
                    affected_features=[],  # Could be enhanced to detect specific features
                    detection_method="error_threshold",
                    timestamp=timezone.now(),
                    recommended_action=self._get_recommended_action(
                        drift_type, magnitude
                    ),
                )

                # Reset for next detection
                self.consecutive_alerts = 0
                self.baseline_error = recent_avg_error
                self.drift_threshold = recent_avg_error * (1 + self.sensitivity)

                return alert
        else:
            self.consecutive_alerts = max(0, self.consecutive_alerts - 1)

        return None

    def _classify_drift_type(self, errors: List[float]) -> str:
        """Classify the type of drift based on error patterns."""
        if len(errors) < 50:
            return "sudden"

        # Analyze error trend
        mid_point = len(errors) // 2
        first_half = errors[:mid_point]
        second_half = errors[mid_point:]

        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)

        # Check for gradual vs sudden change
        change_ratio = second_avg / first_avg if first_avg > 0 else 1.0

        if change_ratio > 1.5:
            return "sudden"
        elif change_ratio > 1.2:
            return "gradual"
        else:
            return "recurring"

    def _get_recommended_action(self, drift_type: str, magnitude: float) -> str:
        """Get recommended action based on drift characteristics."""
        if magnitude > 0.5:  # High magnitude
            if drift_type == "sudden":
                return "immediate_retrain"
            else:
                return "increase_learning_rate"
        elif magnitude > 0.2:  # Medium magnitude
            return "adapt_learning_rate"
        else:  # Low magnitude
            return "monitor_closely"


class IncrementalRandomForest:
    """Incremental Random Forest implementation for streaming data."""

    def __init__(
        self, n_estimators: int = 10, max_depth: int = 5, memory_limit: int = 1000
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.memory_limit = memory_limit

        # Model components
        self.estimators = []
        self.feature_importances_ = None
        self.n_features_ = None

        # Memory management
        self.training_buffer = deque(maxlen=memory_limit)
        self.estimator_ages = []

    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None
    ):
        """Incrementally update the forest with new data."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if self.n_features_ is None:
            self.n_features_ = X.shape[1]

        # Add to training buffer
        for i in range(len(X)):
            weight = sample_weight[i] if sample_weight is not None else 1.0
            self.training_buffer.append((X[i], y[i], weight))

        # Initialize estimators if needed
        if not self.estimators:
            self._initialize_estimators()

        # Update estimators
        self._update_estimators()

        # Update feature importances
        self._update_feature_importances()

    def _initialize_estimators(self):
        """Initialize the ensemble estimators."""
        for i in range(self.n_estimators):
            # Use SGD for incremental learning
            estimator = SGDRegressor(
                learning_rate="adaptive", eta0=0.01, random_state=i
            )
            self.estimators.append(estimator)
            self.estimator_ages.append(0)

    def _update_estimators(self):
        """Update estimators with buffered data."""
        if len(self.training_buffer) < 10:  # Need minimum data
            return

        buffer_data = list(self.training_buffer)
        X_buffer = np.array([item[0] for item in buffer_data])
        y_buffer = np.array([item[1] for item in buffer_data])
        w_buffer = np.array([item[2] for item in buffer_data])

        # Update each estimator with different subsets
        for i, estimator in enumerate(self.estimators):
            # Bootstrap sampling for diversity
            indices = np.random.choice(
                len(buffer_data), size=min(len(buffer_data), 50), replace=True
            )

            X_subset = X_buffer[indices]
            y_subset = y_buffer[indices]
            w_subset = w_buffer[indices]

            # Partial fit
            if hasattr(estimator, "partial_fit"):
                estimator.partial_fit(X_subset, y_subset, sample_weight=w_subset)
            else:
                # If not incremental, retrain with recent data
                estimator.fit(X_subset, y_subset, sample_weight=w_subset)

            self.estimator_ages[i] += 1

        # Refresh old estimators
        self._refresh_old_estimators()

    def _refresh_old_estimators(self):
        """Refresh estimators that are too old."""
        max_age = 100  # Maximum age before refresh

        for i, age in enumerate(self.estimator_ages):
            if age > max_age:
                # Create new estimator
                new_estimator = SGDRegressor(
                    learning_rate="adaptive",
                    eta0=0.01,
                    random_state=i + int(time.time()),
                )

                # Train with recent buffer data
                if len(self.training_buffer) >= 20:
                    recent_data = list(self.training_buffer)[-50:]
                    X_recent = np.array([item[0] for item in recent_data])
                    y_recent = np.array([item[1] for item in recent_data])
                    w_recent = np.array([item[2] for item in recent_data])

                    new_estimator.fit(X_recent, y_recent, sample_weight=w_recent)

                self.estimators[i] = new_estimator
                self.estimator_ages[i] = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        if not self.estimators:
            return np.zeros(len(X))

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        predictions = []
        for estimator in self.estimators:
            if hasattr(estimator, "predict"):
                pred = estimator.predict(X)
                predictions.append(pred)

        if predictions:
            return np.mean(predictions, axis=0)
        else:
            return np.zeros(len(X))

    def _update_feature_importances(self):
        """Update feature importances from ensemble."""
        if not self.estimators or self.n_features_ is None:
            return

        importances = []
        for estimator in self.estimators:
            if hasattr(estimator, "coef_"):
                # For linear models, use absolute coefficients as importance
                imp = np.abs(estimator.coef_)
                if len(imp) == self.n_features_:
                    importances.append(imp)

        if importances:
            self.feature_importances_ = np.mean(importances, axis=0)
        else:
            self.feature_importances_ = np.ones(self.n_features_) / self.n_features_


class IncrementalLearningEngine:
    """Main engine for incremental learning with advanced capabilities."""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

        # Learning components
        self.primary_model = None
        self.backup_models = []
        self.scaler = StandardScaler()

        # Adaptive components
        self.lr_scheduler = AdaptiveLearningRateScheduler()
        self.drift_detector = ConceptDriftDetector()

        # Memory management
        self.instance_buffer = deque(maxlen=10000)
        self.performance_history = deque(maxlen=1000)

        # State tracking
        self.total_samples_processed = 0
        self.last_full_evaluation = timezone.now()
        self.model_version = "incremental_v1.0"

        # Threading
        self.processing_lock = threading.Lock()

        # Cache
        self.cache = caches["default"]

    def initialize_models(self) -> bool:
        """Initialize the incremental learning models."""
        try:
            if not SKLEARN_AVAILABLE:
                logger.error("scikit-learn not available for incremental learning")
                return False

            # Primary incremental model
            self.primary_model = IncrementalRandomForest(
                n_estimators=20, max_depth=8, memory_limit=5000
            )

            # Backup models for comparison.
            # PassiveAggressiveRegressor was deprecated in sklearn 1.8 and removed
            # in 1.10; the sklearn-recommended replacement is SGDRegressor with
            # the PA-1 configuration, which also accepts sample_weight in
            # partial_fit (PA did not).
            self.backup_models = [
                SGDRegressor(learning_rate="adaptive", eta0=0.01, random_state=42),
                SGDRegressor(
                    loss="epsilon_insensitive",
                    penalty=None,
                    learning_rate="pa1",
                    eta0=1.0,
                    random_state=42,
                ),
            ]

            logger.info("Incremental learning models initialized")
            return True

        except Exception as e:
            logger.error(f"Error initializing incremental models: {e}")
            return False

    def process_learning_instance(self, instance: LearningInstance) -> Dict[str, Any]:
        """Process a single learning instance."""
        with self.processing_lock:
            try:
                start_time = time.time()

                # Validate instance
                if not self._validate_instance(instance):
                    return {"success": False, "error": "Invalid instance"}

                # Add to buffer
                self.instance_buffer.append(instance)

                # Extract features and target
                X = np.array([instance.features])
                y = np.array([instance.target])
                weight = np.array([instance.weight])

                # Scale features
                if self.total_samples_processed > 0:
                    X_scaled = self.scaler.transform(X)
                else:
                    X_scaled = self.scaler.fit_transform(X)

                # Make prediction before update (for drift detection)
                if self.primary_model and self.total_samples_processed > 10:
                    prediction = self.primary_model.predict(X_scaled)[0]
                    self.drift_detector.add_prediction(prediction, instance.target)

                    # Check for drift
                    drift_alert = self.drift_detector.detect_drift()
                    if drift_alert:
                        self._handle_drift_alert(drift_alert)

                # Update models
                self._update_models(X_scaled, y, weight)

                # Calculate metrics
                processing_time = (time.time() - start_time) * 1000

                # Update counters
                self.total_samples_processed += 1
                instance.processed = True

                # Periodic evaluation
                if self.total_samples_processed % 100 == 0:
                    self._perform_periodic_evaluation()

                return {
                    "success": True,
                    "processing_time_ms": processing_time,
                    "samples_processed": self.total_samples_processed,
                    "drift_detected": (
                        drift_alert is not None if "drift_alert" in locals() else False
                    ),
                }

            except Exception as e:
                logger.error(f"Error processing learning instance: {e}")
                return {"success": False, "error": str(e)}

    def _validate_instance(self, instance: LearningInstance) -> bool:
        """Validate a learning instance."""
        if not instance.features or len(instance.features) == 0:
            return False

        if not isinstance(instance.target, (int, float)):
            return False

        if not 0 <= instance.weight <= 10:  # Reasonable weight range
            return False

        return True

    def _update_models(self, X: np.ndarray, y: np.ndarray, weight: np.ndarray):
        """Update all models with new data."""
        # Update primary model
        if self.primary_model:
            self.primary_model.partial_fit(X, y, sample_weight=weight)

        # Update backup models
        for model in self.backup_models:
            if hasattr(model, "partial_fit"):
                model.partial_fit(X, y, sample_weight=weight)

        # Update learning rate
        if len(self.performance_history) > 0:
            recent_loss = self.performance_history[-1]
            new_lr = self.lr_scheduler.update_learning_rate(recent_loss)

            # Apply new learning rate to SGD models
            for model in self.backup_models:
                if hasattr(model, "learning_rate") and hasattr(model, "eta0"):
                    model.eta0 = new_lr

    def _handle_drift_alert(self, alert: ConceptDriftAlert):
        """Handle concept drift alert."""
        logger.warning(
            f"Concept drift detected: {alert.drift_type} with magnitude {alert.magnitude:.3f}"
        )

        # Cache the alert
        self.cache.set(f"drift_alert_{alert.alert_id}", asdict(alert), timeout=86400)

        # Take action based on recommendation
        if alert.recommended_action == "immediate_retrain":
            self._trigger_emergency_retrain()
        elif alert.recommended_action == "increase_learning_rate":
            self.lr_scheduler.current_lr = min(
                self.lr_scheduler.max_lr, self.lr_scheduler.current_lr * 1.5
            )
        elif alert.recommended_action == "adapt_learning_rate":
            self.lr_scheduler.current_lr = self.lr_scheduler.initial_lr

    def _trigger_emergency_retrain(self):
        """Trigger emergency retrain of models."""
        try:
            # Retrain with recent buffer data
            if len(self.instance_buffer) >= 100:
                recent_instances = list(self.instance_buffer)[
                    -500:
                ]  # Use last 500 instances

                X = np.array([inst.features for inst in recent_instances])
                y = np.array([inst.target for inst in recent_instances])
                weights = np.array([inst.weight for inst in recent_instances])

                # Scale features
                X_scaled = self.scaler.fit_transform(X)

                # Reinitialize and retrain primary model
                self.primary_model = IncrementalRandomForest(
                    n_estimators=20, max_depth=8, memory_limit=5000
                )

                # Batch train the model
                for i in range(0, len(X_scaled), 50):  # Process in batches
                    batch_X = X_scaled[i : i + 50]
                    batch_y = y[i : i + 50]
                    batch_w = weights[i : i + 50]
                    self.primary_model.partial_fit(
                        batch_X, batch_y, sample_weight=batch_w
                    )

                logger.info("Emergency retrain completed")

        except Exception as e:
            logger.error(f"Error in emergency retrain: {e}")

    def _perform_periodic_evaluation(self):
        """Perform periodic evaluation of model performance."""
        try:
            if len(self.instance_buffer) < 50:
                return

            # Use recent instances for evaluation
            recent_instances = list(self.instance_buffer)[-50:]
            X = np.array([inst.features for inst in recent_instances])
            y = np.array([inst.target for inst in recent_instances])

            X_scaled = self.scaler.transform(X)

            # Evaluate primary model
            if self.primary_model:
                predictions = self.primary_model.predict(X_scaled)
                mse = mean_squared_error(y, predictions)
                mae = mean_absolute_error(y, predictions)
                r2 = r2_score(y, predictions)

                # Store performance metrics
                self.performance_history.append(mse)

                # Log performance
                logger.info(
                    f"Periodic evaluation - MSE: {mse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}"
                )

                # Cache metrics
                metrics = {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "samples_evaluated": len(recent_instances),
                    "timestamp": timezone.now().isoformat(),
                }
                self.cache.set("incremental_learning_metrics", metrics, timeout=3600)

        except Exception as e:
            logger.error(f"Error in periodic evaluation: {e}")

    def predict(self, features: List[float]) -> Tuple[float, float]:
        """Make prediction with confidence estimate."""
        try:
            if not self.primary_model:
                return 0.0, 0.0

            X = np.array([features])
            X_scaled = self.scaler.transform(X)

            # Primary prediction
            primary_pred = self.primary_model.predict(X_scaled)[0]

            # Backup predictions for confidence estimation
            backup_preds = []
            for model in self.backup_models:
                if hasattr(model, "predict"):
                    pred = model.predict(X_scaled)[0]
                    backup_preds.append(pred)

            # Calculate confidence based on agreement
            if backup_preds:
                all_preds = [primary_pred] + backup_preds
                pred_std = np.std(all_preds)
                confidence = max(0.1, 1.0 - (pred_std / max(1.0, np.mean(all_preds))))
            else:
                confidence = 0.8

            return primary_pred, confidence

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.0, 0.0

    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about the current model state."""
        insights = {
            "total_samples_processed": self.total_samples_processed,
            "buffer_size": len(self.instance_buffer),
            "current_learning_rate": self.lr_scheduler.current_lr,
            "model_version": self.model_version,
            "last_evaluation": self.last_full_evaluation.isoformat(),
        }

        # Performance insights
        if self.performance_history:
            recent_performance = list(self.performance_history)[-10:]
            insights["recent_avg_loss"] = np.mean(recent_performance)
            insights["performance_trend"] = self._analyze_performance_trend()

        # Feature importance (if available)
        if self.primary_model and hasattr(self.primary_model, "feature_importances_"):
            if self.primary_model.feature_importances_ is not None:
                insights["feature_importances"] = (
                    self.primary_model.feature_importances_.tolist()
                )

        return insights

    def _analyze_performance_trend(self) -> str:
        """Analyze recent performance trend."""
        if len(self.performance_history) < 10:
            return "insufficient_data"

        recent_losses = list(self.performance_history)[-10:]
        mid_point = len(recent_losses) // 2

        first_half = np.mean(recent_losses[:mid_point])
        second_half = np.mean(recent_losses[mid_point:])

        if second_half < first_half * 0.95:
            return "improving"
        elif second_half > first_half * 1.05:
            return "degrading"
        else:
            return "stable"

    def save_model_state(self, filepath: str) -> bool:
        """Save current model state to file."""
        try:
            state = {
                "primary_model": self.primary_model,
                "backup_models": self.backup_models,
                "scaler": self.scaler,
                "lr_scheduler": self.lr_scheduler,
                "total_samples_processed": self.total_samples_processed,
                "model_version": self.model_version,
                "performance_history": list(self.performance_history),
                "saved_at": timezone.now().isoformat(),
            }

            with open(filepath, "wb") as f:
                pickle.dump(state, f)

            logger.info(f"Model state saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving model state: {e}")
            return False

    def load_model_state(self, filepath: str) -> bool:
        """Load model state from file."""
        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

            self.primary_model = state["primary_model"]
            self.backup_models = state["backup_models"]
            self.scaler = state["scaler"]
            self.lr_scheduler = state["lr_scheduler"]
            self.total_samples_processed = state["total_samples_processed"]
            self.model_version = state["model_version"]
            self.performance_history = deque(state["performance_history"], maxlen=1000)

            logger.info(f"Model state loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            return False


# Usage example and testing
if __name__ == "__main__":
    # Test the incremental learning engine
    engine = IncrementalLearningEngine()

    if engine.initialize_models():
        # Simulate learning instances
        for i in range(100):
            instance = LearningInstance(
                instance_id=f"test_{i}",
                features=np.random.rand(10).tolist(),
                target=np.random.rand() * 100,
                weight=1.0,
                timestamp=timezone.now(),
                query_id=i,
                source="test",
            )

            result = engine.process_learning_instance(instance)
            if i % 20 == 0:
                print(f"Processed {i} instances: {result}")

        # Get insights
        insights = engine.get_model_insights()
        print(f"Model insights: {insights}")

        # Test prediction
        test_features = np.random.rand(10).tolist()
        prediction, confidence = engine.predict(test_features)
        print(f"Test prediction: {prediction:.2f} (confidence: {confidence:.2f})")
