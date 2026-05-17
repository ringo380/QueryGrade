"""
Real-time Feedback Loop Integration for QueryGrade ML System

This module implements a sophisticated real-time feedback integration system that
immediately incorporates user feedback into the ML model's learning process through
streaming updates and incremental learning.
"""

import asyncio
import json
import logging
import pickle
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from django.conf import settings
from django.core.cache import caches
from django.db import transaction
from django.utils import timezone

try:
    import joblib
    from sklearn.base import BaseEstimator
    from sklearn.metrics import mean_squared_error

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning(
        "scikit-learn not available. Real-time learning functionality will be limited."
    )

from analyzer.ml.core.feature_extractor import FeatureExtractor
from analyzer.ml.core.feedback_collector import FeedbackCollector
from analyzer.ml.core.hybrid_grader import HybridQueryGrader
from analyzer.models import (
    FeedbackLearning,
    LearningMetrics,
    MLModel,
    Query,
    QueryAnalysis,
    QueryFeedback,
    TrainingData,
    UserQueryHistory,
)

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """Represents a feedback event in the real-time system."""

    event_id: str
    user_id: int
    query_id: int
    analysis_id: int
    feedback_type: str  # 'rating', 'correction', 'validation'
    feedback_data: Dict[str, Any]
    original_score: float
    original_grade: str
    timestamp: datetime
    processed: bool = False
    processing_time: Optional[float] = None


@dataclass
class ModelUpdateEvent:
    """Represents a model update event."""

    update_id: str
    trigger_type: str  # 'feedback_threshold', 'time_based', 'performance_drift'
    affected_samples: int
    performance_improvement: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


class FeedbackBuffer:
    """Manages buffering and batching of feedback events."""

    def __init__(self, max_size: int = 1000, flush_interval: int = 300):
        self.max_size = max_size
        self.flush_interval = flush_interval  # seconds
        self.buffer = deque(maxlen=max_size)
        self.last_flush = time.time()
        self.lock = threading.Lock()

    def add_feedback(self, feedback_event: FeedbackEvent):
        """Add a feedback event to the buffer."""
        with self.lock:
            self.buffer.append(feedback_event)

    def should_flush(self) -> bool:
        """Determine if buffer should be flushed."""
        with self.lock:
            time_threshold = time.time() - self.last_flush > self.flush_interval
            size_threshold = len(self.buffer) >= self.max_size
            return time_threshold or size_threshold

    def flush(self) -> List[FeedbackEvent]:
        """Flush and return all events in buffer."""
        with self.lock:
            events = list(self.buffer)
            self.buffer.clear()
            self.last_flush = time.time()
            return events

    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)


class OnlineLearningEngine:
    """Implements online/incremental learning capabilities."""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.current_model = None
        self.model_version = None
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.adaptation_window = 100  # Number of samples for adaptation

        # Performance tracking
        self.recent_errors = deque(maxlen=self.adaptation_window)
        self.baseline_error = None

        # Cache for recent predictions and feedback
        self.prediction_cache = caches["default"]

    def load_current_model(self) -> bool:
        """Load the current active model for online learning."""
        try:
            active_model = MLModel.objects.filter(
                model_type="HYBRID_SCORER", status="ACTIVE"
            ).first()

            if not active_model:
                logger.warning("No active model found for online learning")
                return False

            model_path = active_model.file_path
            self.current_model = joblib.load(model_path)
            self.model_version = active_model.version

            logger.info(
                f"Loaded model {active_model.name} v{self.model_version} for online learning"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading model for online learning: {e}")
            return False

    def update_model_incremental(
        self, feedback_events: List[FeedbackEvent]
    ) -> Dict[str, Any]:
        """Update model incrementally with new feedback."""
        if not self.current_model or not SKLEARN_AVAILABLE:
            return {"success": False, "error": "Model or sklearn not available"}

        try:
            # Prepare training data from feedback events
            X, y, weights = self._prepare_incremental_data(feedback_events)

            if len(X) == 0:
                return {"success": False, "error": "No valid training data"}

            # Check if model supports partial_fit
            if hasattr(self.current_model, "partial_fit"):
                # Direct incremental update. Older estimators
                # (e.g. PassiveAggressiveRegressor) do not accept sample_weight in
                # partial_fit, so retry without weights on TypeError rather than
                # let the whole update fail.
                try:
                    self.current_model.partial_fit(X, y, sample_weight=weights)
                except TypeError:
                    self.current_model.partial_fit(X, y)
                update_method = "partial_fit"
            else:
                # Simulate incremental learning with weighted update
                update_method = "weighted_update"
                self._simulate_incremental_update(X, y, weights)

            # Evaluate improvement
            improvement = self._evaluate_incremental_update(X, y)

            # Update performance tracking
            for event in feedback_events:
                error = abs(
                    event.feedback_data.get("corrected_score", event.original_score)
                    - event.original_score
                )
                self.recent_errors.append(error)

            logger.info(
                f"Incremental update completed: {len(feedback_events)} samples, method: {update_method}"
            )

            return {
                "success": True,
                "samples_processed": len(feedback_events),
                "method": update_method,
                "performance_improvement": improvement,
                "avg_recent_error": (
                    np.mean(self.recent_errors) if self.recent_errors else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error in incremental model update: {e}")
            return {"success": False, "error": str(e)}

    def _prepare_incremental_data(
        self, feedback_events: List[FeedbackEvent]
    ) -> Tuple[List, List, List]:
        """Prepare training data from feedback events."""
        X, y, weights = [], [], []

        for event in feedback_events:
            try:
                # Get query and extract features
                query = Query.objects.get(id=event.query_id)
                features = self.feature_extractor.extract_features(query)

                if features is None:
                    continue

                # Get target value from feedback
                if event.feedback_type == "rating":
                    # Convert 1-5 rating to 0-100 score
                    target_score = (event.feedback_data.get("rating", 3) - 1) * 25
                elif event.feedback_type == "correction":
                    # Direct score correction
                    target_score = event.feedback_data.get(
                        "corrected_score", event.original_score
                    )
                else:
                    continue

                # Calculate feedback weight based on user reliability and confidence
                weight = self._calculate_feedback_weight(event)

                X.append(features)
                y.append(target_score)
                weights.append(weight)

            except Exception as e:
                logger.warning(f"Error preparing data for event {event.event_id}: {e}")

        return X, y, weights

    def _calculate_feedback_weight(self, event: FeedbackEvent) -> float:
        """Calculate weight for feedback based on reliability and confidence."""
        base_weight = 1.0

        # User reliability (simplified - could be more sophisticated)
        user_feedback_count = QueryFeedback.objects.filter(
            user_history__user_id=event.user_id
        ).count()
        reliability_multiplier = min(2.0, 1.0 + (user_feedback_count * 0.1))

        # Feedback confidence based on type and data quality
        confidence_multiplier = 1.0
        if event.feedback_type == "rating":
            rating = event.feedback_data.get("rating", 3)
            confidence_multiplier = (
                0.8 if rating in [2, 4] else 1.0
            )  # Less confident for middle ratings
        elif event.feedback_type == "correction":
            confidence_multiplier = 1.5  # Higher confidence for explicit corrections

        # Recency weight (more recent feedback gets higher weight)
        hours_old = (timezone.now() - event.timestamp).total_seconds() / 3600
        recency_multiplier = max(0.5, 1.0 - (hours_old * 0.01))

        return (
            base_weight
            * reliability_multiplier
            * confidence_multiplier
            * recency_multiplier
        )

    def _simulate_incremental_update(self, X: List, y: List, weights: List):
        """Simulate incremental learning for models that don't support partial_fit."""
        # This is a simplified approach - in production, you might use more sophisticated methods
        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)

        # Get current predictions
        current_predictions = self.current_model.predict(X)

        # Calculate weighted error
        errors = (y - current_predictions) * weights

        # Apply momentum-based update (conceptual - actual implementation depends on model type)
        # This would need to be customized based on the specific model architecture
        logger.info(f"Simulated incremental update with {len(X)} samples")

    def _evaluate_incremental_update(self, X: List, y: List) -> float:
        """Evaluate the performance improvement from incremental update."""
        if not X or not y:
            return 0.0

        try:
            X = np.array(X)
            y = np.array(y)

            # Get predictions with updated model
            predictions = self.current_model.predict(X)
            current_error = mean_squared_error(y, predictions)

            # Compare with baseline if available
            if self.baseline_error is not None:
                improvement = self.baseline_error - current_error
                return improvement
            else:
                self.baseline_error = current_error
                return 0.0

        except Exception as e:
            logger.warning(f"Error evaluating incremental update: {e}")
            return 0.0

    def should_trigger_full_retrain(self) -> bool:
        """Determine if a full model retrain should be triggered."""
        if len(self.recent_errors) < self.adaptation_window:
            return False

        # Check for performance degradation
        recent_avg_error = np.mean(list(self.recent_errors)[-20:])  # Last 20 samples
        overall_avg_error = np.mean(self.recent_errors)

        if recent_avg_error > overall_avg_error * 1.5:
            logger.info("Performance degradation detected - triggering full retrain")
            return True

        # Check for concept drift (simplified)
        first_half = list(self.recent_errors)[: self.adaptation_window // 2]
        second_half = list(self.recent_errors)[self.adaptation_window // 2 :]

        if len(first_half) > 0 and len(second_half) > 0:
            drift_ratio = np.mean(second_half) / np.mean(first_half)
            if drift_ratio > 2.0:
                logger.info("Concept drift detected - triggering full retrain")
                return True

        return False


class RealTimeFeedbackProcessor:
    """Main processor for real-time feedback integration."""

    def __init__(self):
        self.feedback_buffer = FeedbackBuffer()
        self.online_engine = OnlineLearningEngine()
        self.feedback_collector = FeedbackCollector()
        self.hybrid_grader = HybridQueryGrader()

        # Processing queues
        self.feedback_queue = Queue()
        self.update_queue = Queue()

        # Control flags
        self.processing_active = False
        self.worker_threads = []

        # Performance monitoring
        self.processed_events = 0
        self.failed_events = 0
        self.last_model_update = timezone.now()

        # Initialize cache
        self.cache = caches["default"]

    def start_processing(self):
        """Start the real-time feedback processing system."""
        if self.processing_active:
            logger.warning("Real-time processing is already active")
            return

        self.processing_active = True

        # Load the current model for online learning
        if not self.online_engine.load_current_model():
            logger.error("Failed to load model for online learning")
            return

        # Start worker threads
        self.worker_threads = [
            threading.Thread(target=self._feedback_processor_worker, daemon=True),
            threading.Thread(target=self._model_updater_worker, daemon=True),
            threading.Thread(target=self._buffer_monitor_worker, daemon=True),
        ]

        for thread in self.worker_threads:
            thread.start()

        logger.info("Real-time feedback processing started")

    def stop_processing(self):
        """Stop the real-time feedback processing system."""
        self.processing_active = False

        # Wait for threads to finish (with timeout)
        for thread in self.worker_threads:
            thread.join(timeout=5)

        logger.info("Real-time feedback processing stopped")

    def submit_feedback(
        self,
        user_id: int,
        query_id: int,
        analysis_id: int,
        feedback_type: str,
        feedback_data: Dict[str, Any],
    ) -> str:
        """Submit feedback for real-time processing."""
        try:
            # Get original analysis for comparison
            analysis = QueryAnalysis.objects.get(id=analysis_id)

            # Create feedback event
            event = FeedbackEvent(
                event_id=f"fb_{int(time.time() * 1000000)}_{user_id}",
                user_id=user_id,
                query_id=query_id,
                analysis_id=analysis_id,
                feedback_type=feedback_type,
                feedback_data=feedback_data,
                original_score=analysis.score,
                original_grade=analysis.grade,
                timestamp=timezone.now(),
            )

            # Add to buffer
            self.feedback_buffer.add_feedback(event)

            # Cache the event for immediate access
            self.cache.set(
                f"feedback_event_{event.event_id}", asdict(event), timeout=3600
            )

            logger.info(f"Feedback submitted: {event.event_id}")
            return event.event_id

        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            raise

    def _feedback_processor_worker(self):
        """Worker thread for processing individual feedback events."""
        while self.processing_active:
            try:
                # Check if buffer should be flushed
                if self.feedback_buffer.should_flush():
                    events = self.feedback_buffer.flush()
                    if events:
                        for event in events:
                            self._process_single_feedback(event)

                time.sleep(1)  # Prevent busy waiting

            except Exception as e:
                logger.error(f"Error in feedback processor worker: {e}")
                time.sleep(5)  # Wait before retrying

    def _process_single_feedback(self, event: FeedbackEvent):
        """Process a single feedback event."""
        try:
            start_time = time.time()

            # Immediate prediction update (if applicable)
            if event.feedback_type == "correction":
                self._apply_immediate_correction(event)

            # Update traditional feedback collection
            self.feedback_collector.collect_feedback_for_query(event.query_id)

            # Mark as processed
            event.processed = True
            event.processing_time = time.time() - start_time

            # Update cache
            self.cache.set(
                f"feedback_event_{event.event_id}", asdict(event), timeout=3600
            )

            self.processed_events += 1

            # Check if model update should be triggered
            if self._should_trigger_model_update():
                self.update_queue.put(("feedback_threshold", event))

        except Exception as e:
            logger.error(f"Error processing feedback event {event.event_id}: {e}")
            self.failed_events += 1

    def _apply_immediate_correction(self, event: FeedbackEvent):
        """Apply immediate correction to model's understanding."""
        # This could involve updating cached predictions or model state
        corrected_score = event.feedback_data.get("corrected_score")
        if corrected_score is not None:
            # Update any cached predictions for this query
            cache_key = f"ml_prediction_{event.query_id}"
            cached_prediction = self.cache.get(cache_key)

            if cached_prediction:
                cached_prediction["corrected_score"] = corrected_score
                cached_prediction["correction_timestamp"] = timezone.now().isoformat()
                self.cache.set(cache_key, cached_prediction, timeout=86400)

    def _model_updater_worker(self):
        """Worker thread for updating the ML model."""
        while self.processing_active:
            try:
                # Check for update triggers
                try:
                    trigger_type, data = self.update_queue.get(timeout=10)
                    self._perform_model_update(trigger_type, data)
                except Empty:
                    continue  # No updates to process

            except Exception as e:
                logger.error(f"Error in model updater worker: {e}")
                time.sleep(5)

    def _perform_model_update(self, trigger_type: str, trigger_data: Any):
        """Perform incremental model update."""
        try:
            # Get recent feedback events for batch update
            events = self.feedback_buffer.flush()

            if not events:
                logger.info("No feedback events to process for model update")
                return

            # Perform incremental update
            result = self.online_engine.update_model_incremental(events)

            # Create update event
            update_event = ModelUpdateEvent(
                update_id=f"upd_{int(time.time() * 1000000)}",
                trigger_type=trigger_type,
                affected_samples=len(events),
                performance_improvement=result.get("performance_improvement", 0),
                timestamp=timezone.now(),
                success=result["success"],
                error_message=result.get("error") if not result["success"] else None,
            )

            # Log update
            logger.info(f"Model update completed: {update_event.update_id}")

            # Check if full retrain is needed
            if self.online_engine.should_trigger_full_retrain():
                self._trigger_full_retrain()

            self.last_model_update = timezone.now()

        except Exception as e:
            logger.error(f"Error performing model update: {e}")

    def _buffer_monitor_worker(self):
        """Worker thread for monitoring buffer and system health."""
        while self.processing_active:
            try:
                # Log system status
                buffer_size = self.feedback_buffer.size()
                if buffer_size > 0:
                    logger.debug(f"Feedback buffer size: {buffer_size}")

                # Check for system health issues
                error_rate = self.failed_events / max(
                    1, self.processed_events + self.failed_events
                )
                if error_rate > 0.1:  # 10% error rate
                    logger.warning(
                        f"High error rate in feedback processing: {error_rate:.2%}"
                    )

                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error in buffer monitor worker: {e}")
                time.sleep(30)

    def _should_trigger_model_update(self) -> bool:
        """Determine if model update should be triggered."""
        # Time-based trigger
        time_since_update = timezone.now() - self.last_model_update
        if time_since_update > timedelta(hours=1):
            return True

        # Buffer size trigger
        if self.feedback_buffer.size() > 50:
            return True

        # Error pattern trigger
        if self.online_engine.should_trigger_full_retrain():
            return True

        return False

    def _trigger_full_retrain(self):
        """Trigger a full model retrain."""
        try:
            # This would typically be handled by the existing training pipeline
            result = self.hybrid_grader.train_model(force_retrain=True)

            if result:
                logger.info(f"Full model retrain triggered successfully: {result.name}")
                # Reload the model in online engine
                self.online_engine.load_current_model()
            else:
                logger.warning("Full model retrain failed or was not needed")

        except Exception as e:
            logger.error(f"Error triggering full retrain: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "processing_active": self.processing_active,
            "buffer_size": self.feedback_buffer.size(),
            "processed_events": self.processed_events,
            "failed_events": self.failed_events,
            "error_rate": self.failed_events
            / max(1, self.processed_events + self.failed_events),
            "last_model_update": self.last_model_update.isoformat(),
            "worker_threads_alive": sum(1 for t in self.worker_threads if t.is_alive()),
            "model_version": self.online_engine.model_version,
            "recent_avg_error": (
                np.mean(self.online_engine.recent_errors)
                if self.online_engine.recent_errors
                else 0
            ),
        }


# Global instance for the application
feedback_processor = RealTimeFeedbackProcessor()


# Django integration functions
def start_realtime_feedback():
    """Start real-time feedback processing (called from Django app startup)."""
    feedback_processor.start_processing()


def stop_realtime_feedback():
    """Stop real-time feedback processing (called from Django app shutdown)."""
    feedback_processor.stop_processing()


def submit_user_feedback(
    user_id: int,
    query_id: int,
    analysis_id: int,
    feedback_type: str,
    feedback_data: Dict[str, Any],
) -> str:
    """Submit user feedback for real-time processing."""
    return feedback_processor.submit_feedback(
        user_id, query_id, analysis_id, feedback_type, feedback_data
    )


def get_feedback_system_status() -> Dict[str, Any]:
    """Get current feedback system status."""
    return feedback_processor.get_system_status()


# Usage example and testing
if __name__ == "__main__":
    # Example usage
    processor = RealTimeFeedbackProcessor()
    processor.start_processing()

    # Simulate feedback submission
    feedback_id = processor.submit_feedback(
        user_id=1,
        query_id=123,
        analysis_id=456,
        feedback_type="correction",
        feedback_data={
            "corrected_score": 85.0,
            "comment": "This query is actually efficient",
        },
    )

    print(f"Submitted feedback: {feedback_id}")

    # Check status
    status = processor.get_system_status()
    print(f"System status: {status}")

    # Stop processing
    time.sleep(5)
    processor.stop_processing()
