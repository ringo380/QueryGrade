"""
Hybrid Query Grader for QueryGrade ML System

This module combines rule-based analysis with machine learning predictions
to provide improved query grading that learns from user feedback.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from django.conf import settings
from django.core.cache import caches
from django.utils import timezone

try:
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. ML functionality will be limited.")

from ...models import LearningMetrics, MLModel, Query, QueryAnalysis, TrainingData
from ...query_analyzer import QueryGrader as RuleBasedGrader
from .feature_extractor import FeatureExtractor
from .feedback_collector import FeedbackCollector

logger = logging.getLogger(__name__)


class HybridQueryGrader:
    """
    Hybrid grader that combines rule-based analysis with machine learning predictions.

    The system starts with primarily rule-based scoring and gradually increases
    ML weight as the model learns from user feedback.
    """

    def __init__(self):
        self.rule_based_grader = RuleBasedGrader()
        self.feature_extractor = FeatureExtractor()
        self.feedback_collector = FeedbackCollector()

        # Hybrid weighting (starts rule-heavy, becomes more ML-heavy over time)
        self.initial_rule_weight = 0.8
        self.initial_ml_weight = 0.2
        self.min_rule_weight = 0.3  # Never go below 30% rule-based
        self.max_ml_weight = 0.7  # Never go above 70% ML

        # Model management
        self.current_model = None
        self.model_confidence = 0.5
        self.model_path = os.path.join(settings.BASE_DIR, "ml_models")
        self.cache = caches["default"]

        # Training thresholds
        self.min_training_samples = 50
        self.retrain_threshold_days = 7
        self.performance_threshold = 0.7

    def analyze_query(
        self, sql_text: str, database_type: str = "", use_ml: bool = True
    ) -> Tuple[Query, QueryAnalysis]:
        """
        Analyze a SQL query using hybrid rule-based + ML approach.

        Args:
            sql_text: The SQL query to analyze
            database_type: Target database type
            use_ml: Whether to include ML predictions

        Returns:
            Tuple of Query and QueryAnalysis objects
        """
        start_time = timezone.now()

        # Get rule-based analysis first
        query, rule_analysis = self.rule_based_grader.analyze_query(
            sql_text, database_type
        )

        if not use_ml or not SKLEARN_AVAILABLE:
            # Return rule-based analysis only
            return query, rule_analysis

        try:
            # Get ML prediction if model is available
            ml_prediction = self._get_ml_prediction(query, database_type)

            if ml_prediction is not None:
                # Combine rule-based and ML predictions
                hybrid_score, hybrid_grade, confidence = self._combine_predictions(
                    rule_analysis.score, ml_prediction, query
                )

                # Update analysis with hybrid results
                rule_analysis.score = hybrid_score
                rule_analysis.grade = hybrid_grade

                # Add ML confidence info to performance notes
                original_notes = rule_analysis.performance_notes or ""
                ml_notes = f" | ML confidence: {confidence:.2f} | ML prediction: {ml_prediction:.1f}"
                rule_analysis.performance_notes = original_notes + ml_notes
                rule_analysis.save()

                # Log hybrid prediction for monitoring
                self._log_prediction(
                    query.id, rule_analysis.score, ml_prediction, confidence
                )

        except Exception as e:
            logger.warning(f"ML prediction failed for query {query.id}: {str(e)}")
            # Fall back to rule-based analysis

        return query, rule_analysis

    def _get_ml_prediction(self, query: Query, database_type: str) -> Optional[float]:
        """Get ML prediction for a query."""
        try:
            # Load current model if not already loaded
            if self.current_model is None:
                self.current_model = self._load_current_model()

            if self.current_model is None:
                return None

            # Extract features for the query
            features = self.feature_extractor.extract_features(query, database_type)
            if features is None:
                return None

            # Apply scaler if one was saved with the model
            import numpy as np

            feature_array = np.array([features])
            scaler = getattr(self, "_model_scaler", None)
            if scaler is not None:
                feature_array = scaler.transform(feature_array)

            prediction = self.current_model.predict(feature_array)[0]

            # Ensure prediction is in valid range (0-100)
            prediction = max(0, min(100, prediction))

            return float(prediction)

        except Exception as e:
            logger.error(f"Error getting ML prediction: {str(e)}")
            return None

    def _combine_predictions(
        self, rule_score: float, ml_score: float, query: Query
    ) -> Tuple[float, str, float]:
        """
        Combine rule-based and ML predictions with dynamic weighting.

        Args:
            rule_score: Score from rule-based analysis (0-100)
            ml_score: Score from ML model (0-100)
            query: Query object for context

        Returns:
            Tuple of (combined_score, grade, confidence)
        """
        # Calculate dynamic weights based on model performance and query characteristics
        rule_weight, ml_weight = self._calculate_dynamic_weights(query)

        # Combine scores
        combined_score = (rule_score * rule_weight) + (ml_score * ml_weight)

        # Calculate confidence based on agreement between methods
        score_difference = abs(rule_score - ml_score)
        agreement_factor = max(
            0, 1 - (score_difference / 50)
        )  # Closer scores = higher confidence
        confidence = (self.model_confidence * ml_weight) + (
            0.9 * rule_weight
        ) * agreement_factor

        # Convert to grade
        combined_grade = self._score_to_grade(combined_score)

        return combined_score, combined_grade, confidence

    def _calculate_dynamic_weights(self, query: Query) -> Tuple[float, float]:
        """Calculate dynamic weights for rule-based vs ML predictions."""
        base_rule_weight = self.initial_rule_weight
        base_ml_weight = self.initial_ml_weight

        # Adjust based on model performance
        if self.model_confidence > 0.8:
            # High confidence model gets more weight
            ml_bonus = min(0.2, (self.model_confidence - 0.8) * 0.5)
            base_ml_weight += ml_bonus
            base_rule_weight -= ml_bonus
        elif self.model_confidence < 0.6:
            # Low confidence model gets less weight
            ml_penalty = min(0.2, (0.6 - self.model_confidence) * 0.5)
            base_ml_weight -= ml_penalty
            base_rule_weight += ml_penalty

        # Adjust based on query complexity
        if query.estimated_complexity > 70:
            # Complex queries - rely more on ML for pattern recognition
            complexity_bonus = min(0.1, (query.estimated_complexity - 70) / 300)
            base_ml_weight += complexity_bonus
            base_rule_weight -= complexity_bonus

        # Ensure weights stay within bounds and sum to 1
        ml_weight = max(0.0, min(self.max_ml_weight, base_ml_weight))
        rule_weight = 1.0 - ml_weight
        rule_weight = max(self.min_rule_weight, rule_weight)
        ml_weight = 1.0 - rule_weight

        return rule_weight, ml_weight

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _load_current_model(self) -> Optional[Any]:
        """Load the current active ML model."""
        try:
            # Get active model from database
            active_model = MLModel.objects.filter(
                model_type="HYBRID_SCORER", status="ACTIVE"
            ).first()

            if not active_model:
                logger.info("No active ML model found")
                return None

            # Load model file
            model_file_path = os.path.join(self.model_path, active_model.file_path)
            if not os.path.exists(model_file_path):
                logger.error(f"Model file not found: {model_file_path}")
                return None

            # Load the model — file may be a raw estimator or a dict bundle
            model_data = joblib.load(model_file_path)
            if isinstance(model_data, dict):
                self._model_scaler = model_data.get("scaler")
                model = model_data["model"]
            else:
                self._model_scaler = None
                model = model_data
            self.model_confidence = active_model.validation_accuracy or 0.5

            logger.info(f"Loaded ML model: {active_model.name} v{active_model.version}")
            return model

        except Exception as e:
            logger.error(f"Error loading ML model: {str(e)}")
            return None

    def train_model(self, force_retrain: bool = False) -> Optional[MLModel]:
        """
        Train a new ML model from accumulated feedback data.

        Args:
            force_retrain: Whether to force retraining even if not needed

        Returns:
            MLModel object if training successful, None otherwise
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Cannot train model: scikit-learn not available")
            return None

        try:
            # Check if retraining is needed
            if not force_retrain and not self._should_retrain():
                logger.info("Model retraining not needed")
                return None

            # Collect training data
            training_data = self.feedback_collector.get_training_dataset(
                min_feedback_count=3, include_validated_only=False
            )

            if len(training_data) < self.min_training_samples:
                logger.warning(
                    f"Insufficient training data: {len(training_data)} < {self.min_training_samples}"
                )
                return None

            # Prepare features and targets
            X, y = self._prepare_training_data(training_data)

            if len(X) == 0:
                logger.error("No valid training samples after feature extraction")
                return None

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            model, scaler = self._train_model_pipeline(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logger.info(f"Model performance - MSE: {mse:.2f}, R2: {r2:.3f}")

            # Save model if performance is acceptable
            if r2 >= self.performance_threshold:
                saved_model = self._save_model(
                    model, scaler, len(training_data), r2, mse
                )
                self._update_model_confidence(r2)
                return saved_model
            else:
                logger.warning(
                    f"Model performance too low: R2={r2:.3f} < {self.performance_threshold}"
                )
                return None

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None

    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        # Check if we have an active model
        active_model = MLModel.objects.filter(
            model_type="HYBRID_SCORER", status="ACTIVE"
        ).first()

        if not active_model:
            return True  # No model exists, should train

        # Check if model is old
        if active_model.created_at < timezone.now() - timedelta(
            days=self.retrain_threshold_days
        ):
            return True

        # Check if we have significant new training data
        new_data_count = TrainingData.objects.filter(
            updated_at__gt=active_model.created_at
        ).count()

        if new_data_count >= 20:  # Threshold for new data
            return True

        return False

    def _prepare_training_data(
        self, training_data: List[TrainingData]
    ) -> Tuple[List, List]:
        """Prepare feature matrix and target vector for training."""
        X = []
        y = []

        for data in training_data:
            try:
                # Extract features for this query
                features = self.feature_extractor.extract_features(data.query)
                if features is not None:
                    X.append(features)
                    # Convert user grade (1-5) to score (0-100)
                    target_score = (data.user_grade_avg - 1) * 25
                    y.append(target_score)
            except Exception as e:
                logger.warning(
                    f"Error extracting features for query {data.query.id}: {str(e)}"
                )
                continue

        return X, y

    def _train_model_pipeline(self, X_train: List, y_train: List) -> Tuple[Any, Any]:
        """Train the ML model pipeline."""
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train ensemble model
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )

        model.fit(X_train_scaled, y_train)

        return model, scaler

    def _save_model(
        self,
        model: Any,
        scaler: Any,
        training_samples: int,
        validation_accuracy: float,
        training_loss: float,
    ) -> MLModel:
        """Save trained model to file and database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"hybrid_grader_{timestamp}.pkl"
        model_file_path = os.path.join(self.model_path, model_filename)

        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)

        # Save model and scaler together
        model_data = {
            "model": model,
            "scaler": scaler,
            "feature_names": self.feature_extractor.get_feature_names(),
            "trained_at": timezone.now(),
        }

        joblib.dump(model_data, model_file_path)

        # Calculate file size and checksum
        file_size = os.path.getsize(model_file_path)

        # Deactivate old models
        MLModel.objects.filter(model_type="HYBRID_SCORER", status="ACTIVE").update(
            status="DEPRECATED"
        )

        # Create database record
        ml_model = MLModel.objects.create(
            name=f"Hybrid Query Grader",
            model_type="HYBRID_SCORER",
            version=timestamp,
            status="ACTIVE",
            file_path=model_filename,
            file_size_bytes=file_size,
            training_accuracy=validation_accuracy,
            validation_accuracy=validation_accuracy,
            training_samples=training_samples,
            deployed_at=timezone.now(),
        )

        logger.info(
            f"Saved model: {model_filename} with accuracy {validation_accuracy:.3f}"
        )
        return ml_model

    def _update_model_confidence(self, r2_score: float):
        """Update internal model confidence based on performance."""
        # Convert R2 score to confidence (0-1 range)
        self.model_confidence = max(0.1, min(0.95, r2_score))

    def _log_prediction(
        self, query_id: int, final_score: float, ml_score: float, confidence: float
    ):
        """Log prediction for monitoring and evaluation."""
        try:
            # Store in cache for recent predictions monitoring
            prediction_data = {
                "query_id": query_id,
                "final_score": final_score,
                "ml_score": ml_score,
                "confidence": confidence,
                "timestamp": timezone.now().isoformat(),
            }

            cache_key = f"ml_prediction_{query_id}"
            self.cache.set(cache_key, prediction_data, timeout=86400)  # 24 hours

        except Exception as e:
            logger.warning(f"Error logging prediction: {str(e)}")

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and performance metrics."""
        try:
            active_model = MLModel.objects.filter(
                model_type="HYBRID_SCORER", status="ACTIVE"
            ).first()

            if not active_model:
                return {
                    "status": "No active model",
                    "model_available": False,
                }

            # Get recent metrics
            recent_metrics = LearningMetrics.objects.filter(model=active_model).first()

            return {
                "status": "Active",
                "model_available": True,
                "model_name": active_model.name,
                "version": active_model.version,
                "training_accuracy": active_model.training_accuracy,
                "validation_accuracy": active_model.validation_accuracy,
                "training_samples": active_model.training_samples,
                "deployed_at": active_model.deployed_at,
                "confidence": self.model_confidence,
                "recent_metrics": recent_metrics.__dict__ if recent_metrics else None,
            }

        except Exception as e:
            logger.error(f"Error getting model status: {str(e)}")
            return {
                "status": "Error",
                "model_available": False,
                "error": str(e),
            }
