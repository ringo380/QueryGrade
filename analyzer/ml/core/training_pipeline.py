"""
Training Pipeline Manager for QueryGrade ML System

This module manages the complete machine learning training pipeline,
including data preparation, model training, validation, and deployment.
"""

import logging
import json
import os
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

from django.conf import settings
from django.utils import timezone
from django.db import transaction

from ...models import Query, QueryFeedback, MLModel, TrainingData, LearningMetrics
from .feature_extractor import FeatureExtractor
from .feedback_collector import FeedbackCollector

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    model_type: str = 'QUERY_GRADER'
    model_name: str = 'query_grader'
    algorithm: str = 'random_forest'  # random_forest, gradient_boosting, neural_network
    test_size: float = 0.2
    validation_size: float = 0.2
    cross_validation_folds: int = 5
    min_training_samples: int = 50
    max_training_samples: int = 10000
    feature_scaling: bool = True
    hyperparameter_tuning: bool = True
    model_versioning: bool = True
    auto_deployment: bool = False
    performance_threshold: float = 0.7  # Minimum accuracy for deployment


@dataclass
class TrainingResult:
    """Results from training pipeline execution."""
    success: bool
    model_version: str
    training_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    feature_importance: Dict[str, float]
    training_time: float
    model_path: str
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class TrainingPipelineManager:
    """Manages the complete ML training pipeline."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.feature_extractor = FeatureExtractor()
        self.feedback_collector = FeedbackCollector()
        self.scaler = StandardScaler() if self.config.feature_scaling else None

        # Ensure model directory exists
        self.model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
        os.makedirs(self.model_dir, exist_ok=True)

    def run_training_pipeline(self, force_retrain: bool = False) -> TrainingResult:
        """
        Run the complete training pipeline.

        Args:
            force_retrain: Force retraining even if recent model exists

        Returns:
            TrainingResult with pipeline execution results
        """
        start_time = timezone.now()

        try:
            logger.info("Starting ML training pipeline")

            # 1. Check if training is needed
            if not force_retrain and self._should_skip_training():
                logger.info("Skipping training - recent model exists")
                return self._get_existing_model_result()

            # 2. Prepare training data
            logger.info("Preparing training data")
            X, y, metadata = self._prepare_training_data()

            if X is None or len(X) < self.config.min_training_samples:
                return TrainingResult(
                    success=False,
                    model_version="",
                    training_accuracy=0.0,
                    validation_accuracy=0.0,
                    test_accuracy=0.0,
                    feature_importance={},
                    training_time=0.0,
                    model_path="",
                    error_message=f"Insufficient training data: {len(X) if X is not None else 0} samples"
                )

            # 3. Split data
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=42
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=self.config.validation_size, random_state=42
            )

            # 4. Feature scaling
            if self.config.feature_scaling:
                X_train = self.scaler.fit_transform(X_train)
                X_val = self.scaler.transform(X_val)
                X_test = self.scaler.transform(X_test)

            # 5. Train model
            logger.info(f"Training {self.config.algorithm} model")
            model = self._create_model()
            model.fit(X_train, y_train)

            # 6. Evaluate model
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            test_predictions = model.predict(X_test)

            training_accuracy = r2_score(y_train, train_predictions)
            validation_accuracy = r2_score(y_val, val_predictions)
            test_accuracy = r2_score(y_test, test_predictions)

            # 7. Cross-validation
            cv_scores = cross_val_score(model, X_train_val, y_train_val,
                                      cv=self.config.cross_validation_folds, scoring='r2')

            # 8. Feature importance
            feature_importance = self._get_feature_importance(model)

            # 9. Generate model version
            model_version = self._generate_model_version()

            # 10. Save model
            model_path = self._save_model(model, model_version)

            # 11. Record training metrics
            self._record_training_metrics(
                model_version, training_accuracy, validation_accuracy,
                test_accuracy, cv_scores, metadata
            )

            # 12. Deploy model if meets threshold
            if self.config.auto_deployment and validation_accuracy >= self.config.performance_threshold:
                self._deploy_model(model_version)

            training_time = (timezone.now() - start_time).total_seconds()

            logger.info(f"Training pipeline completed successfully in {training_time:.2f}s")

            return TrainingResult(
                success=True,
                model_version=model_version,
                training_accuracy=training_accuracy,
                validation_accuracy=validation_accuracy,
                test_accuracy=test_accuracy,
                feature_importance=feature_importance,
                training_time=training_time,
                model_path=model_path,
                metrics={
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'mse_train': mean_squared_error(y_train, train_predictions),
                    'mse_val': mean_squared_error(y_val, val_predictions),
                    'mse_test': mean_squared_error(y_test, test_predictions),
                    'mae_train': mean_absolute_error(y_train, train_predictions),
                    'mae_val': mean_absolute_error(y_val, val_predictions),
                    'mae_test': mean_absolute_error(y_test, test_predictions),
                    'training_samples': len(X_train),
                    'validation_samples': len(X_val),
                    'test_samples': len(X_test),
                }
            )

        except Exception as e:
            training_time = (timezone.now() - start_time).total_seconds()
            logger.error(f"Training pipeline failed: {str(e)}")

            return TrainingResult(
                success=False,
                model_version="",
                training_accuracy=0.0,
                validation_accuracy=0.0,
                test_accuracy=0.0,
                feature_importance={},
                training_time=training_time,
                model_path="",
                error_message=str(e)
            )

    def _should_skip_training(self) -> bool:
        """Check if training should be skipped based on existing models."""
        recent_models = MLModel.objects.filter(
            model_type=self.config.model_type,
            created_at__gte=timezone.now() - timedelta(hours=24)
        ).order_by('-created_at')

        if recent_models.exists():
            latest_model = recent_models.first()
            if latest_model.performance_metrics.get('validation_accuracy', 0) >= self.config.performance_threshold:
                return True

        return False

    def _get_existing_model_result(self) -> TrainingResult:
        """Get result for existing model."""
        latest_model = MLModel.objects.filter(
            model_type=self.config.model_type
        ).order_by('-created_at').first()

        if latest_model:
            metrics = latest_model.performance_metrics
            return TrainingResult(
                success=True,
                model_version=latest_model.version,
                training_accuracy=metrics.get('training_accuracy', 0.0),
                validation_accuracy=metrics.get('validation_accuracy', 0.0),
                test_accuracy=metrics.get('test_accuracy', 0.0),
                feature_importance=metrics.get('feature_importance', {}),
                training_time=0.0,
                model_path=latest_model.file_path,
                metrics=metrics
            )

        return TrainingResult(
            success=False,
            model_version="",
            training_accuracy=0.0,
            validation_accuracy=0.0,
            test_accuracy=0.0,
            feature_importance={},
            training_time=0.0,
            model_path="",
            error_message="No existing model found"
        )

    def _prepare_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """Prepare training data from collected feedback."""
        training_data = TrainingData.objects.all().order_by('-created_date')

        if self.config.max_training_samples:
            training_data = training_data[:self.config.max_training_samples]

        if not training_data.exists():
            logger.warning("No training data available")
            return None, None, {}

        # Extract features and targets
        X = []
        y = []
        weights = []

        for data in training_data:
            if data.features_json and len(data.features_json) > 0:
                X.append(data.features_json)
                y.append(data.target_score)
                weights.append(data.feedback_weight)

        if not X:
            return None, None, {}

        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)

        # Metadata about training data
        metadata = {
            'total_samples': len(X),
            'feature_count': X.shape[1] if len(X.shape) > 1 else 0,
            'target_mean': np.mean(y),
            'target_std': np.std(y),
            'weight_mean': np.mean(weights),
            'data_collection_period': self._get_data_collection_period(training_data)
        }

        logger.info(f"Prepared training data: {metadata}")
        return X, y, metadata

    def _create_model(self):
        """Create ML model based on configuration."""
        if self.config.algorithm == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.config.algorithm == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

    def _get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            feature_names = self.feature_extractor.get_feature_names()
            importance_dict = {}

            for i, importance in enumerate(model.feature_importances_):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = float(importance)

            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return {}

    def _generate_model_version(self) -> str:
        """Generate version string for new model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.model_name}_v{timestamp}"

    def _save_model(self, model, model_version: str) -> str:
        """Save trained model to disk."""
        model_filename = f"{model_version}.pkl"
        model_path = os.path.join(self.model_dir, model_filename)

        # Save model and scaler together
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_names': self.feature_extractor.get_feature_names(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

        return model_path

    def _record_training_metrics(self, model_version: str, training_accuracy: float,
                                validation_accuracy: float, test_accuracy: float,
                                cv_scores: np.ndarray, metadata: Dict[str, Any]):
        """Record training metrics in database."""
        with transaction.atomic():
            # Create MLModel record
            model_record = MLModel.objects.create(
                name=self.config.model_name,
                model_type=self.config.model_type,
                version=model_version,
                file_path=os.path.join(self.model_dir, f"{model_version}.pkl"),
                is_active=False,  # Not active until deployed
                description=f"Trained with {self.config.algorithm} algorithm",
                performance_metrics={
                    'training_accuracy': training_accuracy,
                    'validation_accuracy': validation_accuracy,
                    'test_accuracy': test_accuracy,
                    'cv_mean': float(np.mean(cv_scores)),
                    'cv_std': float(np.std(cv_scores)),
                    'algorithm': self.config.algorithm,
                    'training_samples': metadata.get('total_samples', 0)
                },
                training_data_count=metadata.get('total_samples', 0),
                last_trained=timezone.now()
            )

            # Create LearningMetrics record
            LearningMetrics.objects.create(
                model_version=model_version,
                training_accuracy=training_accuracy,
                validation_accuracy=validation_accuracy,
                feedback_correlation=0.0,  # To be calculated separately
                user_satisfaction_avg=0.0,  # To be calculated from feedback
                total_feedback_count=metadata.get('total_samples', 0)
            )

    def _deploy_model(self, model_version: str):
        """Deploy model by setting it as active."""
        with transaction.atomic():
            # Deactivate all existing models of this type
            MLModel.objects.filter(
                model_type=self.config.model_type,
                is_active=True
            ).update(is_active=False)

            # Activate the new model
            MLModel.objects.filter(
                version=model_version,
                model_type=self.config.model_type
            ).update(is_active=True)

        logger.info(f"Model {model_version} deployed successfully")

    def _get_data_collection_period(self, training_data) -> Dict[str, str]:
        """Get information about data collection period."""
        dates = [data.created_date for data in training_data if data.created_date]

        if dates:
            return {
                'start_date': min(dates).isoformat(),
                'end_date': max(dates).isoformat(),
                'span_days': (max(dates) - min(dates)).days
            }

        return {}

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training pipeline status."""
        latest_model = MLModel.objects.filter(
            model_type=self.config.model_type
        ).order_by('-created_at').first()

        training_data_count = TrainingData.objects.count()
        recent_feedback_count = QueryFeedback.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        ).count()

        return {
            'latest_model': {
                'version': latest_model.version if latest_model else None,
                'is_active': latest_model.is_active if latest_model else False,
                'performance': latest_model.performance_metrics if latest_model else {},
                'created_at': latest_model.created_at.isoformat() if latest_model else None
            },
            'training_data': {
                'total_samples': training_data_count,
                'ready_for_training': training_data_count >= self.config.min_training_samples
            },
            'recent_activity': {
                'feedback_last_week': recent_feedback_count,
                'training_recommended': recent_feedback_count > 10
            },
            'pipeline_config': {
                'algorithm': self.config.algorithm,
                'min_samples': self.config.min_training_samples,
                'auto_deployment': self.config.auto_deployment,
                'performance_threshold': self.config.performance_threshold
            }
        }

    def cleanup_old_models(self, keep_versions: int = 5):
        """Clean up old model files and database records."""
        old_models = MLModel.objects.filter(
            model_type=self.config.model_type,
            is_active=False
        ).order_by('-created_at')[keep_versions:]

        for model in old_models:
            # Remove file if exists
            if os.path.exists(model.file_path):
                try:
                    os.remove(model.file_path)
                    logger.info(f"Removed old model file: {model.file_path}")
                except OSError as e:
                    logger.warning(f"Could not remove model file {model.file_path}: {e}")

            # Remove database record
            model.delete()

        logger.info(f"Cleaned up {len(old_models)} old models")


class TrainingScheduler:
    """Manages automated training scheduling."""

    def __init__(self, pipeline_manager: TrainingPipelineManager):
        self.pipeline_manager = pipeline_manager

    def should_trigger_training(self) -> Tuple[bool, str]:
        """Determine if training should be triggered."""
        # Check if enough new feedback has been collected
        last_training = MLModel.objects.filter(
            model_type=self.pipeline_manager.config.model_type
        ).order_by('-created_at').first()

        if not last_training:
            return True, "No existing model found"

        # Check for new feedback since last training
        new_feedback_count = QueryFeedback.objects.filter(
            created_at__gt=last_training.created_at
        ).count()

        if new_feedback_count >= 50:  # Threshold for retraining
            return True, f"New feedback available: {new_feedback_count} items"

        # Check if model performance has degraded
        recent_metrics = LearningMetrics.objects.filter(
            model_version=last_training.version
        ).first()

        if recent_metrics and recent_metrics.validation_accuracy < 0.6:
            return True, "Model performance below threshold"

        # Check if model is too old
        model_age = timezone.now() - last_training.created_at
        if model_age > timedelta(days=30):
            return True, f"Model is {model_age.days} days old"

        return False, "Training not needed"

    def schedule_training(self) -> Dict[str, Any]:
        """Schedule or execute training based on conditions."""
        should_train, reason = self.should_trigger_training()

        if should_train:
            logger.info(f"Triggering training: {reason}")
            result = self.pipeline_manager.run_training_pipeline()

            return {
                'training_triggered': True,
                'reason': reason,
                'result': result
            }
        else:
            logger.info(f"Training not triggered: {reason}")
            return {
                'training_triggered': False,
                'reason': reason,
                'result': None
            }