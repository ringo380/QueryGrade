"""
Multi-Model Ensemble for QueryGrade ML System

This module implements multiple ML algorithms (Random Forest, XGBoost, Neural Networks)
with automatic model selection and ensemble capabilities for optimal query grading performance.
"""

import logging
import numpy as np
import pickle
import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import os

from django.conf import settings
from django.core.cache import caches
from django.utils import timezone
from django.db import transaction

try:
    # Core ML libraries
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Ensemble functionality will be limited.")

try:
    # XGBoost
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. XGBoost models will be skipped.")

try:
    # Neural Network libraries
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural network models will be skipped.")

from analyzer.models import Query, QueryAnalysis, TrainingData, MLModel
from ..core.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models available in the ensemble."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    GRADIENT_BOOSTING = "gradient_boosting"
    RIDGE = "ridge"
    ELASTIC_NET = "elastic_net"


@dataclass
class ModelConfiguration:
    """Configuration for a specific model type."""
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    preprocessing: str  # 'standard', 'robust', 'none'
    cross_validation_folds: int = 5
    early_stopping: bool = True
    feature_selection: bool = False


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    model_id: str
    model_type: ModelType
    training_score: float
    validation_score: float
    cross_val_mean: float
    cross_val_std: float
    training_time: float
    prediction_time: float
    memory_usage_mb: float
    feature_importance: Optional[List[float]] = None


@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    final_prediction: float
    confidence: float
    model_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    consensus_score: float  # How much models agree
    best_individual_model: str


class RandomForestModel:
    """Enhanced Random Forest model with advanced features."""

    def __init__(self, config: ModelConfiguration):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.trained = False

    def create_model(self, n_features: int) -> RandomForestRegressor:
        """Create Random Forest model with optimized hyperparameters."""
        params = self.config.hyperparameters

        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': 42
        }

        # Merge with provided hyperparameters
        final_params = {**default_params, **params}

        return RandomForestRegressor(**final_params)

    def train(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train the Random Forest model."""
        start_time = time.time()

        # Preprocessing
        X_processed = self._preprocess_features(X)

        # Create and train model
        self.model = self.create_model(X_processed.shape[1])
        self.model.fit(X_processed, y)

        # Calculate performance metrics
        training_score = self.model.score(X_processed, y)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_processed, y,
            cv=self.config.cross_validation_folds,
            scoring='r2'
        )

        training_time = time.time() - start_time
        self.trained = True

        return ModelPerformance(
            model_id=f"rf_{int(time.time())}",
            model_type=ModelType.RANDOM_FOREST,
            training_score=training_score,
            validation_score=cv_scores.mean(),
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std(),
            training_time=training_time,
            prediction_time=0.0,  # Will be measured during prediction
            memory_usage_mb=self._estimate_memory_usage(),
            feature_importance=self.model.feature_importances_.tolist() if self.model else None
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        X_processed = self._preprocess_features(X)
        return self.model.predict(X_processed)

    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """Preprocess features according to configuration."""
        if self.config.preprocessing == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        elif self.config.preprocessing == 'robust':
            if self.scaler is None:
                self.scaler = RobustScaler()
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        else:
            return X

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the model."""
        if self.model is None:
            return 0.0

        # Rough estimation based on number of trees and features
        n_estimators = self.model.n_estimators
        n_features = self.model.n_features_in_
        estimated_mb = (n_estimators * n_features * 8) / (1024 * 1024)  # 8 bytes per float
        return estimated_mb


class XGBoostModel:
    """XGBoost model with advanced features and hyperparameter optimization."""

    def __init__(self, config: ModelConfiguration):
        self.config = config
        self.model = None
        self.scaler = None
        self.trained = False

    def create_model(self, n_features: int) -> 'xgb.XGBRegressor':
        """Create XGBoost model with optimized hyperparameters."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")

        params = self.config.hyperparameters

        default_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }

        final_params = {**default_params, **params}

        return xgb.XGBRegressor(**final_params)

    def train(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train the XGBoost model."""
        start_time = time.time()

        # Preprocessing
        X_processed = self._preprocess_features(X)

        # Create and train model
        self.model = self.create_model(X_processed.shape[1])

        # Training with early stopping
        if self.config.early_stopping and len(X) > 100:
            # Split for early stopping
            split_idx = int(0.8 * len(X))
            X_train, X_val = X_processed[:split_idx], X_processed[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_processed, y)

        # Calculate performance metrics
        training_score = self.model.score(X_processed, y)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_processed, y,
            cv=self.config.cross_validation_folds,
            scoring='r2'
        )

        training_time = time.time() - start_time
        self.trained = True

        return ModelPerformance(
            model_id=f"xgb_{int(time.time())}",
            model_type=ModelType.XGBOOST,
            training_score=training_score,
            validation_score=cv_scores.mean(),
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std(),
            training_time=training_time,
            prediction_time=0.0,
            memory_usage_mb=self._estimate_memory_usage(),
            feature_importance=self.model.feature_importances_.tolist() if self.model else None
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        X_processed = self._preprocess_features(X)
        return self.model.predict(X_processed)

    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """Preprocess features according to configuration."""
        if self.config.preprocessing == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        elif self.config.preprocessing == 'robust':
            if self.scaler is None:
                self.scaler = RobustScaler()
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        else:
            return X

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the model."""
        if self.model is None:
            return 0.0
        # XGBoost models are generally compact
        return 10.0  # Rough estimate in MB


class NeuralNetworkModel:
    """Neural Network model using TensorFlow/Keras."""

    def __init__(self, config: ModelConfiguration):
        self.config = config
        self.model = None
        self.scaler = None
        self.trained = False
        self.history = None

    def create_model(self, n_features: int) -> 'tf.keras.Model':
        """Create neural network model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available")

        params = self.config.hyperparameters

        # Default architecture
        hidden_layers = params.get('hidden_layers', [128, 64, 32])
        dropout_rate = params.get('dropout_rate', 0.3)
        activation = params.get('activation', 'relu')
        learning_rate = params.get('learning_rate', 0.001)

        # Build model
        model = keras.Sequential()

        # Input layer
        model.add(layers.Dense(
            hidden_layers[0],
            activation=activation,
            input_shape=(n_features,)
        ))
        model.add(layers.Dropout(dropout_rate))

        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout_rate))

        # Output layer
        model.add(layers.Dense(1, activation='linear'))

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )

        return model

    def train(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train the neural network model."""
        start_time = time.time()

        # Preprocessing
        X_processed = self._preprocess_features(X)

        # Create model
        self.model = self.create_model(X_processed.shape[1])

        # Training configuration
        params = self.config.hyperparameters
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        validation_split = params.get('validation_split', 0.2)

        # Callbacks
        callbacks = []
        if self.config.early_stopping:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        # Train model
        self.history = self.model.fit(
            X_processed, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )

        # Calculate performance metrics
        training_predictions = self.model.predict(X_processed, verbose=0)
        training_score = r2_score(y, training_predictions.flatten())

        # Validation score from training history
        validation_score = max(self.history.history.get('val_loss', [0]))
        validation_score = 1.0 - validation_score / 100.0  # Convert loss to R2-like score

        training_time = time.time() - start_time
        self.trained = True

        return ModelPerformance(
            model_id=f"nn_{int(time.time())}",
            model_type=ModelType.NEURAL_NETWORK,
            training_score=training_score,
            validation_score=validation_score,
            cross_val_mean=validation_score,
            cross_val_std=0.0,  # Not applicable for NN
            training_time=training_time,
            prediction_time=0.0,
            memory_usage_mb=self._estimate_memory_usage()
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        X_processed = self._preprocess_features(X)
        predictions = self.model.predict(X_processed, verbose=0)
        return predictions.flatten()

    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """Preprocess features for neural network."""
        # Neural networks typically need standardized features
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the neural network."""
        if self.model is None:
            return 0.0

        # Count parameters
        total_params = self.model.count_params()
        estimated_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per parameter
        return estimated_mb


class MultiModelEnsemble:
    """Manages multiple ML models and provides ensemble predictions."""

    def __init__(self):
        self.models = {}
        self.model_performances = {}
        self.feature_extractor = FeatureExtractor()
        self.ensemble_weights = {}
        self.model_configs = self._get_default_configurations()

        # Cache
        self.cache = caches['default']

        # Threading
        self.training_lock = threading.Lock()

    def _get_default_configurations(self) -> Dict[ModelType, ModelConfiguration]:
        """Get default configurations for all model types."""
        return {
            ModelType.RANDOM_FOREST: ModelConfiguration(
                model_type=ModelType.RANDOM_FOREST,
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5
                },
                preprocessing='none',
                cross_validation_folds=5
            ),

            ModelType.XGBOOST: ModelConfiguration(
                model_type=ModelType.XGBOOST,
                hyperparameters={
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.1
                },
                preprocessing='standard',
                cross_validation_folds=5,
                early_stopping=True
            ),

            ModelType.NEURAL_NETWORK: ModelConfiguration(
                model_type=ModelType.NEURAL_NETWORK,
                hyperparameters={
                    'hidden_layers': [128, 64, 32],
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'batch_size': 32
                },
                preprocessing='standard',
                cross_validation_folds=3,
                early_stopping=True
            ),

            ModelType.GRADIENT_BOOSTING: ModelConfiguration(
                model_type=ModelType.GRADIENT_BOOSTING,
                hyperparameters={
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 6
                },
                preprocessing='standard',
                cross_validation_folds=5
            )
        }

    def train_all_models(self, training_data: List['TrainingData']) -> Dict[ModelType, ModelPerformance]:
        """Train all available models with the given training data."""
        with self.training_lock:
            try:
                # Prepare training data
                X, y = self._prepare_training_data(training_data)

                if len(X) == 0:
                    logger.error("No valid training data available")
                    return {}

                logger.info(f"Training ensemble with {len(X)} samples")

                performances = {}

                # Train Random Forest
                if SKLEARN_AVAILABLE:
                    try:
                        rf_model = RandomForestModel(self.model_configs[ModelType.RANDOM_FOREST])
                        rf_performance = rf_model.train(X, y)
                        self.models[ModelType.RANDOM_FOREST] = rf_model
                        performances[ModelType.RANDOM_FOREST] = rf_performance
                        logger.info(f"Random Forest trained: R2={rf_performance.cross_val_mean:.3f}")
                    except Exception as e:
                        logger.error(f"Error training Random Forest: {e}")

                # Train XGBoost
                if XGBOOST_AVAILABLE:
                    try:
                        xgb_model = XGBoostModel(self.model_configs[ModelType.XGBOOST])
                        xgb_performance = xgb_model.train(X, y)
                        self.models[ModelType.XGBOOST] = xgb_model
                        performances[ModelType.XGBOOST] = xgb_performance
                        logger.info(f"XGBoost trained: R2={xgb_performance.cross_val_mean:.3f}")
                    except Exception as e:
                        logger.error(f"Error training XGBoost: {e}")

                # Train Neural Network
                if TENSORFLOW_AVAILABLE:
                    try:
                        nn_model = NeuralNetworkModel(self.model_configs[ModelType.NEURAL_NETWORK])
                        nn_performance = nn_model.train(X, y)
                        self.models[ModelType.NEURAL_NETWORK] = nn_model
                        performances[ModelType.NEURAL_NETWORK] = nn_performance
                        logger.info(f"Neural Network trained: R2={nn_performance.validation_score:.3f}")
                    except Exception as e:
                        logger.error(f"Error training Neural Network: {e}")

                # Train Gradient Boosting
                if SKLEARN_AVAILABLE:
                    try:
                        gb_model = self._create_sklearn_model(ModelType.GRADIENT_BOOSTING)
                        gb_performance = self._train_sklearn_model(gb_model, X, y, ModelType.GRADIENT_BOOSTING)
                        performances[ModelType.GRADIENT_BOOSTING] = gb_performance
                        logger.info(f"Gradient Boosting trained: R2={gb_performance.cross_val_mean:.3f}")
                    except Exception as e:
                        logger.error(f"Error training Gradient Boosting: {e}")

                # Calculate ensemble weights
                self._calculate_ensemble_weights(performances)

                # Store performances
                self.model_performances = performances

                # Cache results
                self._cache_ensemble_results(performances)

                logger.info(f"Ensemble training completed with {len(performances)} models")
                return performances

            except Exception as e:
                logger.error(f"Error in ensemble training: {e}")
                return {}

    def _prepare_training_data(self, training_data: List['TrainingData']) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model training."""
        X = []
        y = []

        for data in training_data:
            try:
                # Extract features
                features = self.feature_extractor.extract_features(data.query)
                if features is not None:
                    X.append(features)
                    # Convert user grade (1-5) to score (0-100)
                    target_score = (data.user_grade_avg - 1) * 25
                    y.append(target_score)
            except Exception as e:
                logger.warning(f"Error extracting features for query {data.query.id}: {e}")

        return np.array(X), np.array(y)

    def _create_sklearn_model(self, model_type: ModelType):
        """Create sklearn model based on type."""
        config = self.model_configs[model_type]
        params = config.hyperparameters

        if model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(**params, random_state=42)
        elif model_type == ModelType.RIDGE:
            return Ridge(**params, random_state=42)
        elif model_type == ModelType.ELASTIC_NET:
            return ElasticNet(**params, random_state=42)
        else:
            raise ValueError(f"Unknown sklearn model type: {model_type}")

    def _train_sklearn_model(self, model, X: np.ndarray, y: np.ndarray,
                           model_type: ModelType) -> ModelPerformance:
        """Train a generic sklearn model."""
        start_time = time.time()

        # Preprocessing
        config = self.model_configs[model_type]
        if config.preprocessing == 'standard':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        # Train model
        model.fit(X_scaled, y)

        # Store model and scaler
        self.models[model_type] = {'model': model, 'scaler': scaler if config.preprocessing == 'standard' else None}

        # Performance metrics
        training_score = model.score(X_scaled, y)
        cv_scores = cross_val_score(model, X_scaled, y, cv=config.cross_validation_folds, scoring='r2')

        training_time = time.time() - start_time

        return ModelPerformance(
            model_id=f"{model_type.value}_{int(time.time())}",
            model_type=model_type,
            training_score=training_score,
            validation_score=cv_scores.mean(),
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std(),
            training_time=training_time,
            prediction_time=0.0,
            memory_usage_mb=1.0  # Rough estimate
        )

    def _calculate_ensemble_weights(self, performances: Dict[ModelType, ModelPerformance]):
        """Calculate weights for ensemble based on model performances."""
        if not performances:
            return

        # Use cross-validation scores for weighting
        scores = {model_type: perf.cross_val_mean for model_type, perf in performances.items()}
        total_score = sum(max(0, score) for score in scores.values())

        if total_score > 0:
            self.ensemble_weights = {
                model_type: max(0, score) / total_score
                for model_type, score in scores.items()
            }
        else:
            # Equal weights if all models perform poorly
            n_models = len(performances)
            self.ensemble_weights = {
                model_type: 1.0 / n_models
                for model_type in performances.keys()
            }

        logger.info(f"Ensemble weights: {self.ensemble_weights}")

    def predict_ensemble(self, features: List[float]) -> EnsembleResult:
        """Make ensemble prediction using all available models."""
        try:
            X = np.array([features])
            model_predictions = {}

            # Get predictions from all models
            for model_type, model in self.models.items():
                try:
                    if isinstance(model, dict):  # sklearn models
                        sklearn_model = model['model']
                        scaler = model.get('scaler')
                        X_input = scaler.transform(X) if scaler else X
                        prediction = sklearn_model.predict(X_input)[0]
                    else:  # Custom model classes
                        prediction = model.predict(X)[0]

                    model_predictions[model_type.value] = float(prediction)

                except Exception as e:
                    logger.warning(f"Error getting prediction from {model_type.value}: {e}")

            if not model_predictions:
                return EnsembleResult(
                    final_prediction=50.0,  # Default score
                    confidence=0.1,
                    model_predictions={},
                    model_weights={},
                    consensus_score=0.0,
                    best_individual_model="none"
                )

            # Calculate weighted ensemble prediction
            weighted_sum = 0.0
            total_weight = 0.0

            for model_type_str, prediction in model_predictions.items():
                model_type = ModelType(model_type_str)
                weight = self.ensemble_weights.get(model_type, 0.0)
                weighted_sum += prediction * weight
                total_weight += weight

            final_prediction = weighted_sum / total_weight if total_weight > 0 else np.mean(list(model_predictions.values()))

            # Calculate consensus score (how much models agree)
            predictions_array = np.array(list(model_predictions.values()))
            consensus_score = 1.0 - (np.std(predictions_array) / max(1.0, np.mean(predictions_array)))
            consensus_score = max(0.0, min(1.0, consensus_score))

            # Calculate confidence based on consensus and model performance
            avg_performance = np.mean([perf.cross_val_mean for perf in self.model_performances.values()])
            confidence = (consensus_score * 0.6) + (avg_performance * 0.4)

            # Find best individual model
            best_model = max(self.model_performances.items(), key=lambda x: x[1].cross_val_mean)[0]

            # Convert weights to string keys for JSON serialization
            model_weights = {model_type.value: weight for model_type, weight in self.ensemble_weights.items()}

            return EnsembleResult(
                final_prediction=final_prediction,
                confidence=confidence,
                model_predictions=model_predictions,
                model_weights=model_weights,
                consensus_score=consensus_score,
                best_individual_model=best_model.value
            )

        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return EnsembleResult(
                final_prediction=50.0,
                confidence=0.1,
                model_predictions={},
                model_weights={},
                consensus_score=0.0,
                best_individual_model="error"
            )

    def get_best_model(self) -> Tuple[ModelType, ModelPerformance]:
        """Get the best performing individual model."""
        if not self.model_performances:
            raise ValueError("No models have been trained")

        best_model_type = max(
            self.model_performances.items(),
            key=lambda x: x[1].cross_val_mean
        )

        return best_model_type

    def _cache_ensemble_results(self, performances: Dict[ModelType, ModelPerformance]):
        """Cache ensemble training results."""
        cache_data = {
            'performances': {model_type.value: asdict(perf) for model_type, perf in performances.items()},
            'weights': {model_type.value: weight for model_type, weight in self.ensemble_weights.items()},
            'trained_at': timezone.now().isoformat(),
            'model_count': len(performances)
        }

        self.cache.set('ensemble_training_results', cache_data, timeout=86400)

    def save_ensemble(self, filepath: str) -> bool:
        """Save the entire ensemble to file."""
        try:
            ensemble_data = {
                'models': {},
                'performances': self.model_performances,
                'weights': self.ensemble_weights,
                'configs': self.model_configs,
                'saved_at': timezone.now().isoformat()
            }

            # Save models
            for model_type, model in self.models.items():
                model_path = f"{filepath}_{model_type.value}.pkl"
                joblib.dump(model, model_path)
                ensemble_data['models'][model_type.value] = model_path

            # Save ensemble metadata
            with open(f"{filepath}_ensemble.json", 'w') as f:
                json.dump(ensemble_data, f, indent=2, default=str)

            logger.info(f"Ensemble saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
            return False

    def load_ensemble(self, filepath: str) -> bool:
        """Load ensemble from file."""
        try:
            # Load ensemble metadata
            with open(f"{filepath}_ensemble.json", 'r') as f:
                ensemble_data = json.load(f)

            # Load models
            for model_type_str, model_path in ensemble_data['models'].items():
                model_type = ModelType(model_type_str)
                model = joblib.load(model_path)
                self.models[model_type] = model

            # Load other data
            self.model_performances = ensemble_data['performances']
            self.ensemble_weights = {
                ModelType(k): v for k, v in ensemble_data['weights'].items()
            }

            logger.info(f"Ensemble loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            return False


# Global ensemble instance
model_ensemble = MultiModelEnsemble()


# Django integration functions
def train_model_ensemble(training_data: List['TrainingData']) -> Dict[str, Any]:
    """Train the complete model ensemble."""
    performances = model_ensemble.train_all_models(training_data)
    return {
        'success': len(performances) > 0,
        'models_trained': list(performances.keys()),
        'best_model': model_ensemble.get_best_model()[0].value if performances else None,
        'training_results': {model_type.value: asdict(perf) for model_type, perf in performances.items()}
    }


def get_ensemble_prediction(features: List[float]) -> Dict[str, Any]:
    """Get ensemble prediction for features."""
    result = model_ensemble.predict_ensemble(features)
    return asdict(result)


def get_ensemble_status() -> Dict[str, Any]:
    """Get current ensemble status."""
    return {
        'models_available': [model_type.value for model_type in model_ensemble.models.keys()],
        'ensemble_weights': {model_type.value: weight for model_type, weight in model_ensemble.ensemble_weights.items()},
        'performances': {model_type.value: asdict(perf) for model_type, perf in model_ensemble.model_performances.items()},
        'best_model': model_ensemble.get_best_model()[0].value if model_ensemble.model_performances else None
    }


# Usage example
if __name__ == "__main__":
    # Test the ensemble system
    ensemble = MultiModelEnsemble()

    # Generate dummy training data
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.rand(100) * 100

    print("Multi-model ensemble system initialized")
    print(f"Available model types: {[mt.value for mt in ModelType]}")
    print(f"Libraries available - sklearn: {SKLEARN_AVAILABLE}, XGBoost: {XGBOOST_AVAILABLE}, TensorFlow: {TENSORFLOW_AVAILABLE}")