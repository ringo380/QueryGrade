"""
Model Manager for QueryGrade ML System

Centralized model loading, versioning, caching, and lifecycle management.
Consolidates model loading logic from hybrid_grader.py, multi_model_ensemble.py,
and training_pipeline.py into a single, reusable service.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from django.conf import settings
from django.utils import timezone

from ...models import MLModel

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of ML models in the system."""

    HYBRID_SCORER = "HYBRID_SCORER"
    QUERY_GRADER = "QUERY_GRADER"
    RANDOM_FOREST = "RANDOM_FOREST"
    XGBOOST = "XGBOOST"
    GRADIENT_BOOSTING = "GRADIENT_BOOSTING"
    NEURAL_NETWORK = "NEURAL_NETWORK"
    ENSEMBLE = "ENSEMBLE"


class ModelStatus(str, Enum):
    """Model deployment status."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    TRAINING = "TRAINING"
    FAILED = "FAILED"
    DEPRECATED = "DEPRECATED"


@dataclass
class LoadedModel:
    """Container for loaded model with metadata."""

    model: Any  # The actual ML model object
    model_id: int
    model_type: str
    version: str
    confidence: float  # Validation accuracy or confidence score
    file_path: str
    created_at: datetime
    performance_metrics: Dict[str, Any]
    load_time: float  # Time taken to load the model


@dataclass
class ModelCacheEntry:
    """Cache entry for model with TTL tracking."""

    loaded_model: LoadedModel
    cached_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class ModelManager:
    """
    Centralized manager for ML model loading, versioning, and caching.

    Features:
    - Single source of truth for model loading
    - In-memory caching with TTL
    - Version management and fallback
    - Performance tracking
    - Thread-safe model access
    - Automatic cleanup and memory management
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        cache_ttl: int = 3600,  # 1 hour default cache TTL
        enable_cache: bool = True,
    ):
        """
        Initialize ModelManager.

        Args:
            model_dir: Directory where models are stored
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to enable in-memory caching
        """
        self.model_dir = model_dir or os.path.join(
            settings.BASE_DIR, "analyzer", "ml", "models"
        )
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache

        # In-memory cache for loaded models
        self._model_cache: Dict[str, ModelCacheEntry] = {}

        # Ensure model directory exists
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"ModelManager initialized with model_dir: {self.model_dir}")

    def load_active_model(
        self, model_type: str, use_cache: bool = True
    ) -> Optional[LoadedModel]:
        """
        Load the currently active model of specified type.

        Args:
            model_type: Type of model to load (e.g., 'HYBRID_SCORER')
            use_cache: Whether to use cached model if available

        Returns:
            LoadedModel object or None if no active model found
        """
        cache_key = f"active_model_{model_type}"

        # Check cache first
        if use_cache and self.enable_cache:
            cached_model = self._get_from_cache(cache_key)
            if cached_model:
                logger.debug(f"Loaded {model_type} from cache")
                return cached_model

        try:
            # Query database for active model
            active_model = (
                MLModel.objects.filter(model_type=model_type, status="ACTIVE")
                .order_by("-created_at")
                .first()
            )

            if not active_model:
                logger.info(f"No active model found for type: {model_type}")
                return None

            # Load the model file
            loaded_model = self._load_model_from_db(active_model)

            # Cache the loaded model
            if self.enable_cache and loaded_model:
                self._add_to_cache(cache_key, loaded_model)

            return loaded_model

        except Exception as e:
            logger.error(
                f"Error loading active model {model_type}: {str(e)}", exc_info=True
            )
            return None

    def load_model_by_id(
        self, model_id: int, use_cache: bool = True
    ) -> Optional[LoadedModel]:
        """
        Load a specific model by database ID.

        Args:
            model_id: Database ID of the model
            use_cache: Whether to use cached model if available

        Returns:
            LoadedModel object or None if not found
        """
        cache_key = f"model_id_{model_id}"

        # Check cache
        if use_cache and self.enable_cache:
            cached_model = self._get_from_cache(cache_key)
            if cached_model:
                return cached_model

        try:
            model_record = MLModel.objects.get(id=model_id)
            loaded_model = self._load_model_from_db(model_record)

            if self.enable_cache and loaded_model:
                self._add_to_cache(cache_key, loaded_model)

            return loaded_model

        except MLModel.DoesNotExist:
            logger.error(f"Model with ID {model_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}", exc_info=True)
            return None

    def load_model_by_version(
        self, model_type: str, version: str, use_cache: bool = True
    ) -> Optional[LoadedModel]:
        """
        Load a specific model version.

        Args:
            model_type: Type of model
            version: Model version string
            use_cache: Whether to use cache

        Returns:
            LoadedModel object or None
        """
        cache_key = f"model_{model_type}_{version}"

        if use_cache and self.enable_cache:
            cached_model = self._get_from_cache(cache_key)
            if cached_model:
                return cached_model

        try:
            model_record = MLModel.objects.filter(
                model_type=model_type, version=version
            ).first()

            if not model_record:
                logger.warning(f"Model {model_type} v{version} not found")
                return None

            loaded_model = self._load_model_from_db(model_record)

            if self.enable_cache and loaded_model:
                self._add_to_cache(cache_key, loaded_model)

            return loaded_model

        except Exception as e:
            logger.error(
                f"Error loading model {model_type} v{version}: {str(e)}", exc_info=True
            )
            return None

    def load_all_active_models(self) -> Dict[str, LoadedModel]:
        """
        Load all active models across all types.

        Returns:
            Dictionary mapping model_type to LoadedModel
        """
        active_models = {}

        try:
            model_records = MLModel.objects.filter(status="ACTIVE")

            for record in model_records:
                loaded_model = self._load_model_from_db(record)
                if loaded_model:
                    active_models[record.model_type] = loaded_model

            logger.info(f"Loaded {len(active_models)} active models")
            return active_models

        except Exception as e:
            logger.error(f"Error loading all active models: {str(e)}", exc_info=True)
            return {}

    def _load_model_from_db(self, model_record: MLModel) -> Optional[LoadedModel]:
        """
        Load model from database record and file.

        Args:
            model_record: MLModel database record

        Returns:
            LoadedModel object or None on failure
        """
        start_time = timezone.now()

        try:
            # Construct full file path
            model_file_path = os.path.join(self.model_dir, model_record.file_path)

            # Validate file exists
            if not os.path.exists(model_file_path):
                logger.error(f"Model file not found: {model_file_path}")
                return None

            # Load the model using joblib
            model = joblib.load(model_file_path)

            # Calculate load time
            load_time = (timezone.now() - start_time).total_seconds()

            # Create LoadedModel container
            loaded_model = LoadedModel(
                model=model,
                model_id=model_record.id,
                model_type=model_record.model_type,
                version=model_record.version,
                confidence=model_record.validation_accuracy or 0.5,
                file_path=model_file_path,
                created_at=model_record.created_at,
                performance_metrics=model_record.performance_metrics or {},
                load_time=load_time,
            )

            logger.info(
                f"Loaded model: {model_record.name} v{model_record.version} "
                f"(type: {model_record.model_type}, confidence: {loaded_model.confidence:.3f}, "
                f"load_time: {load_time:.3f}s)"
            )

            return loaded_model

        except Exception as e:
            logger.error(
                f"Error loading model file {model_record.file_path}: {str(e)}",
                exc_info=True,
            )
            return None

    def save_model(
        self,
        model: Any,
        model_type: str,
        version: str,
        performance_metrics: Dict[str, Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        activate: bool = False,
    ) -> Optional[int]:
        """
        Save a trained model to disk and database.

        Args:
            model: Trained model object
            model_type: Type of model
            version: Version string
            performance_metrics: Performance metrics dictionary
            name: Optional model name
            description: Optional description
            activate: Whether to activate this model immediately

        Returns:
            Model ID if successful, None otherwise
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_type}_{version}_{timestamp}.pkl"
            file_path = os.path.join(self.model_dir, filename)

            # Save model to disk
            joblib.dump(model, file_path)
            logger.info(f"Saved model to {file_path}")

            # Create database record
            model_record = MLModel.objects.create(
                name=name or f"{model_type} v{version}",
                model_type=model_type,
                version=version,
                file_path=filename,  # Store relative path
                performance_metrics=performance_metrics,
                validation_accuracy=performance_metrics.get("validation_accuracy", 0.0),
                description=description or f"Auto-saved {model_type} model",
                status="ACTIVE" if activate else "INACTIVE",
            )

            # If activating, deactivate other models of same type
            if activate:
                self._activate_model(model_record.id, model_type)

            logger.info(f"Saved model to database: ID={model_record.id}")
            return model_record.id

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            return None

    def _activate_model(self, model_id: int, model_type: str):
        """
        Activate a model and deactivate others of same type.

        Args:
            model_id: ID of model to activate
            model_type: Type of model
        """
        try:
            # Deactivate all other models of this type
            MLModel.objects.filter(model_type=model_type, status="ACTIVE").exclude(
                id=model_id
            ).update(status="INACTIVE")

            # Activate the target model
            MLModel.objects.filter(id=model_id).update(status="ACTIVE")

            # Clear cache for this model type
            cache_key = f"active_model_{model_type}"
            if cache_key in self._model_cache:
                del self._model_cache[cache_key]

            logger.info(f"Activated model {model_id} for type {model_type}")

        except Exception as e:
            logger.error(f"Error activating model: {str(e)}", exc_info=True)

    def _get_from_cache(self, cache_key: str) -> Optional[LoadedModel]:
        """
        Retrieve model from in-memory cache.

        Args:
            cache_key: Cache key

        Returns:
            LoadedModel if valid cache entry exists, None otherwise
        """
        if cache_key not in self._model_cache:
            return None

        cache_entry = self._model_cache[cache_key]

        # Check if cache is expired
        age = (timezone.now() - cache_entry.cached_at).total_seconds()
        if age > self.cache_ttl:
            del self._model_cache[cache_key]
            return None

        # Update access tracking
        cache_entry.access_count += 1
        cache_entry.last_accessed = timezone.now()

        return cache_entry.loaded_model

    def _add_to_cache(self, cache_key: str, loaded_model: LoadedModel):
        """
        Add model to in-memory cache.

        Args:
            cache_key: Cache key
            loaded_model: LoadedModel to cache
        """
        cache_entry = ModelCacheEntry(
            loaded_model=loaded_model, cached_at=timezone.now(), access_count=0
        )
        self._model_cache[cache_key] = cache_entry

    def clear_cache(self, model_type: Optional[str] = None):
        """
        Clear model cache.

        Args:
            model_type: If provided, only clear cache for this type
        """
        if model_type:
            # Clear specific model type
            keys_to_delete = [
                key for key in self._model_cache.keys() if model_type in key
            ]
            for key in keys_to_delete:
                del self._model_cache[key]
            logger.info(f"Cleared cache for model type: {model_type}")
        else:
            # Clear all cache
            self._model_cache.clear()
            logger.info("Cleared all model cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about model cache.

        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self._model_cache)
        total_access = sum(entry.access_count for entry in self._model_cache.values())

        entries_by_type = {}
        for key, entry in self._model_cache.items():
            model_type = entry.loaded_model.model_type
            entries_by_type[model_type] = entries_by_type.get(model_type, 0) + 1

        return {
            "total_cached_models": total_entries,
            "total_cache_hits": total_access,
            "models_by_type": entries_by_type,
            "cache_ttl_seconds": self.cache_ttl,
            "cache_enabled": self.enable_cache,
        }

    def list_available_models(
        self, model_type: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available models with metadata.

        Args:
            model_type: Filter by model type
            status: Filter by status

        Returns:
            List of model metadata dictionaries
        """
        queryset = MLModel.objects.all()

        if model_type:
            queryset = queryset.filter(model_type=model_type)
        if status:
            queryset = queryset.filter(status=status)

        queryset = queryset.order_by("-created_at")

        models = []
        for record in queryset:
            models.append(
                {
                    "id": record.id,
                    "name": record.name,
                    "type": record.model_type,
                    "version": record.version,
                    "status": record.status,
                    "accuracy": record.validation_accuracy,
                    "created_at": record.created_at,
                    "file_path": record.file_path,
                    "metrics": record.performance_metrics,
                }
            )

        return models

    def cleanup_old_models(
        self, model_type: str, keep_count: int = 5, dry_run: bool = True
    ) -> Tuple[int, int]:
        """
        Clean up old inactive models, keeping only recent ones.

        Args:
            model_type: Type of models to clean
            keep_count: Number of models to keep
            dry_run: If True, only report what would be deleted

        Returns:
            Tuple of (models_deleted, files_deleted)
        """
        try:
            # Get inactive models, ordered by creation date
            old_models = MLModel.objects.filter(
                model_type=model_type, status="INACTIVE"
            ).order_by("-created_at")[keep_count:]

            models_to_delete = list(old_models)
            files_deleted = 0

            if dry_run:
                logger.info(
                    f"DRY RUN: Would delete {len(models_to_delete)} old {model_type} models"
                )
                return len(models_to_delete), 0

            # Delete model files and database records
            for model in models_to_delete:
                file_path = os.path.join(self.model_dir, model.file_path)

                # Delete file if it exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                    files_deleted += 1

                # Delete database record
                model.delete()

            logger.info(
                f"Cleaned up {len(models_to_delete)} old models, deleted {files_deleted} files"
            )
            return len(models_to_delete), files_deleted

        except Exception as e:
            logger.error(f"Error during model cleanup: {str(e)}", exc_info=True)
            return 0, 0


# Global singleton instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """
    Get the global ModelManager singleton instance.

    Returns:
        ModelManager instance
    """
    global _model_manager

    if _model_manager is None:
        _model_manager = ModelManager()

    return _model_manager
