"""
Machine learning models for ML system tracking and training.

This module contains models for tracking ML model versions, training data,
performance metrics, and learning from user feedback.
"""
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from .query_models import Query
from .user_models import UserQueryHistory


class MLModel(models.Model):
    """Model to track machine learning model versions and metadata."""

    MODEL_TYPES = [
        ('QUERY_GRADER', 'Query Grader'),
        ('FEATURE_EXTRACTOR', 'Feature Extractor'),
        ('FEEDBACK_PREDICTOR', 'Feedback Predictor'),
        ('HYBRID_SCORER', 'Hybrid Scorer'),
    ]

    MODEL_STATUS = [
        ('TRAINING', 'Training'),
        ('ACTIVE', 'Active'),
        ('DEPRECATED', 'Deprecated'),
        ('ARCHIVED', 'Archived'),
    ]

    name = models.CharField(max_length=100, help_text="Human-readable model name")
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    version = models.CharField(max_length=20, help_text="Model version (e.g., '1.0.0')")
    status = models.CharField(max_length=12, choices=MODEL_STATUS, default='TRAINING')

    # Model file information
    file_path = models.CharField(max_length=500, help_text="Path to the model file")
    file_size_bytes = models.BigIntegerField(default=0)
    checksum = models.CharField(max_length=64, help_text="SHA256 checksum of model file")

    # Training metadata
    training_accuracy = models.FloatField(null=True, blank=True, help_text="Training accuracy (0-1)")
    validation_accuracy = models.FloatField(null=True, blank=True, help_text="Validation accuracy (0-1)")
    training_samples = models.IntegerField(default=0, help_text="Number of training samples used")
    training_time_seconds = models.IntegerField(default=0, help_text="Training time in seconds")

    # Deployment metadata
    deployed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    # Performance tracking
    total_predictions = models.BigIntegerField(default=0)
    average_prediction_time_ms = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['model_type', 'version']
        indexes = [
            models.Index(fields=['model_type', 'status']),
            models.Index(fields=['version']),
        ]

    def __str__(self):
        return f"{self.name} v{self.version} ({self.status})"


class TrainingData(models.Model):
    """Model to store aggregated training data from user feedback."""

    query = models.ForeignKey(Query, on_delete=models.CASCADE, related_name='training_samples')

    # Ground truth data
    user_grade_avg = models.FloatField(help_text="Average grade from user feedback (1-5 scale)")
    user_grade_count = models.IntegerField(default=0, help_text="Number of user ratings")
    user_grade_stddev = models.FloatField(default=0.0, help_text="Standard deviation of user ratings")

    # System predictions for comparison
    system_grade = models.CharField(max_length=1, help_text="Original system grade (A-F)")
    system_score = models.FloatField(help_text="Original system score (0-100)")

    # Aggregated feedback metrics
    accuracy_rating_avg = models.FloatField(null=True, blank=True)
    usefulness_rating_avg = models.FloatField(null=True, blank=True)
    clarity_rating_avg = models.FloatField(null=True, blank=True)

    # Context features
    database_type = models.CharField(max_length=50, blank=True)
    query_complexity = models.IntegerField(default=0)
    table_count = models.IntegerField(default=0)
    join_count = models.IntegerField(default=0)

    # Training metadata
    is_validated = models.BooleanField(default=False, help_text="Whether this sample has been validated")
    validation_source = models.CharField(max_length=100, blank=True, help_text="Source of validation (manual, benchmark, etc.)")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['query', 'is_validated']),
            models.Index(fields=['database_type']),
            models.Index(fields=['user_grade_count']),
        ]

    def __str__(self):
        return f"Training data for Query {self.query.id} - Avg Grade: {self.user_grade_avg:.1f}"


class LearningMetrics(models.Model):
    """Model to track machine learning model performance over time."""

    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='metrics')

    # Performance metrics
    accuracy = models.FloatField(help_text="Model accuracy (0-1)")
    precision = models.FloatField(help_text="Model precision (0-1)")
    recall = models.FloatField(help_text="Model recall (0-1)")
    f1_score = models.FloatField(help_text="F1 score (0-1)")

    # User satisfaction metrics
    user_agreement_rate = models.FloatField(help_text="Rate of user agreement with predictions (0-1)")
    avg_user_rating = models.FloatField(help_text="Average user rating for model predictions (1-5)")

    # Operational metrics
    prediction_count = models.IntegerField(default=0, help_text="Number of predictions made")
    avg_prediction_time_ms = models.FloatField(help_text="Average prediction time in milliseconds")
    error_rate = models.FloatField(default=0.0, help_text="Error rate (0-1)")

    # Time period
    measurement_period_start = models.DateTimeField()
    measurement_period_end = models.DateTimeField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['model', '-created_at']),
            models.Index(fields=['measurement_period_start']),
        ]

    def __str__(self):
        return f"Metrics for {self.model.name} - Accuracy: {self.accuracy:.3f}"


class FeedbackLearning(models.Model):
    """Model to track learning from individual feedback instances."""

    user_history = models.OneToOneField(UserQueryHistory, on_delete=models.CASCADE, related_name='learning_data')

    # Original prediction
    original_grade = models.CharField(max_length=1)
    original_score = models.FloatField()
    original_confidence = models.FloatField(default=0.5, help_text="Model confidence (0-1)")

    # User feedback impact
    feedback_grade_equivalent = models.FloatField(help_text="User feedback converted to grade scale (0-100)")
    grade_difference = models.FloatField(help_text="Difference between user and system grade")
    feedback_weight = models.FloatField(default=1.0, help_text="Weight of this feedback for learning")

    # Learning application
    was_used_for_training = models.BooleanField(default=False)
    training_session = models.CharField(max_length=50, blank=True, help_text="ID of training session that used this feedback")

    # Meta-learning
    user_reliability_score = models.FloatField(default=0.5, help_text="Reliability score of the user providing feedback")
    context_similarity_score = models.FloatField(default=0.0, help_text="Similarity to other queries in training set")

    created_at = models.DateTimeField(default=timezone.now)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['was_used_for_training']),
            models.Index(fields=['feedback_weight']),
        ]

    def __str__(self):
        return f"Learning data for {self.user_history.user.username} - Grade diff: {self.grade_difference:.1f}"
