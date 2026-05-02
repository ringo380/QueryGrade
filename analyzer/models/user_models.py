"""
User interaction models for query history and feedback.

This module contains models for tracking user query submissions
and collecting user feedback on analysis quality.
"""

from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

from .query_models import Query


class UserQueryHistory(models.Model):
    """Model to track user query submissions and maintain history."""

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="query_history"
    )
    query = models.ForeignKey(Query, on_delete=models.CASCADE)

    # User context
    submitted_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=255, blank=True)

    # Query context
    database_type = models.CharField(
        max_length=50, blank=True, help_text="e.g., MySQL, PostgreSQL"
    )
    database_version = models.CharField(max_length=50, blank=True)
    use_case_notes = models.TextField(
        blank=True, help_text="User's notes about the query purpose"
    )

    # Feedback tracking
    was_helpful = models.BooleanField(
        null=True, blank=True, help_text="User feedback on analysis quality"
    )
    feedback_comments = models.TextField(blank=True)

    class Meta:
        ordering = ["-submitted_at"]
        indexes = [
            models.Index(fields=["user", "-submitted_at"]),
            models.Index(fields=["submitted_at"]),
        ]
        verbose_name_plural = "User query histories"

    def __str__(self):
        return f"{self.user.username} - {self.query.query_type} query at {self.submitted_at}"


class QueryFeedback(models.Model):
    """Model to store user feedback on analysis quality."""

    RATING_CHOICES = [
        (1, "1 - Very Poor"),
        (2, "2 - Poor"),
        (3, "3 - Average"),
        (4, "4 - Good"),
        (5, "5 - Excellent"),
    ]

    user_history = models.OneToOneField(
        UserQueryHistory, on_delete=models.CASCADE, related_name="detailed_feedback"
    )

    # Detailed feedback
    accuracy_rating = models.IntegerField(choices=RATING_CHOICES, null=True, blank=True)
    usefulness_rating = models.IntegerField(
        choices=RATING_CHOICES, null=True, blank=True
    )
    clarity_rating = models.IntegerField(choices=RATING_CHOICES, null=True, blank=True)

    # Improvement suggestions
    suggestions = models.TextField(
        blank=True, help_text="User suggestions for improvement"
    )
    would_recommend = models.BooleanField(null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Feedback for {self.user_history.user.username} - Query {self.user_history.query.id}"
