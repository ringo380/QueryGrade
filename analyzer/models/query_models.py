"""
Core query models for SQL query storage and analysis.

This module contains the fundamental models for storing SQL queries
and their analysis results.
"""

from django.db import models
from django.utils import timezone


class Query(models.Model):
    """Model to store SQL queries submitted for analysis."""

    QUERY_TYPES = [
        ("SELECT", "Select"),
        ("INSERT", "Insert"),
        ("UPDATE", "Update"),
        ("DELETE", "Delete"),
        ("CREATE", "Create"),
        ("ALTER", "Alter"),
        ("DROP", "Drop"),
        ("UNKNOWN", "Unknown"),
    ]

    sql_text = models.TextField(help_text="The SQL query text")
    query_type = models.CharField(max_length=10, choices=QUERY_TYPES, default="UNKNOWN")
    query_hash = models.CharField(
        max_length=64, db_index=True, help_text="MD5 hash of normalized query"
    )
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    # Query complexity metrics
    estimated_complexity = models.IntegerField(
        default=0, help_text="Complexity score (0-100)"
    )
    table_count = models.IntegerField(default=0)
    join_count = models.IntegerField(default=0)
    where_conditions = models.IntegerField(default=0)
    subquery_count = models.IntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["query_hash"]),
            models.Index(fields=["query_type"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"{self.query_type} query ({self.id}): {self.sql_text[:50]}..."


class QueryAnalysis(models.Model):
    """Model to store analysis results and grades for queries."""

    GRADE_CHOICES = [
        ("A", "A - Excellent"),
        ("B", "B - Good"),
        ("C", "C - Average"),
        ("D", "D - Poor"),
        ("F", "F - Failing"),
    ]

    query = models.OneToOneField(
        Query, on_delete=models.CASCADE, related_name="analysis"
    )

    # Grading
    grade = models.CharField(max_length=1, choices=GRADE_CHOICES)
    score = models.FloatField(help_text="Numeric score (0-100)")

    # Analysis results
    issues_found = models.JSONField(default=list, help_text="List of issues identified")
    recommendations = models.JSONField(
        default=list, help_text="List of improvement recommendations"
    )
    performance_notes = models.TextField(
        blank=True, help_text="Performance analysis notes"
    )

    # Analysis metadata
    analysis_version = models.CharField(max_length=10, default="1.0")
    execution_time_ms = models.IntegerField(
        default=0, help_text="Analysis execution time in milliseconds"
    )
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["grade"]),
            models.Index(fields=["score"]),
        ]

    def __str__(self):
        return (
            f"Analysis for Query {self.query.id}: Grade {self.grade} ({self.score:.1f})"
        )
