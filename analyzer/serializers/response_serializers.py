"""
API response formatting serializers.

This module contains serializers for formatting API responses for
query grading and batch analysis operations. These are not tied to
models but provide structured API output.
"""
from rest_framework import serializers


class QueryGradeResponseSerializer(serializers.Serializer):
    """Serializer for query grading API responses."""

    query_id = serializers.IntegerField(help_text="Unique ID of the analyzed query")
    analysis_id = serializers.IntegerField(help_text="Unique ID of the analysis result")
    grade = serializers.CharField(help_text="Letter grade (A-F)")
    score = serializers.FloatField(help_text="Numeric score (0-100)")
    query_type = serializers.CharField(help_text="Type of SQL query")
    complexity = serializers.IntegerField(help_text="Complexity score (0-100)")
    issues_found = serializers.ListField(
        child=serializers.CharField(),
        help_text="List of issues identified in the query"
    )
    recommendations = serializers.ListField(
        child=serializers.CharField(),
        help_text="List of improvement recommendations"
    )
    performance_notes = serializers.CharField(help_text="Performance analysis notes")
    execution_time_ms = serializers.IntegerField(help_text="Analysis execution time in milliseconds")
    created_at = serializers.DateTimeField(help_text="Timestamp when analysis was created")


class BatchQueryResponseSerializer(serializers.Serializer):
    """Serializer for batch query analysis responses."""

    total_queries = serializers.IntegerField(help_text="Total number of queries analyzed")
    successful_analyses = serializers.IntegerField(help_text="Number of successful analyses")
    failed_analyses = serializers.IntegerField(help_text="Number of failed analyses")
    average_grade = serializers.CharField(help_text="Average grade across all successful analyses")
    average_score = serializers.FloatField(help_text="Average score across all successful analyses")
    results = QueryGradeResponseSerializer(many=True, help_text="Individual analysis results")
    summary = serializers.CharField(help_text="Summary of batch analysis results")
    created_at = serializers.DateTimeField(help_text="Timestamp when batch analysis was created")
