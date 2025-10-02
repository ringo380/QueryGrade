"""
API request validation serializers.

This module contains serializers for validating and cleaning incoming
API request data for query grading and batch analysis operations.
"""
from rest_framework import serializers


class QueryGradeRequestSerializer(serializers.Serializer):
    """Serializer for query grading API requests."""

    sql_text = serializers.CharField(
        max_length=10000,
        help_text="The SQL query to analyze",
        error_messages={
            'blank': 'SQL query text is required.',
            'max_length': 'SQL query is too long. Maximum 10,000 characters allowed.'
        }
    )
    database_type = serializers.CharField(
        max_length=50,
        required=False,
        default='',
        help_text="Database type (mysql, postgresql, sqlite, oracle, sqlserver)"
    )
    database_version = serializers.CharField(
        max_length=50,
        required=False,
        default='',
        help_text="Database version for version-specific recommendations"
    )
    use_case_notes = serializers.CharField(
        max_length=1000,
        required=False,
        default='',
        help_text="Optional notes about the query's intended use case"
    )

    def validate_sql_text(self, value):
        """Validate SQL text is not empty."""
        if not value.strip():
            raise serializers.ValidationError("SQL query cannot be empty.")
        return value.strip()


class BatchQueryRequestSerializer(serializers.Serializer):
    """Serializer for batch query analysis requests."""

    queries = serializers.ListField(
        child=serializers.CharField(max_length=10000),
        min_length=1,
        max_length=20,
        help_text="List of SQL queries to analyze (max 20 queries)"
    )
    database_type = serializers.CharField(
        max_length=50,
        required=False,
        default='',
        help_text="Database type for all queries"
    )
    database_version = serializers.CharField(
        max_length=50,
        required=False,
        default='',
        help_text="Database version for all queries"
    )
    analysis_notes = serializers.CharField(
        max_length=1000,
        required=False,
        default='',
        help_text="Optional notes about the batch analysis context"
    )

    def validate_queries(self, value):
        """Validate each query in the batch."""
        validated_queries = []
        for i, query in enumerate(value, 1):
            query = query.strip()
            if not query:
                raise serializers.ValidationError(f"Query {i} cannot be empty.")
            validated_queries.append(query)
        return validated_queries
