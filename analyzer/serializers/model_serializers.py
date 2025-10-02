"""
Model-based serializers for Django ORM models.

This module contains serializers for converting QueryGrade database models
to/from JSON representations for API responses.
"""
from rest_framework import serializers
from django.contrib.auth.models import User
from ..models import Query, QueryAnalysis, UserQueryHistory, QueryFeedback


class QuerySerializer(serializers.ModelSerializer):
    """Serializer for Query model."""

    class Meta:
        model = Query
        fields = [
            'id', 'sql_text', 'query_type', 'query_hash', 'created_at', 'updated_at',
            'estimated_complexity', 'table_count', 'join_count', 'where_conditions', 'subquery_count'
        ]
        read_only_fields = ['id', 'query_hash', 'created_at', 'updated_at', 'estimated_complexity',
                           'table_count', 'join_count', 'where_conditions', 'subquery_count']


class QueryAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for QueryAnalysis model."""

    query = QuerySerializer(read_only=True)

    class Meta:
        model = QueryAnalysis
        fields = [
            'id', 'query', 'grade', 'score', 'issues_found', 'recommendations',
            'performance_notes', 'analysis_version', 'execution_time_ms', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model."""

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'date_joined']
        read_only_fields = ['id', 'date_joined']


class UserQueryHistorySerializer(serializers.ModelSerializer):
    """Serializer for UserQueryHistory model."""

    user = UserSerializer(read_only=True)
    query = QuerySerializer(read_only=True)

    class Meta:
        model = UserQueryHistory
        fields = [
            'id', 'user', 'query', 'submitted_at', 'ip_address', 'user_agent',
            'database_type', 'database_version', 'use_case_notes', 'was_helpful', 'feedback_comments'
        ]
        read_only_fields = ['id', 'user', 'query', 'submitted_at', 'ip_address', 'user_agent']


class QueryFeedbackSerializer(serializers.ModelSerializer):
    """Serializer for QueryFeedback model."""

    user_history = UserQueryHistorySerializer(read_only=True)

    class Meta:
        model = QueryFeedback
        fields = [
            'id', 'user_history', 'accuracy_rating', 'usefulness_rating', 'clarity_rating',
            'suggestions', 'would_recommend', 'created_at'
        ]
        read_only_fields = ['id', 'user_history', 'created_at']


class QueryHistoryListSerializer(serializers.ModelSerializer):
    """Simplified serializer for query history lists."""

    grade = serializers.CharField(source='query.analysis.grade', read_only=True)
    score = serializers.FloatField(source='query.analysis.score', read_only=True)
    query_type = serializers.CharField(source='query.query_type', read_only=True)
    query_preview = serializers.SerializerMethodField()

    class Meta:
        model = UserQueryHistory
        fields = [
            'id', 'query_type', 'grade', 'score', 'query_preview',
            'database_type', 'database_version', 'submitted_at', 'was_helpful'
        ]
        read_only_fields = ['id', 'submitted_at']

    def get_query_preview(self, obj):
        """Get a truncated preview of the SQL query."""
        sql_text = obj.query.sql_text
        if len(sql_text) > 100:
            return sql_text[:100] + "..."
        return sql_text
