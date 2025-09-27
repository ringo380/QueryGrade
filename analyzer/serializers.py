from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Query, QueryAnalysis, UserQueryHistory, QueryFeedback


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