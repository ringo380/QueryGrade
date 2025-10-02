# QueryGrade Service Layer

This package contains the business logic layer for QueryGrade, following the **Service Layer Pattern**. Services act as a facade between views and domain logic, handling complex workflows and coordinating multiple operations.

## Architecture Overview

```
┌─────────────────┐
│     Views       │  ← Thin controllers (HTTP handling, validation)
│  (Presentation) │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│    Services     │  ← Business logic layer (orchestration, rules)
│  (This Package) │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Models/DB      │  ← Data layer (persistence, queries)
│   (Database)    │
└─────────────────┘
```

## Services

### 1. QueryAnalysisService

**Purpose**: Handles all SQL query analysis operations

**Responsibilities**:
- Orchestrate traditional rule-based query analysis
- Integrate ML-enhanced analysis
- Create user history records
- Handle SQL syntax errors gracefully
- Format error messages for user display

**Usage**:
```python
from analyzer.services import QueryAnalysisService
from analyzer.services.query_analysis_service import QueryAnalysisRequest

service = QueryAnalysisService()

request = QueryAnalysisRequest(
    sql_query="SELECT * FROM users WHERE id = 1",
    user=request.user,
    database_type='mysql',
    ip_address='127.0.0.1',
    user_agent='Mozilla/5.0',
    enable_ml=True
)

result = service.analyze_query_for_user(request)

# Access results
query = result.query
analysis = result.analysis
user_history = result.user_history
ml_analysis = result.ml_analysis  # Optional
```

**Key Methods**:
- `analyze_query_for_user(request)` - Main entry point for query analysis
- `get_analysis_by_id(analysis_id)` - Retrieve existing analysis

---

### 2. FeedbackService

**Purpose**: Handles all user feedback collection and processing

**Responsibilities**:
- Create and update detailed feedback
- Handle quick thumbs up/down feedback
- Generate learning records for ML training
- Calculate user reliability scores
- Aggregate feedback statistics

**Usage**:
```python
from analyzer.services import FeedbackService
from analyzer.services.feedback_service import FeedbackSubmission

service = FeedbackService()

# Submit detailed feedback
submission = FeedbackSubmission(
    user=request.user,
    analysis_id=analysis.id,
    accuracy_rating=5,
    usefulness_rating=4,
    clarity_rating=5,
    suggestions='Very helpful analysis!',
    would_recommend=True
)

result = service.submit_detailed_feedback(submission)

if result.success:
    print(f"Feedback {'created' if result.created else 'updated'}")
else:
    print(f"Error: {result.error}")

# Quick feedback
quick_submission = FeedbackSubmission(
    user=request.user,
    analysis_id=analysis.id,
    is_helpful=True
)

quick_result = service.submit_quick_feedback(quick_submission)
```

**Key Methods**:
- `submit_detailed_feedback(submission)` - Submit full feedback form
- `submit_quick_feedback(submission)` - Submit thumbs up/down
- `get_feedback_for_analysis(user, analysis_id)` - Retrieve feedback
- `get_feedback_statistics(user=None)` - Get aggregate stats

---

### 3. DatabaseIntrospectionService

**Purpose**: Handles database schema analysis and introspection

**Responsibilities**:
- Manage database connections
- Analyze schema structure (tables, columns, indexes, foreign keys)
- Generate optimization recommendations
- Identify missing indexes
- Detect foreign key issues

**Usage**:
```python
from analyzer.services import DatabaseIntrospectionService
from analyzer.services.database_introspection_service import DatabaseConnectionConfig

service = DatabaseIntrospectionService()

# Connect to database
config = DatabaseConnectionConfig(
    engine='mysql',
    name='my_database',
    host='localhost',
    port=3306,
    username='user',
    password='pass'
)

success, error = service.connect_to_database(config)

if success:
    # Analyze schema
    result = service.analyze_schema(schema='public')

    print(f"Tables: {len(result.tables)}")
    print(f"Recommendations: {len(result.recommendations)}")
    print(f"Missing Indexes: {len(result.missing_indexes)}")
    print(f"Statistics: {result.statistics}")

    # Always close connection
    service.close_connection()
else:
    print(f"Connection error: {error}")
```

**Key Methods**:
- `connect_to_database(config)` - Establish database connection
- `analyze_schema(schema=None)` - Analyze database schema
- `close_connection()` - Close database connection

---

## Design Principles

### 1. **Single Responsibility**
Each service handles one domain area (query analysis, feedback, database introspection)

### 2. **Dependency Injection**
Services accept data via DTOs (Data Transfer Objects) rather than request objects

### 3. **Return Results, Not Responses**
Services return domain objects/DTOs, views handle HTTP responses

### 4. **Exception Handling**
Services raise domain exceptions (ValueError, etc.), views handle user feedback

### 5. **Testability**
Services are easily unit-testable without HTTP layer

---

## Data Transfer Objects (DTOs)

### Query Analysis
- `QueryAnalysisRequest` - Input for analysis operations
- `QueryAnalysisResult` - Output containing analysis results

### Feedback
- `FeedbackSubmission` - Input for feedback operations
- `FeedbackResult` - Output containing operation results

### Database Introspection
- `DatabaseConnectionConfig` - Database connection parameters
- `SchemaAnalysisResult` - Schema analysis output

---

## Benefits

### Before Service Layer (Logic in Views)
```python
# 50+ lines of business logic mixed with HTTP handling
def grade_query(request):
    if request.method == 'POST':
        form = QueryGradeForm(request.POST)
        if form.is_valid():
            # 30 lines of analysis logic here
            # ML orchestration
            # Error handling
            # User history creation
            # ...
            return redirect('results')
```

### After Service Layer (Thin Views)
```python
# 10 lines, clear separation of concerns
def grade_query(request):
    if request.method == 'POST':
        form = QueryGradeForm(request.POST)
        if form.is_valid():
            # Build request DTO
            service_request = QueryAnalysisRequest(...)

            # Call service
            result = service.analyze_query_for_user(service_request)

            # Handle HTTP response
            return redirect('results', analysis_id=result.analysis.id)
```

### Advantages
1. ✅ **Testable** - Services can be unit tested independently
2. ✅ **Reusable** - Same logic for views, APIs, CLI commands
3. ✅ **Maintainable** - Business logic in one place
4. ✅ **Clear** - Views focus on HTTP, services on business rules
5. ✅ **Type-Safe** - DTOs provide clear contracts

---

## Testing

### Service Tests
Tests are in `analyzer/test_services.py`

Run service tests:
```bash
python manage.py test analyzer.test_services
```

### Example Test
```python
def test_analyze_simple_query(self):
    """Test analyzing a simple SELECT query."""
    request = QueryAnalysisRequest(
        sql_query="SELECT * FROM users",
        user=self.user,
        database_type='mysql',
        enable_ml=False
    )

    result = self.service.analyze_query_for_user(request)

    self.assertIsNotNone(result.query)
    self.assertIsNotNone(result.analysis)
```

---

## Migration Guide

### Converting Views to Use Services

**Before** (Business Logic in View):
```python
def grade_query(request):
    # 50 lines of mixed concerns
    query, analysis = analyze_query(sql_text, db_type)
    ml_result = run_ml_analysis(...)
    user_history = UserQueryHistory.objects.create(...)
    # ...
```

**After** (Using Service):
```python
def grade_query(request):
    service = QueryAnalysisService()

    request_dto = QueryAnalysisRequest(
        sql_query=form.cleaned_data['sql_query'],
        user=request.user,
        database_type=form.cleaned_data['database_type'],
        ip_address=get_client_ip(request),
        user_agent=request.META.get('HTTP_USER_AGENT', '')
    )

    try:
        result = service.analyze_query_for_user(request_dto)
        # Store ML analysis in session if present
        if result.ml_analysis:
            request.session['ml_analysis'] = result.ml_analysis
        return redirect('results', analysis_id=result.analysis.id)
    except ValueError as e:
        messages.error(request, str(e))
        return render(request, 'form.html', {'form': form})
```

---

## Future Enhancements

Potential services to add:
- **BatchProcessingService** - Handle batch query analysis
- **OptimizationService** - Query optimization and rewriting
- **NotificationService** - User notifications and alerts
- **ReportingService** - Generate analytics reports
- **CacheService** - Centralize caching logic

---

**Version**: 1.0.0
**Created**: 2025-10-02
**Status**: Production-Ready ✅
