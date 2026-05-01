# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QueryGrade is a comprehensive Django-based SQL query analysis and database optimization platform powered by machine learning. The application serves three primary purposes:

1. **SQL Query Grading** ✅: Users paste individual SQL queries to receive performance grades (A-F) with specific feedback and improvement recommendations
2. **System Query Analysis** 🚧: Analyze queries from database logs to identify optimization opportunities within database ecosystem context
3. **Database Architecture Optimization** 🚧: Analyze database architecture and recommend structural improvements

**Current Implementation Status**: Phase 1 (SQL Query Grading) is **complete** with a sophisticated hybrid ML system that learns from user feedback. The platform includes rule-based analysis, ML-powered predictions, comprehensive feedback collection, and automated model training.

## Core Architecture

### Django Project Structure
- **Main project**: `querygrade/` - Django configuration, Celery setup
- **Main app**: `analyzer/` - Core query analysis functionality
- **Key modules**:
  - `analyzer/query_analyzer.py` - Core query grading engine facade (delegates to modular analyzers)
  - `analyzer/analyzers/` - **NEW: Modular analyzer architecture (9 specialized analyzers)**:
    - `base.py` - Base analyzer class and orchestrator
    - `select_analyzer.py` - SELECT clause efficiency
    - `join_analyzer.py` - JOIN analysis
    - `where_analyzer.py` - WHERE clause optimization
    - `indexing_analyzer.py` - **NEW: Index detection and optimization**
    - `subquery_analyzer.py` - **NEW: Subquery pattern optimization**
    - `orderby_analyzer.py` - **NEW: Sorting efficiency**
    - `groupby_analyzer.py` - **NEW: Aggregation optimization**
    - `database/` - Database-specific analyzers (MySQL, PostgreSQL, SQLite, Oracle, SQL Server)
  - `analyzer/views/` - Modular views package:
    - `auth_views.py` - Authentication flows
    - `query_grading_views.py` - Query analysis
    - `history_views.py` - User history
    - `feedback_views.py` - Feedback collection
    - `upload_views.py` - File processing
  - `analyzer/models.py` - Comprehensive models for queries, analysis, feedback, ML tracking
  - `analyzer/forms.py` - Query input, feedback collection, database connection forms
  - `analyzer/ml/` - Machine learning subsystem:
    - `hybrid_grader.py` - Hybrid rule-based + ML grading system
    - `feature_extractor.py` - Extracts 41+ features from SQL queries
    - `feedback_collector.py` - Processes user feedback for ML training
    - `training_pipeline.py` - Automated model training and deployment
    - `documentation_loader.py` - Imports best practices from SQL documentation
  - `analyzer/tasks.py` - Celery async tasks for heavy processing
  - `analyzer/api_views.py` - REST API endpoints
  - `analyzer/management/commands/` - CLI commands for ML operations

### Query Grading Data Flow (Primary Feature)

1. **Query Submission**: User pastes SQL query via web form (`QueryGradeForm`)
2. **Analysis Phase**:
   - Query normalized and hashed for caching
   - `QueryGrader.analyze_query()` orchestrates 9 specialized analyzers
   - Each analyzer examines specific aspects (SELECT, JOIN, WHERE, ORDER BY, GROUP BY, indexes, subqueries)
   - Detects 18+ issue types across performance, efficiency, and best practices
   - Generates 27+ recommendation types with actionable examples
   - Calculates base score (0-100) and letter grade (A-F)
3. **ML Enhancement** (if enabled):
   - `FeatureExtractor` extracts 41+ numerical features
   - `HybridQueryGrader` combines rule-based score with ML prediction
   - Confidence-based weighting adjusts between rules and ML
4. **Results Display**:
   - Grade, score, issues list, and recommendations shown
   - Quick feedback UI (thumbs up/down) for learning
   - Query saved to `UserQueryHistory` for user
5. **Feedback Loop**:
   - User feedback collected via `QueryFeedback` model
   - `FeedbackCollector` processes feedback into training data
   - Periodic retraining improves ML predictions

### Analyzer Architecture (NEW - Phase 1 Complete)

**9 Specialized Analyzers** using Strategy Pattern:

**Core Clause Analyzers**:
1. **SelectAnalyzer** - SELECT clause efficiency (SELECT *, DISTINCT, COUNT(*), scalar subqueries)
2. **JoinAnalyzer** - JOIN analysis (types, conditions, cross joins, implicit joins)
3. **WhereAnalyzer** - WHERE clause optimization (functions on columns, OR conditions, type mismatches)
4. **OrderByAnalyzer** - Sorting efficiency (ORDER BY without LIMIT, functions, expressions, RAND())
5. **GroupByAnalyzer** - Aggregation optimization (GROUP BY indexes, HAVING vs WHERE, DISTINCT in aggregates)

**Performance Analyzers**:
6. **IndexingAnalyzer** - Index detection (missing indexes, LIKE wildcards, composite indexes, covering indexes)
7. **SubqueryAnalyzer** - Subquery patterns (correlated subqueries, scalar subqueries, IN vs EXISTS, CTEs)

**Database-Specific Analyzers**:
8. **MySQLAnalyzer** - MySQL patterns (storage engines, full-text search, query hints)
9. **PostgreSQLAnalyzer** - PostgreSQL features (window functions, DISTINCT ON, advanced indexes)

**Detection Capabilities** (18+ issue types):
- High Severity: LIKE leading wildcard, function on indexed columns, scalar subqueries, NOT IN, ORDER BY RAND()
- Medium Severity: SELECT *, functions in WHERE/ORDER BY/GROUP BY, correlated subqueries, HAVING without aggregates
- Low Severity: Many GROUP BY columns, DISTINCT with GROUP BY, ORDER BY in subquery

**Recommendation Types** (27+ types):
- Index-related (7): WHERE, range, JOIN, ORDER BY, GROUP BY, composite, covering
- Query rewriting (6): JOIN instead of subquery, NOT EXISTS, CTEs, eliminate scalar subqueries
- Function optimization (4): Avoid functions on columns, computed columns for ORDER BY/GROUP BY
- Filter optimization (2): Move HAVING to WHERE, column order by selectivity
- Other optimizations (8+): ADD LIMIT, remove DISTINCT, COUNT optimization, etc.

### Machine Learning System

**Hybrid Grading Architecture**:
- Starts with rule-based grading (QueryGrader)
- ML model trained on user feedback + documentation benchmarks
- Dynamic weighting: high confidence → more ML, low confidence → more rules
- Models stored in `MLModel` table with versioning and performance tracking

**Feature Engineering** (41+ features):
- Query structure: table count, join count, where conditions, subqueries
- Complexity metrics: nesting depth, aggregation usage, distinct operations
- Performance indicators: SELECT *, function on columns, wildcards in LIKE
- Database-specific patterns: storage engines, index hints, lock types

**Training Pipeline**:
- Automated via `train_ml_model` management command
- Algorithms: Random Forest, Gradient Boosting, XGBoost, LightGBM
- Validation: Cross-validation, train/test split, performance benchmarking
- Deployment: Best model auto-deployed to production

**Feedback Collection**:
- Quick feedback: thumbs up/down on analysis accuracy
- Detailed feedback: ratings for accuracy, usefulness, clarity
- User reliability scoring: consistent users weighted higher
- Feedback converted to training samples via `FeedbackLearning` model

## Development Commands

### Local Development Setup
```bash
# Install dependencies (includes ML libraries: tensorflow, torch, transformers, xgboost, lightgbm)
pip install -r requirements.txt

# Database migrations (sets up Query, QueryAnalysis, MLModel, TrainingData, etc.)
python manage.py migrate

# Create superuser for admin access
python manage.py createsuperuser

# Collect static files for production
python manage.py collectstatic --noinput

# Run development server
python manage.py runserver

# Start Redis (required for Celery and caching)
redis-server

# Start Celery worker (for async tasks)
celery -A querygrade worker -l info

# Start Celery beat (for scheduled tasks)
celery -A querygrade beat -l info
```

### ML System Commands
```bash
# Train ML model with user feedback
python manage.py train_ml_model --algorithm random_forest --min-samples 50

# Train with specific algorithm and validation
python manage.py train_ml_model --algorithm xgboost --validation-split 0.2 --cross-validation 5

# Process user feedback into training data
python manage.py process_ml_feedback --min-reliability 0.6

# Load best practices from SQL documentation
python manage.py load_documentation --source mysql --url https://dev.mysql.com/doc/

# View ML system analytics
python manage.py ml_analytics --metrics accuracy,f1_score,user_satisfaction

# Manage ML models (list, activate, deactivate, cleanup)
python manage.py manage_ml_models list
python manage.py manage_ml_models activate <model_id>
python manage.py manage_ml_models cleanup --keep 5
```

### Testing
```bash
# Run all tests (includes unit, integration, ML tests)
python manage.py test

# Run specific test suites
python manage.py test analyzer                           # All analyzer tests
python manage.py test analyzer.tests.TestQueryGrader     # Query grading tests
python manage.py test analyzer.ml.tests                  # ML system tests
python manage.py test analyzer.test_api                  # API tests

# Run specific test files
python manage.py test analyzer.test_query_grader         # Query grading
python manage.py test analyzer.test_integration          # Integration tests
python manage.py test analyzer.ml.tests.test_hybrid_grader  # Hybrid ML tests
```

### Docker Development
```bash
# Build and run with Docker Compose (includes PostgreSQL and Redis)
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Configuration

### Environment Variables
```bash
# Database (Production)
DB_ENGINE=django.db.backends.postgresql
DB_NAME=querygrade
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# Redis/Celery
REDIS_URL=redis://localhost:6379/1
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# ML System
ML_ENABLED=True                    # Enable ML features
ML_HYBRID_GRADING=True            # Use hybrid rule+ML grading
ML_AUTO_RETRAIN=True              # Automatic model retraining
ML_FEEDBACK_COLLECTION=True       # Collect user feedback
ML_MIN_TRAINING_SAMPLES=50        # Minimum samples before training
ML_RETRAIN_THRESHOLD_DAYS=7       # Days between retraining checks
ML_PERFORMANCE_THRESHOLD=0.7      # Minimum model accuracy

# Security
DEBUG=False                        # NEVER True in production
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

### Redis Cache Configuration
The system uses **4 separate Redis databases** for different caching purposes:
- **DB 1**: Default cache (query results, general caching)
- **DB 2**: Process cache (ML results, feature extractions)
- **DB 3**: Query analysis cache (analysis results, 2-hour TTL)
- **DB 4**: Template cache (rendered templates, 30-minute TTL)

### ML System Settings (settings.py)
```python
ML_ENABLED = True                      # Global ML toggle
ML_MODEL_PATH = 'analyzer/ml/models'   # Model storage location
ML_MIN_TRAINING_SAMPLES = 50           # Min samples for training
ML_RETRAIN_THRESHOLD_DAYS = 7          # Retraining frequency
ML_PERFORMANCE_THRESHOLD = 0.7         # Min accuracy threshold
```

## Deployment (Railway)

- Live at `querygrade.com` / `querygrade.net`. Auto-deploy on push to `main` via the GitHub-connected Railway service.
- `railway.toml`: `preDeployCommand` runs in a separate container, so `collectstatic` MUST be in `startCommand` — its filesystem doesn't carry over to the runtime container.
- Wrap startCommand in `sh -c '...'` — Railway exec's directly, so `$PORT` won't expand otherwise.
- `set -e` fails in startCommand (`set: executable not found`). Use `&&` chaining instead.
- Postgres `default_transaction_isolation=read_committed` needs no override; the server default is already correct.
- `BASE_DIR/logs/` must be `mkdir -p`'d at settings import time — Railway containers have no `logs/` dir, RotatingFileHandler crashes Django setup otherwise.
- Manual deploy: `railway up --service querygrade --detach` (uploads local working tree, bypasses GitHub).
- Set env var without triggering deploy: `railway variables --service querygrade --set "KEY=VAL" --skip-deploys`.
- `railway add -d postgres` needs a TTY: `script -q /dev/null railway add -d postgres`.
- Slim/worker split: `requirements-prod.txt` (web, no tensorflow/torch/transformers) + `requirements-worker.txt` (full ML). ML imports gated to `analyzer/ml/ensemble/multi_model.py`.
- Check live deploy status: `railway status --json | python3 -c "import json,sys; d=json.load(sys.stdin); [print(e['node']['serviceName'], '->', e['node']['latestDeployment']['status']) for e in d['environments']['edges'][0]['node']['serviceInstances']['edges']]"`
- PR merge requires `gh pr merge <N> --admin --squash --delete-branch` — required checks (Test Suite, Security Scan, Trivy) stay PENDING due to a GitHub Actions billing block on this repo.

## Key Dependencies

### Core Framework
- Django 4.0-5.0 - Web framework
- djangorestframework 3.14+ - REST API
- djangorestframework-simplejwt 5.2+ - JWT authentication
- celery 5.2+ - Async task processing
- redis 4.0+ - Caching and message broker
- gunicorn 20.1 - WSGI server
- psycopg2-binary 2.9+ - PostgreSQL adapter

### Machine Learning Stack
- **sklearn**: Random Forest, Gradient Boosting, feature scaling
- **xgboost** 1.7+: Gradient boosting optimization
- **lightgbm** 3.3+: Efficient gradient boosting
- **tensorflow** 2.10+: Deep learning (future neural network models)
- **torch** 1.13+: PyTorch (flexible model architectures)
- **transformers** 4.25+: NLP for query semantic analysis
- **sentence-transformers** 2.2+: Query embedding generation
- **joblib** 1.2+: Model serialization and caching

### Data Processing
- pandas 2.2+ - Data manipulation
- numpy 1.26+ - Numerical operations
- sqlparse 0.4+ - SQL parsing and analysis

### Documentation & Web
- beautifulsoup4 4.12+ - HTML parsing for documentation loading
- requests 2.31+ - HTTP requests for external docs

## API Endpoints

### REST API (requires authentication)
```bash
POST /api/analyze/                    # Analyze single query
POST /api/batch-analyze/              # Batch query analysis
GET  /api/history/                    # User query history
POST /api/feedback/<analysis_id>/     # Submit feedback
GET  /api/models/                     # List ML models
GET  /api/analytics/                  # ML performance metrics
```

### Web Views
```bash
GET  /                                # Home (log upload)
GET  /grade/                          # Query grading interface
POST /grade/                          # Submit query for grading
GET  /grade/results/<id>/             # View analysis results
POST /feedback/<id>/                  # Submit detailed feedback
POST /feedback/quick/<id>/            # Quick thumbs up/down
GET  /history/                        # User query history
GET  /ml/dashboard/                   # ML performance dashboard (admin)
GET  /login/                          # User login
GET  /register/                       # User registration
```

## Testing Strategy

> **📚 For comprehensive testing guidance**, see:
> - **[TESTING.md](TESTING.md)** - Complete testing guide with examples, best practices, and troubleshooting
> - **[INTEGRATION_TEST_FIX_SUMMARY.md](INTEGRATION_TEST_FIX_SUMMARY.md)** - Case study of solving cache initialization issues in tests

### Test Coverage
- **Unit Tests**: Individual components (QueryGrader, FeatureExtractor, etc.)
- **Integration Tests**: End-to-end workflows (query submission → grading → feedback)
- **ML Tests**: Model training, prediction, feedback processing
- **API Tests**: REST endpoint validation
- **Database Tests**: Model relationships, constraints, queries

### Key Test Files
- `analyzer/test_query_grader.py` - Query analysis logic (28 tests)
- `analyzer/test_integration_refactored.py` - Full workflow tests with proper cache handling (5 tests)
- `analyzer/test_integration.py` - Legacy integration tests (deprecated, use refactored version)
- `analyzer/test_api.py` - API endpoint tests
- `analyzer/ml/tests/test_hybrid_grader.py` - ML grading tests
- `analyzer/ml/tests/test_feature_extractor.py` - Feature engineering
- `analyzer/ml/tests/test_feedback_collector.py` - Feedback processing

### Testing Best Practices

#### Cache Management in Tests
**Critical**: The global `query_cache` singleton in `analyzer/performance.py` is instantiated at module import time, *before* test settings apply. This causes tests to use production cache instead of DummyCache.

**Solution**: Always reinitialize cache in test setUp():
```python
def setUp(self):
    from analyzer.performance import query_cache
    from django.core.cache import caches

    # Force query_cache to use test cache backend
    query_cache.cache = caches['query_analysis_cache']

    # Clear all caches
    for cache_name in ['default', 'query_analysis_cache', 'process_cache', 'template_cache']:
        try:
            caches[cache_name].clear()
        except:
            pass
```

#### TransactionTestCase vs TestCase
When `ATOMIC_REQUESTS=True` (production setting), use `TransactionTestCase` for integration tests:
- `TestCase` wraps each test in a transaction, conflicting with `ATOMIC_REQUESTS`
- `TransactionTestCase` allows proper transaction control
- Requires manual cleanup in `tearDown()` (no automatic rollback)
- Use factory methods for consistent test object creation

**Example**:
```python
from django.test import TransactionTestCase

class IntegrationTestCase(TransactionTestCase):
    def tearDown(self):
        # Manual cleanup required
        UserQueryHistory.objects.all().delete()
        QueryAnalysis.objects.all().delete()
        Query.objects.all().delete()
        User.objects.all().delete()
```

#### Fetching Test Objects
Always fetch objects by explicit ID to avoid cache interference:
```python
# ❌ BAD - may return cached objects
simple_query_obj = Query.objects.first()
complex_query_obj = Query.objects.last()

# ✅ GOOD - fetch by ID from response
analysis_id = int(response.url.split('/')[-2])
analysis = QueryAnalysis.objects.get(id=analysis_id)
query_obj = analysis.query
```

#### Test Configuration
Required `@override_settings` for integration tests:
```python
@override_settings(
    RATELIMIT_ENABLE=False,  # Disable rate limiting
    CACHES={
        'default': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
        'query_analysis_cache': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
        'process_cache': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
        'template_cache': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'}
    }
)
```

## Performance Considerations

### Caching Strategy
- Query analysis results cached by query hash (2 hours)
- ML predictions cached with feature fingerprints
- Template fragments cached (30 minutes)
- Redis used for all cache backends with compression

### Async Processing
- Heavy operations (log processing, model training) via Celery
- Task queues: `heavy_processing`, `light_processing`, `maintenance`
- Worker memory limits: 200MB per child, 1000 tasks per child

### Database Optimization
- Connection pooling enabled (CONN_MAX_AGE=600)
- Indexed fields: query_hash, user+timestamp, grades, scores
- Database query optimization via select_related() and prefetch_related()
- SQLite WAL mode for development

## Security Features

### XSS Protection
- Content Security Policy (CSP) configured
- Script/style sources restricted to self + CDN
- Django template auto-escaping enabled
- Custom security middleware for enhanced protection

### CSRF Protection
- CSRF tokens required for all POST requests
- Cookie-based CSRF with 1-hour expiration
- Custom CSRF failure handling

### Authentication
- User authentication required for query submission
- JWT tokens for API access
- Session cookies with HTTPOnly, Secure, SameSite flags
- Password validation: 12+ chars, not common, not similar to username

### Rate Limiting
- IP-based rate limiting: 5 requests/minute for unauthenticated
- User-based rate limiting: 10 requests/minute for authenticated
- API throttling: 100/hour anonymous, 1000/hour authenticated

## Development Workflow Tips

### Working with ML Models
1. Collect feedback: Users submit queries and provide ratings
2. Process feedback: `python manage.py process_ml_feedback`
3. Train model: `python manage.py train_ml_model --algorithm xgboost`
4. Monitor performance: Visit `/ml/dashboard/` or run `ml_analytics` command
5. Model auto-deployed if accuracy > threshold

### Debugging Query Analysis
- Set `DEBUG=True` in settings.py
- Check logs in `logs/security.log`
- Use Django shell: `python manage.py shell`
  ```python
  from analyzer.query_analyzer import QueryGrader
  grader = QueryGrader()
  query, analysis = grader.analyze_query("SELECT * FROM users")
  print(analysis.grade, analysis.score, analysis.issues_found)
  ```

**For debugging test failures**, see:
- [TESTING.md - Troubleshooting Checklist](TESTING.md#troubleshooting-checklist)
- [INTEGRATION_TEST_FIX_SUMMARY.md - Common Issues](INTEGRATION_TEST_FIX_SUMMARY.md#common-issues-and-solutions)

### Adding New Analysis Rules
The analyzer uses a modular architecture with specialized analyzer classes. To add new rules:

**Option 1: Add to existing analyzer**
1. Identify relevant analyzer in `analyzer/analyzers/` (e.g., `where_analyzer.py`, `join_analyzer.py`)
2. Add detection logic to the analyzer's `analyze()` method
3. Append issues/recommendations to `context.issues` or `context.recommendations`
4. Update scoring in `analyzer/analyzers/base.py` `_calculate_score()` if needed
5. Add test case to `analyzer/test_query_grader.py`
6. Run tests: `python manage.py test analyzer.test_query_grader`

**Option 2: Create new analyzer**
1. Create new file in `analyzer/analyzers/` (e.g., `security_analyzer.py`)
2. Inherit from `BaseAnalyzer` and implement `analyze()` and `name` property
3. Register in `analyzer/analyzers/base.py` `_initialize_analyzers()`
4. Add comprehensive test cases
5. Run full test suite: `python manage.py test analyzer`

**Example new analyzer**:
```python
from .base import BaseAnalyzer, AnalysisContext

class SecurityAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "SecurityAnalyzer"

    def analyze(self, context: AnalysisContext) -> None:
        sql_upper = context.sql_text.upper()

        # Detect SQL injection patterns
        if re.search(r";\s*DROP\s+TABLE", sql_upper):
            context.issues.append({
                'type': 'sql_injection',
                'severity': 'critical',
                'message': 'Potential SQL injection detected'
            })
```

## UI Patterns (Tailwind, post-#14)

- Django template escaping: use `|escapejs` only inside JS string literals (e.g. `var x = "{{ val|escapejs }}"`). For HTML `data-*` attributes, Django's default auto-escaping is correct — it converts `"` to `&quot;` which browsers decode properly when JS reads `dataset.*`. `|escapejs` on an attribute would produce `\"` and break attribute parsing.
- All templates extend `'analyzer/base.html'` (NOT `'base.html'`). Loads Tailwind CDN + Inter font.
- `getCsrfToken()` and `showToast(msg, type)` are global helpers from `base.html` — use these in new JS rather than rolling your own.
- Form widgets get input classes auto-applied via `base.html` `DOMContentLoaded` hook; no need to add Tailwind classes manually to Django form output.
- Grade pill convention: A=emerald, B=lime, C=amber, D=orange, F=red (`bg-{color}-100 text-{color}-700`). See `account.html` and `grade_results.html` for the canonical pattern.
- URL prefix: `analyzer/urls.py` is mounted at root in `querygrade/urls.py`. Fetch from `/ml/api/...` NOT `/analyzer/ml/api/...`.
- Read-only SQL display: use `<textarea class="sql-display">` converted by CodeMirror (readOnly, material-darker). See `compare_results.html` / `batch_results.html` for the pattern. Never use `<pre><code>` for SQL display.
- Copy functions (`copyFullReport`, `copyAllRecommendations`, `copyRewriteSuggestions`) read from DOM elements — use `el.value || el.textContent` for textareas, `el.textContent` for code/span. When changing element types, audit all copy functions.
- CodeMirror in hidden tabs: always call `refresh()` inside `setTimeout(0)` after unhiding, so the browser repaints before CodeMirror remeasures.

## Known Issues

- `enhanced_grade_results.html` renders `{{ analysis.issues_found|safe }}` into JS — pre-existing XSS risk; don't widen this pattern. (`grade_results.html` was fixed: now uses `{{ analysis.issues_found|json_script:"issues-json" }}` + `JSON.parse(document.getElementById('issues-json').textContent)`.)

## Form Validation

- `analyzer/forms/validators.py:validate_sql_query()` is the single input gate for all query entry points (single grade, compare, batch). Changes here affect all three.
- Analyzer pipeline (`analyzer/analyzers/`) is SELECT-optimized — non-SELECT queries no-op through most analyzers and get a `performance_notes` warning added in `base.py`.

## Analytics (GA4)

- GA4 property `QueryGrade` (`G-YXW6942WVH`) is live in production. Env var `GA4_MEASUREMENT_ID` controls activation — empty → no script tag rendered (local dev default). Set in Railway, not in `.env`.
- Tracked events: `query_graded` (grade, score, analysis_type), `feedback_submitted` (feedback_type, was_helpful), `sign_up` (one-shot via `?signup=1` redirect, checked in `base.html`). Custom dimensions: `grade`, `analysis_type`, `user_type`, `feedback_type`.
- Adding any third-party JS endpoint (analytics, error tracking, external API) requires updating `CSP_SCRIPT_SRC` and/or `CSP_CONNECT_SRC` in `querygrade/settings.py` — `connect-src` defaults to `'self'` only and silently blocks otherwise.
- Global template vars belong in `analyzer/context_processors.py` (registered in `settings.TEMPLATES[0]['OPTIONS']['context_processors']`), not in every view. `ga4_settings` is the existing example.
- One-shot post-redirect client events use the `?<flag>=1` query-param pattern: view sets it on redirect, base.html checks `request.GET.<flag>` to fire `gtag('event', ...)` exactly once.

## Anonymous trial system

- Anonymous visitors may grade up to `ANON_TRIAL_CAP` (default 3, env-configurable) queries per session. Cap/count/remaining are computed by `anon_trial_state(request)` in `analyzer/views/utils.py` — import from there, do not duplicate inline.
- Anonymous results are tracked in `session[ANON_ANALYSIS_SESSION_KEY]` (list of `analysis.id` ints). Access check in `grade_results`: `analysis_id not in session[ANON_ANALYSIS_SESSION_KEY]` — IDs are stored and compared as integers.
- All `grade_form.html` render paths must include `trial_exhausted` in context (the template gates the entire `<form>` on it). Missing it renders the form even for exhausted anon users.
- `grade_query()` has **4 render paths**: exhausted-trial early return, ValueError catch, generic Exception catch, and the bottom GET/failed-validation render. Any new context var (e.g. `db_versions_json`) must be added to all paths the form actually renders in.
- JS that references `#gradeForm` must guard with `const f = document.getElementById('gradeForm'); if (f) { ... }` — the element is absent when `trial_exhausted` is true.
- Rate-limit stacking: use a callable key (`_anon_ip_key`) that returns `None` for authenticated users so the IP-keyed anon limit doesn't accidentally cap auth users on shared/NAT IPs. `django-ratelimit` skips the check when the key function returns `None`.

## View context conventions (gotchas)

- `grade_results`, `compare_results`, `batch_results`: result objects expose `result.analysis.grade` / `.issues_found` / `.recommendations` (NOT `result.grade` / `.issues` / `.suggestions`).
- `QueryAnalysis` uses `OneToOneField(Query, related_name='analysis')` — access via `query.analysis.grade`, NOT `query.queryanalysis_set.last`. There is no `queryanalysis_set`.
- `UserQueryHistory` timestamp field is `submitted_at` (NOT `created_at`). Order by `-submitted_at`.
- `compare_results` / `batch_results`: `result.query_text` (string) and `result.query` (Query model instance) — not interchangeable.
- `feedback_analytics`: stats wrapped in a `statistics` dict (`statistics.total_feedback`, `statistics.avg_accuracy`, etc.).
- `ml_dashboard`: requires `is_staff_or_superuser`; API endpoints under `/ml/api/*` are decorated with `@cache_page(60 * 5)` — bust by deploying or stripping cache.
