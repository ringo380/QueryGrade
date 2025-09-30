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
  - `analyzer/query_analyzer.py` - Core query grading engine (rule-based)
  - `analyzer/views.py` - Web interface with authentication & query submission
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
   - `QueryGrader.analyze_query()` applies rule-based analysis
   - Checks for 10+ performance issues (SELECT *, function on columns, etc.)
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

### Test Coverage
- **Unit Tests**: Individual components (QueryGrader, FeatureExtractor, etc.)
- **Integration Tests**: End-to-end workflows (query submission → grading → feedback)
- **ML Tests**: Model training, prediction, feedback processing
- **API Tests**: REST endpoint validation
- **Database Tests**: Model relationships, constraints, queries

### Key Test Files
- `analyzer/test_query_grader.py` - Query analysis logic
- `analyzer/test_integration.py` - Full workflow tests
- `analyzer/test_api.py` - API endpoint tests
- `analyzer/ml/tests/test_hybrid_grader.py` - ML grading tests
- `analyzer/ml/tests/test_feature_extractor.py` - Feature engineering
- `analyzer/ml/tests/test_feedback_collector.py` - Feedback processing

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

### Adding New Analysis Rules
1. Edit `analyzer/query_analyzer.py`
2. Add rule method to `QueryGrader` class
3. Update `_calculate_score()` to incorporate new rule
4. Add test case in `analyzer/test_query_grader.py`
5. Run tests: `python manage.py test analyzer.test_query_grader`