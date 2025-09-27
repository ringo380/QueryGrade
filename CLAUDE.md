# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QueryGrade is a comprehensive Django-based SQL query analysis and database optimization platform. The application serves three primary purposes:

1. **SQL Query Grading**: Allow users to paste individual SQL queries into a text field to receive a performance grade with specific feedback and improvement recommendations
2. **System Query Analysis**: Analyze queries running within users' database systems (via log analysis) to identify optimization opportunities within their current database ecosystem context
3. **Database Architecture Optimization**: Analyze database architecture and recommend structural improvements to enhance efficiency and functionality relative to server software and codebase requirements

**Current Implementation Status**: The application currently implements only basic log file anomaly detection (part of goal #2). The primary query grading interface (#1) and comprehensive database architecture analysis (#3) need to be implemented.

## Core Architecture

### Django Project Structure
- **Main project**: `querygrade/` - Contains Django configuration
- **Main app**: `analyzer/` - Core functionality for log analysis
- **Key modules**:
  - `analyzer/parser.py` - Log parsing and ML-based anomaly detection
  - `analyzer/views.py` - Web interface views with authentication
  - `analyzer/models.py` - Currently minimal (no custom models)
  - `analyzer/forms.py` - File upload form with validation

### Data Flow (Current vs Intended)

**Current Implementation (Log Analysis Only)**:
1. User uploads MySQL log file via web form (`UploadLogForm`)
2. File is temporarily stored and processed by appropriate parser:
   - `process_slow_log()` for slow query logs
   - `process_general_log()` for general query logs
3. Parser performs feature engineering on queries (length, joins, conditions, etc.)
4. Isolation Forest algorithm detects anomalies
5. Results are paginated and displayed in web interface
6. Temporary files are cleaned up

**Intended Implementation (Full Feature Set)**:
1. **Query Grading Interface**: User pastes SQL query → Analysis engine provides grade (A-F) + specific improvement recommendations
2. **System Analysis**: Log file upload → Contextual analysis of queries within user's database ecosystem → Optimization recommendations
3. **Database Architecture Analysis**: Database schema analysis → Structural optimization recommendations for server/codebase context

### Machine Learning Components
- **Feature Engineering**: Query length, JOIN count, WHERE conditions, subqueries, query type classification
- **Anomaly Detection**: Isolation Forest with different contamination rates (10% for slow logs, 5% for general logs)
- **Performance Monitoring**: Decorators track execution time and memory usage
- **Caching**: Results cached for 1 hour using Django's cache framework

## Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Database setup
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic

# Run development server
python manage.py runserver

# Process log files directly (CLI)
python analyzer/parser.py slow /path/to/slowlog.log
python analyzer/parser.py general /path/to/generallog.log
```

### Docker Development
```bash
# Build and run with Docker Compose (includes PostgreSQL)
docker-compose up --build

# Build Docker image only
docker build -t querygrade .
```

### Testing
```bash
# Run Django tests
python manage.py test

# Run specific app tests
python manage.py test analyzer
```

## Configuration Notes

### Database Configuration
- **Development**: SQLite (default)
- **Production**: PostgreSQL via environment variables:
  - `DB_ENGINE`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`

### Authentication
- User authentication required for all main views
- Login/logout/register views implemented
- Uses Django's built-in authentication system
- Login redirects to `/`, logout redirects to login

### File Handling
- Uploaded files validated for `.log` extension and `text/plain` content type
- Files temporarily stored in Django's media directory
- Automatic cleanup after processing

### Internationalization
- Multi-language support configured (English/Spanish)
- Locale files in `locale/` directory
- Templates use Django's `{% load static %}` pattern

## Deployment

### Kubernetes
- Deployment configuration in `k8s/`
- Includes QueryGrade app, Prometheus monitoring, and Grafana dashboards
- Configured for 3 replicas by default

### Production Settings
- Set `DEBUG=False` in production
- Configure `ALLOWED_HOSTS`
- Use environment variables for sensitive settings
- WhiteNoise configured for static file serving

## Key Dependencies

### Core Framework
- Django 4.0-5.0
- gunicorn for WSGI server
- psycopg2-binary for PostgreSQL

### Machine Learning & Data
- pandas for data manipulation
- scikit-learn for Isolation Forest
- numpy for numerical operations
- matplotlib for potential visualizations

### Development Notes
- Uses Django's cache framework with separate 'process_cache' for ML results
- Performance profiling built into parser functions
- Comprehensive error handling with user-friendly messages
- Pagination implemented for large result sets (10 items per page)

## Implementation Roadmap

### Phase 1: Query Grading Interface (Primary Goal)
- Create form for single query input (textarea)
- Implement query analysis engine with grading algorithm
- Build feedback system with specific improvement recommendations
- Design grade display (A-F scale with explanations)

### Phase 2: Enhanced System Analysis
- Extend current log analysis with contextual database ecosystem insights
- Add database connection capabilities for schema introspection
- Implement query optimization recommendations based on actual database structure

### Phase 3: Database Architecture Analysis
- Schema analysis tools
- Index optimization recommendations
- Table structure analysis
- Relationship optimization suggestions
- Server and application context integration

## Current Limitations
- **Missing Primary Feature**: No query grading interface for individual queries
- **Limited Scope**: Only processes log files, doesn't analyze live database connections
- **No Architecture Analysis**: Cannot analyze database schema or recommend structural improvements
- **File Upload Only**: No text input interface for direct query analysis

## File Upload Constraints (Current Implementation)
- Only `.log` files accepted
- Must be `text/plain` content type
- Files are processed synchronously (consider async processing for large files)

## Security Considerations
- CSRF protection enabled
- File upload validation implemented
- User authentication required
- Secret key should be changed in production
- Database credentials via environment variables