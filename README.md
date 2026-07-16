# QueryGrade

QueryGrade is a comprehensive Django-based SQL query analysis and database optimization platform designed to help developers and database administrators improve query performance and database architecture.

## Vision & Core Purposes

QueryGrade serves three primary functions, in order of priority:

### 1. 🎯 SQL Query Grading (Primary Feature)
Get instant feedback on individual SQL queries by pasting them into a simple interface:
- **Letter Grade (A-F)**: Immediate performance assessment
- **Specific Feedback**: Detailed recommendations for improvement
- **Best Practices**: Learn optimal query writing techniques
- **Performance Metrics**: Understand query complexity and efficiency

### 2. 📊 System Query Analysis
Analyze queries running within your existing database ecosystem:
- **Log File Analysis**: Upload slow query logs and general query logs
- **Contextual Insights**: Recommendations tailored to your specific database structure
- **Anomaly Detection**: Identify problematic queries using machine learning
- **System-Specific Optimization**: Suggestions based on your current database setup

### 3. 🏗️ Database Architecture Optimization
Comprehensive database architecture analysis and recommendations:
- **Schema Analysis**: Review table structures and relationships
- **Index Optimization**: Identify missing or redundant indexes
- **Architecture Recommendations**: Structural improvements for your specific server environment
- **Application Context**: Optimization suggestions that consider your codebase and server configuration

## Current Implementation Status

**✅ Phase 1 COMPLETE**: SQL Query Grading Interface is fully functional and ready to use!

**🎯 Current Focus**: Quality improvements, comprehensive testing, and enhanced analysis patterns.

**🚀 Next Priority**: Integration features connecting query grading with log analysis and database architecture optimization.

## Current Features

### 🎯 SQL Query Grading (PRIMARY FEATURE)
- **Letter Grades (A-F)**: Instant performance assessment for any SQL query
- **Smart Analysis Engine**: Detects 10+ common performance issues
- **Detailed Feedback**: Specific recommendations with examples
- **Query History**: Track your progress and view past analyses
- **Professional UI**: Dark mode interface with syntax highlighting
- **Example Queries**: Built-in examples for testing different scenarios

### 📊 Log File Analysis
- Upload MySQL slow query logs and general query logs
- Machine learning-based anomaly detection using Isolation Forest
- Identify problematic queries automatically
- Paginated results with detailed analysis

### 🔐 Core Platform Features
- User authentication and secure session management
- Query result caching for improved performance
- Responsive design for desktop and mobile
- Export-ready analysis results
- Real-time SQL validation

## Development Roadmap

### ✅ Phase 1: Query Grading Interface (COMPLETE)
- [x] Create query input form with syntax highlighting
- [x] Implement query analysis engine with grading algorithm
- [x] Build comprehensive feedback system
- [x] Design intuitive grade display (A-F with explanations)
- [x] Add query optimization suggestions
- [x] User query history and tracking
- [x] Professional dark mode UI

### Phase 2: Enhanced System Analysis
- [x] Basic log file anomaly detection (✅ Complete)
- [ ] Database connection capabilities for schema introspection
- [ ] Contextual analysis based on actual database structure
- [ ] Enhanced optimization recommendations

### Phase 3: Database Architecture Analysis
- [ ] Schema analysis and visualization tools
- [ ] Index optimization recommendations
- [ ] Table relationship analysis
- [ ] Server and application context integration

## Requirements

- Python 3.x
- Django 4.0-5.0
- pandas 2.2.3+
- scikit-learn 1.5.2+
- numpy 1.26.4+
- sqlparse (for SQL parsing)
- matplotlib (for visualizations)
- psycopg2-binary (for PostgreSQL support)
- gunicorn (for production deployment)

## 🚀 Quick Start

### Try the Live Query Grader

1. **Clone and setup**:
    ```bash
    git clone https://github.com/yourusername/QueryGrade.git
    cd QueryGrade
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Initialize database**:
    ```bash
    python manage.py migrate
    python manage.py createsuperuser  # Optional: create admin user
    ```

4. **Start the server**:
    ```bash
    python manage.py runserver
    ```

5. **Start grading queries**:
    - Open http://127.0.0.1:8000 in your browser
    - Register for a new account or login
    - Click "Grade My Query" to start analyzing SQL queries instantly!

### Example Usage

Paste any SQL query to get instant feedback:

```sql
-- This query will get a poor grade (D or F)
SELECT *
FROM users u
WHERE UPPER(u.email) LIKE '%@GMAIL.%'
  AND u.id NOT IN (SELECT user_id FROM orders)
ORDER BY u.created_at;

-- This query will get a good grade (A or B)
SELECT u.id, u.name, u.email, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2023-01-01'
  AND u.status = 'active'
GROUP BY u.id, u.name, u.email
ORDER BY order_count DESC;
```

## ✨ Features Overview

### 🎯 Smart Query Analysis
- **Performance Grading**: Get letter grades (A-F) for any SQL query
- **Issue Detection**: Identifies SELECT *, inefficient JOINs, missing indexes, and more
- **Optimization Tips**: Specific recommendations with examples
- **Database Support**: MySQL, PostgreSQL, SQLite, Oracle, SQL Server

### 📊 Analysis Results
- **Detailed Scoring**: Understand exactly why your query got its grade
- **Visual Feedback**: Color-coded results with severity indicators
- **Historical Tracking**: View your query analysis history
- **Export Options**: Print or save results for documentation

### 🔍 What QueryGrade Detects
- ❌ `SELECT *` usage (inefficient column selection)
- ❌ Functions on columns in WHERE clauses (prevents index usage)
- ❌ Leading wildcards in LIKE patterns (`LIKE '%text'`)
- ❌ Cartesian products (missing JOIN conditions)
- ❌ Excessive JOINs (performance bottlenecks)
- ❌ NOT IN with potential NULL issues
- ❌ Complex subqueries that could be optimized
- ✅ Proper indexing opportunities
- ✅ Query structure improvements
- ✅ Best practice recommendations

### 📈 Log File Analysis
Upload MySQL log files for batch analysis:
- Slow query log processing
- General query log analysis
- ML-based anomaly detection
- Performance pattern identification

## 🐳 Docker Deployment

For containerized deployment with PostgreSQL:

```bash
docker-compose up --build
```

## Usage

### Current Features (Log Analysis)
1. **Login/Register**: Create an account or log in
2. **Upload Logs**: Select slow query log or general query log
3. **View Results**: Review detected anomalies with scores and recommendations

### Coming Soon (Query Grading)
1. **Paste Query**: Enter your SQL query in the text area
2. **Get Grade**: Receive instant A-F grade with detailed feedback
3. **Learn & Improve**: Follow specific recommendations to optimize your query

## Contributing

We welcome contributions! Priority areas:

1. **Query Grading Engine**: Help build the core query analysis and grading system
2. **UI/UX Improvements**: Enhance the user interface for better query input and feedback
3. **Database Integrations**: Add support for different database systems
4. **Analysis Algorithms**: Improve query optimization recommendations

### Developer Documentation

Before contributing, please review:
- **[CLAUDE.md](CLAUDE.md)** - Complete project documentation, architecture, and development guide
- **[TESTING.md](TESTING.md)** - Testing best practices and troubleshooting guide
- **[INTEGRATION_TEST_FIX_SUMMARY.md](INTEGRATION_TEST_FIX_SUMMARY.md)** - Case study on cache issues in tests

## Project Structure

```
QueryGrade/
├── querygrade/          # Django project settings
├── analyzer/            # Main application
│   ├── parser.py        # Log parsing and ML analysis
│   ├── views.py         # Web interface views
│   ├── templates/       # HTML templates
│   └── static/          # CSS, JS, images
├── scripts/             # Developer utilities (og:image card generation)
├── requirements.txt     # Python dependencies (full, includes test deps)
├── requirements-prod.txt # Slim deps for the deployed web service
├── Dockerfile          # Local dev image, used by docker-compose
├── Dockerfile.web      # Production web image (see railway.toml)
├── docker-compose.yml  # Multi-service deployment
└── k8s/               # Kubernetes deployment files
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.