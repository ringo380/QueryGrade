# QueryGrade Testing Guide

This document provides comprehensive guidance for writing and maintaining tests for the QueryGrade project.

## Related Documentation
- **[CLAUDE.md](CLAUDE.md)** - Project overview, architecture, and development setup
- **[INTEGRATION_TEST_FIX_SUMMARY.md](INTEGRATION_TEST_FIX_SUMMARY.md)** - Detailed analysis of the integration test cache issue fix
- **[README.md](README.md)** - Project introduction and quick start guide

## Table of Contents
- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Critical Testing Considerations](#critical-testing-considerations)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Quick Start

### Running Tests
```bash
# Run all tests
python manage.py test

# Run specific test file
python manage.py test analyzer.test_query_grader

# Run specific test class
python manage.py test analyzer.test_integration_refactored.RefactoredQueryGradingIntegrationTestCase

# Run specific test method
python manage.py test analyzer.test_query_grader.TestQueryGrader.test_select_star_detection

# Run with verbosity
python manage.py test analyzer -v 2
```

### Test Coverage Status
- ✅ Query Grader Unit Tests: 28/28 passing
- ✅ Integration Tests (Refactored): 5/5 passing
- ✅ Total: 33/33 passing (100%)

## Test Structure

### Test Files Organization
```
analyzer/
├── test_query_grader.py              # Unit tests for query analysis (28 tests)
├── test_integration_refactored.py    # Integration tests with proper cache handling (5 tests)
├── test_integration.py               # DEPRECATED: Legacy integration tests
├── test_api.py                       # API endpoint tests
├── test_database_analysis.py         # Database-specific tests
└── ml/
    └── tests/
        ├── test_hybrid_grader.py     # ML grading tests
        ├── test_feature_extractor.py # Feature engineering tests
        └── test_feedback_collector.py # Feedback processing tests
```

### Test Categories
1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test end-to-end workflows
3. **ML Tests**: Test machine learning components
4. **API Tests**: Test REST API endpoints
5. **Database Tests**: Test model relationships and constraints

## Critical Testing Considerations

### 1. Cache Management (⚠️ CRITICAL)

**Problem**: The global `query_cache` singleton in `analyzer/performance.py:310` is instantiated at module import time, BEFORE Django's `@override_settings` decorator applies test settings. This causes tests to use the production Redis cache instead of DummyCache.

**Symptoms**:
- Tests return cached Query objects not in test database
- `Query.objects.filter(id=X).exists()` returns `False` for objects that were just created
- Foreign key constraint errors during test teardown
- Inconsistent test results

**Solution**: Always reinitialize cache in test `setUp()`:

```python
def setUp(self):
    from analyzer.performance import query_cache
    from django.core.cache import caches

    # CRITICAL: Reinitialize query_cache with test cache backend
    query_cache.cache = caches['query_analysis_cache']

    # Clear all caches
    for cache_name in ['default', 'query_analysis_cache', 'process_cache', 'template_cache']:
        try:
            caches[cache_name].clear()
        except:
            pass  # Cache might not exist
```

### 2. TransactionTestCase vs TestCase

**When to use TransactionTestCase**:
- Integration tests that test full request/response cycles
- Tests that interact with views decorated with `@transaction.non_atomic_requests`
- When `ATOMIC_REQUESTS=True` in settings (production setting)

**When to use TestCase**:
- Unit tests that don't interact with Django request handling
- Tests that don't require transaction management
- Tests that benefit from automatic database rollback

**Key Differences**:
```python
# TestCase (faster, automatic cleanup)
from django.test import TestCase

class UnitTestCase(TestCase):
    # Automatic transaction rollback after each test
    # Cannot test transaction-related behavior
    # Faster execution

# TransactionTestCase (slower, manual cleanup)
from django.test import TransactionTestCase

class IntegrationTestCase(TransactionTestCase):
    def tearDown(self):
        # REQUIRED: Manual cleanup (no automatic rollback)
        UserQueryHistory.objects.all().delete()
        QueryAnalysis.objects.all().delete()
        Query.objects.all().delete()
        User.objects.all().delete()
```

### 3. Required Test Configuration

All integration tests MUST use these settings:

```python
@override_settings(
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'query_analysis_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'process_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'template_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        }
    }
)
class MyIntegrationTestCase(TransactionTestCase):
    ...
```

## Common Issues and Solutions

### Issue 1: Foreign Key Constraint Errors

**Error**:
```
IntegrityError: analyzer_userqueryhistory.query_id contains a value '1'
that does not have a corresponding value in analyzer_query.id
```

**Root Cause**: Cache returning stale Query objects that aren't in the test database.

**Solution**:
1. Reinitialize cache in setUp() (see [Cache Management](#1-cache-management--critical))
2. Fetch objects by explicit ID, not `.first()` or `.last()`

### Issue 2: Objects Not Persisting

**Symptoms**:
- `Query.objects.count()` returns 0 after creating objects
- Objects have IDs but `exists()` returns False
- Tests pass individually but fail when run together

**Root Causes**:
1. Cache interference (see [Cache Management](#1-cache-management--critical))
2. Using `TestCase` instead of `TransactionTestCase`
3. Missing `ATOMIC_REQUESTS` handling

**Solution**:
1. Use `TransactionTestCase` for integration tests
2. Reinitialize cache in setUp()
3. Use factory methods for test object creation
4. Fetch objects by explicit ID

### Issue 3: Cached Results Contaminating Tests

**Symptoms**:
- Debug output shows "Returning CACHED result" even with DummyCache
- Same Query/Analysis returned for different SQL inputs
- Complex query shows same complexity as simple query

**Root Cause**: Global singleton initialized before test settings apply.

**Solution**: Reinitialize cache in setUp() method.

### Issue 4: Test Database Issues

**Error**:
```
django.db.utils.OperationalError: no such table: analyzer_query
```

**Solution**:
```bash
# Recreate test database
python manage.py migrate --run-syncdb

# Or run tests with --keepdb flag for faster subsequent runs
python manage.py test --keepdb
```

## Best Practices

### 1. Factory Methods for Test Data

Create reusable factory methods for common test objects:

```python
def create_test_user(username='testuser', password='testpass123', email='test@example.com'):
    """Factory method to create a test user."""
    with transaction.atomic():
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )
    return user
```

### 2. Fetch Objects by Explicit ID

**❌ BAD** - May return cached objects:
```python
simple_query = Query.objects.first()
complex_query = Query.objects.last()
```

**✅ GOOD** - Fetch by explicit ID:
```python
# Get ID from response URL
analysis_id = int(response.url.split('/')[-2])
analysis = QueryAnalysis.objects.get(id=analysis_id)
query = analysis.query
```

### 3. Isolate Test Dependencies

Each test should be independent and not rely on other tests:

```python
def test_query_grading(self):
    # Create all necessary objects within the test
    user = create_test_user()
    self.client.force_login(user)

    # Test the functionality
    response = self.client.post(reverse('grade_query'), {...})

    # Verify results
    self.assertEqual(response.status_code, 302)
```

### 4. Clean Up Test Data

Always clean up in `tearDown()` for TransactionTestCase:

```python
def tearDown(self):
    """Clean up test data in reverse dependency order."""
    UserQueryHistory.objects.all().delete()  # Has FK to Query
    QueryAnalysis.objects.all().delete()      # Has FK to Query
    Query.objects.all().delete()              # Referenced by others
    User.objects.all().delete()               # Last
```

### 5. Use Descriptive Test Names

```python
# ❌ BAD
def test_query_1(self):
    ...

# ✅ GOOD
def test_select_star_detection_with_large_table(self):
    """Test that SELECT * is flagged as an issue on large tables."""
    ...
```

### 6. Test Both Success and Failure Cases

```python
def test_valid_query_submission(self):
    """Test that valid queries are accepted and analyzed."""
    ...

def test_invalid_query_rejection(self):
    """Test that invalid queries are rejected with appropriate errors."""
    ...

def test_empty_query_handling(self):
    """Test that empty queries return validation errors."""
    ...
```

## Examples

### Example 1: Complete Integration Test

```python
from django.test import TransactionTestCase, Client, override_settings
from django.contrib.auth.models import User
from django.urls import reverse
from django.db import transaction
from analyzer.models import Query, QueryAnalysis, UserQueryHistory


def create_test_user(username='testuser', password='testpass123', email='test@example.com'):
    """Factory method to create a test user."""
    with transaction.atomic():
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )
    return user


@override_settings(
    RATELIMIT_ENABLE=False,
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'query_analysis_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'process_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        },
        'template_cache': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        }
    }
)
class QueryGradingIntegrationTestCase(TransactionTestCase):
    """Integration tests for query grading workflow."""

    def setUp(self):
        """Set up test client and user."""
        self.client = Client(enforce_csrf_checks=False)

        # CRITICAL: Reinitialize cache with test backend
        from analyzer.performance import query_cache
        from django.core.cache import caches
        query_cache.cache = caches['query_analysis_cache']

        # Clear all caches
        for cache_name in ['default', 'query_analysis_cache', 'process_cache', 'template_cache']:
            try:
                caches[cache_name].clear()
            except:
                pass

        # Create test user
        self.test_user = create_test_user(
            username='integrationuser',
            email='integration@example.com',
            password='testpass123'
        )

        # Force login
        self.client.force_login(self.test_user)

    def tearDown(self):
        """Clean up test data."""
        UserQueryHistory.objects.all().delete()
        QueryAnalysis.objects.all().delete()
        Query.objects.all().delete()
        User.objects.all().delete()

    def test_full_query_grading_workflow(self):
        """Test the complete workflow from submission to results."""
        test_query = "SELECT id, name FROM users WHERE status = 'active'"

        # Submit query for grading
        response = self.client.post(reverse('grade_query'), {
            'sql_query': test_query,
            'database_type': 'mysql'
        })

        # Should redirect to results page
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith('/grade/enhanced/'))

        # Verify objects were created
        self.assertEqual(Query.objects.count(), 1)
        self.assertEqual(QueryAnalysis.objects.count(), 1)
        self.assertEqual(UserQueryHistory.objects.count(), 1)

        # Fetch by explicit ID
        analysis_id = int(response.url.split('/')[-2])
        analysis = QueryAnalysis.objects.get(id=analysis_id)
        query = analysis.query

        # Verify analysis results
        self.assertIn(analysis.grade, ['A', 'B', 'C', 'D', 'F'])
        self.assertGreaterEqual(analysis.score, 0)
        self.assertLessEqual(analysis.score, 100)
        self.assertEqual(query.query_type, 'SELECT')
```

### Example 2: Unit Test for Analyzer

```python
from django.test import TestCase
from analyzer.analyzers import QueryGrader
from analyzer.exceptions import EmptyQueryError


class TestQueryGrader(TestCase):
    """Unit tests for QueryGrader analyzer."""

    def setUp(self):
        """Set up test grader."""
        self.grader = QueryGrader()

    def test_select_star_detection(self):
        """Test that SELECT * is detected and flagged."""
        sql = "SELECT * FROM users"
        query, analysis = self.grader.analyze_query(sql, 'mysql')

        # Check that issue was detected
        issues = analysis.issues_found
        select_star_issues = [i for i in issues if i.get('type') == 'select_star']
        self.assertGreater(len(select_star_issues), 0)

    def test_empty_query_raises_error(self):
        """Test that empty queries raise EmptyQueryError."""
        with self.assertRaises(EmptyQueryError):
            self.grader.analyze_query('', 'mysql')

        with self.assertRaises(EmptyQueryError):
            self.grader.analyze_query('   ', 'mysql')

    def test_complex_query_higher_complexity(self):
        """Test that complex queries have higher complexity scores."""
        simple_sql = "SELECT id FROM users"
        complex_sql = """
            SELECT u.id, COUNT(o.id)
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.status = 'active'
            GROUP BY u.id
        """

        simple_query, _ = self.grader.analyze_query(simple_sql, 'mysql')
        complex_query, _ = self.grader.analyze_query(complex_sql, 'mysql')

        self.assertGreater(
            complex_query.estimated_complexity,
            simple_query.estimated_complexity
        )
```

## Troubleshooting Checklist

If tests are failing, work through this checklist:

- [ ] Are you using `TransactionTestCase` for integration tests?
- [ ] Did you reinitialize `query_cache` in `setUp()`?
- [ ] Are you using `@override_settings` with all 4 DummyCache backends?
- [ ] Are you clearing caches in `setUp()`?
- [ ] Are you fetching objects by explicit ID instead of `.first()`/`.last()`?
- [ ] Did you implement `tearDown()` to clean up test data?
- [ ] Are you using factory methods for test object creation?
- [ ] Is `RATELIMIT_ENABLE=False` in test settings?
- [ ] Did you run migrations? (`python manage.py migrate`)
- [ ] Are you using `--keepdb` for faster subsequent test runs?

## Additional Resources

### QueryGrade Documentation
- **[INTEGRATION_TEST_FIX_SUMMARY.md](INTEGRATION_TEST_FIX_SUMMARY.md)** - In-depth case study of solving the cache initialization issue
- **[CLAUDE.md](CLAUDE.md)** - Complete project documentation including testing section
- **[README.md](README.md)** - Project overview and setup instructions
- **[analyzer/test_integration_refactored.py](analyzer/test_integration_refactored.py)** - Working integration test implementation with detailed comments

### Django & Testing Resources
- [Django Testing Documentation](https://docs.djangoproject.com/en/stable/topics/testing/)
- [Django TestCase vs TransactionTestCase](https://docs.djangoproject.com/en/stable/topics/testing/tools/#transactiontestcase)
- [pytest-django](https://pytest-django.readthedocs.io/) - Alternative testing framework

## Getting Help

If you encounter testing issues:

1. Check this document's [Common Issues and Solutions](#common-issues-and-solutions)
2. Review the [Troubleshooting Checklist](#troubleshooting-checklist)
3. Examine `analyzer/test_integration_refactored.py` for working examples
4. Run tests with `-v 2` for verbose output
5. Add debug prints to understand test flow
6. Check that all migrations are applied
