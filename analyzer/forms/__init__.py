"""
Forms package for QueryGrade analyzer.

This package contains all form classes and validators organized by domain:
- validators: validate_log_file, validate_sql_query
- upload_forms: UploadLogForm
- query_forms: QueryGradeForm, QueryCompareForm, BatchQueryForm
- feedback_forms: QueryFeedbackForm
- database_forms: DatabaseConnectionForm

All forms and validators are exported from this module for backward compatibility
with existing imports: `from analyzer.forms import QueryGradeForm`
"""

from .validators import validate_log_file, validate_sql_query
from .upload_forms import UploadLogForm
from .query_forms import QueryGradeForm, QueryCompareForm, BatchQueryForm
from .feedback_forms import QueryFeedbackForm
from .database_forms import DatabaseConnectionForm

__all__ = [
    # Validators
    'validate_log_file',
    'validate_sql_query',
    # Upload forms
    'UploadLogForm',
    # Query forms
    'QueryGradeForm',
    'QueryCompareForm',
    'BatchQueryForm',
    # Feedback forms
    'QueryFeedbackForm',
    # Database forms
    'DatabaseConnectionForm',
]
