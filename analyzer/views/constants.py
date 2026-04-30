"""
Shared constants across view modules.
"""

# Grade colors for template rendering
GRADE_COLORS = {
    'A': 'success',  # Green
    'B': 'info',     # Blue
    'C': 'warning',  # Yellow
    'D': 'orange',   # Orange
    'F': 'danger'    # Red
}

# Pagination settings
DEFAULT_HISTORY_PAGE_SIZE = 10
MAX_BATCH_SIZE = 100

# Rate limiting (for reference, actual limits in decorators)
QUERY_RATE_LIMIT = '20/m'
FEEDBACK_RATE_LIMIT = '10/m'
LOGIN_RATE_LIMIT = '5/5m'
REGISTER_RATE_LIMIT = '3/h'
ANON_QUERY_RATE_LIMIT = '5/m'

# Anonymous trial
ANON_TRIAL_CAP = 3
ANON_ANALYSIS_SESSION_KEY = 'anon_analysis_ids'
ANON_TRIAL_COUNT_KEY = 'anon_trial_count'
ANON_ANALYSIS_HISTORY_LIMIT = 10