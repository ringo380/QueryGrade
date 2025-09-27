from django import forms
from django.core.exceptions import ValidationError
import os
import sqlparse

def validate_log_file(file):
    """
    Enhanced validator function to ensure the uploaded file is a valid and secure log file.

    Args:
        file: The uploaded file.

    Raises:
        forms.ValidationError: If the file is not a valid log file or poses security risks.
    """
    import magic
    import mimetypes

    # File size validation (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    if file.size > max_size:
        raise ValidationError(f"File too large. Maximum size is {max_size // (1024*1024)}MB.")

    # Validate file extension
    valid_extensions = ['.log', '.txt']
    if not any(file.name.lower().endswith(ext) for ext in valid_extensions):
        raise ValidationError("Invalid file extension. Only .log and .txt files are allowed.")

    # Validate MIME type using multiple methods
    valid_mime_types = ['text/plain', 'application/octet-stream']

    # Primary MIME type check
    if file.content_type not in valid_mime_types:
        raise ValidationError(f"Invalid file type: {file.content_type}. Only text files are allowed.")

    # Secondary MIME type validation using python-magic if available
    try:
        file.seek(0)
        file_content = file.read(1024)  # Read first 1KB for magic detection
        file.seek(0)  # Reset file pointer

        detected_mime = magic.from_buffer(file_content, mime=True)
        if detected_mime not in ['text/plain', 'application/octet-stream', 'text/x-log']:
            raise ValidationError(f"File content does not match expected format. Detected: {detected_mime}")
    except ImportError:
        # python-magic not available, use mimetypes as fallback
        guessed_type, _ = mimetypes.guess_type(file.name)
        if guessed_type and guessed_type not in valid_mime_types:
            raise ValidationError(f"Invalid file type based on filename: {guessed_type}")

    # Check for potentially malicious filename patterns
    import re
    dangerous_patterns = [
        r'\.\./',  # Directory traversal
        r'[<>:"|?*]',  # Invalid filename characters
        r'\x00',  # Null bytes
        r'\.exe$|\.bat$|\.cmd$|\.com$|\.scr$|\.pif$',  # Executable extensions
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, file.name, re.IGNORECASE):
            raise ValidationError("Filename contains invalid or potentially dangerous characters.")

    # Validate filename length
    if len(file.name) > 255:
        raise ValidationError("Filename is too long. Maximum length is 255 characters.")

    # Basic content validation - check for text file characteristics
    try:
        file.seek(0)
        sample_content = file.read(512).decode('utf-8', errors='ignore')
        file.seek(0)

        # Check for potentially malicious content patterns
        malicious_patterns = [
            r'<script[^>]*>',  # JavaScript injection
            r'<iframe[^>]*>',  # Iframe injection
            r'(?i)exec\s*\(',  # Code execution attempts
            r'(?i)eval\s*\(',  # Eval attempts
            r'(?i)system\s*\(',  # System command execution
            r'(?i)shell_exec\s*\(',  # Shell execution
            r'%[0-9a-fA-F]{2}%[0-9a-fA-F]{2}',  # URL encoding (potential obfuscation)
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding (potential obfuscation)
        ]

        for pattern in malicious_patterns:
            if re.search(pattern, sample_content, re.IGNORECASE):
                raise ValidationError("File contains potentially malicious content patterns.")

        # Check for excessive binary data (should be mostly text)
        printable_chars = sum(1 for c in sample_content if c.isprintable() or c.isspace())
        if len(sample_content) > 0 and (printable_chars / len(sample_content)) < 0.8:
            raise ValidationError("File contains too much binary data. Please ensure it's a valid text log file.")

        # Check if content looks like a log file (contains timestamps, common log patterns)
        log_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # Date pattern
            r'\d{2}:\d{2}:\d{2}',  # Time pattern
            r'(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)',  # SQL keywords
            r'(INFO|WARN|ERROR|DEBUG)',  # Log levels
            r'mysql-slow',  # MySQL slow log identifier
            r'mysqld',  # MySQL daemon references
            r'Time:\s*\d+',  # Query time references
        ]

        has_log_content = any(re.search(pattern, sample_content, re.IGNORECASE) for pattern in log_patterns)
        if not has_log_content and len(sample_content.strip()) > 0:
            raise ValidationError("File does not appear to contain valid log data.")

        # Additional validation: check for minimum file size
        if file.size < 10:  # Files smaller than 10 bytes are suspicious
            raise ValidationError("File is too small to be a valid log file.")

        # Check for excessive line length (potential attack vector)
        lines = sample_content.split('\n')
        max_line_length = max(len(line) for line in lines) if lines else 0
        if max_line_length > 5000:  # Reasonable limit for log lines
            raise ValidationError("File contains unusually long lines. Please verify it's a valid log file.")

    except UnicodeDecodeError:
        raise ValidationError("File contains invalid text encoding. Please ensure it's a valid text file.")

def validate_sql_query(sql_text):
    """
    Enhanced validator function to ensure the SQL query is valid, parseable, and secure.

    Args:
        sql_text (str): The SQL query text to validate.

    Raises:
        forms.ValidationError: If the SQL query is invalid or potentially malicious.
    """
    if not sql_text or not sql_text.strip():
        raise ValidationError("SQL query cannot be empty.")

    # Length validation
    if len(sql_text) > 10000:  # 10KB limit
        raise ValidationError("SQL query is too long. Maximum length is 10,000 characters.")

    # Basic security checks - block dangerous patterns
    dangerous_patterns = [
        r'(?i)\b(exec|execute|sp_executesql)\b',  # Stored procedure execution
        r'(?i)\b(xp_cmdshell|xp_regread|xp_regwrite)\b',  # System commands
        r'(?i)\b(alter\s+table|drop\s+table|drop\s+database|truncate)\b',  # DDL operations
        r'(?i)\b(insert\s+into|update\s+|delete\s+from)\b',  # DML operations
        r'(?i)\b(create\s+|drop\s+|alter\s+)\b',  # Schema modifications
        r'(?i)\b(grant\s+|revoke\s+)\b',  # Permission changes
        r'(?i)(\-\-|\#|\/\*)',  # Comment injection attempts
        r'(?i)\b(union\s+(?:all\s+)?select)\b',  # Union-based injection
        r'(?i)\b(information_schema|sys\.|pg_catalog)\b',  # System schema access
        r'(?i)\b(char|ascii|substring|mid|left|right)\s*\(',  # String manipulation functions
        r'(?i)\b(sleep|waitfor|benchmark)\s*\(',  # Time-based attacks
        r'(?i)\b(load_file|into\s+outfile|into\s+dumpfile)\b',  # File operations
    ]

    import re
    for pattern in dangerous_patterns:
        if re.search(pattern, sql_text):
            raise ValidationError("SQL query contains potentially dangerous operations that are not allowed for analysis.")

    # Check for excessive semicolons (potential multi-statement injection)
    semicolon_count = sql_text.count(';')
    if semicolon_count > 1:
        raise ValidationError("Multiple SQL statements are not allowed. Please submit one query at a time.")

    try:
        # Try to parse the SQL query
        parsed = sqlparse.parse(sql_text)
        if not parsed:
            raise ValidationError("Unable to parse the SQL query.")

        # Check if it contains at least one statement
        if not any(stmt.tokens for stmt in parsed):
            raise ValidationError("No valid SQL statements found.")

        # Validate that it's a SELECT statement (read-only)
        first_stmt = parsed[0] if parsed else None
        if first_stmt:
            # Get the first non-whitespace token
            first_token = None
            for token in first_stmt.flatten():
                if not token.is_whitespace:
                    first_token = token
                    break

            if first_token and first_token.ttype is sqlparse.tokens.DML:
                if first_token.value.upper() != 'SELECT':
                    raise ValidationError("Only SELECT queries are allowed for analysis.")
            elif first_token and first_token.ttype is sqlparse.tokens.Keyword:
                if first_token.value.upper() not in ['SELECT', 'WITH']:
                    raise ValidationError("Only SELECT and WITH queries are allowed for analysis.")

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Invalid SQL query: {str(e)}")

class UploadLogForm(forms.Form):
    """
    Form for uploading MySQL log files.
    """
    LOG_TYPE_CHOICES = [
        ('slow', 'Slow Query Log'),
        ('general', 'General Query Log'),
    ]
    log_type = forms.ChoiceField(choices=LOG_TYPE_CHOICES, label='Select Log Type')
    log_file = forms.FileField(label='Choose Log File', validators=[validate_log_file])
    use_async = forms.BooleanField(
        required=False,
        initial=False,
        label='Process in background',
        help_text='Enable for large files (recommended for files > 10MB)'
    )


class QueryGradeForm(forms.Form):
    """
    Form for submitting SQL queries for grading and analysis.
    """
    DATABASE_CHOICES = [
        ('', 'Select Database (Optional)'),
        ('mysql', 'MySQL'),
        ('postgresql', 'PostgreSQL'),
        ('sqlite', 'SQLite'),
        ('oracle', 'Oracle'),
        ('sqlserver', 'SQL Server'),
        ('other', 'Other'),
    ]

    sql_query = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 12,
            'placeholder': 'Paste your SQL query here...\n\nExample:\nSELECT u.name, u.email, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.created_at >= \'2023-01-01\'\nGROUP BY u.id, u.name, u.email\nORDER BY order_count DESC;',
            'style': 'font-family: monospace; font-size: 14px;'
        }),
        label='SQL Query',
        help_text='Enter your SQL query for analysis and grading',
        validators=[validate_sql_query]
    )

    database_type = forms.ChoiceField(
        choices=DATABASE_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label='Database Type',
        help_text='Optional: Specify your database type for more targeted recommendations'
    )

    database_version = forms.CharField(
        max_length=50,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 8.0, 13.4, etc.'
        }),
        label='Database Version',
        help_text='Optional: Database version for version-specific recommendations'
    )

    use_case_notes = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Optional: Describe what this query is used for, expected data volume, frequency of execution, etc.'
        }),
        required=False,
        label='Use Case Notes',
        help_text='Optional: Additional context about how this query is used'
    )
    use_async = forms.BooleanField(
        required=False,
        initial=False,
        label='Process in background',
        help_text='Enable for large query batches (recommended for > 10 queries)'
    )

    def clean_sql_query(self):
        """Additional cleaning for SQL query field."""
        sql_query = self.cleaned_data.get('sql_query')
        if sql_query:
            # Remove excessive whitespace while preserving structure
            sql_query = '\n'.join(line.strip() for line in sql_query.split('\n') if line.strip())
        return sql_query


class QueryCompareForm(forms.Form):
    """
    Form for comparing multiple SQL queries side-by-side.
    """
    DATABASE_CHOICES = [
        ('', 'Select Database (Optional)'),
        ('mysql', 'MySQL'),
        ('postgresql', 'PostgreSQL'),
        ('sqlite', 'SQLite'),
        ('oracle', 'Oracle'),
        ('sqlserver', 'SQL Server'),
        ('other', 'Other'),
    ]

    query_1 = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control sql-editor',
            'rows': 10,
            'placeholder': 'Enter your first SQL query...\n\nExample:\nSELECT * FROM users WHERE active = 1;',
            'style': 'font-family: monospace; font-size: 14px;'
        }),
        label='Query 1',
        help_text='First query to compare',
        validators=[validate_sql_query]
    )

    query_1_name = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Original Query, Current Implementation, etc.'
        }),
        label='Query 1 Name',
        help_text='Optional: Give this query a descriptive name'
    )

    query_2 = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control sql-editor',
            'rows': 10,
            'placeholder': 'Enter your second SQL query...\n\nExample:\nSELECT id, name, email FROM users WHERE active = 1;',
            'style': 'font-family: monospace; font-size: 14px;'
        }),
        label='Query 2',
        help_text='Second query to compare',
        validators=[validate_sql_query]
    )

    query_2_name = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Optimized Query, Alternative Version, etc.'
        }),
        label='Query 2 Name',
        help_text='Optional: Give this query a descriptive name'
    )

    query_3 = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control sql-editor',
            'rows': 10,
            'placeholder': 'Enter your third SQL query (optional)...',
            'style': 'font-family: monospace; font-size: 14px;'
        }),
        required=False,
        label='Query 3 (Optional)',
        help_text='Third query to compare (optional)'
    )

    query_3_name = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Best Practice Version, Final Optimized, etc.'
        }),
        label='Query 3 Name',
        help_text='Optional: Give this query a descriptive name'
    )

    database_type = forms.ChoiceField(
        choices=DATABASE_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label='Database Type',
        help_text='Optional: Specify database type for all queries'
    )

    comparison_notes = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Optional: Add notes about what you\'re comparing or what differences you expect to see...'
        }),
        required=False,
        label='Comparison Notes',
        help_text='Optional: Context about the comparison'
    )

    def clean_query_3(self):
        """Validate third query if provided."""
        query_3 = self.cleaned_data.get('query_3')
        if query_3 and query_3.strip():
            validate_sql_query(query_3)
            # Remove excessive whitespace while preserving structure
            query_3 = '\n'.join(line.strip() for line in query_3.split('\n') if line.strip())
        return query_3

    def clean_query_1(self):
        """Additional cleaning for query 1."""
        query_1 = self.cleaned_data.get('query_1')
        if query_1:
            query_1 = '\n'.join(line.strip() for line in query_1.split('\n') if line.strip())
        return query_1

    def clean_query_2(self):
        """Additional cleaning for query 2."""
        query_2 = self.cleaned_data.get('query_2')
        if query_2:
            query_2 = '\n'.join(line.strip() for line in query_2.split('\n') if line.strip())
        return query_2


class BatchQueryForm(forms.Form):
    """
    Form for analyzing multiple SQL queries in batch.
    """
    DATABASE_CHOICES = [
        ('', 'Select Database (Optional)'),
        ('mysql', 'MySQL'),
        ('postgresql', 'PostgreSQL'),
        ('sqlite', 'SQLite'),
        ('oracle', 'Oracle'),
        ('sqlserver', 'SQL Server'),
        ('other', 'Other'),
    ]

    queries_text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control sql-editor',
            'rows': 15,
            'placeholder': 'Enter multiple SQL queries separated by semicolons...\n\nExample:\nSELECT * FROM users WHERE active = 1;\nSELECT COUNT(*) FROM orders;\nSELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nGROUP BY u.id;',
            'style': 'font-family: monospace; font-size: 14px;'
        }),
        label='SQL Queries',
        help_text='Enter multiple SQL queries separated by semicolons. Each query will be analyzed individually.',
        validators=[validate_sql_query]
    )

    database_type = forms.ChoiceField(
        choices=DATABASE_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label='Database Type',
        help_text='Optional: Specify database type for all queries'
    )

    database_version = forms.CharField(
        max_length=50,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 8.0, 13.4, etc.'
        }),
        label='Database Version',
        help_text='Optional: Database version for version-specific recommendations'
    )

    analysis_notes = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Optional: Describe the context for these queries, their purpose, expected data volume, etc.'
        }),
        required=False,
        label='Analysis Notes',
        help_text='Optional: Additional context about these queries'
    )

    def clean_queries_text(self):
        """Parse and validate multiple SQL queries."""
        queries_text = self.cleaned_data.get('queries_text')
        if not queries_text:
            raise forms.ValidationError("Please provide at least one SQL query.")

        # Split queries by semicolon and clean them
        raw_queries = queries_text.split(';')
        cleaned_queries = []

        for i, query in enumerate(raw_queries, 1):
            query = query.strip()
            if query:  # Skip empty queries
                try:
                    validate_sql_query(query)
                    cleaned_queries.append(query)
                except forms.ValidationError as e:
                    raise forms.ValidationError(f"Error in query {i}: {e}")

        if not cleaned_queries:
            raise forms.ValidationError("No valid SQL queries found. Please check your input.")

        if len(cleaned_queries) > 20:  # Limit to prevent abuse
            raise forms.ValidationError("Too many queries. Please limit to 20 queries per batch.")

        # Store cleaned queries for later use
        self.cleaned_queries = cleaned_queries
        return queries_text

    def get_parsed_queries(self):
        """Get the list of parsed and validated queries."""
        return getattr(self, 'cleaned_queries', [])


class QueryFeedbackForm(forms.Form):
    """Form for collecting user feedback on query analysis quality."""

    RATING_CHOICES = [
        (1, '1 - Very Poor'),
        (2, '2 - Poor'),
        (3, '3 - Average'),
        (4, '4 - Good'),
        (5, '5 - Excellent'),
    ]

    accuracy_rating = forms.ChoiceField(
        choices=RATING_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='How accurate was the analysis?',
        required=True
    )

    usefulness_rating = forms.ChoiceField(
        choices=RATING_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='How useful were the recommendations?',
        required=True
    )

    clarity_rating = forms.ChoiceField(
        choices=RATING_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='How clear was the feedback?',
        required=True
    )

    suggestions = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'What could we improve? Any suggestions for better analysis or recommendations?'
        }),
        required=False,
        label='Suggestions for Improvement',
        help_text='Optional: Help us make QueryGrade better'
    )

    would_recommend = forms.BooleanField(
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        required=False,
        label='Would you recommend QueryGrade to others?',
        help_text='Optional'
    )


class DatabaseConnectionForm(forms.Form):
    """Form for configuring database connections for introspection."""

    ENGINE_CHOICES = [
        ('mysql', 'MySQL'),
        ('postgresql', 'PostgreSQL'),
        ('sqlite', 'SQLite'),
    ]

    engine = forms.ChoiceField(
        choices=ENGINE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label='Database Engine',
        help_text='Select your database type'
    )

    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'database_name'
        }),
        label='Database Name',
        help_text='Name of the database to connect to'
    )

    host = forms.CharField(
        max_length=100,
        required=False,
        initial='localhost',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'localhost'
        }),
        label='Host',
        help_text='Database server hostname (leave empty for SQLite)'
    )

    port = forms.CharField(
        max_length=10,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '3306 (MySQL) or 5432 (PostgreSQL)'
        }),
        label='Port',
        help_text='Database server port (leave empty for default)'
    )

    user = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'username'
        }),
        label='Username',
        help_text='Database username (leave empty for SQLite)'
    )

    password = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'password'
        }),
        label='Password',
        help_text='Database password (leave empty for SQLite)'
    )

    schema = forms.CharField(
        max_length=100,
        required=False,
        initial='public',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'public'
        }),
        label='Schema',
        help_text='Database schema (PostgreSQL only, optional)'
    )

    def clean(self):
        """Validate database connection parameters."""
        cleaned_data = super().clean()
        engine = cleaned_data.get('engine')
        name = cleaned_data.get('name')
        host = cleaned_data.get('host')
        user = cleaned_data.get('user')
        password = cleaned_data.get('password')

        if engine == 'sqlite':
            # SQLite only needs the database name (file path)
            if not name:
                raise forms.ValidationError("Database name (file path) is required for SQLite.")
        else:
            # MySQL and PostgreSQL need host, user, and typically password
            if not host:
                cleaned_data['host'] = 'localhost'
            if not user:
                raise forms.ValidationError(f"Username is required for {engine}.")
            # Password is often required but can be empty in some configurations

        return cleaned_data

    def get_connection_config(self):
        """Return configuration dict for DatabaseIntrospector."""
        data = self.cleaned_data
        return {
            'engine': data['engine'],
            'name': data['name'],
            'host': data.get('host', 'localhost'),
            'port': data.get('port', ''),
            'user': data.get('user', ''),
            'password': data.get('password', ''),
            'schema': data.get('schema', 'public')
        }
