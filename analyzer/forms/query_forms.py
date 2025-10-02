from django import forms
from .validators import validate_sql_query


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
            raise forms.ValidationError(
                "✍️ No queries provided. "
                "Please enter one or more SQL queries separated by semicolons."
            )

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
                    raise forms.ValidationError(
                        f"❌ Error in query #{i}:\n{str(e)}\n\n"
                        f"💡 Fix this query and try again, or remove it to analyze the others."
                    )

        if not cleaned_queries:
            raise forms.ValidationError(
                "❓ No valid queries found. "
                "Make sure queries are separated by semicolons and contain valid SQL syntax."
            )

        if len(cleaned_queries) > 20:  # Limit to prevent abuse
            raise forms.ValidationError(
                f"📊 Too many queries ({len(cleaned_queries)} found). "
                f"Maximum allowed is 20 queries per batch. "
                f"Please split your queries into multiple batches."
            )

        # Store cleaned queries for later use
        self.cleaned_queries = cleaned_queries
        return queries_text

    def get_parsed_queries(self):
        """Get the list of parsed and validated queries."""
        return getattr(self, 'cleaned_queries', [])
