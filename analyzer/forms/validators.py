import os
import re

import sqlparse
from django import forms
from django.core.exceptions import ValidationError


def validate_log_file(file):
    """
    Enhanced validator function to ensure the uploaded file is a valid and secure log file.

    Args:
        file: The uploaded file.

    Raises:
        forms.ValidationError: If the file is not a valid log file or poses security risks.
    """
    import mimetypes

    import magic

    # File size validation (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    if file.size > max_size:
        file_mb = file.size / (1024 * 1024)
        raise ValidationError(
            f"📁 File is too large ({file_mb:.1f}MB). "
            f"Maximum allowed size is 50MB. "
            f"Try splitting your log file or uploading a smaller time range."
        )

    # Validate file extension
    valid_extensions = [".log", ".txt"]
    if not any(file.name.lower().endswith(ext) for ext in valid_extensions):
        file_ext = os.path.splitext(file.name)[1]
        raise ValidationError(
            f"❌ Invalid file type '{file_ext}'. "
            f"Please upload a .log or .txt file containing your database query logs."
        )

    # Validate MIME type using multiple methods
    valid_mime_types = ["text/plain", "application/octet-stream"]

    # Primary MIME type check
    if file.content_type not in valid_mime_types:
        raise ValidationError(
            f"🚫 Unsupported file format. "
            f"Expected a plain text log file, but received '{file.content_type}'. "
            f"Please convert your file to .txt or .log format."
        )

    # Secondary MIME type validation using python-magic if available
    try:
        file.seek(0)
        file_content = file.read(1024)  # Read first 1KB for magic detection
        file.seek(0)  # Reset file pointer

        detected_mime = magic.from_buffer(file_content, mime=True)
        if detected_mime not in [
            "text/plain",
            "application/octet-stream",
            "text/x-log",
        ]:
            raise ValidationError(
                f"⚠️ File content appears to be '{detected_mime}', not a text log file. "
                f"Please ensure you're uploading an actual database log file."
            )
    except ImportError:
        # python-magic not available, use mimetypes as fallback
        guessed_type, _ = mimetypes.guess_type(file.name)
        if guessed_type and guessed_type not in valid_mime_types:
            raise ValidationError(
                f"Invalid file type based on filename: {guessed_type}"
            )

    # Check for potentially malicious filename patterns
    dangerous_patterns = [
        r"\.\./",  # Directory traversal
        r'[<>:"|?*]',  # Invalid filename characters
        r"\x00",  # Null bytes
        r"\.exe$|\.bat$|\.cmd$|\.com$|\.scr$|\.pif$",  # Executable extensions
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, file.name, re.IGNORECASE):
            raise ValidationError(
                "⛔ Invalid filename. "
                "Please rename your file to remove special characters, directory paths, or executable extensions. "
                "Use only letters, numbers, hyphens, and underscores."
            )

    # Validate filename length
    if len(file.name) > 255:
        raise ValidationError(
            f"📏 Filename is too long ({len(file.name)} characters). "
            f"Maximum allowed is 255 characters. Please rename your file to something shorter."
        )

    # Basic content validation - check for text file characteristics
    try:
        file.seek(0)
        sample_content = file.read(512).decode("utf-8", errors="ignore")
        file.seek(0)

        # Check for potentially malicious content patterns
        malicious_patterns = [
            r"<script[^>]*>",  # JavaScript injection
            r"<iframe[^>]*>",  # Iframe injection
            r"(?i)exec\s*\(",  # Code execution attempts
            r"(?i)eval\s*\(",  # Eval attempts
            r"(?i)system\s*\(",  # System command execution
            r"(?i)shell_exec\s*\(",  # Shell execution
            r"%[0-9a-fA-F]{2}%[0-9a-fA-F]{2}",  # URL encoding (potential obfuscation)
            r"\\x[0-9a-fA-F]{2}",  # Hex encoding (potential obfuscation)
        ]

        for pattern in malicious_patterns:
            if re.search(pattern, sample_content, re.IGNORECASE):
                raise ValidationError(
                    "🔒 Security check failed. "
                    "File appears to contain code or script content. "
                    "Please upload a clean database log file without embedded scripts."
                )

        # Check for excessive binary data (should be mostly text)
        printable_chars = sum(
            1 for c in sample_content if c.isprintable() or c.isspace()
        )
        if len(sample_content) > 0 and (printable_chars / len(sample_content)) < 0.8:
            binary_percent = (
                (len(sample_content) - printable_chars) / len(sample_content)
            ) * 100
            raise ValidationError(
                f"📊 File appears to be binary ({binary_percent:.0f}% non-text). "
                f"Database logs should be plain text. "
                f"If this is a compressed file, please extract it first."
            )

        # Check if content looks like a log file (contains timestamps, common log patterns)
        log_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # Date pattern
            r"\d{2}:\d{2}:\d{2}",  # Time pattern
            r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)",  # SQL keywords
            r"(INFO|WARN|ERROR|DEBUG)",  # Log levels
            r"mysql-slow",  # MySQL slow log identifier
            r"mysqld",  # MySQL daemon references
            r"Time:\s*\d+",  # Query time references
        ]

        has_log_content = any(
            re.search(pattern, sample_content, re.IGNORECASE)
            for pattern in log_patterns
        )
        if not has_log_content and len(sample_content.strip()) > 0:
            raise ValidationError(
                "📋 File doesn't look like a database log. "
                "Expected to find SQL queries, timestamps, or log entries. "
                "Make sure you're uploading a MySQL slow query log or general query log."
            )

        # Additional validation: check for minimum file size
        if file.size < 10:  # Files smaller than 10 bytes are suspicious
            raise ValidationError(
                "📉 File is suspiciously small (< 10 bytes). "
                "Please check that you've uploaded the correct file with actual log content."
            )

        # Check for excessive line length (potential attack vector)
        lines = sample_content.split("\n")
        max_line_length = max(len(line) for line in lines) if lines else 0
        if max_line_length > 5000:  # Reasonable limit for log lines
            raise ValidationError(
                "📏 File contains unusually long lines (> 5000 characters). "
                "This doesn't look like a standard log file. "
                "Please verify the file format and try again."
            )

    except UnicodeDecodeError:
        raise ValidationError(
            "🔤 File encoding error. "
            "The file appears to use a non-UTF-8 encoding or contains binary data. "
            "Please save your log file as UTF-8 text and try again."
        )


def validate_sql_query(sql_text):
    """
    Enhanced validator function to ensure the SQL query is valid, parseable, and secure.

    Args:
        sql_text (str): The SQL query text to validate.

    Raises:
        forms.ValidationError: If the SQL query is invalid or potentially malicious.
    """
    if not sql_text or not sql_text.strip():
        raise ValidationError(
            "✍️ No SQL query provided. "
            "Please enter a SQL query in the text area above to analyze."
        )

    # Length validation
    if len(sql_text) > 10000:  # 10KB limit
        char_count = len(sql_text)
        raise ValidationError(
            f"📝 Query is too long ({char_count:,} characters). "
            f"Maximum allowed is 10,000 characters. "
            f"Try breaking complex queries into smaller parts for analysis."
        )

    # Basic security checks - block dangerous patterns
    dangerous_patterns = [
        r"(?i)\b(exec|execute|sp_executesql)\b",  # Stored procedure execution
        r"(?i)\b(xp_cmdshell|xp_regread|xp_regwrite)\b",  # System commands
        r"(?i)\b(alter\s+table|drop\s+table|drop\s+database|truncate)\b",  # DDL operations
        r"(?i)\b(insert\s+into|update\s+|delete\s+from)\b",  # DML operations
        r"(?i)\b(create\s+|drop\s+|alter\s+)\b",  # Schema modifications
        r"(?i)\b(grant\s+|revoke\s+)\b",  # Permission changes
        r"(?i)(\-\-|\#|\/\*)",  # Comment injection attempts
        r"(?i)\b(union\s+(?:all\s+)?select)\b",  # Union-based injection
        r"(?i)\b(information_schema|sys\.|pg_catalog)\b",  # System schema access
        r"(?i)\b(char|ascii|substring|mid|left|right)\s*\(",  # String manipulation functions
        r"(?i)\b(sleep|waitfor|benchmark)\s*\(",  # Time-based attacks
        r"(?i)\b(load_file|into\s+outfile|into\s+dumpfile)\b",  # File operations
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, sql_text):
            raise ValidationError(
                "🚫 Query contains restricted operations. "
                "For security, we only analyze SELECT queries. "
                "Data modification (INSERT, UPDATE, DELETE) and schema changes (CREATE, DROP, ALTER) are not allowed."
            )

    # Check for excessive semicolons (potential multi-statement injection)
    semicolon_count = sql_text.count(";")
    if semicolon_count > 1:
        raise ValidationError(
            "⚠️ Multiple statements detected. "
            "Please submit one query at a time for analysis. "
            "For batch analysis, use the 'Batch Query' feature instead."
        )

    try:
        # Try to parse the SQL query
        parsed = sqlparse.parse(sql_text)
        if not parsed:
            raise ValidationError(
                "❓ Unable to parse your SQL query. "
                "Please check your syntax for errors like unmatched quotes, missing parentheses, or typos."
            )

        # Check if it contains at least one statement
        if not any(stmt.tokens for stmt in parsed):
            raise ValidationError(
                "🔍 No valid SQL statement found. "
                "Please ensure you've entered a complete SQL query with proper syntax."
            )

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
                if first_token.value.upper() != "SELECT":
                    statement_type = first_token.value.upper()
                    raise ValidationError(
                        f"🛑 {statement_type} statements are not supported. "
                        f"This tool only analyzes read-only SELECT queries for performance optimization. "
                        f"Data modifications are not allowed for security reasons."
                    )
            elif first_token and first_token.ttype is sqlparse.tokens.Keyword:
                if first_token.value.upper() not in ["SELECT", "WITH"]:
                    statement_type = first_token.value.upper()
                    raise ValidationError(
                        f"🛑 {statement_type} statements are not supported. "
                        f"Only SELECT queries (including WITH/CTE) can be analyzed. "
                        f"Try submitting a SELECT query to get performance insights."
                    )

    except ValidationError:
        raise
    except Exception as e:
        error_msg = str(e)
        raise ValidationError(
            f"⚠️ SQL syntax error: {error_msg}\n\n"
            f"💡 Common issues: Check for missing commas, unmatched parentheses, "
            f"incorrect table/column names, or reserved keyword usage."
        )
