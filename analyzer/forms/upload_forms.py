from django import forms

from .validators import validate_log_file


class UploadLogForm(forms.Form):
    """
    Form for uploading MySQL log files.
    """

    LOG_TYPE_CHOICES = [
        ("slow", "Slow Query Log"),
        ("general", "General Query Log"),
    ]
    log_type = forms.ChoiceField(choices=LOG_TYPE_CHOICES, label="Select Log Type")
    log_file = forms.FileField(label="Choose Log File", validators=[validate_log_file])
    use_async = forms.BooleanField(
        required=False,
        initial=False,
        label="Process in background",
        help_text="Enable for large files (recommended for files > 10MB)",
    )
