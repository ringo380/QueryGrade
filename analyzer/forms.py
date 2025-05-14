from django import forms

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

def validate_log_file(file):
    """
    Validator function to ensure the uploaded file is a valid log file.

    Args:
        file: The uploaded file.

    Raises:
        forms.ValidationError: If the file is not a valid log file.
    """
    valid_mime_types = ['text/plain']
    valid_extensions = ['.log']

    if not file.content_type in valid_mime_types:
        raise forms.ValidationError("Unsupported file type.")

    if not any(file.name.lower().endswith(ext) for ext in valid_extensions):
        raise forms.ValidationError("Unsupported file extension.")
