from django import forms

class UploadLogForm(forms.Form):
    LOG_TYPE_CHOICES = [
        ('slow', 'Slow Query Log'),
        ('general', 'General Query Log'),
    ]
    log_type = forms.ChoiceField(choices=LOG_TYPE_CHOICES, label='Select Log Type')
    log_file = forms.FileField(label='Choose Log File')
