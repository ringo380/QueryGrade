from django import forms


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
