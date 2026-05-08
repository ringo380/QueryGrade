"""Form for creating/editing persisted ``UserDatabaseConnection`` rows."""

from __future__ import annotations

from django import forms

from analyzer.models import UserDatabaseConnection


class SavedConnectionForm(forms.ModelForm):
    """ModelForm for UserDatabaseConnection.

    The model stores ``password_encrypted``, but the form exposes a plain
    ``password`` field; ``save()`` writes the ciphertext via
    ``UserDatabaseConnection.set_password``.
    """

    password = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=False),
        help_text=(
            "Leave blank to keep the existing password when editing. "
            "SQLite ignores this field."
        ),
    )

    class Meta:
        model = UserDatabaseConnection
        fields = ["name", "engine", "host", "port", "db_name", "db_user", "schema"]
        widgets = {
            "host": forms.TextInput(attrs={"placeholder": "localhost"}),
            "port": forms.NumberInput(attrs={"placeholder": "5432 / 3306"}),
            "db_name": forms.TextInput(
                attrs={"placeholder": "database name (or /path/to/db.sqlite)"}
            ),
            "schema": forms.TextInput(attrs={"placeholder": "public"}),
        }

    def clean(self):
        data = super().clean()
        engine = data.get("engine")
        if engine in {"postgresql", "mysql"}:
            if not data.get("db_user"):
                raise forms.ValidationError(
                    "Username is required for PostgreSQL/MySQL connections."
                )
        if engine == "sqlite" and not data.get("db_name"):
            raise forms.ValidationError(
                "SQLite database file path is required (in 'Database name')."
            )
        return data

    def save(self, commit: bool = True):
        instance: UserDatabaseConnection = super().save(commit=False)
        plaintext = self.cleaned_data.get("password") or ""
        if plaintext:
            instance.set_password(plaintext)
        elif not instance.pk:
            instance.set_password("")
        if commit:
            instance.save()
        return instance
