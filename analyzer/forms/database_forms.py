from django import forms


class DatabaseConnectionForm(forms.Form):
    """Form for configuring database connections for introspection."""

    ENGINE_CHOICES = [
        ("mysql", "MySQL"),
        ("postgresql", "PostgreSQL"),
        ("sqlite", "SQLite"),
    ]

    engine = forms.ChoiceField(
        choices=ENGINE_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
        label="Database Engine",
        help_text="Select your database type",
    )

    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "database_name"}
        ),
        label="Database Name",
        help_text="Name of the database to connect to",
    )

    host = forms.CharField(
        max_length=100,
        required=False,
        initial="localhost",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "localhost"}
        ),
        label="Host",
        help_text="Database server hostname (leave empty for SQLite)",
    )

    port = forms.CharField(
        max_length=10,
        required=False,
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "3306 (MySQL) or 5432 (PostgreSQL)",
            }
        ),
        label="Port",
        help_text="Database server port (leave empty for default)",
    )

    user = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "username"}
        ),
        label="Username",
        help_text="Database username (leave empty for SQLite)",
    )

    password = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.PasswordInput(
            attrs={"class": "form-control", "placeholder": "password"}
        ),
        label="Password",
        help_text="Database password (leave empty for SQLite)",
    )

    schema = forms.CharField(
        max_length=100,
        required=False,
        initial="public",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "public"}
        ),
        label="Schema",
        help_text="Database schema (PostgreSQL only, optional)",
    )

    def clean(self):
        """Validate database connection parameters."""
        cleaned_data = super().clean()
        engine = cleaned_data.get("engine")
        name = cleaned_data.get("name")
        host = cleaned_data.get("host")
        user = cleaned_data.get("user")
        password = cleaned_data.get("password")

        if engine == "sqlite":
            # SQLite only needs the database name (file path)
            if not name:
                raise forms.ValidationError(
                    "📁 Database file path is required for SQLite. "
                    "Example: /path/to/database.db or database.sqlite"
                )
        else:
            # MySQL and PostgreSQL need host, user, and typically password
            if not host:
                cleaned_data["host"] = "localhost"
            if not user:
                db_type = engine.upper() if engine else "database"
                raise forms.ValidationError(
                    f"👤 Username is required to connect to {db_type}. "
                    f"Please enter your database username."
                )
            # Password is often required but can be empty in some configurations

        return cleaned_data

    def get_connection_config(self):
        """Return configuration dict for DatabaseIntrospector."""
        data = self.cleaned_data
        return {
            "engine": data["engine"],
            "name": data["name"],
            "host": data.get("host", "localhost"),
            "port": data.get("port", ""),
            "user": data.get("user", ""),
            "password": data.get("password", ""),
            "schema": data.get("schema", "public"),
        }
