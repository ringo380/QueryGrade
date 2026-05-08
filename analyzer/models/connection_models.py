"""Persisted database connection profiles for live schema introspection.

Used by the IndexRecommender pipeline to introspect a user's database for
schema-aware index suggestions (issue #7). Passwords are stored encrypted
via ``analyzer.services.connection_crypto``.
"""

from __future__ import annotations

from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

from analyzer.services.connection_crypto import decrypt, encrypt


class UserDatabaseConnection(models.Model):
    """A user's saved database-connection profile."""

    ENGINE_CHOICES = [
        ("postgresql", "PostgreSQL"),
        ("mysql", "MySQL"),
        ("sqlite", "SQLite"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="db_connections"
    )
    name = models.CharField(
        max_length=100,
        help_text="Friendly label, e.g. 'staging-orders'",
    )
    engine = models.CharField(max_length=20, choices=ENGINE_CHOICES)
    host = models.CharField(max_length=255, blank=True, default="localhost")
    port = models.PositiveIntegerField(null=True, blank=True)
    db_name = models.CharField(
        max_length=255,
        help_text="Database name, or absolute file path for SQLite",
    )
    db_user = models.CharField(max_length=100, blank=True, default="")
    password_encrypted = models.TextField(blank=True, default="")
    schema = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="PostgreSQL schema; ignored for MySQL/SQLite",
    )

    created_at = models.DateTimeField(default=timezone.now)
    last_used_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [("user", "name")]
        ordering = ["-last_used_at", "-created_at"]
        indexes = [models.Index(fields=["user", "-last_used_at"])]

    def __str__(self) -> str:
        return f"{self.user.username}:{self.name} ({self.engine})"

    def set_password(self, plaintext: str) -> None:
        self.password_encrypted = encrypt(plaintext) if plaintext else ""

    def get_password(self) -> str:
        return decrypt(self.password_encrypted) if self.password_encrypted else ""

    def to_connection_config(self) -> dict:
        """Shape expected by ``analyzer.database_introspector.DatabaseIntrospector``."""
        default_port = {"postgresql": 5432, "mysql": 3306}.get(self.engine)
        return {
            "engine": self.engine,
            "name": self.db_name,
            "host": self.host or "localhost",
            "port": str(self.port or default_port or ""),
            "user": self.db_user,
            "password": self.get_password(),
            "schema": self.schema or ("public" if self.engine == "postgresql" else ""),
        }

    def touch(self) -> None:
        """Mark this connection as recently used (for ordering in pickers)."""
        self.last_used_at = timezone.now()
        self.save(update_fields=["last_used_at"])
