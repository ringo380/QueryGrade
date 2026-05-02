"""Idempotently seed an admin from env vars on first deploy.

Reads SUPERUSER_EMAIL and SUPERUSER_PASSWORD. No-ops if either is unset
or if a superuser already exists. Safe to call on every boot.
"""

import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        "Create a superuser from SUPERUSER_EMAIL / SUPERUSER_PASSWORD if none exists."
    )

    def handle(self, *args, **options):
        User = get_user_model()
        email = os.environ.get("SUPERUSER_EMAIL", "").strip()
        password = os.environ.get("SUPERUSER_PASSWORD", "").strip()
        username = (
            os.environ.get("SUPERUSER_USERNAME", "").strip() or email.split("@")[0]
            if email
            else ""
        )

        if not email or not password:
            self.stdout.write(
                "init_superuser: SUPERUSER_EMAIL/SUPERUSER_PASSWORD not set, skipping."
            )
            return

        if User.objects.filter(is_superuser=True).exists():
            self.stdout.write("init_superuser: superuser already exists, skipping.")
            return

        User.objects.create_superuser(username=username, email=email, password=password)
        self.stdout.write(
            self.style.SUCCESS(f"init_superuser: created {username} <{email}>")
        )
