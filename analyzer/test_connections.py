"""Tests for connection crypto + UserDatabaseConnection CRUD (issue #7 Layer 1)."""

from __future__ import annotations

import os
from unittest import mock

from cryptography.fernet import Fernet
from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from analyzer.models import UserDatabaseConnection
from analyzer.services import connection_crypto

# A fixed key shared by every test in this module. We can't rely on the
# DEBUG fallback because Django runs tests with DEBUG=False.
TEST_FERNET_KEY = Fernet.generate_key().decode()


def _patch_key():
    return mock.patch.dict(os.environ, {"DB_CONNECTION_KEY": TEST_FERNET_KEY})


class ConnectionCryptoTests(TestCase):
    def setUp(self):
        connection_crypto._get_fernet.cache_clear()
        self._key_patch = _patch_key()
        self._key_patch.start()
        self.addCleanup(self._key_patch.stop)
        self.addCleanup(connection_crypto._get_fernet.cache_clear)

    def test_round_trip(self):
        token = connection_crypto.encrypt("hunter2")
        self.assertNotEqual(token, "hunter2")
        self.assertEqual(connection_crypto.decrypt(token), "hunter2")

    def test_empty_inputs(self):
        self.assertEqual(connection_crypto.encrypt(""), "")
        self.assertEqual(connection_crypto.decrypt(""), "")

    def test_changing_key_breaks_decrypt(self):
        token = connection_crypto.encrypt("secret")
        new_key = Fernet.generate_key().decode()
        with mock.patch.dict(os.environ, {"DB_CONNECTION_KEY": new_key}):
            connection_crypto._get_fernet.cache_clear()
            with self.assertRaises(connection_crypto.ConnectionCryptoError):
                connection_crypto.decrypt(token)

    def test_invalid_key_raises(self):
        self._key_patch.stop()  # remove valid override for this test
        with mock.patch.dict(os.environ, {"DB_CONNECTION_KEY": "not-a-fernet-key"}):
            connection_crypto._get_fernet.cache_clear()
            with self.assertRaises(connection_crypto.ConnectionCryptoError):
                connection_crypto.encrypt("x")
        self._key_patch.start()  # restore for cleanup ordering


class UserDatabaseConnectionModelTests(TestCase):
    def setUp(self):
        connection_crypto._get_fernet.cache_clear()
        self._key_patch = _patch_key()
        self._key_patch.start()
        self.addCleanup(self._key_patch.stop)
        self.addCleanup(connection_crypto._get_fernet.cache_clear)
        self.user = User.objects.create_user(username="alice", password="pw")

    def test_password_round_trip_via_model(self):
        c = UserDatabaseConnection(
            user=self.user,
            name="staging",
            engine="postgresql",
            host="db.example.com",
            port=5432,
            db_name="orders",
            db_user="reader",
        )
        c.set_password("secret-pw")
        c.save()

        c.refresh_from_db()
        self.assertNotEqual(c.password_encrypted, "secret-pw")
        self.assertEqual(c.get_password(), "secret-pw")

    def test_to_connection_config_postgres_defaults(self):
        c = UserDatabaseConnection.objects.create(
            user=self.user,
            name="pg",
            engine="postgresql",
            db_name="orders",
            db_user="reader",
        )
        c.set_password("p")
        cfg = c.to_connection_config()
        self.assertEqual(cfg["engine"], "postgresql")
        self.assertEqual(cfg["port"], "5432")
        self.assertEqual(cfg["password"], "p")
        self.assertEqual(cfg["schema"], "public")

    def test_to_connection_config_sqlite_no_schema(self):
        c = UserDatabaseConnection.objects.create(
            user=self.user,
            name="local",
            engine="sqlite",
            db_name="/tmp/test.db",
        )
        cfg = c.to_connection_config()
        self.assertEqual(cfg["engine"], "sqlite")
        self.assertEqual(cfg["schema"], "")


class ConnectionViewTests(TestCase):
    def setUp(self):
        connection_crypto._get_fernet.cache_clear()
        self._key_patch = _patch_key()
        self._key_patch.start()
        self.addCleanup(self._key_patch.stop)
        self.addCleanup(connection_crypto._get_fernet.cache_clear)
        self.user = User.objects.create_user(username="bob", password="pw")
        self.client = Client()
        self.client.login(username="bob", password="pw")

    def test_list_empty(self):
        resp = self.client.get(reverse("connections_list"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "No saved connections yet")

    def test_create_persists_and_encrypts(self):
        resp = self.client.post(
            reverse("connection_create"),
            {
                "name": "prod",
                "engine": "postgresql",
                "host": "db",
                "port": 5432,
                "db_name": "orders",
                "db_user": "reader",
                "password": "supersecret",
                "schema": "public",
            },
        )
        self.assertEqual(resp.status_code, 302)
        c = UserDatabaseConnection.objects.get(user=self.user, name="prod")
        self.assertEqual(c.get_password(), "supersecret")
        self.assertNotIn("supersecret", c.password_encrypted)

    def test_other_users_connections_hidden(self):
        other = User.objects.create_user(username="eve", password="pw")
        UserDatabaseConnection.objects.create(
            user=other, name="eves-db", engine="sqlite", db_name="/tmp/x.db"
        )
        resp = self.client.get(reverse("connections_list"))
        self.assertNotContains(resp, "eves-db")

    def test_delete_requires_ownership(self):
        other = User.objects.create_user(username="eve2", password="pw")
        c = UserDatabaseConnection.objects.create(
            user=other, name="eves-db", engine="sqlite", db_name="/tmp/x.db"
        )
        resp = self.client.post(reverse("connection_delete", args=[c.id]))
        self.assertEqual(resp.status_code, 404)
        self.assertTrue(UserDatabaseConnection.objects.filter(pk=c.id).exists())

    def test_login_required(self):
        self.client.logout()
        resp = self.client.get(reverse("connections_list"))
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login/", resp.url)
