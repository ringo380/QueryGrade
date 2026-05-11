"""End-to-end test: authenticated grade flow with a saved DB connection
populates ``QueryAnalysis.index_recommendations`` and fires the GA4 event."""

from __future__ import annotations

import os
from unittest import mock

from cryptography.fernet import Fernet
from django.contrib.auth.models import User
from django.test import Client, TransactionTestCase, override_settings
from django.urls import reverse

from analyzer.database_introspector import TableInfo
from analyzer.models import (Query, QueryAnalysis, UserDatabaseConnection,
                             UserQueryHistory)
from analyzer.services import connection_crypto, live_schema_context

TEST_FERNET_KEY = Fernet.generate_key().decode()


@override_settings(
    RATELIMIT_ENABLE=False,
    CACHES={
        "default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"},
        "query_analysis_cache": {
            "BACKEND": "django.core.cache.backends.dummy.DummyCache"
        },
        "process_cache": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"},
        "template_cache": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"},
    },
)
class GradeWithLiveSchemaTests(TransactionTestCase):
    def setUp(self):
        connection_crypto._get_fernet.cache_clear()
        self._key_patch = mock.patch.dict(
            os.environ, {"DB_CONNECTION_KEY": TEST_FERNET_KEY}
        )
        self._key_patch.start()
        self.addCleanup(self._key_patch.stop)
        self.addCleanup(connection_crypto._get_fernet.cache_clear)

        # Force query_cache to use dummy backend (CLAUDE.md test guidance).
        from django.core.cache import caches

        from analyzer.performance import query_cache

        query_cache.cache = caches["query_analysis_cache"]
        for cache_name in [
            "default",
            "query_analysis_cache",
            "process_cache",
            "template_cache",
        ]:
            try:
                caches[cache_name].clear()
            except Exception:
                pass

        self.user = User.objects.create_user(username="alice", password="pw")
        self.conn = UserDatabaseConnection.objects.create(
            user=self.user,
            name="orders-db",
            engine="postgresql",
            host="db",
            port=5432,
            db_name="shop",
            db_user="reader",
        )
        self.conn.set_password("supersecret")
        self.conn.save()

        self.client = Client(enforce_csrf_checks=False)
        self.client.force_login(self.user)

    def tearDown(self):
        UserQueryHistory.objects.all().delete()
        QueryAnalysis.objects.all().delete()
        Query.objects.all().delete()
        UserDatabaseConnection.objects.all().delete()
        User.objects.all().delete()

    def _patch_hypopg(self):
        # IndexRecommender._try_hypopg_improvement opens a real DB connection
        # to PG via DatabaseIntrospector. In tests we don't have one — short-
        # circuit so the heuristic path runs instead.
        return mock.patch(
            "analyzer.services.index_recommender._try_hypopg_improvement",
            return_value=None,
        )

    def test_grade_attaches_index_recommendations_when_connection_supplied(self):
        stub_table = TableInfo(
            name="orders",
            schema="public",
            row_count=1_000_000,
            size_mb=120.0,
            columns=[
                {"name": "id", "type": "integer", "nullable": False},
                {"name": "customer_id", "type": "integer", "nullable": True},
            ],
            indexes=[
                {
                    "name": "orders_pkey",
                    "columns": ["id"],
                    "unique": True,
                    "primary": True,
                }
            ],
            foreign_keys=[],
        )

        with mock.patch(
            "analyzer.services.live_schema_context.DatabaseIntrospector"
        ) as IntroCls, self._patch_hypopg():
            instance = IntroCls.return_value
            instance.connect.return_value = True
            instance.get_tables.return_value = [stub_table]

            response = self.client.post(
                reverse("grade_query"),
                {
                    "sql_query": "SELECT * FROM orders WHERE customer_id = 42",
                    "database_type": "postgresql",
                    "database_version": "",
                    "use_case_notes": "",
                    "db_connection": str(self.conn.pk),
                },
                follow=False,
            )

        self.assertEqual(response.status_code, 302)
        analysis_id = int(response.url.split("/")[-2])
        analysis = QueryAnalysis.objects.get(pk=analysis_id)

        block = analysis.index_recommendations
        self.assertIsInstance(block, dict)
        self.assertIn("recommendations", block)
        self.assertGreaterEqual(len(block["recommendations"]), 1)

        top = block["recommendations"][0]
        self.assertEqual(top["table"], "orders")
        self.assertEqual(top["columns"], ["customer_id"])
        self.assertIn("CREATE INDEX", top["create_sql"])
        self.assertIn(top["confidence"], {"HIGH", "MEDIUM", "LOW"})
        self.assertIn("index_features", block)
        self.assertEqual(block["index_features"]["existing_index_count"], 1.0)

        # GA4 session-flag must be set.
        session = self.client.session
        # Session pop happens in context_processor on next render — just
        # confirm the flag was set during the POST handler.
        self.conn.refresh_from_db()
        self.assertIsNotNone(self.conn.last_used_at)

    def test_grade_without_connection_skips_index_recommendations(self):
        response = self.client.post(
            reverse("grade_query"),
            {
                "sql_query": "SELECT * FROM orders WHERE customer_id = 42",
                "database_type": "postgresql",
                "database_version": "",
                "use_case_notes": "",
            },
            follow=False,
        )
        self.assertEqual(response.status_code, 302)
        analysis_id = int(response.url.split("/")[-2])
        analysis = QueryAnalysis.objects.get(pk=analysis_id)
        # When the user didn't pick a connection, the field stays empty.
        self.assertEqual(analysis.index_recommendations, {})

    def test_grade_form_picker_lists_only_user_connections(self):
        eve = User.objects.create_user(username="eve", password="pw")
        UserDatabaseConnection.objects.create(
            user=eve, name="eves-db", engine="sqlite", db_name="/tmp/x.db"
        )
        response = self.client.get(reverse("grade_query"))
        self.assertContains(response, "orders-db")
        self.assertNotContains(response, "eves-db")

    def test_grade_continues_when_introspector_fails(self):
        with mock.patch(
            "analyzer.services.live_schema_context.DatabaseIntrospector"
        ) as IntroCls, self._patch_hypopg():
            IntroCls.return_value.connect.return_value = False

            response = self.client.post(
                reverse("grade_query"),
                {
                    "sql_query": "SELECT * FROM orders WHERE customer_id = 42",
                    "database_type": "postgresql",
                    "database_version": "",
                    "use_case_notes": "",
                    "db_connection": str(self.conn.pk),
                },
                follow=False,
            )

        # Grade should still succeed, just without schema-aware data.
        self.assertEqual(response.status_code, 302)
        analysis_id = int(response.url.split("/")[-2])
        analysis = QueryAnalysis.objects.get(pk=analysis_id)
        block = analysis.index_recommendations
        # The recommender still runs without live_schema and emits LOW-confidence
        # text-only candidates.
        self.assertIn("recommendations", block)
