"""End-to-end test: authenticated grade flow with a saved DB connection
populates ``QueryAnalysis.schema_insights`` and renders the insights panel.

Mirrors ``test_grade_with_live_schema.py`` — mocks the introspector so no real
database or HypoPG is touched (the HypoPG path must always be mocked)."""

from __future__ import annotations

import os
from unittest import mock

from cryptography.fernet import Fernet
from django.contrib.auth.models import User
from django.test import Client, TransactionTestCase, override_settings
from django.urls import reverse

from analyzer.database_introspector import TableInfo
from analyzer.models import (
    Query,
    QueryAnalysis,
    UserDatabaseConnection,
    UserQueryHistory,
)
from analyzer.services import connection_crypto

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
class GradeWithSchemaInsightsTests(TransactionTestCase):
    def setUp(self):
        connection_crypto._get_fernet.cache_clear()
        self._key_patch = mock.patch.dict(
            os.environ, {"DB_CONNECTION_KEY": TEST_FERNET_KEY}
        )
        self._key_patch.start()
        self.addCleanup(self._key_patch.stop)
        self.addCleanup(connection_crypto._get_fernet.cache_clear)

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

        self.user = User.objects.create_user(username="bob", password="pw")
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
        return mock.patch(
            "analyzer.services.index_recommender._try_hypopg_improvement",
            return_value=None,
        )

    def _orders_customers(self):
        """orders.customer_id (nullable FK) -> customers.id; one-to-many."""
        orders = TableInfo(
            name="orders",
            schema="public",
            row_count=1_000_000,
            size_mb=120.0,
            last_analyzed=None,  # never analyzed -> stale_stats info insight
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
            foreign_keys=[
                {
                    "name": "fk_cust",
                    "columns": ["customer_id"],
                    "referenced_table": "customers",
                    "referenced_columns": ["id"],
                }
            ],
        )
        customers = TableInfo(
            name="customers",
            schema="public",
            row_count=10_000,
            last_analyzed=None,
            columns=[{"name": "id", "type": "integer", "nullable": False}],
            indexes=[
                {
                    "name": "customers_pkey",
                    "columns": ["id"],
                    "unique": True,
                    "primary": True,
                }
            ],
            foreign_keys=[],
        )
        return [orders, customers]

    def test_grade_attaches_schema_insights(self):
        sql = "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id"
        with mock.patch(
            "analyzer.services.live_schema_context.DatabaseIntrospector"
        ) as IntroCls, self._patch_hypopg():
            instance = IntroCls.return_value
            instance.connect.return_value = True
            instance.get_tables.return_value = self._orders_customers()

            response = self.client.post(
                reverse("grade_query"),
                {
                    "sql_query": sql,
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

        block = analysis.schema_insights
        self.assertIsInstance(block, dict)
        self.assertIn("insights", block)
        types = {i["type"] for i in block["insights"]}
        # Nullable FK + one-to-many fan-out + never-analyzed stats all fire.
        self.assertIn("nullable_fk", types)
        self.assertIn("fk_fanout", types)
        self.assertIn("stale_stats", types)

        # Panel renders on the results page.
        page = self.client.get(reverse("grade_results", args=[analysis_id]))
        self.assertContains(page, 'id="schema-insights"')

    def test_grade_without_connection_leaves_schema_insights_empty(self):
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
        self.assertEqual(analysis.schema_insights, {})

    def test_advisor_failure_does_not_break_grade(self):
        with mock.patch(
            "analyzer.services.live_schema_context.DatabaseIntrospector"
        ) as IntroCls, self._patch_hypopg(), mock.patch(
            "analyzer.services.schema_advisor.SchemaAdvisor.analyze",
            side_effect=RuntimeError("boom"),
        ):
            instance = IntroCls.return_value
            instance.connect.return_value = True
            instance.get_tables.return_value = self._orders_customers()

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
        # Grade still succeeds; index recs still saved despite advisor failure.
        self.assertEqual(response.status_code, 302)
        analysis_id = int(response.url.split("/")[-2])
        analysis = QueryAnalysis.objects.get(pk=analysis_id)
        self.assertEqual(analysis.schema_insights, {})
        self.assertIn("recommendations", analysis.index_recommendations)
