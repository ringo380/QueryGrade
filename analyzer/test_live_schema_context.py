"""Tests for LiveSchemaContext caching, dataclass round-trip, and the
``DatabaseStatisticsManager._fetch_live_statistics`` integration."""

from __future__ import annotations

import os
from unittest import mock

from cryptography.fernet import Fernet
from django.contrib.auth.models import User
from django.test import TestCase, override_settings

from analyzer.models import UserDatabaseConnection
from analyzer.services import connection_crypto, live_schema_context
from analyzer.services.live_schema_context import (IndexSnapshot,
                                                   LiveSchemaContext,
                                                   TableSnapshot,
                                                   build_live_context,
                                                   schema_fingerprint)

TEST_FERNET_KEY = Fernet.generate_key().decode()


@override_settings(
    CACHES={
        "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
        "query_analysis_cache": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "live_schema_test",
        },
    }
)
class LiveSchemaContextTests(TestCase):
    def setUp(self):
        connection_crypto._get_fernet.cache_clear()
        self._patch = mock.patch.dict(
            os.environ, {"DB_CONNECTION_KEY": TEST_FERNET_KEY}
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(connection_crypto._get_fernet.cache_clear)

        self.user = User.objects.create_user(username="u", password="p")
        self.conn = UserDatabaseConnection.objects.create(
            user=self.user,
            name="t",
            engine="postgresql",
            host="db",
            port=5432,
            db_name="orders",
            db_user="reader",
        )
        live_schema_context._cache().clear()

    def _fake_table(self, name="orders", row_count=1_000_000, has_idx=False):
        return TableSnapshot(
            name=name,
            schema="public",
            row_count=row_count,
            size_mb=12.5,
            columns=[
                {"name": "id", "type": "integer", "nullable": False},
                {"name": "customer_id", "type": "integer", "nullable": True},
            ],
            indexes=(
                [
                    IndexSnapshot(
                        name=f"{name}_pkey", columns=["id"], unique=True, primary=True
                    )
                ]
                + (
                    [IndexSnapshot(name=f"{name}_cust_idx", columns=["customer_id"])]
                    if has_idx
                    else []
                )
            ),
            foreign_keys=[],
        )

    def test_to_from_cache_dict_round_trip(self):
        ctx = LiveSchemaContext(
            engine="postgresql",
            database="orders",
            schema="public",
            fetched_at="2026-05-07T00:00:00",
            tables={"orders": self._fake_table()},
        )
        data = ctx.to_cache_dict()
        ctx2 = LiveSchemaContext.from_cache_dict(data)
        self.assertEqual(ctx2.engine, "postgresql")
        self.assertIn("orders", ctx2.tables)
        self.assertEqual(ctx2.tables["orders"].row_count, 1_000_000)
        self.assertEqual(ctx2.tables["orders"].indexes[0].columns, ["id"])
        self.assertTrue(ctx2.tables["orders"].indexes[0].primary)

    def test_get_table_case_insensitive(self):
        ctx = LiveSchemaContext(
            engine="postgresql",
            database="orders",
            tables={"Orders": self._fake_table(name="Orders")},
        )
        self.assertIsNotNone(ctx.get_table("orders"))
        self.assertIsNotNone(ctx.get_table("ORDERS"))
        self.assertIsNone(ctx.get_table("missing"))

    def test_schema_fingerprint_stable(self):
        ctx_a = LiveSchemaContext(
            engine="postgresql", database="d", tables={"orders": self._fake_table()}
        )
        ctx_b = LiveSchemaContext(
            engine="postgresql", database="d", tables={"orders": self._fake_table()}
        )
        self.assertEqual(schema_fingerprint(ctx_a), schema_fingerprint(ctx_b))
        ctx_c = LiveSchemaContext(
            engine="postgresql",
            database="d",
            tables={"orders": self._fake_table(has_idx=True)},
        )
        self.assertNotEqual(schema_fingerprint(ctx_a), schema_fingerprint(ctx_c))

    def test_hydrate_statistics_manager(self):
        from analyzer.ml.integration.database_stats import \
            DatabaseStatisticsManager

        ctx = LiveSchemaContext(
            engine="postgresql",
            database="d",
            tables={"orders": self._fake_table(has_idx=True)},
        )
        mgr = DatabaseStatisticsManager()
        ctx.hydrate_statistics_manager(mgr)

        self.assertIn("orders", mgr.table_stats)
        self.assertEqual(mgr.table_stats["orders"].row_count, 1_000_000)
        self.assertEqual(len(mgr.index_stats["orders"]), 2)
        unique_indexes = [i for i in mgr.index_stats["orders"] if i.unique]
        self.assertEqual(len(unique_indexes), 1)
        self.assertEqual(unique_indexes[0].selectivity, 1.0)
        self.assertIn("customer_id", mgr.column_stats["orders"])

    def test_fetch_live_statistics_accepts_context(self):
        from analyzer.ml.integration.database_stats import \
            DatabaseStatisticsManager

        ctx = LiveSchemaContext(
            engine="postgresql", database="d", tables={"orders": self._fake_table()}
        )
        mgr = DatabaseStatisticsManager()
        mgr._fetch_live_statistics(ctx)
        self.assertIn("orders", mgr.table_stats)

    def test_build_live_context_caches_after_fetch(self):
        fake_ctx = LiveSchemaContext(
            engine="postgresql",
            database="orders",
            tables={"orders": self._fake_table()},
        )
        # First call: simulate a successful introspection by short-circuiting
        # the introspector with a stub that returns canned tables.
        from analyzer.database_introspector import TableInfo

        stub_table = TableInfo(
            name="orders",
            schema="public",
            row_count=1_000_000,
            size_mb=12.5,
            columns=[{"name": "id", "type": "integer", "nullable": False}],
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
        ) as IntroCls:
            instance = IntroCls.return_value
            instance.connect.return_value = True
            instance.get_tables.return_value = [stub_table]

            ctx = build_live_context(self.conn)
            self.assertIsNotNone(ctx)
            self.assertIn("orders", ctx.tables)

            # Second call must hit cache (no new connect).
            instance.connect.reset_mock()
            ctx2 = build_live_context(self.conn)
            self.assertIsNotNone(ctx2)
            instance.connect.assert_not_called()

    def test_build_live_context_returns_none_on_connect_failure(self):
        with mock.patch(
            "analyzer.services.live_schema_context.DatabaseIntrospector"
        ) as IntroCls:
            IntroCls.return_value.connect.return_value = False
            self.assertIsNone(build_live_context(self.conn))
