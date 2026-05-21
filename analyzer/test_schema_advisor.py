"""Unit tests for ``SchemaAdvisor`` (issue #6, scope A).

Each insight is exercised against synthetic, in-memory ``LiveSchemaContext``
fixtures — no database, no introspection. Covers the positive path, the
negative/suppressed path, and the ``skipped`` path for each of the four
insight types.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from django.test import SimpleTestCase

from analyzer.services.live_schema_context import (
    IndexSnapshot,
    LiveSchemaContext,
    TableSnapshot,
)
from analyzer.services.schema_advisor import STALE_DAYS, SchemaAdvisor


def _ctx(engine="postgresql", tables=None):
    ctx = LiveSchemaContext(engine=engine, database="test")
    for t in tables or []:
        ctx.tables[t.name] = t
    return ctx


def _types(result):
    return {i.type for i in result.insights}


class FanOutTests(SimpleTestCase):
    def _schema(self):
        # orders.customer_id -> customers.id (one-to-many, no unique on FK col)
        orders = TableSnapshot(
            name="orders",
            row_count=500_000,
            columns=[
                {"name": "id", "type": "int", "nullable": False},
                {"name": "customer_id", "type": "int", "nullable": False},
            ],
            indexes=[
                IndexSnapshot(
                    name="orders_pkey", columns=["id"], unique=True, primary=True
                )
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
        customers = TableSnapshot(
            name="customers",
            row_count=10_000,
            columns=[{"name": "id", "type": "int", "nullable": False}],
            indexes=[
                IndexSnapshot(
                    name="customers_pkey", columns=["id"], unique=True, primary=True
                )
            ],
            foreign_keys=[],
        )
        return _ctx(tables=[orders, customers])

    def test_fan_out_flagged_without_aggregation(self):
        sql = "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id"
        result = SchemaAdvisor(live_schema=self._schema()).analyze(sql)
        self.assertIn("fk_fanout", _types(result))

    def test_fan_out_suppressed_with_aggregation(self):
        sql = (
            "SELECT customers.id, COUNT(*) FROM orders "
            "JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.id"
        )
        result = SchemaAdvisor(live_schema=self._schema()).analyze(sql)
        self.assertNotIn("fk_fanout", _types(result))

    def test_fan_out_suppressed_when_fk_is_unique(self):
        schema = self._schema()
        # Make the FK column unique → one-to-one, no fan-out.
        schema.tables["orders"].indexes.append(
            IndexSnapshot(name="uq_cust", columns=["customer_id"], unique=True)
        )
        sql = "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id"
        result = SchemaAdvisor(live_schema=schema).analyze(sql)
        self.assertNotIn("fk_fanout", _types(result))


class NullableFkTests(SimpleTestCase):
    def _schema(self, nullable):
        orders = TableSnapshot(
            name="orders",
            row_count=1000,
            columns=[
                {"name": "id", "type": "int", "nullable": False},
                {"name": "customer_id", "type": "int", "nullable": nullable},
            ],
            indexes=[],
            foreign_keys=[
                {
                    "name": "fk_cust",
                    "columns": ["customer_id"],
                    "referenced_table": "customers",
                    "referenced_columns": ["id"],
                }
            ],
        )
        customers = TableSnapshot(
            name="customers",
            row_count=100,
            columns=[{"name": "id", "type": "int", "nullable": False}],
        )
        return _ctx(tables=[orders, customers])

    def test_nullable_fk_flagged(self):
        sql = "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id"
        result = SchemaAdvisor(live_schema=self._schema(nullable=True)).analyze(sql)
        self.assertIn("nullable_fk", _types(result))

    def test_non_nullable_fk_not_flagged(self):
        sql = "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id"
        result = SchemaAdvisor(live_schema=self._schema(nullable=False)).analyze(sql)
        self.assertNotIn("nullable_fk", _types(result))


class LowCardinalityTests(SimpleTestCase):
    def _schema(self, ndv):
        col = {"name": "status", "type": "varchar", "nullable": False}
        if ndv is not None:
            col["ndv"] = ndv
        users = TableSnapshot(
            name="users",
            row_count=1_000_000,
            columns=[
                {"name": "id", "type": "int", "nullable": False},
                col,
            ],
        )
        return _ctx(tables=[users])

    def test_low_cardinality_flagged(self):
        sql = "SELECT * FROM users WHERE status = 'active'"
        result = SchemaAdvisor(live_schema=self._schema(ndv=3)).analyze(sql)
        self.assertIn("low_cardinality", _types(result))

    def test_high_cardinality_not_flagged(self):
        sql = "SELECT * FROM users WHERE status = 'active'"
        result = SchemaAdvisor(live_schema=self._schema(ndv=900_000)).analyze(sql)
        self.assertNotIn("low_cardinality", _types(result))

    def test_missing_ndv_is_skipped(self):
        sql = "SELECT * FROM users WHERE status = 'active'"
        result = SchemaAdvisor(live_schema=self._schema(ndv=None)).analyze(sql)
        self.assertNotIn("low_cardinality", _types(result))
        self.assertTrue(any("distinct-value" in s for s in result.skipped))


class StaleStatsTests(SimpleTestCase):
    def _schema(self, engine="postgresql", last_analyzed=None):
        users = TableSnapshot(
            name="users",
            row_count=1000,
            last_analyzed=last_analyzed,
            columns=[{"name": "id", "type": "int", "nullable": False}],
        )
        return _ctx(engine=engine, tables=[users])

    def test_never_analyzed_is_info(self):
        sql = "SELECT * FROM users WHERE id = 1"
        result = SchemaAdvisor(live_schema=self._schema(last_analyzed=None)).analyze(
            sql
        )
        stale = [i for i in result.insights if i.type == "stale_stats"]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0].severity, "info")

    def test_old_stats_flagged_medium(self):
        old = (datetime.now(timezone.utc) - timedelta(days=STALE_DAYS + 5)).isoformat()
        sql = "SELECT * FROM users WHERE id = 1"
        result = SchemaAdvisor(live_schema=self._schema(last_analyzed=old)).analyze(sql)
        stale = [i for i in result.insights if i.type == "stale_stats"]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0].severity, "medium")

    def test_fresh_stats_not_flagged(self):
        fresh = datetime.now(timezone.utc).isoformat()
        sql = "SELECT * FROM users WHERE id = 1"
        result = SchemaAdvisor(live_schema=self._schema(last_analyzed=fresh)).analyze(
            sql
        )
        self.assertNotIn("stale_stats", _types(result))

    def test_sqlite_is_skipped(self):
        sql = "SELECT * FROM users WHERE id = 1"
        result = SchemaAdvisor(
            live_schema=self._schema(engine="sqlite", last_analyzed=None)
        ).analyze(sql)
        self.assertNotIn("stale_stats", _types(result))
        self.assertTrue(any("last-ANALYZE" in s for s in result.skipped))


class GuardTests(SimpleTestCase):
    def test_no_live_schema_returns_skipped(self):
        result = SchemaAdvisor(live_schema=None).analyze("SELECT 1")
        self.assertEqual(result.insights, [])
        self.assertTrue(result.skipped)

    def test_result_is_json_serializable(self):
        import json

        orders = TableSnapshot(
            name="orders",
            row_count=1000,
            columns=[{"name": "customer_id", "type": "int", "nullable": True}],
            foreign_keys=[
                {
                    "name": "fk",
                    "columns": ["customer_id"],
                    "referenced_table": "customers",
                    "referenced_columns": ["id"],
                }
            ],
        )
        result = SchemaAdvisor(live_schema=_ctx(tables=[orders])).analyze(
            "SELECT * FROM orders WHERE customer_id = 1"
        )
        # Round-trips cleanly (matches what the view stores on the JSONField).
        json.dumps(result.to_dict())
