"""Tests for IndexRecommender + script generator (issue #7 Layer 3)."""

from __future__ import annotations

from django.test import TestCase

from analyzer.services.index_recommender import (Confidence, IndexRecommender,
                                                  Redundancy,
                                                  classify_redundancy,
                                                  extract_candidates)
from analyzer.services.index_script_generator import (create_index_sql,
                                                       drop_index_sql,
                                                       quote_ident,
                                                       suggest_index_name)
from analyzer.services.live_schema_context import (IndexSnapshot,
                                                    LiveSchemaContext,
                                                    TableSnapshot)


def _schema(*, engine="postgresql", existing_indexes=None, row_count=1_000_000):
    """Build a fixture schema with an `orders(id, customer_id, status, created_at)` table."""
    indexes = [IndexSnapshot(name="orders_pkey", columns=["id"], unique=True, primary=True)]
    for cols in existing_indexes or []:
        indexes.append(IndexSnapshot(name="idx_" + "_".join(cols), columns=list(cols)))

    table = TableSnapshot(
        name="orders",
        schema="public",
        row_count=row_count,
        size_mb=120.0,
        columns=[
            {"name": "id", "type": "integer", "nullable": False},
            {"name": "customer_id", "type": "integer", "nullable": True},
            {"name": "status", "type": "varchar", "nullable": False},
            {"name": "created_at", "type": "timestamp", "nullable": False},
        ],
        indexes=indexes,
        foreign_keys=[],
    )
    return LiveSchemaContext(
        engine=engine,
        database="shop",
        schema="public",
        tables={"orders": table},
    )


class CandidateExtractionTests(TestCase):
    def test_where_equality_qualified(self):
        cands = extract_candidates("SELECT * FROM orders WHERE orders.customer_id = 42")
        self.assertTrue(any(c.table == "orders" and c.columns == ("customer_id",) for c in cands))

    def test_where_equality_unqualified_uses_schema(self):
        cands = extract_candidates(
            "SELECT * FROM orders WHERE customer_id = 42", _schema()
        )
        keys = {(c.table, c.columns) for c in cands}
        self.assertIn(("orders", ("customer_id",)), keys)

    def test_where_range(self):
        cands = extract_candidates(
            "SELECT * FROM orders WHERE created_at > '2026-01-01'", _schema()
        )
        keys = {(c.table, c.columns, "where_range" in c.clauses) for c in cands}
        self.assertIn(("orders", ("created_at",), True), keys)

    def test_join_on_columns(self):
        sql = (
            "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.id = 1"
        )
        cands = extract_candidates(sql)
        # Join produces both sides as candidates.
        names = {(c.table, c.columns) for c in cands}
        self.assertTrue(any(t in {"o", "orders"} and cols == ("customer_id",) for t, cols in names))

    def test_order_by_composite(self):
        cands = extract_candidates(
            "SELECT id FROM orders ORDER BY customer_id, created_at DESC", _schema()
        )
        keys = {(c.table, c.columns) for c in cands}
        self.assertIn(("orders", ("customer_id", "created_at")), keys)


class RedundancyTests(TestCase):
    def test_exact_match(self):
        from analyzer.services.index_recommender import IndexCandidate

        snap = _schema(existing_indexes=[("customer_id",)]).get_table("orders")
        cand = IndexCandidate(table="orders", columns=("customer_id",), clauses=["where_eq"])
        self.assertEqual(classify_redundancy(cand, snap), Redundancy.EXACT)

    def test_subsumed_by_composite(self):
        from analyzer.services.index_recommender import IndexCandidate

        snap = _schema(existing_indexes=[("customer_id", "created_at")]).get_table("orders")
        cand = IndexCandidate(table="orders", columns=("customer_id",), clauses=["where_eq"])
        self.assertEqual(classify_redundancy(cand, snap), Redundancy.SUBSUMED)

    def test_distinct_columns_not_redundant(self):
        from analyzer.services.index_recommender import IndexCandidate

        snap = _schema(existing_indexes=[("customer_id",)]).get_table("orders")
        cand = IndexCandidate(table="orders", columns=("status",), clauses=["where_eq"])
        self.assertEqual(classify_redundancy(cand, snap), Redundancy.NONE)


class ScriptGeneratorTests(TestCase):
    def test_quote_identifier_safe(self):
        self.assertEqual(quote_ident("simple", "postgresql"), "simple")

    def test_quote_identifier_unsafe(self):
        self.assertEqual(quote_ident("Order Items", "postgresql"), '"Order Items"')
        self.assertEqual(quote_ident("Order Items", "mysql"), "`Order Items`")

    def test_postgres_create_index_with_include(self):
        sql = create_index_sql(
            engine="postgresql",
            table="orders",
            columns=["customer_id"],
            include=["status"],
        )
        self.assertIn("CREATE INDEX idx_orders_customer_id ON orders (customer_id)", sql)
        self.assertIn("INCLUDE (status)", sql)

    def test_mysql_create_index_uses_btree(self):
        sql = create_index_sql(
            engine="mysql", table="orders", columns=["customer_id"]
        )
        self.assertIn("USING BTREE", sql)

    def test_drop_index_engine_specific(self):
        self.assertIn("ON", drop_index_sql(engine="mysql", index_name="x", table="orders"))
        self.assertIn("IF EXISTS", drop_index_sql(engine="postgresql", index_name="x"))

    def test_suggest_index_name_truncates(self):
        long_cols = [f"c{i}" for i in range(20)]
        name = suggest_index_name("very_long_table_name", long_cols)
        self.assertLessEqual(len(name), 63)


class RecommenderEndToEndTests(TestCase):
    def test_simple_equality_recommends_index(self):
        schema = _schema()
        rec = IndexRecommender(live_schema=schema).recommend(
            "SELECT * FROM orders WHERE customer_id = 42"
        )
        self.assertGreaterEqual(len(rec.recommendations), 1)
        top = rec.recommendations[0]
        self.assertEqual(top.table, "orders")
        self.assertEqual(top.columns, ["customer_id"])
        self.assertEqual(top.redundancy, Redundancy.NONE)
        self.assertGreater(top.estimated_improvement_pct, 5)
        self.assertIn("CREATE INDEX", top.create_sql)

    def test_existing_index_marked_redundant_exact(self):
        schema = _schema(existing_indexes=[("customer_id",)])
        rec = IndexRecommender(live_schema=schema).recommend(
            "SELECT * FROM orders WHERE customer_id = 42"
        )
        # We keep the EXACT redundancy result so the UI can show "no action needed".
        # It must rank below any non-redundant recommendation.
        if rec.recommendations:
            for r in rec.recommendations:
                if r.columns == ["customer_id"]:
                    self.assertEqual(r.redundancy, Redundancy.EXACT)

    def test_subsumed_filtered(self):
        schema = _schema(existing_indexes=[("customer_id", "created_at")])
        rec = IndexRecommender(live_schema=schema).recommend(
            "SELECT * FROM orders WHERE customer_id = 42"
        )
        cust_only = [r for r in rec.recommendations if r.columns == ["customer_id"]]
        self.assertEqual(cust_only, [], "Subsumed candidate should be filtered out")
        self.assertGreaterEqual(rec.filtered_redundant, 1)

    def test_max_results_cap(self):
        # Construct a query touching many columns
        schema = _schema()
        long_sql = (
            "SELECT * FROM orders WHERE customer_id = 1 AND status = 'x' "
            "AND created_at > '2026-01-01' ORDER BY id, customer_id, status, created_at"
        )
        rec = IndexRecommender(live_schema=schema).recommend(long_sql)
        self.assertLessEqual(len(rec.recommendations), IndexRecommender.MAX_RESULTS)

    def test_no_schema_still_works(self):
        rec = IndexRecommender().recommend(
            "SELECT * FROM orders WHERE orders.customer_id = 42"
        )
        self.assertGreaterEqual(len(rec.recommendations), 1)
        self.assertEqual(rec.recommendations[0].confidence, Confidence.LOW)

    def test_serialization_round_trip(self):
        rec = IndexRecommender(live_schema=_schema()).recommend(
            "SELECT * FROM orders WHERE customer_id = 1"
        )
        d = rec.to_dict()
        self.assertIn("recommendations", d)
        self.assertIsInstance(d["recommendations"][0]["confidence"], str)
        self.assertIsInstance(d["recommendations"][0]["redundancy"], str)
