"""Schema-aware non-index insights (issue #6, scope A).

A companion to ``IndexRecommender`` that surfaces *structural* and
*statistical* problems a live schema reveals — things an index alone won't
fix. Four insight types:

1. **fk_fanout** — a JOIN follows a one-to-many foreign-key edge and the query
   has no aggregation/DISTINCT, so the result set multiplies (row fan-out).
2. **nullable_fk** — a foreign-key column referenced by the query is nullable,
   so an INNER JOIN silently drops orphan rows.
3. **low_cardinality** — a column filtered/joined on has very few distinct
   values (real NDV from planner stats), making any index there low-selectivity.
4. **stale_stats** — a referenced table's planner statistics are old or were
   never gathered, so cost estimates (including our own HypoPG path) are
   unreliable.

Design mirrors ``IndexRecommender``: a pure service taking ``(connection,
live_schema)`` and a query string, returning a JSON-serializable result. Each
insight is independently guarded — a failure in one never suppresses the
others, and missing data yields a human-readable ``skipped`` reason rather than
an error. Reuses ``index_recommender.extract_candidates`` for column/clause
extraction so the two services agree on what the query touches.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from analyzer.services.index_recommender import extract_candidates
from analyzer.services.live_schema_context import LiveSchemaContext, TableSnapshot

logger = logging.getLogger(__name__)

# Tunable thresholds (mirror the MAX_RESULTS = 5 convention in index_recommender).
LOW_NDV_ABS = (
    16  # absolute distinct-value count at/below which a column is "low cardinality"
)
LOW_NDV_RATIO = (
    0.01  # distinct/row_count ratio at/below which a column is low-cardinality
)
MIN_ROWS_FOR_NDV = 1000  # only flag low cardinality on tables larger than this
STALE_DAYS = 14  # planner stats older than this are flagged stale

_AGG_OR_DISTINCT = re.compile(
    r"\b(?:DISTINCT|COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY)\b", re.IGNORECASE
)

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass
class SchemaInsight:
    type: str  # fk_fanout | nullable_fk | low_cardinality | stale_stats
    severity: str  # high | medium | low | info
    title: str
    detail: str
    table: str = ""
    columns: List[str] = field(default_factory=list)
    suggestion: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def _dedupe_key(self) -> tuple:
        return (self.type, self.table, tuple(c.lower() for c in self.columns))


@dataclass
class SchemaInsightResult:
    insights: List[SchemaInsight]
    skipped: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "insights": [i.to_dict() for i in self.insights],
            "skipped": list(self.skipped),
        }


def _has_unique_index_on(table_snap: TableSnapshot, columns) -> bool:
    """True if an index on *exactly* these columns (any order-prefix match) is unique.

    A primary key or an unrelated unique index does not count — the FK columns
    themselves must be the unique key for the relationship to be one-to-one.
    """
    target = tuple(c.lower() for c in columns)
    for idx in table_snap.indexes:
        if idx.unique and idx.column_key() == target:
            return True
    return False


class SchemaAdvisor:
    def __init__(
        self,
        *,
        connection=None,
        live_schema: Optional[LiveSchemaContext] = None,
    ):
        self.connection = connection
        self.live_schema = live_schema
        self.engine = (
            connection.engine
            if connection is not None
            else (live_schema.engine if live_schema else "")
        )

    # ------------------------------------------------------------------ public

    def analyze(self, sql: str) -> SchemaInsightResult:
        insights: List[SchemaInsight] = []
        skipped: List[str] = []

        if not self.live_schema:
            return SchemaInsightResult(
                insights=[], skipped=["No live schema available."]
            )

        candidates = extract_candidates(sql, self.live_schema)

        for runner in (
            self._fan_out_warnings,
            self._nullable_fk,
            self._low_cardinality,
            self._stale_stats,
        ):
            try:
                runner(sql, candidates, insights, skipped)
            except Exception:  # one insight failing must not sink the rest
                logger.exception("SchemaAdvisor insight %s failed", runner.__name__)

        # Dedupe (a column hit by several clauses can produce duplicates) and
        # sort by severity for stable, useful display ordering.
        seen = set()
        deduped: List[SchemaInsight] = []
        for ins in insights:
            key = ins._dedupe_key()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ins)
        deduped.sort(key=lambda i: _SEVERITY_ORDER.get(i.severity, 9))

        return SchemaInsightResult(
            insights=deduped, skipped=list(dict.fromkeys(skipped))
        )

    # ------------------------------------------------------------- insight #1

    def _fan_out_warnings(self, sql, candidates, insights, skipped) -> None:
        """Warn when a JOIN over a one-to-many FK edge multiplies result rows."""
        if _AGG_OR_DISTINCT.search(sql or ""):
            return  # aggregation/DISTINCT collapses the fan-out

        join_tables = {c.table for c in candidates if "join" in c.clauses and c.table}
        if len(join_tables) < 2:
            return

        schema = self.live_schema
        for tname in join_tables:
            child = schema.get_table(tname)
            if not child:
                continue
            for fk in child.foreign_keys:
                parent_name = fk.get("referenced_table")
                fk_cols = fk.get("columns") or []
                if not parent_name or not fk_cols:
                    continue
                # Only relevant if the parent is also joined in this query.
                if not any(parent_name.lower() == jt.lower() for jt in join_tables):
                    continue
                # One-to-many unless the child's FK columns are themselves unique.
                if _has_unique_index_on(child, fk_cols):
                    continue
                insights.append(
                    SchemaInsight(
                        type="fk_fanout",
                        severity="medium",
                        title=f"JOIN may multiply rows: {tname} → {parent_name}",
                        detail=(
                            f"`{tname}` references `{parent_name}` via "
                            f"({', '.join(fk_cols)}) as a one-to-many relationship "
                            f"(no unique constraint on those columns). Joining them "
                            f"without aggregation or DISTINCT can multiply result rows."
                        ),
                        table=tname,
                        columns=list(fk_cols),
                        suggestion=(
                            "Aggregate the many-side, add DISTINCT, or confirm the "
                            "fan-out is intentional."
                        ),
                    )
                )

    # ------------------------------------------------------------- insight #2

    def _nullable_fk(self, sql, candidates, insights, skipped) -> None:
        """Flag nullable FK columns used in JOINs (INNER JOIN drops orphan rows)."""
        ref_tables = {c.table for c in candidates if c.table}
        is_inner = not re.search(
            r"\b(?:LEFT|RIGHT|FULL)\s+(?:OUTER\s+)?JOIN\b", sql or "", re.IGNORECASE
        )
        schema = self.live_schema
        for tname in ref_tables:
            snap = schema.get_table(tname)
            if not snap:
                continue
            nullable_map = {
                c["name"].lower(): bool(c.get("nullable", True)) for c in snap.columns
            }
            for fk in snap.foreign_keys:
                fk_cols = fk.get("columns") or []
                nullable_cols = [
                    c for c in fk_cols if nullable_map.get(c.lower(), True)
                ]
                if not nullable_cols:
                    continue
                # Only interesting if the query actually joins on/filters this table.
                if not any(
                    cand.table.lower() == tname.lower()
                    and ("join" in cand.clauses or "where_eq" in cand.clauses)
                    for cand in candidates
                ):
                    continue
                insights.append(
                    SchemaInsight(
                        type="nullable_fk",
                        severity="medium",
                        title=f"Nullable foreign key on {tname}",
                        detail=(
                            f"Foreign-key column(s) ({', '.join(nullable_cols)}) on "
                            f"`{tname}` are nullable. "
                            + (
                                "An INNER JOIN here silently drops rows whose FK "
                                "is NULL."
                                if is_inner
                                else "Rows with NULL FK won't match on this join."
                            )
                        ),
                        table=tname,
                        columns=list(nullable_cols),
                        suggestion=(
                            "Use a LEFT JOIN to retain orphan rows, or enforce "
                            "NOT NULL if the relationship is mandatory."
                        ),
                    )
                )

    # ------------------------------------------------------------- insight #3

    def _low_cardinality(self, sql, candidates, insights, skipped) -> None:
        """Warn that filtering/joining a low-NDV column yields poor selectivity."""
        schema = self.live_schema
        any_ndv_known = False
        missing_ndv = False
        for cand in candidates:
            if not ({"where_eq", "where_range", "join"} & set(cand.clauses)):
                continue
            snap = schema.get_table(cand.table)
            if not snap:
                continue
            ndv_map = {
                c["name"].lower(): c.get("ndv") for c in snap.columns if "ndv" in c
            }
            for col in cand.columns:
                ndv = ndv_map.get(col.lower())
                if ndv is None:
                    missing_ndv = True
                    continue
                any_ndv_known = True
                row_count = snap.row_count or 0
                if row_count <= MIN_ROWS_FOR_NDV:
                    continue
                ratio = (ndv / row_count) if row_count else 1.0
                if ndv <= LOW_NDV_ABS or ratio <= LOW_NDV_RATIO:
                    insights.append(
                        SchemaInsight(
                            type="low_cardinality",
                            severity="low",
                            title=f"Low-cardinality predicate: {cand.table}.{col}",
                            detail=(
                                f"`{cand.table}.{col}` has only ~{ndv:,} distinct "
                                f"value(s) across {row_count:,} rows. Filtering or "
                                f"joining on it is low-selectivity, so an index there "
                                f"may not help much."
                            ),
                            table=cand.table,
                            columns=[col],
                            suggestion=(
                                "Combine with a more selective column in a composite "
                                "index, or filter on a higher-cardinality column."
                            ),
                        )
                    )
        if missing_ndv and not any_ndv_known:
            skipped.append(
                "Low-cardinality analysis skipped: this database doesn't expose "
                "per-column distinct-value statistics."
            )

    # ------------------------------------------------------------- insight #4

    def _stale_stats(self, sql, candidates, insights, skipped) -> None:
        """Flag referenced tables whose planner statistics are stale or absent."""
        schema = self.live_schema
        ref_tables = {c.table for c in candidates if c.table}
        if not ref_tables:
            return

        if self.engine and self.engine not in ("postgresql", "mysql"):
            skipped.append(
                "Statistics-staleness check skipped: this database doesn't track a "
                "last-ANALYZE timestamp."
            )
            return

        now = datetime.now(timezone.utc)
        for tname in ref_tables:
            snap = schema.get_table(tname)
            if not snap:
                continue
            last = snap.last_analyzed
            if not last:
                insights.append(
                    SchemaInsight(
                        type="stale_stats",
                        severity="info",
                        title=f"No planner statistics for {tname}",
                        detail=(
                            f"`{tname}` has no recorded statistics refresh. Cost-based "
                            f"estimates (including index recommendations) may be off."
                        ),
                        table=tname,
                        suggestion=f"Run ANALYZE {tname}; to gather fresh statistics.",
                    )
                )
                continue
            try:
                parsed = datetime.fromisoformat(last)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            age_days = (now - parsed).days
            if age_days >= STALE_DAYS:
                insights.append(
                    SchemaInsight(
                        type="stale_stats",
                        severity="medium",
                        title=f"Stale statistics on {tname}",
                        detail=(
                            f"`{tname}` statistics were last refreshed ~{age_days} "
                            f"days ago. Cost estimates may no longer reflect the data."
                        ),
                        table=tname,
                        suggestion=f"Run ANALYZE {tname}; to refresh statistics.",
                    )
                )
