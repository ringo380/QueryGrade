"""Live database schema snapshot used by the IndexRecommender pipeline.

A ``LiveSchemaContext`` is a frozen view of:

  * tables (with row counts and size estimates)
  * columns
  * indexes (current state — used for redundancy detection)
  * foreign keys

…sourced from a user's connected database via ``DatabaseIntrospector`` and
cached in Redis (``query_analysis_cache`` backend, 2 hour TTL — same as
analysis results) so the IndexRecommender doesn't re-introspect on every
grade.

The context also exposes ``hydrate_statistics_manager`` which fills a
``DatabaseStatisticsManager`` (analyzer/ml/integration/database_stats.py)
with the introspected data so its existing selectivity / cardinality
estimators can run without modification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from django.core.cache import caches

from analyzer.database_introspector import DatabaseIntrospector

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60 * 60 * 2  # 2 hours, matching query_analysis_cache convention
CACHE_PREFIX = "live_schema:v1:"


@dataclass
class IndexSnapshot:
    name: str
    columns: List[str]
    unique: bool = False
    primary: bool = False

    def column_key(self) -> tuple:
        return tuple(c.lower() for c in self.columns)


@dataclass
class TableSnapshot:
    name: str
    schema: str = ""
    row_count: Optional[int] = None
    size_mb: Optional[float] = None
    columns: List[Dict] = field(default_factory=list)
    indexes: List[IndexSnapshot] = field(default_factory=list)
    foreign_keys: List[Dict] = field(default_factory=list)

    def column_names(self) -> List[str]:
        return [c["name"] for c in self.columns]

    def index_columns_set(self) -> List[tuple]:
        return [idx.column_key() for idx in self.indexes]


@dataclass
class LiveSchemaContext:
    """Frozen snapshot of a user's database schema."""

    engine: str
    database: str
    schema: str = ""
    fetched_at: str = ""
    tables: Dict[str, TableSnapshot] = field(default_factory=dict)

    def get_table(self, name: str) -> Optional[TableSnapshot]:
        if not name:
            return None
        # Case-insensitive lookup.
        lname = name.lower()
        for tname, tbl in self.tables.items():
            if tname.lower() == lname:
                return tbl
        return None

    def hydrate_statistics_manager(self, manager) -> None:
        """Populate a ``DatabaseStatisticsManager`` from this snapshot.

        The manager is consumed by ``analyzer/ml/integration/database_stats.py``
        for selectivity / cardinality estimates already used elsewhere.
        """
        from analyzer.ml.integration.database_stats import (
            ColumnStatistics,
            DataDistribution,
            IndexStatistics,
            TableStatistics,
        )

        for tbl in self.tables.values():
            manager.table_stats[tbl.name] = TableStatistics(
                table_name=tbl.name,
                row_count=tbl.row_count or 0,
                page_count=0,
                avg_row_size=0,
                total_size_mb=tbl.size_mb or 0.0,
                last_analyzed=datetime.utcnow(),
            )
            for idx in tbl.indexes:
                manager.index_stats[tbl.name].append(
                    IndexStatistics(
                        index_name=idx.name,
                        table_name=tbl.name,
                        columns=list(idx.columns),
                        unique=idx.unique,
                        clustered=idx.primary,
                        index_depth=3,
                        leaf_pages=0,
                        fragmentation_percentage=0.0,
                        avg_key_size=0,
                        # Unique indexes are maximally selective. Otherwise
                        # use a conservative default; downstream estimators
                        # refine via row count / NDV when available.
                        selectivity=1.0 if idx.unique else 0.1,
                    )
                )
            for col in tbl.columns:
                manager.column_stats[tbl.name][col["name"]] = ColumnStatistics(
                    column_name=col["name"],
                    table_name=tbl.name,
                    data_type=str(col.get("type", "")),
                    nullable=bool(col.get("nullable", True)),
                    distinct_values=tbl.row_count or 0,
                    null_percentage=0.0,
                    avg_length=col.get("max_length") or 0,
                    max_length=col.get("max_length") or 0,
                    min_value=None,
                    max_value=None,
                    distribution=DataDistribution.UNIFORM,
                )

    def to_cache_dict(self) -> dict:
        return {
            "engine": self.engine,
            "database": self.database,
            "schema": self.schema,
            "fetched_at": self.fetched_at,
            "tables": {
                name: {
                    **asdict(tbl),
                    "indexes": [asdict(i) for i in tbl.indexes],
                }
                for name, tbl in self.tables.items()
            },
        }

    @classmethod
    def from_cache_dict(cls, data: dict) -> "LiveSchemaContext":
        ctx = cls(
            engine=data.get("engine", ""),
            database=data.get("database", ""),
            schema=data.get("schema", ""),
            fetched_at=data.get("fetched_at", ""),
        )
        for name, tdata in (data.get("tables") or {}).items():
            indexes = [IndexSnapshot(**i) for i in tdata.get("indexes") or []]
            tbl_kwargs = {k: v for k, v in tdata.items() if k != "indexes"}
            tbl_kwargs["indexes"] = indexes
            ctx.tables[name] = TableSnapshot(**tbl_kwargs)
        return ctx


def _cache_key(connection_id: int) -> str:
    return f"{CACHE_PREFIX}{connection_id}"


def _cache():
    try:
        return caches["query_analysis_cache"]
    except Exception:  # fallback for environments without that cache configured
        return caches["default"]


def build_live_context(
    connection,
    *,
    force_refresh: bool = False,
) -> Optional[LiveSchemaContext]:
    """Build (or return cached) a ``LiveSchemaContext`` for a saved connection.

    Args:
        connection: ``UserDatabaseConnection`` instance.
        force_refresh: bypass the Redis cache.

    Returns ``None`` if the connection cannot be established. Errors are
    logged; the caller should treat absence as "no live schema available"
    and continue with text-only analysis.
    """
    cache = _cache()
    key = _cache_key(connection.pk)

    if not force_refresh:
        cached = cache.get(key)
        if cached:
            try:
                return LiveSchemaContext.from_cache_dict(cached)
            except Exception:
                logger.warning("Stale schema cache for %s; refetching", connection.pk)

    cfg = connection.to_connection_config()
    introspector = DatabaseIntrospector(cfg)
    if not introspector.connect():
        logger.info("Live schema: connect failed for %s", connection.pk)
        return None

    schema = cfg.get("schema") or ""
    tables = introspector.get_tables(schema=schema or None)

    ctx = LiveSchemaContext(
        engine=cfg["engine"],
        database=cfg.get("name", ""),
        schema=schema,
        fetched_at=datetime.utcnow().isoformat(timespec="seconds"),
    )
    for t in tables:
        ctx.tables[t.name] = TableSnapshot(
            name=t.name,
            schema=t.schema,
            row_count=t.row_count,
            size_mb=t.size_mb,
            columns=list(t.columns),
            indexes=[
                IndexSnapshot(
                    name=i["name"],
                    columns=list(i["columns"]),
                    unique=bool(i.get("unique", False)),
                    primary=bool(i.get("primary", False)),
                )
                for i in t.indexes
            ],
            foreign_keys=list(t.foreign_keys),
        )

    try:
        cache.set(key, ctx.to_cache_dict(), CACHE_TTL_SECONDS)
    except Exception:
        logger.exception("Failed to cache live schema for %s", connection.pk)

    return ctx


def invalidate_live_context(connection_id: int) -> None:
    try:
        _cache().delete(_cache_key(connection_id))
    except Exception:
        logger.exception("Failed to invalidate live schema cache")


def schema_fingerprint(ctx: LiveSchemaContext) -> str:
    """Stable hash of the schema shape — used as a feature/cache key seed."""
    payload = json.dumps(
        {
            n: {
                "cols": [c["name"] for c in t.columns],
                "idx": [list(i.column_key()) for i in t.indexes],
                "fk": [
                    (fk.get("columns"), fk.get("referenced_table"))
                    for fk in t.foreign_keys
                ],
            }
            for n, t in sorted(ctx.tables.items())
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
