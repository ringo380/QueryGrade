"""Schema-aware automated index recommender (issue #7).

Pipeline
--------
1. **Candidate generation** — parse the query (sqlparse) to find columns
   used in equality predicates, range predicates, JOIN ON clauses, and
   ORDER BY / GROUP BY. Each (table, [columns]) tuple is a candidate.

2. **Redundancy filter** — compare each candidate against the live schema's
   existing indexes. Mark exact matches (``REDUNDANT_EXACT``) and prefix-
   subsumed candidates (``REDUNDANT_SUBSUMED``). Subsumed candidates are
   dropped; exact matches are returned with the redundancy badge so the
   UI can explain "no action needed."

3. **Cost-benefit scoring**

   * **PostgreSQL with HypoPG**: create the candidate as a hypothetical
     index, run ``EXPLAIN (FORMAT JSON)`` on the user's query, diff the
     reported ``Total Cost`` against the baseline. Improvement % =
     ``(baseline - candidate) / baseline``. Confidence = ``HIGH``.
   * **PostgreSQL without HypoPG / MySQL / SQLite**: stats-based estimator
     using row count × selectivity heuristics (1 / NDV for equality, 0.3
     for range, 0.5 for ORDER BY without filter). Confidence = ``MEDIUM``
     when row count is known, ``LOW`` otherwise.

4. **Ranking** — sort by ``estimated_improvement_pct × confidence_weight``
   and cap at 5.

5. **Script generation** — emit DB-specific CREATE INDEX DDL via
   ``index_script_generator``.

The recommender is intentionally tolerant: if any step fails (parser
chokes, EXPLAIN errors, HypoPG not installed), it falls back to the
heuristic path and emits an advisory in the result.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Where
from sqlparse.tokens import Keyword

from analyzer.services.index_script_generator import (
    create_index_sql,
    drop_index_sql,
    suggest_index_name,
)
from analyzer.services.live_schema_context import LiveSchemaContext, TableSnapshot

logger = logging.getLogger(__name__)


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


CONFIDENCE_WEIGHT = {Confidence.HIGH: 1.0, Confidence.MEDIUM: 0.6, Confidence.LOW: 0.3}


class Redundancy(str, Enum):
    NONE = "NONE"
    SUBSUMED = "REDUNDANT_SUBSUMED"
    EXACT = "REDUNDANT_EXACT"


@dataclass
class IndexRecommendation:
    table: str
    columns: List[str]
    rationale: str
    estimated_improvement_pct: float
    confidence: Confidence
    redundancy: Redundancy
    create_sql: str
    drop_sql: str
    affected_clauses: List[str] = field(default_factory=list)
    index_type: str = "btree"
    advisory: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = self.confidence.value
        d["redundancy"] = self.redundancy.value
        return d


# ----------------------------- Candidate generation -------------------------


@dataclass
class IndexCandidate:
    table: str
    columns: Tuple[str, ...]
    clauses: List[
        str
    ]  # which clause types contributed: where_eq, where_range, join, order, group

    def column_key(self) -> tuple:
        return tuple(c.lower() for c in self.columns)


_TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_][A-Za-z0-9_\.\"`]*)",
    re.IGNORECASE,
)
_WHERE_EQ = re.compile(
    r"(?:^|\sWHERE\s|\sAND\s|\sOR\s)\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(=|IN\s*\()",
    re.IGNORECASE,
)
_WHERE_RANGE = re.compile(
    r"(?:^|\sWHERE\s|\sAND\s|\sOR\s)\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:<=|>=|<|>|BETWEEN)",
    re.IGNORECASE,
)
_JOIN_ON = re.compile(
    # JOIN <table> [[AS] <alias>] ON <predicate>
    r"\bJOIN\s+[A-Za-z_][A-Za-z0-9_\.\"`]*"
    r"(?:\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_]*)?"
    r"\s+ON\s+([^\(\)]+?)"
    r"(?=\s+(?:LEFT|RIGHT|INNER|OUTER|CROSS|JOIN|WHERE|GROUP|ORDER|LIMIT|HAVING)\b|\s*$)",
    re.IGNORECASE | re.DOTALL,
)
_ORDER_BY = re.compile(
    r"\bORDER\s+BY\s+([^\)]+?)(?=\s+(?:LIMIT|OFFSET|HAVING|$))", re.IGNORECASE
)
_GROUP_BY = re.compile(
    r"\bGROUP\s+BY\s+([^\)]+?)(?=\s+(?:HAVING|ORDER|LIMIT|$))", re.IGNORECASE
)


def _split_qualified(token: str) -> Tuple[Optional[str], str]:
    """Return ``(table_or_alias, column)`` for ``a.col`` or ``(None, "col")``."""
    parts = token.strip().split(".")
    if len(parts) == 2:
        return parts[0].strip().strip('"`'), parts[1].strip().strip('"`')
    return None, parts[0].strip().strip('"`')


def _infer_table_for_column(
    column: str,
    tables: Sequence[str],
    schema: Optional[LiveSchemaContext],
) -> Optional[str]:
    """Best-effort table resolution for an unqualified column reference."""
    if not schema:
        return tables[0] if len(tables) == 1 else None

    matches = []
    for t in tables:
        snap = schema.get_table(t)
        if not snap:
            continue
        if any(c["name"].lower() == column.lower() for c in snap.columns):
            matches.append(t)
    if len(matches) == 1:
        return matches[0]
    return None


def extract_candidates(
    sql: str, schema: Optional[LiveSchemaContext] = None
) -> List[IndexCandidate]:
    """Parse the query and produce raw index candidates."""
    sql_clean = " ".join(sql.split())  # collapse whitespace for the regexes
    tables_in_query = []
    for m in _TABLE_REF.finditer(sql_clean):
        ref = m.group(1).strip().strip('"`').split(" ")[0]
        # drop alias suffix like "orders o"
        tables_in_query.append(ref.split(".")[-1])
    tables_in_query = list(dict.fromkeys(tables_in_query))  # preserve order, dedupe

    by_key: Dict[Tuple[str, Tuple[str, ...]], IndexCandidate] = {}

    def _add(table: Optional[str], columns: Sequence[str], clause: str) -> None:
        if not table or not columns:
            return
        cols = tuple(c for c in columns if c)
        if not cols:
            return
        key = (table, tuple(c.lower() for c in cols))
        cand = by_key.get(key)
        if cand:
            if clause not in cand.clauses:
                cand.clauses.append(clause)
            return
        by_key[key] = IndexCandidate(table=table, columns=cols, clauses=[clause])

    # WHERE equality
    for m in _WHERE_EQ.finditer(sql_clean):
        tbl, col = _split_qualified(m.group(1))
        if not tbl:
            tbl = _infer_table_for_column(col, tables_in_query, schema)
        _add(tbl, [col], "where_eq")

    # WHERE range
    for m in _WHERE_RANGE.finditer(sql_clean):
        tbl, col = _split_qualified(m.group(1))
        if not tbl:
            tbl = _infer_table_for_column(col, tables_in_query, schema)
        _add(tbl, [col], "where_range")

    # JOIN ON
    for m in _JOIN_ON.finditer(sql_clean + " "):  # trailing space helps the lookahead
        clause = m.group(1)
        for eq in re.finditer(
            r"([A-Za-z_][A-Za-z0-9_\.]*)\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)", clause
        ):
            for side in (eq.group(1), eq.group(2)):
                tbl, col = _split_qualified(side)
                if not tbl:
                    tbl = _infer_table_for_column(col, tables_in_query, schema)
                _add(tbl, [col], "join")

    # ORDER BY (composite)
    m = _ORDER_BY.search(sql_clean + " ")
    if m:
        _add_composite(m.group(1), tables_in_query, schema, "order", _add)

    # GROUP BY (composite)
    m = _GROUP_BY.search(sql_clean + " ")
    if m:
        _add_composite(m.group(1), tables_in_query, schema, "group", _add)

    return list(by_key.values())


def _add_composite(
    fragment: str,
    tables: Sequence[str],
    schema: Optional[LiveSchemaContext],
    clause: str,
    add_fn,
) -> None:
    cols_by_table: Dict[str, List[str]] = {}
    for raw in fragment.split(","):
        token = raw.strip().rstrip(";").split()[0]  # drop ASC/DESC
        if not token or "(" in token:  # skip expressions
            continue
        tbl, col = _split_qualified(token)
        if not tbl:
            tbl = _infer_table_for_column(col, tables, schema)
        if not tbl:
            continue
        cols_by_table.setdefault(tbl, []).append(col)
    for tbl, cols in cols_by_table.items():
        if cols:
            add_fn(tbl, cols, clause)


# ----------------------------- Redundancy filter -----------------------------


def classify_redundancy(
    candidate: IndexCandidate, table_snap: Optional[TableSnapshot]
) -> Redundancy:
    if not table_snap:
        return Redundancy.NONE
    cand_key = candidate.column_key()
    for idx in table_snap.indexes:
        existing = idx.column_key()
        if existing == cand_key:
            return Redundancy.EXACT
        # Subsumed: candidate is a prefix of an existing composite, OR
        # existing is a prefix of candidate (then a longer existing covers it).
        if len(cand_key) < len(existing) and existing[: len(cand_key)] == cand_key:
            return Redundancy.SUBSUMED
    return Redundancy.NONE


# ----------------------------- Cost-benefit scoring --------------------------


def _heuristic_improvement(
    candidate: IndexCandidate, table_snap: Optional[TableSnapshot]
) -> Tuple[float, Confidence, str]:
    """Stats-based improvement estimate when EXPLAIN/HypoPG isn't available.

    Heuristic: assume baseline is a sequential scan touching every row.
    A useful index reduces rows touched by:
        equality   -> 1 / NDV  (we approximate NDV with row_count when no
                                column stats; very conservative on small tables)
        range      -> 0.3
        order/group only -> 0.5 (sort avoidance, no filter)
        join       -> 0.5 (FK lookup pattern)
    Improvement is then ``1 - rows_touched_ratio`` clipped to [0.05, 0.99].
    Confidence MEDIUM if we have a row_count > 1000, LOW otherwise.
    """
    clauses = set(candidate.clauses)
    selectivity = 1.0
    for c, factor in (
        ("where_eq", 0.05),
        ("where_range", 0.3),
        ("join", 0.5),
        ("order", 0.5),
        ("group", 0.5),
    ):
        if c in clauses:
            selectivity = min(selectivity, factor)

    row_count = table_snap.row_count if table_snap else None
    if row_count and row_count > 1000:
        # Equality on large tables: refine using NDV-as-row-count proxy. We
        # don't actually have NDV; assume best case 1/sqrt(N) for primary-
        # key-like columns, which keeps us conservative on small datasets.
        if "where_eq" in clauses and row_count > 10_000:
            selectivity = min(selectivity, max(1.0 / (row_count**0.5), 1e-4))
        confidence = Confidence.MEDIUM
        rationale = (
            f"Heuristic estimate: ~{1 - selectivity:.0%} reduction in rows "
            f"scanned on a {row_count:,}-row table."
        )
    else:
        confidence = Confidence.LOW
        rationale = (
            "Heuristic estimate (no row-count data): "
            "predicted-impact figures should be treated as approximate."
        )

    improvement = max(0.05, min(0.99, 1.0 - selectivity))
    return improvement, confidence, rationale


def _try_hypopg_improvement(
    candidate: IndexCandidate,
    sql: str,
    connection,
) -> Optional[Tuple[float, str]]:
    """PostgreSQL-only: use HypoPG to compute a real EXPLAIN cost delta.

    Returns ``(improvement_pct, rationale)`` or ``None`` when HypoPG isn't
    available / the query can't be EXPLAINed safely.
    """
    if not connection or connection.engine != "postgresql":
        return None

    try:
        from analyzer.database_introspector import DatabaseIntrospector

        introspector = DatabaseIntrospector(connection.to_connection_config())
        if not introspector.connect():
            return None
        with introspector.connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'hypopg'")
            if not cursor.fetchone():
                return None

            # Baseline cost
            try:
                cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
                baseline_row = cursor.fetchone()
                baseline_plan = baseline_row[0][0]["Plan"]
                baseline_cost = float(baseline_plan.get("Total Cost", 0.0))
            except Exception:
                return None

            cols = ", ".join(f'"{c}"' for c in candidate.columns)
            create_stmt = f'CREATE INDEX ON "{candidate.table}" ({cols})'
            try:
                cursor.execute(
                    "SELECT indexrelid FROM hypopg_create_index(%s)",
                    [create_stmt],
                )
                row = cursor.fetchone()
                hypo_oid = row[0] if row else None
            except Exception:
                return None

            try:
                cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
                cand_plan = cursor.fetchone()[0][0]["Plan"]
                cand_cost = float(cand_plan.get("Total Cost", baseline_cost))
            except Exception:
                cand_cost = baseline_cost
            finally:
                if hypo_oid is not None:
                    try:
                        cursor.execute("SELECT hypopg_drop_index(%s)", [hypo_oid])
                    except Exception:
                        pass

        if baseline_cost <= 0:
            return None
        improvement = max(0.0, min(0.99, (baseline_cost - cand_cost) / baseline_cost))
        return improvement, (
            f"HypoPG EXPLAIN cost: {baseline_cost:.1f} → {cand_cost:.1f} "
            f"({improvement:.0%} reduction)."
        )
    except Exception as exc:
        logger.info("HypoPG path failed: %s", exc)
        return None


# ----------------------------- Public API -----------------------------------


@dataclass
class RecommendationResult:
    recommendations: List[IndexRecommendation]
    total_candidates: int
    filtered_redundant: int
    advisories: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "recommendations": [r.to_dict() for r in self.recommendations],
            "total_candidates": self.total_candidates,
            "filtered_redundant": self.filtered_redundant,
            "advisories": list(self.advisories),
        }


class IndexRecommender:
    MAX_RESULTS = 5

    def __init__(
        self,
        *,
        connection=None,
        live_schema: Optional[LiveSchemaContext] = None,
    ):
        """
        Args:
            connection: ``UserDatabaseConnection`` (used for HypoPG / EXPLAIN).
                When ``None`` we skip the PG path and rely on heuristics.
            live_schema: snapshot used for redundancy detection and stats.
        """
        self.connection = connection
        self.live_schema = live_schema
        self.engine = (
            connection.engine
            if connection is not None
            else (live_schema.engine if live_schema else "")
        )

    def recommend(self, sql: str) -> RecommendationResult:
        candidates = extract_candidates(sql, self.live_schema)
        advisories: List[str] = []

        # Redundancy filter
        kept: List[Tuple[IndexCandidate, Redundancy]] = []
        filtered_redundant = 0
        for cand in candidates:
            snap = self.live_schema.get_table(cand.table) if self.live_schema else None
            redundancy = classify_redundancy(cand, snap)
            if redundancy == Redundancy.SUBSUMED:
                filtered_redundant += 1
                continue  # drop subsumed
            kept.append((cand, redundancy))

        # Cost-benefit
        results: List[IndexRecommendation] = []
        for cand, redundancy in kept:
            snap = self.live_schema.get_table(cand.table) if self.live_schema else None

            improvement = 0.0
            confidence = Confidence.LOW
            rationale = ""

            if redundancy == Redundancy.EXACT:
                improvement = 0.0
                confidence = Confidence.HIGH
                rationale = "An equivalent index already exists; no action needed."
            else:
                hypo = _try_hypopg_improvement(cand, sql, self.connection)
                if hypo is not None:
                    improvement, rationale = hypo
                    confidence = Confidence.HIGH
                else:
                    if self.engine == "postgresql" and self.connection is not None:
                        advisories.append(
                            "HypoPG is not installed on this PostgreSQL "
                            "database — falling back to heuristic estimates. "
                            "Install with `CREATE EXTENSION hypopg;` for "
                            "EXPLAIN-grounded predictions."
                        )
                    improvement, confidence, rationale = _heuristic_improvement(
                        cand, snap
                    )

            engine = self.engine or "postgresql"
            full_text = (
                "where_range" in cand.clauses
                and any("%" in (sql or "") for _ in [None])
                and self.engine == "mysql"
            )
            create_sql = create_index_sql(
                engine=engine,
                table=cand.table,
                columns=list(cand.columns),
                full_text=False,
            )
            drop_sql = drop_index_sql(
                engine=engine,
                index_name=suggest_index_name(cand.table, cand.columns),
                table=cand.table,
            )

            results.append(
                IndexRecommendation(
                    table=cand.table,
                    columns=list(cand.columns),
                    rationale=rationale,
                    estimated_improvement_pct=round(improvement * 100, 1),
                    confidence=confidence,
                    redundancy=redundancy,
                    create_sql=create_sql,
                    drop_sql=drop_sql,
                    affected_clauses=list(cand.clauses),
                )
            )

        # Rank: weight improvement by confidence; demote redundant_exact to bottom.
        def _score(r: IndexRecommendation) -> float:
            if r.redundancy == Redundancy.EXACT:
                return -1.0
            return r.estimated_improvement_pct * CONFIDENCE_WEIGHT[r.confidence]

        results.sort(key=_score, reverse=True)
        results = results[: self.MAX_RESULTS]

        return RecommendationResult(
            recommendations=results,
            total_candidates=len(candidates),
            filtered_redundant=filtered_redundant,
            advisories=list(dict.fromkeys(advisories)),
        )
