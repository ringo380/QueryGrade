"""Database-specific CREATE INDEX / DROP INDEX DDL generation.

Used by ``IndexRecommender`` to emit copy-pasteable scripts. We deliberately
quote identifiers conservatively so column names with mixed case or reserved
words don't break the generated DDL.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

# Conservative identifier validator: alphanumeric + underscore. Anything
# else gets quoted using engine-specific delimiters.
_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

ENGINE_QUOTES = {
    "postgresql": '"',
    "sqlite": '"',
    "mysql": "`",
}


def quote_ident(name: str, engine: str) -> str:
    if not name:
        return name
    if _SAFE_IDENT.match(name):
        return name
    q = ENGINE_QUOTES.get(engine, '"')
    escaped = name.replace(q, q + q)
    return f"{q}{escaped}{q}"


def suggest_index_name(table: str, columns: Iterable[str], suffix: str = "") -> str:
    base = "_".join(["idx", table] + list(columns))
    base = re.sub(r"[^A-Za-z0-9_]", "_", base).strip("_").lower()
    if suffix:
        base = f"{base}_{suffix.lower()}"
    return base[:63]  # Postgres identifier limit


def create_index_sql(
    *,
    engine: str,
    table: str,
    columns: List[str],
    index_name: Optional[str] = None,
    unique: bool = False,
    include: Optional[List[str]] = None,
    full_text: bool = False,
) -> str:
    """Generate ``CREATE INDEX`` DDL for the given engine."""
    name = index_name or suggest_index_name(table, columns)
    qname = quote_ident(name, engine)
    qtable = quote_ident(table, engine)
    qcols = ", ".join(quote_ident(c, engine) for c in columns)

    if engine == "postgresql":
        unique_kw = "UNIQUE " if unique else ""
        include_clause = ""
        if include:
            qinc = ", ".join(quote_ident(c, engine) for c in include)
            include_clause = f" INCLUDE ({qinc})"
        return f"CREATE {unique_kw}INDEX {qname} ON {qtable} ({qcols}){include_clause};"

    if engine == "mysql":
        if full_text:
            return f"CREATE FULLTEXT INDEX {qname} ON {qtable} ({qcols});"
        unique_kw = "UNIQUE " if unique else ""
        return f"CREATE {unique_kw}INDEX {qname} ON {qtable} ({qcols}) USING BTREE;"

    if engine == "sqlite":
        unique_kw = "UNIQUE " if unique else ""
        return f"CREATE {unique_kw}INDEX {qname} ON {qtable} ({qcols});"

    # Fallback: ANSI-ish.
    unique_kw = "UNIQUE " if unique else ""
    return f"CREATE {unique_kw}INDEX {qname} ON {qtable} ({qcols});"


def drop_index_sql(*, engine: str, index_name: str, table: Optional[str] = None) -> str:
    qname = quote_ident(index_name, engine)
    if engine == "mysql" and table:
        qtable = quote_ident(table, engine)
        return f"DROP INDEX {qname} ON {qtable};"
    return f"DROP INDEX IF EXISTS {qname};"
