import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import sqlparse
from django.db import connections

logger = logging.getLogger(__name__)


@dataclass
class TableInfo:
    """Information about a database table."""

    name: str
    schema: str = "public"
    row_count: Optional[int] = None
    size_mb: Optional[float] = None
    columns: List[Dict[str, Any]] = None
    indexes: List[Dict[str, Any]] = None
    foreign_keys: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.columns is None:
            self.columns = []
        if self.indexes is None:
            self.indexes = []
        if self.foreign_keys is None:
            self.foreign_keys = []


@dataclass
class IndexInfo:
    """Information about a database index."""

    name: str
    table: str
    columns: List[str]
    unique: bool = False
    primary: bool = False
    type: str = "btree"
    size_mb: Optional[float] = None


@dataclass
class ColumnInfo:
    """Information about a table column."""

    name: str
    type: str
    nullable: bool = True
    default: Optional[str] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None


class DatabaseIntrospector:
    """
    Database introspection utility for analyzing database schemas
    and providing context-aware query recommendations.
    """

    def __init__(self, database_config: Dict[str, str]):
        """
        Initialize database introspector.

        Args:
            database_config: Dictionary with connection parameters
                {
                    'engine': 'mysql' | 'postgresql' | 'sqlite',
                    'host': 'localhost',
                    'port': '3306',
                    'name': 'database_name',
                    'user': 'username',
                    'password': 'password'
                }
        """
        self.config = database_config
        self.connection = None
        self.introspection = None
        self._tables_cache = {}
        self._indexes_cache = {}

    def connect(self) -> bool:
        """
        Establish database connection.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create Django database configuration
            db_config = self._create_django_db_config()

            # Note: In a real implementation, you'd want to create a temporary
            # connection rather than modifying Django's connections
            connection_name = f"introspection_{id(self)}"
            connections.databases[connection_name] = db_config

            self.connection = connections[connection_name]
            self.introspection = self.connection.introspection

            # Test connection
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")

            logger.info(f"Successfully connected to {self.config['engine']} database")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def _create_django_db_config(self) -> Dict[str, Any]:
        """Create Django database configuration from our config."""
        engine_map = {
            "mysql": "django.db.backends.mysql",
            "postgresql": "django.db.backends.postgresql",
            "sqlite": "django.db.backends.sqlite3",
        }

        config = {
            "ENGINE": engine_map.get(
                self.config["engine"], "django.db.backends.postgresql"
            ),
            "NAME": self.config["name"],
        }

        if self.config["engine"] != "sqlite":
            config.update(
                {
                    "HOST": self.config.get("host", "localhost"),
                    "PORT": self.config.get("port", ""),
                    "USER": self.config.get("user", ""),
                    "PASSWORD": self.config.get("password", ""),
                }
            )

        return config

    def get_tables(self, schema: str = None) -> List[TableInfo]:
        """
        Get list of tables in the database.

        Args:
            schema: Schema name (for PostgreSQL)

        Returns:
            List of TableInfo objects
        """
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            with self.connection.cursor() as cursor:
                table_names = self.introspection.get_table_list(cursor)
                tables = []

                for table_info in table_names:
                    table_name = table_info.name

                    # Get table details
                    table = TableInfo(
                        name=table_name,
                        schema=schema or "public",
                        columns=self._get_table_columns(cursor, table_name),
                        indexes=self._get_table_indexes(cursor, table_name),
                        foreign_keys=self._get_foreign_keys(cursor, table_name),
                    )

                    # Try to get row count and size (database-specific)
                    try:
                        table.row_count = self._get_table_row_count(cursor, table_name)
                        table.size_mb = self._get_table_size(cursor, table_name)
                    except Exception as e:
                        logger.debug(
                            f"Could not get size info for table {table_name}: {e}"
                        )

                    tables.append(table)
                    self._tables_cache[table_name] = table

                return tables

        except Exception as e:
            logger.error(f"Error getting tables: {e}")
            return []

    def _get_table_columns(self, cursor, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        try:
            descriptions = self.introspection.get_table_description(cursor, table_name)
            columns = []

            for desc in descriptions:
                column = {
                    "name": desc.name,
                    "type": desc.type_code,
                    "nullable": desc.null_ok,
                    "max_length": desc.size,
                    "precision": desc.precision,
                    "scale": desc.scale,
                }
                columns.append(column)

            return columns
        except Exception as e:
            logger.debug(f"Error getting columns for {table_name}: {e}")
            return []

    def _get_table_indexes(self, cursor, table_name: str) -> List[Dict[str, Any]]:
        """Get index information for a table."""
        try:
            indexes = self.introspection.get_indexes(cursor, table_name)
            index_list = []

            for index_name, index_info in indexes.items():
                index_dict = {
                    "name": index_name,
                    "columns": index_info["columns"],
                    "unique": index_info["unique"],
                    "primary": index_info["primary_key"],
                }
                index_list.append(index_dict)

            return index_list
        except Exception as e:
            logger.debug(f"Error getting indexes for {table_name}: {e}")
            return []

    def _get_foreign_keys(self, cursor, table_name: str) -> List[Dict[str, Any]]:
        """Get foreign key information for a table."""
        try:
            constraints = self.introspection.get_constraints(cursor, table_name)
            foreign_keys = []

            for constraint_name, constraint_info in constraints.items():
                if constraint_info["foreign_key"]:
                    fk = {
                        "name": constraint_name,
                        "columns": constraint_info["columns"],
                        "referenced_table": constraint_info["foreign_key"][0],
                        "referenced_columns": constraint_info["foreign_key"][1],
                    }
                    foreign_keys.append(fk)

            return foreign_keys
        except Exception as e:
            logger.debug(f"Error getting foreign keys for {table_name}: {e}")
            return []

    def _get_table_row_count(self, cursor, table_name: str) -> Optional[int]:
        """Get approximate row count for a table."""
        try:
            if self.config["engine"] == "mysql":
                cursor.execute(
                    """
                    SELECT table_rows
                    FROM information_schema.tables
                    WHERE table_name = %s
                """,
                    [table_name],
                )
            elif self.config["engine"] == "postgresql":
                cursor.execute(
                    """
                    SELECT reltuples::bigint AS estimate
                    FROM pg_class
                    WHERE relname = %s
                """,
                    [table_name],
                )
            else:
                # Fallback: exact count (can be slow for large tables)
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")  # nosec

            result = cursor.fetchone()
            return int(result[0]) if result and result[0] is not None else None

        except Exception as e:
            logger.debug(f"Error getting row count for {table_name}: {e}")
            return None

    def _get_table_size(self, cursor, table_name: str) -> Optional[float]:
        """Get table size in MB."""
        try:
            if self.config["engine"] == "mysql":
                cursor.execute(
                    """
                    SELECT ROUND(((data_length + index_length) / 1024 / 1024), 2) AS size_mb
                    FROM information_schema.tables
                    WHERE table_name = %s
                """,
                    [table_name],
                )
            elif self.config["engine"] == "postgresql":
                cursor.execute(
                    """
                    SELECT ROUND(pg_total_relation_size(%s) / 1024.0 / 1024.0, 2) AS size_mb
                """,
                    [table_name],
                )
            else:
                return None

            result = cursor.fetchone()
            return float(result[0]) if result and result[0] is not None else None

        except Exception as e:
            logger.debug(f"Error getting size for {table_name}: {e}")
            return None

    def analyze_query_context(self, sql_query: str) -> Dict[str, Any]:
        """
        Analyze query in the context of the actual database schema.

        Args:
            sql_query: SQL query to analyze

        Returns:
            Dictionary with contextual analysis results
        """
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            # Parse the query to extract table and column references
            parsed = sqlparse.parse(sql_query)[0]
            table_references = self._extract_table_references(parsed)
            column_references = self._extract_column_references(parsed)

            # Get schema information for referenced tables
            context = {
                "tables_referenced": table_references,
                "columns_referenced": column_references,
                "schema_analysis": {},
                "recommendations": [],
                "potential_issues": [],
            }

            # Analyze each referenced table
            for table_name in table_references:
                if table_name in self._tables_cache:
                    table_info = self._tables_cache[table_name]
                else:
                    # Get table info if not cached
                    with self.connection.cursor() as cursor:
                        table_info = self._get_table_info(cursor, table_name)
                        self._tables_cache[table_name] = table_info

                if table_info:
                    context["schema_analysis"][table_name] = {
                        "row_count": table_info.row_count,
                        "size_mb": table_info.size_mb,
                        "columns": len(table_info.columns),
                        "indexes": len(table_info.indexes),
                        "has_primary_key": any(
                            idx.get("primary", False) for idx in table_info.indexes
                        ),
                    }

                    # Generate context-aware recommendations
                    recommendations = self._generate_context_recommendations(
                        sql_query, table_name, table_info, column_references
                    )
                    context["recommendations"].extend(recommendations)

            return context

        except Exception as e:
            logger.error(f"Error analyzing query context: {e}")
            return {"error": str(e), "tables_referenced": [], "recommendations": []}

    def _get_table_info(self, cursor, table_name: str) -> Optional[TableInfo]:
        """Get complete table information."""
        try:
            table = TableInfo(
                name=table_name,
                columns=self._get_table_columns(cursor, table_name),
                indexes=self._get_table_indexes(cursor, table_name),
                foreign_keys=self._get_foreign_keys(cursor, table_name),
            )

            table.row_count = self._get_table_row_count(cursor, table_name)
            table.size_mb = self._get_table_size(cursor, table_name)

            return table
        except Exception as e:
            logger.debug(f"Error getting table info for {table_name}: {e}")
            return None

    def _extract_table_references(self, parsed_query) -> List[str]:
        """Extract table names referenced in the query."""
        tables = []

        def extract_from_token(token):
            if token.ttype is None and hasattr(token, "tokens"):
                for subtoken in token.tokens:
                    extract_from_token(subtoken)
            elif token.ttype in (sqlparse.tokens.Name, None):
                token_str = str(token).strip()
                # Simple heuristic: look for FROM and JOIN clauses
                if any(keyword in str(token).upper() for keyword in ["FROM", "JOIN"]):
                    # Extract table names after FROM/JOIN
                    words = token_str.split()
                    for i, word in enumerate(words):
                        if word.upper() in [
                            "FROM",
                            "JOIN",
                            "INNER",
                            "LEFT",
                            "RIGHT",
                            "FULL",
                            "OUTER",
                        ]:
                            if i + 1 < len(words):
                                table_name = words[i + 1].strip("(),")
                                if table_name and table_name.upper() not in [
                                    "SELECT",
                                    "WHERE",
                                    "ORDER",
                                    "GROUP",
                                ]:
                                    tables.append(table_name)

        extract_from_token(parsed_query)
        return list(set(tables))  # Remove duplicates

    def _extract_column_references(self, parsed_query) -> List[str]:
        """Extract column names referenced in the query."""
        columns = []
        query_str = str(parsed_query)

        # Simple extraction - could be improved with more sophisticated parsing
        # Look for SELECT columns
        select_match = re.search(
            r"SELECT\s+(.*?)\s+FROM", query_str, re.IGNORECASE | re.DOTALL
        )
        if select_match:
            select_part = select_match.group(1)
            # Split by commas and clean up
            column_parts = [col.strip() for col in select_part.split(",")]
            for col in column_parts:
                # Remove aliases, functions, etc. - basic cleanup
                col = re.sub(r"\s+AS\s+\w+", "", col, flags=re.IGNORECASE)
                col = col.split(".")[-1]  # Take column name after table prefix
                if col != "*" and col.isalnum():
                    columns.append(col)

        return columns

    def _generate_context_recommendations(
        self, query: str, table_name: str, table_info: TableInfo, columns: List[str]
    ) -> List[str]:
        """Generate context-aware recommendations based on schema analysis."""
        recommendations = []

        # Check for missing indexes on queried columns
        indexed_columns = set()
        for index in table_info.indexes:
            indexed_columns.update(index.get("columns", []))

        for column in columns:
            if column not in indexed_columns and column in [
                col["name"] for col in table_info.columns
            ]:
                recommendations.append(
                    f"Consider adding an index on {table_name}.{column} for better query performance"
                )

        # Check for large table queries without LIMIT
        if table_info.row_count and table_info.row_count > 100000:
            if "LIMIT" not in query.upper():
                recommendations.append(
                    f"Table {table_name} has {table_info.row_count:,} rows. "
                    f"Consider adding a LIMIT clause to prevent large result sets"
                )

        # Check for SELECT * on tables with many columns
        if len(table_info.columns) > 10 and "SELECT *" in query.upper():
            recommendations.append(
                f"Table {table_name} has {len(table_info.columns)} columns. "
                f"Consider selecting only needed columns instead of SELECT *"
            )

        # Check for missing foreign key usage in JOINs
        if "JOIN" in query.upper():
            fk_columns = [
                fk["columns"][0] for fk in table_info.foreign_keys if fk["columns"]
            ]
            if fk_columns:
                recommendations.append(
                    f"Ensure JOIN conditions use foreign key columns: {', '.join(fk_columns)}"
                )

        return recommendations

    def get_execution_plan(self, sql_query: str) -> Optional[Dict[str, Any]]:
        """
        Get query execution plan (database-specific).

        Args:
            sql_query: SQL query to analyze

        Returns:
            Dictionary with execution plan information
        """
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            with self.connection.cursor() as cursor:
                if self.config["engine"] == "postgresql":
                    cursor.execute(f"EXPLAIN (FORMAT JSON) {sql_query}")
                    result = cursor.fetchone()
                    return {"plan": result[0], "format": "json"}

                elif self.config["engine"] == "mysql":
                    cursor.execute(f"EXPLAIN FORMAT=JSON {sql_query}")
                    result = cursor.fetchone()
                    return {"plan": result[0], "format": "json"}

                else:
                    cursor.execute(f"EXPLAIN QUERY PLAN {sql_query}")
                    result = cursor.fetchall()
                    return {"plan": result, "format": "rows"}

        except Exception as e:
            logger.error(f"Error getting execution plan: {e}")
            return None

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
