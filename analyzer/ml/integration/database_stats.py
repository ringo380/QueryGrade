"""
Database Statistics Integration System

This module integrates real database statistics and metadata to provide
context-aware feature extraction and query analysis.
"""

import hashlib
import json
import logging
import pickle
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class StatisticType(Enum):
    """Types of database statistics"""

    TABLE_SIZE = "table_size"
    INDEX_STATISTICS = "index_statistics"
    COLUMN_STATISTICS = "column_statistics"
    HISTOGRAM = "histogram"
    DEPENDENCY = "dependency"
    QUERY_FREQUENCY = "query_frequency"
    CACHE_STATISTICS = "cache_statistics"
    LOCK_STATISTICS = "lock_statistics"
    IO_STATISTICS = "io_statistics"


class DataDistribution(Enum):
    """Types of data distributions"""

    UNIFORM = "uniform"
    NORMAL = "normal"
    SKEWED = "skewed"
    BIMODAL = "bimodal"
    EXPONENTIAL = "exponential"
    CUSTOM = "custom"


@dataclass
class TableStatistics:
    """Statistics for a database table"""

    table_name: str
    row_count: int
    page_count: int
    avg_row_size: int
    total_size_mb: float
    last_analyzed: datetime
    fragmentation_percentage: float = 0.0
    compression_ratio: float = 1.0
    partition_count: int = 1
    clustered_index: Optional[str] = None
    fill_factor: float = 0.9


@dataclass
class IndexStatistics:
    """Statistics for a database index"""

    index_name: str
    table_name: str
    columns: List[str]
    unique: bool
    clustered: bool
    index_depth: int
    leaf_pages: int
    fragmentation_percentage: float
    avg_key_size: int
    selectivity: float  # 0-1, higher is more selective
    usage_count: int = 0
    last_used: Optional[datetime] = None
    maintenance_cost: float = 0.0


@dataclass
class ColumnStatistics:
    """Statistics for a database column"""

    column_name: str
    table_name: str
    data_type: str
    nullable: bool
    distinct_values: int
    null_percentage: float
    avg_length: int
    max_length: int
    min_value: Any
    max_value: Any
    distribution: DataDistribution
    histogram_buckets: List[Tuple[Any, int]] = field(default_factory=list)
    correlation_with_pk: float = 0.0
    most_common_values: List[Tuple[Any, float]] = field(default_factory=list)


@dataclass
class RelationshipStatistics:
    """Statistics for table relationships"""

    parent_table: str
    child_table: str
    parent_column: str
    child_column: str
    relationship_type: str  # one-to-one, one-to-many, many-to-many
    join_selectivity: float
    orphan_percentage: float = 0.0
    cascade_impact: int = 0  # Number of affected rows on delete


@dataclass
class WorkloadStatistics:
    """Workload-level statistics"""

    total_queries: int
    queries_per_second: float
    read_write_ratio: float
    avg_query_time_ms: float
    cache_hit_ratio: float
    deadlock_count: int
    peak_connections: int
    avg_lock_wait_time_ms: float
    temp_space_usage_mb: float
    query_patterns: Dict[str, int] = field(default_factory=dict)


class DatabaseStatisticsManager:
    """Manages database statistics and provides context-aware features"""

    def __init__(self, database_type: str = "generic"):
        self.database_type = database_type
        self.logger = logging.getLogger(__name__)

        # Statistics storage
        self.table_stats: Dict[str, TableStatistics] = {}
        self.index_stats: Dict[str, List[IndexStatistics]] = defaultdict(list)
        self.column_stats: Dict[str, Dict[str, ColumnStatistics]] = defaultdict(dict)
        self.relationship_stats: List[RelationshipStatistics] = []
        self.workload_stats: Optional[WorkloadStatistics] = None

        # Query history for pattern analysis
        self.query_history = deque(maxlen=10000)
        self.query_performance_history = {}

        # Cache for computed statistics
        self.statistics_cache = {}
        self.cache_ttl = timedelta(hours=1)

    def load_statistics(self, source: str, format: str = "json"):
        """Load database statistics from a source"""
        try:
            if format == "json":
                with open(source, "r") as f:
                    data = json.load(f)
                    self._parse_json_statistics(data)
            elif format == "pickle":
                with open(source, "rb") as f:
                    data = pickle.load(f)
                    self._load_pickle_statistics(data)
            elif format == "live":
                # Connect to live database
                self._fetch_live_statistics(source)

            self.logger.info(f"Loaded statistics from {source}")

        except Exception as e:
            self.logger.error(f"Error loading statistics: {e}")

    def _parse_json_statistics(self, data: Dict[str, Any]):
        """Parse statistics from JSON format"""
        # Parse table statistics
        if "tables" in data:
            for table_data in data["tables"]:
                table_stat = TableStatistics(
                    table_name=table_data["name"],
                    row_count=table_data.get("row_count", 0),
                    page_count=table_data.get("page_count", 0),
                    avg_row_size=table_data.get("avg_row_size", 100),
                    total_size_mb=table_data.get("total_size_mb", 0),
                    last_analyzed=datetime.fromisoformat(
                        table_data.get("last_analyzed", datetime.now().isoformat())
                    ),
                    fragmentation_percentage=table_data.get("fragmentation", 0),
                    clustered_index=table_data.get("clustered_index"),
                )
                self.table_stats[table_stat.table_name] = table_stat

        # Parse index statistics
        if "indexes" in data:
            for index_data in data["indexes"]:
                index_stat = IndexStatistics(
                    index_name=index_data["name"],
                    table_name=index_data["table"],
                    columns=index_data["columns"],
                    unique=index_data.get("unique", False),
                    clustered=index_data.get("clustered", False),
                    index_depth=index_data.get("depth", 3),
                    leaf_pages=index_data.get("leaf_pages", 100),
                    fragmentation_percentage=index_data.get("fragmentation", 0),
                    avg_key_size=index_data.get("avg_key_size", 10),
                    selectivity=index_data.get("selectivity", 0.1),
                )
                self.index_stats[index_stat.table_name].append(index_stat)

        # Parse column statistics
        if "columns" in data:
            for col_data in data["columns"]:
                col_stat = ColumnStatistics(
                    column_name=col_data["name"],
                    table_name=col_data["table"],
                    data_type=col_data["data_type"],
                    nullable=col_data.get("nullable", True),
                    distinct_values=col_data.get("distinct_values", 100),
                    null_percentage=col_data.get("null_percentage", 0),
                    avg_length=col_data.get("avg_length", 10),
                    max_length=col_data.get("max_length", 100),
                    min_value=col_data.get("min_value"),
                    max_value=col_data.get("max_value"),
                    distribution=DataDistribution(
                        col_data.get("distribution", "uniform")
                    ),
                )
                self.column_stats[col_stat.table_name][col_stat.column_name] = col_stat

    def _fetch_live_statistics(self, connection_string):
        """Fetch statistics from a live database connection.

        Accepts either:
          * a ``UserDatabaseConnection`` instance (preferred), or
          * a ``LiveSchemaContext`` (already-fetched snapshot).

        The historical signature was a connection_string placeholder. The
        method is kept duck-typed so callers can pass whichever they have.
        """
        from analyzer.services.live_schema_context import (
            LiveSchemaContext, build_live_context)

        ctx = None
        if isinstance(connection_string, LiveSchemaContext):
            ctx = connection_string
        elif hasattr(connection_string, "to_connection_config"):
            ctx = build_live_context(connection_string)
        else:
            self.logger.warning(
                "_fetch_live_statistics: unsupported source %r", type(connection_string)
            )
            return

        if ctx is None:
            return
        ctx.hydrate_statistics_manager(self)

    def _load_pickle_statistics(self, data: Dict[str, Any]):
        """Load statistics from pickle format"""
        self.table_stats = data.get("table_stats", {})
        self.index_stats = data.get("index_stats", defaultdict(list))
        self.column_stats = data.get("column_stats", defaultdict(dict))
        self.relationship_stats = data.get("relationship_stats", [])
        self.workload_stats = data.get("workload_stats")

    def get_table_statistics(self, table_name: str) -> Optional[TableStatistics]:
        """Get statistics for a specific table"""
        return self.table_stats.get(table_name)

    def get_index_statistics(self, table_name: str) -> List[IndexStatistics]:
        """Get all indexes for a table"""
        return self.index_stats.get(table_name, [])

    def get_column_statistics(
        self, table_name: str, column_name: str
    ) -> Optional[ColumnStatistics]:
        """Get statistics for a specific column"""
        return self.column_stats.get(table_name, {}).get(column_name)

    def estimate_selectivity(
        self, table_name: str, column_name: str, operator: str, value: Any
    ) -> float:
        """
        Estimate the selectivity of a predicate
        Returns a value between 0 and 1, where 0 means no rows and 1 means all rows
        """
        col_stats = self.get_column_statistics(table_name, column_name)

        if not col_stats:
            # Default selectivity estimates
            if operator == "=":
                return 0.01
            elif operator in ["<", ">", "<=", ">="]:
                return 0.3
            elif operator == "BETWEEN":
                return 0.1
            elif operator == "LIKE":
                return 0.25
            else:
                return 0.5

        # Use column statistics for better estimates
        if operator == "=":
            if col_stats.distinct_values > 0:
                base_selectivity = 1.0 / col_stats.distinct_values

                # Check if value is in most common values
                for common_val, frequency in col_stats.most_common_values:
                    if common_val == value:
                        return frequency

                return base_selectivity
            return 0.01

        elif operator in ["<", "<="]:
            if col_stats.min_value is not None and col_stats.max_value is not None:
                try:
                    range_fraction = (value - col_stats.min_value) / (
                        col_stats.max_value - col_stats.min_value
                    )
                    return max(0.0, min(1.0, range_fraction))
                except Exception:
                    return 0.3
            return 0.3

        elif operator in [">", ">="]:
            if col_stats.min_value is not None and col_stats.max_value is not None:
                try:
                    range_fraction = (col_stats.max_value - value) / (
                        col_stats.max_value - col_stats.min_value
                    )
                    return max(0.0, min(1.0, range_fraction))
                except Exception:
                    return 0.3
            return 0.3

        return 0.5

    def estimate_join_cardinality(
        self,
        left_table: str,
        right_table: str,
        join_column_left: str,
        join_column_right: str,
    ) -> int:
        """Estimate the cardinality of a join result"""
        left_stats = self.get_table_statistics(left_table)
        right_stats = self.get_table_statistics(right_table)

        if not left_stats or not right_stats:
            return 10000  # Default estimate

        left_col_stats = self.get_column_statistics(left_table, join_column_left)
        right_col_stats = self.get_column_statistics(right_table, join_column_right)

        # Check for foreign key relationship
        for rel in self.relationship_stats:
            if (
                rel.parent_table == left_table
                and rel.child_table == right_table
                and rel.parent_column == join_column_left
                and rel.child_column == join_column_right
            ):
                # Use relationship statistics
                return int(right_stats.row_count * (1 - rel.orphan_percentage))

        # Estimate based on column statistics
        if left_col_stats and right_col_stats:
            # Use the smaller distinct count as the join factor
            join_factor = min(
                left_col_stats.distinct_values, right_col_stats.distinct_values
            )
            if join_factor > 0:
                return int((left_stats.row_count * right_stats.row_count) / join_factor)

        # Default: Cartesian product reduced by 10x
        return int((left_stats.row_count * right_stats.row_count) / 10)

    def suggest_index(
        self, table_name: str, columns: List[str], query_pattern: str = None
    ) -> Dict[str, Any]:
        """Suggest an index based on statistics and query pattern"""
        suggestion = {
            "table": table_name,
            "columns": columns,
            "type": "nonclustered",
            "estimated_benefit": 0.0,
            "reasoning": [],
        }

        table_stats = self.get_table_statistics(table_name)
        if not table_stats:
            suggestion["reasoning"].append("No table statistics available")
            return suggestion

        # Check column statistics
        total_selectivity = 1.0
        for column in columns:
            col_stats = self.get_column_statistics(table_name, column)
            if col_stats:
                selectivity = 1.0 / max(col_stats.distinct_values, 1)
                total_selectivity *= selectivity

                # High cardinality columns are good for indexing
                if col_stats.distinct_values > table_stats.row_count * 0.7:
                    suggestion["reasoning"].append(f"{column} has high cardinality")
                    suggestion["estimated_benefit"] += 0.3

        # Check if this would be a covering index
        existing_indexes = self.get_index_statistics(table_name)
        is_covering = False
        for index in existing_indexes:
            if set(columns).issubset(set(index.columns)):
                suggestion["reasoning"].append(
                    "Columns already covered by existing index"
                )
                suggestion["estimated_benefit"] = 0.0
                return suggestion

        # Estimate benefit based on selectivity
        if total_selectivity < 0.01:  # Very selective
            suggestion["estimated_benefit"] += 0.5
            suggestion["reasoning"].append("High selectivity index")

        # Check query frequency
        if query_pattern and self.workload_stats:
            pattern_frequency = self.workload_stats.query_patterns.get(query_pattern, 0)
            if pattern_frequency > 100:
                suggestion["estimated_benefit"] += 0.2
                suggestion["reasoning"].append(
                    f"Frequently used in {query_pattern} queries"
                )

        # Determine index type
        if len(columns) == 1 and columns[0] == table_stats.clustered_index:
            suggestion["type"] = "clustered"
        elif total_selectivity < 0.001:
            suggestion["type"] = "unique"

        return suggestion

    def analyze_data_skew(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Analyze data skew for a column"""
        col_stats = self.get_column_statistics(table_name, column_name)

        if not col_stats:
            return {
                "skew_detected": False,
                "skew_type": "unknown",
                "skew_severity": 0.0,
            }

        analysis = {
            "skew_detected": False,
            "skew_type": "none",
            "skew_severity": 0.0,
            "recommendations": [],
        }

        # Check distribution type
        if col_stats.distribution == DataDistribution.SKEWED:
            analysis["skew_detected"] = True
            analysis["skew_type"] = "statistical"

        # Check for value concentration
        if col_stats.most_common_values:
            top_value_frequency = col_stats.most_common_values[0][1]
            if top_value_frequency > 0.5:
                analysis["skew_detected"] = True
                analysis["skew_type"] = "value_concentration"
                analysis["skew_severity"] = top_value_frequency
                analysis["recommendations"].append(
                    "Consider filtered statistics or indexes"
                )

        # Check for null skew
        if col_stats.null_percentage > 0.7:
            analysis["skew_detected"] = True
            analysis["skew_type"] = "null_heavy"
            analysis["skew_severity"] = col_stats.null_percentage
            analysis["recommendations"].append(
                "Consider filtered index excluding NULLs"
            )

        # Check histogram for uneven distribution
        if col_stats.histogram_buckets:
            bucket_sizes = [count for _, count in col_stats.histogram_buckets]
            if bucket_sizes:
                mean_size = np.mean(bucket_sizes)
                std_size = np.std(bucket_sizes)
                cv = std_size / mean_size if mean_size > 0 else 0

                if cv > 1.0:  # Coefficient of variation > 1 indicates high skew
                    analysis["skew_detected"] = True
                    analysis["skew_type"] = "histogram_skew"
                    analysis["skew_severity"] = min(cv / 2.0, 1.0)
                    analysis["recommendations"].append(
                        "Update statistics with higher sampling rate"
                    )

        return analysis

    def get_correlated_columns(
        self, table_name: str, column_name: str, threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Find columns that are correlated with the given column"""
        correlated = []

        target_stats = self.get_column_statistics(table_name, column_name)
        if not target_stats:
            return correlated

        for other_column, other_stats in self.column_stats.get(table_name, {}).items():
            if other_column == column_name:
                continue

            # Simple correlation check based on cardinality
            if target_stats.distinct_values > 0 and other_stats.distinct_values > 0:
                cardinality_ratio = min(
                    target_stats.distinct_values, other_stats.distinct_values
                ) / max(target_stats.distinct_values, other_stats.distinct_values)

                if cardinality_ratio > threshold:
                    correlated.append((other_column, cardinality_ratio))

        return sorted(correlated, key=lambda x: x[1], reverse=True)

    def estimate_memory_grant(self, query_analysis: Dict[str, Any]) -> float:
        """Estimate memory grant required for a query in MB"""
        memory_mb = 1.0  # Base memory

        # Check for sorts
        if "ORDER BY" in str(query_analysis):
            # Estimate rows to sort
            estimated_rows = 10000  # Default
            for table in query_analysis.get("tables", []):
                if table in self.table_stats:
                    estimated_rows = min(
                        estimated_rows, self.table_stats[table].row_count
                    )

            # Memory for sorting (rough estimate)
            memory_mb += estimated_rows * 0.0001  # 0.1KB per row

        # Check for hash joins
        if "joins" in query_analysis:
            for join in query_analysis["joins"]:
                # Hash table memory
                if "hash" in str(join).lower():
                    build_table_rows = 10000  # Default estimate
                    memory_mb += (
                        build_table_rows * 0.0002
                    )  # 0.2KB per row for hash table

        # Check for aggregations
        if "GROUP BY" in str(query_analysis):
            distinct_groups = 1000  # Default estimate
            memory_mb += distinct_groups * 0.001  # 1KB per group

        return memory_mb

    def get_partition_elimination_hints(
        self, table_name: str, filters: List[Dict[str, Any]]
    ) -> List[str]:
        """Get hints for partition elimination based on filters"""
        hints = []

        table_stats = self.get_table_statistics(table_name)
        if not table_stats or table_stats.partition_count <= 1:
            return hints

        for filter_info in filters:
            # Check if filter column is a partition key (simplified check)
            if "date" in filter_info.get("column", "").lower():
                hints.append(
                    f"Partition elimination possible on {filter_info['column']}"
                )

        return hints

    def track_query_performance(
        self,
        query_hash: str,
        execution_time_ms: float,
        rows_returned: int,
        cpu_time_ms: float = 0,
    ):
        """Track query performance for historical analysis"""
        if query_hash not in self.query_performance_history:
            self.query_performance_history[query_hash] = {
                "executions": 0,
                "total_time_ms": 0,
                "avg_time_ms": 0,
                "min_time_ms": float("inf"),
                "max_time_ms": 0,
                "total_rows": 0,
                "avg_rows": 0,
                "total_cpu_ms": 0,
            }

        perf = self.query_performance_history[query_hash]
        perf["executions"] += 1
        perf["total_time_ms"] += execution_time_ms
        perf["avg_time_ms"] = perf["total_time_ms"] / perf["executions"]
        perf["min_time_ms"] = min(perf["min_time_ms"], execution_time_ms)
        perf["max_time_ms"] = max(perf["max_time_ms"], execution_time_ms)
        perf["total_rows"] += rows_returned
        perf["avg_rows"] = perf["total_rows"] / perf["executions"]
        perf["total_cpu_ms"] += cpu_time_ms

    def get_query_performance_trend(self, query_hash: str) -> Dict[str, Any]:
        """Get performance trend for a query"""
        if query_hash not in self.query_performance_history:
            return {"trend": "unknown", "volatility": 0.0}

        perf = self.query_performance_history[query_hash]

        trend = {
            "executions": perf["executions"],
            "avg_time_ms": perf["avg_time_ms"],
            "trend": "stable",
            "volatility": 0.0,
        }

        # Calculate volatility
        if perf["avg_time_ms"] > 0:
            volatility = (perf["max_time_ms"] - perf["min_time_ms"]) / perf[
                "avg_time_ms"
            ]
            trend["volatility"] = volatility

            if volatility > 2.0:
                trend["trend"] = "highly_volatile"
            elif volatility > 1.0:
                trend["trend"] = "volatile"
            else:
                trend["trend"] = "stable"

        return trend

    def export_statistics(self, output_path: str, format: str = "json"):
        """Export current statistics to a file"""
        try:
            if format == "json":
                data = {
                    "tables": [
                        self._table_stats_to_dict(ts)
                        for ts in self.table_stats.values()
                    ],
                    "indexes": [
                        self._index_stats_to_dict(idx)
                        for idx_list in self.index_stats.values()
                        for idx in idx_list
                    ],
                    "columns": [
                        self._column_stats_to_dict(cs)
                        for col_dict in self.column_stats.values()
                        for cs in col_dict.values()
                    ],
                }

                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)

            elif format == "pickle":
                data = {
                    "table_stats": self.table_stats,
                    "index_stats": dict(self.index_stats),
                    "column_stats": dict(self.column_stats),
                    "relationship_stats": self.relationship_stats,
                    "workload_stats": self.workload_stats,
                }

                with open(output_path, "wb") as f:
                    pickle.dump(data, f)

            self.logger.info(f"Exported statistics to {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting statistics: {e}")

    def _table_stats_to_dict(self, stats: TableStatistics) -> Dict[str, Any]:
        """Convert TableStatistics to dictionary"""
        return {
            "name": stats.table_name,
            "row_count": stats.row_count,
            "page_count": stats.page_count,
            "avg_row_size": stats.avg_row_size,
            "total_size_mb": stats.total_size_mb,
            "last_analyzed": stats.last_analyzed.isoformat(),
            "fragmentation": stats.fragmentation_percentage,
            "clustered_index": stats.clustered_index,
        }

    def _index_stats_to_dict(self, stats: IndexStatistics) -> Dict[str, Any]:
        """Convert IndexStatistics to dictionary"""
        return {
            "name": stats.index_name,
            "table": stats.table_name,
            "columns": stats.columns,
            "unique": stats.unique,
            "clustered": stats.clustered,
            "depth": stats.index_depth,
            "leaf_pages": stats.leaf_pages,
            "fragmentation": stats.fragmentation_percentage,
            "avg_key_size": stats.avg_key_size,
            "selectivity": stats.selectivity,
        }

    def _column_stats_to_dict(self, stats: ColumnStatistics) -> Dict[str, Any]:
        """Convert ColumnStatistics to dictionary"""
        return {
            "name": stats.column_name,
            "table": stats.table_name,
            "data_type": stats.data_type,
            "nullable": stats.nullable,
            "distinct_values": stats.distinct_values,
            "null_percentage": stats.null_percentage,
            "avg_length": stats.avg_length,
            "max_length": stats.max_length,
            "min_value": stats.min_value,
            "max_value": stats.max_value,
            "distribution": stats.distribution.value,
        }


def generate_context_aware_features(
    query: str, stats_manager: DatabaseStatisticsManager
) -> Dict[str, Any]:
    """
    Generate context-aware features for a query using database statistics
    """
    features = {
        "table_features": {},
        "index_features": {},
        "selectivity_features": {},
        "cardinality_features": {},
        "performance_features": {},
        "optimization_hints": [],
    }

    # Extract tables from query (simplified)
    table_pattern = re.compile(r"FROM\s+([^\s,]+)|JOIN\s+([^\s]+)", re.IGNORECASE)
    tables = [match[0] or match[1] for match in table_pattern.findall(query)]

    for table in tables:
        # Get table features
        table_stats = stats_manager.get_table_statistics(table)
        if table_stats:
            features["table_features"][table] = {
                "row_count": table_stats.row_count,
                "size_mb": table_stats.total_size_mb,
                "fragmentation": table_stats.fragmentation_percentage,
            }

        # Get index features
        indexes = stats_manager.get_index_statistics(table)
        if indexes:
            features["index_features"][table] = [
                {
                    "name": idx.index_name,
                    "columns": idx.columns,
                    "selectivity": idx.selectivity,
                    "usage_count": idx.usage_count,
                }
                for idx in indexes
            ]

    # Extract WHERE conditions and estimate selectivity
    where_match = re.search(
        r"WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)", query, re.IGNORECASE
    )
    if where_match:
        where_clause = where_match.group(1)
        # Simple parsing of conditions
        conditions = re.split(r"\s+AND\s+|\s+OR\s+", where_clause, flags=re.IGNORECASE)

        for condition in conditions:
            # Try to extract column, operator, value
            match = re.match(
                r"(\w+\.)?(\w+)\s*([=<>]+|LIKE|IN|BETWEEN)\s*(.+)",
                condition,
                re.IGNORECASE,
            )
            if match:
                table_alias, column, operator, value = match.groups()
                # Estimate selectivity
                for table in tables:
                    selectivity = stats_manager.estimate_selectivity(
                        table, column, operator, value
                    )
                    features["selectivity_features"][f"{table}.{column}"] = selectivity

    # Calculate query hash for performance tracking
    query_hash = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()
    perf_trend = stats_manager.get_query_performance_trend(query_hash)
    features["performance_features"] = perf_trend

    # Add optimization hints based on statistics
    for table in tables:
        # Check for missing indexes
        table_stats = stats_manager.get_table_statistics(table)
        if table_stats and table_stats.row_count > 10000:
            if not stats_manager.get_index_statistics(table):
                features["optimization_hints"].append(f"Table {table} has no indexes")

        # Check for data skew
        for column_name in stats_manager.column_stats.get(table, {}).keys():
            skew_analysis = stats_manager.analyze_data_skew(table, column_name)
            if skew_analysis["skew_detected"]:
                features["optimization_hints"].extend(skew_analysis["recommendations"])

    return features


if __name__ == "__main__":
    # Example usage
    manager = DatabaseStatisticsManager()

    # Create sample statistics
    sample_stats = {
        "tables": [
            {
                "name": "customers",
                "row_count": 100000,
                "page_count": 5000,
                "avg_row_size": 200,
                "total_size_mb": 20.0,
                "last_analyzed": datetime.now().isoformat(),
                "fragmentation": 5.0,
                "clustered_index": "customer_id",
            },
            {
                "name": "orders",
                "row_count": 1000000,
                "page_count": 50000,
                "avg_row_size": 150,
                "total_size_mb": 150.0,
                "last_analyzed": datetime.now().isoformat(),
                "fragmentation": 10.0,
                "clustered_index": "order_id",
            },
        ],
        "indexes": [
            {
                "name": "idx_customer_email",
                "table": "customers",
                "columns": ["email"],
                "unique": True,
                "clustered": False,
                "depth": 3,
                "leaf_pages": 100,
                "fragmentation": 2.0,
                "avg_key_size": 30,
                "selectivity": 0.99,
            }
        ],
        "columns": [
            {
                "name": "customer_id",
                "table": "customers",
                "data_type": "int",
                "nullable": False,
                "distinct_values": 100000,
                "null_percentage": 0.0,
                "avg_length": 4,
                "max_length": 4,
                "min_value": 1,
                "max_value": 100000,
                "distribution": "uniform",
            },
            {
                "name": "order_date",
                "table": "orders",
                "data_type": "datetime",
                "nullable": False,
                "distinct_values": 365,
                "null_percentage": 0.0,
                "avg_length": 8,
                "max_length": 8,
                "min_value": "2023-01-01",
                "max_value": "2023-12-31",
                "distribution": "uniform",
            },
        ],
    }

    # Save sample statistics
    with open("/tmp/sample_db_stats.json", "w") as f:
        json.dump(sample_stats, f, indent=2)

    # Load and use statistics
    manager.load_statistics("/tmp/sample_db_stats.json", format="json")

    # Test query
    test_query = "SELECT * FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date >= '2023-06-01'"

    features = generate_context_aware_features(test_query, manager)

    print("=== Context-Aware Features ===")
    for category, data in features.items():
        print(f"\n{category.upper()}:")
        print(json.dumps(data, indent=2, default=str))
