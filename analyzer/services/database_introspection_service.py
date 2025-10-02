"""
Database Introspection Service

Handles all business logic related to database introspection including:
- Database connection management
- Schema analysis
- Index recommendations
- Foreign key analysis
- Performance recommendations
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConnectionConfig:
    """DTO for database connection configuration"""
    engine: str
    name: str
    host: str = ''
    port: int = 0
    username: str = ''
    password: str = ''
    schema: str = ''


@dataclass
class SchemaAnalysisResult:
    """DTO for schema analysis results"""
    tables: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    missing_indexes: List[Dict[str, Any]]
    foreign_key_issues: List[Dict[str, Any]]


class DatabaseIntrospectionService:
    """
    Service for handling database introspection and analysis.

    This service encapsulates the business logic for:
    - Connecting to databases
    - Analyzing schema structure
    - Generating optimization recommendations
    - Identifying missing indexes
    - Analyzing foreign key relationships
    """

    def __init__(self):
        """Initialize the database introspection service."""
        self.introspector = None

    def connect_to_database(self, config: DatabaseConnectionConfig) -> tuple[bool, Optional[str]]:
        """
        Connect to a database using the provided configuration.

        Args:
            config: DatabaseConnectionConfig containing connection details

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        from ..database_introspector import DatabaseIntrospector

        try:
            db_config = {
                'engine': config.engine,
                'name': config.name,
                'host': config.host,
                'port': config.port,
                'user': config.username,
                'password': config.password,
                'schema': config.schema
            }

            self.introspector = DatabaseIntrospector(db_config)

            if self.introspector.connect():
                logger.info(f"Successfully connected to {config.engine} database '{config.name}'")
                return True, None
            else:
                return False, "Failed to connect to the database. Please check your connection parameters."

        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False, f"Database connection error: {str(e)}"

    def analyze_schema(self, schema: Optional[str] = None) -> SchemaAnalysisResult:
        """
        Analyze database schema and generate recommendations.

        Args:
            schema: Optional schema name to analyze

        Returns:
            SchemaAnalysisResult containing analysis details

        Raises:
            ValueError: If not connected to a database
        """
        if not self.introspector:
            raise ValueError("Not connected to a database. Call connect_to_database first.")

        try:
            # Get all tables in the database
            tables = self.introspector.get_tables(schema)

            # Analyze each table
            analyzed_tables = []
            all_recommendations = []
            missing_indexes = []
            foreign_key_issues = []

            for table_name in tables:
                # Get table details
                columns = self.introspector.get_columns(table_name, schema)
                indexes = self.introspector.get_indexes(table_name, schema)
                foreign_keys = self.introspector.get_foreign_keys(table_name, schema)

                table_info = {
                    'name': table_name,
                    'columns': columns,
                    'indexes': indexes,
                    'foreign_keys': foreign_keys,
                    'column_count': len(columns),
                    'index_count': len(indexes)
                }

                analyzed_tables.append(table_info)

                # Generate recommendations for this table
                table_recommendations = self._analyze_table(
                    table_name,
                    columns,
                    indexes,
                    foreign_keys
                )

                all_recommendations.extend(table_recommendations)

                # Check for missing indexes
                missing = self._find_missing_indexes(table_name, columns, indexes, foreign_keys)
                missing_indexes.extend(missing)

                # Check for foreign key issues
                fk_issues = self._analyze_foreign_keys(table_name, foreign_keys, indexes)
                foreign_key_issues.extend(fk_issues)

            # Calculate statistics
            statistics = self._calculate_statistics(analyzed_tables)

            return SchemaAnalysisResult(
                tables=analyzed_tables,
                recommendations=all_recommendations,
                statistics=statistics,
                missing_indexes=missing_indexes,
                foreign_key_issues=foreign_key_issues
            )

        except Exception as e:
            logger.error(f"Error analyzing schema: {e}")
            raise

    def close_connection(self) -> None:
        """Close the database connection."""
        if self.introspector:
            self.introspector.close()
            self.introspector = None

    def _analyze_table(
        self,
        table_name: str,
        columns: List[Dict],
        indexes: List[Dict],
        foreign_keys: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Analyze a table and generate recommendations."""
        recommendations = []

        # Check if table has a primary key
        has_pk = any(col.get('primary_key') for col in columns)
        if not has_pk:
            recommendations.append({
                'severity': 'high',
                'table': table_name,
                'type': 'missing_primary_key',
                'message': f"Table '{table_name}' does not have a primary key. Consider adding one for data integrity.",
                'recommendation': f"ALTER TABLE {table_name} ADD PRIMARY KEY (id);"
            })

        # Check for large VARCHAR fields without indexes
        for col in columns:
            if col.get('type', '').upper().startswith('VARCHAR'):
                length = col.get('max_length', 0)
                if length > 255 and not self._column_has_index(col['name'], indexes):
                    recommendations.append({
                        'severity': 'medium',
                        'table': table_name,
                        'column': col['name'],
                        'type': 'large_varchar_no_index',
                        'message': f"Large VARCHAR({length}) column '{col['name']}' in '{table_name}' may benefit from an index if used in WHERE clauses.",
                        'recommendation': f"CREATE INDEX idx_{table_name}_{col['name']} ON {table_name} ({col['name']});"
                    })

        # Check for timestamp columns without indexes
        timestamp_cols = [
            col for col in columns
            if any(t in col.get('type', '').upper() for t in ['TIMESTAMP', 'DATETIME', 'DATE'])
        ]
        for col in timestamp_cols:
            if not self._column_has_index(col['name'], indexes):
                recommendations.append({
                    'severity': 'low',
                    'table': table_name,
                    'column': col['name'],
                    'type': 'timestamp_no_index',
                    'message': f"Timestamp column '{col['name']}' in '{table_name}' often benefits from indexing for date range queries.",
                    'recommendation': f"CREATE INDEX idx_{table_name}_{col['name']} ON {table_name} ({col['name']});"
                })

        return recommendations

    def _find_missing_indexes(
        self,
        table_name: str,
        columns: List[Dict],
        indexes: List[Dict],
        foreign_keys: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Find foreign keys without corresponding indexes."""
        missing_indexes = []

        for fk in foreign_keys:
            fk_column = fk.get('column')
            if fk_column and not self._column_has_index(fk_column, indexes):
                missing_indexes.append({
                    'table': table_name,
                    'column': fk_column,
                    'references': fk.get('references'),
                    'recommendation': f"CREATE INDEX idx_{table_name}_{fk_column} ON {table_name} ({fk_column});"
                })

        return missing_indexes

    def _analyze_foreign_keys(
        self,
        table_name: str,
        foreign_keys: List[Dict],
        indexes: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Analyze foreign keys for issues."""
        issues = []

        for fk in foreign_keys:
            # Check if foreign key has an index
            fk_column = fk.get('column')
            if fk_column and not self._column_has_index(fk_column, indexes):
                issues.append({
                    'table': table_name,
                    'column': fk_column,
                    'issue': 'missing_index',
                    'severity': 'high',
                    'message': f"Foreign key column '{fk_column}' in '{table_name}' should have an index for optimal JOIN performance."
                })

        return issues

    def _column_has_index(self, column_name: str, indexes: List[Dict]) -> bool:
        """Check if a column has an index."""
        for index in indexes:
            # Check if column is the first column in the index
            # (composite indexes are less effective for this column alone)
            columns = index.get('columns', [])
            if columns and columns[0] == column_name:
                return True
        return False

    def _calculate_statistics(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate database statistics."""
        total_columns = sum(table['column_count'] for table in tables)
        total_indexes = sum(table['index_count'] for table in tables)
        total_foreign_keys = sum(len(table['foreign_keys']) for table in tables)

        return {
            'total_tables': len(tables),
            'total_columns': total_columns,
            'total_indexes': total_indexes,
            'total_foreign_keys': total_foreign_keys,
            'avg_columns_per_table': round(total_columns / len(tables), 2) if tables else 0,
            'avg_indexes_per_table': round(total_indexes / len(tables), 2) if tables else 0
        }
