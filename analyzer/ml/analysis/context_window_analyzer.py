"""
Context Window Analyzer

Advanced analysis of multi-statement SQL queries with semantic understanding of:
- Statement parsing and execution flow
- Transaction boundary detection
- Statement dependencies and data flow
- Context propagation between statements
- Batch vs sequential execution patterns
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


class StatementType(Enum):
    """Classification of SQL statement types"""

    SELECT = "select"  # Data retrieval
    INSERT = "insert"  # Data insertion
    UPDATE = "update"  # Data modification
    DELETE = "delete"  # Data deletion
    CREATE = "create"  # Object creation
    ALTER = "alter"  # Object modification
    DROP = "drop"  # Object deletion
    TRUNCATE = "truncate"  # Table truncation
    BEGIN = "begin"  # Transaction start
    COMMIT = "commit"  # Transaction commit
    ROLLBACK = "rollback"  # Transaction rollback
    DECLARE = "declare"  # Variable declaration
    CALL = "call"  # Procedure/function call
    UNKNOWN = "unknown"  # Cannot determine type


class TransactionScope(Enum):
    """Transaction scope classification"""

    AUTO_COMMIT = "auto_commit"  # Individual statements auto-commit
    EXPLICIT = "explicit"  # BEGIN...COMMIT/ROLLBACK
    IMPLICIT = "implicit"  # Implicit transaction with auto-commit
    UNKNOWN = "unknown"  # Cannot determine scope


@dataclass
class SQLStatement:
    """Represents a single SQL statement in a batch"""

    statement_text: str
    position: int  # Position in batch (0-indexed)
    statement_type: StatementType = StatementType.UNKNOWN
    tables_read: Set[str] = field(default_factory=set)
    tables_written: Set[str] = field(default_factory=set)
    complexity_score: float = 0.0
    line_count: int = 0
    character_count: int = 0
    estimated_rows_affected: int = 0
    has_transaction_control: bool = False


@dataclass
class ContextFlow:
    """Represents data flow between statements"""

    from_statement_idx: int
    to_statement_idx: int
    flow_type: str  # "read_after_write", "write_dependency", "data_dependency"
    affected_tables: Set[str] = field(default_factory=set)


@dataclass
class ContextWindowAnalysis:
    """Complete analysis of multi-statement query context"""

    total_statement_count: int = 0
    statements: List[SQLStatement] = field(default_factory=list)
    statement_types: Dict[str, int] = field(default_factory=dict)  # type -> count
    transaction_scope: TransactionScope = TransactionScope.UNKNOWN
    has_explicit_transaction: bool = False
    transaction_complexity: float = 0.0
    data_flows: List[ContextFlow] = field(default_factory=list)
    tables_accessed: Set[str] = field(default_factory=set)
    overall_complexity_score: float = 0.0
    execution_risk_level: str = "low"  # low/medium/high/critical
    optimization_opportunities: List[str] = field(default_factory=list)
    batch_vs_sequential_recommendation: str = "sequential"  # batch or sequential
    cumulative_impact: Dict[str, float] = field(default_factory=dict)  # metric -> value


class ContextWindowAnalyzer:
    """Advanced analyzer for multi-statement SQL query context"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for statement analysis"""
        # Statement type patterns
        self.select_pattern = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
        self.insert_pattern = re.compile(r"^\s*INSERT\b", re.IGNORECASE)
        self.update_pattern = re.compile(r"^\s*UPDATE\b", re.IGNORECASE)
        self.delete_pattern = re.compile(r"^\s*DELETE\b", re.IGNORECASE)
        self.create_pattern = re.compile(r"^\s*CREATE\b", re.IGNORECASE)
        self.alter_pattern = re.compile(r"^\s*ALTER\b", re.IGNORECASE)
        self.drop_pattern = re.compile(r"^\s*DROP\b", re.IGNORECASE)
        self.truncate_pattern = re.compile(r"^\s*TRUNCATE\b", re.IGNORECASE)
        self.begin_pattern = re.compile(r"^\s*BEGIN\b", re.IGNORECASE)
        self.commit_pattern = re.compile(r"^\s*COMMIT\b", re.IGNORECASE)
        self.rollback_pattern = re.compile(r"^\s*ROLLBACK\b", re.IGNORECASE)
        self.declare_pattern = re.compile(r"^\s*DECLARE\b", re.IGNORECASE)
        self.call_pattern = re.compile(r"^\s*CALL\b", re.IGNORECASE)

        # FROM/INTO pattern for table extraction
        self.table_pattern = re.compile(
            r"\b(?:FROM|INTO|UPDATE|JOIN|DELETE\s+FROM)\s+(?:\w+\.)?(\w+)",
            re.IGNORECASE,
        )

    def analyze_context_window(self, query: str) -> ContextWindowAnalysis:
        """Analyze multi-statement query context"""
        try:
            analysis = ContextWindowAnalysis()

            # Split statements by semicolon
            statements = self._split_statements(query)

            if len(statements) == 0:
                return analysis

            # Analyze each statement
            for idx, stmt_text in enumerate(statements):
                stmt = SQLStatement(
                    statement_text=stmt_text,
                    position=idx,
                    line_count=stmt_text.count("\n") + 1,
                    character_count=len(stmt_text),
                )

                # Classify type
                stmt.statement_type = self._classify_statement_type(stmt_text)

                # Extract table references
                stmt.tables_read = self._extract_read_tables(stmt_text)
                stmt.tables_written = self._extract_write_tables(stmt_text)

                # Calculate complexity
                stmt.complexity_score = self._calculate_statement_complexity(stmt)

                # Check for transaction control
                stmt.has_transaction_control = self._has_transaction_control(stmt_text)

                analysis.statements.append(stmt)
                analysis.tables_accessed.update(stmt.tables_read)
                analysis.tables_accessed.update(stmt.tables_written)

            analysis.total_statement_count = len(statements)

            # Collect statistics
            for stmt in analysis.statements:
                type_name = stmt.statement_type.value
                analysis.statement_types[type_name] = (
                    analysis.statement_types.get(type_name, 0) + 1
                )

            # Detect transaction scope
            analysis.transaction_scope = self._detect_transaction_scope(
                analysis.statements
            )
            analysis.has_explicit_transaction = (
                analysis.transaction_scope == TransactionScope.EXPLICIT
            )

            # Build data flows
            analysis.data_flows = self._build_data_flows(analysis.statements)

            # Calculate overall complexity
            analysis.overall_complexity_score = self._calculate_overall_complexity(
                analysis.statements
            )

            # Assess execution risk
            analysis.execution_risk_level = self._assess_execution_risk(analysis)

            # Generate recommendations
            analysis.optimization_opportunities = self._generate_recommendations(
                analysis
            )
            analysis.batch_vs_sequential_recommendation = (
                self._recommend_execution_mode(analysis)
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing context window: {e}")
            return ContextWindowAnalysis()

    def _split_statements(self, query: str) -> List[str]:
        """Split query into individual statements"""
        statements = []
        current = ""
        in_string = False
        string_char = None
        paren_depth = 0

        for i, char in enumerate(query):
            # Handle string literals
            if char in ('"', "'") and (i == 0 or query[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False

            # Only process semicolon outside strings
            if not in_string:
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                elif char == ";" and paren_depth == 0:
                    stmt = current.strip()
                    if stmt:
                        statements.append(stmt)
                    current = ""
                    continue

            current += char

        # Add final statement if exists
        stmt = current.strip()
        if stmt:
            statements.append(stmt)

        return statements

    def _classify_statement_type(self, stmt_text: str) -> StatementType:
        """Classify the type of SQL statement"""
        stmt_upper = stmt_text.upper().strip()

        if self.select_pattern.match(stmt_upper):
            return StatementType.SELECT
        elif self.insert_pattern.match(stmt_upper):
            return StatementType.INSERT
        elif self.update_pattern.match(stmt_upper):
            return StatementType.UPDATE
        elif self.delete_pattern.match(stmt_upper):
            return StatementType.DELETE
        elif self.create_pattern.match(stmt_upper):
            return StatementType.CREATE
        elif self.alter_pattern.match(stmt_upper):
            return StatementType.ALTER
        elif self.drop_pattern.match(stmt_upper):
            return StatementType.DROP
        elif self.truncate_pattern.match(stmt_upper):
            return StatementType.TRUNCATE
        elif self.begin_pattern.match(stmt_upper):
            return StatementType.BEGIN
        elif self.commit_pattern.match(stmt_upper):
            return StatementType.COMMIT
        elif self.rollback_pattern.match(stmt_upper):
            return StatementType.ROLLBACK
        elif self.declare_pattern.match(stmt_upper):
            return StatementType.DECLARE
        elif self.call_pattern.match(stmt_upper):
            return StatementType.CALL
        else:
            return StatementType.UNKNOWN

    def _extract_read_tables(self, stmt_text: str) -> Set[str]:
        """Extract tables being read in SELECT/FROM"""
        if "SELECT" not in stmt_text.upper():
            return set()

        tables = set()
        for match in self.table_pattern.finditer(stmt_text):
            if (
                "FROM"
                in stmt_text.upper()[max(0, match.start() - 20) : match.start()].upper()
                or "JOIN"
                in stmt_text.upper()[max(0, match.start() - 20) : match.start()].upper()
            ):
                tables.add(match.group(1))

        return tables

    def _extract_write_tables(self, stmt_text: str) -> Set[str]:
        """Extract tables being written to (INSERT/UPDATE/DELETE)"""
        tables = set()

        # INSERT INTO table
        insert_match = re.search(r"INSERT\s+INTO\s+(\w+)", stmt_text, re.IGNORECASE)
        if insert_match:
            tables.add(insert_match.group(1))

        # UPDATE table
        update_match = re.search(r"UPDATE\s+(\w+)", stmt_text, re.IGNORECASE)
        if update_match:
            tables.add(update_match.group(1))

        # DELETE FROM table
        delete_match = re.search(r"DELETE\s+FROM\s+(\w+)", stmt_text, re.IGNORECASE)
        if delete_match:
            tables.add(delete_match.group(1))

        return tables

    def _calculate_statement_complexity(self, stmt: SQLStatement) -> float:
        """Calculate complexity score for a statement"""
        score = 0.0

        # Type complexity
        if stmt.statement_type == StatementType.SELECT:
            score += 0.2
        elif stmt.statement_type in [
            StatementType.INSERT,
            StatementType.UPDATE,
            StatementType.DELETE,
        ]:
            score += 0.15
        elif stmt.statement_type in [StatementType.CREATE, StatementType.ALTER]:
            score += 0.1
        elif stmt.statement_type == StatementType.TRUNCATE:
            score += 0.05

        # Size factor (normalize to 0-0.3)
        line_factor = min(0.3, stmt.line_count / 50)
        score += line_factor

        # Character count factor
        char_factor = min(0.2, stmt.character_count / 1000)
        score += char_factor

        return min(1.0, score)

    def _has_transaction_control(self, stmt_text: str) -> bool:
        """Check if statement has transaction control keywords"""
        stmt_upper = stmt_text.upper()
        return bool(re.search(r"\b(BEGIN|COMMIT|ROLLBACK|SAVEPOINT)\b", stmt_upper))

    def _detect_transaction_scope(
        self, statements: List[SQLStatement]
    ) -> TransactionScope:
        """Detect transaction scope from statements"""
        has_begin = any(s.statement_type == StatementType.BEGIN for s in statements)
        has_commit = any(s.statement_type == StatementType.COMMIT for s in statements)
        has_rollback = any(
            s.statement_type == StatementType.ROLLBACK for s in statements
        )

        if has_begin and (has_commit or has_rollback):
            return TransactionScope.EXPLICIT

        if len(statements) == 1:
            return TransactionScope.AUTO_COMMIT

        return TransactionScope.IMPLICIT

    def _build_data_flows(self, statements: List[SQLStatement]) -> List[ContextFlow]:
        """Build data flow dependencies between statements"""
        flows = []

        for i in range(len(statements) - 1):
            current = statements[i]
            next_stmt = statements[i + 1]

            # Find tables written by current and read by next
            overlapping_tables = current.tables_written & next_stmt.tables_read

            if overlapping_tables:
                flow = ContextFlow(
                    from_statement_idx=i,
                    to_statement_idx=i + 1,
                    flow_type="read_after_write",
                    affected_tables=overlapping_tables,
                )
                flows.append(flow)

        return flows

    def _calculate_overall_complexity(self, statements: List[SQLStatement]) -> float:
        """Calculate overall complexity for multi-statement batch"""
        if not statements:
            return 0.0

        avg_complexity = sum(s.complexity_score for s in statements) / len(statements)

        # Batch complexity factor (more statements = higher complexity)
        batch_factor = min(0.3, len(statements) * 0.05)

        # Mixed statement types factor
        types = set(
            s.statement_type
            for s in statements
            if s.statement_type != StatementType.UNKNOWN
        )
        type_variety = min(0.2, len(types) * 0.05)

        return min(1.0, avg_complexity + batch_factor + type_variety)

    def _assess_execution_risk(self, analysis: ContextWindowAnalysis) -> str:
        """Assess execution risk for multi-statement batch"""
        risk_score = 0.0

        # Explicit transaction complexity
        if analysis.has_explicit_transaction:
            risk_score += 0.2

        # Write operations risk
        write_stmts = [s for s in analysis.statements if s.tables_written]
        if len(write_stmts) > 1:
            risk_score += min(0.3, len(write_stmts) * 0.1)

        # Complex dependencies
        if len(analysis.data_flows) > 0:
            risk_score += min(0.2, len(analysis.data_flows) * 0.05)

        # Overall complexity
        risk_score += analysis.overall_complexity_score * 0.2

        # Determine risk level
        if risk_score >= 0.7:
            return "critical"
        elif risk_score >= 0.5:
            return "high"
        elif risk_score >= 0.25:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self, analysis: ContextWindowAnalysis) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Multiple write operations
        write_stmts = [s for s in analysis.statements if s.tables_written]
        if len(write_stmts) > 2:
            recommendations.append(
                "Multiple write operations detected - verify they don't have conflicts"
            )

        # Circular dependencies
        if self._has_circular_dependencies(analysis.data_flows):
            recommendations.append(
                "Circular data dependencies detected - review statement order"
            )

        # Large batch
        if len(analysis.statements) > 10:
            recommendations.append(
                "Large batch of statements - consider breaking into smaller transactions"
            )

        # No explicit transaction with writes
        if not analysis.has_explicit_transaction and len(write_stmts) > 1:
            recommendations.append(
                "Multiple writes without explicit transaction - consider adding BEGIN/COMMIT"
            )

        return recommendations

    def _has_circular_dependencies(self, flows: List[ContextFlow]) -> bool:
        """Check for circular dependencies in data flows"""
        # Simple circular detection: if A depends on B and B depends on A
        for flow in flows:
            reverse_flows = [
                f for f in flows if f.from_statement_idx == flow.to_statement_idx
            ]
            if any(
                f.to_statement_idx == flow.from_statement_idx for f in reverse_flows
            ):
                return True
        return False

    def _recommend_execution_mode(self, analysis: ContextWindowAnalysis) -> str:
        """Recommend batch vs sequential execution"""
        # If statements are independent, can batch
        if len(analysis.data_flows) == 0 and not analysis.has_explicit_transaction:
            return "batch"
        return "sequential"

    def get_execution_flow_summary(self, analysis: ContextWindowAnalysis) -> Dict:
        """Get a summary of execution flow"""
        return {
            "total_statements": analysis.total_statement_count,
            "statement_types": analysis.statement_types,
            "transaction_scope": analysis.transaction_scope.value,
            "data_dependencies": len(analysis.data_flows),
            "tables_accessed": len(analysis.tables_accessed),
            "execution_mode": analysis.batch_vs_sequential_recommendation,
            "risk_level": analysis.execution_risk_level,
        }
