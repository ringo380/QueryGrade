"""
Query Mutation Engine for QueryGrade ML System

This module provides sophisticated SQL query mutation capabilities to generate
semantically equivalent but syntactically different query variations for training data augmentation.
"""

import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import sqlparse
from sqlparse import keywords

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of mutations that can be applied to queries."""

    ALIAS_VARIATION = "alias_variation"
    CASE_VARIATION = "case_variation"
    WHITESPACE_VARIATION = "whitespace_variation"
    KEYWORD_SYNONYM = "keyword_synonym"
    CONDITION_REORDER = "condition_reorder"
    JOIN_SYNTAX = "join_syntax"
    SUBQUERY_TRANSFORMATION = "subquery_transformation"
    FUNCTION_EQUIVALENT = "function_equivalent"
    LITERAL_VARIATION = "literal_variation"
    PARENTHESES_VARIATION = "parentheses_variation"


@dataclass
class MutationRule:
    """A rule defining how to mutate queries."""

    mutation_type: MutationType
    pattern: str  # Regex pattern to match
    replacement_options: List[str]  # Possible replacements
    probability: float = 1.0  # Probability of applying this mutation
    preserve_semantics: bool = True  # Whether this preserves query semantics
    complexity_impact: int = 0  # How this affects query complexity (-5 to +5)
    database_specific: List[str] = None  # Database types this applies to


@dataclass
class MutationResult:
    """Result of applying mutations to a query."""

    original_query: str
    mutated_query: str
    mutations_applied: List[MutationType]
    semantic_preserved: bool
    complexity_change: int
    success: bool
    error_message: Optional[str] = None


class QueryAliasGenerator:
    """Generates meaningful aliases for tables and columns."""

    def __init__(self):
        self.table_aliases = {
            "users": ["u", "usr", "user_tbl", "people"],
            "orders": ["o", "ord", "order_tbl", "purchases"],
            "products": ["p", "prod", "item", "product_tbl"],
            "customers": ["c", "cust", "client", "customer_tbl"],
            "employees": ["e", "emp", "staff", "employee_tbl"],
            "departments": ["d", "dept", "division", "department_tbl"],
        }

        self.column_aliases = {
            "id": ["pk", "key", "identifier"],
            "name": ["nm", "title", "label"],
            "email": ["mail", "email_addr", "e_mail"],
            "created_at": ["created", "create_time", "creation_date"],
            "updated_at": ["updated", "update_time", "last_modified"],
        }

    def generate_table_alias(self, table_name: str) -> str:
        """Generate an alias for a table name."""
        table_lower = table_name.lower()
        if table_lower in self.table_aliases:
            return random.choice(self.table_aliases[table_lower])
        else:
            # Generate short alias from table name
            if len(table_name) <= 2:
                return table_name
            elif len(table_name) <= 4:
                return table_name[:2]
            else:
                return table_name[0] + table_name[len(table_name) // 2]

    def generate_column_alias(self, column_name: str) -> str:
        """Generate an alias for a column name."""
        column_lower = column_name.lower()
        if column_lower in self.column_aliases:
            return random.choice(self.column_aliases[column_lower])
        else:
            # Generate short alias from column name
            if len(column_name) <= 3:
                return column_name
            else:
                return column_name[:3]


class QueryMutationEngine:
    """Main engine for mutating SQL queries."""

    def __init__(self):
        self.alias_generator = QueryAliasGenerator()
        self.mutation_rules = self._initialize_mutation_rules()
        self.keyword_synonyms = self._initialize_keyword_synonyms()

    def _initialize_mutation_rules(self) -> List[MutationRule]:
        """Initialize the set of mutation rules."""
        rules = [
            # Case variations
            MutationRule(
                mutation_type=MutationType.CASE_VARIATION,
                pattern=r"\b(SELECT|FROM|WHERE|JOIN|GROUP|ORDER|BY|HAVING|UNION|INSERT|UPDATE|DELETE)\b",
                replacement_options=["lowercase", "uppercase", "mixed"],
                probability=0.8,
                preserve_semantics=True,
            ),
            # Alias variations
            MutationRule(
                mutation_type=MutationType.ALIAS_VARIATION,
                pattern=r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b",
                replacement_options=["add_alias", "remove_alias", "change_alias"],
                probability=0.6,
                preserve_semantics=True,
            ),
            # JOIN syntax variations
            MutationRule(
                mutation_type=MutationType.JOIN_SYNTAX,
                pattern=r"\bINNER\s+JOIN\b",
                replacement_options=["JOIN", "INNER JOIN"],
                probability=0.7,
                preserve_semantics=True,
            ),
            # Condition reordering
            MutationRule(
                mutation_type=MutationType.CONDITION_REORDER,
                pattern=r"WHERE\s+(.+?)(?:\s+(?:GROUP|ORDER|HAVING|LIMIT|$))",
                replacement_options=["reorder_and", "reorder_or"],
                probability=0.5,
                preserve_semantics=True,
            ),
            # Function equivalents
            MutationRule(
                mutation_type=MutationType.FUNCTION_EQUIVALENT,
                pattern=r"\bCOUNT\(\*\)\b",
                replacement_options=["COUNT(1)", "COUNT(*)"],
                probability=0.8,
                preserve_semantics=True,
            ),
            # Literal variations
            MutationRule(
                mutation_type=MutationType.LITERAL_VARIATION,
                pattern=r"'([^']+)'",
                replacement_options=["double_quotes", "single_quotes"],
                probability=0.6,
                preserve_semantics=True,
                database_specific=["mysql", "sqlite"],
            ),
        ]
        return rules

    def _initialize_keyword_synonyms(self) -> Dict[str, List[str]]:
        """Initialize keyword synonyms for different databases."""
        return {
            "INNER JOIN": ["JOIN", "INNER JOIN"],
            "LEFT OUTER JOIN": ["LEFT JOIN", "LEFT OUTER JOIN"],
            "RIGHT OUTER JOIN": ["RIGHT JOIN", "RIGHT OUTER JOIN"],
            "COUNT(*)": ["COUNT(*)", "COUNT(1)"],
            "AUTOINCREMENT": ["AUTO_INCREMENT", "AUTOINCREMENT"],  # SQLite vs MySQL
            "LIMIT": ["LIMIT", "TOP"],  # Standard vs SQL Server
        }

    def mutate_query(
        self, query: str, num_mutations: int = 3, database_type: str = "generic"
    ) -> List[MutationResult]:
        """
        Generate multiple mutations of a SQL query.

        Args:
            query: Original SQL query
            num_mutations: Number of mutations to generate
            database_type: Target database type

        Returns:
            List of mutation results
        """
        results = []

        for i in range(num_mutations):
            try:
                # Select random mutations to apply
                mutations_to_apply = self._select_mutations(database_type)

                # Apply mutations
                mutated_query = query
                applied_mutations = []
                complexity_change = 0
                semantic_preserved = True

                for mutation_type in mutations_to_apply:
                    mutation_result = self._apply_mutation(
                        mutated_query, mutation_type, database_type
                    )

                    if mutation_result["success"]:
                        mutated_query = mutation_result["query"]
                        applied_mutations.append(mutation_type)
                        complexity_change += mutation_result.get("complexity_change", 0)
                        semantic_preserved = semantic_preserved and mutation_result.get(
                            "semantic_preserved", True
                        )

                # Validate the mutated query
                is_valid = self._validate_mutated_query(mutated_query)

                result = MutationResult(
                    original_query=query,
                    mutated_query=mutated_query,
                    mutations_applied=applied_mutations,
                    semantic_preserved=semantic_preserved and is_valid,
                    complexity_change=complexity_change,
                    success=is_valid and len(applied_mutations) > 0,
                )

                results.append(result)

            except Exception as e:
                logger.warning(f"Error in mutation {i}: {e}")
                results.append(
                    MutationResult(
                        original_query=query,
                        mutated_query=query,
                        mutations_applied=[],
                        semantic_preserved=False,
                        complexity_change=0,
                        success=False,
                        error_message=str(e),
                    )
                )

        return results

    def _select_mutations(self, database_type: str) -> List[MutationType]:
        """Select which mutations to apply based on probability and database type."""
        selected = []

        for rule in self.mutation_rules:
            # Check database compatibility
            if rule.database_specific and database_type not in rule.database_specific:
                continue

            # Check probability
            if random.random() <= rule.probability:
                selected.append(rule.mutation_type)

        # Ensure we have at least one mutation
        if not selected and self.mutation_rules:
            selected.append(random.choice(self.mutation_rules).mutation_type)

        return selected[:3]  # Limit to 3 mutations per query

    def _apply_mutation(
        self, query: str, mutation_type: MutationType, database_type: str
    ) -> Dict[str, Any]:
        """Apply a specific mutation to a query."""
        try:
            if mutation_type == MutationType.CASE_VARIATION:
                return self._apply_case_variation(query)
            elif mutation_type == MutationType.ALIAS_VARIATION:
                return self._apply_alias_variation(query)
            elif mutation_type == MutationType.WHITESPACE_VARIATION:
                return self._apply_whitespace_variation(query)
            elif mutation_type == MutationType.KEYWORD_SYNONYM:
                return self._apply_keyword_synonym(query, database_type)
            elif mutation_type == MutationType.CONDITION_REORDER:
                return self._apply_condition_reorder(query)
            elif mutation_type == MutationType.JOIN_SYNTAX:
                return self._apply_join_syntax_variation(query)
            elif mutation_type == MutationType.FUNCTION_EQUIVALENT:
                return self._apply_function_equivalent(query)
            elif mutation_type == MutationType.LITERAL_VARIATION:
                return self._apply_literal_variation(query, database_type)
            elif mutation_type == MutationType.PARENTHESES_VARIATION:
                return self._apply_parentheses_variation(query)
            else:
                return {
                    "success": False,
                    "query": query,
                    "error": "Unknown mutation type",
                }

        except Exception as e:
            logger.warning(f"Error applying {mutation_type}: {e}")
            return {"success": False, "query": query, "error": str(e)}

    def _apply_case_variation(self, query: str) -> Dict[str, Any]:
        """Apply case variations to SQL keywords."""
        variation_type = random.choice(["lowercase", "uppercase", "mixed"])

        keywords_pattern = r"\b(SELECT|FROM|WHERE|JOIN|INNER|LEFT|RIGHT|OUTER|ON|GROUP|ORDER|BY|HAVING|UNION|ALL|DISTINCT|AS|AND|OR|NOT|IN|EXISTS|LIKE|BETWEEN|IS|NULL|COUNT|SUM|AVG|MIN|MAX|INSERT|INTO|VALUES|UPDATE|SET|DELETE|CREATE|TABLE|ALTER|DROP|INDEX)\b"  # noqa: E501

        def case_replacer(match):
            keyword = match.group(1)
            if variation_type == "lowercase":
                return keyword.lower()
            elif variation_type == "uppercase":
                return keyword.upper()
            else:  # mixed
                return (
                    keyword.capitalize() if random.random() > 0.5 else keyword.lower()
                )

        mutated_query = re.sub(
            keywords_pattern, case_replacer, query, flags=re.IGNORECASE
        )

        return {
            "success": True,
            "query": mutated_query,
            "semantic_preserved": True,
            "complexity_change": 0,
        }

    def _apply_alias_variation(self, query: str) -> Dict[str, Any]:
        """Apply table and column alias variations."""
        try:
            parsed = sqlparse.parse(query)[0]

            # Find table references and add/modify aliases
            tokens = list(parsed.flatten())
            mutated_tokens = []
            i = 0

            while i < len(tokens):
                token = tokens[i]

                # Look for table names after FROM or JOIN
                if token.ttype in keywords.Keyword and token.value.upper() in [
                    "FROM",
                    "JOIN",
                    "INNER",
                    "LEFT",
                    "RIGHT",
                ]:

                    mutated_tokens.append(token)

                    # Skip to the table name
                    j = i + 1
                    while j < len(tokens) and (
                        tokens[j].is_whitespace
                        or tokens[j].value.upper()
                        in ["INNER", "LEFT", "RIGHT", "OUTER", "JOIN"]
                    ):
                        mutated_tokens.append(tokens[j])
                        j += 1

                    # Add alias if we find a table name
                    if j < len(tokens) and tokens[j].ttype is None:
                        table_name = tokens[j].value
                        mutated_tokens.append(tokens[j])

                        # Check if there's already an alias
                        k = j + 1
                        while k < len(tokens) and tokens[k].is_whitespace:
                            mutated_tokens.append(tokens[k])
                            k += 1

                        # Add alias if none exists
                        if (
                            k >= len(tokens)
                            or tokens[k].ttype in keywords.Keyword
                            or tokens[k].value in [",", "(", ")"]
                        ):

                            alias = self.alias_generator.generate_table_alias(
                                table_name
                            )
                            mutated_tokens.append(
                                sqlparse.sql.Token(tokens.Whitespace, " ")
                            )
                            mutated_tokens.append(sqlparse.sql.Token(None, alias))

                        i = k - 1
                    else:
                        i = j - 1
                else:
                    mutated_tokens.append(token)

                i += 1

            mutated_query = "".join(str(token) for token in mutated_tokens)

            return {
                "success": True,
                "query": mutated_query,
                "semantic_preserved": True,
                "complexity_change": 0,
            }

        except Exception as e:
            logger.warning(f"Error in alias variation: {e}")
            return {"success": False, "query": query, "error": str(e)}

    def _apply_whitespace_variation(self, query: str) -> Dict[str, Any]:
        """Apply whitespace variations while preserving syntax."""
        variations = [
            # Normalize multiple spaces to single space
            lambda q: re.sub(r"\s+", " ", q),
            # Add extra spaces around operators
            lambda q: re.sub(r"([=<>!]+)", r" \1 ", q),
            # Normalize spaces around commas
            lambda q: re.sub(r"\s*,\s*", ", ", q),
            # Normalize spaces around parentheses
            lambda q: re.sub(r"\s*\(\s*", "(", q),
            lambda q: re.sub(r"\s*\)\s*", ")", q),
        ]

        variation = random.choice(variations)
        mutated_query = variation(query.strip())

        return {
            "success": True,
            "query": mutated_query,
            "semantic_preserved": True,
            "complexity_change": 0,
        }

    def _apply_keyword_synonym(self, query: str, database_type: str) -> Dict[str, Any]:
        """Apply keyword synonym replacements."""
        mutated_query = query

        for keyword, synonyms in self.keyword_synonyms.items():
            if keyword in query.upper():
                synonym = random.choice(synonyms)
                # Case-preserving replacement
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                mutated_query = pattern.sub(synonym, mutated_query)
                break

        return {
            "success": mutated_query != query,
            "query": mutated_query,
            "semantic_preserved": True,
            "complexity_change": 0,
        }

    def _apply_condition_reorder(self, query: str) -> Dict[str, Any]:
        """Reorder conditions in WHERE clauses."""
        where_pattern = r"WHERE\s+(.+?)(?=\s+(?:GROUP|ORDER|HAVING|LIMIT|UNION|$))"
        match = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)

        if not match:
            return {"success": False, "query": query, "error": "No WHERE clause found"}

        where_conditions = match.group(1).strip()

        # Split on AND/OR and reorder
        and_parts = re.split(r"\s+AND\s+", where_conditions, flags=re.IGNORECASE)
        if len(and_parts) > 1:
            random.shuffle(and_parts)
            new_where = " AND ".join(and_parts)
            mutated_query = query.replace(match.group(1), new_where)

            return {
                "success": True,
                "query": mutated_query,
                "semantic_preserved": True,
                "complexity_change": 0,
            }

        return {"success": False, "query": query, "error": "No conditions to reorder"}

    def _apply_join_syntax_variation(self, query: str) -> Dict[str, Any]:
        """Apply JOIN syntax variations."""
        variations = [
            (r"\bINNER\s+JOIN\b", "JOIN"),
            (r"\bJOIN\b(?!\s+ON)", "INNER JOIN"),
            (r"\bLEFT\s+OUTER\s+JOIN\b", "LEFT JOIN"),
            (r"\bRIGHT\s+OUTER\s+JOIN\b", "RIGHT JOIN"),
        ]

        for pattern, replacement in variations:
            if re.search(pattern, query, re.IGNORECASE):
                mutated_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                return {
                    "success": True,
                    "query": mutated_query,
                    "semantic_preserved": True,
                    "complexity_change": 0,
                }

        return {"success": False, "query": query, "error": "No JOIN syntax to vary"}

    def _apply_function_equivalent(self, query: str) -> Dict[str, Any]:
        """Apply function equivalent transformations."""
        equivalents = [
            (r"\bCOUNT\(\*\)", "COUNT(1)"),
            (r"\bCOUNT\(1\)", "COUNT(*)"),
        ]

        for pattern, replacement in equivalents:
            if re.search(pattern, query, re.IGNORECASE):
                mutated_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                return {
                    "success": True,
                    "query": mutated_query,
                    "semantic_preserved": True,
                    "complexity_change": 0,
                }

        return {"success": False, "query": query, "error": "No functions to vary"}

    def _apply_literal_variation(
        self, query: str, database_type: str
    ) -> Dict[str, Any]:
        """Apply literal value variations."""
        if database_type in ["mysql", "sqlite"]:
            # Convert single quotes to double quotes or vice versa
            if "'" in query and '"' not in query:
                mutated_query = query.replace("'", '"')
                return {
                    "success": True,
                    "query": mutated_query,
                    "semantic_preserved": True,
                    "complexity_change": 0,
                }
            elif '"' in query and "'" not in query:
                mutated_query = query.replace('"', "'")
                return {
                    "success": True,
                    "query": mutated_query,
                    "semantic_preserved": True,
                    "complexity_change": 0,
                }

        return {"success": False, "query": query, "error": "No literals to vary"}

    def _apply_parentheses_variation(self, query: str) -> Dict[str, Any]:
        """Add or remove optional parentheses."""
        # Add parentheses around complex conditions
        condition_pattern = r"(\w+\s*[=<>!]+\s*\w+)\s+AND\s+(\w+\s*[=<>!]+\s*\w+)"
        match = re.search(condition_pattern, query)

        if match:
            full_condition = match.group(0)
            parenthesized = f"({match.group(1)}) AND ({match.group(2)})"
            mutated_query = query.replace(full_condition, parenthesized)

            return {
                "success": True,
                "query": mutated_query,
                "semantic_preserved": True,
                "complexity_change": 1,
            }

        return {
            "success": False,
            "query": query,
            "error": "No conditions for parentheses",
        }

    def _validate_mutated_query(self, query: str) -> bool:
        """Validate that the mutated query is syntactically correct."""
        try:
            parsed = sqlparse.parse(query)
            return len(parsed) > 0 and len(str(parsed[0]).strip()) > 0
        except Exception:
            return False

    def generate_semantic_variations(self, query: str, count: int = 5) -> List[str]:
        """Generate semantic variations that preserve query meaning."""
        variations = []

        for _ in range(count):
            mutations = self.mutate_query(query, num_mutations=random.randint(1, 3))
            for mutation in mutations:
                if mutation.success and mutation.semantic_preserved:
                    variations.append(mutation.mutated_query)
                    break

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for variation in variations:
            if variation not in seen:
                seen.add(variation)
                unique_variations.append(variation)

        return unique_variations[:count]

    def generate_complexity_variants(self, query: str) -> Dict[str, List[str]]:
        """Generate variants with different complexity levels."""
        variants = {"simplified": [], "equivalent": [], "enhanced": []}

        mutations = self.mutate_query(query, num_mutations=10)

        for mutation in mutations:
            if not mutation.success:
                continue

            if mutation.complexity_change < 0:
                variants["simplified"].append(mutation.mutated_query)
            elif mutation.complexity_change == 0:
                variants["equivalent"].append(mutation.mutated_query)
            else:
                variants["enhanced"].append(mutation.mutated_query)

        return variants


# Usage examples and utility functions
def test_mutation_engine():
    """Test the mutation engine with sample queries."""
    engine = QueryMutationEngine()

    test_queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id",
        "SELECT name FROM products WHERE price > 100 AND category = 'electronics'",
    ]

    for query in test_queries:
        print(f"\nOriginal: {query}")
        mutations = engine.mutate_query(query, num_mutations=3)

        for i, mutation in enumerate(mutations, 1):
            if mutation.success:
                print(f"Mutation {i}: {mutation.mutated_query}")
                print(f"  Applied: {[m.value for m in mutation.mutations_applied]}")
                print(f"  Semantic preserved: {mutation.semantic_preserved}")
            else:
                print(f"Mutation {i}: Failed - {mutation.error_message}")


if __name__ == "__main__":
    test_mutation_engine()
