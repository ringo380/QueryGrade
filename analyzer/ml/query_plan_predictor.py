"""
Query Execution Plan Prediction System

This module predicts query execution plans without actually executing queries,
providing insights into potential performance characteristics and optimization opportunities.
"""

import re
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import json


class PlanNodeType(Enum):
    """Types of execution plan nodes"""
    TABLE_SCAN = "table_scan"
    INDEX_SCAN = "index_scan"
    INDEX_SEEK = "index_seek"
    NESTED_LOOP = "nested_loop"
    HASH_JOIN = "hash_join"
    MERGE_JOIN = "merge_join"
    SORT = "sort"
    AGGREGATE = "aggregate"
    FILTER = "filter"
    PROJECT = "project"
    UNION = "union"
    SUBQUERY = "subquery"
    TEMP_TABLE = "temp_table"
    MATERIALIZE = "materialize"


class CostCategory(Enum):
    """Categories of query execution costs"""
    CPU = "cpu"
    IO = "io"
    MEMORY = "memory"
    NETWORK = "network"


@dataclass
class PlanNode:
    """Represents a node in the predicted execution plan"""
    node_type: PlanNodeType
    estimated_cost: float
    estimated_rows: int
    actual_rows: Optional[int] = None
    cost_breakdown: Dict[CostCategory, float] = field(default_factory=dict)
    children: List['PlanNode'] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    optimization_hints: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlanPrediction:
    """Complete predicted execution plan for a query"""
    root_node: PlanNode
    total_cost: float
    estimated_time_ms: float
    memory_usage_mb: float
    io_operations: int
    cpu_operations: int
    parallelism_degree: int
    bottleneck_nodes: List[PlanNode] = field(default_factory=list)
    optimization_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.5
    plan_complexity: float = 0.0
    alternative_plans: List['ExecutionPlanPrediction'] = field(default_factory=list)


class QueryPlanPredictor:
    """Predicts query execution plans based on query structure and statistics"""

    def __init__(self, database_type: str = 'generic'):
        self.database_type = database_type
        self.logger = logging.getLogger(__name__)
        self._initialize_cost_models()
        self._initialize_statistics_cache()

    def _initialize_cost_models(self):
        """Initialize cost models for different operations"""
        self.cost_models = {
            PlanNodeType.TABLE_SCAN: {
                'cpu_per_row': 0.01,
                'io_per_page': 1.0,
                'memory_per_row': 0.001
            },
            PlanNodeType.INDEX_SCAN: {
                'cpu_per_row': 0.005,
                'io_per_page': 0.5,
                'memory_per_row': 0.0005
            },
            PlanNodeType.INDEX_SEEK: {
                'cpu_per_row': 0.001,
                'io_per_seek': 0.1,
                'memory_per_row': 0.0001
            },
            PlanNodeType.NESTED_LOOP: {
                'cpu_multiplier': 2.0,
                'memory_base': 1.0
            },
            PlanNodeType.HASH_JOIN: {
                'cpu_per_row': 0.01,
                'memory_per_row': 0.01,
                'build_cost': 10.0
            },
            PlanNodeType.MERGE_JOIN: {
                'cpu_per_row': 0.005,
                'memory_per_row': 0.001,
                'sort_cost': 5.0
            },
            PlanNodeType.SORT: {
                'cpu_complexity': 1.5,  # n log n
                'memory_per_row': 0.01,
                'spill_threshold': 100000
            },
            PlanNodeType.AGGREGATE: {
                'cpu_per_group': 0.02,
                'memory_per_group': 0.01
            }
        }

    def _initialize_statistics_cache(self):
        """Initialize cache for table/index statistics"""
        self.statistics_cache = {
            'table_sizes': defaultdict(lambda: 10000),  # Default 10k rows
            'index_selectivity': defaultdict(lambda: 0.1),  # Default 10% selectivity
            'column_cardinality': defaultdict(lambda: 100),  # Default 100 distinct values
            'table_pages': defaultdict(lambda: 1000),  # Default 1000 pages
            'index_height': defaultdict(lambda: 3)  # Default B-tree height of 3
        }

    def predict_execution_plan(self, query: str, table_statistics: Dict[str, Any] = None) -> ExecutionPlanPrediction:
        """Predict the execution plan for a given query"""
        try:
            # Update statistics if provided
            if table_statistics:
                self._update_statistics(table_statistics)

            # Parse and analyze query
            query_analysis = self._analyze_query_structure(query)

            # Build execution plan tree
            root_node = self._build_plan_tree(query_analysis)

            # Calculate costs
            self._calculate_plan_costs(root_node)

            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(root_node)

            # Find optimization opportunities
            optimizations = self._find_optimization_opportunities(root_node, query_analysis)

            # Create prediction
            prediction = ExecutionPlanPrediction(
                root_node=root_node,
                total_cost=root_node.estimated_cost,
                estimated_time_ms=self._estimate_execution_time(root_node),
                memory_usage_mb=self._estimate_memory_usage(root_node),
                io_operations=self._count_io_operations(root_node),
                cpu_operations=self._count_cpu_operations(root_node),
                parallelism_degree=self._estimate_parallelism(root_node),
                bottleneck_nodes=bottlenecks,
                optimization_opportunities=optimizations,
                confidence_score=self._calculate_confidence(query_analysis),
                plan_complexity=self._calculate_plan_complexity(root_node)
            )

            # Generate alternative plans
            prediction.alternative_plans = self._generate_alternative_plans(query_analysis)

            return prediction

        except Exception as e:
            self.logger.error(f"Error predicting execution plan: {e}")
            # Return a basic prediction on error
            return self._create_fallback_prediction()

    def _analyze_query_structure(self, query: str) -> Dict[str, Any]:
        """Analyze the structure of the SQL query"""
        query_upper = query.upper()

        analysis = {
            'type': self._determine_query_type(query_upper),
            'tables': self._extract_tables(query),
            'joins': self._extract_joins(query),
            'filters': self._extract_filters(query),
            'aggregations': self._extract_aggregations(query),
            'sorting': self._extract_sorting(query),
            'grouping': self._extract_grouping(query),
            'subqueries': self._extract_subqueries(query),
            'set_operations': self._extract_set_operations(query),
            'hints': self._extract_query_hints(query)
        }

        return analysis

    def _determine_query_type(self, query_upper: str) -> str:
        """Determine the primary type of the query"""
        if query_upper.startswith('SELECT'):
            if 'GROUP BY' in query_upper:
                return 'aggregate'
            elif 'JOIN' in query_upper:
                return 'join'
            else:
                return 'simple_select'
        elif query_upper.startswith('INSERT'):
            return 'insert'
        elif query_upper.startswith('UPDATE'):
            return 'update'
        elif query_upper.startswith('DELETE'):
            return 'delete'
        else:
            return 'other'

    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from the query"""
        tables = []

        # Extract from FROM clause
        from_pattern = re.compile(r'FROM\s+([^\s,]+)', re.IGNORECASE)
        tables.extend(from_pattern.findall(query))

        # Extract from JOIN clauses
        join_pattern = re.compile(r'JOIN\s+([^\s]+)', re.IGNORECASE)
        tables.extend(join_pattern.findall(query))

        return list(set(tables))

    def _extract_joins(self, query: str) -> List[Dict[str, Any]]:
        """Extract join information from the query"""
        joins = []
        query_upper = query.upper()

        join_types = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN']

        for join_type in join_types:
            if join_type in query_upper:
                pattern = re.compile(f'{join_type}\\s+([^\\s]+)\\s+ON\\s+([^WHERE^GROUP^ORDER]+)', re.IGNORECASE)
                matches = pattern.findall(query)
                for match in matches:
                    joins.append({
                        'type': join_type.replace(' JOIN', '').lower(),
                        'table': match[0],
                        'condition': match[1].strip()
                    })

        return joins

    def _extract_filters(self, query: str) -> List[Dict[str, Any]]:
        """Extract WHERE clause conditions"""
        filters = []

        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)

            # Simple condition extraction (can be enhanced)
            conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_clause, flags=re.IGNORECASE)

            for condition in conditions:
                filters.append({
                    'condition': condition.strip(),
                    'selectivity': self._estimate_filter_selectivity(condition)
                })

        return filters

    def _extract_aggregations(self, query: str) -> List[str]:
        """Extract aggregation functions from the query"""
        agg_pattern = re.compile(r'\b(COUNT|SUM|AVG|MAX|MIN|STDDEV|VARIANCE)\s*\(', re.IGNORECASE)
        return agg_pattern.findall(query)

    def _extract_sorting(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract ORDER BY information"""
        order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:LIMIT|$)', query, re.IGNORECASE)
        if order_match:
            order_clause = order_match.group(1)
            columns = [col.strip() for col in order_clause.split(',')]
            return {
                'columns': columns,
                'estimated_rows': self._estimate_sort_rows(query)
            }
        return None

    def _extract_grouping(self, query: str) -> Optional[List[str]]:
        """Extract GROUP BY columns"""
        group_match = re.search(r'GROUP\s+BY\s+(.+?)(?:HAVING|ORDER|LIMIT|$)', query, re.IGNORECASE)
        if group_match:
            group_clause = group_match.group(1)
            return [col.strip() for col in group_clause.split(',')]
        return None

    def _extract_subqueries(self, query: str) -> List[str]:
        """Extract subqueries from the query"""
        subqueries = []
        # Find nested SELECT statements
        pattern = re.compile(r'\((\s*SELECT[^)]+)\)', re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(query)
        subqueries.extend(matches)
        return subqueries

    def _extract_set_operations(self, query: str) -> List[str]:
        """Extract set operations (UNION, INTERSECT, EXCEPT)"""
        set_ops = []
        for op in ['UNION', 'UNION ALL', 'INTERSECT', 'EXCEPT']:
            if op in query.upper():
                set_ops.append(op)
        return set_ops

    def _extract_query_hints(self, query: str) -> List[str]:
        """Extract query hints if present"""
        hints = []
        # SQL Server style hints
        hint_pattern = re.compile(r'/\*\+(.+?)\*/', re.DOTALL)
        hints.extend(hint_pattern.findall(query))
        # MySQL style hints
        force_pattern = re.compile(r'(FORCE|USE|IGNORE)\s+INDEX', re.IGNORECASE)
        hints.extend(force_pattern.findall(query))
        return hints

    def _build_plan_tree(self, query_analysis: Dict[str, Any]) -> PlanNode:
        """Build the execution plan tree based on query analysis"""
        # Start with the root node (usually projection)
        root = PlanNode(
            node_type=PlanNodeType.PROJECT,
            estimated_cost=0.0,
            estimated_rows=1000
        )

        current_node = root

        # Add sorting if present
        if query_analysis['sorting']:
            sort_node = PlanNode(
                node_type=PlanNodeType.SORT,
                estimated_cost=0.0,
                estimated_rows=current_node.estimated_rows,
                properties={'columns': query_analysis['sorting']['columns']}
            )
            current_node.children.append(sort_node)
            current_node = sort_node

        # Add aggregation if present
        if query_analysis['aggregations'] or query_analysis['grouping']:
            agg_node = PlanNode(
                node_type=PlanNodeType.AGGREGATE,
                estimated_cost=0.0,
                estimated_rows=self._estimate_group_count(query_analysis),
                properties={
                    'aggregations': query_analysis['aggregations'],
                    'grouping': query_analysis['grouping']
                }
            )
            current_node.children.append(agg_node)
            current_node = agg_node

        # Add joins
        if query_analysis['joins']:
            join_node = self._build_join_tree(query_analysis)
            current_node.children.append(join_node)
        else:
            # Single table access
            table_node = self._build_table_access_node(query_analysis)
            current_node.children.append(table_node)

        # Add subqueries
        for subquery in query_analysis['subqueries']:
            subquery_node = PlanNode(
                node_type=PlanNodeType.SUBQUERY,
                estimated_cost=0.0,
                estimated_rows=100,
                properties={'query': subquery}
            )
            current_node.children.append(subquery_node)

        return root

    def _build_join_tree(self, query_analysis: Dict[str, Any]) -> PlanNode:
        """Build join nodes for the execution plan"""
        joins = query_analysis['joins']
        tables = query_analysis['tables']

        if not joins:
            return self._build_table_access_node(query_analysis)

        # Determine join type based on analysis
        join_type = self._determine_join_algorithm(joins[0], query_analysis)

        join_node = PlanNode(
            node_type=join_type,
            estimated_cost=0.0,
            estimated_rows=self._estimate_join_cardinality(joins[0], query_analysis)
        )

        # Add table access nodes as children
        for table in tables[:2]:  # Simple two-table join for now
            table_node = self._build_table_access_node(query_analysis, table)
            join_node.children.append(table_node)

        # Handle additional joins recursively
        if len(joins) > 1:
            next_join = self._build_join_tree({
                **query_analysis,
                'joins': joins[1:],
                'tables': tables[2:]
            })
            join_node.children.append(next_join)

        return join_node

    def _build_table_access_node(self, query_analysis: Dict[str, Any], table_name: str = None) -> PlanNode:
        """Build table access node (scan or seek)"""
        if not table_name and query_analysis['tables']:
            table_name = query_analysis['tables'][0]

        # Determine access method based on filters
        if self._has_index_friendly_filters(query_analysis['filters'], table_name):
            node_type = PlanNodeType.INDEX_SEEK
            estimated_rows = int(self.statistics_cache['table_sizes'][table_name] * 0.01)
        elif self._has_covering_index(table_name, query_analysis):
            node_type = PlanNodeType.INDEX_SCAN
            estimated_rows = int(self.statistics_cache['table_sizes'][table_name] * 0.1)
        else:
            node_type = PlanNodeType.TABLE_SCAN
            estimated_rows = self.statistics_cache['table_sizes'][table_name]

        return PlanNode(
            node_type=node_type,
            estimated_cost=0.0,
            estimated_rows=estimated_rows,
            properties={'table': table_name}
        )

    def _determine_join_algorithm(self, join: Dict[str, Any], query_analysis: Dict[str, Any]) -> PlanNodeType:
        """Determine the best join algorithm based on statistics"""
        # Simple heuristics for join algorithm selection
        join_condition = join['condition']

        # Check for equality join
        if '=' in join_condition and not ('>' in join_condition or '<' in join_condition):
            # Estimate table sizes
            left_size = self.statistics_cache['table_sizes'].get(query_analysis['tables'][0], 10000)
            right_size = self.statistics_cache['table_sizes'].get(join['table'], 10000)

            # Use nested loop for small tables
            if min(left_size, right_size) < 100:
                return PlanNodeType.NESTED_LOOP
            # Use hash join for medium tables
            elif max(left_size, right_size) < 100000:
                return PlanNodeType.HASH_JOIN
            # Use merge join for large sorted tables
            else:
                return PlanNodeType.MERGE_JOIN
        else:
            # Non-equality joins typically use nested loop
            return PlanNodeType.NESTED_LOOP

    def _calculate_plan_costs(self, node: PlanNode):
        """Calculate costs for each node in the plan tree"""
        # Calculate costs for children first
        for child in node.children:
            self._calculate_plan_costs(child)

        # Calculate node's own cost
        if node.node_type in self.cost_models:
            model = self.cost_models[node.node_type]

            if node.node_type == PlanNodeType.TABLE_SCAN:
                node.cost_breakdown[CostCategory.CPU] = node.estimated_rows * model['cpu_per_row']
                node.cost_breakdown[CostCategory.IO] = self.statistics_cache['table_pages'][node.properties.get('table', '')] * model['io_per_page']
                node.cost_breakdown[CostCategory.MEMORY] = node.estimated_rows * model['memory_per_row']

            elif node.node_type == PlanNodeType.INDEX_SEEK:
                node.cost_breakdown[CostCategory.CPU] = node.estimated_rows * model['cpu_per_row']
                node.cost_breakdown[CostCategory.IO] = model['io_per_seek'] * self.statistics_cache['index_height'][node.properties.get('table', '')]
                node.cost_breakdown[CostCategory.MEMORY] = node.estimated_rows * model['memory_per_row']

            elif node.node_type == PlanNodeType.HASH_JOIN:
                if len(node.children) >= 2:
                    build_rows = node.children[0].estimated_rows
                    probe_rows = node.children[1].estimated_rows
                    node.cost_breakdown[CostCategory.CPU] = (build_rows + probe_rows) * model['cpu_per_row'] + model['build_cost']
                    node.cost_breakdown[CostCategory.MEMORY] = build_rows * model['memory_per_row']

            elif node.node_type == PlanNodeType.SORT:
                rows = node.estimated_rows
                node.cost_breakdown[CostCategory.CPU] = rows * np.log2(max(rows, 2)) * model['cpu_complexity']
                node.cost_breakdown[CostCategory.MEMORY] = rows * model['memory_per_row']
                if rows > model['spill_threshold']:
                    node.cost_breakdown[CostCategory.IO] = rows / 1000  # Spill to disk

        # Sum up total cost
        node.estimated_cost = sum(node.cost_breakdown.values())

        # Add children's costs
        for child in node.children:
            node.estimated_cost += child.estimated_cost

    def _identify_bottlenecks(self, root: PlanNode, threshold: float = 0.3) -> List[PlanNode]:
        """Identify nodes that are bottlenecks in the execution plan"""
        bottlenecks = []
        total_cost = root.estimated_cost

        def find_bottlenecks(node: PlanNode):
            if node.estimated_cost > total_cost * threshold:
                bottlenecks.append(node)
                # Add optimization hints
                if node.node_type == PlanNodeType.TABLE_SCAN:
                    node.optimization_hints.append("Consider adding an index")
                elif node.node_type == PlanNodeType.SORT:
                    node.optimization_hints.append("Consider pre-sorting or adding a sorted index")
                elif node.node_type == PlanNodeType.NESTED_LOOP:
                    node.optimization_hints.append("Consider using hash join for large tables")

            for child in node.children:
                find_bottlenecks(child)

        find_bottlenecks(root)
        return bottlenecks

    def _find_optimization_opportunities(self, root: PlanNode, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find potential optimization opportunities"""
        opportunities = []

        # Check for missing indexes
        for filter_info in query_analysis['filters']:
            if filter_info['selectivity'] < 0.1:  # High selectivity filter
                opportunities.append({
                    'type': 'missing_index',
                    'description': f"Consider adding index for condition: {filter_info['condition']}",
                    'impact': 'high',
                    'estimated_improvement': 0.5
                })

        # Check for unnecessary sorting
        if query_analysis['sorting'] and not query_analysis['aggregations']:
            opportunities.append({
                'type': 'unnecessary_sort',
                'description': "Consider using an indexed column for ordering",
                'impact': 'medium',
                'estimated_improvement': 0.3
            })

        # Check for subquery optimization
        if query_analysis['subqueries']:
            opportunities.append({
                'type': 'subquery_optimization',
                'description': "Consider rewriting subqueries as joins",
                'impact': 'medium',
                'estimated_improvement': 0.2
            })

        return opportunities

    def _estimate_execution_time(self, node: PlanNode) -> float:
        """Estimate execution time in milliseconds"""
        # Simple model: 1 cost unit = 0.1ms
        return node.estimated_cost * 0.1

    def _estimate_memory_usage(self, node: PlanNode) -> float:
        """Estimate memory usage in megabytes"""
        def sum_memory(n: PlanNode) -> float:
            memory = n.cost_breakdown.get(CostCategory.MEMORY, 0.0)
            for child in n.children:
                memory += sum_memory(child)
            return memory

        return sum_memory(node)

    def _count_io_operations(self, node: PlanNode) -> int:
        """Count total I/O operations"""
        def sum_io(n: PlanNode) -> float:
            io = n.cost_breakdown.get(CostCategory.IO, 0.0)
            for child in n.children:
                io += sum_io(child)
            return io

        return int(sum_io(node))

    def _count_cpu_operations(self, node: PlanNode) -> int:
        """Count total CPU operations"""
        def sum_cpu(n: PlanNode) -> float:
            cpu = n.cost_breakdown.get(CostCategory.CPU, 0.0)
            for child in n.children:
                cpu += sum_cpu(child)
            return cpu

        return int(sum_cpu(node) * 1000)  # Convert to operation count

    def _estimate_parallelism(self, node: PlanNode) -> int:
        """Estimate degree of parallelism possible"""
        # Check for operations that can be parallelized
        parallelizable_ops = [PlanNodeType.TABLE_SCAN, PlanNodeType.AGGREGATE, PlanNodeType.HASH_JOIN]

        max_parallelism = 1

        def check_parallelism(n: PlanNode):
            nonlocal max_parallelism
            if n.node_type in parallelizable_ops:
                # Estimate based on data size
                if n.estimated_rows > 10000:
                    max_parallelism = max(max_parallelism, min(8, n.estimated_rows // 10000))

            for child in n.children:
                check_parallelism(child)

        check_parallelism(node)
        return max_parallelism

    def _calculate_confidence(self, query_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the prediction"""
        confidence = 0.5  # Base confidence

        # Increase confidence based on available information
        if query_analysis['tables']:
            confidence += 0.1
        if query_analysis['filters']:
            confidence += 0.1
        if not query_analysis['subqueries']:  # Simpler queries are more predictable
            confidence += 0.1
        if not query_analysis['hints']:  # No hints means standard optimization
            confidence += 0.1
        if len(query_analysis['joins']) <= 2:  # Fewer joins are more predictable
            confidence += 0.1

        return min(1.0, confidence)

    def _calculate_plan_complexity(self, node: PlanNode) -> float:
        """Calculate overall plan complexity"""
        def count_nodes(n: PlanNode) -> int:
            count = 1
            for child in n.children:
                count += count_nodes(child)
            return count

        node_count = count_nodes(node)

        # Normalize complexity score (0-1)
        return min(1.0, node_count / 20.0)

    def _generate_alternative_plans(self, query_analysis: Dict[str, Any]) -> List[ExecutionPlanPrediction]:
        """Generate alternative execution plans"""
        alternatives = []

        # For now, just return empty list
        # In a real implementation, this would generate different join orders,
        # access methods, etc.

        return alternatives

    def _estimate_filter_selectivity(self, condition: str) -> float:
        """Estimate selectivity of a filter condition"""
        condition_lower = condition.lower()

        # Simple heuristics
        if '=' in condition and 'id' in condition_lower:
            return 0.001  # Primary key lookup
        elif '=' in condition:
            return 0.01  # Equality on indexed column
        elif 'between' in condition_lower:
            return 0.1  # Range query
        elif 'like' in condition_lower:
            if condition_lower.count('%') >= 2:
                return 0.5  # Wildcard on both sides
            else:
                return 0.2  # Wildcard on one side
        elif 'in' in condition_lower:
            return 0.05  # IN clause
        else:
            return 0.3  # Default

    def _estimate_sort_rows(self, query: str) -> int:
        """Estimate number of rows to sort"""
        # Check for LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            return int(limit_match.group(1))
        else:
            # Estimate based on filters
            return 10000  # Default

    def _estimate_group_count(self, query_analysis: Dict[str, Any]) -> int:
        """Estimate number of groups after GROUP BY"""
        if query_analysis['grouping']:
            # Estimate based on column cardinality
            estimated_groups = 1
            for col in query_analysis['grouping']:
                estimated_groups *= self.statistics_cache['column_cardinality'].get(col, 10)
            return min(estimated_groups, 10000)
        return 100  # Default

    def _estimate_join_cardinality(self, join: Dict[str, Any], query_analysis: Dict[str, Any]) -> int:
        """Estimate result cardinality after join"""
        # Simple estimation based on join type and table sizes
        left_size = 10000  # Default
        right_size = 10000  # Default

        if join['type'] == 'inner':
            return int(min(left_size, right_size) * 0.1)
        elif join['type'] == 'left':
            return left_size
        elif join['type'] == 'right':
            return right_size
        elif join['type'] == 'full':
            return left_size + right_size
        elif join['type'] == 'cross':
            return left_size * right_size
        else:
            return int((left_size + right_size) / 2)

    def _has_index_friendly_filters(self, filters: List[Dict[str, Any]], table_name: str) -> bool:
        """Check if filters are suitable for index usage"""
        for filter_info in filters:
            if filter_info['selectivity'] < 0.01:  # Very selective filter
                return True
        return False

    def _has_covering_index(self, table_name: str, query_analysis: Dict[str, Any]) -> bool:
        """Check if query can use a covering index"""
        # Simplified check - in reality would check actual index definitions
        return False

    def _update_statistics(self, statistics: Dict[str, Any]):
        """Update internal statistics cache"""
        if 'table_sizes' in statistics:
            self.statistics_cache['table_sizes'].update(statistics['table_sizes'])
        if 'index_selectivity' in statistics:
            self.statistics_cache['index_selectivity'].update(statistics['index_selectivity'])
        if 'column_cardinality' in statistics:
            self.statistics_cache['column_cardinality'].update(statistics['column_cardinality'])

    def _create_fallback_prediction(self) -> ExecutionPlanPrediction:
        """Create a fallback prediction when analysis fails"""
        root = PlanNode(
            node_type=PlanNodeType.TABLE_SCAN,
            estimated_cost=100.0,
            estimated_rows=10000
        )

        return ExecutionPlanPrediction(
            root_node=root,
            total_cost=100.0,
            estimated_time_ms=10.0,
            memory_usage_mb=1.0,
            io_operations=100,
            cpu_operations=10000,
            parallelism_degree=1,
            confidence_score=0.1,
            plan_complexity=0.5
        )


def visualize_plan(prediction: ExecutionPlanPrediction) -> str:
    """Generate a text visualization of the execution plan"""
    lines = []

    def format_node(node: PlanNode, indent: int = 0):
        prefix = "  " * indent + "→ " if indent > 0 else ""
        lines.append(f"{prefix}{node.node_type.value.upper()}")
        lines.append(f"{'  ' * (indent + 1)}Cost: {node.estimated_cost:.2f}")
        lines.append(f"{'  ' * (indent + 1)}Rows: {node.estimated_rows}")

        if node.optimization_hints:
            lines.append(f"{'  ' * (indent + 1)}Hints: {', '.join(node.optimization_hints)}")

        for child in node.children:
            format_node(child, indent + 1)

    lines.append("=== PREDICTED EXECUTION PLAN ===")
    format_node(prediction.root_node)
    lines.append("")
    lines.append(f"Total Cost: {prediction.total_cost:.2f}")
    lines.append(f"Estimated Time: {prediction.estimated_time_ms:.2f} ms")
    lines.append(f"Memory Usage: {prediction.memory_usage_mb:.2f} MB")
    lines.append(f"Parallelism Degree: {prediction.parallelism_degree}")
    lines.append(f"Confidence: {prediction.confidence_score:.2%}")

    if prediction.bottleneck_nodes:
        lines.append("\nBOTTLENECKS:")
        for bottleneck in prediction.bottleneck_nodes:
            lines.append(f"  - {bottleneck.node_type.value}: {bottleneck.optimization_hints}")

    if prediction.optimization_opportunities:
        lines.append("\nOPTIMIZATION OPPORTUNITIES:")
        for opp in prediction.optimization_opportunities:
            lines.append(f"  - [{opp['impact']}] {opp['description']}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    predictor = QueryPlanPredictor()

    test_query = """
    SELECT c.customer_name, SUM(o.total_amount) as total_spent
    FROM customers c
    INNER JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_date >= '2023-01-01'
    GROUP BY c.customer_id, c.customer_name
    ORDER BY total_spent DESC
    LIMIT 10
    """

    # Provide some statistics
    statistics = {
        'table_sizes': {
            'customers': 50000,
            'orders': 500000
        },
        'column_cardinality': {
            'customer_id': 50000,
            'order_date': 365
        }
    }

    prediction = predictor.predict_execution_plan(test_query, statistics)
    print(visualize_plan(prediction))