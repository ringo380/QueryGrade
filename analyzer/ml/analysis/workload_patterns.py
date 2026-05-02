"""
Workload Pattern Recognition System

This module identifies and analyzes workload patterns in database query streams,
detecting recurring patterns, time-based trends, and workload characteristics.
"""

import hashlib
import logging
import pickle
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


class WorkloadType(Enum):
    """Types of database workloads"""

    OLTP = "oltp"  # Online Transaction Processing
    OLAP = "olap"  # Online Analytical Processing
    HYBRID = "hybrid"  # Mixed workload
    BATCH = "batch"  # Batch processing
    STREAMING = "streaming"  # Real-time streaming
    MAINTENANCE = "maintenance"  # Maintenance operations


class PatternType(Enum):
    """Types of query patterns"""

    SEQUENTIAL = "sequential"  # Sequential access pattern
    RANDOM = "random"  # Random access pattern
    TEMPORAL = "temporal"  # Time-based pattern
    PERIODIC = "periodic"  # Recurring at regular intervals
    BURST = "burst"  # Sudden spike in activity
    GRADUAL = "gradual"  # Gradual increase/decrease


class TimeWindow(Enum):
    """Time windows for analysis"""

    MINUTE = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 604800
    MONTH = 2592000


@dataclass
class QueryPattern:
    """Represents a discovered query pattern"""

    pattern_id: str
    pattern_type: PatternType
    query_template: str
    frequency: int
    avg_execution_time: float
    time_distribution: Dict[int, int]  # hour -> count
    parameter_variations: List[Dict[str, Any]]
    tables_involved: Set[str]
    operations: Set[str]  # SELECT, INSERT, UPDATE, DELETE
    confidence: float


@dataclass
class WorkloadProfile:
    """Profile of database workload characteristics"""

    workload_type: WorkloadType
    start_time: datetime
    end_time: datetime
    total_queries: int
    unique_patterns: int
    queries_per_second: float
    read_write_ratio: float
    avg_query_complexity: float
    peak_hours: List[int]
    quiet_hours: List[int]
    dominant_patterns: List[QueryPattern]
    anomaly_score: float
    resource_usage: Dict[str, float]  # CPU, Memory, IO percentages


@dataclass
class TemporalPattern:
    """Time-based pattern in workload"""

    pattern_name: str
    time_window: TimeWindow
    recurrence_type: str  # daily, weekly, monthly
    peak_times: List[Tuple[time, time]]  # start, end times
    intensity_profile: List[float]  # Normalized intensity over time
    confidence: float
    next_occurrence: datetime


class WorkloadPatternRecognizer:
    """Recognizes and analyzes patterns in database workloads"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Pattern storage
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.temporal_patterns: List[TemporalPattern] = []
        self.workload_profiles: List[WorkloadProfile] = []

        # Query stream buffer
        self.query_buffer = deque(maxlen=10000)
        self.query_timestamps = deque(maxlen=10000)

        # Pattern detection models
        self.clustering_model = None
        self.scaler = StandardScaler()

        # Statistics
        self.pattern_statistics = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0,
                "min_time": float("inf"),
                "max_time": 0,
                "last_seen": None,
            }
        )

    def process_query_stream(
        self, queries: List[Tuple[str, datetime, float]]
    ) -> WorkloadProfile:
        """
        Process a stream of queries to identify patterns
        Args:
            queries: List of (query, timestamp, execution_time) tuples
        """
        if not queries:
            return self._create_empty_profile()

        # Sort queries by timestamp
        queries.sort(key=lambda x: x[1])

        # Update buffer
        for query, timestamp, exec_time in queries:
            self.query_buffer.append((query, exec_time))
            self.query_timestamps.append(timestamp)

        # Extract features from queries
        query_features = self._extract_query_features(queries)

        # Identify query patterns
        patterns = self._identify_query_patterns(query_features)

        # Detect temporal patterns
        temporal_patterns = self._detect_temporal_patterns(queries)

        # Analyze workload characteristics
        workload_profile = self._analyze_workload_characteristics(
            queries, patterns, temporal_patterns
        )

        # Store results
        self.workload_profiles.append(workload_profile)

        return workload_profile

    def _extract_query_features(
        self, queries: List[Tuple[str, datetime, float]]
    ) -> np.ndarray:
        """Extract features from queries for pattern recognition"""
        features = []

        for query, timestamp, exec_time in queries:
            query_upper = query.upper()

            # Basic features
            query_length = len(query)
            num_tables = len(re.findall(r"\bFROM\b|\bJOIN\b", query_upper))
            num_conditions = len(re.findall(r"\bWHERE\b|\bAND\b|\bOR\b", query_upper))
            num_aggregates = len(
                re.findall(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", query_upper)
            )
            has_group_by = 1 if "GROUP BY" in query_upper else 0
            has_order_by = 1 if "ORDER BY" in query_upper else 0
            has_subquery = 1 if query_upper.count("SELECT") > 1 else 0

            # Operation type
            is_select = 1 if query_upper.startswith("SELECT") else 0
            is_insert = 1 if query_upper.startswith("INSERT") else 0
            is_update = 1 if query_upper.startswith("UPDATE") else 0
            is_delete = 1 if query_upper.startswith("DELETE") else 0

            # Temporal features
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0

            # Performance features
            log_exec_time = np.log1p(exec_time)

            features.append(
                [
                    query_length / 1000,  # Normalize
                    num_tables,
                    num_conditions,
                    num_aggregates,
                    has_group_by,
                    has_order_by,
                    has_subquery,
                    is_select,
                    is_insert,
                    is_update,
                    is_delete,
                    hour_of_day / 24,  # Normalize
                    day_of_week / 7,  # Normalize
                    is_weekend,
                    log_exec_time,
                ]
            )

        return np.array(features)

    def _identify_query_patterns(self, features: np.ndarray) -> List[QueryPattern]:
        """Identify recurring query patterns using clustering"""
        patterns = []

        if len(features) < 10:
            return patterns

        try:
            # Normalize features
            features_normalized = self.scaler.fit_transform(features)

            # Cluster queries to find patterns
            clustering = DBSCAN(eps=0.3, min_samples=5).fit(features_normalized)
            labels = clustering.labels_

            # Analyze each cluster
            unique_labels = set(labels) - {-1}  # Exclude noise points

            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]

                if len(cluster_indices) >= 5:  # Minimum pattern size
                    # Extract pattern characteristics
                    pattern = self._create_pattern_from_cluster(
                        cluster_indices, features
                    )
                    if pattern:
                        patterns.append(pattern)

        except Exception as e:
            self.logger.error(f"Error identifying patterns: {e}")

        return patterns

    def _create_pattern_from_cluster(
        self, indices: np.ndarray, features: np.ndarray
    ) -> Optional[QueryPattern]:
        """Create a QueryPattern from a cluster of similar queries"""
        try:
            cluster_features = features[indices]

            # Determine pattern type
            exec_times = cluster_features[:, -1]
            time_std = np.std(exec_times)

            if time_std < 0.1:
                pattern_type = PatternType.SEQUENTIAL
            elif time_std > 1.0:
                pattern_type = PatternType.RANDOM
            else:
                pattern_type = PatternType.PERIODIC

            # Generate pattern ID
            pattern_id = hashlib.md5(
                str(cluster_features.mean(axis=0)).encode(), usedforsecurity=False
            ).hexdigest()[:8]

            # Extract dominant operations
            operations = set()
            if cluster_features[:, 7].mean() > 0.5:
                operations.add("SELECT")
            if cluster_features[:, 8].mean() > 0.5:
                operations.add("INSERT")
            if cluster_features[:, 9].mean() > 0.5:
                operations.add("UPDATE")
            if cluster_features[:, 10].mean() > 0.5:
                operations.add("DELETE")

            # Time distribution
            hours = (cluster_features[:, 11] * 24).astype(int)
            time_distribution = dict(Counter(hours))

            return QueryPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                query_template="",  # Would need actual queries to generate template
                frequency=len(indices),
                avg_execution_time=np.mean(exec_times),
                time_distribution=time_distribution,
                parameter_variations=[],
                tables_involved=set(),
                operations=operations,
                confidence=0.8,
            )

        except Exception as e:
            self.logger.error(f"Error creating pattern: {e}")
            return None

    def _detect_temporal_patterns(
        self, queries: List[Tuple[str, datetime, float]]
    ) -> List[TemporalPattern]:
        """Detect time-based patterns in query workload"""
        temporal_patterns = []

        if len(queries) < 100:
            return temporal_patterns

        try:
            # Analyze query distribution over time
            timestamps = [q[1] for q in queries]

            # Group by hour of day
            hourly_distribution = defaultdict(list)
            for ts in timestamps:
                hourly_distribution[ts.hour].append(ts)

            # Find peak and quiet hours
            hour_counts = {
                hour: len(times) for hour, times in hourly_distribution.items()
            }
            avg_count = np.mean(list(hour_counts.values()))
            std_count = np.std(list(hour_counts.values()))

            peak_hours = [
                hour
                for hour, count in hour_counts.items()
                if count > avg_count + std_count
            ]
            quiet_hours = [
                hour
                for hour, count in hour_counts.items()
                if count < avg_count - std_count
            ]

            # Detect daily patterns
            if peak_hours:
                daily_pattern = TemporalPattern(
                    pattern_name="daily_peak",
                    time_window=TimeWindow.DAY,
                    recurrence_type="daily",
                    peak_times=self._hours_to_time_ranges(peak_hours),
                    intensity_profile=[
                        hour_counts.get(h, 0) / max(hour_counts.values())
                        for h in range(24)
                    ],
                    confidence=0.7,
                    next_occurrence=self._predict_next_occurrence(peak_hours[0]),
                )
                temporal_patterns.append(daily_pattern)

            # Detect weekly patterns
            weekly_distribution = defaultdict(list)
            for ts in timestamps:
                weekly_distribution[ts.weekday()].append(ts)

            weekday_counts = {
                day: len(times) for day, times in weekly_distribution.items()
            }

            # Check for weekend pattern
            weekday_avg = np.mean([weekday_counts.get(d, 0) for d in range(5)])
            weekend_avg = np.mean([weekday_counts.get(d, 0) for d in [5, 6]])

            if abs(weekday_avg - weekend_avg) > std_count:
                weekly_pattern = TemporalPattern(
                    pattern_name="weekday_weekend_difference",
                    time_window=TimeWindow.WEEK,
                    recurrence_type="weekly",
                    peak_times=[],
                    intensity_profile=[
                        weekday_counts.get(d, 0) / max(weekday_counts.values())
                        for d in range(7)
                    ],
                    confidence=0.6,
                    next_occurrence=self._predict_next_weekday(),
                )
                temporal_patterns.append(weekly_pattern)

        except Exception as e:
            self.logger.error(f"Error detecting temporal patterns: {e}")

        return temporal_patterns

    def _analyze_workload_characteristics(
        self,
        queries: List[Tuple[str, datetime, float]],
        query_patterns: List[QueryPattern],
        temporal_patterns: List[TemporalPattern],
    ) -> WorkloadProfile:
        """Analyze overall workload characteristics"""
        if not queries:
            return self._create_empty_profile()

        # Calculate basic statistics
        timestamps = [q[1] for q in queries]
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = (end_time - start_time).total_seconds()

        total_queries = len(queries)
        queries_per_second = total_queries / duration if duration > 0 else 0

        # Determine read/write ratio
        read_count = sum(1 for q, _, _ in queries if q.upper().startswith("SELECT"))
        write_count = total_queries - read_count
        read_write_ratio = read_count / write_count if write_count > 0 else float("inf")

        # Determine workload type
        workload_type = self._determine_workload_type(
            queries, read_write_ratio, query_patterns
        )

        # Calculate complexity
        avg_complexity = self._calculate_average_complexity(queries)

        # Extract peak and quiet hours from temporal patterns
        peak_hours = []
        quiet_hours = []
        for pattern in temporal_patterns:
            if pattern.pattern_name == "daily_peak":
                peak_hours = [t[0].hour for t in pattern.peak_times]

        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(queries, query_patterns)

        # Estimate resource usage
        resource_usage = self._estimate_resource_usage(queries)

        return WorkloadProfile(
            workload_type=workload_type,
            start_time=start_time,
            end_time=end_time,
            total_queries=total_queries,
            unique_patterns=len(query_patterns),
            queries_per_second=queries_per_second,
            read_write_ratio=read_write_ratio,
            avg_query_complexity=avg_complexity,
            peak_hours=peak_hours,
            quiet_hours=quiet_hours,
            dominant_patterns=query_patterns[:5],  # Top 5 patterns
            anomaly_score=anomaly_score,
            resource_usage=resource_usage,
        )

    def _determine_workload_type(
        self,
        queries: List[Tuple[str, datetime, float]],
        read_write_ratio: float,
        patterns: List[QueryPattern],
    ) -> WorkloadType:
        """Determine the type of workload"""
        # OLTP characteristics: High write ratio, simple queries, low complexity
        # OLAP characteristics: High read ratio, complex queries, aggregations
        # Batch: Periodic patterns, bulk operations
        # Streaming: Continuous flow, consistent rate

        avg_exec_time = np.mean([exec_time for _, _, exec_time in queries])

        # Check for OLTP
        if read_write_ratio < 2 and avg_exec_time < 100:  # Fast, balanced reads/writes
            return WorkloadType.OLTP

        # Check for OLAP
        if read_write_ratio > 10 and avg_exec_time > 1000:  # Read-heavy, slow queries
            return WorkloadType.OLAP

        # Check for batch processing
        for pattern in patterns:
            if pattern.pattern_type == PatternType.PERIODIC:
                return WorkloadType.BATCH

        # Check for streaming
        timestamps = [q[1] for q in queries]
        if len(timestamps) > 100:
            time_diffs = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
            if np.std(time_diffs) < np.mean(time_diffs) * 0.1:  # Consistent rate
                return WorkloadType.STREAMING

        # Default to hybrid
        return WorkloadType.HYBRID

    def _calculate_average_complexity(
        self, queries: List[Tuple[str, datetime, float]]
    ) -> float:
        """Calculate average query complexity"""
        complexities = []

        for query, _, _ in queries:
            query_upper = query.upper()

            complexity = 0
            complexity += len(re.findall(r"\bJOIN\b", query_upper)) * 2
            complexity += len(re.findall(r"\bGROUP BY\b", query_upper)) * 3
            complexity += len(re.findall(r"\bHAVING\b", query_upper)) * 2
            complexity += query_upper.count("SELECT") - 1  # Subqueries
            complexity += len(
                re.findall(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", query_upper)
            )

            complexities.append(complexity)

        return np.mean(complexities) if complexities else 0

    def _calculate_anomaly_score(
        self, queries: List[Tuple[str, datetime, float]], patterns: List[QueryPattern]
    ) -> float:
        """Calculate anomaly score for the workload"""
        if not patterns:
            return 0.5  # Neutral score if no patterns

        # Calculate what percentage of queries match known patterns
        pattern_matches = sum(p.frequency for p in patterns)
        total_queries = len(queries)

        if total_queries == 0:
            return 0.0

        match_ratio = pattern_matches / total_queries

        # Anomaly score is inverse of match ratio
        anomaly_score = 1.0 - match_ratio

        return min(1.0, max(0.0, anomaly_score))

    def _estimate_resource_usage(
        self, queries: List[Tuple[str, datetime, float]]
    ) -> Dict[str, float]:
        """Estimate resource usage based on query characteristics"""
        cpu_score = 0.0
        memory_score = 0.0
        io_score = 0.0

        for query, _, exec_time in queries:
            query_upper = query.upper()

            # CPU-intensive operations
            cpu_score += len(re.findall(r"\bJOIN\b", query_upper)) * 0.2
            cpu_score += (
                len(re.findall(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", query_upper)) * 0.1
            )
            cpu_score += (exec_time / 1000) * 0.1  # Long-running queries use more CPU

            # Memory-intensive operations
            memory_score += len(re.findall(r"\bGROUP BY\b", query_upper)) * 0.3
            memory_score += len(re.findall(r"\bORDER BY\b", query_upper)) * 0.2
            memory_score += len(re.findall(r"\bDISTINCT\b", query_upper)) * 0.2

            # IO-intensive operations
            if "SELECT *" in query_upper:
                io_score += 0.5
            io_score += len(re.findall(r"\bFROM\b", query_upper)) * 0.2

        # Normalize scores (0-100%)
        num_queries = len(queries)
        if num_queries > 0:
            cpu_score = min(100, (cpu_score / num_queries) * 100)
            memory_score = min(100, (memory_score / num_queries) * 100)
            io_score = min(100, (io_score / num_queries) * 100)

        return {"cpu": cpu_score, "memory": memory_score, "io": io_score}

    def _hours_to_time_ranges(self, hours: List[int]) -> List[Tuple[time, time]]:
        """Convert list of hours to time ranges"""
        if not hours:
            return []

        hours.sort()
        ranges = []
        start = hours[0]
        end = hours[0]

        for h in hours[1:]:
            if h == end + 1:
                end = h
            else:
                ranges.append((time(start, 0), time(end, 59)))
                start = end = h

        ranges.append((time(start, 0), time(end, 59)))
        return ranges

    def _predict_next_occurrence(self, hour: int) -> datetime:
        """Predict next occurrence of a daily pattern"""
        now = datetime.now()
        next_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)

        if next_time <= now:
            next_time += timedelta(days=1)

        return next_time

    def _predict_next_weekday(self) -> datetime:
        """Predict next weekday occurrence"""
        now = datetime.now()
        days_until_monday = (7 - now.weekday()) % 7

        if days_until_monday == 0 and now.hour >= 9:  # Past Monday morning
            days_until_monday = 7

        return now + timedelta(days=days_until_monday)

    def _create_empty_profile(self) -> WorkloadProfile:
        """Create an empty workload profile"""
        return WorkloadProfile(
            workload_type=WorkloadType.HYBRID,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_queries=0,
            unique_patterns=0,
            queries_per_second=0.0,
            read_write_ratio=1.0,
            avg_query_complexity=0.0,
            peak_hours=[],
            quiet_hours=[],
            dominant_patterns=[],
            anomaly_score=0.0,
            resource_usage={"cpu": 0.0, "memory": 0.0, "io": 0.0},
        )

    def predict_future_workload(self, time_horizon: timedelta) -> Dict[str, Any]:
        """Predict future workload based on historical patterns"""
        predictions = {
            "expected_queries": 0,
            "expected_patterns": [],
            "peak_periods": [],
            "resource_requirements": {},
            "confidence": 0.0,
        }

        if not self.workload_profiles:
            return predictions

        # Analyze historical profiles
        recent_profiles = self.workload_profiles[-10:]  # Last 10 profiles

        # Average statistics
        avg_qps = np.mean([p.queries_per_second for p in recent_profiles])
        predictions["expected_queries"] = int(avg_qps * time_horizon.total_seconds())

        # Predict patterns
        pattern_counts = Counter()
        for profile in recent_profiles:
            for pattern in profile.dominant_patterns:
                pattern_counts[pattern.pattern_id] += 1

        predictions["expected_patterns"] = [
            pid for pid, _ in pattern_counts.most_common(5)
        ]

        # Predict peak periods
        all_peak_hours = []
        for profile in recent_profiles:
            all_peak_hours.extend(profile.peak_hours)

        if all_peak_hours:
            peak_hour_counts = Counter(all_peak_hours)
            predictions["peak_periods"] = [
                hour for hour, _ in peak_hour_counts.most_common(3)
            ]

        # Predict resource requirements
        avg_resources = {
            "cpu": np.mean([p.resource_usage.get("cpu", 0) for p in recent_profiles]),
            "memory": np.mean(
                [p.resource_usage.get("memory", 0) for p in recent_profiles]
            ),
            "io": np.mean([p.resource_usage.get("io", 0) for p in recent_profiles]),
        }
        predictions["resource_requirements"] = avg_resources

        # Calculate confidence based on consistency
        if len(recent_profiles) >= 5:
            qps_std = np.std([p.queries_per_second for p in recent_profiles])
            qps_mean = np.mean([p.queries_per_second for p in recent_profiles])
            if qps_mean > 0:
                cv = qps_std / qps_mean
                predictions["confidence"] = max(0.0, min(1.0, 1.0 - cv))

        return predictions

    def export_patterns(self, output_path: str):
        """Export discovered patterns to a file"""
        try:
            export_data = {
                "query_patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "pattern_type": p.pattern_type.value,
                        "frequency": p.frequency,
                        "avg_execution_time": p.avg_execution_time,
                        "confidence": p.confidence,
                        "operations": list(p.operations),
                    }
                    for p in self.query_patterns.values()
                ],
                "temporal_patterns": [
                    {
                        "pattern_name": p.pattern_name,
                        "time_window": p.time_window.value,
                        "recurrence_type": p.recurrence_type,
                        "confidence": p.confidence,
                        "next_occurrence": p.next_occurrence.isoformat(),
                    }
                    for p in self.temporal_patterns
                ],
                "workload_profiles": [
                    {
                        "workload_type": p.workload_type.value,
                        "start_time": p.start_time.isoformat(),
                        "end_time": p.end_time.isoformat(),
                        "total_queries": p.total_queries,
                        "queries_per_second": p.queries_per_second,
                        "read_write_ratio": p.read_write_ratio,
                        "anomaly_score": p.anomaly_score,
                    }
                    for p in self.workload_profiles[-100:]  # Last 100 profiles
                ],
            }

            with open(output_path, "wb") as f:
                pickle.dump(export_data, f)

            self.logger.info(f"Exported patterns to {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting patterns: {e}")


def analyze_workload(query_log: List[Tuple[str, datetime, float]]) -> Dict[str, Any]:
    """
    Convenience function to analyze workload and return results
    """
    recognizer = WorkloadPatternRecognizer()
    profile = recognizer.process_query_stream(query_log)

    # Predict future workload
    future_prediction = recognizer.predict_future_workload(timedelta(hours=24))

    return {
        "current_profile": {
            "workload_type": profile.workload_type.value,
            "queries_per_second": profile.queries_per_second,
            "read_write_ratio": profile.read_write_ratio,
            "avg_complexity": profile.avg_query_complexity,
            "unique_patterns": profile.unique_patterns,
            "peak_hours": profile.peak_hours,
            "anomaly_score": profile.anomaly_score,
            "resource_usage": profile.resource_usage,
        },
        "patterns": [
            {
                "pattern_id": p.pattern_id,
                "type": p.pattern_type.value,
                "frequency": p.frequency,
                "avg_time": p.avg_execution_time,
            }
            for p in profile.dominant_patterns
        ],
        "future_prediction": future_prediction,
    }


if __name__ == "__main__":
    # Example usage
    import random
    from datetime import datetime, timedelta

    # Generate sample query log
    query_log = []
    base_time = datetime.now() - timedelta(hours=24)

    for i in range(1000):
        # Generate different types of queries
        query_type = random.choice(["select", "insert", "update"])

        if query_type == "select":
            if random.random() < 0.3:  # Complex analytical query
                query = """
                SELECT c.customer_name, SUM(o.total) as total_spent
                FROM customers c
                JOIN orders o ON c.id = o.customer_id
                WHERE o.date >= '2023-01-01'
                GROUP BY c.customer_name
                ORDER BY total_spent DESC
                """
                exec_time = random.uniform(500, 2000)
            else:  # Simple lookup
                query = "SELECT * FROM users WHERE id = 123"
                exec_time = random.uniform(1, 10)
        elif query_type == "insert":
            query = "INSERT INTO logs (message, timestamp) VALUES ('test', NOW())"
            exec_time = random.uniform(5, 20)
        else:
            query = "UPDATE users SET last_login = NOW() WHERE id = 456"
            exec_time = random.uniform(10, 50)

        timestamp = base_time + timedelta(seconds=i * 86.4)  # Spread over 24 hours

        # Add some temporal patterns (peak during business hours)
        if 9 <= timestamp.hour <= 17:
            exec_time *= 1.5  # Slower during peak hours

        query_log.append((query, timestamp, exec_time))

    # Analyze workload
    results = analyze_workload(query_log)

    print("=== Workload Analysis Results ===")
    print("\nCurrent Profile:")
    for key, value in results["current_profile"].items():
        print(f"  {key}: {value}")

    print("\nDominant Patterns:")
    for pattern in results["patterns"]:
        print(
            f"  Pattern {pattern['pattern_id']}: {pattern['type']} (freq: {pattern['frequency']})"
        )

    print("\nFuture Prediction (next 24 hours):")
    for key, value in results["future_prediction"].items():
        print(f"  {key}: {value}")
