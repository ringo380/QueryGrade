"""
SQL Query Analyzers Package

This package provides modular, extensible SQL query analysis using the Strategy pattern.
Each analyzer focuses on a specific aspect of query analysis.

Main components:
- QueryGrader: Orchestrates all analyzers
- BaseAnalyzer: Abstract base class for all analyzers
- Specialized analyzers: SELECT, JOIN, WHERE, indexing, etc.
- Database-specific analyzers: MySQL, PostgreSQL, SQLite, Oracle, SQL Server

Usage:
    from analyzer.analyzers import QueryGrader

    grader = QueryGrader()
    query, analysis = grader.analyze_query("SELECT * FROM users")
"""

# Convenience functions for backward compatibility
# Main orchestrator class
from .base import BaseAnalyzer, QueryGrader, analyze_query, grade_single_query

__all__ = [
    "QueryGrader",
    "BaseAnalyzer",
    "analyze_query",
    "grade_single_query",
]

__version__ = "2.0.0"
