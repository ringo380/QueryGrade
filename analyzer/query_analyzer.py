"""
SQL Query Analyzer - Facade Module

This module provides backward compatibility with the original QueryGrader API
while delegating to the new modular analyzer architecture.

The original monolithic implementation has been refactored into:
- analyzer/analyzers/base.py - Core orchestration
- analyzer/analyzers/select_analyzer.py - SELECT analysis
- analyzer/analyzers/join_analyzer.py - JOIN analysis
- analyzer/analyzers/where_analyzer.py - WHERE analysis
- analyzer/analyzers/database/ - Database-specific analyzers

For new code, prefer importing from analyzer.analyzers directly:
    from analyzer.analyzers import QueryGrader, analyze_query

This facade maintains compatibility with existing code:
    from analyzer.query_analyzer import QueryGrader, analyze_query
"""

from typing import Tuple, Optional
from .models import Query, QueryAnalysis

# Import the new modular implementation
from .analyzers import QueryGrader as ModularQueryGrader
from .analyzers import analyze_query as modular_analyze_query
from .analyzers import grade_single_query as modular_grade_single_query


# Legacy class name for backward compatibility
class QueryGrader(ModularQueryGrader):
    """
    Legacy QueryGrader class for backward compatibility.

    This class delegates to the new modular QueryGrader implementation
    in analyzer.analyzers.base while maintaining the original API.

    For new code, import from analyzer.analyzers instead:
        from analyzer.analyzers import QueryGrader
    """
    pass


# Legacy convenience functions for backward compatibility
def analyze_query(sql_text: str, database_type: str = '', use_ml: Optional[bool] = None) -> Tuple[Query, QueryAnalysis]:
    """
    Convenience function to analyze a SQL query with optional ML integration.

    Maintains backward compatibility with the original API.

    Args:
        sql_text (str): The SQL query to analyze
        database_type (str): The target database type
        use_ml (bool, optional): Whether to use ML-enhanced grading

    Returns:
        Tuple[Query, QueryAnalysis]: The created Query and QueryAnalysis objects

    Note:
        This function delegates to analyzer.analyzers.analyze_query
        For new code, import from analyzer.analyzers directly
    """
    return modular_analyze_query(sql_text, database_type, use_ml)


def grade_single_query(sql_text: str, database_type: str = '',
                      database_version: str = '', use_ml: Optional[bool] = None) -> QueryAnalysis:
    """
    Convenience function to grade a single SQL query and return just the analysis.

    Args:
        sql_text (str): The SQL query to analyze
        database_type (str): The target database type
        database_version (str): The database version
        use_ml (bool, optional): Whether to use ML-enhanced grading

    Returns:
        QueryAnalysis: The analysis object with grade and recommendations

    Note:
        This function delegates to analyzer.analyzers.grade_single_query
        For new code, import from analyzer.analyzers directly
    """
    return modular_grade_single_query(sql_text, database_type, database_version, use_ml)


# For reference: The legacy implementation is preserved in query_analyzer_legacy.py
# Original file size: 1,235 lines
# New modular architecture:
#   - base.py: ~360 lines
#   - select_analyzer.py: ~100 lines
#   - join_analyzer.py: ~90 lines
#   - where_analyzer.py: ~120 lines
#   - utils.py: ~180 lines
#   - database/mysql_analyzer.py: ~100 lines
#   - database/postgresql_analyzer.py: ~100 lines
#   Total: ~1,050 lines across 7 focused modules
#
# Benefits:
#   - Clear separation of concerns
#   - Easy to test individual analyzers
#   - Extensible architecture for new analyzers
#   - Backward compatible with existing code