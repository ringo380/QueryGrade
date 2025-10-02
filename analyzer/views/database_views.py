"""
Database introspection and context-aware analysis views.

This module handles:
- Database connection management
- Schema analysis and introspection
- Context-aware query analysis using live database metadata
"""
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django_ratelimit.decorators import ratelimit

from ..forms import DatabaseConnectionForm, QueryGradeForm
from ..models import UserQueryHistory
from ..query_analyzer import analyze_query
from ..query_optimizer import optimize_query_from_analysis
from .utils import get_client_ip
from .constants import GRADE_COLORS

logger = logging.getLogger(__name__)


@login_required
@ratelimit(key='user', rate='5/m', method='POST', block=True)
def database_analyze(request):
    """
    View for database architecture analysis using live database connections.

    This view allows users to connect to their database and analyze
    the schema, table structure, indexes, and query performance.
    """
    from ..database_introspector import DatabaseIntrospector

    if request.method == 'POST':
        form = DatabaseConnectionForm(request.POST)
        if form.is_valid():
            try:
                # Get database configuration
                db_config = form.get_connection_config()

                # Initialize database introspector
                introspector = DatabaseIntrospector(db_config)

                # Test connection
                if introspector.connect():
                    # Store connection config in session for analysis
                    request.session['db_config'] = db_config
                    messages.success(request, f"Successfully connected to {db_config['engine']} database '{db_config['name']}'!")
                    return redirect('database_schema')
                else:
                    messages.error(request, "Failed to connect to the database. Please check your connection parameters.")

            except Exception as e:
                logger.error(f"Database connection error for user {request.user.username}: {e}")
                messages.error(request, f"Database connection error: {str(e)}")
            finally:
                # Ensure connection is closed
                if 'introspector' in locals():
                    introspector.close()

    else:
        form = DatabaseConnectionForm()

    return render(request, 'analyzer/database_analyze.html', {'form': form})


@login_required
def database_schema(request):
    """
    Display database schema information and analysis.

    Shows tables, columns, indexes, foreign keys, and provides
    recommendations for schema optimization.
    """
    from ..database_introspector import DatabaseIntrospector

    # Get connection config from session
    db_config = request.session.get('db_config')
    if not db_config:
        messages.error(request, "No database connection found. Please connect to a database first.")
        return redirect('database_analyze')

    try:
        # Initialize database introspector
        introspector = DatabaseIntrospector(db_config)

        if introspector.connect():
            # Get all tables in the database
            tables = introspector.get_tables(db_config.get('schema'))

            # Generate schema analysis and recommendations
            schema_analysis = analyze_database_schema(tables, db_config)

            context = {
                'db_config': db_config,
                'tables': tables,
                'schema_analysis': schema_analysis,
                'table_count': len(tables),
                'total_columns': sum(len(table.columns) for table in tables),
                'total_indexes': sum(len(table.indexes) for table in tables),
                'total_foreign_keys': sum(len(table.foreign_keys) for table in tables),
            }

            return render(request, 'analyzer/database_schema.html', context)

        else:
            messages.error(request, "Failed to reconnect to the database. Please check your connection.")
            return redirect('database_analyze')

    except Exception as e:
        logger.error(f"Schema analysis error for user {request.user.username}: {e}")
        messages.error(request, f"Schema analysis error: {str(e)}")
        return redirect('database_analyze')
    finally:
        # Ensure connection is closed
        if 'introspector' in locals():
            introspector.close()


@login_required
def query_with_context(request):
    """
    Enhanced query grading with database context.

    Uses the connected database schema to provide more targeted
    recommendations based on actual table structure, indexes, etc.
    """
    from ..database_introspector import DatabaseIntrospector

    # Get connection config from session
    db_config = request.session.get('db_config')
    if not db_config:
        messages.error(request, "No database connection found. Please connect to a database first.")
        return redirect('database_analyze')

    if request.method == 'POST':
        form = QueryGradeForm(request.POST)
        if form.is_valid():
            try:
                sql_query = form.cleaned_data['sql_query']

                # Initialize database introspector
                introspector = DatabaseIntrospector(db_config)

                if introspector.connect():
                    # Analyze query with database context
                    context_analysis = introspector.analyze_query_context(sql_query)

                    # Get execution plan if supported
                    execution_plan = introspector.get_execution_plan(sql_query)

                    # Standard query analysis
                    query, analysis = analyze_query(
                        sql_query,
                        form.cleaned_data.get('database_type') or db_config['engine']
                    )

                    # Create user history record
                    user_history = UserQueryHistory.objects.create(
                        user=request.user,
                        query=query,
                        database_type=db_config['engine'],
                        database_version=form.cleaned_data.get('database_version', ''),
                        use_case_notes=form.cleaned_data.get('use_case_notes', ''),
                        ip_address=get_client_ip(request),
                        user_agent=request.META.get('HTTP_USER_AGENT', '')[:255]
                    )

                    # Store context analysis in session for results page
                    request.session['context_analysis'] = context_analysis
                    request.session['execution_plan'] = execution_plan

                    return redirect('contextualized_results', analysis_id=analysis.id)

                else:
                    messages.error(request, "Failed to reconnect to the database. Using standard analysis.")
                    return redirect('grade_query')

            except Exception as e:
                logger.error(f"Contextualized query analysis error for user {request.user.username}: {e}")
                messages.error(request, f"Analysis error: {str(e)}")
            finally:
                # Ensure connection is closed
                if 'introspector' in locals():
                    introspector.close()

    else:
        form = QueryGradeForm()
        # Pre-populate database type from connection
        if db_config:
            form.fields['database_type'].initial = db_config['engine']

    context = {
        'form': form,
        'db_config': db_config,
    }

    return render(request, 'analyzer/query_with_context.html', context)


@login_required
def contextualized_results(request, analysis_id):
    """
    Display query analysis results with database context.

    Shows standard analysis plus context-aware recommendations
    based on the actual database schema.
    """
    try:
        # Get the analysis
        user_history = UserQueryHistory.objects.get(
            query__analysis__id=analysis_id,
            user=request.user
        )
        analysis = user_history.query.analysis

        # Get context analysis from session
        context_analysis = request.session.get('context_analysis', {})
        execution_plan = request.session.get('execution_plan')
        db_config = request.session.get('db_config', {})

        context = {
            'analysis': analysis,
            'query': user_history.query,
            'user_history': user_history,
            'context_analysis': context_analysis,
            'execution_plan': execution_plan,
            'db_config': db_config,
            'has_context': bool(context_analysis),
            'grade_colors': GRADE_COLORS
        }

        return render(request, 'analyzer/contextualized_results.html', context)

    except UserQueryHistory.DoesNotExist:
        messages.error(request, "Analysis not found or you don't have permission to view it.")
        return redirect('query_history')
    except Exception as e:
        logger.error(f"Error displaying contextualized results for user {request.user.username}: {e}")
        messages.error(request, f"Error loading results: {str(e)}")
        return redirect('query_history')


def analyze_database_schema(tables, db_config):
    """
    Analyze database schema and provide optimization recommendations.

    Args:
        tables: List of TableInfo objects
        db_config: Database configuration dict

    Returns:
        Dict with analysis results and recommendations
    """
    analysis = {
        'recommendations': [],
        'issues': [],
        'performance_notes': [],
        'statistics': {
            'large_tables': [],
            'tables_without_primary_key': [],
            'tables_with_many_columns': [],
            'unused_indexes': [],
            'missing_foreign_key_indexes': []
        }
    }

    # Analyze each table
    for table in tables:
        # Check for missing primary keys
        has_primary_key = any(idx.get('primary', False) for idx in table.indexes)
        if not has_primary_key:
            analysis['issues'].append(f"Table '{table.name}' lacks a primary key")
            analysis['statistics']['tables_without_primary_key'].append(table.name)

        # Check for tables with many columns
        if len(table.columns) > 20:
            analysis['issues'].append(f"Table '{table.name}' has {len(table.columns)} columns (consider normalization)")
            analysis['statistics']['tables_with_many_columns'].append({
                'name': table.name,
                'column_count': len(table.columns)
            })

        # Check for large tables
        if table.row_count and table.row_count > 1000000:
            analysis['performance_notes'].append(f"Table '{table.name}' has {table.row_count:,} rows (consider partitioning)")
            analysis['statistics']['large_tables'].append({
                'name': table.name,
                'row_count': table.row_count,
                'size_mb': table.size_mb
            })

        # Check foreign key indexes
        for fk in table.foreign_keys:
            if fk['columns']:
                fk_column = fk['columns'][0]
                # Check if there's an index on the foreign key column
                indexed_columns = set()
                for idx in table.indexes:
                    indexed_columns.update(idx.get('columns', []))

                if fk_column not in indexed_columns:
                    analysis['issues'].append(f"Foreign key column '{table.name}.{fk_column}' should have an index")
                    analysis['statistics']['missing_foreign_key_indexes'].append({
                        'table': table.name,
                        'column': fk_column,
                        'references': f"{fk['referenced_table']}.{fk['referenced_columns'][0] if fk['referenced_columns'] else '?'}"
                    })

    # Generate general recommendations
    if analysis['statistics']['tables_without_primary_key']:
        analysis['recommendations'].append(
            "Add primary keys to all tables for better performance and replication support"
        )

    if analysis['statistics']['tables_with_many_columns']:
        analysis['recommendations'].append(
            "Consider normalizing tables with many columns to improve query performance"
        )

    if analysis['statistics']['large_tables']:
        analysis['recommendations'].append(
            "Consider partitioning large tables to improve query performance"
        )

    if analysis['statistics']['missing_foreign_key_indexes']:
        analysis['recommendations'].append(
            "Add indexes on foreign key columns to improve JOIN performance"
        )

    # Database-specific recommendations
    if db_config['engine'] == 'mysql':
        analysis['recommendations'].append(
            "Consider using InnoDB storage engine for all tables with foreign keys"
        )
    elif db_config['engine'] == 'postgresql':
        analysis['recommendations'].append(
            "Consider using partial indexes for queries with frequent WHERE conditions"
        )

    return analysis
