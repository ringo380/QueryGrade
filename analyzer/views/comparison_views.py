"""
Query comparison and batch analysis views.

This module handles:
- Query comparison (side-by-side analysis of multiple queries)
- Batch query analysis
- Comparison result generation
"""
import logging
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator

from ..forms import QueryCompareForm, BatchQueryForm
from ..query_analyzer import analyze_query, grade_single_query
from ..query_optimizer import optimize_query_from_analysis
from .constants import GRADE_COLORS

logger = logging.getLogger(__name__)


@login_required
def query_compare(request):
    """
    Handles the SQL query comparison interface.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == 'POST':
        form = QueryCompareForm(request.POST)
        if form.is_valid():
            # Store the comparison data in session for results page
            comparison_data = {
                'query_1': form.cleaned_data['query_1'],
                'query_1_name': form.cleaned_data.get('query_1_name', 'Query 1'),
                'query_2': form.cleaned_data['query_2'],
                'query_2_name': form.cleaned_data.get('query_2_name', 'Query 2'),
                'query_3': form.cleaned_data.get('query_3'),
                'query_3_name': form.cleaned_data.get('query_3_name', 'Query 3'),
                'database_type': form.cleaned_data.get('database_type', ''),
                'comparison_notes': form.cleaned_data.get('comparison_notes', ''),
            }

            # Store in session
            request.session['comparison_data'] = comparison_data

            return redirect('compare_results')
        else:
            messages.error(request, "Please correct the errors in the form below.")
    else:
        form = QueryCompareForm()

    return render(request, 'analyzer/query_compare.html', {'form': form})


@login_required
def compare_results(request):
    """
    Display the comparison results for multiple queries.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    # Get comparison data from session
    comparison_data = request.session.get('comparison_data')
    if not comparison_data:
        messages.error(request, "No comparison data found. Please submit queries for comparison.")
        return redirect('query_compare')

    try:
        results = []
        queries_to_analyze = [
            (comparison_data['query_1'], comparison_data['query_1_name']),
            (comparison_data['query_2'], comparison_data['query_2_name']),
        ]

        # Add third query if provided
        if comparison_data.get('query_3') and comparison_data['query_3'].strip():
            queries_to_analyze.append((comparison_data['query_3'], comparison_data['query_3_name']))

        # Analyze each query
        for query_sql, query_name in queries_to_analyze:
            try:
                query, analysis = analyze_query(query_sql, comparison_data['database_type'])

                # Generate optimization suggestions
                optimization_result = None
                if analysis.issues_found and len(analysis.issues_found) > 0:
                    try:
                        optimization_result = optimize_query_from_analysis(
                            query_sql,
                            analysis.issues_found,
                            comparison_data['database_type']
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate optimization suggestions: {e}")

                results.append({
                    'name': query_name,
                    'query': query,
                    'analysis': analysis,
                    'optimization': optimization_result,
                })
            except Exception as e:
                logger.error(f"Failed to analyze query '{query_name}': {e}")
                results.append({
                    'name': query_name,
                    'query': None,
                    'analysis': None,
                    'optimization': None,
                    'error': str(e)
                })

        # Generate comparison summary
        comparison_summary = generate_comparison_summary(results)

        context = {
            'results': results,
            'comparison_data': comparison_data,
            'comparison_summary': comparison_summary,
            'grade_colors': GRADE_COLORS
        }

        return render(request, 'analyzer/compare_results.html', context)

    except Exception as e:
        logger.error(f"Unexpected error in comparison: {e}")
        messages.error(request, "An unexpected error occurred while comparing queries.")
        return redirect('query_compare')


def generate_comparison_summary(results):
    """
    Generate a summary comparing the results of multiple queries.

    Args:
        results: List of query analysis results.

    Returns:
        dict: Summary comparison data.
    """
    summary = {
        'best_grade': None,
        'worst_grade': None,
        'best_query': None,
        'worst_query': None,
        'common_issues': [],
        'unique_issues': {},
        'performance_ranking': [],
        'recommendations': []
    }

    valid_results = [r for r in results if r.get('analysis') and not r.get('error')]

    if not valid_results:
        return summary

    # Grade mapping for comparison
    grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}

    # Find best and worst grades
    grades = [(r, grade_values.get(r['analysis'].grade, 0)) for r in valid_results]
    grades.sort(key=lambda x: x[1], reverse=True)

    if grades:
        summary['best_grade'] = grades[0][0]['analysis'].grade
        summary['best_query'] = grades[0][0]['name']
        summary['worst_grade'] = grades[-1][0]['analysis'].grade
        summary['worst_query'] = grades[-1][0]['name']
        summary['performance_ranking'] = [
            {
                'name': r[0]['name'],
                'grade': r[0]['analysis'].grade,
                'score': r[0]['analysis'].score
            }
            for r in grades
        ]

    # Analyze common and unique issues
    all_issues = []
    issue_by_query = {}

    for result in valid_results:
        if result['analysis'].issues_found:
            query_issues = [issue.get('type', 'UNKNOWN') for issue in result['analysis'].issues_found]
            all_issues.extend(query_issues)
            issue_by_query[result['name']] = query_issues

    # Find common issues (appear in multiple queries)
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1

    summary['common_issues'] = [
        issue for issue, count in issue_counts.items()
        if count > 1 and len(valid_results) > 1
    ]

    # Find unique issues per query
    for query_name, issues in issue_by_query.items():
        unique = [issue for issue in issues if issue_counts[issue] == 1]
        if unique:
            summary['unique_issues'][query_name] = unique

    # Generate recommendations
    if len(valid_results) > 1:
        best_result = grades[0][0] if grades else None
        worst_result = grades[-1][0] if grades else None

        if best_result and worst_result and best_result != worst_result:
            summary['recommendations'].append(
                f"Consider using patterns from '{best_result['name']}' (Grade {best_result['analysis'].grade}) "
                f"to improve '{worst_result['name']}' (Grade {worst_result['analysis'].grade})"
            )

        if summary['common_issues']:
            summary['recommendations'].append(
                f"All queries share these issues: {', '.join(summary['common_issues'])}. "
                "Focus on fixing these common problems first."
            )

    # Convert summary to formatted text
    formatted_summary = []

    if summary['best_query'] and summary['worst_query']:
        if summary['best_query'] == summary['worst_query']:
            formatted_summary.append(f"All queries performed equally with grade {summary['best_grade']}.")
        else:
            formatted_summary.append(f"Best performing query: '{summary['best_query']}' (Grade {summary['best_grade']})")
            formatted_summary.append(f"Query needing most improvement: '{summary['worst_query']}' (Grade {summary['worst_grade']})")

    if summary['performance_ranking']:
        formatted_summary.append("\nPerformance Ranking:")
        for i, query in enumerate(summary['performance_ranking'], 1):
            formatted_summary.append(f"{i}. {query['name']} - Grade {query['grade']} ({query['score']:.1f} points)")

    if summary['common_issues']:
        formatted_summary.append(f"\nCommon issues across all queries: {', '.join(summary['common_issues'])}")

    if summary['unique_issues']:
        formatted_summary.append("\nQuery-specific issues:")
        for query, issues in summary['unique_issues'].items():
            formatted_summary.append(f"• {query}: {', '.join(issues)}")

    if summary['recommendations']:
        formatted_summary.append("\nRecommendations:")
        for rec in summary['recommendations']:
            formatted_summary.append(f"• {rec}")

    return '\n'.join(formatted_summary) if formatted_summary else "No comparison data available."


@login_required
def batch_analysis(request):
    """
    View for batch analysis of multiple SQL queries.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == 'POST':
        form = BatchQueryForm(request.POST)
        if form.is_valid():
            try:
                # Get parsed queries from the form
                queries = form.get_parsed_queries()

                # Store batch data in session
                batch_data = {
                    'queries': queries,
                    'database_type': form.cleaned_data.get('database_type', ''),
                    'database_version': form.cleaned_data.get('database_version', ''),
                    'analysis_notes': form.cleaned_data.get('analysis_notes', ''),
                    'total_queries': len(queries)
                }
                request.session['batch_data'] = batch_data

                return redirect('batch_results')

            except Exception as e:
                messages.error(request, f"An error occurred while processing your queries: {str(e)}")

    else:
        form = BatchQueryForm()

    return render(request, 'analyzer/batch_analysis.html', {'form': form})


@login_required
def batch_results(request):
    """
    Display batch analysis results for multiple queries.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    # Get batch data from session
    batch_data = request.session.get('batch_data')
    if not batch_data:
        messages.error(request, "No batch analysis data found. Please submit queries for analysis.")
        return redirect('batch_analysis')

    try:
        results = []
        queries = batch_data.get('queries', [])

        # Analyze each query individually
        for i, query_text in enumerate(queries, 1):
            try:
                analysis = grade_single_query(
                    query_text,
                    database_type=batch_data.get('database_type'),
                    database_version=batch_data.get('database_version')
                )

                results.append({
                    'query_number': i,
                    'query_text': query_text,
                    'analysis': analysis,
                    'error': None
                })

            except Exception as e:
                results.append({
                    'query_number': i,
                    'query_text': query_text,
                    'analysis': None,
                    'error': str(e)
                })

        # Generate batch summary statistics
        batch_summary = generate_batch_summary(results, batch_data)

        context = {
            'results': results,
            'batch_data': batch_data,
            'batch_summary': batch_summary,
            'grade_colors': GRADE_COLORS
        }

        return render(request, 'analyzer/batch_results.html', context)

    except Exception as e:
        messages.error(request, f"Unexpected error in batch analysis: {str(e)}")
        return redirect('batch_analysis')


def generate_batch_summary(results, batch_data):
    """
    Generate summary statistics for batch query analysis.

    Args:
        results: List of query analysis results.
        batch_data: Batch processing data.

    Returns:
        str: Formatted summary statistics.
    """
    successful_results = [r for r in results if r['analysis'] and not r['error']]
    failed_results = [r for r in results if r['error']]

    if not successful_results and not failed_results:
        return "No results to summarize."

    summary_lines = []

    # Basic statistics
    total_queries = len(results)
    successful_count = len(successful_results)
    failed_count = len(failed_results)

    summary_lines.append(f"Batch Analysis Summary")
    summary_lines.append(f"Total Queries Analyzed: {total_queries}")
    summary_lines.append(f"Successfully Analyzed: {successful_count}")
    if failed_count > 0:
        summary_lines.append(f"Failed to Analyze: {failed_count}")

    if successful_results:
        # Grade distribution
        grades = [r['analysis'].grade for r in successful_results]
        grade_counts = {grade: grades.count(grade) for grade in set(grades)}

        summary_lines.append(f"\nGrade Distribution:")
        for grade in ['A', 'B', 'C', 'D', 'F']:
            if grade in grade_counts:
                summary_lines.append(f"• Grade {grade}: {grade_counts[grade]} queries")

        # Average score
        scores = [r['analysis'].score for r in successful_results]
        avg_score = sum(scores) / len(scores)
        summary_lines.append(f"\nAverage Score: {avg_score:.1f}/100")

        # Best and worst queries
        best_result = max(successful_results, key=lambda x: x['analysis'].score)
        worst_result = min(successful_results, key=lambda x: x['analysis'].score)

        summary_lines.append(f"\nBest Query: Query #{best_result['query_number']} (Grade {best_result['analysis'].grade}, {best_result['analysis'].score:.1f} points)")
        if best_result != worst_result:
            summary_lines.append(f"Query Needing Most Improvement: Query #{worst_result['query_number']} (Grade {worst_result['analysis'].grade}, {worst_result['analysis'].score:.1f} points)")

        # Common issues analysis
        all_issues = []
        for result in successful_results:
            if result['analysis'].issues_found:
                all_issues.extend(result['analysis'].issues_found)

        if all_issues:
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

            # Show issues that appear in multiple queries
            common_issues = {issue: count for issue, count in issue_counts.items() if count > 1}
            if common_issues:
                summary_lines.append(f"\nCommon Issues Found:")
                for issue, count in sorted(common_issues.items(), key=lambda x: x[1], reverse=True):
                    summary_lines.append(f"• {issue}: Found in {count} queries")

    if failed_count > 0:
        summary_lines.append(f"\nNote: {failed_count} queries failed analysis due to errors.")

    return '\n'.join(summary_lines)
