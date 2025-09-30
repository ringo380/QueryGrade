from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.contrib import messages
from django.core.paginator import Paginator
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.core.cache import caches
from django.views.decorators.http import require_http_methods, require_POST
from django_ratelimit.decorators import ratelimit
from django.views.decorators.cache import never_cache
from .forms import UploadLogForm, QueryGradeForm, QueryCompareForm, BatchQueryForm, QueryFeedbackForm, DatabaseConnectionForm
from .parser import process_slow_log, process_general_log
from .query_analyzer import analyze_query, grade_single_query
from .query_optimizer import optimize_query_from_analysis
from .ml.unified_query_analyzer import UnifiedQueryAnalyzer, AnalysisRequest
import asyncio
import json
from .models import Query, QueryAnalysis, UserQueryHistory, QueryFeedback
from .tasks import process_log_file_async, batch_analyze_queries, analyze_database_schema_async, generate_performance_report
from .performance import optimize_view_performance, PerformanceMonitor, memory_optimizer
from django.db import models
import logging
import tempfile
import os
import uuid
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

def analyze(request):
    """
    Handles the analyze view.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    return render(request, 'analyzer/index.html')

@ratelimit(key='ip', rate='5/m', method='POST', block=True)
@ratelimit(key='user', rate='10/m', method='POST', block=True)
def index(request):
    """
    Handles the upload and processing of MySQL log files.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if not request.user.is_authenticated:
        return redirect('login')

    if request.method == 'POST':
        form = UploadLogForm(request.POST, request.FILES)
        if form.is_valid():
            log_type = form.cleaned_data['log_type']
            log_file = request.FILES['log_file']

            # Check if async processing is requested
            use_async = form.cleaned_data.get('use_async', False)

            if use_async:
                # Save file to temp directory for async processing with enhanced security
                try:
                    # Create secure temp file with restrictive permissions
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix='.log',
                        prefix='secure_upload_',
                        dir=tempfile.gettempdir()
                    )

                    # Set restrictive file permissions (owner read/write only)
                    os.chmod(temp_file.name, 0o600)

                    # Write file content securely
                    file_content = log_file.read()
                    temp_file.write(file_content)
                    temp_file.close()

                    # Additional security check after write
                    if os.path.getsize(temp_file.name) != len(file_content):
                        os.unlink(temp_file.name)  # Clean up
                        logger.error(f"File size mismatch during upload for user {request.user.username}")
                        messages.error(request, "File upload verification failed. Please try again.")
                        return HttpResponseRedirect(reverse('index'))

                    # Start async task
                    task = process_log_file_async.delay(temp_file.name, log_type, request.user.id)

                    # Store task ID in session for progress tracking
                    request.session['log_processing_task_id'] = task.id

                    messages.info(request, f"File upload successful! Your {log_type} log is being processed in the background. You'll be notified when it's complete.")
                    return redirect('async_processing_status')

                except (OSError, IOError) as e:
                    logger.error(f"Secure file upload failed for user {request.user.username}: {e}")
                    messages.error(request, "File upload failed due to security restrictions. Please try again.")
                    return HttpResponseRedirect(reverse('index'))
            else:
                # Synchronous processing with enhanced security
                try:
                    # Create secure storage with custom settings
                    fs = FileSystemStorage()

                    # Generate secure filename to prevent path traversal
                    secure_filename = f"secure_{uuid.uuid4().hex}_{log_file.name}"
                    secure_filename = secure_filename.replace('..', '').replace('/', '').replace('\\', '')

                    # Save file with security checks
                    filename = fs.save(secure_filename, log_file)
                    uploaded_file_url = fs.path(filename)

                    # Set restrictive file permissions
                    os.chmod(uploaded_file_url, 0o600)

                except (OSError, IOError) as e:
                    logger.error(f"Secure file storage failed for user {request.user.username}: {e}")
                    messages.error(request, "File storage failed due to security restrictions. Please try again.")
                    return HttpResponseRedirect(reverse('index'))

                try:
                    # Analyze the log file
                    if log_type == 'slow':
                        df_anomalies = process_slow_log(uploaded_file_url)
                    elif log_type == 'general':
                        df_anomalies = process_general_log(uploaded_file_url)

                    # Prepare data for template
                    anomalies = df_anomalies.to_dict('records') if df_anomalies is not None else []

                    # Check if any anomalies were found
                    if not anomalies:
                        messages.info(request, "No anomalies detected in your log file. This suggests your queries are performing well!")

                    # Delete the uploaded file after processing
                    fs.delete(filename)

                    # Paginate the results
                    paginator = Paginator(anomalies, 10)  # Show 10 anomalies per page
                    page_number = request.GET.get('page')
                    page_obj = paginator.get_page(page_number)

                    return render(request, 'analyzer/results.html', {'page_obj': page_obj})
                except FileNotFoundError:
                    logger.error(f"Log file not found: {uploaded_file_url}")
                    messages.error(request, "The uploaded file could not be found. Please try uploading again.")
                    # Clean up if file exists
                    try:
                        fs.delete(filename)
                    except FileNotFoundError:
                        logger.debug(f"File {filename} already deleted or not found")
                    except Exception as e:
                        logger.warning(f"Failed to delete file {filename}: {e}")
                    return HttpResponseRedirect(reverse('index'))
                except pd.errors.EmptyDataError:
                    logger.warning(f"Empty log file uploaded by user {request.user.username}")
                    messages.warning(request, "The uploaded log file appears to be empty. Please upload a valid MySQL log file.")
                    # Clean up
                    try:
                        fs.delete(filename)
                    except FileNotFoundError:
                        logger.debug(f"File {filename} already deleted or not found")
                    except Exception as e:
                        logger.warning(f"Failed to delete file {filename}: {e}")
                    return HttpResponseRedirect(reverse('index'))
                except pd.errors.ParserError as e:
                    logger.warning(f"Log file parsing error for user {request.user.username}: {e}")
                    messages.error(request, "Unable to parse the log file format. Please ensure you're uploading a valid MySQL log file.")
                    # Clean up
                    try:
                        fs.delete(filename)
                    except FileNotFoundError:
                        logger.debug(f"File {filename} already deleted or not found")
                    except Exception as e:
                        logger.warning(f"Failed to delete file {filename}: {e}")
                    return HttpResponseRedirect(reverse('index'))
                except Exception as e:
                    logger.error(f"Unexpected error processing log file for user {request.user.username}: {e}")
                    messages.error(request, "An unexpected error occurred while processing the log file. Please check the file format and try again.")
                    # Clean up
                    try:
                        fs.delete(filename)
                    except FileNotFoundError:
                        logger.debug(f"File {filename} already deleted or not found")
                    except Exception as e:
                        logger.warning(f"Failed to delete file {filename}: {e}")
                    return HttpResponseRedirect(reverse('index'))
        else:
            messages.error(request, "There was an error with the form submission. Please check the fields and try again.")
    else:
        form = UploadLogForm()
    return render(request, 'analyzer/index.html', {'form': form})

@ratelimit(key='ip', rate='5/5m', method='POST', block=True)
@never_cache
def login_view(request):
    """
    Handles the login view.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
    else:
        form = AuthenticationForm()
    return render(request, 'analyzer/login.html', {'form': form})

def logout_view(request):
    """
    Handles the logout view.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    logout(request)
    return redirect('login')

@ratelimit(key='ip', rate='3/h', method='POST', block=True)
def register_view(request):
    """
    Handles the registration view.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'analyzer/register.html', {'form': form})


@login_required
@ratelimit(key='user', rate='20/m', method='POST', block=True)
@PerformanceMonitor.time_function("grade_query_view")
def grade_query(request):
    """
    Handles the SQL query grading interface.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == 'POST':
        form = QueryGradeForm(request.POST)
        if form.is_valid():
            sql_query = form.cleaned_data['sql_query']
            database_type = form.cleaned_data.get('database_type', '')
            database_version = form.cleaned_data.get('database_version', '')
            use_case_notes = form.cleaned_data.get('use_case_notes', '')

            try:
                # Analyze the query with both traditional and ML analysis
                query, analysis = analyze_query(sql_query, database_type)

                # Enhanced ML analysis
                ml_analysis = None
                try:
                    unified_analyzer = UnifiedQueryAnalyzer()
                    analysis_request = AnalysisRequest(
                        query=sql_query,
                        user_id=str(request.user.id),
                        database_type=database_type,
                        database_version=database_version,
                        context={
                            'use_case': use_case_notes,
                            'user_agent': request.META.get('HTTP_USER_AGENT', ''),
                            'ip_address': get_client_ip(request)
                        }
                    )

                    # Run async analysis in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        ml_analysis = loop.run_until_complete(
                            unified_analyzer.analyze_query(analysis_request)
                        )
                    finally:
                        loop.close()

                    # Store ML analysis in session for results page
                    request.session['ml_analysis'] = {
                        'semantic_metrics': ml_analysis.semantic_metrics,
                        'performance_prediction': ml_analysis.performance_prediction,
                        'feedback': ml_analysis.feedback,
                        'recommendations': ml_analysis.recommendations,
                        'personalized_feedback': ml_analysis.personalized_feedback,
                        'rewrite_suggestions': ml_analysis.rewrite_suggestions
                    }

                    logger.info(f"Enhanced ML analysis completed for user {request.user.username}")

                except Exception as ml_error:
                    logger.warning(f"ML analysis failed for user {request.user.username}: {ml_error}")
                    # Continue with traditional analysis even if ML fails

                # Create user history record
                user_history = UserQueryHistory.objects.create(
                    user=request.user,
                    query=query,
                    ip_address=get_client_ip(request),
                    user_agent=request.META.get('HTTP_USER_AGENT', '')[:255],
                    database_type=database_type,
                    database_version=database_version,
                    use_case_notes=use_case_notes
                )

                # Redirect to enhanced results page
                return redirect('enhanced_grade_results', analysis_id=analysis.id)

            except ValueError as e:
                # Handle SQL syntax errors with specific feedback
                error_msg = str(e)
                if "typos in keywords" in error_msg:
                    messages.error(request, "SQL syntax error: Your query contains apparent typos in SQL keywords. Please check your spelling.")
                elif "Unable to parse" in error_msg:
                    messages.error(request, "SQL parsing error: We couldn't parse your SQL query. Please check the syntax and try again.")
                elif "No SQL keywords found" in error_msg:
                    messages.error(request, "Invalid input: No SQL keywords detected. Please enter a valid SQL query.")
                else:
                    messages.error(request, f"SQL error: {error_msg}")
                logger.warning(f"SQL syntax error for user {request.user.username}: {e}")
                return render(request, 'analyzer/grade_form.html', {'form': form})
            except Exception as e:
                logger.error(f"Unexpected error analyzing query for user {request.user.username}: {e}")
                messages.error(request, "An unexpected error occurred while analyzing your query. Please try again or contact support if the problem persists.")
                return render(request, 'analyzer/grade_form.html', {'form': form})
        else:
            messages.error(request, "Please correct the errors in the form below.")
    else:
        form = QueryGradeForm()

    return render(request, 'analyzer/grade_form.html', {'form': form})


@login_required
def grade_results(request, analysis_id):
    """
    Display the grading results for a query analysis.

    Args:
        request: The HTTP request object.
        analysis_id: ID of the QueryAnalysis object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    try:
        analysis = get_object_or_404(QueryAnalysis, id=analysis_id)
    except:
        messages.error(request, "The requested analysis could not be found.")
        return redirect('grade_query')

    # Check if the current user has access to this analysis
    try:
        user_history = UserQueryHistory.objects.get(
            user=request.user,
            query=analysis.query
        )
    except UserQueryHistory.DoesNotExist:
        logger.warning(f"User {request.user.username} attempted to access analysis {analysis_id} without permission")
        messages.error(request, "You don't have permission to view this analysis.")
        return redirect('grade_query')

    # Generate optimization suggestions if there are issues
    optimization_result = None
    if analysis.issues_found and len(analysis.issues_found) > 0:
        try:
            database_type = user_history.database_type if user_history.database_type else ''
            optimization_result = optimize_query_from_analysis(
                analysis.query.sql_text,
                analysis.issues_found,
                database_type
            )
        except Exception as e:
            logger.warning(f"Failed to generate optimization suggestions: {e}")

    context = {
        'analysis': analysis,
        'query': analysis.query,
        'user_history': user_history,
        'optimization_result': optimization_result,
        'grade_colors': {
            'A': 'success',  # Green
            'B': 'info',     # Blue
            'C': 'warning',  # Yellow
            'D': 'orange',   # Orange
            'F': 'danger'    # Red
        }
    }

    return render(request, 'analyzer/grade_results.html', context)


@login_required
def enhanced_grade_results(request, analysis_id):
    """
    Display enhanced grading results with ML analysis for a query.

    Args:
        request: The HTTP request object.
        analysis_id: ID of the QueryAnalysis object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    try:
        analysis = get_object_or_404(QueryAnalysis, id=analysis_id)
    except:
        messages.error(request, "The requested analysis could not be found.")
        return redirect('grade_query')

    # Check if the current user has access to this analysis
    try:
        user_history = UserQueryHistory.objects.get(
            user=request.user,
            query=analysis.query
        )
    except UserQueryHistory.DoesNotExist:
        logger.warning(f"User {request.user.username} attempted to access analysis {analysis_id} without permission")
        messages.error(request, "You don't have permission to view this analysis.")
        return redirect('grade_query')

    # Get ML analysis from session
    ml_analysis = request.session.get('ml_analysis', {})

    # Generate optimization suggestions if there are issues
    optimization_result = None
    if analysis.issues_found and len(analysis.issues_found) > 0:
        try:
            database_type = user_history.database_type if user_history.database_type else ''
            optimization_result = optimize_query_from_analysis(
                analysis.query.sql_text,
                analysis.issues_found,
                database_type
            )
        except Exception as e:
            logger.warning(f"Failed to generate optimization suggestions: {e}")

    # Process ML analysis data for template
    processed_ml_analysis = {}
    if ml_analysis:
        try:
            processed_ml_analysis = {
                'has_ml_analysis': True,
                'semantic_score': ml_analysis.get('semantic_metrics', {}).get('overall_score', 0),
                'complexity_level': ml_analysis.get('semantic_metrics', {}).get('complexity_level', 'Unknown'),
                'query_intent': ml_analysis.get('semantic_metrics', {}).get('query_intent', 'Unknown'),
                'performance_prediction': ml_analysis.get('performance_prediction', {}),
                'feedback': ml_analysis.get('feedback', {}),
                'recommendations': ml_analysis.get('recommendations', []),
                'personalized_feedback': ml_analysis.get('personalized_feedback', {}),
                'rewrite_suggestions': ml_analysis.get('rewrite_suggestions', [])
            }
        except Exception as e:
            logger.warning(f"Error processing ML analysis data: {e}")
            processed_ml_analysis = {'has_ml_analysis': False}
    else:
        processed_ml_analysis = {'has_ml_analysis': False}

    context = {
        'analysis': analysis,
        'query': analysis.query,
        'user_history': user_history,
        'optimization_result': optimization_result,
        'ml_analysis': processed_ml_analysis,
        'grade_colors': {
            'A': 'success',  # Green
            'B': 'info',     # Blue
            'C': 'warning',  # Yellow
            'D': 'orange',   # Orange
            'F': 'danger'    # Red
        }
    }

    return render(request, 'analyzer/enhanced_grade_results.html', context)


@login_required
def user_query_history(request):
    """
    Display the user's query history.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    history = UserQueryHistory.objects.filter(user=request.user).select_related('query', 'query__analysis')

    # Paginate the results
    paginator = Paginator(history, 10)  # Show 10 queries per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'analyzer/query_history.html', {'page_obj': page_obj})


def get_client_ip(request):
    """
    Get the client's IP address from the request.

    Args:
        request: The HTTP request object.

    Returns:
        str: The client's IP address.
    """
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


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
            'grade_colors': {
                'A': 'success',  # Green
                'B': 'info',     # Blue
                'C': 'warning',  # Yellow
                'D': 'orange',   # Orange
                'F': 'danger'    # Red
            }
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
            'grade_colors': {
                'A': 'success',  # Green
                'B': 'info',     # Blue
                'C': 'warning',  # Orange
                'D': 'danger',   # Red
                'F': 'danger',   # Red
            }
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


@login_required
def submit_feedback(request, analysis_id):
    """Submit feedback for a query analysis."""
    try:
        user_history = UserQueryHistory.objects.get(
            query__analysis__id=analysis_id,
            user=request.user
        )
    except UserQueryHistory.DoesNotExist:
        logger.warning(f"User {request.user.username} attempted to provide feedback for non-existent analysis {analysis_id}")
        messages.error(request, "Analysis not found or you don't have permission to provide feedback.")
        return redirect('query_history')

    # Check if feedback already exists
    existing_feedback = QueryFeedback.objects.filter(user_history=user_history).first()

    if request.method == 'POST':
        form = QueryFeedbackForm(request.POST)
        if form.is_valid():
            if existing_feedback:
                # Update existing feedback
                existing_feedback.accuracy_rating = int(form.cleaned_data['accuracy_rating'])
                existing_feedback.usefulness_rating = int(form.cleaned_data['usefulness_rating'])
                existing_feedback.clarity_rating = int(form.cleaned_data['clarity_rating'])
                existing_feedback.suggestions = form.cleaned_data['suggestions']
                existing_feedback.would_recommend = form.cleaned_data['would_recommend']
                existing_feedback.save()

                logger.info(f"User {request.user.username} updated feedback for analysis {analysis_id}")
                messages.success(request, "Thank you! Your feedback has been updated.")
            else:
                # Create new feedback
                feedback = QueryFeedback.objects.create(
                    user_history=user_history,
                    accuracy_rating=int(form.cleaned_data['accuracy_rating']),
                    usefulness_rating=int(form.cleaned_data['usefulness_rating']),
                    clarity_rating=int(form.cleaned_data['clarity_rating']),
                    suggestions=form.cleaned_data['suggestions'],
                    would_recommend=form.cleaned_data['would_recommend']
                )

                logger.info(f"User {request.user.username} submitted feedback for analysis {analysis_id}")
                messages.success(request, "Thank you for your feedback! It helps us improve QueryGrade.")

            # Update user history feedback flags
            user_history.was_helpful = True
            user_history.feedback_comments = form.cleaned_data['suggestions']
            user_history.save()

            return redirect('grade_results', analysis_id=analysis_id)
        else:
            messages.error(request, "Please correct the errors in the feedback form.")
    else:
        # Pre-populate form if feedback exists
        initial_data = {}
        if existing_feedback:
            initial_data = {
                'accuracy_rating': existing_feedback.accuracy_rating,
                'usefulness_rating': existing_feedback.usefulness_rating,
                'clarity_rating': existing_feedback.clarity_rating,
                'suggestions': existing_feedback.suggestions,
                'would_recommend': existing_feedback.would_recommend,
            }
        form = QueryFeedbackForm(initial=initial_data)

    context = {
        'form': form,
        'user_history': user_history,
        'analysis': user_history.query.analysis,
        'existing_feedback': existing_feedback,
    }

    return render(request, 'analyzer/feedback_form.html', context)


@login_required
@require_POST
def quick_feedback(request, analysis_id):
    """Handle quick thumbs up/down feedback via AJAX."""
    import json
    from django.http import JsonResponse
    from django.views.decorators.csrf import csrf_exempt
    from django.views.decorators.http import require_POST

    try:
        # Get the user's query history for this analysis
        user_history = UserQueryHistory.objects.get(
            query__analysis__id=analysis_id,
            user=request.user
        )
    except UserQueryHistory.DoesNotExist:
        logger.warning(f"User {request.user.username} attempted quick feedback for non-existent analysis {analysis_id}")
        return JsonResponse({
            'success': False,
            'error': 'Analysis not found or you don\'t have permission to provide feedback.'
        }, status=404)

    try:
        # Parse JSON data
        data = json.loads(request.body)
        was_helpful = data.get('was_helpful')

        if was_helpful is None:
            return JsonResponse({
                'success': False,
                'error': 'Missing feedback data'
            }, status=400)

        # Update the user history with quick feedback
        user_history.was_helpful = bool(was_helpful)
        user_history.save()

        # Try to create or update the FeedbackLearning record for ML training
        try:
            from .ml.feedback_collector import FeedbackCollector
            from .models import FeedbackLearning

            # Convert thumbs up/down to grade equivalent (1-5 scale)
            feedback_grade = 4.0 if was_helpful else 2.0
            feedback_score = (feedback_grade - 1) * 25  # Convert to 0-100 scale

            analysis = user_history.query.analysis
            grade_difference = feedback_score - analysis.score

            # Create or update learning record
            learning_record, created = FeedbackLearning.objects.update_or_create(
                user_history=user_history,
                defaults={
                    'original_grade': analysis.grade,
                    'original_score': analysis.score,
                    'original_confidence': 0.8,  # Default system confidence
                    'feedback_grade_equivalent': feedback_score,
                    'grade_difference': grade_difference,
                    'feedback_weight': 0.7,  # Medium weight for quick feedback
                    'user_reliability_score': 0.5,  # Default for new feedback
                    'context_similarity_score': 0.0,
                }
            )

            logger.info(f"{'Created' if created else 'Updated'} ML learning record for analysis {analysis_id}")

        except Exception as ml_error:
            # ML processing failed but don't fail the entire request
            logger.warning(f"ML processing failed for quick feedback: {str(ml_error)}")

        logger.info(f"User {request.user.username} submitted quick feedback ({'helpful' if was_helpful else 'not helpful'}) for analysis {analysis_id}")

        return JsonResponse({
            'success': True,
            'message': f'Thank you for your feedback! This helps us improve QueryGrade.',
            'feedback_type': 'helpful' if was_helpful else 'not_helpful'
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)

    except Exception as e:
        logger.error(f"Error processing quick feedback for analysis {analysis_id}: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred while processing your feedback. Please try again.'
        }, status=500)


@login_required
def feedback_analytics(request):
    """View feedback analytics (admin only for now)."""
    if not request.user.is_staff:
        messages.error(request, "You don't have permission to view feedback analytics.")
        return redirect('grade_query')

    # Get feedback statistics
    feedback_data = QueryFeedback.objects.all()
    total_feedback = feedback_data.count()

    if total_feedback > 0:
        avg_accuracy = feedback_data.aggregate(avg=models.Avg('accuracy_rating'))['avg'] or 0
        avg_usefulness = feedback_data.aggregate(avg=models.Avg('usefulness_rating'))['avg'] or 0
        avg_clarity = feedback_data.aggregate(avg=models.Avg('clarity_rating'))['avg'] or 0

        recommend_count = feedback_data.filter(would_recommend=True).count()
        recommend_percentage = (recommend_count / total_feedback) * 100 if total_feedback > 0 else 0

        # Get recent feedback
        recent_feedback = feedback_data.order_by('-created_at')[:10]

        context = {
            'total_feedback': total_feedback,
            'avg_accuracy': round(avg_accuracy, 2),
            'avg_usefulness': round(avg_usefulness, 2),
            'avg_clarity': round(avg_clarity, 2),
            'recommend_percentage': round(recommend_percentage, 2),
            'recent_feedback': recent_feedback,
        }
    else:
        context = {
            'total_feedback': 0,
            'message': 'No feedback data available yet.'
        }

    return render(request, 'analyzer/feedback_analytics.html', context)


@login_required
@ratelimit(key='user', rate='5/m', method='POST', block=True)
def database_analyze(request):
    """
    View for database architecture analysis using live database connections.

    This view allows users to connect to their database and analyze
    the schema, table structure, indexes, and query performance.
    """
    from .database_introspector import DatabaseIntrospector

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
    from .database_introspector import DatabaseIntrospector

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
    from .database_introspector import DatabaseIntrospector

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
                        ip_address=request.META.get('REMOTE_ADDR'),
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


# Async Processing Views

@login_required
def async_processing_status(request):
    """View to show async processing status."""
    task_id = request.session.get('log_processing_task_id')

    if not task_id:
        messages.error(request, "No processing task found.")
        return redirect('index')

    context = {
        'task_id': task_id,
        'page_title': 'Processing Status'
    }

    return render(request, 'analyzer/async_status.html', context)


@login_required
@require_http_methods(["GET"])
def check_task_status(request, task_id):
    """Check the status of an async task."""
    from celery.result import AsyncResult

    try:
        task = AsyncResult(task_id)

        response_data = {
            'task_id': task_id,
            'status': task.status,
            'ready': task.ready()
        }

        if task.ready():
            if task.successful():
                result = task.result
                response_data.update({
                    'success': True,
                    'result': result
                })

                # Clear task ID from session
                if request.session.get('log_processing_task_id') == task_id:
                    del request.session['log_processing_task_id']

            else:
                response_data.update({
                    'success': False,
                    'error': str(task.info)
                })
        else:
            response_data['progress'] = getattr(task.info, 'current', 0) if hasattr(task.info, 'current') else 0

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error checking task status: {str(e)}")
        return JsonResponse({
            'task_id': task_id,
            'status': 'ERROR',
            'success': False,
            'error': 'Failed to check task status'
        })


@login_required
def async_results(request):
    """Display results from async processing."""
    cache_key = request.GET.get('cache_key')

    if not cache_key:
        messages.error(request, "No results found.")
        return redirect('index')

    cache = caches['process_cache']
    results = cache.get(cache_key)

    if not results:
        messages.error(request, "Results have expired or are not available.")
        return redirect('index')

    # Handle different types of results
    if 'anomalies' in results:
        # Log file processing results
        anomalies = results.get('anomalies', [])

        # Paginate the results
        paginator = Paginator(anomalies, 10)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        context = {
            'page_obj': page_obj,
            'processing_time': results.get('processing_time', 0),
            'total_queries': len(anomalies),
            'anomaly_count': len([a for a in anomalies if a.get('is_anomaly', False)])
        }

        return render(request, 'analyzer/results.html', context)

    # Handle other result types (batch analysis, schema analysis, etc.)
    return render(request, 'analyzer/async_results.html', {'results': results})


@login_required
def batch_analysis_view(request):
    """Handle batch query analysis."""
    if request.method == 'POST':
        form = BatchQueryForm(request.POST)
        if form.is_valid():
            queries_text = form.cleaned_data['queries']
            use_async = form.cleaned_data.get('use_async', False)

            # Parse queries (split by semicolon and clean)
            query_list = [q.strip() for q in queries_text.split(';') if q.strip()]

            if not query_list:
                messages.error(request, "No valid queries found.")
                return redirect('batch_analysis')

            if use_async:
                # Prepare query data for async processing
                query_data_list = []
                for query in query_list:
                    query_data_list.append({
                        'sql_query': query,
                        'database_version': form.cleaned_data.get('database_version', ''),
                        'use_case_notes': f"Batch query {len(query_data_list) + 1}",
                        'expected_performance': 'medium'
                    })

                # Start async batch analysis
                task = batch_analyze_queries.delay(query_data_list, request.user.id)
                request.session['batch_analysis_task_id'] = task.id

                messages.info(request, f"Batch analysis started for {len(query_list)} queries. Processing in background...")
                return redirect('async_processing_status')

            else:
                # Synchronous batch processing
                results = []
                for i, query in enumerate(query_list):
                    try:
                        query_obj, analysis = analyze_query(
                            sql_query=query,
                            user=request.user,
                            database_version=form.cleaned_data.get('database_version', ''),
                            use_case_notes=f"Batch query {i + 1}"
                        )
                        results.append({
                            'query': query,
                            'analysis': analysis,
                            'success': True
                        })
                    except Exception as e:
                        results.append({
                            'query': query,
                            'error': str(e),
                            'success': False
                        })

                return render(request, 'analyzer/batch_results.html', {
                    'results': results,
                    'total_queries': len(query_list)
                })
    else:
        form = BatchQueryForm()

    return render(request, 'analyzer/batch_analysis.html', {'form': form})


@login_required
def performance_report_view(request):
    """Generate and display performance report."""
    if request.method == 'POST':
        date_range_days = int(request.POST.get('date_range_days', 30))
        use_async = request.POST.get('use_async') == 'on'

        if use_async:
            task = generate_performance_report.delay(request.user.id, date_range_days)
            request.session['report_task_id'] = task.id

            messages.info(request, "Performance report generation started. This may take a few minutes...")
            return redirect('async_processing_status')
        else:
            # Synchronous report generation (for smaller datasets)
            from .tasks import generate_performance_report

            # Get the task function directly (not the Celery task)
            from datetime import timedelta
            from django.db.models import Avg, Count, Q
            from django.utils import timezone

            end_date = timezone.now()
            start_date = end_date - timedelta(days=date_range_days)

            analyses = QueryAnalysis.objects.filter(
                query__user=request.user,
                created_at__gte=start_date,
                created_at__lte=end_date
            )

            # Generate report data
            total_queries = analyses.count()
            avg_grade = analyses.aggregate(avg_grade=Avg('grade'))['avg_grade'] or 0

            grade_distribution = {}
            for grade in ['A', 'B', 'C', 'D', 'F']:
                count = analyses.filter(grade=grade).count()
                grade_distribution[grade] = {
                    'count': count,
                    'percentage': (count / total_queries * 100) if total_queries > 0 else 0
                }

            report_data = {
                'user_info': {
                    'username': request.user.username,
                    'report_period': f"{start_date.date()} to {end_date.date()}"
                },
                'summary': {
                    'total_queries_analyzed': total_queries,
                    'average_grade': round(avg_grade, 2),
                    'grade_distribution': grade_distribution
                }
            }

            return render(request, 'analyzer/performance_report.html', {
                'report': report_data,
                'date_range_days': date_range_days
            })

    return render(request, 'analyzer/performance_report.html', {
        'date_range_options': [7, 14, 30, 90, 365]
    })


@require_http_methods(["POST"])
@ratelimit(key='user', rate='10/m', method='POST', block=True)
def api_unified_query_analysis(request):
    """
    API endpoint for unified ML-enhanced query analysis.

    Accepts JSON payload with query and context information,
    returns comprehensive ML analysis results.
    """
    try:
        # Parse JSON request
        data = json.loads(request.body)
        sql_query = data.get('query', '').strip()

        if not sql_query:
            return JsonResponse({
                'success': False,
                'error': 'Query is required'
            }, status=400)

        # Extract additional parameters
        database_type = data.get('database_type', 'generic')
        database_version = data.get('database_version', '')
        context = data.get('context', {})
        user_id = str(request.user.id) if request.user.is_authenticated else 'anonymous'

        # Create analysis request
        analysis_request = AnalysisRequest(
            query=sql_query,
            user_id=user_id,
            database_type=database_type,
            database_version=database_version,
            context=context
        )

        # Run unified ML analysis
        unified_analyzer = UnifiedQueryAnalyzer()

        # Execute async analysis in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            ml_result = loop.run_until_complete(
                unified_analyzer.analyze_query(analysis_request)
            )
        finally:
            loop.close()

        # Traditional analysis for comparison
        traditional_analysis = None
        try:
            query_obj, traditional_analysis = analyze_query(sql_query, database_type)
        except Exception as e:
            logger.warning(f"Traditional analysis failed: {e}")

        # Prepare response
        response_data = {
            'success': True,
            'analysis_id': str(uuid.uuid4()),
            'query': sql_query,
            'ml_analysis': {
                'semantic_metrics': ml_result.semantic_metrics,
                'performance_prediction': ml_result.performance_prediction,
                'feedback': ml_result.feedback,
                'recommendations': ml_result.recommendations,
                'personalized_feedback': ml_result.personalized_feedback,
                'rewrite_suggestions': ml_result.rewrite_suggestions,
                'confidence_score': ml_result.confidence_score,
                'analysis_timestamp': ml_result.analysis_timestamp
            }
        }

        # Include traditional analysis if available
        if traditional_analysis:
            response_data['traditional_analysis'] = {
                'grade': traditional_analysis.grade,
                'score': traditional_analysis.score,
                'issues_found': traditional_analysis.issues_found,
                'recommendations': traditional_analysis.recommendations,
                'execution_time_ms': traditional_analysis.execution_time_ms
            }

        # Log API usage
        logger.info(f"API unified analysis completed for user {user_id}, query length: {len(sql_query)}")

        return JsonResponse(response_data)

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON payload'
        }, status=400)

    except Exception as e:
        logger.error(f"API unified analysis error: {e}")
        return JsonResponse({
            'success': False,
            'error': 'Internal server error during analysis'
        }, status=500)


def csrf_failure(request, reason=""):
    """
    Custom CSRF failure view for enhanced security logging and user experience.
    """
    logger.warning(
        f"CSRF failure: {reason} | "
        f"IP: {get_client_ip(request)} | "
        f"User: {getattr(request.user, 'username', 'anonymous')} | "
        f"Path: {request.path} | "
        f"Referer: {request.META.get('HTTP_REFERER', 'None')}"
    )

    # Check if this is an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'error': 'CSRF verification failed',
            'message': 'Security token expired. Please refresh the page and try again.',
            'reload_required': True
        }, status=403)

    # For regular requests, render a user-friendly error page
    return render(request, 'analyzer/csrf_failure.html', {
        'reason': reason,
    }, status=403)


def get_client_ip(request):
    """
    Get the client's IP address from the request.
    Helper function for security logging.
    """
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
