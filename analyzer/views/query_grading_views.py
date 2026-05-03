"""
SQL query grading and analysis views.

This module handles the core query grading functionality including:
- Single query grading
- Query results display
- ML-enhanced analysis
- Query comparison
- Batch analysis
"""

import asyncio
import logging

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render
from django_ratelimit.decorators import ratelimit

from ..db_versions import DATABASE_VERSIONS
from ..forms import BatchQueryForm, QueryCompareForm, QueryGradeForm
from ..ml.analysis.unified_analyzer import (AnalysisRequest,
                                            UnifiedQueryAnalyzer)
from ..models import Query, QueryAnalysis, UserQueryHistory
from ..performance import PerformanceMonitor
from ..query_analyzer import analyze_query
from ..query_optimizer import optimize_query_from_analysis
from .constants import (ANON_ANALYSIS_HISTORY_LIMIT, ANON_ANALYSIS_SESSION_KEY,
                        ANON_QUERY_RATE_LIMIT, ANON_TRIAL_COUNT_KEY,
                        GRADE_COLORS, QUERY_RATE_LIMIT)
from .utils import anon_trial_state, get_client_ip

logger = logging.getLogger(__name__)


def _anon_ip_key(group, request):
    """Rate-limit key: returns the client IP for anonymous users, None for authenticated (skip)."""
    if request.user.is_authenticated:
        return None
    return get_client_ip(request)


@transaction.non_atomic_requests
@ratelimit(key=_anon_ip_key, rate=ANON_QUERY_RATE_LIMIT, method="POST", block=True)
@ratelimit(key="user", rate=QUERY_RATE_LIMIT, method="POST", block=True)
@PerformanceMonitor.time_function("grade_query_view")
def grade_query(request):
    """
    Handles the SQL query grading interface.

    Anonymous visitors may grade up to ANON_TRIAL_CAP queries per session;
    authenticated users get full ML analysis, history, and feedback features.
    """
    is_anon = not request.user.is_authenticated
    cap, count, remaining = anon_trial_state(request)

    if request.method == "POST":
        if is_anon and remaining <= 0:
            return render(
                request,
                "analyzer/grade_form.html",
                {
                    "form": QueryGradeForm(),
                    "recent_queries": [],
                    "is_anonymous_trial": True,
                    "trial_exhausted": True,
                    "trial_cap": cap,
                    "trial_remaining": 0,
                    "db_versions": DATABASE_VERSIONS,
                    "gtag_event": "trial_exhausted",
                    "gtag_params": {"trial_cap": cap, "queries_used": count},
                },
            )

        form = QueryGradeForm(request.POST)
        if form.is_valid():
            sql_query = form.cleaned_data["sql_query"]
            database_type = form.cleaned_data.get("database_type", "")
            database_version = form.cleaned_data.get("database_version", "")
            use_case_notes = form.cleaned_data.get("use_case_notes", "")

            try:
                # Analyze the query (persists Query + QueryAnalysis with no user FK)
                query, analysis = analyze_query(sql_query, database_type)
                logger.info(
                    f"Query created: ID={query.id}, Analysis created: ID={analysis.id}"
                )

                if is_anon:
                    # Track which analyses this anon visitor may view
                    ids = list(request.session.get(ANON_ANALYSIS_SESSION_KEY, []))
                    ids.append(analysis.id)
                    request.session[ANON_ANALYSIS_SESSION_KEY] = ids[
                        -ANON_ANALYSIS_HISTORY_LIMIT:
                    ]
                    request.session[ANON_TRIAL_COUNT_KEY] = count + 1
                    request.session.modified = True
                    return redirect("grade_results", analysis_id=analysis.id)

                # Authenticated path: ML enhancement + history persistence
                try:
                    unified_analyzer = UnifiedQueryAnalyzer()
                    analysis_request = AnalysisRequest(
                        query=sql_query,
                        user_id=str(request.user.id),
                        database_type=database_type,
                        database_version=database_version,
                        context={
                            "use_case": use_case_notes,
                            "user_agent": request.META.get("HTTP_USER_AGENT", ""),
                            "ip_address": get_client_ip(request),
                        },
                    )

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        ml_analysis = loop.run_until_complete(
                            unified_analyzer.analyze_query(analysis_request)
                        )
                    finally:
                        loop.close()

                    request.session["ml_analysis"] = {
                        "semantic_metrics": ml_analysis.semantic_metrics,
                        "performance_prediction": ml_analysis.performance_prediction,
                        "feedback": ml_analysis.feedback,
                        "recommendations": ml_analysis.recommendations,
                        "personalized_feedback": ml_analysis.personalized_feedback,
                        "rewrite_suggestions": ml_analysis.rewrite_suggestions,
                    }

                    logger.info(
                        f"Enhanced ML analysis completed for user {request.user.username}"
                    )

                except Exception as ml_error:
                    logger.warning(
                        f"ML analysis failed for user {request.user.username}: {ml_error}"
                    )

                logger.info(
                    f"About to create UserQueryHistory with query.id={query.id}, user={request.user}"
                )
                user_history = UserQueryHistory.objects.create(
                    user=request.user,
                    query=query,
                    ip_address=get_client_ip(request),
                    user_agent=request.META.get("HTTP_USER_AGENT", "")[:255],
                    database_type=database_type,
                    database_version=database_version,
                    use_case_notes=use_case_notes,
                )
                logger.info(f"UserQueryHistory created: ID={user_history.id}")

                return redirect("enhanced_grade_results", analysis_id=analysis.id)

            except ValueError as e:
                error_msg = str(e)
                if "typos in keywords" in error_msg:
                    messages.error(
                        request,
                        "SQL syntax error: Your query contains apparent typos in SQL keywords. Please check your spelling.",
                    )
                elif "Unable to parse" in error_msg:
                    messages.error(
                        request,
                        "SQL parsing error: We couldn't parse your SQL query. Please check the syntax and try again.",
                    )
                elif "No SQL keywords found" in error_msg:
                    messages.error(
                        request,
                        "Invalid input: No SQL keywords detected. Please enter a valid SQL query.",
                    )
                else:
                    messages.error(request, f"SQL error: {error_msg}")
                who = "anonymous" if is_anon else request.user.username
                logger.warning(f"SQL syntax error for user {who}: {e}")
                return render(
                    request,
                    "analyzer/grade_form.html",
                    {
                        "form": form,
                        "recent_queries": [],
                        "is_anonymous_trial": is_anon,
                        "trial_exhausted": is_anon and remaining <= 0,
                        "trial_cap": cap,
                        "trial_remaining": remaining,
                        "db_versions": DATABASE_VERSIONS,
                    },
                )
            except Exception as e:
                who = "anonymous" if is_anon else request.user.username
                logger.error(f"Unexpected error analyzing query for user {who}: {e}")
                messages.error(request, f"An unexpected error occurred: {str(e)}")
                return render(
                    request,
                    "analyzer/grade_form.html",
                    {
                        "form": form,
                        "recent_queries": [],
                        "is_anonymous_trial": is_anon,
                        "trial_exhausted": is_anon and remaining <= 0,
                        "trial_cap": cap,
                        "trial_remaining": remaining,
                        "db_versions": DATABASE_VERSIONS,
                    },
                )
        else:
            messages.error(request, "Please correct the errors in the form below.")
    else:
        form = QueryGradeForm()

    recent_queries = (
        (
            UserQueryHistory.objects.filter(user=request.user)
            .select_related("query")
            .order_by("-submitted_at")[:10]
        )
        if request.user.is_authenticated
        else []
    )

    return render(
        request,
        "analyzer/grade_form.html",
        {
            "form": form,
            "recent_queries": recent_queries,
            "is_anonymous_trial": is_anon,
            "trial_exhausted": is_anon and remaining <= 0,
            "trial_cap": cap,
            "trial_remaining": remaining,
            "db_versions": DATABASE_VERSIONS,
        },
    )


def grade_results(request, analysis_id):
    """
    Display the grading results for a query analysis.

    Authenticated users access analyses they own (via UserQueryHistory).
    Anonymous users access analyses created in their session (tracked in
    session[ANON_ANALYSIS_SESSION_KEY]).
    """
    try:
        analysis = get_object_or_404(QueryAnalysis, id=analysis_id)
    except Exception:
        messages.error(request, "The requested analysis could not be found.")
        return redirect("grade_query")

    is_anon = not request.user.is_authenticated
    user_history = None
    database_type = ""

    if is_anon:
        allowed_ids = request.session.get(ANON_ANALYSIS_SESSION_KEY, [])
        if analysis_id not in allowed_ids:
            logger.warning(
                f"Anonymous visitor attempted to access analysis {analysis_id} without session grant"
            )
            messages.error(request, "Please register or sign in to view this analysis.")
            return redirect("index")
    else:
        user_history = (
            UserQueryHistory.objects.filter(user=request.user, query=analysis.query)
            .order_by("-submitted_at")
            .first()
        )
        if user_history is None:
            logger.warning(
                f"User {request.user.username} attempted to access analysis {analysis_id} without permission"
            )
            messages.error(request, "You don't have permission to view this analysis.")
            return redirect("grade_query")
        database_type = user_history.database_type or ""

    # Generate optimization suggestions if there are issues
    optimization_result = None
    if analysis.issues_found and len(analysis.issues_found) > 0:
        try:
            optimization_result = optimize_query_from_analysis(
                analysis.query.sql_text, analysis.issues_found, database_type
            )
        except Exception as e:
            logger.warning(f"Failed to generate optimization suggestions: {e}")

    # Only show the inline upgrade CTA when the anon user is close to their cap
    # (last 10 grades / 80% used). Earlier on, the navbar credits pill is enough.
    show_upgrade_cta = False
    if is_anon:
        _, _, _remaining = anon_trial_state(request)
        show_upgrade_cta = _remaining <= 10

    context = {
        "analysis": analysis,
        "query": analysis.query,
        "user_history": user_history,
        "optimization_result": optimization_result,
        "grade_colors": GRADE_COLORS,
        "is_anonymous": is_anon,
        "show_upgrade_cta": show_upgrade_cta,
    }

    return render(request, "analyzer/grade_results.html", context)


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
    except Exception:
        messages.error(request, "The requested analysis could not be found.")
        return redirect("grade_query")

    # Check if the current user has access to this analysis
    user_history = (
        UserQueryHistory.objects.filter(user=request.user, query=analysis.query)
        .order_by("-submitted_at")
        .first()
    )
    if user_history is None:
        logger.warning(
            f"User {request.user.username} attempted to access analysis {analysis_id} without permission"
        )
        messages.error(request, "You don't have permission to view this analysis.")
        return redirect("grade_query")

    # Get ML analysis from session
    ml_analysis = request.session.get("ml_analysis", {})

    # Generate optimization suggestions if there are issues
    optimization_result = None
    if analysis.issues_found and len(analysis.issues_found) > 0:
        try:
            database_type = (
                user_history.database_type if user_history.database_type else ""
            )
            optimization_result = optimize_query_from_analysis(
                analysis.query.sql_text, analysis.issues_found, database_type
            )
        except Exception as e:
            logger.warning(f"Failed to generate optimization suggestions: {e}")

    # Process ML analysis data for template
    processed_ml_analysis = {}
    if ml_analysis:
        try:
            processed_ml_analysis = {
                "has_ml_analysis": True,
                "semantic_score": ml_analysis.get("semantic_metrics", {}).get(
                    "overall_score", 0
                ),
                "complexity_level": ml_analysis.get("semantic_metrics", {}).get(
                    "complexity_level", "Unknown"
                ),
                "query_intent": ml_analysis.get("semantic_metrics", {}).get(
                    "query_intent", "Unknown"
                ),
                "performance_prediction": ml_analysis.get("performance_prediction", {}),
                "feedback": ml_analysis.get("feedback", {}),
                "recommendations": ml_analysis.get("recommendations", []),
                "personalized_feedback": ml_analysis.get("personalized_feedback", {}),
                "rewrite_suggestions": ml_analysis.get("rewrite_suggestions", []),
            }
        except Exception as e:
            logger.warning(f"Error processing ML analysis data: {e}")
            processed_ml_analysis = {"has_ml_analysis": False}
    else:
        processed_ml_analysis = {"has_ml_analysis": False}

    context = {
        "analysis": analysis,
        "query": analysis.query,
        "user_history": user_history,
        "optimization_result": optimization_result,
        "ml_analysis": processed_ml_analysis,
        "grade_colors": GRADE_COLORS,
    }

    return render(request, "analyzer/enhanced_grade_results.html", context)


@login_required
def compare_queries(request):
    """
    Handles the SQL query comparison interface.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == "POST":
        form = QueryCompareForm(request.POST)
        if form.is_valid():
            # Store the comparison data in session for results page
            comparison_data = {
                "query_1": form.cleaned_data["query_1"],
                "query_1_name": form.cleaned_data.get("query_1_name", "Query 1"),
                "query_2": form.cleaned_data["query_2"],
                "query_2_name": form.cleaned_data.get("query_2_name", "Query 2"),
                "query_3": form.cleaned_data.get("query_3"),
                "query_3_name": form.cleaned_data.get("query_3_name", "Query 3"),
                "database_type": form.cleaned_data.get("database_type", ""),
                "comparison_notes": form.cleaned_data.get("comparison_notes", ""),
            }

            # Store in session
            request.session["comparison_data"] = comparison_data

            return redirect("compare_results")
        else:
            messages.error(request, "Please correct the errors in the form below.")
    else:
        form = QueryCompareForm()

    return render(request, "analyzer/query_compare.html", {"form": form})


@login_required
def batch_grade_queries(request):
    """
    Handles batch query grading interface.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == "POST":
        form = BatchQueryForm(request.POST)
        if form.is_valid():
            # Process batch queries
            queries_text = form.cleaned_data["queries"]
            database_type = form.cleaned_data.get("database_type", "")

            # Split queries by delimiter (assuming newline or semicolon)
            queries = [q.strip() for q in queries_text.split(";") if q.strip()]

            # Store in session for processing
            request.session["batch_queries"] = {
                "queries": queries,
                "database_type": database_type,
            }

            return redirect("batch_results")
        else:
            messages.error(request, "Please correct the errors in the form below.")
    else:
        form = BatchQueryForm()

    return render(request, "analyzer/batch_grade_form.html", {"form": form})


@login_required
def batch_results(request):
    """
    Display batch query grading results.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    batch_data = request.session.get("batch_queries")
    if not batch_data:
        messages.error(
            request, "No batch queries found. Please submit queries for analysis."
        )
        return redirect("batch_grade_queries")

    results = []
    for idx, query_sql in enumerate(batch_data["queries"], 1):
        try:
            query, analysis = analyze_query(query_sql, batch_data["database_type"])
            results.append(
                {"index": idx, "query": query, "analysis": analysis, "error": None}
            )
        except Exception as e:
            logger.error(f"Failed to analyze batch query {idx}: {e}")
            results.append(
                {"index": idx, "query": None, "analysis": None, "error": str(e)}
            )

    context = {
        "results": results,
        "batch_data": batch_data,
        "grade_colors": GRADE_COLORS,
    }

    return render(request, "analyzer/batch_results.html", context)
