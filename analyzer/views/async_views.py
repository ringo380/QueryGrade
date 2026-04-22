"""
Asynchronous processing and task status views.

This module handles:
- Async processing status display
- Task progress checking
- Async results retrieval
- Batch analysis job management
- Performance report generation
"""

import asyncio
import json
import logging
from datetime import timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.cache import caches
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST
from django_ratelimit.decorators import ratelimit

from ..forms import BatchQueryForm
from ..ml.analysis.unified_analyzer import AnalysisRequest, UnifiedQueryAnalyzer
from ..models import QueryAnalysis
from ..query_analyzer import analyze_query

logger = logging.getLogger(__name__)


@login_required
def async_processing_status(request):
    """View to show async processing status."""
    task_id = (
        request.session.get("log_processing_task_id")
        or request.session.get("batch_analysis_task_id")
        or request.session.get("report_task_id")
    )

    if not task_id:
        messages.error(request, "No processing task found.")
        return redirect("index")

    context = {"task_id": task_id, "page_title": "Processing Status"}

    return render(request, "analyzer/async_status.html", context)


@login_required
@require_http_methods(["GET"])
def check_task_status(request, task_id):
    """Check the status of an async task."""
    from celery.result import AsyncResult

    try:
        task = AsyncResult(task_id)

        response_data = {
            "task_id": task_id,
            "status": task.status,
            "ready": task.ready(),
        }

        if task.ready():
            if task.successful():
                result = task.result
                response_data.update({"success": True, "result": result})

                # Clear task ID from session
                session_keys = [
                    "log_processing_task_id",
                    "batch_analysis_task_id",
                    "report_task_id",
                ]
                for key in session_keys:
                    if request.session.get(key) == task_id:
                        del request.session[key]

            else:
                response_data.update({"success": False, "error": str(task.info)})
        else:
            response_data["progress"] = (
                getattr(task.info, "current", 0) if hasattr(task.info, "current") else 0
            )

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error checking task status: {str(e)}")
        return JsonResponse(
            {
                "task_id": task_id,
                "status": "ERROR",
                "success": False,
                "error": "Failed to check task status",
            }
        )


@login_required
def async_results(request):
    """Display results from async processing."""
    cache_key = request.GET.get("cache_key")

    if not cache_key:
        messages.error(request, "No results found.")
        return redirect("index")

    cache = caches["process_cache"]
    results = cache.get(cache_key)

    if not results:
        messages.error(request, "Results have expired or are not available.")
        return redirect("index")

    # Handle different types of results
    if "anomalies" in results:
        # Log file processing results
        anomalies = results.get("anomalies", [])

        # Paginate the results
        paginator = Paginator(anomalies, 10)
        page_number = request.GET.get("page")
        page_obj = paginator.get_page(page_number)

        context = {
            "page_obj": page_obj,
            "processing_time": results.get("processing_time", 0),
            "total_queries": len(anomalies),
            "anomaly_count": len([a for a in anomalies if a.get("is_anomaly", False)]),
        }

        return render(request, "analyzer/results.html", context)

    # Handle other result types (batch analysis, schema analysis, etc.)
    return render(request, "analyzer/async_results.html", {"results": results})


@login_required
def batch_analysis_view(request):
    """Handle batch query analysis with async processing."""
    if request.method == "POST":
        form = BatchQueryForm(request.POST)
        if form.is_valid():
            queries_text = form.cleaned_data["queries"]
            use_async = form.cleaned_data.get("use_async", False)

            # Parse queries (split by semicolon and clean)
            query_list = [q.strip() for q in queries_text.split(";") if q.strip()]

            if not query_list:
                messages.error(request, "No valid queries found.")
                return redirect("batch_analysis")

            if use_async:
                from ..tasks import batch_analyze_queries

                # Prepare query data for async processing
                query_data_list = []
                for query in query_list:
                    query_data_list.append(
                        {
                            "sql_query": query,
                            "database_version": form.cleaned_data.get(
                                "database_version", ""
                            ),
                            "use_case_notes": f"Batch query {len(query_data_list) + 1}",
                            "expected_performance": "medium",
                        }
                    )

                # Start async batch analysis
                task = batch_analyze_queries.delay(query_data_list, request.user.id)
                request.session["batch_analysis_task_id"] = task.id

                messages.info(
                    request,
                    f"Batch analysis started for {len(query_list)} queries. Processing in background...",
                )
                return redirect("async_processing_status")

            else:
                # Synchronous batch processing
                results = []
                for i, query in enumerate(query_list):
                    try:
                        query_obj, analysis = analyze_query(
                            sql_query=query,
                            database_version=form.cleaned_data.get(
                                "database_version", ""
                            ),
                        )
                        results.append(
                            {"query": query, "analysis": analysis, "success": True}
                        )
                    except Exception as e:
                        results.append(
                            {"query": query, "error": str(e), "success": False}
                        )

                return render(
                    request,
                    "analyzer/batch_results.html",
                    {"results": results, "total_queries": len(query_list)},
                )
    else:
        form = BatchQueryForm()

    return render(request, "analyzer/batch_analysis.html", {"form": form})


@login_required
def performance_report_view(request):
    """Generate and display performance report."""
    if request.method == "POST":
        date_range_days = int(request.POST.get("date_range_days", 30))
        use_async = request.POST.get("use_async") == "on"

        if use_async:
            from ..tasks import generate_performance_report

            task = generate_performance_report.delay(request.user.id, date_range_days)
            request.session["report_task_id"] = task.id

            messages.info(
                request,
                "Performance report generation started. This may take a few minutes...",
            )
            return redirect("async_processing_status")
        else:
            # Synchronous report generation (for smaller datasets)
            end_date = timezone.now()
            start_date = end_date - timedelta(days=date_range_days)

            analyses = QueryAnalysis.objects.filter(
                query__userqueryhistory__user=request.user,
                created_at__gte=start_date,
                created_at__lte=end_date,
            )

            # Generate report data
            total_queries = analyses.count()

            # Calculate average grade (convert letter grades to numeric values)
            grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
            numeric_grades = [grade_values.get(a.grade, 0) for a in analyses]
            avg_grade = (
                sum(numeric_grades) / len(numeric_grades) if numeric_grades else 0
            )

            grade_distribution = {}
            for grade in ["A", "B", "C", "D", "F"]:
                count = analyses.filter(grade=grade).count()
                grade_distribution[grade] = {
                    "count": count,
                    "percentage": (
                        (count / total_queries * 100) if total_queries > 0 else 0
                    ),
                }

            report_data = {
                "user_info": {
                    "username": request.user.username,
                    "report_period": f"{start_date.date()} to {end_date.date()}",
                },
                "summary": {
                    "total_queries_analyzed": total_queries,
                    "average_grade": round(avg_grade, 2),
                    "grade_distribution": grade_distribution,
                },
            }

            return render(
                request,
                "analyzer/performance_report.html",
                {"report": report_data, "date_range_days": date_range_days},
            )

    return render(
        request,
        "analyzer/performance_report.html",
        {"date_range_options": [7, 14, 30, 90, 365]},
    )


@require_POST
@ratelimit(key="user", rate="10/m", method="POST", block=True)
def api_unified_query_analysis(request):
    """
    API endpoint for unified ML-enhanced query analysis.

    Accepts JSON payload with query and context information,
    returns comprehensive ML analysis results.
    """
    try:
        import uuid

        # Parse JSON request
        data = json.loads(request.body)
        sql_query = data.get("query", "").strip()

        if not sql_query:
            return JsonResponse(
                {"success": False, "error": "Query is required"}, status=400
            )

        # Extract additional parameters
        database_type = data.get("database_type", "generic")
        database_version = data.get("database_version", "")
        context = data.get("context", {})
        user_id = str(request.user.id) if request.user.is_authenticated else "anonymous"

        # Create analysis request
        analysis_request = AnalysisRequest(
            query=sql_query,
            user_id=user_id,
            database_type=database_type,
            database_version=database_version,
            context=context,
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
            "success": True,
            "analysis_id": str(uuid.uuid4()),
            "query": sql_query,
            "ml_analysis": {
                "semantic_metrics": ml_result.semantic_metrics,
                "performance_prediction": ml_result.performance_prediction,
                "feedback": ml_result.feedback,
                "recommendations": ml_result.recommendations,
                "personalized_feedback": ml_result.personalized_feedback,
                "rewrite_suggestions": ml_result.rewrite_suggestions,
                "confidence_score": ml_result.confidence_score,
                "analysis_timestamp": ml_result.analysis_timestamp,
            },
        }

        # Include traditional analysis if available
        if traditional_analysis:
            response_data["traditional_analysis"] = {
                "grade": traditional_analysis.grade,
                "score": traditional_analysis.score,
                "issues_found": traditional_analysis.issues_found,
                "recommendations": traditional_analysis.recommendations,
                "execution_time_ms": traditional_analysis.execution_time_ms,
            }

        # Log API usage
        logger.info(
            f"API unified analysis completed for user {user_id}, query length: {len(sql_query)}"
        )

        return JsonResponse(response_data)

    except json.JSONDecodeError:
        return JsonResponse(
            {"success": False, "error": "Invalid JSON payload"}, status=400
        )

    except Exception as e:
        logger.error(f"API unified analysis error: {e}")
        return JsonResponse(
            {"success": False, "error": "Internal server error during analysis"},
            status=500,
        )
