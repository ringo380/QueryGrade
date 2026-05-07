"""
Log file upload and async processing views.
"""

import logging
import os
import tempfile
import uuid

import pandas as pd
from celery.result import AsyncResult
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.cache import caches
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from django_ratelimit.decorators import ratelimit

from ..forms import QueryGradeForm, UploadLogForm
from ..parser import process_general_log, process_slow_log
from ..tasks import process_log_file_async
from .utils import anon_trial_state

logger = logging.getLogger(__name__)


def analyze(request):
    """
    Handles the analyze view.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    return render(request, "analyzer/index.html")


@ratelimit(key="ip", rate="5/m", method="POST", block=True)
@ratelimit(key="user", rate="10/m", method="POST", block=True)
def index(request):
    """
    Handles the upload and processing of MySQL log files.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if not request.user.is_authenticated:
        # Anonymous landing: show inline trial grade form + upgrade CTAs
        if request.method == "POST":
            return redirect("login")
        cap, _count, remaining = anon_trial_state(request)
        return render(
            request,
            "analyzer/index.html",
            {
                "grade_form": QueryGradeForm(),
                "is_anonymous_trial": True,
                "trial_cap": cap,
                "trial_remaining": remaining,
                "trial_exhausted": remaining <= 0,
            },
        )

    if request.method == "POST":
        form = UploadLogForm(request.POST, request.FILES)
        if form.is_valid():
            log_type = form.cleaned_data["log_type"]
            log_file = request.FILES["log_file"]

            # Check if async processing is requested
            use_async = form.cleaned_data.get("use_async", False)

            if use_async:
                # Save file to temp directory for async processing with enhanced security
                try:
                    # Create secure temp file with restrictive permissions
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".log",
                        prefix="secure_upload_",
                        dir=tempfile.gettempdir(),
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
                        logger.error(
                            f"File size mismatch during upload for user {request.user.username}"
                        )
                        messages.error(
                            request,
                            "File upload verification failed. Please try again.",
                        )
                        return HttpResponseRedirect(reverse("index"))

                    # Start async task
                    task = process_log_file_async.delay(
                        temp_file.name, log_type, request.user.id
                    )

                    # Store task ID in session for progress tracking
                    request.session["log_processing_task_id"] = task.id

                    messages.info(
                        request,
                        f"File upload successful! Your {log_type} log is being processed in the background. You'll be notified when it's complete.",
                    )
                    request.session["_pending_gtag_event"] = "log_file_uploaded"
                    request.session["_pending_gtag_params"] = {
                        "log_type": log_type,
                        "processing_mode": "async",
                    }
                    return redirect("async_processing_status")

                except (OSError, IOError) as e:
                    logger.error(
                        f"Secure file upload failed for user {request.user.username}: {e}"
                    )
                    messages.error(
                        request,
                        "File upload failed due to security restrictions. Please try again.",
                    )
                    return HttpResponseRedirect(reverse("index"))
            else:
                # Synchronous processing with enhanced security
                try:
                    # Create secure storage with custom settings
                    fs = FileSystemStorage()

                    # Generate secure filename to prevent path traversal
                    secure_filename = f"secure_{uuid.uuid4().hex}_{log_file.name}"
                    secure_filename = (
                        secure_filename.replace("..", "")
                        .replace("/", "")
                        .replace("\\", "")
                    )

                    # Save file with security checks
                    filename = fs.save(secure_filename, log_file)
                    uploaded_file_url = fs.path(filename)

                    # Set restrictive file permissions
                    os.chmod(uploaded_file_url, 0o600)

                except (OSError, IOError) as e:
                    logger.error(
                        f"Secure file storage failed for user {request.user.username}: {e}"
                    )
                    messages.error(
                        request,
                        "File storage failed due to security restrictions. Please try again.",
                    )
                    return HttpResponseRedirect(reverse("index"))

                try:
                    # Analyze the log file
                    if log_type == "slow":
                        df_anomalies = process_slow_log(uploaded_file_url)
                    elif log_type == "general":
                        df_anomalies = process_general_log(uploaded_file_url)

                    # Prepare data for template
                    anomalies = (
                        df_anomalies.to_dict("records")
                        if df_anomalies is not None
                        else []
                    )

                    # Check if any anomalies were found
                    if not anomalies:
                        messages.info(
                            request,
                            "No anomalies detected in your log file. This suggests your queries are performing well!",
                        )

                    # Delete the uploaded file after processing
                    fs.delete(filename)

                    # Paginate the results
                    paginator = Paginator(anomalies, 10)  # Show 10 anomalies per page
                    page_number = request.GET.get("page")
                    page_obj = paginator.get_page(page_number)

                    request.session["_pending_gtag_event"] = "log_file_uploaded"
                    request.session["_pending_gtag_params"] = {
                        "log_type": log_type,
                        "processing_mode": "sync",
                    }
                    return render(
                        request, "analyzer/results.html", {"page_obj": page_obj}
                    )
                except FileNotFoundError:
                    logger.error(f"Log file not found: {uploaded_file_url}")
                    messages.error(
                        request,
                        "The uploaded file could not be found. Please try uploading again.",
                    )
                    _cleanup_file(fs, filename)
                    return HttpResponseRedirect(reverse("index"))
                except pd.errors.EmptyDataError:
                    logger.warning(
                        f"Empty log file uploaded by user {request.user.username}"
                    )
                    messages.warning(
                        request,
                        "The uploaded log file appears to be empty. Please upload a valid MySQL log file.",
                    )
                    _cleanup_file(fs, filename)
                    return HttpResponseRedirect(reverse("index"))
                except pd.errors.ParserError as e:
                    logger.warning(
                        f"Log file parsing error for user {request.user.username}: {e}"
                    )
                    messages.error(
                        request,
                        "Unable to parse the log file format. Please ensure you're uploading a valid MySQL log file.",
                    )
                    _cleanup_file(fs, filename)
                    return HttpResponseRedirect(reverse("index"))
                except Exception as e:
                    logger.error(
                        f"Unexpected error processing log file for user {request.user.username}: {e}"
                    )
                    messages.error(
                        request,
                        "An unexpected error occurred while processing the log file. Please check the file format and try again.",
                    )
                    _cleanup_file(fs, filename)
                    return HttpResponseRedirect(reverse("index"))
        else:
            messages.error(
                request,
                "There was an error with the form submission. Please check the fields and try again.",
            )
    else:
        form = UploadLogForm()
    return render(request, "analyzer/index.html", {"form": form})


@login_required
def async_processing_status(request):
    """View to show async processing status."""
    task_id = request.session.get("log_processing_task_id")

    if not task_id:
        messages.error(request, "No processing task found.")
        return redirect("index")

    context = {"task_id": task_id, "page_title": "Processing Status"}

    return render(request, "analyzer/async_status.html", context)


@login_required
@require_http_methods(["GET"])
def check_task_status(request, task_id):
    """Check the status of an async task."""
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
                if request.session.get("log_processing_task_id") == task_id:
                    del request.session["log_processing_task_id"]

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
    else:
        messages.error(request, "Unknown result format.")
        return redirect("index")


def _cleanup_file(fs, filename):
    """Helper function to clean up uploaded files."""
    try:
        fs.delete(filename)
    except FileNotFoundError:
        logger.debug(f"File {filename} already deleted or not found")
    except Exception as e:
        logger.warning(f"Failed to delete file {filename}: {e}")
