"""CRUD views for ``UserDatabaseConnection`` rows.

Routes (mounted in ``analyzer/urls.py``):
    /connections/                 connections_list
    /connections/new/             connection_create
    /connections/<id>/edit/       connection_edit
    /connections/<id>/delete/     connection_delete
    /connections/<id>/test/       connection_test  (POST, JSON)

A successful test pings the database via ``DatabaseIntrospector.test_connection``
without persisting anything new — used as a sanity check before saving a
profile and before running the IndexRecommender pipeline.
"""

from __future__ import annotations

import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods, require_POST

from analyzer.forms import SavedConnectionForm
from analyzer.models import UserDatabaseConnection

logger = logging.getLogger(__name__)


@login_required
def connections_list(request):
    connections = UserDatabaseConnection.objects.filter(user=request.user)
    return render(
        request,
        "analyzer/connections/list.html",
        {"connections": connections},
    )


@login_required
@require_http_methods(["GET", "POST"])
def connection_create(request):
    if request.method == "POST":
        form = SavedConnectionForm(request.POST)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.user = request.user
            instance.save()
            messages.success(request, f"Saved connection '{instance.name}'.")
            return redirect("connections_list")
    else:
        form = SavedConnectionForm()
    return render(
        request,
        "analyzer/connections/form.html",
        {"form": form, "mode": "create"},
    )


@login_required
@require_http_methods(["GET", "POST"])
def connection_edit(request, connection_id: int):
    instance = get_object_or_404(
        UserDatabaseConnection, pk=connection_id, user=request.user
    )
    if request.method == "POST":
        form = SavedConnectionForm(request.POST, instance=instance)
        if form.is_valid():
            form.save()
            messages.success(request, f"Updated connection '{instance.name}'.")
            return redirect("connections_list")
    else:
        form = SavedConnectionForm(instance=instance)
    return render(
        request,
        "analyzer/connections/form.html",
        {"form": form, "mode": "edit", "instance": instance},
    )


@login_required
@require_http_methods(["GET", "POST"])
def connection_delete(request, connection_id: int):
    instance = get_object_or_404(
        UserDatabaseConnection, pk=connection_id, user=request.user
    )
    if request.method == "POST":
        name = instance.name
        instance.delete()
        messages.success(request, f"Deleted connection '{name}'.")
        return redirect("connections_list")
    return render(
        request,
        "analyzer/connections/confirm_delete.html",
        {"instance": instance},
    )


@login_required
@require_POST
def connection_test(request, connection_id: int):
    """Ping the saved DB and return JSON ``{ok, error?}``."""
    instance = get_object_or_404(
        UserDatabaseConnection, pk=connection_id, user=request.user
    )
    try:
        from analyzer.database_introspector import DatabaseIntrospector

        introspector = DatabaseIntrospector(instance.to_connection_config())
        if introspector.connect():
            instance.touch()
            return JsonResponse({"ok": True})
        return JsonResponse(
            {"ok": False, "error": "Could not establish a connection. Check logs."},
            status=400,
        )
    except Exception as exc:  # pragma: no cover - depends on user network
        logger.exception("Connection test failed for %s", instance.pk)
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
