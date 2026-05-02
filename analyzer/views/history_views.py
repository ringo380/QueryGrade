"""
User query history views.
"""

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import render

from ..models import UserQueryHistory
from .constants import DEFAULT_HISTORY_PAGE_SIZE


@login_required
def query_history(request):
    """
    Display the user's query history.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    history = UserQueryHistory.objects.filter(user=request.user).select_related(
        "query", "query__analysis"
    )

    # Paginate the results
    paginator = Paginator(history, DEFAULT_HISTORY_PAGE_SIZE)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(request, "analyzer/query_history.html", {"page_obj": page_obj})
