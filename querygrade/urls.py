from django.contrib import admin
from django.contrib.sitemaps.views import sitemap
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.urls import include, path

from analyzer import views
from analyzer.sitemaps import sitemaps


def robots_txt(request):
    content = render_to_string(
        "robots.txt",
        {"scheme": request.scheme, "host": request.get_host()},
    )
    return HttpResponse(content, content_type="text/plain")


urlpatterns = [
    path("admin/", admin.site.urls),
    path(
        "sitemap.xml",
        sitemap,
        {"sitemaps": sitemaps},
        name="django.contrib.sitemaps.views.sitemap",
    ),
    path("robots.txt", robots_txt, name="robots_txt"),
    path("", include("analyzer.urls")),
    path("api/", include("analyzer.api_urls")),
    # Custom authentication URLs are in analyzer.urls, not using Django's built-in auth URLs
]

# Custom error handlers
handler404 = "analyzer.views.error_views.custom_404"
handler500 = "analyzer.views.error_views.custom_500"
