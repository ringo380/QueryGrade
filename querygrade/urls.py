from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("analyzer.urls")),
    path("api/", include("analyzer.api_urls")),
    # Custom authentication URLs are in analyzer.urls, not using Django's built-in auth URLs
]

# Custom error handlers
handler404 = "analyzer.views.error_views.custom_404"
handler500 = "analyzer.views.error_views.custom_500"
