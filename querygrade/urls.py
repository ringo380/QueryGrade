from django.contrib import admin
from django.urls import path, include
from analyzer import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('analyzer.urls')),
    path('api/', include('analyzer.api_urls')),
    path('accounts/', include('django.contrib.auth.urls')),  # Add this line for authentication URLs
]
