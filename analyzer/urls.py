from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze, name='analyze'),
    path('login/', views.login_view, name='login'),  # Add this line for login view
    path('logout/', views.logout_view, name='logout'),  # Add this line for logout view
    path('register/', views.register_view, name='register'),  # Add this line for register view
]
