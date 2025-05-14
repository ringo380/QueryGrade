from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages
from django.core.paginator import Paginator
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .forms import UploadLogForm
from .parser import process_slow_log, process_general_log  # Import the updated functions
import logging

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

            # Save the uploaded file
            fs = FileSystemStorage()
            filename = fs.save(log_file.name, log_file)
            uploaded_file_url = fs.path(filename)

            try:
                # Analyze the log file
                if log_type == 'slow':
                    df_anomalies = process_slow_log(uploaded_file_url)
                elif log_type == 'general':
                    df_anomalies = process_general_log(uploaded_file_url)

                # Prepare data for template
                anomalies = df_anomalies.to_dict('records') if df_anomalies is not None else []

                # Delete the uploaded file after processing
                fs.delete(filename)

                # Paginate the results
                paginator = Paginator(anomalies, 10)  # Show 10 anomalies per page
                page_number = request.GET.get('page')
                page_obj = paginator.get_page(page_number)

                return render(request, 'analyzer/results.html', {'page_obj': page_obj})
            except Exception as e:
                logger.error(f"Error processing log file: {e}")
                messages.error(request, "An error occurred while processing the log file. Please check the file format and try again.")
                return HttpResponseRedirect(reverse('index'))
        else:
            messages.error(request, "There was an error with the form submission. Please check the fields and try again.")
    else:
        form = UploadLogForm()
    return render(request, 'analyzer/index.html', {'form': form})

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
