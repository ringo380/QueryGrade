from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from .forms import UploadLogForm
from .parser import parse_mysql_slow_log, parse_mysql_general_log  # Import your main functions


def index(request):
    if request.method == 'POST':
        form = UploadLogForm(request.POST, request.FILES)
        if form.is_valid():
            log_type = form.cleaned_data['log_type']
            log_file = request.FILES['log_file']

            # Save the uploaded file
            fs = FileSystemStorage()
            filename = fs.save(log_file.name, log_file)
            uploaded_file_url = fs.path(filename)

            # Initialize df_anomalies
            df_anomalies = None

            # Analyze the log file
            if log_type == 'slow':
                df, df_anomalies = parse_mysql_slow_log(uploaded_file_url)
            elif log_type == 'general':
                df, df_anomalies = parse_mysql_general_log(uploaded_file_url)

            # Prepare data for template
            anomalies = df_anomalies.to_dict('records') if df_anomalies is not None else []

            # Delete the uploaded file after processing
            fs.delete(filename)

            return render(request, 'analyzer/results.html', {'anomalies': anomalies})
    else:
        form = UploadLogForm()
    return render(request, 'analyzer/index.html', {'form': form})