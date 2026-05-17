import os

from celery import Celery
from celery.schedules import crontab

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "querygrade.settings")

app = Celery("querygrade")

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Load task modules from all registered Django apps.
app.autodiscover_tasks()


# Beat schedule. Picked up only by the `celery beat` process — the worker
# process ignores this. Beat runs as its own Railway service; see
# Dockerfile.beat. Granularity is 15 min for ML monitoring (easily meets
# the issue #5 SLA of "drift detected within 24h" and "alert response <5
# min"; the alert is delivered synchronously on detection).
app.conf.beat_schedule = {
    "monitor-ml-models": {
        "task": "analyzer.tasks.monitor_ml_models",
        "schedule": crontab(minute="*/15"),
    },
}


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
