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
    # Retention. Django never deletes expired session rows on its own, so
    # django_session grows without bound - it was 4,905 rows / 1.8 MB of a
    # 12 MB database with zero registered users. Daily at 04:10 UTC, off the
    # :00/:15 marks so it never contends with monitor-ml-models.
    #
    # NOTE: nothing in this schedule is running right now. The worker and beat
    # services are stopped for cost while QueryGrade has no users (issue #133),
    # so these three entries describe what beat WOULD do, not what is
    # happening. The session purge is the one piece that still runs, from
    # SessionPurgeMiddleware on the web service. The other two do not.
    # Restore both services before QueryGrade takes real traffic.
    "purge-expired-sessions": {
        "task": "analyzer.tasks.purge_expired_sessions",
        "schedule": crontab(hour=4, minute=10),
    },
    # cleanup_temp_files has always documented itself as "should be run via
    # celery beat every hour" but was never actually scheduled.
    "cleanup-temp-files": {
        "task": "analyzer.tasks.cleanup_temp_files",
        "schedule": crontab(minute=40),
    },
}


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
