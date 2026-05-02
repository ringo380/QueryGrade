from django.conf import settings


def ga4_settings(request):
    return {
        "GA4_MEASUREMENT_ID": getattr(settings, "GA4_MEASUREMENT_ID", ""),
    }
