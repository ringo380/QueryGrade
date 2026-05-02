from django.conf import settings


def ga4_settings(request):
    """Expose GA4 config + one-shot session-based event flag to templates.

    Views set ``request.session['_pending_gtag_event']`` (and optionally
    ``_pending_gtag_params``) after a successful action; this processor
    pops them so base.html fires the gtag event exactly once on the next
    rendered page.
    """
    ctx = {
        "GA4_MEASUREMENT_ID": getattr(settings, "GA4_MEASUREMENT_ID", ""),
    }

    session = getattr(request, "session", None)
    if session is not None:
        event = session.pop("_pending_gtag_event", None)
        params = session.pop("_pending_gtag_params", None)
        if event:
            ctx["gtag_event"] = event
            if params is not None:
                ctx["gtag_params"] = params

    return ctx


def anon_trial(request):
    """Expose anonymous trial state (cap/count/remaining) for the navbar credits indicator.

    Returns empty dict for authenticated users so the indicator renders only for anon visitors.
    """
    user = getattr(request, "user", None)
    if user is not None and user.is_authenticated:
        return {}
    if getattr(request, "session", None) is None:
        return {}

    from analyzer.views.utils import anon_trial_state

    try:
        cap, count, remaining = anon_trial_state(request)
    except Exception:
        return {}
    return {
        "anon_trial_cap": cap,
        "anon_trial_count": count,
        "anon_trial_remaining": remaining,
    }
