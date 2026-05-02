"""
Server-side Google Analytics 4 Measurement Protocol client.

Used for events that originate from Celery tasks, scheduled jobs, or other
server contexts where no browser is rendering a page (and gtag.js can't fire).

Configuration:
- ``GA4_MEASUREMENT_ID``: shared with the gtag config (e.g. ``G-XXXX``)
- ``GA4_API_SECRET``: a Measurement Protocol API secret created in
  GA4 Admin → Data Streams → <stream> → Measurement Protocol API secrets

If either is missing, ``send_ga4_event`` becomes a silent no-op so the
codebase stays runnable in dev/test environments without analytics setup.

Failures (network, 4xx, 5xx) are logged at WARNING and swallowed — analytics
must never break business logic.
"""

import logging
from typing import Any, Dict, Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

_MP_ENDPOINT = "https://www.google-analytics.com/mp/collect"
_TIMEOUT_SECONDS = 2.0


def synthetic_client_id(user_id: int | str) -> str:
    """Stable pseudo client_id for server-only events.

    GA4's Measurement Protocol requires a ``client_id`` even when there is
    no browser. Use a deterministic, non-PII string keyed on the user so
    repeated server events from the same user are grouped consistently.
    """
    return f"server.{user_id}"


def send_ga4_event(
    client_id: str,
    event_name: str,
    params: Optional[Dict[str, Any]] = None,
    user_id: Optional[int | str] = None,
    *,
    timeout: float = _TIMEOUT_SECONDS,
) -> bool:
    """Fire one GA4 event via Measurement Protocol.

    Returns True on a 2xx response, False on any failure (including a missing
    config). Never raises.
    """
    measurement_id = getattr(settings, "GA4_MEASUREMENT_ID", "") or ""
    api_secret = getattr(settings, "GA4_API_SECRET", "") or ""

    if not measurement_id or not api_secret:
        # Dev / test path: nothing to do.
        return False

    payload: Dict[str, Any] = {
        "client_id": client_id,
        "events": [{"name": event_name, "params": params or {}}],
    }
    if user_id is not None:
        payload["user_id"] = str(user_id)

    try:
        resp = requests.post(
            _MP_ENDPOINT,
            params={"measurement_id": measurement_id, "api_secret": api_secret},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        logger.warning("GA4 MP request failed for event %s: %s", event_name, exc)
        return False

    if 200 <= resp.status_code < 300:
        return True

    logger.warning(
        "GA4 MP non-2xx for event %s: status=%s body=%s",
        event_name,
        resp.status_code,
        resp.text[:200],
    )
    return False
