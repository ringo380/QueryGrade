"""Unit tests for analyzer.analytics (server-side GA4 Measurement Protocol)
and for the session-based one-shot gtag event handoff.
"""

import re
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, override_settings

from analyzer.analytics import send_ga4_event, synthetic_client_id


class SyntheticClientIdTests(SimpleTestCase):
    def test_stable_for_same_user(self):
        self.assertEqual(synthetic_client_id(42), synthetic_client_id(42))

    def test_distinct_for_different_users(self):
        self.assertNotEqual(synthetic_client_id(1), synthetic_client_id(2))

    def test_accepts_str(self):
        self.assertEqual(synthetic_client_id("abc"), "server.abc")


@override_settings(GA4_MEASUREMENT_ID="", GA4_API_SECRET="")
class NoopWhenUnconfiguredTests(SimpleTestCase):
    def test_returns_false_without_measurement_id(self):
        with patch("analyzer.analytics.requests.post") as mock_post:
            ok = send_ga4_event("c1", "test_event", {"k": "v"})
        self.assertFalse(ok)
        mock_post.assert_not_called()


@override_settings(GA4_MEASUREMENT_ID="G-TEST", GA4_API_SECRET="secret123")
class HappyPathTests(SimpleTestCase):
    def test_sends_expected_shape(self):
        mock_resp = MagicMock(status_code=204, text="")
        with patch(
            "analyzer.analytics.requests.post", return_value=mock_resp
        ) as mock_post:
            ok = send_ga4_event(
                client_id="cid-1",
                event_name="log_analysis_completed",
                params={"log_type": "slow", "total_queries": 5},
                user_id=42,
            )
        self.assertTrue(ok)
        mock_post.assert_called_once()
        kwargs = mock_post.call_args.kwargs
        self.assertEqual(
            kwargs["params"], {"measurement_id": "G-TEST", "api_secret": "secret123"}
        )
        body = kwargs["json"]
        self.assertEqual(body["client_id"], "cid-1")
        self.assertEqual(body["user_id"], "42")
        self.assertEqual(len(body["events"]), 1)
        self.assertEqual(body["events"][0]["name"], "log_analysis_completed")
        self.assertEqual(body["events"][0]["params"]["log_type"], "slow")

    def test_omits_user_id_when_not_provided(self):
        mock_resp = MagicMock(status_code=200, text="")
        with patch(
            "analyzer.analytics.requests.post", return_value=mock_resp
        ) as mock_post:
            send_ga4_event("cid", "anon_event", {})
        body = mock_post.call_args.kwargs["json"]
        self.assertNotIn("user_id", body)

    def test_swallows_request_exceptions(self):
        import requests as _requests

        with patch(
            "analyzer.analytics.requests.post",
            side_effect=_requests.ConnectionError("boom"),
        ):
            ok = send_ga4_event("cid", "test", {})
        self.assertFalse(ok)

    def test_returns_false_on_4xx(self):
        mock_resp = MagicMock(status_code=400, text="bad request")
        with patch("analyzer.analytics.requests.post", return_value=mock_resp):
            ok = send_ga4_event("cid", "test", {})
        self.assertFalse(ok)


@override_settings(
    GA4_MEASUREMENT_ID="G-TESTID123",
    # Any {% static %} in base.html needs a collectstatic manifest under the
    # default storage, and that manifest is a gitignored build artifact.
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        },
    },
)
class PendingGtagEventTests(TestCase):
    """The session-based one-shot event handoff, end to end.

    A view sets request.session['_pending_gtag_event'], the ga4_settings
    context processor pops it, and base.html fires it on the next rendered
    page. Nothing about a broken handoff raises an error - the event simply
    never reaches GA4, and the only symptom is a silent gap in a report
    weeks later (issue #106). These tests drive real requests so a
    regression fails here instead.
    """

    PASSWORD = "handoff-pw-12345"

    def setUp(self):
        self.user = User.objects.create_user(username="handoff", password=self.PASSWORD)

    def rendered_event(self, response):
        """The event name the session handoff fires, or None.

        Anchored to the `params || {}` call the pending-event block emits, so
        it cannot accidentally match the unrelated client-side events
        (web_vital, nav_clicked) that base.html also fires on every page.
        """
        match = re.search(
            r"gtag\('event', '([^']+)', params \|\| \{\}\)",
            response.content.decode(),
        )
        return match.group(1) if match else None

    def test_login_fires_user_login_on_the_next_page(self):
        response = self.client.post(
            "/login/", {"username": "handoff", "password": self.PASSWORD}
        )

        # Precondition: a failed login would also produce no event, and the
        # test would pass while proving nothing.
        self.assertEqual(response.status_code, 302)
        self.assertIn("_auth_user_id", self.client.session)

        self.assertEqual(
            self.rendered_event(self.client.get(response["Location"])), "user_login"
        )

    def test_logout_survives_the_session_flush(self):
        """logout() flushes the session, then the view writes the flag into
        the new empty one. The flag has to survive that."""
        self.client.login(username="handoff", password=self.PASSWORD)
        response = self.client.post("/logout/")

        self.assertEqual(response.status_code, 302)
        self.assertNotIn("_auth_user_id", self.client.session)

        self.assertEqual(
            self.rendered_event(self.client.get(response["Location"])), "user_logout"
        )

    def test_event_fires_exactly_once(self):
        """The pop has to mark the session dirty, or the flag survives and
        every subsequent page re-fires the event, inflating the count."""
        self.client.post("/login/", {"username": "handoff", "password": self.PASSWORD})
        self.assertEqual(self.client.session.get("_pending_gtag_event"), "user_login")

        self.assertEqual(self.rendered_event(self.client.get("/")), "user_login")
        # Second render of the same page must be silent.
        self.assertIsNone(self.rendered_event(self.client.get("/")))
        self.assertIsNone(self.client.session.get("_pending_gtag_event"))

    def test_params_ride_along_with_the_event(self):
        self.client.post("/login/", {"username": "handoff", "password": self.PASSWORD})
        html = self.client.get("/").content.decode()

        self.assertIn("gtag-params-data", html)
        self.assertIn("password", html)

    def test_no_event_renders_without_a_pending_flag(self):
        """Guards the tests above: if base.html fired an event unconditionally
        they would pass no matter what the handoff did."""
        self.assertIsNone(self.rendered_event(self.client.get("/")))
