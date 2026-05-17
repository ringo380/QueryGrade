"""Unit tests for analyzer.analytics (server-side GA4 Measurement Protocol)."""

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

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
