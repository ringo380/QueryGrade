"""Tests for the crawler-facing surface: sitemap.xml, robots.txt, and meta tags.

These fail quietly in production: a login-gated URL in the sitemap or a
Disallow rule that contradicts it does not raise an error, it just costs
indexing, and only shows up in Search Console weeks later. The checks here
exercise the real responses so a regression surfaces at test time instead.
"""

import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from django.test import TestCase

SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def sitemap_locs(client):
    """Return the <loc> values from sitemap.xml, parsed as real XML."""
    response = client.get("/sitemap.xml")
    root = ET.fromstring(response.content)
    return [el.text for el in root.findall(".//s:loc", SITEMAP_NS)]


class SitemapTests(TestCase):
    def test_sitemap_is_served_as_well_formed_xml(self):
        response = self.client.get("/sitemap.xml")

        self.assertEqual(response.status_code, 200)
        self.assertIn("xml", response["Content-Type"])
        # Raises ParseError if malformed, which fails the test.
        ET.fromstring(response.content)

    def test_sitemap_is_not_empty(self):
        # Guards the other tests in this class: they all iterate over the
        # locs, so an empty sitemap would vacuously pass every one of them.
        self.assertGreater(len(sitemap_locs(self.client)), 0)

    def test_every_url_is_absolute_https(self):
        for loc in sitemap_locs(self.client):
            with self.subTest(loc=loc):
                self.assertEqual(urlparse(loc).scheme, "https")
                self.assertTrue(urlparse(loc).netloc)

    def test_every_url_is_reachable_by_an_anonymous_crawler(self):
        """A sitemap URL must return 200 without logging in.

        A login_required view 302s to /login/, which Google records as
        "Page with redirect" and drops from the index.
        """
        for loc in sitemap_locs(self.client):
            path = urlparse(loc).path
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(
                    response.status_code,
                    200,
                    f"{path} returned {response.status_code} to an anonymous "
                    f"request; it is not indexable and must not be advertised "
                    f"in the sitemap.",
                )


class RobotsTxtTests(TestCase):
    def test_robots_txt_is_served_as_plain_text(self):
        response = self.client.get("/robots.txt")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/plain")

    def test_sitemap_url_is_absolute_and_honours_forwarded_proto(self):
        """Behind Railway's proxy the app speaks http, so request.scheme has to
        pick https up from X-Forwarded-Proto (SECURE_PROXY_SSL_HEADER) or the
        advertised sitemap URL is wrong in production."""
        response = self.client.get(
            "/robots.txt",
            HTTP_HOST="querygrade.com",
            HTTP_X_FORWARDED_PROTO="https",
        )

        self.assertIn(
            "Sitemap: https://querygrade.com/sitemap.xml",
            response.content.decode(),
        )

    def test_no_disallow_rule_blocks_an_advertised_url(self):
        """robots.txt must not contradict the sitemap."""
        robots = self.client.get("/robots.txt").content.decode()
        disallowed = [
            line.split(":", 1)[1].strip()
            for line in robots.splitlines()
            if line.startswith("Disallow:")
        ]
        self.assertGreater(len(disallowed), 0, "expected some Disallow rules")

        for loc in sitemap_locs(self.client):
            path = urlparse(loc).path
            for rule in disallowed:
                with self.subTest(path=path, rule=rule):
                    self.assertFalse(
                        path.startswith(rule),
                        f"sitemap advertises {path} but robots.txt disallows "
                        f"{rule}",
                    )


class MetaTagTests(TestCase):
    def test_home_page_has_exactly_one_self_referencing_canonical(self):
        response = self.client.get(
            "/", HTTP_HOST="querygrade.com", HTTP_X_FORWARDED_PROTO="https"
        )
        html = response.content.decode()

        self.assertEqual(html.count('rel="canonical"'), 1)
        self.assertIn(
            '<link rel="canonical" href="https://querygrade.com/">', html
        )

    def test_home_page_is_indexable_and_describes_itself(self):
        response = self.client.get("/")
        html = response.content.decode()

        self.assertIn('<meta name="robots" content="index, follow">', html)
        self.assertIn('<meta name="description" content="', html)
        self.assertIn('<meta property="og:title" content="', html)
