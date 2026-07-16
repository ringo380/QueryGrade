"""Sitemap definitions for QueryGrade.

Exposes the public, indexable entry points so search engines can discover
the site. Authenticated/result pages and API endpoints are deliberately
excluded.
"""

from django.contrib.sitemaps import Sitemap
from django.urls import reverse


class StaticViewSitemap(Sitemap):
    """Sitemap for the public, crawlable pages of the app."""

    protocol = "https"
    changefreq = "weekly"

    # (url name, priority)
    # Only list pages an anonymous crawler can actually reach. A login_required
    # view redirects to /login/, which search engines index as a redirect and
    # drop, so listing one here costs crawl budget and trust in the sitemap.
    # test_seo.py enforces this by fetching every URL below anonymously.
    pages = [
        ("index", 1.0),
        ("grade_query", 0.9),
        ("register", 0.5),
        ("login", 0.4),
    ]

    def items(self):
        return self.pages

    def location(self, item):
        return reverse(item[0])

    def priority(self, item):
        return item[1]


sitemaps = {
    "static": StaticViewSitemap,
}
