"""
Tests for the anonymous trial flow on the query grader.

Anonymous visitors may grade up to ANON_TRIAL_CAP queries per session and view
the basic results page. Authenticated users retain the full ML-enhanced flow
with UserQueryHistory persistence and feedback.
"""
from django.test import TransactionTestCase, Client, override_settings
from django.contrib.auth.models import User
from django.urls import reverse
from django.db import transaction

from analyzer.models import Query, QueryAnalysis, UserQueryHistory
from analyzer.views.constants import (
    ANON_ANALYSIS_SESSION_KEY,
    ANON_TRIAL_COUNT_KEY,
)


SIMPLE_QUERY = "SELECT id, name FROM users WHERE id = 1;"


@override_settings(
    RATELIMIT_ENABLE=False,
    ANON_TRIAL_CAP=3,
    CACHES={
        'default': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
        'query_analysis_cache': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
        'process_cache': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
        'template_cache': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
    }
)
class AnonymousTrialTestCase(TransactionTestCase):
    """Verify anon users can grade queries without login, up to the trial cap."""

    def setUp(self):
        self.client = Client(enforce_csrf_checks=False)

        # Reinitialize the global query_cache singleton with DummyCache
        from analyzer.performance import query_cache
        from django.core.cache import caches
        query_cache.cache = caches['query_analysis_cache']
        for cache_name in ['default', 'query_analysis_cache', 'process_cache', 'template_cache']:
            try:
                caches[cache_name].clear()
            except Exception:
                pass

    def tearDown(self):
        UserQueryHistory.objects.all().delete()
        QueryAnalysis.objects.all().delete()
        Query.objects.all().delete()
        User.objects.all().delete()

    # ---------- Landing & form rendering ----------

    def test_anon_landing_renders_inline_grade_form(self):
        """GET / for anon shows the inline grade form + trial banner."""
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Try QueryGrade free')
        self.assertContains(response, '3 of 3 grades left')
        self.assertContains(response, 'Create free account')

    def test_anon_grade_page_shows_trial_banner(self):
        """GET /grade/ for anon shows the trial banner and form."""
        response = self.client.get(reverse('grade_query'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Trial mode')
        self.assertContains(response, 'SQL Query Grader')

    # ---------- POST /grade/ ----------

    def test_anon_post_grades_query_without_login(self):
        """Anon POST creates Query+QueryAnalysis, skips UserQueryHistory, increments session."""
        response = self.client.post(reverse('grade_query'), {
            'sql_query': SIMPLE_QUERY,
            'database_type': 'mysql',
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            response.url.startswith('/grade/results/'),
            f"Anon should redirect to basic grade_results, got {response.url}"
        )

        self.assertEqual(Query.objects.count(), 1)
        self.assertEqual(QueryAnalysis.objects.count(), 1)
        self.assertEqual(
            UserQueryHistory.objects.count(), 0,
            "Anon should NOT create UserQueryHistory"
        )

        session = self.client.session
        self.assertEqual(session.get(ANON_TRIAL_COUNT_KEY), 1)
        self.assertEqual(len(session.get(ANON_ANALYSIS_SESSION_KEY, [])), 1)

    def test_anon_can_view_results_for_session_analysis(self):
        """Anon may GET /grade/results/<id>/ when id is in the session list."""
        post = self.client.post(reverse('grade_query'), {
            'sql_query': SIMPLE_QUERY,
            'database_type': 'mysql',
        })
        analysis_id = int(post.url.rstrip('/').split('/')[-1])

        response = self.client.get(reverse('grade_results', args=[analysis_id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Unlock advanced reporting')
        # Feedback widgets should NOT render for anon
        self.assertNotContains(response, 'Was this analysis helpful?')

    def test_anon_blocked_from_other_session_analyses(self):
        """Anon cannot view an analysis_id not in their session."""
        # Create an analysis as some other authenticated user
        user = User.objects.create_user(username='other', password='x')
        self.client.force_login(user)
        post = self.client.post(reverse('grade_query'), {
            'sql_query': SIMPLE_QUERY,
            'database_type': 'mysql',
        })
        analysis_id = QueryAnalysis.objects.first().id
        self.client.logout()

        # Now as anon, attempt to view
        response = self.client.get(reverse('grade_results', args=[analysis_id]))
        self.assertEqual(response.status_code, 302)

    # ---------- Trial cap ----------

    def test_anon_trial_cap_blocks_fourth_post(self):
        """After ANON_TRIAL_CAP grades, a further POST shows the 'trial used' page."""
        for _ in range(3):
            response = self.client.post(reverse('grade_query'), {
                'sql_query': SIMPLE_QUERY,
                'database_type': 'mysql',
            })
            self.assertEqual(response.status_code, 302)

        # 4th grade attempt should NOT redirect; should render the exhausted UI
        response = self.client.post(reverse('grade_query'), {
            'sql_query': SIMPLE_QUERY,
            'database_type': 'mysql',
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "used all 3 free trials")

        # Counter unchanged at cap (still 3)
        self.assertEqual(self.client.session.get(ANON_TRIAL_COUNT_KEY), 3)

    def test_anon_landing_after_cap_shows_register_cta_only(self):
        """GET / after trial cap is hit hides the form and shows the upsell."""
        for _ in range(3):
            self.client.post(reverse('grade_query'), {
                'sql_query': SIMPLE_QUERY,
                'database_type': 'mysql',
            })

        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "used all 3 free trials")

    # ---------- Feedback gates ----------

    def test_anon_cannot_submit_feedback(self):
        """Feedback endpoints remain login-required."""
        post = self.client.post(reverse('grade_query'), {
            'sql_query': SIMPLE_QUERY,
            'database_type': 'mysql',
        })
        analysis_id = int(post.url.rstrip('/').split('/')[-1])

        response = self.client.get(reverse('submit_feedback', args=[analysis_id]))
        self.assertIn(response.status_code, (302, 301))
        self.assertIn('login', response.url)

    def test_anon_cannot_view_enhanced_results(self):
        """Enhanced results stays login-only."""
        post = self.client.post(reverse('grade_query'), {
            'sql_query': SIMPLE_QUERY,
            'database_type': 'mysql',
        })
        analysis_id = int(post.url.rstrip('/').split('/')[-1])

        response = self.client.get(reverse('enhanced_grade_results', args=[analysis_id]))
        self.assertIn(response.status_code, (302, 301))
        self.assertIn('login', response.url)


@override_settings(
    RATELIMIT_ENABLE=False,
    ANON_TRIAL_CAP=3,
    CACHES={
        'default': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
        'query_analysis_cache': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
        'process_cache': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
        'template_cache': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
    }
)
class AuthenticatedRegressionTestCase(TransactionTestCase):
    """Confirm the authenticated flow still creates UserQueryHistory and goes to enhanced results."""

    def setUp(self):
        self.client = Client(enforce_csrf_checks=False)
        from analyzer.performance import query_cache
        from django.core.cache import caches
        query_cache.cache = caches['query_analysis_cache']
        for cache_name in ['default', 'query_analysis_cache', 'process_cache', 'template_cache']:
            try:
                caches[cache_name].clear()
            except Exception:
                pass

        with transaction.atomic():
            self.user = User.objects.create_user(
                username='authuser', password='pw12345678', email='a@b.com'
            )
        self.client.force_login(self.user)

    def tearDown(self):
        UserQueryHistory.objects.all().delete()
        QueryAnalysis.objects.all().delete()
        Query.objects.all().delete()
        User.objects.all().delete()

    def test_authenticated_post_creates_history_and_redirects_to_enhanced(self):
        response = self.client.post(reverse('grade_query'), {
            'sql_query': SIMPLE_QUERY,
            'database_type': 'mysql',
        })
        self.assertEqual(response.status_code, 302)
        self.assertIn('/grade/enhanced/', response.url)
        self.assertEqual(UserQueryHistory.objects.count(), 1)
        self.assertEqual(UserQueryHistory.objects.first().user, self.user)

        # No anon session keys should be set for an authed flow
        self.assertNotIn(ANON_TRIAL_COUNT_KEY, self.client.session)
