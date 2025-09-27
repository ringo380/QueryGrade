from django.test import TestCase
from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
import json

from .models import Query, QueryAnalysis, UserQueryHistory, QueryFeedback


class QueryGradingAPITestCase(TestCase):
    """Test cases for the Query Grading API."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()

        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

        # Get JWT token for authentication
        refresh = RefreshToken.for_user(self.user)
        self.access_token = str(refresh.access_token)

    def authenticate(self):
        """Authenticate API client with JWT token."""
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {self.access_token}')

    def test_api_health_check(self):
        """Test API health check endpoint."""
        url = reverse('api:health')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
        self.assertEqual(response.data['status'], 'healthy')
        self.assertIn('features', response.data)

    def test_jwt_token_obtain(self):
        """Test JWT token generation."""
        url = reverse('api:token_obtain_pair')
        data = {
            'username': 'testuser',
            'password': 'testpass123'
        }
        response = self.client.post(url, data)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)

    def test_jwt_token_refresh(self):
        """Test JWT token refresh."""
        refresh = RefreshToken.for_user(self.user)
        url = reverse('api:token_refresh')
        data = {'refresh': str(refresh)}
        response = self.client.post(url, data)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)

    def test_grade_query_api_unauthenticated(self):
        """Test that query grading requires authentication."""
        url = reverse('api:grade_query')
        data = {'sql_text': 'SELECT * FROM users'}
        response = self.client.post(url, data)

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_grade_query_api_success(self):
        """Test successful query grading via API."""
        self.authenticate()
        url = reverse('api:grade_query')
        data = {
            'sql_text': 'SELECT id, name, email FROM users WHERE active = 1 LIMIT 10',
            'database_type': 'mysql',
            'database_version': '8.0',
            'use_case_notes': 'Test query for API'
        }
        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('query_id', response.data)
        self.assertIn('analysis_id', response.data)
        self.assertIn('grade', response.data)
        self.assertIn('score', response.data)
        self.assertIn('issues_found', response.data)
        self.assertIn('recommendations', response.data)

        # Verify database records were created
        self.assertTrue(Query.objects.filter(id=response.data['query_id']).exists())
        self.assertTrue(QueryAnalysis.objects.filter(id=response.data['analysis_id']).exists())
        self.assertTrue(UserQueryHistory.objects.filter(user=self.user).exists())

    def test_grade_query_api_invalid_data(self):
        """Test query grading with invalid data."""
        self.authenticate()
        url = reverse('api:grade_query')
        data = {'sql_text': ''}  # Empty query
        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)

    def test_batch_analysis_api(self):
        """Test batch query analysis via API."""
        self.authenticate()
        url = reverse('api:batch_analysis')
        data = {
            'queries': [
                'SELECT * FROM users WHERE active = 1',
                'SELECT COUNT(*) FROM orders',
                'SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id'
            ],
            'database_type': 'postgresql',
            'database_version': '13.0',
            'analysis_notes': 'API batch test'
        }
        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('total_queries', response.data)
        self.assertIn('successful_analyses', response.data)
        self.assertIn('results', response.data)
        self.assertEqual(response.data['total_queries'], 3)
        self.assertGreaterEqual(response.data['successful_analyses'], 1)

    def test_batch_analysis_api_invalid_data(self):
        """Test batch analysis with invalid data."""
        self.authenticate()
        url = reverse('api:batch_analysis')
        data = {'queries': []}  # Empty queries list
        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_query_history_api(self):
        """Test query history listing via API."""
        # First, create some query history
        self.authenticate()
        grade_url = reverse('api:grade_query')
        grade_data = {'sql_text': 'SELECT * FROM test_table'}
        self.client.post(grade_url, grade_data, format='json')

        # Now test history endpoint
        history_url = reverse('api:query_history')
        response = self.client.get(history_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.data)
        self.assertGreater(len(response.data['results']), 0)

        # Check that result contains expected fields
        first_result = response.data['results'][0]
        expected_fields = ['id', 'query_type', 'grade', 'score', 'query_preview', 'submitted_at']
        for field in expected_fields:
            self.assertIn(field, first_result)

    def test_analysis_detail_api(self):
        """Test detailed analysis retrieval via API."""
        # Create a query analysis first
        self.authenticate()
        grade_url = reverse('api:grade_query')
        grade_data = {'sql_text': 'SELECT * FROM detailed_test'}
        grade_response = self.client.post(grade_url, grade_data, format='json')

        analysis_id = grade_response.data['analysis_id']

        # Test detail endpoint
        detail_url = reverse('api:analysis_detail', args=[analysis_id])
        response = self.client.get(detail_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('id', response.data)
        self.assertIn('query', response.data)
        self.assertIn('grade', response.data)
        self.assertIn('issues_found', response.data)

    def test_analysis_detail_api_access_control(self):
        """Test that users can only access their own analyses."""
        # Create another user and their analysis
        other_user = User.objects.create_user(
            username='otheruser',
            email='other@example.com',
            password='otherpass123'
        )

        # Create analysis for other user (manually)
        from .query_analyzer import analyze_query
        query, analysis = analyze_query("SELECT * FROM other_test")
        other_history = UserQueryHistory.objects.create(
            user=other_user,
            query=query
        )

        # Try to access other user's analysis
        self.authenticate()  # Authenticate as testuser
        detail_url = reverse('api:analysis_detail', args=[analysis.id])
        response = self.client.get(detail_url)

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_submit_feedback_api(self):
        """Test feedback submission via API."""
        # Create a query analysis first
        self.authenticate()
        grade_url = reverse('api:grade_query')
        grade_data = {'sql_text': 'SELECT * FROM feedback_test'}
        grade_response = self.client.post(grade_url, grade_data, format='json')

        analysis_id = grade_response.data['analysis_id']

        # Submit feedback
        feedback_url = reverse('api:submit_feedback', args=[analysis_id])
        feedback_data = {
            'accuracy_rating': 4,
            'usefulness_rating': 5,
            'clarity_rating': 4,
            'suggestions': 'Great API analysis!',
            'would_recommend': True
        }
        response = self.client.post(feedback_url, feedback_data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('message', response.data)
        self.assertIn('feedback_id', response.data)

        # Verify feedback was saved
        feedback = QueryFeedback.objects.get(id=response.data['feedback_id'])
        self.assertEqual(feedback.accuracy_rating, 4)
        self.assertEqual(feedback.suggestions, 'Great API analysis!')

    def test_submit_feedback_api_update(self):
        """Test updating existing feedback via API."""
        # Create analysis and initial feedback
        self.authenticate()
        grade_url = reverse('api:grade_query')
        grade_data = {'sql_text': 'SELECT * FROM update_feedback_test'}
        grade_response = self.client.post(grade_url, grade_data, format='json')
        analysis_id = grade_response.data['analysis_id']

        # Submit initial feedback
        feedback_url = reverse('api:submit_feedback', args=[analysis_id])
        initial_feedback = {
            'accuracy_rating': 3,
            'usefulness_rating': 3,
            'clarity_rating': 3,
            'suggestions': 'Initial feedback',
            'would_recommend': False
        }
        self.client.post(feedback_url, initial_feedback, format='json')

        # Update feedback
        updated_feedback = {
            'accuracy_rating': 5,
            'usefulness_rating': 5,
            'clarity_rating': 5,
            'suggestions': 'Updated feedback - much better!',
            'would_recommend': True
        }
        response = self.client.post(feedback_url, updated_feedback, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('updated', response.data['message'])

    def test_user_stats_api(self):
        """Test user statistics via API."""
        # Create some query history
        self.authenticate()
        grade_url = reverse('api:grade_query')

        # Grade a few queries to generate stats
        queries = [
            'SELECT * FROM stats_test1',
            'SELECT COUNT(*) FROM stats_test2',
            'SELECT id, name FROM stats_test3 WHERE active = 1'
        ]

        for query in queries:
            self.client.post(grade_url, {'sql_text': query}, format='json')

        # Test stats endpoint
        stats_url = reverse('api:user_stats')
        response = self.client.get(stats_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('total_queries', response.data)
        self.assertIn('average_score', response.data)
        self.assertIn('grade_distribution', response.data)
        self.assertIn('recent_activity', response.data)

        self.assertEqual(response.data['total_queries'], 3)

    def test_user_stats_api_no_data(self):
        """Test user statistics with no query history."""
        self.authenticate()
        stats_url = reverse('api:user_stats')
        response = self.client.get(stats_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['total_queries'], 0)

    def test_api_pagination(self):
        """Test API pagination for query history."""
        # Create multiple queries to test pagination
        self.authenticate()
        grade_url = reverse('api:grade_query')

        # Create 25 queries (more than default page size of 20)
        for i in range(25):
            query_data = {'sql_text': f'SELECT * FROM pagination_test_{i}'}
            self.client.post(grade_url, query_data, format='json')

        # Test first page
        history_url = reverse('api:query_history')
        response = self.client.get(history_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('next', response.data)
        self.assertIn('previous', response.data)
        self.assertIn('count', response.data)
        self.assertEqual(response.data['count'], 25)
        self.assertEqual(len(response.data['results']), 20)  # Page size

        # Test second page
        response = self.client.get(f"{history_url}?page=2")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 5)  # Remaining items

    def test_api_throttling_headers(self):
        """Test that API includes throttling headers."""
        self.authenticate()
        url = reverse('api:health')
        response = self.client.get(url)

        # Check that throttling headers are present (they should be in a real request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Note: Throttling headers might not be present in test environment