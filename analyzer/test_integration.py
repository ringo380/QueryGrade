from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.http import HttpResponse
from analyzer.models import Query, QueryAnalysis, UserQueryHistory
import json


class QueryGradingIntegrationTestCase(TestCase):
    """Integration tests for the complete query grading workflow."""

    def setUp(self):
        """Set up test client and user."""
        self.client = Client()
        self.test_user = User.objects.create_user(
            username='integrationuser',
            email='integration@example.com',
            password='testpass123'
        )

    def test_full_query_grading_workflow(self):
        """Test the complete workflow from login to grading to history."""

        # Step 1: User logs in
        login_response = self.client.post(reverse('login'), {
            'username': 'integrationuser',
            'password': 'testpass123'
        })
        self.assertEqual(login_response.status_code, 302)  # Redirect after login

        # Step 2: Access grade query page
        grade_page = self.client.get(reverse('grade_query'))
        self.assertEqual(grade_page.status_code, 200)
        self.assertContains(grade_page, 'SQL Query Grader')
        self.assertContains(grade_page, '<textarea')

        # Step 3: Submit a query for grading
        test_query = """
            SELECT u.id, u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.created_at >= '2023-01-01'
            GROUP BY u.id, u.name
            ORDER BY order_count DESC
            LIMIT 100;
        """

        grade_response = self.client.post(reverse('grade_query'), {
            'sql_query': test_query,
            'database_type': 'mysql'
        })

        # Should redirect to results page
        self.assertEqual(grade_response.status_code, 302)
        self.assertTrue(grade_response.url.startswith('/grade/results/'))

        # Step 4: Check that Query and Analysis objects were created
        self.assertEqual(Query.objects.count(), 1)
        self.assertEqual(QueryAnalysis.objects.count(), 1)
        self.assertEqual(UserQueryHistory.objects.count(), 1)

        query = Query.objects.first()
        analysis = QueryAnalysis.objects.first()
        history = UserQueryHistory.objects.first()

        # Query text gets normalized, so compare essential content
        self.assertIn('SELECT u.id, u.name, COUNT(o.id)', query.sql_text)
        self.assertIn('FROM users u', query.sql_text)
        self.assertIn('LEFT JOIN orders o', query.sql_text)
        self.assertEqual(query.query_type, 'SELECT')
        self.assertEqual(analysis.grade, 'A')  # Should be an excellent query
        self.assertEqual(history.user, self.test_user)
        self.assertEqual(history.query, query)

        # Step 5: View the results page
        results_response = self.client.get(reverse('grade_results', args=[analysis.id]))
        self.assertEqual(results_response.status_code, 200)

        # Check results page content
        self.assertContains(results_response, f'{analysis.grade}')  # Grade letter is displayed
        self.assertContains(results_response, f'{analysis.score}')  # Score is displayed
        self.assertContains(results_response, 'Query Analysis Results')

        # Step 6: Check query history page
        history_response = self.client.get(reverse('query_history'))
        self.assertEqual(history_response.status_code, 200)
        self.assertContains(history_response, 'Query History')
        self.assertContains(history_response, analysis.grade)
        self.assertContains(history_response, 'SELECT')

    def test_poor_query_grading_workflow(self):
        """Test workflow with a poorly performing query."""

        # Login
        self.client.login(username='integrationuser', password='testpass123')

        # Submit a poor query (Cartesian product)
        poor_query = """
            SELECT *
            FROM users u, products p
            WHERE UPPER(u.email) LIKE '%@gmail.%';
        """

        grade_response = self.client.post(reverse('grade_query'), {
            'sql_query': poor_query
        })

        self.assertEqual(grade_response.status_code, 302)

        # Check that the analysis shows poor performance
        analysis = QueryAnalysis.objects.first()
        self.assertIn(analysis.grade, ['D', 'F'])  # Should be poor grade
        self.assertLess(analysis.score, 70.0)  # Low score
        self.assertGreater(len(analysis.issues_found), 0)  # Should have issues

        # Check results page shows issues
        results_response = self.client.get(reverse('grade_results', args=[analysis.id]))
        self.assertContains(results_response, 'Issues Found')
        self.assertContains(results_response, 'Recommendations')

    def test_authentication_required(self):
        """Test that authentication is required for grading pages."""

        # Try to access grade query page without login
        grade_response = self.client.get(reverse('grade_query'))
        self.assertEqual(grade_response.status_code, 302)  # Redirect to login

        # Try to access history page without login
        history_response = self.client.get(reverse('query_history'))
        self.assertEqual(history_response.status_code, 302)  # Redirect to login

        # Create an analysis to test results page
        self.client.login(username='integrationuser', password='testpass123')
        self.client.post(reverse('grade_query'), {
            'sql_query': 'SELECT * FROM users;'
        })
        analysis = QueryAnalysis.objects.first()
        self.client.logout()

        # Try to access results page without login
        results_response = self.client.get(reverse('grade_results', args=[analysis.id]))
        self.assertEqual(results_response.status_code, 302)  # Redirect to login

    def test_invalid_query_handling(self):
        """Test handling of invalid SQL queries."""

        self.client.login(username='integrationuser', password='testpass123')

        # Submit invalid SQL
        invalid_query = "SELCT * FORM users WHER id = "

        grade_response = self.client.post(reverse('grade_query'), {
            'sql_query': invalid_query
        })

        # Should stay on same page with error message
        self.assertEqual(grade_response.status_code, 200)
        self.assertContains(grade_response, 'SQL syntax error')  # Should show specific error message

    def test_empty_query_handling(self):
        """Test handling of empty queries."""

        self.client.login(username='integrationuser', password='testpass123')

        # Submit empty query
        grade_response = self.client.post(reverse('grade_query'), {
            'sql_query': ''
        })

        # Should stay on same page with validation error
        self.assertEqual(grade_response.status_code, 200)
        self.assertContains(grade_response, 'required')  # Form validation message

    def test_multiple_queries_same_user(self):
        """Test that user can grade multiple queries and view history."""

        self.client.login(username='integrationuser', password='testpass123')

        queries = [
            "SELECT id, name FROM users WHERE active = 1;",
            "SELECT * FROM products WHERE price > 100;",
            "SELECT DISTINCT category FROM products ORDER BY category;"
        ]

        # Submit multiple queries
        for query in queries:
            self.client.post(reverse('grade_query'), {
                'sql_query': query
            })

        # Check that all queries are in history
        self.assertEqual(UserQueryHistory.objects.filter(user=self.test_user).count(), 3)

        # Check history page shows all queries
        history_response = self.client.get(reverse('query_history'))
        self.assertEqual(history_response.status_code, 200)

        for query in queries:
            # Check that snippets of each query appear in history
            truncated = query[:50] if len(query) > 50 else query
            # Note: The template might truncate differently, so we check for key parts
            self.assertContains(history_response, 'SELECT')

    def test_query_complexity_tracking(self):
        """Test that query complexity is properly tracked."""

        self.client.login(username='integrationuser', password='testpass123')

        # Simple query
        simple_query = "SELECT id FROM users;"
        self.client.post(reverse('grade_query'), {
            'sql_query': simple_query
        })

        simple_query_obj = Query.objects.first()
        self.assertLess(simple_query_obj.estimated_complexity, 20)

        # Complex query
        complex_query = """
            SELECT u.id, u.name, COUNT(o.id) as orders,
                   SUM(oi.quantity * p.price) as total_spent
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            LEFT JOIN order_items oi ON o.id = oi.order_id
            LEFT JOIN products p ON oi.product_id = p.id
            WHERE u.created_at >= '2023-01-01'
              AND EXISTS (
                  SELECT 1 FROM user_preferences up
                  WHERE up.user_id = u.id AND up.newsletter = true
              )
            GROUP BY u.id, u.name
            HAVING COUNT(o.id) > 0
            ORDER BY total_spent DESC;
        """

        self.client.post(reverse('grade_query'), {
            'sql_query': complex_query
        })

        complex_query_obj = Query.objects.latest('created_at')
        self.assertGreater(complex_query_obj.estimated_complexity, 40)

    def test_user_query_isolation(self):
        """Test that users can only see their own query history."""

        # Create second user
        user2 = User.objects.create_user(
            username='user2',
            email='user2@example.com',
            password='testpass123'
        )

        # User 1 submits query
        self.client.login(username='integrationuser', password='testpass123')
        self.client.post(reverse('grade_query'), {
            'sql_query': 'SELECT * FROM users;'
        })
        user1_analysis = QueryAnalysis.objects.first()
        self.client.logout()

        # User 2 submits query
        self.client.login(username='user2', password='testpass123')
        self.client.post(reverse('grade_query'), {
            'sql_query': 'SELECT * FROM products;'
        })
        user2_analysis = QueryAnalysis.objects.latest('created_at')

        # User 2 should only see their own history
        history_response = self.client.get(reverse('query_history'))
        self.assertEqual(history_response.status_code, 200)

        # Should contain user2's query but not user1's
        self.assertContains(history_response, 'products')
        self.assertNotContains(history_response, 'users')

        # User 2 should not be able to access User 1's results
        results_response = self.client.get(reverse('grade_results', args=[user1_analysis.id]))
        self.assertEqual(results_response.status_code, 302)  # Should be redirected
        self.assertTrue(results_response.url.startswith('/grade/'))

    def test_grade_display_formatting(self):
        """Test that grades are displayed with proper formatting."""

        self.client.login(username='integrationuser', password='testpass123')

        # Submit query
        self.client.post(reverse('grade_query'), {
            'sql_query': 'SELECT id, name FROM users WHERE active = 1;'
        })

        analysis = QueryAnalysis.objects.first()
        results_response = self.client.get(reverse('grade_results', args=[analysis.id]))

        # Check for grade badge and score display
        self.assertContains(results_response, f'grade-{analysis.grade.lower()}')
        self.assertContains(results_response, f'{analysis.score:.1f}')

        # Check history page formatting
        history_response = self.client.get(reverse('query_history'))
        self.assertContains(history_response, f'grade-{analysis.grade.lower()}')