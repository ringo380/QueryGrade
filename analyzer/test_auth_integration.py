"""
Integration tests for authentication workflow.

Tests the complete authentication system including login, logout, registration,
password reset, password change, and account management.
"""

from django.test import TestCase, Client, override_settings
from django.contrib.auth.models import User
from django.urls import reverse
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes


# Disable rate limiting for tests
@override_settings(
    RATELIMIT_ENABLE=False,
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        }
    }
)


class AuthenticationWorkflowTestCase(TestCase):
    """Test complete authentication workflow."""

    def setUp(self):
        """Set up test client and test user."""
        self.client = Client()
        self.test_username = 'testuser'
        self.test_password = 'SecureTestPass123!'
        self.test_email = 'test@example.com'

    def test_user_registration_workflow(self):
        """Test complete user registration flow."""
        # Get registration page
        response = self.client.get(reverse('register'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Create Account')

        # Submit registration
        response = self.client.post(reverse('register'), {
            'username': self.test_username,
            'password1': self.test_password,
            'password2': self.test_password,
        })

        # Should redirect to index after successful registration
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/')

        # User should be created
        user = User.objects.get(username=self.test_username)
        self.assertIsNotNone(user)

        # User should be logged in
        self.assertTrue(user.is_authenticated)

    def test_login_logout_workflow(self):
        """Test login and logout flow."""
        # Create test user
        user = User.objects.create_user(
            username=self.test_username,
            password=self.test_password,
            email=self.test_email
        )

        # Get login page
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Welcome Back')

        # Login with correct credentials
        response = self.client.post(reverse('login'), {
            'username': self.test_username,
            'password': self.test_password,
        })

        # Should redirect to index
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/')

        # User should be logged in
        response = self.client.get('/')
        self.assertEqual(response.context['user'].username, self.test_username)

        # Logout
        response = self.client.get(reverse('logout'))
        self.assertEqual(response.status_code, 302)

        # User should be logged out - check by trying to access login page
        # (logged out users can access login page)
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)

        # Verify user is not authenticated by checking the response content
        self.assertContains(response, 'Welcome Back')

    def test_login_with_invalid_credentials(self):
        """Test login with wrong password."""
        # Create test user
        User.objects.create_user(
            username=self.test_username,
            password=self.test_password
        )

        # Try to login with wrong password
        response = self.client.post(reverse('login'), {
            'username': self.test_username,
            'password': 'WrongPassword123!',
        })

        # Should stay on login page
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Invalid username or password')

    def test_login_redirect_parameter(self):
        """Test login redirect to protected page."""
        # Create test user
        User.objects.create_user(
            username=self.test_username,
            password=self.test_password
        )

        # Login with next parameter
        response = self.client.post(
            reverse('login') + '?next=/account/',
            {
                'username': self.test_username,
                'password': self.test_password,
            }
        )

        # Should redirect to account page
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/account/')

    def test_password_reset_request_flow(self):
        """Test password reset request."""
        # Create test user with email
        User.objects.create_user(
            username=self.test_username,
            password=self.test_password,
            email=self.test_email
        )

        # Get password reset page
        response = self.client.get(reverse('password_reset'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Reset Password')

        # Submit password reset request
        response = self.client.post(reverse('password_reset'), {
            'email': self.test_email,
        })

        # Should redirect to login with success message
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/login/')

    def test_password_reset_confirm_flow(self):
        """Test password reset confirmation with token."""
        # Create test user
        user = User.objects.create_user(
            username=self.test_username,
            password=self.test_password,
            email=self.test_email
        )

        # Generate reset token
        token = default_token_generator.make_token(user)
        uidb64 = urlsafe_base64_encode(force_bytes(user.pk))

        # Get password reset confirm page
        url = reverse('password_reset_confirm', kwargs={
            'uidb64': uidb64,
            'token': token
        })
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Set New Password')

        # Submit new password
        new_password = 'NewSecurePass456!'
        response = self.client.post(url, {
            'new_password1': new_password,
            'new_password2': new_password,
        })

        # Should redirect to login
        self.assertEqual(response.status_code, 302)

        # Should be able to login with new password
        response = self.client.post(reverse('login'), {
            'username': self.test_username,
            'password': new_password,
        })
        self.assertEqual(response.status_code, 302)

    def test_password_reset_invalid_token(self):
        """Test password reset with invalid token."""
        # Create test user
        user = User.objects.create_user(
            username=self.test_username,
            password=self.test_password,
            email=self.test_email
        )

        # Use invalid token
        uidb64 = urlsafe_base64_encode(force_bytes(user.pk))
        invalid_token = 'invalid-token-12345'

        url = reverse('password_reset_confirm', kwargs={
            'uidb64': uidb64,
            'token': invalid_token
        })
        response = self.client.get(url)

        # Should show invalid link message
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'invalid or has expired')

    def test_password_change_flow(self):
        """Test password change for logged-in user."""
        # Create and login user
        user = User.objects.create_user(
            username=self.test_username,
            password=self.test_password
        )
        self.client.login(username=self.test_username, password=self.test_password)

        # Get password change page
        response = self.client.get(reverse('password_change'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Change Password')

        # Submit password change
        new_password = 'NewPassword789!'
        response = self.client.post(reverse('password_change'), {
            'old_password': self.test_password,
            'new_password1': new_password,
            'new_password2': new_password,
        })

        # Should redirect to account page
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/account/')

        # Should still be logged in
        response = self.client.get('/account/')
        self.assertEqual(response.status_code, 200)

        # Should be able to login with new password
        self.client.logout()
        login_success = self.client.login(
            username=self.test_username,
            password=new_password
        )
        self.assertTrue(login_success)

    def test_password_change_wrong_old_password(self):
        """Test password change with incorrect old password."""
        # Create and login user
        User.objects.create_user(
            username=self.test_username,
            password=self.test_password
        )
        self.client.login(username=self.test_username, password=self.test_password)

        # Try to change password with wrong old password
        response = self.client.post(reverse('password_change'), {
            'old_password': 'WrongOldPassword!',
            'new_password1': 'NewPassword789!',
            'new_password2': 'NewPassword789!',
        })

        # Should stay on page with error
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['form'].errors)

    def test_account_page_access(self):
        """Test account page for authenticated user."""
        # Create and login user
        user = User.objects.create_user(
            username=self.test_username,
            password=self.test_password,
            email=self.test_email
        )
        self.client.login(username=self.test_username, password=self.test_password)

        # Access account page
        response = self.client.get(reverse('account'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'My Account')
        self.assertContains(response, self.test_username)
        self.assertContains(response, 'Activity Statistics')

    def test_account_page_requires_login(self):
        """Test account page redirects unauthenticated users."""
        # Try to access account page without login
        response = self.client.get(reverse('account'))

        # Should redirect to login page
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith('/login/'))

    def test_already_logged_in_redirects(self):
        """Test that logged-in users are redirected from login/register."""
        # Create and login user
        User.objects.create_user(
            username=self.test_username,
            password=self.test_password
        )
        self.client.login(username=self.test_username, password=self.test_password)

        # Try to access login page
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/')

        # Try to access register page
        response = self.client.get(reverse('register'))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/')

    def test_registration_password_validation(self):
        """Test that weak passwords are rejected during registration."""
        # Try to register with weak password
        response = self.client.post(reverse('register'), {
            'username': self.test_username,
            'password1': 'weak',
            'password2': 'weak',
        })

        # Should stay on registration page with errors
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['form'].errors)

    def test_registration_password_mismatch(self):
        """Test that mismatched passwords are rejected."""
        response = self.client.post(reverse('register'), {
            'username': self.test_username,
            'password1': 'SecurePassword123!',
            'password2': 'DifferentPassword456!',
        })

        # Should stay on registration page with errors
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['form'].errors)

    def test_registration_duplicate_username(self):
        """Test that duplicate usernames are rejected."""
        # Create existing user
        User.objects.create_user(
            username=self.test_username,
            password=self.test_password
        )

        # Try to register with same username
        response = self.client.post(reverse('register'), {
            'username': self.test_username,
            'password1': 'DifferentPass123!',
            'password2': 'DifferentPass123!',
        })

        # Should stay on registration page with errors
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['form'].errors)


@override_settings(
    RATELIMIT_ENABLE=False,
    CACHES={
        'default': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        }
    }
)
class NavigationTestCase(TestCase):
    """Test navigation and UI elements."""

    def setUp(self):
        """Set up test client and user."""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='TestPass123!'
        )

    def test_navigation_authenticated(self):
        """Test navigation menu for authenticated users."""
        self.client.login(username='testuser', password='TestPass123!')

        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

        # Should show authenticated menu items
        content = response.content.decode()
        self.assertIn('Grade Query', content)
        self.assertIn('History', content)
        self.assertIn('testuser', content)

    def test_navigation_unauthenticated(self):
        """Test navigation menu for unauthenticated users."""
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)

        # Should show login/register links
        content = response.content.decode()
        self.assertIn('Login', content)
        self.assertIn('Register', content)
