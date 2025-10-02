# QueryGrade Authentication System

## Overview
Complete authentication system with user management, password recovery, and session security.

## Features Implemented

### 1. User Authentication ✅
- **Login** (`/login/`)
  - Session-based authentication
  - Rate limiting (5 attempts per 5 minutes)
  - Redirect support for protected pages
  - Success/error messages
  - "Forgot password" link

- **Logout** (`/logout/`)
  - Secure session termination
  - Confirmation message with username

- **Registration** (`/register/`)
  - Username and password creation
  - Password strength validation (12+ chars)
  - Rate limiting (3 attempts per hour)
  - Auto-login after registration
  - Link to login page

### 2. Password Management ✅
- **Password Reset Request** (`/password-reset/`)
  - Email-based reset flow
  - Token generation with 24-hour expiration
  - Rate limiting (3 attempts per hour)
  - Development mode: displays reset URL in console

- **Password Reset Confirmation** (`/password-reset-confirm/<uidb64>/<token>/`)
  - Secure token validation
  - Password strength validation
  - Invalid/expired link handling

- **Password Change** (`/password-change/`)
  - For logged-in users only
  - Requires old password
  - Session preservation after change
  - Redirect to account page

### 3. User Account Management ✅
- **Account Page** (`/account/`)
  - Profile information (username, email, join date, last login)
  - Activity statistics (queries analyzed, feedback given)
  - Recent query history (last 5 queries)
  - Quick access to password change
  - Links to full query history

### 4. Navigation & UX ✅
- **User Dropdown Menu**
  - Displays username in navigation
  - Dropdown with:
    - My Account
    - Change Password
    - Logout
  - Clean dark mode UI
  - Responsive design

### 5. Security Features ✅
- **Session Security**
  - HTTPOnly cookies
  - SameSite protection (Lax)
  - 1-hour session timeout
  - Session expires on browser close

- **CSRF Protection**
  - Token-based validation
  - 1-hour token expiration
  - Custom failure view

- **Password Validation**
  - Minimum 12 characters
  - Not similar to username
  - Not common passwords
  - Not entirely numeric

- **Rate Limiting**
  - Login: 5 attempts per 5 minutes (IP-based)
  - Registration: 3 attempts per hour (IP-based)
  - Password reset: 3 attempts per hour (IP-based)

- **XSS Protection**
  - Content Security Policy (CSP)
  - Secure headers
  - Auto-escaping in templates

## URL Patterns

```
/login/                              - User login
/logout/                             - User logout
/register/                           - New user registration
/password-reset/                     - Request password reset
/password-reset-confirm/<uidb64>/<token>/  - Confirm password reset
/password-change/                    - Change password (auth required)
/account/                            - User account page (auth required)
```

## Templates

All templates use dark mode theme and are mobile-responsive:

- `analyzer/templates/analyzer/login.html`
- `analyzer/templates/analyzer/register.html`
- `analyzer/templates/analyzer/password_reset.html`
- `analyzer/templates/analyzer/password_reset_confirm.html`
- `analyzer/templates/analyzer/password_reset_email.html`
- `analyzer/templates/analyzer/password_change.html`
- `analyzer/templates/analyzer/account.html`

## Configuration

### Email Settings (Development)
```python
# Email backend prints to console in development
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# For production, configure SMTP:
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'your-email@gmail.com'
EMAIL_HOST_PASSWORD = 'your-app-password'
DEFAULT_FROM_EMAIL = 'noreply@querygrade.com'
```

### Session Settings
```python
SESSION_COOKIE_SECURE = True  # HTTPS only in production
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
SESSION_COOKIE_AGE = 3600  # 1 hour
```

### CSRF Settings
```python
CSRF_COOKIE_SECURE = True  # HTTPS only in production
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = 'Lax'
CSRF_USE_SESSIONS = True
CSRF_COOKIE_AGE = 3600  # 1 hour
```

## Testing the Workflow

### 1. Create an Account
```bash
# Navigate to http://localhost:8000/register/
# Enter username and password (12+ chars)
# Auto-logged in after creation
```

### 2. View Account
```bash
# Click on username dropdown in navigation
# Select "My Account"
# View profile info and statistics
```

### 3. Change Password
```bash
# From account page or dropdown menu
# Enter old password and new password (12+ chars)
# Session maintained after change
```

### 4. Reset Password (Forgot Password)
```bash
# Click "Forgot your password?" on login page
# Enter email address
# Check console for reset link (development)
# Click link and set new password
```

### 5. Logout
```bash
# Click on username dropdown
# Select "Logout"
# Redirected to login page
```

## Future Enhancements (Optional)

### Email Verification
- Send verification email on registration
- Verify email before allowing login
- Re-send verification email option

### Two-Factor Authentication (2FA)
- TOTP-based 2FA
- Backup codes
- SMS verification option

### Social Authentication
- Google OAuth
- GitHub OAuth
- Microsoft OAuth

### Account Features
- Email address change
- Account deletion
- Export user data
- Login history/sessions

### Security Enhancements
- Failed login notifications
- Suspicious activity alerts
- IP-based restrictions
- Password expiration policies

## Production Deployment Checklist

Before deploying to production, ensure:

- [ ] `DEBUG = False` in settings.py
- [ ] Set strong `SECRET_KEY` (50+ random characters)
- [ ] Configure `ALLOWED_HOSTS` with your domain
- [ ] Set `SESSION_COOKIE_SECURE = True`
- [ ] Set `CSRF_COOKIE_SECURE = True`
- [ ] Configure SMTP email backend
- [ ] Enable HTTPS/SSL
- [ ] Set `SECURE_HSTS_SECONDS = 31536000`
- [ ] Set `SECURE_SSL_REDIRECT = True`
- [ ] Configure proper `CSRF_TRUSTED_ORIGINS`
- [ ] Set up monitoring and logging
- [ ] Configure backup strategy

## Notes

- All authentication views use rate limiting to prevent brute force attacks
- Password reset tokens expire after 24 hours
- Sessions expire after 1 hour of inactivity
- All forms include CSRF protection
- Templates are mobile-responsive with dark mode by default
- Email backend prints to console in development mode
- Production requires proper SMTP configuration

## File Locations

### Views
- `analyzer/views/auth_views.py` - All authentication views

### Templates
- `analyzer/templates/analyzer/` - All auth templates

### URLs
- `analyzer/urls.py` - URL patterns for auth endpoints

### Settings
- `querygrade/settings.py` - Security and email configuration
