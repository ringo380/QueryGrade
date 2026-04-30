from pathlib import Path
import os
import sys
from django.utils.translation import gettext_lazy as _
from decouple import config, Csv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
# Generate a new secret key with: python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
SECRET_KEY = config('SECRET_KEY', default='django-insecure-CHANGE-ME-IN-PRODUCTION')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config('DEBUG', default=False, cast=bool)

ALLOWED_HOSTS = config(
    'ALLOWED_HOSTS',
    default='localhost,127.0.0.1,querygrade.com,querygrade.net,.up.railway.app',
    cast=Csv(),
)

# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework_simplejwt",
    "django_ratelimit",
    "analyzer",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "analyzer.middleware.CSRFFailureMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "csp.middleware.CSPMiddleware",
    "analyzer.middleware.EnhancedSecurityMiddleware",
]

ROOT_URLCONF = "querygrade.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        'DIRS': [os.path.join(BASE_DIR, 'analyzer/templates')],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "analyzer.context_processors.ga4_settings",
            ],
        },
    },
]

WSGI_APPLICATION = "querygrade.wsgi.application"

# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": os.environ.get("DB_ENGINE", "django.db.backends.sqlite3"),
        "NAME": os.environ.get("DB_NAME", BASE_DIR / "db.sqlite3"),
        "USER": os.environ.get("DB_USER", ""),
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST": os.environ.get("DB_HOST", ""),
        "PORT": os.environ.get("DB_PORT", ""),
        "OPTIONS": {
            # Empty options for SQLite to avoid init_command issues
        } if os.environ.get("DB_ENGINE", "django.db.backends.sqlite3") == "django.db.backends.sqlite3" else {
            # PostgreSQL connection pooling and performance settings
            'sslmode': 'prefer',
            'connect_timeout': 10,
            # Postgres default isolation is already 'read committed' — no need to override.
        },
        "CONN_MAX_AGE": 600,  # Connection pooling - 10 minutes
        "CONN_HEALTH_CHECKS": True,
        "ATOMIC_REQUESTS": True,  # Wrap each request in a transaction
    }
}

# Database performance optimizations
DATABASE_CONNECTION_POOLING = True
DATABASE_ENGINE_OPTIONS = {
    'sqlite3': {
        'PRAGMA foreign_keys': '1',
        'PRAGMA journal_mode': 'WAL',
        'PRAGMA synchronous': 'NORMAL',
        'PRAGMA cache_size': '10000',
        'PRAGMA temp_store': 'MEMORY',
    }
}

# Machine Learning Configuration
ML_ENABLED = os.environ.get('ML_ENABLED', 'True').lower() in ('true', '1', 'yes', 'on')
ML_MODEL_PATH = os.path.join(BASE_DIR, 'analyzer', 'ml', 'models')
ML_MIN_TRAINING_SAMPLES = int(os.environ.get('ML_MIN_TRAINING_SAMPLES', '50'))
ML_RETRAIN_THRESHOLD_DAYS = int(os.environ.get('ML_RETRAIN_THRESHOLD_DAYS', '7'))
ML_PERFORMANCE_THRESHOLD = float(os.environ.get('ML_PERFORMANCE_THRESHOLD', '0.7'))

# ML Feature Flags
ML_HYBRID_GRADING = os.environ.get('ML_HYBRID_GRADING', 'True').lower() in ('true', '1', 'yes', 'on')
ML_AUTO_RETRAIN = os.environ.get('ML_AUTO_RETRAIN', 'True').lower() in ('true', '1', 'yes', 'on')
ML_FEEDBACK_COLLECTION = os.environ.get('ML_FEEDBACK_COLLECTION', 'True').lower() in ('true', '1', 'yes', 'on')

# Password validation
# https://docs.djangoproject.com/en/4.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 12,
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.0/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True

LOCALE_PATHS = [os.path.join(BASE_DIR, 'locale')]

LANGUAGES = [
    ('en', _('English')),
    ('es', _('Spanish')),
]

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.0/howto/static-files/

STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Default primary key field type
# https://docs.djangoproject.com/en/4.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Google Analytics 4 — empty disables instrumentation entirely (no script tag rendered).
GA4_MEASUREMENT_ID = config('GA4_MEASUREMENT_ID', default='')

# Enhanced Cache configuration for performance optimization
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            },
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'IGNORE_EXCEPTIONS': True,
        },
        'KEY_PREFIX': 'querygrade',
        'TIMEOUT': 300,  # 5 minutes default
        'VERSION': 1,
    },
    'process_cache': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/2'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            },
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'IGNORE_EXCEPTIONS': True,
        },
        'KEY_PREFIX': 'querygrade_process',
        'TIMEOUT': 3600,  # 1 hour cache
        'VERSION': 1,
    },
    'query_analysis_cache': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/3'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 30,
                'retry_on_timeout': True,
            },
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'IGNORE_EXCEPTIONS': True,
        },
        'KEY_PREFIX': 'querygrade_analysis',
        'TIMEOUT': 7200,  # 2 hours cache for analysis results
        'VERSION': 1,
    },
    'template_cache': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/4'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 20,
                'retry_on_timeout': True,
            },
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'IGNORE_EXCEPTIONS': True,
        },
        'KEY_PREFIX': 'querygrade_templates',
        'TIMEOUT': 1800,  # 30 minutes for template cache
        'VERSION': 1,
    }
}

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Redirect to home URL after login (Default redirects to /accounts/profile/)
LOGIN_REDIRECT_URL = '/'

# Redirect to login URL if user tries to access a login required page and is not logged in
LOGIN_URL = '/login/'

# Email Configuration for Password Reset
EMAIL_BACKEND = os.environ.get(
    'EMAIL_BACKEND',
    'django.core.mail.backends.console.EmailBackend'  # Development: prints to console
)
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', '587'))
EMAIL_USE_TLS = os.environ.get('EMAIL_USE_TLS', 'True').lower() in ('true', '1', 'yes', 'on')
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', '')
DEFAULT_FROM_EMAIL = os.environ.get('DEFAULT_FROM_EMAIL', 'noreply@querygrade.com')

# Django REST Framework configuration
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour'
    }
}

# JWT Configuration
from datetime import timedelta
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
}

# Celery Configuration
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE
CELERY_ENABLE_UTC = True

# Celery task settings
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes
CELERY_TASK_SOFT_TIME_LIMIT = 25 * 60  # 25 minutes
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_DISABLE_RATE_LIMITS = False

# Celery result expiration
CELERY_RESULT_EXPIRES = 3600  # 1 hour

# Security Settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_HSTS_SECONDS = 31536000 if not DEBUG else 0  # 1 year in production
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
X_FRAME_OPTIONS = 'DENY'
SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'

# Enhanced Content Security Policy for XSS Protection
CSP_DEFAULT_SRC = ("'self'",)
CSP_SCRIPT_SRC = (
    "'self'",
    "'unsafe-inline'",  # Temporarily allow for CodeMirror
    "https://cdnjs.cloudflare.com",  # For CodeMirror CDN
    "'unsafe-eval'",  # For CodeMirror functionality
    "https://*.googletagmanager.com",  # GA4 gtag.js loader
)
CSP_STYLE_SRC = (
    "'self'",
    "'unsafe-inline'",  # For inline styles
    "https://cdnjs.cloudflare.com",  # For external stylesheets
)
CSP_IMG_SRC = ("'self'", "data:", "https:")
CSP_FONT_SRC = ("'self'", "https://cdnjs.cloudflare.com")
CSP_CONNECT_SRC = (
    "'self'",
    "https://*.google-analytics.com",  # GA4 collection
    "https://*.analytics.google.com",  # GA4 collection (regional)
    "https://*.googletagmanager.com",  # GA4 config fetch
)
CSP_FRAME_ANCESTORS = ("'none'",)
CSP_FRAME_SRC = ("'none'",)
CSP_OBJECT_SRC = ("'none'",)
CSP_MEDIA_SRC = ("'self'",)
CSP_CHILD_SRC = ("'none'",)
CSP_BASE_URI = ("'self'",)
CSP_FORM_ACTION = ("'self'",)
CSP_WORKER_SRC = ("'self'",)

# Additional XSS Protection Headers
CSP_BLOCK_ALL_MIXED_CONTENT = True
CSP_UPGRADE_INSECURE_REQUESTS = not DEBUG  # Only in production
CSP_INCLUDE_NONCE_IN = ["script-src", "style-src"]

# Session Security
SESSION_COOKIE_SECURE = not DEBUG
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
SESSION_COOKIE_AGE = 3600  # 1 hour

# Enhanced CSRF Protection
CSRF_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = 'Lax'
CSRF_USE_SESSIONS = True
CSRF_COOKIE_AGE = 3600  # 1 hour
CSRF_TRUSTED_ORIGINS = config(
    'CSRF_TRUSTED_ORIGINS',
    default='https://querygrade.com,https://querygrade.net,https://querygrade-production.up.railway.app',
    cast=Csv(),
)
CSRF_FAILURE_VIEW = 'analyzer.views.csrf_failure'

# Enhanced XSS Protection Settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True

# Template security settings
TEMPLATES[0]['OPTIONS']['context_processors'].extend([
    'django.template.context_processors.csrf',
])

# Additional security settings for XSS prevention
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB limit already set above
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024   # 10MB limit already set above
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Security headers for XSS protection
SECURE_CROSS_ORIGIN_OPENER_POLICY = "same-origin"

# Rate Limiting Settings
# Temporarily disabled for development - enable in production
RATELIMIT_ENABLE = os.environ.get('RATELIMIT_ENABLE', 'False').lower() in ('true', '1', 'yes', 'on')
RATELIMIT_USE_CACHE = 'default'

# Anonymous trial: per-session cap on free query grades for unauthenticated visitors
ANON_TRIAL_CAP = int(os.environ.get('ANON_TRIAL_CAP', '3'))

# File Upload Security
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
FILE_UPLOAD_PERMISSIONS = 0o644

# Performance Optimizations
USE_ETAGS = True
USE_TZ = True

# Static files optimization
STATICFILES_STORAGE = 'whitenoise.storage.CompressedStaticFilesStorage'

# Template performance
TEMPLATES[0]['OPTIONS']['context_processors'].extend([
    'django.template.context_processors.request',
])

# Query optimization settings
DATABASE_QUERY_CACHE_ENABLED = True
QUERY_ANALYSIS_CACHE_TIMEOUT = 7200  # 2 hours

# Performance monitoring
PERFORMANCE_MONITORING_ENABLED = DEBUG
SLOW_QUERY_THRESHOLD = 1.0  # Log queries taking longer than 1 second

# Background task optimization
CELERY_TASK_ROUTES = {
    'analyzer.tasks.process_log_file_async': {'queue': 'heavy_processing'},
    'analyzer.tasks.batch_analyze_queries': {'queue': 'heavy_processing'},
    'analyzer.tasks.analyze_database_schema_async': {'queue': 'heavy_processing'},
    'analyzer.tasks.generate_performance_report': {'queue': 'light_processing'},
    'analyzer.tasks.cleanup_temp_files': {'queue': 'maintenance'},
}

# Memory optimization
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1000
CELERY_WORKER_MAX_MEMORY_PER_CHILD = 200000  # 200MB


# Logging for Security Events
# Ensure log dir exists so the file handler doesn't crash Django setup at import time.
# In ephemeral container envs (Railway/etc) the dir may not exist.
_LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'filters': {
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'console_debug_only': {
            'level': 'INFO',
            'filters': ['require_debug_true'],
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(_LOG_DIR, 'security.log'),
            'maxBytes': 1024*1024*5,  # 5 MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'propagate': True,
        },
        'django.security': {
            'handlers': ['file', 'console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'analyzer': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
