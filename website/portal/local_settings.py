import os
import secrets
import sys
from .settings import *
sys.path.insert(0, str(BASE_DIR.parent / 'src'))

DEBUG = os.environ.get('SMS_DEBUG', '1') == '1'
key_file = BASE_DIR / '.local-secret'
if not key_file.exists():
    key_file.write_text(secrets.token_urlsafe(64), encoding='utf-8')
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY') or key_file.read_text().strip()
if not DEBUG and not os.environ.get('DJANGO_SECRET_KEY'):
    raise RuntimeError('Set DJANGO_SECRET_KEY for deployment')
ALLOWED_HOSTS = os.environ.get('SMS_ALLOWED_HOSTS', 'localhost,127.0.0.1,testserver').split(',')
INSTALLED_APPS += ['workspace', 'allauth', 'allauth.account', 'allauth.socialaccount', 'allauth.socialaccount.providers.google']
MIDDLEWARE += ['allauth.account.middleware.AccountMiddleware']
TEMPLATES[0]['DIRS'] = [BASE_DIR / 'templates']
TEMPLATES[0]['OPTIONS']['context_processors'].append('workspace.context.auth_options')
AUTHENTICATION_BACKENDS = ['django.contrib.auth.backends.ModelBackend', 'allauth.account.auth_backends.AuthenticationBackend']
LOGIN_URL = '/accounts/login/'
LOGIN_REDIRECT_URL = '/'
ACCOUNT_LOGOUT_REDIRECT_URL = '/accounts/login/'
ACCOUNT_LOGIN_METHODS = {'username', 'email'}
ACCOUNT_SIGNUP_FIELDS = ['email*', 'username*', 'password1*', 'password2*']
ACCOUNT_EMAIL_VERIFICATION = 'none'
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
SOCIALACCOUNT_PROVIDERS = {'google': {'SCOPE': ['profile', 'email'], 'OAUTH_PKCE_ENABLED': True}}
GOOGLE_ENABLED = bool(os.environ.get('GOOGLE_CLIENT_ID') and os.environ.get('GOOGLE_CLIENT_SECRET'))
if GOOGLE_ENABLED:
    SOCIALACCOUNT_PROVIDERS['google']['APPS'] = [{'client_id': os.environ['GOOGLE_CLIENT_ID'], 'secret': os.environ['GOOGLE_CLIENT_SECRET'], 'key': ''}]
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_SSL_REDIRECT = not DEBUG
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024
LLM1_URL = os.environ.get('LLM1_URL', '')
LLM1_TOKEN = os.environ.get('LLM1_TOKEN')
SMS_INFERENCE = os.environ.get('SMS_INFERENCE', 'local')
