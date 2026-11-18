from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from django.contrib.auth import get_user_model
from django.test import Client, TestCase, override_settings

@override_settings(SMS_INFERENCE='remote')
class WebsiteTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user('analyst', 'analyst@example.com', 'A-strong-password-784!')

    def test_authentication_and_csrf_boundaries(self):
        for url in ['/', '/chat/', '/report/', '/download/', '/progress/']:
            self.assertEqual(self.client.get(url).status_code, 302)
        self.client.force_login(self.user)
        self.assertContains(self.client.get('/'), 'Generate SRS Word document')
        secure = Client(enforce_csrf_checks=True)
        secure.force_login(self.user)
        self.assertEqual(secure.post('/chat/', {'message': 'hello'}).status_code, 403)

    def test_signup_login_logout(self):
        response = self.client.post('/accounts/signup/', {'username': 'newanalyst', 'email': 'new@example.com',
            'password1': 'Another-strong-password-891!', 'password2': 'Another-strong-password-891!'})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(get_user_model().objects.filter(username='newanalyst').exists())
        self.client.post('/accounts/logout/')
        response = self.client.post('/accounts/login/', {'login': 'newanalyst', 'password': 'Another-strong-password-891!'})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.get('/').status_code, 200)

    @patch('workspace.views.remote')
    def test_chat_report_and_user_isolation(self, remote):
        import threading
        model = Mock()
        remote.return_value = (model, threading.Lock())
        model.predict.return_value = 'How many users and which authentication policies?'
        self.client.force_login(self.user)
        response = self.client.post('/chat/', {'message': 'Campus VPN access manager'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.client.session['conversation']), 2)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'srs.docx'
            path.write_bytes(b'test-private-report')
            model.predict.return_value = (str(path), 'report.md', 'SRS generated')
            self.assertEqual(self.client.post('/report/').status_code, 200)
            result = self.client.get('/download/')
            self.assertEqual(b''.join(result.streaming_content), b'test-private-report')
            other = Client()
            other.force_login(get_user_model().objects.create_user('other'))
            self.assertEqual(other.get('/download/').status_code, 404)
            self.assertNotIn('Campus VPN access manager', other.get('/').content.decode())

    @override_settings(GOOGLE_ENABLED=True, SOCIALACCOUNT_PROVIDERS={'google': {'APPS': [{'client_id': 'test-client', 'secret': 'test-secret', 'key': ''}]}})
    def test_google_button_and_oauth_redirect(self):
        self.assertContains(self.client.get('/accounts/login/'), 'Continue with Google')
        response = self.client.post('/accounts/google/login/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('accounts.google.com', response.url)
        self.assertIn('state=', response.url)

    def test_empty_report(self):
        self.client.force_login(self.user)
        self.assertEqual(self.client.post('/report/').status_code, 400)

    def test_report_progress_is_private(self):
        from workspace.views import report_progress
        self.client.force_login(self.user)
        key = (self.user.pk, self.client.session.session_key)
        report_progress[key] = {'status': 'Generating section 3/32'}
        try:
            self.assertEqual(self.client.get('/progress/').json()['status'], 'Generating section 3/32')
            other = Client()
            other.force_login(get_user_model().objects.create_user('progress-other'))
            self.assertEqual(other.get('/progress/').json()['status'], 'Waiting for model')
        finally:
            report_progress.pop(key)
