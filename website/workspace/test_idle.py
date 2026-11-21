from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.sessions.models import Session
from django.test import Client, TestCase
from django.utils import timezone

from .models import IdleSession, SavedWorkspace


class IdleTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user('idle-user')
        self.client.force_login(self.user)
        self.client.get('/')
        self.key = self.client.session.session_key

    def age(self, seconds):
        value = timezone.now() - timedelta(seconds=seconds)
        IdleSession.objects.filter(session_key=self.key).update(last_activity=value)
        return value

    def test_public_home_and_warning_markup(self):
        self.assertContains(Client().get('/'), 'Sign in to open your workspace.')
        page = self.client.get('/')
        self.assertEqual(page['Cache-Control'], 'no-store')
        self.assertContains(page, 'Stay signed in')
        self.assertContains(page, 'theme-toggle')
        self.assertContains(page, 'pointermove')

    def test_polling_does_not_extend_idle_and_warning_starts_at_nineteen_minutes(self):
        before = self.age(19 * 60)
        result = self.client.get('/session/status/').json()
        self.assertGreater(result['remaining_seconds'], 58)
        self.assertLessEqual(result['remaining_seconds'], 60)
        self.assertEqual(result['warning_remaining_seconds'], 60)
        self.client.get('/progress/')
        self.assertEqual(IdleSession.objects.get(session_key=self.key).last_activity, before)

    def test_expiry_invalidates_session_and_redirects_to_public_home(self):
        self.age(20 * 60)
        response = self.client.get('/llm2/')
        self.assertRedirects(response, '/', fetch_redirect_response=False)
        self.assertFalse(Session.objects.filter(session_key=self.key).exists())
        self.assertNotIn('_auth_user_id', self.client.session)
        self.assertContains(self.client.get('/'), 'Sign in to open')

    def test_expired_activity_cannot_renew_session(self):
        self.age(20 * 60)
        response = self.client.post('/session/activity/')
        self.assertEqual(response.status_code, 401)
        self.assertTrue(response.json()['expired'])
        self.assertFalse(Session.objects.filter(session_key=self.key).exists())

    def test_stay_signed_in_resets_shared_tab_clock_and_saves_draft(self):
        self.age(19 * 60)
        tab = Client(); tab.cookies = self.client.cookies.copy()
        response = tab.post('/session/activity/', {'studio': 'llm2', 'draft': 'Draft VPN project notes'})
        self.assertEqual(response.status_code, 200)
        self.assertGreater(self.client.get('/session/status/').json()['remaining_seconds'], 1198)
        self.assertEqual(SavedWorkspace.objects.get(user=self.user).state['llm2_draft'], 'Draft VPN project notes')

    def test_activity_endpoint_requires_csrf(self):
        secure = Client(enforce_csrf_checks=True)
        secure.force_login(self.user)
        secure.get('/')
        before = IdleSession.objects.get(session_key=secure.session.session_key).last_activity
        self.assertEqual(secure.post('/session/activity/').status_code, 403)
        self.assertEqual(IdleSession.objects.get(session_key=secure.session.session_key).last_activity, before)

    def test_different_user_cannot_restore_saved_work(self):
        self.client.post('/session/activity/', {'studio': 'llm2', 'draft': 'Private draft'})
        self.age(1200); self.client.get('/session/status/')
        self.client.force_login(self.user)
        self.assertContains(self.client.get('/llm2/'), 'Private draft')
        other = Client(); other.force_login(get_user_model().objects.create_user('idle-other'))
        self.assertNotContains(other.get('/llm2/'), 'Private draft')

    def test_late_report_survives_expiry_without_restoring_old_login(self):
        session = self.client.session
        session['conversation'] = [{'role': 'user', 'content': 'Network controller'}]
        session.save()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.docx'; path.write_bytes(b'recovered-private-report')
            def finish(*args, **kwargs):
                Session.objects.filter(session_key=self.key).delete()
                IdleSession.objects.filter(session_key=self.key).delete()
                return str(path), '', 'Finished'
            with patch('sms_ac.local_inference.get_engine', return_value=SimpleNamespace(report=finish)):
                response = self.client.post('/report/')
            self.assertEqual(response.status_code, 401)
            self.assertNotIn('sessionid', response.cookies)
            self.assertFalse(Session.objects.filter(session_key=self.key).exists())
            self.assertEqual(SavedWorkspace.objects.get(user=self.user).state['report_path'], str(path))
            self.client.force_login(self.user)
            response = self.client.get('/download/')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(b''.join(response.streaming_content), b'recovered-private-report')

    def test_reset_tombstones_do_not_restore_old_project_on_next_login(self):
        self.client.post('/session/activity/', {'studio': 'llm2', 'draft': 'Old project'})
        self.client.post('/llm2/new/')
        self.client.post('/accounts/logout/')
        self.client.force_login(self.user)
        self.assertNotContains(self.client.get('/llm2/'), 'Old project')

    def test_llm2_report_survives_expiry_and_remains_private(self):
        session = self.client.session
        session['llm2_history'] = [{'role': 'user', 'content': 'Network review'}]
        session.save()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'sdlc.docx'; path.write_bytes(b'private-lifecycle-report')
            def finish(*args, **kwargs):
                Session.objects.filter(session_key=self.key).delete()
                IdleSession.objects.filter(session_key=self.key).delete()
                return SimpleNamespace(markdown='Finished report', source_warning=())
            with patch('workspace.llm2_views.ROOT', Path(directory)), patch('workspace.llm2_views.get_engine', return_value=SimpleNamespace(lock=RLock())), patch('workspace.llm2_views.create_report') as create:
                def result(*args, **kwargs):
                    output = kwargs['output']; output.parent.mkdir(parents=True); output.write_bytes(b'private-lifecycle-report')
                    return finish()
                create.side_effect = result
                response = self.client.post('/llm2/report/')
            self.assertEqual(response.status_code, 401)
            self.assertNotIn('sessionid', response.cookies)
            self.client.force_login(self.user)
            response = self.client.get('/llm2/download/')
            self.assertEqual(b''.join(response.streaming_content), b'private-lifecycle-report')
            other = Client(); other.force_login(get_user_model().objects.create_user('report-other'))
            self.assertEqual(other.get('/llm2/download/').status_code, 404)
