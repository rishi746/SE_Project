import io
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from docx import Document


class LLM2WebsiteTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user('lifecycle-analyst')
        self.client.force_login(self.user)

    def test_separate_site_and_login_required(self):
        self.assertContains(self.client.get('/', SERVER_PORT='8001'), 'SDLC studio')
        self.assertContains(self.client.get('/', SERVER_PORT='8000'), 'SRS studio')
        anonymous = Client()
        for endpoint in ('', 'upload/', 'chat/', 'report/', 'download/', 'new/'):
            self.assertEqual(anonymous.get('/llm2/' + endpoint).status_code, 302)
        guarded = Client(enforce_csrf_checks=True)
        guarded.force_login(self.user)
        self.assertEqual(guarded.post('/llm2/chat/', {'message': 'Hello'}).status_code, 403)

    def test_rejects_bad_upload_and_empty_report(self):
        file = SimpleUploadedFile('bad.docx', b'not a document')
        self.assertEqual(self.client.post('/llm2/upload/', {'document': file}).status_code, 400)
        self.assertEqual(self.client.post('/llm2/report/').status_code, 400)

    @patch('workspace.llm2_views.discuss', return_value='How much verification is required?')
    def test_upload_chat_and_reset_are_separate_from_llm1(self, discuss):
        with TemporaryDirectory() as folder, patch('workspace.llm2_views.ROOT', Path(folder)):
            doc = Document(); doc.add_paragraph('Stable network-controller migration with verification.')
            data = io.BytesIO(); doc.save(data)
            response = self.client.post('/llm2/upload/', {'document': SimpleUploadedFile('design.docx', data.getvalue())})
            self.assertEqual(response.status_code, 200)
            session = self.client.session
            session['conversation'] = [{'role': 'user', 'content': 'Separate LLM1 project'}]
            session.save()
            self.assertEqual(self.client.post('/llm2/chat/', {'message': 'Two-person team.'}).status_code, 200)
            self.assertEqual(len(self.client.session['llm2_history']), 2)
            self.assertIsInstance(discuss.call_args.kwargs['design_docx'], bytes)
            other = Client()
            other.force_login(get_user_model().objects.create_user('lifecycle-other'))
            self.assertNotIn('design.docx', other.get('/llm2/').content.decode())
            self.client.post('/llm2/new/')
            self.assertEqual(self.client.session['conversation'][0]['content'], 'Separate LLM1 project')
            self.assertNotIn('llm2_source', self.client.session)

    @patch('workspace.llm2_views.get_engine', return_value=SimpleNamespace(lock=threading.RLock()))
    @patch('workspace.llm2_views.create_report')
    def test_private_report_download(self, create_report, engine):
        with TemporaryDirectory() as folder:
            doc = Path(folder) / 'report.docx'; doc.write_bytes(b'private llm2 result')
            create_report.return_value = SimpleNamespace(markdown='Detailed recommendation', source_warning=())
            with patch('workspace.llm2_views.ROOT', Path(folder)):
                # Use the mocked report destination created by the view.
                def create(*args, **kwargs):
                    path = kwargs['output']; path.parent.mkdir(parents=True); path.write_bytes(doc.read_bytes())
                    return create_report.return_value
                create_report.side_effect = create
                session = self.client.session; session['llm2_history'] = [{'role':'user','content':'Stable migration'}]; session.save()
                self.assertEqual(self.client.post('/llm2/report/').status_code, 200)
                response = self.client.get('/llm2/download/')
                self.assertEqual(b''.join(response.streaming_content), b'private llm2 result')
                other = Client(); other.force_login(get_user_model().objects.create_user('download-other'))
                self.assertEqual(other.get('/llm2/download/').status_code, 404)
