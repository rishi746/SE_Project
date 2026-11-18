"""Real loopback chat/report/download smoke test using a disposable account."""
import json
import os
import re
import sys
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'website'))
os.environ['DJANGO_SETTINGS_MODULE'] = 'portal.local_settings'
import django
django.setup()
import requests
from django.contrib.auth import get_user_model, SESSION_KEY, BACKEND_SESSION_KEY, HASH_SESSION_KEY
from django.contrib.sessions.backends.db import SessionStore

out = ROOT / 'outputs/local-runtime-check'
out.mkdir(parents=True, exist_ok=True)
user = get_user_model().objects.create_user('runtime-check-' + uuid.uuid4().hex[:8])
session = SessionStore()
session[SESSION_KEY] = str(user.pk)
session[BACKEND_SESSION_KEY] = 'django.contrib.auth.backends.ModelBackend'
session[HASH_SESSION_KEY] = user.get_session_auth_hash()
session.create()
try:
    http = requests.Session()
    http.cookies.set('sessionid', session.session_key)
    page = http.get('http://127.0.0.1:8000/', timeout=20)
    page.raise_for_status()
    assert 'Local GPU model' in page.text
    csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', page.text)[1]
    headers = {'X-CSRFToken': csrf, 'Referer': 'http://127.0.0.1:8000/'}
    answer = http.post('http://127.0.0.1:8000/chat/', headers=headers,
        data={'message': 'I need to develop a VPN access management application for my office.'}, timeout=180)
    answer.raise_for_status()
    (out / 'chat.json').write_text(json.dumps(answer.json(), indent=2), encoding='utf-8')
    print('Real website chat passed:', answer.json()['reply'], flush=True)
    if '--chat-only' in sys.argv:
        print('Final website chat check passed; existing report test results retained.', flush=True)
        sys.exit(0)
    details = ('The office has 80 employees and one site. Thirty remote users connect through an existing VPN gateway; '
        'the project is the access-management web application, not a new VPN protocol. Employees request access, '
        'managers approve it, and network administrators provision or revoke it. Auditors have read-only log access. '
        'Use the existing identity provider with MFA. Maintain immutable audit records for 90 days. '
        'Integrate through the existing gateway API, subject to discovery. Target 99.5 percent monthly availability '
        'and response times under two seconds for ordinary pages, proposed targets to validate. '
        'Two developers have three months for a pilot; budget is unknown. A stakeholder can review weekly. '
        'Do not implement tunnel encryption or routing. Record missing protocol/API details as TBD.')
    answer = http.post('http://127.0.0.1:8000/chat/', headers=headers, data={'message': details}, timeout=180)
    answer.raise_for_status()
    print('Clarification follow-up passed. Generating full SRS through website...', flush=True)
    result = http.post('http://127.0.0.1:8000/report/', headers=headers, timeout=2400)
    (out / 'report-response.json').write_text(result.text, encoding='utf-8')
    result.raise_for_status()
    file = http.get('http://127.0.0.1:8000/download/', timeout=30)
    file.raise_for_status()
    path = out / 'vpn-srs.docx'
    path.write_bytes(file.content)
    from docx import Document
    doc = Document(path)
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith('Heading')]
    assert len(headings) >= 32, headings
    assert any('Overall Description' in p for p in headings)
    assert any('Specific Requirements' in p for p in headings)
    assert zipfile.is_zipfile(path)
    (out / 'result.json').write_text(json.dumps({'chat': 'passed', 'report': 'passed',
        'download_bytes': len(file.content), 'heading_count': len(headings)}, indent=2))
    print('Full report and private Word download passed:', path, flush=True)
finally:
    session.delete()
    user.delete()
