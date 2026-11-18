"""Exercise actual LLM2 upload, GPU chat, complete report and private download."""
import json
import os
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'website'))
os.environ['DJANGO_SETTINGS_MODULE'] = 'portal.local_settings'
import django
django.setup()
import requests
from django.contrib.auth import get_user_model, SESSION_KEY, BACKEND_SESSION_KEY, HASH_SESSION_KEY
from django.contrib.sessions.backends.db import SessionStore
from docx import Document
from sms_ac.llm2_chat import _report_problem

output = ROOT / 'outputs/local-llm2-check'
output.mkdir(parents=True, exist_ok=True)
for attempt in range(20):
    try:
        if requests.get('http://127.0.0.1:8001/', timeout=2).status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError('The warmed local LLM2 website did not become ready.')
source = ROOT / 'outputs/local-srs/a413bb657d7b48669242a17d22405456/srs.docx'
assert source.is_file(), 'The real LLM1 SRS test file is required.'
user = get_user_model().objects.create_user('llm2-check-' + uuid4().hex[:8])
session = SessionStore()
session[SESSION_KEY] = str(user.pk)
session[BACKEND_SESSION_KEY] = 'django.contrib.auth.backends.ModelBackend'
session[HASH_SESSION_KEY] = user.get_session_auth_hash()
session.create()
try:
    started = time.monotonic()
    http = requests.Session()
    http.cookies.set('sessionid', session.session_key)
    response = http.get('http://127.0.0.1:8001/', timeout=20)
    response.raise_for_status()
    assert 'Choose a lifecycle.' in response.text
    csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', response.text)[1]
    headers = {'X-CSRFToken': csrf, 'Referer': 'http://127.0.0.1:8001/'}
    with source.open('rb') as file:
        response = http.post('http://127.0.0.1:8001/llm2/upload/', headers=headers,
            files={'document': ('office-vpn-srs.docx', file, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}, timeout=30)
    response.raise_for_status()
    print('Real LLM1 SRS upload passed:', response.json(), flush=True)
    response = http.post('http://127.0.0.1:8001/llm2/chat/', headers=headers,
        data={'message': 'Use the uploaded office VPN SRS. What decision factors still need clarification?'}, timeout=1200)
    response.raise_for_status()
    (output / 'chat.json').write_text(json.dumps(response.json(), indent=2), encoding='utf-8')
    print('Actual local LLM2 chat passed:', response.json()['reply'], flush=True)
    response = http.post('http://127.0.0.1:8001/llm2/chat/', headers=headers,
        data={'message': 'The team has two developers and three months for a pilot, with weekly stakeholder reviews. The core access-request workflow is agreed, but the gateway integration still needs investigation. Budget is unknown. Keep the final choice provisional where evidence is missing.'}, timeout=1200)
    response.raise_for_status()
    print('Follow-up passed. Generating detailed recommendation through the website...', flush=True)
    response = http.post('http://127.0.0.1:8001/llm2/report/', headers=headers, timeout=1800)
    (output / 'report-response.json').write_text(response.text, encoding='utf-8')
    response.raise_for_status()
    report = response.json()['report']
    assert _report_problem(report) is None
    (output / 'recommendation.md').write_text(report, encoding='utf-8')
    response = http.get('http://127.0.0.1:8001/llm2/download/', timeout=30)
    response.raise_for_status()
    path = output / 'SDLC-Recommendation.docx'
    path.write_bytes(response.content)
    doc = Document(path)
    numbered = sum(p.style.name == 'List Number' for p in doc.paragraphs)
    assert numbered >= 20, numbered
    result = {'upload': 'passed', 'chat': 'passed', 'report': 'passed', 'private_download': 'passed',
        'bytes': len(response.content), 'numbered_points': numbered,
        'elapsed_seconds': round(time.monotonic() - started, 1)}
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('End-to-end LLM2 website test passed:', result, flush=True)
finally:
    session.delete()
    user.delete()
