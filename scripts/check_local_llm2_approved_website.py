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

output = ROOT / 'outputs/local-llm2-approved315-check'
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
source = output / 'network-controller-migration.docx'
fixture = Document()
fixture.add_heading('Network controller migration requirements', 0)
for paragraph in [
    'Replace an existing network-access controller with an as-is implementation on a supported server. No new functionality is permitted. The approved access rules, administrator roles, and external gateway API are fixed.',
    'The existing gateway and identity provider are proven. There is no exploratory hardware or uncertain integration. The specification is signed and stable; the customer accepts one complete delivery.',
    'Verification is the central delivery obligation. Each approved requirement must map to an acceptance test; each component and interface design needs a corresponding test. The new controller must match existing authorization decisions and revocation behavior.',
    'An unapproved request must never grant network access. Tests must cover denied access, expired access, role restrictions, gateway failure and audit records. Compare old and new outputs on the approved test cases.',
    'Two developers have five months. A named customer representative can review test plans and acceptance results. No recurring feature discovery or early user-interface prototype is needed. Recommend the suitable development approach with evidence and limitations.'
]: fixture.add_paragraph(paragraph)
fixture.save(source)
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
    assert 'Optional project context' in response.text
    assert 'Your message' not in response.text and 'clarif' not in response.text.lower()
    csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', response.text)[1]
    headers = {'X-CSRFToken': csrf, 'Referer': 'http://127.0.0.1:8001/'}
    with source.open('rb') as file:
        response = http.post('http://127.0.0.1:8001/llm2/upload/', headers=headers,
            files={'document': (source.name, file, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}, timeout=30)
    response.raise_for_status()
    print('Network migration SRS upload passed:', response.json(), flush=True)
    response = http.post('http://127.0.0.1:8001/llm2/chat/', headers=headers,
        data={'message': 'Summarize the decisive lifecycle selection factors in two sentences. Do not ask questions.'}, timeout=1200)
    response.raise_for_status()
    (output / 'chat.json').write_text(json.dumps(response.json(), indent=2), encoding='utf-8')
    print('Actual local LLM2 chat passed:', response.json()['reply'], flush=True)
    print('Generating detailed recommendation through the website...', flush=True)
    response = http.post('http://127.0.0.1:8001/llm2/report/', headers=headers, timeout=1800)
    (output / 'report-response.json').write_text(response.text, encoding='utf-8')
    response.raise_for_status()
    report = response.json()['report']
    assert _report_problem(report) is None
    quality_check = {'expected': 'V-model', 'actual': report.splitlines()[0], 'passed': report.splitlines()[0].endswith(': V-model')}
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
        'elapsed_seconds': round(time.monotonic() - started, 1), 'classification_check': quality_check}
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    response = http.post('http://127.0.0.1:8000/chat/', headers={'X-CSRFToken': csrf, 'Referer': 'http://127.0.0.1:8000/'}, data={'message': 'I need an office VPN access request application. What details do you need?'}, timeout=1200)
    response.raise_for_status()
    result['llm1_switch_back'] = 'passed'
    result['llm1_reply'] = response.json()
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Functional website checks:', result, flush=True)
    assert quality_check['passed'], 'The model choice failed the verification-focused reference case; see result.json.'
finally:
    session.delete()
    user.delete()
