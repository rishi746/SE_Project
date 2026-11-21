"""Exercise real SRS generation and private download with an isolated test account."""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'website'), str(ROOT / 'src')]
os.environ['DJANGO_SETTINGS_MODULE'] = 'portal.local_settings'
import django
django.setup()
import requests
from django.contrib.auth import get_user_model, SESSION_KEY, BACKEND_SESSION_KEY, HASH_SESSION_KEY
from django.contrib.sessions.backends.db import SessionStore
from docx import Document
from sms_ac.srs import SRS_SECTIONS

out = ROOT / 'outputs/srs-failure-repro'
out.mkdir(exist_ok=True)
history = json.loads((out / 'history.json').read_text(encoding='utf-8'))
user = get_user_model().objects.create_user('srs-check-' + uuid.uuid4().hex[:10])
session = SessionStore()
session[SESSION_KEY] = str(user.pk)
session[BACKEND_SESSION_KEY] = 'django.contrib.auth.backends.ModelBackend'
session[HASH_SESSION_KEY] = user.get_session_auth_hash()
session['conversation'] = history
session.create()
try:
    http = requests.Session()
    http.cookies.set('sessionid', session.session_key)
    base = 'http://127.0.0.1:8000'
    page = http.get(base + '/', timeout=20)
    page.raise_for_status()
    csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', page.text)[1]
    headers = {'X-CSRFToken': csrf, 'Referer': base + '/'}
    worker = requests.Session()
    worker.cookies.update(http.cookies)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.post, base + '/report/', headers=headers, timeout=5400)
        previous = None
        while not future.done():
            try:
                result = http.get(base + '/progress/', timeout=10)
                result.raise_for_status()
                status = result.json()['status']
                if status != previous:
                    print(status, flush=True)
                    previous = status
                # Simulate an active browser during this long end-to-end test.
                http.post(base + '/session/activity/', headers=headers, timeout=10).raise_for_status()
            except requests.RequestException as error:
                print(type(error).__name__, flush=True)
            time.sleep(15)
        result = future.result()
    (out / 'report-response.json').write_text(result.text, encoding='utf-8')
    result.raise_for_status()
    download = http.get(base + result.json()['download'], timeout=30)
    download.raise_for_status()
    path = out / 'vpn-srs.docx'
    path.write_bytes(download.content)
    doc = Document(path)
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith('Heading')]
    assert headings == SRS_SECTIONS + ['Index'], headings
    text = '\n'.join(p.text for p in doc.paragraphs)
    assert 'Table of Contents' in text and 'IEEE 830' not in text
    assert not any('References' in h for h in headings)
    assert any('Non Functional Requirements' in h for h in headings)
    summary = {'report_http_status': result.status_code, 'download_http_status': download.status_code,
               'download_bytes': len(download.content), 'headings': len(headings),
               'elapsed_seconds': round(time.monotonic() - started, 1)}
    (out / 'result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
finally:
    session.delete()
    user.delete()
