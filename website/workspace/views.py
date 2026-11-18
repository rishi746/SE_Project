import logging
import threading
from pathlib import Path
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

logger = logging.getLogger(__name__)
# ponytail: single-process development server; use a worker queue before multi-worker deployment.
clients = {}
locks = {}
registry_lock = threading.Lock()
report_progress = {}

def progress_key(request):
    return (request.user.pk, request.session.session_key)

@login_required
def progress(request):
    with registry_lock:
        return JsonResponse(report_progress.get(progress_key(request), {'status': 'Waiting for model'}))

def remote(request):
    if not settings.LLM1_URL:
        raise ValueError('Configure LLM1_URL with the running Kaggle SRS app URL.')
    key = (request.user.pk, request.session.session_key)
    with registry_lock:
        if key not in clients:
            from gradio_client import Client
            clients[key] = Client(settings.LLM1_URL, hf_token=settings.LLM1_TOKEN, verbose=False)
            locks[key] = threading.Lock()
    return clients[key], locks[key]

@login_required
def home(request):
    if request.get_port() == '8001':
        from .llm2_views import home as llm2_home
        return llm2_home(request)
    return render(request, 'workspace.html', {'history': request.session.get('conversation', []),
        'local': settings.SMS_INFERENCE == 'local', 'online': bool(settings.LLM1_URL),
        'report_ready': bool(request.session.get('report_path'))})

@login_required
@require_POST
def chat(request):
    message = request.POST.get('message', '').strip()
    if not message or len(message) > 10000:
        return JsonResponse({'error': 'Enter a message of 1–10,000 characters.'}, status=400)
    history = request.session.get('conversation', [])
    if sum(len(t['content']) for t in history) + len(message) > 60000:
        return JsonResponse({'error': 'This discussion is full. Generate your SRS or start a new chat.'}, status=400)
    try:
        if settings.SMS_INFERENCE == 'local':
            from sms_ac.local_inference import get_engine
            result = get_engine().chat(message, history)
        else:
            client, lock = remote(request)
            with lock:
                result = client.predict(message, history, api_name='/website_chat')
        if not isinstance(result, str) or not result.strip():
            raise ValueError('Empty model reply')
        history += [{'role': 'user', 'content': message}, {'role': 'assistant', 'content': result}]
        request.session['conversation'] = history
        request.session.pop('report_path', None)
        return JsonResponse({'reply': result})
    except ValueError as error:
        return JsonResponse({'error': str(error)}, status=400)
    except Exception:
        logger.exception('LLM1 chat failed')
        return JsonResponse({'error': 'The model is unavailable. Your discussion is preserved. Check the local model startup log and retry.'}, status=503)

@login_required
@require_POST
def report(request):
    history = request.session.get('conversation', [])
    if not history:
        return JsonResponse({'error': 'Discuss your project before generating the SRS.'}, status=400)
    try:
        if settings.SMS_INFERENCE == 'local':
            from sms_ac.local_inference import get_engine
            def update(index, total, section):
                with registry_lock:
                    report_progress[progress_key(request)] = {'status': f'Generating section {index}/{total}: {section}'}
            result = get_engine().report(history, progress=update)
        else:
            client, lock = remote(request)
            with lock:
                result = client.predict(history, api_name='/website_report')
        path = Path(result[0])
        if not path.is_file() or path.suffix.lower() != '.docx':
            raise ValueError('No Word document returned')
        request.session['report_path'] = str(path)
        return JsonResponse({'download': '/download/', 'status': result[2]})
    except ValueError as error:
        return JsonResponse({'error': str(error)}, status=400)
    except Exception:
        logger.exception('SRS generation failed')
        return JsonResponse({'error': 'The SRS could not be generated. The discussion is saved; check the model log and retry.'}, status=503)
    finally:
        with registry_lock:
            report_progress.pop(progress_key(request), None)

@login_required
def download(request):
    path = request.session.get('report_path')
    if not path or not Path(path).is_file():
        raise Http404('Generate an SRS first')
    return FileResponse(open(path, 'rb'), as_attachment=True, filename='Networking-SRS.docx')

@login_required
@require_POST
def new_chat(request):
    request.session.pop('conversation', None)
    request.session.pop('report_path', None)
    return JsonResponse({'ok': True})
