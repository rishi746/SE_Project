"""Authenticated LLM2 workspace. All model calls remain in the shared local process."""
import logging
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from sms_ac.documents import MAX_FILE_BYTES, read_design_docx
from sms_ac.llm2_chat import discuss, create_report
from sms_ac.local_inference import LocalSDLCGenerator, get_engine

logger = logging.getLogger(__name__)
ROOT = settings.BASE_DIR.parent / 'outputs/local-llm2'


@login_required
def home(request):
    return render(request, 'llm2.html', {'history': request.session.get('llm2_history', []),
        'source_name': request.session.get('llm2_source_name', ''),
        'report_ready': bool(request.session.get('llm2_report'))})


def source(request):
    path = request.session.get('llm2_source')
    return Path(path).read_bytes() if path else None


@login_required
@require_POST
def upload(request):
    file = request.FILES.get('document')
    if not file or file.size > MAX_FILE_BYTES:
        return JsonResponse({'error': 'Upload a .docx file up to 8 MiB.'}, status=400)
    try:
        data = file.read()
        extracted = read_design_docx(data, filename=file.name)
        if not extracted.text.strip():
            raise ValueError('The Word document contains no readable text.')
        folder = ROOT / uuid4().hex
        folder.mkdir(parents=True)
        path = folder / 'source.docx'
        path.write_bytes(data)
        request.session['llm2_source'] = str(path)
        request.session['llm2_source_name'] = Path(file.name).name
        request.session['llm2_history'] = []
        request.session.pop('llm2_report', None)
        return JsonResponse({'name': Path(file.name).name, 'characters': len(extracted.text),
            'warnings': list(extracted.warnings)})
    except ValueError as error:
        return JsonResponse({'error': str(error)}, status=400)


@login_required
@require_POST
def chat(request):
    message = request.POST.get('message', '').strip()
    if not message or len(message) > 10000:
        return JsonResponse({'error': 'Enter a message of 1–10,000 characters.'}, status=400)
    history = request.session.get('llm2_history', [])
    try:
        answer = discuss(LocalSDLCGenerator(), history, message, design_docx=source(request),
            filename=request.session.get('llm2_source_name', 'source.docx'))
        request.session['llm2_history'] = history + [{'role': 'user', 'content': message},
            {'role': 'assistant', 'content': answer}]
        request.session.pop('llm2_report', None)
        return JsonResponse({'reply': answer})
    except ValueError as error:
        return JsonResponse({'error': str(error)}, status=400)
    except Exception:
        logger.exception('Local LLM2 chat failed')
        return JsonResponse({'error': 'LLM2 could not answer. Your document and discussion are saved; check the local runtime log.'}, status=503)


@login_required
@require_POST
def report(request):
    if not request.session.get('llm2_source') and not request.session.get('llm2_history'):
        return JsonResponse({'error': 'Upload a Word document or describe your project first.'}, status=400)
    try:
        output = ROOT / uuid4().hex / 'sdlc-recommendation.docx'
        # Keep the adapter selected throughout generation and any repair attempt.
        with get_engine().lock:
            result = create_report(LocalSDLCGenerator(), history=request.session.get('llm2_history', []),
                design_docx=source(request), filename=request.session.get('llm2_source_name', 'source.docx'),
                output=output)
        output.with_suffix('.md').write_text(result.markdown, encoding='utf-8')
        request.session['llm2_report'] = str(output)
        return JsonResponse({'download': '/llm2/download/', 'report': result.markdown,
            'warnings': list(result.source_warning)})
    except ValueError as error:
        return JsonResponse({'error': str(error)}, status=400)
    except Exception:
        logger.exception('Local LLM2 Word report failed')
        return JsonResponse({'error': 'The detailed report could not be completed. Your inputs are saved; check the runtime log and retry.'}, status=503)


@login_required
def download(request):
    path = request.session.get('llm2_report')
    if not path or not Path(path).is_file():
        raise Http404('Generate your recommendation first')
    return FileResponse(open(path, 'rb'), as_attachment=True, filename='SDLC-Recommendation.docx')


@login_required
@require_POST
def new(request):
    for key in ('llm2_history', 'llm2_source', 'llm2_source_name', 'llm2_report'):
        request.session.pop(key, None)
    return JsonResponse({'ok': True})
