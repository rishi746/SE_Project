"""Server-enforced idle expiry and per-account recovery of saved project work."""
from copy import deepcopy
from datetime import timedelta

from django.conf import settings
from django.contrib.auth import logout
from django.contrib.sessions.exceptions import SessionInterrupted
from django.contrib.sessions.middleware import SessionMiddleware
from django.contrib.sessions.models import Session
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import redirect
from django.utils import timezone
from django.utils.deprecation import MiddlewareMixin
from django.views.decorators.http import require_GET, require_POST

from .models import IdleSession, SavedWorkspace

WORKSPACE_KEYS = (
    'conversation', 'report_path', 'llm1_draft', 'llm2_history', 'llm2_source',
    'llm2_source_name', 'llm2_report', 'llm2_draft',
)
HUMAN_POSTS = {'/chat/', '/report/', '/new/', '/llm2/chat/', '/llm2/report/', '/llm2/upload/', '/llm2/new/'}
API_PATHS = HUMAN_POSTS | {'/progress/', '/session/status/', '/session/activity/'}


def expired_response(request):
    if request.path in API_PATHS:
        return JsonResponse({'expired': True, 'redirect': '/', 'error': 'Your session expired. Sign in to continue your saved project.'}, status=401)
    return redirect('/')


def workspace_snapshot(session):
    return deepcopy({key: session[key] for key in WORKSPACE_KEYS if key in session})


class SafeSessionMiddleware(SessionMiddleware):
    def process_response(self, request, response):
        try:
            return super().process_response(request, response)
        except SessionInterrupted:
            # A different tab can expire the session while a model call finishes.
            # The idle middleware has already saved its result by account.
            return expired_response(request)


class IdleMiddleware(MiddlewareMixin):
    def process_request(self, request):
        if not request.user.is_authenticated:
            return
        request._workspace_user_id = request.user.pk
        request._idle_key = request.session.session_key
        saved, created = SavedWorkspace.objects.get_or_create(user_id=request.user.pk,
            defaults={'state': workspace_snapshot(request.session)})
        # Import older session-only fields once; None marks an explicit project reset.
        legacy = {key: request.session[key] for key in WORKSPACE_KEYS if key in request.session and key not in saved.state}
        if legacy:
            saved.state = {**saved.state, **legacy}
            saved.save(update_fields=['state'])
        for key in WORKSPACE_KEYS:
            if key in saved.state:
                if saved.state[key] is None:
                    request.session.pop(key, None)
                elif request.session.get(key) != saved.state[key]:
                    request.session[key] = saved.state[key]
        request._workspace_before = workspace_snapshot(request.session)
        record, _ = IdleSession.objects.get_or_create(session_key=request._idle_key,
            defaults={'user_id': request.user.pk})
        cutoff = timezone.now() - timedelta(seconds=settings.IDLE_TIMEOUT_SECONDS)
        if record.user_id != request.user.pk or record.last_activity <= cutoff:
            IdleSession.objects.filter(session_key=request._idle_key).delete()
            logout(request)
            return expired_response(request)

    def process_view(self, request, view_func, view_args, view_kwargs):
        # CsrfViewMiddleware runs before this hook. Polling and downloads never reset idle time.
        if request.user.is_authenticated and request.method == 'POST' and request.path in HUMAN_POSTS:
            IdleSession.objects.filter(session_key=request.session.session_key, user_id=request.user.pk).update(last_activity=timezone.now())

    def process_response(self, request, response):
        user_id = getattr(request, '_workspace_user_id', None)
        if user_id:
            response['Cache-Control'] = 'no-store'
        if not user_id:
            # Login creates/rotates the session before its response is saved.
            if hasattr(request, 'user') and request.user.is_authenticated and request.session.session_key:
                IdleSession.objects.get_or_create(session_key=request.session.session_key,
                    defaults={'user_id': request.user.pk})
            return response
        if not request.user.is_authenticated:
            IdleSession.objects.filter(session_key=request._idle_key).delete()
            return response
        before = request._workspace_before
        after = workspace_snapshot(request.session)
        changed = {key for key in WORKSPACE_KEYS if before.get(key) != after.get(key) or (key in before) != (key in after)}
        if changed:
            with transaction.atomic():
                saved = SavedWorkspace.objects.select_for_update().get(user_id=user_id)
                state = dict(saved.state)
                for key in changed:
                    if key in after:
                        state[key] = after[key]
                    else:
                        state[key] = None
                saved.state = state
                saved.save(update_fields=['state'])
        cutoff = timezone.now() - timedelta(seconds=settings.IDLE_TIMEOUT_SECONDS)
        active = IdleSession.objects.filter(session_key=request._idle_key, user_id=user_id,
            last_activity__gt=cutoff).exists()
        if not active or not Session.objects.filter(session_key=request._idle_key).exists():
            # Do not let an old, long-running response overwrite a newer login cookie.
            Session.objects.filter(session_key=request._idle_key).delete()
            IdleSession.objects.filter(session_key=request._idle_key).delete()
            request.session.modified = False
            request.session.accessed = False
            return expired_response(request)
        return response


def session_status(request):
    if not request.user.is_authenticated:
        return JsonResponse({'expired': True, 'redirect': '/'}, status=401)
    record = IdleSession.objects.filter(session_key=request.session.session_key, user_id=request.user.pk).first()
    if record is None:
        return JsonResponse({'expired': True, 'redirect': '/'}, status=401)
    remaining = settings.IDLE_TIMEOUT_SECONDS - (timezone.now() - record.last_activity).total_seconds()
    response = JsonResponse({'remaining_seconds': max(0, remaining),
        'warning_remaining_seconds': settings.IDLE_WARNING_SECONDS})
    response['Cache-Control'] = 'no-store'
    return response


@require_GET
def status(request):
    return session_status(request)


@require_POST
def activity(request):
    if not request.user.is_authenticated:
        return JsonResponse({'expired': True, 'redirect': '/'}, status=401)
    draft = request.POST.get('draft')
    studio = request.POST.get('studio')
    if draft is not None:
        if studio not in ('llm1', 'llm2') or len(draft) > 10000:
            return JsonResponse({'error': 'Invalid project draft.'}, status=400)
        request.session[studio + '_draft'] = draft
    cutoff = timezone.now() - timedelta(seconds=settings.IDLE_TIMEOUT_SECONDS)
    updated = IdleSession.objects.filter(session_key=request.session.session_key, user_id=request.user.pk,
        last_activity__gt=cutoff).update(last_activity=timezone.now())
    if not updated:
        logout(request)
        return expired_response(request)
    return session_status(request)
