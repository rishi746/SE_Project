from django.conf import settings
def auth_options(request):
    return {'google_enabled': settings.GOOGLE_ENABLED,
        'llm2_site': request.get_port() == '8001' or request.path.startswith('/llm2/')}
