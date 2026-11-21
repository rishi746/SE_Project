from django.conf import settings
from django.db import models
from django.utils import timezone


class SavedWorkspace(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    state = models.JSONField(default=dict)


class IdleSession(models.Model):
    session_key = models.CharField(max_length=40, primary_key=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    last_activity = models.DateTimeField(default=timezone.now)
