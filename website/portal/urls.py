"""
URL configuration for portal project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import include, path
from workspace import views
from workspace import llm2_views

urlpatterns = [
    path('accounts/', include('allauth.urls')),
    path('', views.home), path('chat/', views.chat), path('report/', views.report),
    path('new/', views.new_chat), path('download/', views.download), path('progress/', views.progress),
    path('llm2/', llm2_views.home), path('llm2/upload/', llm2_views.upload),
    path('llm2/chat/', llm2_views.chat), path('llm2/report/', llm2_views.report),
    path('llm2/download/', llm2_views.download), path('llm2/new/', llm2_views.new),
]
