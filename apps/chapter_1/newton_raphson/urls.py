# apps/chapter_1/newton_raphson/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.newton_raphson, name='newton_raphson'),
]