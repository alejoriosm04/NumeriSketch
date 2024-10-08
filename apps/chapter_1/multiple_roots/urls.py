from django.urls import path
from . import views

urlpatterns = [
    path('', views.multiple_roots, name='multiple_roots'),
]
