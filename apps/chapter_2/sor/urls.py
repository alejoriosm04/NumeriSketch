from django.urls import path
from . import views

urlpatterns = [
    path('', views.sor, name='sor'),
]
