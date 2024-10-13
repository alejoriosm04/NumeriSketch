from django.urls import path
from . import views

urlpatterns = [
    path('', views.gauss_seidel, name='gauss_seidel'),
]
