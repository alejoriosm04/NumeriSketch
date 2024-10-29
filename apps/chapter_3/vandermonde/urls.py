from django.urls import path
from . import views

urlpatterns = [
    path('', views.vandermonde, name='vandermonde'),
]
