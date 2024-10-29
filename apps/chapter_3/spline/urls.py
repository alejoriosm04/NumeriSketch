from django.urls import path
from . import views

urlpatterns = [
    path('', views.spline, name='spline'),
]
