from django.urls import path
from . import views

urlpatterns = [
    path('', views.fixed_point, name='fixed_point'),
]


