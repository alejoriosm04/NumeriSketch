from django.urls import path
from . import views

urlpatterns = [
    path('', views.jacobi_view, name='jacobi'),
]
