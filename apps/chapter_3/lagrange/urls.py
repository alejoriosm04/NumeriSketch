from django.urls import path
from . import views

urlpatterns = [
    path('', views.lagrange_view, name='lagrange'),
]
