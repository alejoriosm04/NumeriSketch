from django.urls import path
from . import views

urlpatterns = [
    path('', views.falseposition_view, name='falseposition'),
]