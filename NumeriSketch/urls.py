"""
URL configuration for NumeriSketch project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views 
urlpatterns = [
    path('', views.home_view, name='home'),
    path('admin/', admin.site.urls),
    path('bisection/', include('apps.chapter_1.bisection.urls')),
    path('fixed_point/', include('apps.chapter_1.fixed_point.urls')),
    # path('false_position/', include('apps.chapter_1.false_position.urls')),
    path('newton_raphson/', include('apps.chapter_1.newton_raphson.urls')),
    path('multiple_roots/', include('apps.chapter_1.multiple_roots.urls')),
]
