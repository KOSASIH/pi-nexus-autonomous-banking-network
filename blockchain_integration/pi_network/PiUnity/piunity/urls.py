# piunity/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('utility/', include('utility_creation.urls')),
]
