from django.urls import path

from .views import JobListCreateView, JobRetrieveUpdateDestroyView

urlpatterns = [
    path("jobs/", JobListCreateView.as_view(), name="job_list_create"),
    path(
        "jobs/<int:pk>/",
        JobRetrieveUpdateDestroyView.as_view(),
        name="job_retrieve_update_destroy",
    ),
]
