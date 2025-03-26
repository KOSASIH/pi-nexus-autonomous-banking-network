from django.urls import path

from .views import (
    AccountListCreateView,
    AccountRetrieveUpdateDestroyView,
    TransactionListCreateView,
    UserListCreateView,
    UserRetrieveUpdateDestroyView,
)

urlpatterns = [
    path("users/", UserListCreateView.as_view(), name="user_list_create"),
    path(
        "users/<int:pk>/",
        UserRetrieveUpdateDestroyView.as_view(),
        name="user_retrieve_update_destroy",
    ),
    path("accounts/", AccountListCreateView.as_view(), name="account_list_create"),
    path(
        "accounts/<int:pk>/",
        AccountRetrieveUpdateDestroyView.as_view(),
        name="account_retrieve_update_destroy",
    ),
    path(
        "transactions/",
        TransactionListCreateView.as_view(),
        name="transaction_list_create",
    ),
]
