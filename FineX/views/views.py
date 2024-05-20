from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404
from rest_framework import generics, permissions

from .exceptions import ValidationError
from .models import Account, Transaction
from .serializers import AccountSerializer, TransactionSerializer, UserSerializer

User = get_user_model()


class UserListCreateView(generics.ListCreateAPIView):
    """
    A view for listing and creating users.
    """

    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAdminUser]


class UserRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    """
    A view for retrieving, updating, and deleting users.
    """

    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAdminUser]


class AccountListCreateView(generics.ListCreateAPIView):
    """
    A view for listing and creating accounts.
    """

    queryset = Account.objects.all()
    serializer_class = AccountSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        """
        A method for creating an account for the authenticated user.
        """
        user = self.request.user
        serializer.save(user=user)


class AccountRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    """
    A view for retrieving, updating, and deleting accounts.
    """

    queryset = Account.objects.all()
    serializer_class = AccountSerializer
    permission_classes = [permissions.IsAuthenticated]


class TransactionListCreateView(generics.ListCreateAPIView):
    """
    A view for listing and creating transactions.
    """

    queryset = Transaction.objects.all()
    serializer_class = TransactionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        """
        A method for creating a transaction for the authenticated user.
        """
        account_from = get_object_or_404(
            Account, account_number=self.request.data["account_from"]
        )
        account_to = get_object_or_404(
            Account, account_number=self.request.data["account_to"]
        )
        if self.request.data["transaction_type"] == "debit":
            if self.request.data["amount"] > account_from.balance:
                raise ValidationError("Insufficient funds.")
            account_from.balance -= self.request.data["amount"]
            account_from.save()
            account_to.balance += self.request.data["amount"]
            account_to.save()
        serializer.save(account_from=account_from, account_to=account_to)
