from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Account, Transaction
from .exceptions import ValidationError

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    """
    A serializer for the User model.
    """
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'is_staff')

class AccountSerializer(serializers.ModelSerializer):
    """
    A serializer for the Account model.
    """
    class Meta:
        model = Account
        fields = ('id', 'account_number', 'balance', 'user')

    def validate(self, data):
        """
        A validation method for the Account serializer.
        """
        user = self.context['request'].user
        if data['user'].id != user.id:
            raise ValidationError('You do not have permission to create an account for another user.')
        return data

class TransactionSerializer(serializers.ModelSerializer):
    """
    A serializer for the Transaction model.
    """
    class Meta:
        model = Transaction
        fields = ('id', 'transaction_type', 'amount', 'account_from', 'account_to', 'timestamp')

    def validate(self, data):
        """
        A validation method for the Transaction serializer.
        """
        account_from = Account.objects.get(account_number=data['account_from'])
        account_to = Account.objects.get(account_number=data['account_to'])
        if account_from.user != account_to.user:
            raise ValidationError('Transactions can only be made between accounts owned by the same user.')
        if data['transaction_type'] == 'debit' and data['amount'] > account_from.balance:
            raise ValidationError('Insufficient funds.')
        return data
