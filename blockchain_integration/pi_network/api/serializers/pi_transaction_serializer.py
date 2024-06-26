# pi_transaction_serializer.py
from rest_framework import serializers
from.models import PITransaction

class PITransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PITransaction
        fields = ['tx_id', 'ender', 'ecipient', 'amount', 'timestamp', 'data']
