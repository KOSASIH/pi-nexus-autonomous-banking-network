# pi_transaction.py
from django.db import models
from django.contrib.postgres.fields import JSONField

class PITransaction(models.Model):
    tx_id = models.CharField(max_length=255, unique=True)
    sender = models.CharField(max_length=255)
    recipient = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=20, decimal_places=8)
    timestamp = models.DateTimeField(auto_now_add=True)
    data = JSONField(default=dict)

    def __str__(self):
        return f"Tx {self.tx_id}: {self.sender} -> {self.recipient} ({self.amount})"
