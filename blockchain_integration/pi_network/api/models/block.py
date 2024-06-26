# block.py
from django.db import models
import hashlib
import time

class Block(models.Model):
    index = models.IntegerField(unique=True)
    previous_hash = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    transactions = models.ManyToManyField('PITransaction')
    nonce = models.IntegerField(default=0)
    hash = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f"Block {self.index}"

    def calculate_hash(self):
        # Calculate the block's hash using the previous hash, transactions, and nonce
        pass
