# utility_creation/utility_models.py

from django.db import models
from django.contrib.auth.models import AbstractUser

class UtilityUser(AbstractUser):
    utility_id = models.CharField(max_length=20, unique=True)
    mainnet_address = models.CharField(max_length=42)
    testnet_address = models.CharField(max_length=42)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class UtilityContract(models.Model):
    contract_address = models.CharField(max_length=42)
    contract_abi = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class UtilityTransaction(models.Model):
    user = models.ForeignKey(UtilityUser, on_delete=models.CASCADE)
    transaction_hash = models.CharField(max_length=66)
    transaction_status = models.CharField(max_length=10, choices=[('pending', 'Pending'), ('success', 'Success'), ('failed', 'Failed')], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
