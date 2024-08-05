# mainnet_migration/migration_models.py

from django.db import models
from django.contrib.auth.models import AbstractUser
from .utils import generate_migration_id

class MigrationUser(AbstractUser):
    migration_id = models.CharField(max_length=20, unique=True, default=generate_migration_id)
    mainnet_address = models.CharField(max_length=42)
    testnet_address = models.CharField(max_length=42)
    migration_status = models.CharField(max_length=10, choices=[('pending', 'Pending'), ('in_progress', 'In Progress'), ('completed', 'Completed')], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class MigrationTransaction(models.Model):
    user = models.ForeignKey(MigrationUser, on_delete=models.CASCADE)
    transaction_hash = models.CharField(max_length=66)
    transaction_status = models.CharField(max_length=10, choices=[('pending', 'Pending'), ('success', 'Success'), ('failed', 'Failed')], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class MigrationContract(models.Model):
    contract_address = models.CharField(max_length=42)
    contract_abi = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
