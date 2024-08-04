# kyc_models.py

from django.db import models
from django.contrib.auth.models import AbstractUser
from .utils import generate_kyc_id

class KYCUser(AbstractUser):
    kyc_id = models.CharField(max_length=20, unique=True, default=generate_kyc_id)
    id_card_number = models.CharField(max_length=20, blank=True, null=True)
    id_card_type = models.CharField(max_length=10, choices=[('passport', 'Passport'), ('id_card', 'ID Card'), ('driver_license', 'Driver License')], blank=True, null=True)
    face_verification_status = models.CharField(max_length=10, choices=[('pending', 'Pending'), ('verified', 'Verified'), ('rejected', 'Rejected')], default='pending')
    document_verification_status = models.CharField(max_length=10, choices=[('pending', 'Pending'), ('verified', 'Verified'), ('rejected', 'Rejected')], default='pending')
    biometric_verification_status = models.CharField(max_length=10, choices=[('pending', 'Pending'), ('verified', 'Verified'), ('rejected', 'Rejected')], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class IDCard(models.Model):
    user = models.ForeignKey(KYCUser, on_delete=models.CASCADE)
    id_card_number = models.CharField(max_length=20)
    id_card_type = models.CharField(max_length=10, choices=[('passport', 'Passport'), ('id_card', 'ID Card'), ('driver_license', 'Driver License')])
    issue_date = models.DateField()
    expiration_date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class FaceVerification(models.Model):
    user = models.ForeignKey(KYCUser, on_delete=models.CASCADE)
    face_image = models.ImageField(upload_to='face_images')
    verification_status = models.CharField(max_length=10, choices=[('pending', 'Pending'), ('verified', 'Verified'), ('rejected', 'Rejected')], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class DocumentVerification(models.Model):
    user = models.ForeignKey(KYCUser, on_delete=models.CASCADE)
    document_type = models.CharField(max_length=10, choices=[('utility_bill', 'Utility Bill'), ('bank_statement', 'Bank Statement')])
    document_image = models.ImageField(upload_to='document_images')
    verification_status = models.CharField(max_length=10, choices=[('pending', 'Pending'), ('verified', 'Verified'), ('rejected', 'Rejected')], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class BiometricVerification(models.Model):
    user = models.ForeignKey(KYCUser, on_delete=models.CASCADE)
    biometric_data = models.TextField()
    verification_status = models.CharField(max_length=10, choices=[('pending', 'Pending'), ('verified', 'Verified'), ('rejected', 'Rejected')], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
