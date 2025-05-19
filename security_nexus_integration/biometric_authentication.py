"""
Biometric Authentication Module for Pi-Nexus Autonomous Banking Network

This module implements multi-modal biometric authentication, continuous behavioral biometrics,
and decentralized biometric identity verification for the banking network.
"""

import os
import time
import uuid
import hashlib
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime

# Import the quantum-resistant cryptography module
import sys
sys.path.append('/workspace/pi-nexus-autonomous-banking-network')
from quantum_nexus_integration.quantum_resistant_cryptography import QuantumResistantCrypto, QUANTUM_SECURITY_LEVEL_2


class BiometricTemplate:
    """
    Represents a biometric template for a specific modality.
    
    This class encapsulates the biometric data and metadata for a specific
    biometric modality (e.g., fingerprint, face, voice, etc.).
    """
    
    def __init__(self, 
                user_id: str,
                modality: str,
                template_data: bytes,
                quality_score: float,
                creation_timestamp: Optional[datetime] = None):
        """
        Initialize a biometric template.
        
        Args:
            user_id: The ID of the user this template belongs to
            modality: The biometric modality (e.g., 'fingerprint', 'face', 'voice')
            template_data: The biometric template data (encrypted)
            quality_score: The quality score of the template (0.0 to 1.0)
            creation_timestamp: The timestamp when the template was created
        """
        self.template_id = str(uuid.uuid4())
        self.user_id = user_id
        self.modality = modality
        self.template_data = template_data
        self.quality_score = quality_score
        self.creation_timestamp = creation_timestamp or datetime.utcnow()
        self.last_used_timestamp = None
        self.use_count = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the template to a dictionary.
        
        Returns:
            A dictionary representation of the template
        """
        return {
            'template_id': self.template_id,
            'user_id': self.user_id,
            'modality': self.modality,
            'template_data': self.template_data.hex(),
            'quality_score': self.quality_score,
            'creation_timestamp': self.creation_timestamp.isoformat(),
            'last_used_timestamp': self.last_used_timestamp.isoformat() if self.last_used_timestamp else None,
            'use_count': self.use_count
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiometricTemplate':
        """
        Create a template from a dictionary.
        
        Args:
            data: A dictionary representation of the template
            
        Returns:
            A BiometricTemplate instance
        """
        template = cls(
            user_id=data['user_id'],
            modality=data['modality'],
            template_data=bytes.fromhex(data['template_data']),
            quality_score=data['quality_score'],
            creation_timestamp=datetime.fromisoformat(data['creation_timestamp'])
        )
        
        template.template_id = data['template_id']
        template.use_count = data['use_count']
        
        if data['last_used_timestamp']:
            template.last_used_timestamp = datetime.fromisoformat(data['last_used_timestamp'])
            
        return template
        
    def update_usage(self) -> None:
        """
        Update the usage statistics for this template.
        """
        self.last_used_timestamp = datetime.utcnow()
        self.use_count += 1


class BiometricUser:
    """
    Represents a user with multiple biometric templates.
    
    This class encapsulates a user's biometric identity, including templates
    for multiple biometric modalities and behavioral biometric profiles.
    """
    
    def __init__(self, 
                user_id: str,
                username: str,
                email: Optional[str] = None,
                phone: Optional[str] = None):
        """
        Initialize a biometric user.
        
        Args:
            user_id: The unique ID of the user
            username: The username of the user
            email: The email address of the user
            phone: The phone number of the user
        """
        self.user_id = user_id
        self.username = username
        self.email = email
        self.phone = phone
        self.templates = {}  # Mapping from template_id to BiometricTemplate
        self.behavioral_profiles = {}  # Mapping from profile_type to behavioral profile data
        self.creation_timestamp = datetime.utcnow()
        self.last_authentication_timestamp = None
        self.authentication_count = 0
        self.failed_authentication_count = 0
        
    def add_template(self, template: BiometricTemplate) -> None:
        """
        Add a biometric template for this user.
        
        Args:
            template: The biometric template to add
        """
        self.templates[template.template_id] = template
        
    def remove_template(self, template_id: str) -> bool:
        """
        Remove a biometric template for this user.
        
        Args:
            template_id: The ID of the template to remove
            
        Returns:
            True if the template was removed, False otherwise
        """
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False
        
    def get_templates_by_modality(self, modality: str) -> List[BiometricTemplate]:
        """
        Get all templates for a specific modality.
        
        Args:
            modality: The biometric modality to get templates for
            
        Returns:
            A list of templates for the specified modality
        """
        return [template for template in self.templates.values() if template.modality == modality]
        
    def add_behavioral_profile(self, profile_type: str, profile_data: bytes) -> None:
        """
        Add a behavioral biometric profile for this user.
        
        Args:
            profile_type: The type of behavioral profile (e.g., 'typing', 'mouse', 'gait')
            profile_data: The behavioral profile data
        """
        self.behavioral_profiles[profile_type] = profile_data
        
    def remove_behavioral_profile(self, profile_type: str) -> bool:
        """
        Remove a behavioral biometric profile for this user.
        
        Args:
            profile_type: The type of behavioral profile to remove
            
        Returns:
            True if the profile was removed, False otherwise
        """
        if profile_type in self.behavioral_profiles:
            del self.behavioral_profiles[profile_type]
            return True
        return False
        
    def update_authentication_stats(self, success: bool) -> None:
        """
        Update the authentication statistics for this user.
        
        Args:
            success: Whether the authentication was successful
        """
        self.last_authentication_timestamp = datetime.utcnow()
        
        if success:
            self.authentication_count += 1
        else:
            self.failed_authentication_count += 1
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the user to a dictionary.
        
        Returns:
            A dictionary representation of the user
        """
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'phone': self.phone,
            'templates': {template_id: template.to_dict() for template_id, template in self.templates.items()},
            'behavioral_profiles': {profile_type: profile_data.hex() for profile_type, profile_data in self.behavioral_profiles.items()},
            'creation_timestamp': self.creation_timestamp.isoformat(),
            'last_authentication_timestamp': self.last_authentication_timestamp.isoformat() if self.last_authentication_timestamp else None,
            'authentication_count': self.authentication_count,
            'failed_authentication_count': self.failed_authentication_count
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiometricUser':
        """
        Create a user from a dictionary.
        
        Args:
            data: A dictionary representation of the user
            
        Returns:
            A BiometricUser instance
        """
        user = cls(
            user_id=data['user_id'],
            username=data['username'],
            email=data['email'],
            phone=data['phone']
        )
        
        user.creation_timestamp = datetime.fromisoformat(data['creation_timestamp'])
        user.authentication_count = data['authentication_count']
        user.failed_authentication_count = data['failed_authentication_count']
        
        if data['last_authentication_timestamp']:
            user.last_authentication_timestamp = datetime.fromisoformat(data['last_authentication_timestamp'])
            
        for template_data in data['templates'].values():
            template = BiometricTemplate.from_dict(template_data)
            user.templates[template.template_id] = template
            
        for profile_type, profile_data_hex in data['behavioral_profiles'].items():
            user.behavioral_profiles[profile_type] = bytes.fromhex(profile_data_hex)
            
        return user


class BiometricAuthenticator:
    """
    Implements multi-modal biometric authentication.
    
    This class provides methods for enrolling users, verifying biometric samples,
    and managing biometric templates and user identities.
    """
    
    def __init__(self, security_level=QUANTUM_SECURITY_LEVEL_2):
        """
        Initialize the biometric authenticator.
        
        Args:
            security_level: The security level to use for cryptographic operations
        """
        self.security_level = security_level
        self.crypto = QuantumResistantCrypto(security_level=security_level)
        self.users = {}  # Mapping from user_id to BiometricUser
        self.templates_by_user = {}  # Mapping from user_id to list of template_ids
        self.modality_thresholds = {
            'fingerprint': 0.85,
            'face': 0.80,
            'voice': 0.75,
            'iris': 0.90,
            'retina': 0.95,
            'palm': 0.85,
            'gait': 0.70,
            'typing': 0.75,
            'mouse': 0.70
        }
        
    def _generate_user_id(self) -> str:
        """
        Generate a unique user ID.
        
        Returns:
            A unique user ID
        """
        return str(uuid.uuid4())
        
    def _encrypt_template(self, template_data: bytes, user_id: str) -> bytes:
        """
        Encrypt a biometric template.
        
        Args:
            template_data: The raw biometric template data
            user_id: The ID of the user the template belongs to
            
        Returns:
            The encrypted template data
        """
        # In a real implementation, this would use a secure encryption scheme
        # For now, we'll use the quantum-resistant crypto module
        
        # Derive a key from the user ID
        key, salt = self.crypto.generate_symmetric_key(user_id.encode())
        
        # Encrypt the template data
        iv, ciphertext, tag = self.crypto.symmetric_encrypt(key, template_data)
        
        # Combine the IV, ciphertext, and tag for storage
        return salt + iv + tag + ciphertext
        
    def _decrypt_template(self, encrypted_data: bytes, user_id: str) -> bytes:
        """
        Decrypt a biometric template.
        
        Args:
            encrypted_data: The encrypted biometric template data
            user_id: The ID of the user the template belongs to
            
        Returns:
            The decrypted template data
        """
        # Extract the salt, IV, tag, and ciphertext
        salt = encrypted_data[:32]
        iv = encrypted_data[32:48]
        tag = encrypted_data[48:64]
        ciphertext = encrypted_data[64:]
        
        # Derive the key from the user ID and salt
        key, _ = self.crypto.generate_symmetric_key(user_id.encode(), salt)
        
        # Decrypt the template data
        plaintext = self.crypto.symmetric_decrypt(key, iv, ciphertext, tag)
        
        return plaintext
        
    def _simulate_biometric_matching(self, template_data: bytes, sample_data: bytes, modality: str) -> float:
        """
        Simulate biometric matching between a template and a sample.
        
        In a real implementation, this would use actual biometric matching algorithms.
        
        Args:
            template_data: The template data to match against
            sample_data: The sample data to match
            modality: The biometric modality
            
        Returns:
            A match score between 0.0 and 1.0
        """
        # This is a simulation of biometric matching
        # In a real implementation, this would use actual biometric matching algorithms
        
        # For simulation purposes, we'll generate a random match score
        # with a bias towards high scores for "genuine" matches
        
        # Simulate a "genuine" match with 80% probability
        if random.random() < 0.8:
            # Generate a high match score (0.7 to 1.0)
            return random.uniform(0.7, 1.0)
        else:
            # Generate a low match score (0.0 to 0.7)
            return random.uniform(0.0, 0.7)
        
    def enroll_user(self, username: str, email: Optional[str] = None, phone: Optional[str] = None) -> str:
        """
        Enroll a new user in the biometric system.
        
        Args:
            username: The username of the user
            email: The email address of the user
            phone: The phone number of the user
            
        Returns:
            The ID of the newly enrolled user
        """
        user_id = self._generate_user_id()
        
        user = BiometricUser(
            user_id=user_id,
            username=username,
            email=email,
            phone=phone
        )
        
        self.users[user_id] = user
        self.templates_by_user[user_id] = []
        
        return user_id
        
    def add_biometric_template(self, user_id: str, modality: str, template_data: bytes) -> str:
        """
        Add a biometric template for a user.
        
        Args:
            user_id: The ID of the user
            modality: The biometric modality
            template_data: The raw biometric template data
            
        Returns:
            The ID of the newly added template
        """
        if user_id not in self.users:
            raise ValueError(f"User with ID {user_id} not found")
            
        # Encrypt the template data
        encrypted_data = self._encrypt_template(template_data, user_id)
        
        # Create a new template
        template = BiometricTemplate(
            user_id=user_id,
            modality=modality,
            template_data=encrypted_data,
            quality_score=random.uniform(0.8, 1.0)  # Simulate a quality score
        )
        
        # Add the template to the user
        self.users[user_id].add_template(template)
        
        # Add the template ID to the templates_by_user mapping
        self.templates_by_user[user_id].append(template.template_id)
        
        return template.template_id
        
    def remove_biometric_template(self, user_id: str, template_id: str) -> bool:
        """
        Remove a biometric template for a user.
        
        Args:
            user_id: The ID of the user
            template_id: The ID of the template to remove
            
        Returns:
            True if the template was removed, False otherwise
        """
        if user_id not in self.users:
            return False
            
        if template_id not in self.templates_by_user[user_id]:
            return False
            
        # Remove the template from the user
        result = self.users[user_id].remove_template(template_id)
        
        # Remove the template ID from the templates_by_user mapping
        if result:
            self.templates_by_user[user_id].remove(template_id)
            
        return result
        
    def verify_biometric(self, user_id: str, modality: str, sample_data: bytes) -> Tuple[bool, float]:
        """
        Verify a biometric sample against a user's templates.
        
        Args:
            user_id: The ID of the user
            modality: The biometric modality
            sample_data: The biometric sample data
            
        Returns:
            A tuple containing (success, confidence)
        """
        if user_id not in self.users:
            return False, 0.0
            
        user = self.users[user_id]
        
        # Get all templates for the specified modality
        templates = user.get_templates_by_modality(modality)
        
        if not templates:
            return False, 0.0
            
        # Find the best match among all templates
        best_match_score = 0.0
        best_match_template = None
        
        for template in templates:
            # Decrypt the template data
            decrypted_data = self._decrypt_template(template.template_data, user_id)
            
            # Match the sample against the template
            match_score = self._simulate_biometric_matching(decrypted_data, sample_data, modality)
            
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_template = template
                
        # Check if the best match score exceeds the threshold for this modality
        threshold = self.modality_thresholds.get(modality, 0.8)
        success = best_match_score >= threshold
        
        # Update usage statistics for the matched template
        if success and best_match_template:
            best_match_template.update_usage()
            
        # Update authentication statistics for the user
        user.update_authentication_stats(success)
        
        return success, best_match_score
        
    def multi_factor_authenticate(self, user_id: str, 
                                 modality_samples: Dict[str, bytes],
                                 required_modalities: Optional[List[str]] = None,
                                 min_modalities: int = 2) -> Tuple[bool, Dict[str, Tuple[bool, float]]]:
        """
        Perform multi-factor biometric authentication.
        
        Args:
            user_id: The ID of the user
            modality_samples: A dictionary mapping modalities to sample data
            required_modalities: A list of modalities that must be verified
            min_modalities: The minimum number of modalities that must be verified
            
        Returns:
            A tuple containing (success, results)
        """
        if user_id not in self.users:
            return False, {}
            
        # Verify each modality
        results = {}
        successful_modalities = 0
        
        for modality, sample_data in modality_samples.items():
            success, confidence = self.verify_biometric(user_id, modality, sample_data)
            results[modality] = (success, confidence)
            
            if success:
                successful_modalities += 1
                
        # Check if all required modalities were verified
        if required_modalities:
            for modality in required_modalities:
                if modality not in modality_samples or not results.get(modality, (False, 0.0))[0]:
                    return False, results
                    
        # Check if the minimum number of modalities were verified
        if successful_modalities < min_modalities:
            return False, results
            
        return True, results
        
    def add_behavioral_profile(self, user_id: str, profile_type: str, profile_data: bytes) -> bool:
        """
        Add a behavioral biometric profile for a user.
        
        Args:
            user_id: The ID of the user
            profile_type: The type of behavioral profile
            profile_data: The behavioral profile data
            
        Returns:
            True if the profile was added, False otherwise
        """
        if user_id not in self.users:
            return False
            
        # Encrypt the profile data
        encrypted_data = self._encrypt_template(profile_data, user_id)
        
        # Add the profile to the user
        self.users[user_id].add_behavioral_profile(profile_type, encrypted_data)
        
        return True
        
    def verify_behavioral_biometric(self, user_id: str, profile_type: str, sample_data: bytes) -> Tuple[bool, float]:
        """
        Verify a behavioral biometric sample against a user's profile.
        
        Args:
            user_id: The ID of the user
            profile_type: The type of behavioral profile
            sample_data: The behavioral sample data
            
        Returns:
            A tuple containing (success, confidence)
        """
        if user_id not in self.users:
            return False, 0.0
            
        user = self.users[user_id]
        
        if profile_type not in user.behavioral_profiles:
            return False, 0.0
            
        # Decrypt the profile data
        encrypted_data = user.behavioral_profiles[profile_type]
        decrypted_data = self._decrypt_template(encrypted_data, user_id)
        
        # Match the sample against the profile
        match_score = self._simulate_biometric_matching(decrypted_data, sample_data, profile_type)
        
        # Check if the match score exceeds the threshold for this profile type
        threshold = self.modality_thresholds.get(profile_type, 0.7)
        success = match_score >= threshold
        
        # Update authentication statistics for the user
        user.update_authentication_stats(success)
        
        return success, match_score
        
    def continuous_authentication(self, user_id: str, 
                                 behavioral_samples: Dict[str, bytes],
                                 window_size: int = 10,
                                 threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Perform continuous behavioral authentication.
        
        Args:
            user_id: The ID of the user
            behavioral_samples: A dictionary mapping profile types to sample data
            window_size: The number of samples to consider in the authentication window
            threshold: The threshold for successful authentication
            
        Returns:
            A tuple containing (success, confidence)
        """
        if user_id not in self.users:
            return False, 0.0
            
        # Verify each behavioral profile
        scores = []
        
        for profile_type, sample_data in behavioral_samples.items():
            success, confidence = self.verify_behavioral_biometric(user_id, profile_type, sample_data)
            scores.append(confidence)
            
        # Calculate the overall confidence score
        if not scores:
            return False, 0.0
            
        overall_confidence = sum(scores) / len(scores)
        
        # Check if the overall confidence exceeds the threshold
        success = overall_confidence >= threshold
        
        return success, overall_confidence
        
    def get_user(self, user_id: str) -> Optional[BiometricUser]:
        """
        Get a user by ID.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            The user if found, None otherwise
        """
        return self.users.get(user_id)
        
    def get_user_by_username(self, username: str) -> Optional[BiometricUser]:
        """
        Get a user by username.
        
        Args:
            username: The username of the user
            
        Returns:
            The user if found, None otherwise
        """
        for user in self.users.values():
            if user.username == username:
                return user
                
        return None
        
    def get_user_templates(self, user_id: str) -> List[BiometricTemplate]:
        """
        Get all templates for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            A list of templates for the user
        """
        if user_id not in self.users:
            return []
            
        return list(self.users[user_id].templates.values())
        
    def get_user_behavioral_profiles(self, user_id: str) -> List[str]:
        """
        Get all behavioral profile types for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            A list of behavioral profile types for the user
        """
        if user_id not in self.users:
            return []
            
        return list(self.users[user_id].behavioral_profiles.keys())


class DecentralizedBiometricIdentity:
    """
    Implements decentralized biometric identity verification.
    
    This class provides methods for creating and verifying decentralized
    biometric identities using blockchain and zero-knowledge proofs.
    """
    
    def __init__(self, security_level=QUANTUM_SECURITY_LEVEL_2):
        """
        Initialize the decentralized biometric identity system.
        
        Args:
            security_level: The security level to use for cryptographic operations
        """
        self.security_level = security_level
        self.crypto = QuantumResistantCrypto(security_level=security_level)
        self.identities = {}  # Mapping from identity_id to identity data
        self.identity_proofs = {}  # Mapping from identity_id to list of proofs
        
    def _generate_identity_id(self) -> str:
        """
        Generate a unique identity ID.
        
        Returns:
            A unique identity ID
        """
        return str(uuid.uuid4())
        
    def _hash_biometric_data(self, biometric_data: bytes) -> bytes:
        """
        Create a secure hash of biometric data.
        
        Args:
            biometric_data: The biometric data to hash
            
        Returns:
            The hash value
        """
        return self.crypto.hash(biometric_data)
        
    def _generate_zero_knowledge_proof(self, biometric_data: bytes, identity_id: str) -> bytes:
        """
        Generate a zero-knowledge proof for biometric data.
        
        In a real implementation, this would use actual zero-knowledge proof algorithms.
        
        Args:
            biometric_data: The biometric data to create a proof for
            identity_id: The ID of the identity
            
        Returns:
            The zero-knowledge proof
        """
        # This is a simulation of zero-knowledge proof generation
        # In a real implementation, this would use actual zero-knowledge proof algorithms
        
        # For simulation purposes, we'll just create a hash of the biometric data and identity ID
        combined_data = biometric_data + identity_id.encode()
        proof = self.crypto.hash(combined_data)
        
        return proof
        
    def _verify_zero_knowledge_proof(self, proof: bytes, biometric_data: bytes, identity_id: str) -> bool:
        """
        Verify a zero-knowledge proof for biometric data.
        
        In a real implementation, this would use actual zero-knowledge proof algorithms.
        
        Args:
            proof: The zero-knowledge proof to verify
            biometric_data: The biometric data to verify against
            identity_id: The ID of the identity
            
        Returns:
            True if the proof is valid, False otherwise
        """
        # This is a simulation of zero-knowledge proof verification
        # In a real implementation, this would use actual zero-knowledge proof algorithms
        
        # For simulation purposes, we'll just recreate the hash and compare
        combined_data = biometric_data + identity_id.encode()
        expected_proof = self.crypto.hash(combined_data)
        
        return proof == expected_proof
        
    def create_identity(self, biometric_data: Dict[str, bytes], metadata: Dict[str, Any]) -> str:
        """
        Create a new decentralized biometric identity.
        
        Args:
            biometric_data: A dictionary mapping modalities to biometric data
            metadata: Additional metadata for the identity
            
        Returns:
            The ID of the newly created identity
        """
        identity_id = self._generate_identity_id()
        
        # Hash the biometric data
        biometric_hashes = {modality: self._hash_biometric_data(data).hex()
                          for modality, data in biometric_data.items()}
        
        # Generate zero-knowledge proofs
        proofs = {modality: self._generate_zero_knowledge_proof(data, identity_id).hex()
                for modality, data in biometric_data.items()}
        
        # Create the identity
        identity = {
            'identity_id': identity_id,
            'biometric_hashes': biometric_hashes,
            'metadata': metadata,
            'creation_timestamp': datetime.utcnow().isoformat(),
            'last_verification_timestamp': None,
            'verification_count': 0
        }
        
        # Store the identity and proofs
        self.identities[identity_id] = identity
        self.identity_proofs[identity_id] = proofs
        
        return identity_id
        
    def verify_identity(self, identity_id: str, biometric_data: Dict[str, bytes]) -> Tuple[bool, Dict[str, bool]]:
        """
        Verify a decentralized biometric identity.
        
        Args:
            identity_id: The ID of the identity to verify
            biometric_data: A dictionary mapping modalities to biometric data
            
        Returns:
            A tuple containing (success, results)
        """
        if identity_id not in self.identities:
            return False, {}
            
        identity = self.identities[identity_id]
        proofs = self.identity_proofs[identity_id]
        
        # Verify each modality
        results = {}
        
        for modality, data in biometric_data.items():
            if modality not in identity['biometric_hashes'] or modality not in proofs:
                results[modality] = False
                continue
                
            # Verify the zero-knowledge proof
            proof = bytes.fromhex(proofs[modality])
            result = self._verify_zero_knowledge_proof(proof, data, identity_id)
            results[modality] = result
            
        # Check if all modalities were verified successfully
        success = all(results.values()) and len(results) > 0
        
        # Update verification statistics
        if success:
            identity['last_verification_timestamp'] = datetime.utcnow().isoformat()
            identity['verification_count'] += 1
            
        return success, results
        
    def get_identity(self, identity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an identity by ID.
        
        Args:
            identity_id: The ID of the identity
            
        Returns:
            The identity if found, None otherwise
        """
        return self.identities.get(identity_id)
        
    def update_identity_metadata(self, identity_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata for an identity.
        
        Args:
            identity_id: The ID of the identity
            metadata: The new metadata
            
        Returns:
            True if the metadata was updated, False otherwise
        """
        if identity_id not in self.identities:
            return False
            
        # Update the metadata
        self.identities[identity_id]['metadata'].update(metadata)
        
        return True
        
    def add_biometric_modality(self, identity_id: str, modality: str, biometric_data: bytes) -> bool:
        """
        Add a new biometric modality to an identity.
        
        Args:
            identity_id: The ID of the identity
            modality: The biometric modality
            biometric_data: The biometric data
            
        Returns:
            True if the modality was added, False otherwise
        """
        if identity_id not in self.identities:
            return False
            
        identity = self.identities[identity_id]
        
        # Hash the biometric data
        biometric_hash = self._hash_biometric_data(biometric_data).hex()
        
        # Generate a zero-knowledge proof
        proof = self._generate_zero_knowledge_proof(biometric_data, identity_id).hex()
        
        # Add the modality to the identity
        identity['biometric_hashes'][modality] = biometric_hash
        self.identity_proofs[identity_id][modality] = proof
        
        return True
        
    def remove_biometric_modality(self, identity_id: str, modality: str) -> bool:
        """
        Remove a biometric modality from an identity.
        
        Args:
            identity_id: The ID of the identity
            modality: The biometric modality to remove
            
        Returns:
            True if the modality was removed, False otherwise
        """
        if identity_id not in self.identities:
            return False
            
        identity = self.identities[identity_id]
        
        if modality not in identity['biometric_hashes']:
            return False
            
        # Remove the modality from the identity
        del identity['biometric_hashes'][modality]
        
        if modality in self.identity_proofs[identity_id]:
            del self.identity_proofs[identity_id][modality]
            
        return True


# Example usage
def example_usage():
    # Create a biometric authenticator
    authenticator = BiometricAuthenticator()
    
    # Enroll a user
    user_id = authenticator.enroll_user(
        username="alice",
        email="alice@example.com",
        phone="+1234567890"
    )
    
    print(f"Enrolled user: {user_id}")
    
    # Add biometric templates for the user
    fingerprint_template_id = authenticator.add_biometric_template(
        user_id=user_id,
        modality="fingerprint",
        template_data=os.urandom(1024)  # Simulated fingerprint template
    )
    
    face_template_id = authenticator.add_biometric_template(
        user_id=user_id,
        modality="face",
        template_data=os.urandom(2048)  # Simulated face template
    )
    
    voice_template_id = authenticator.add_biometric_template(
        user_id=user_id,
        modality="voice",
        template_data=os.urandom(4096)  # Simulated voice template
    )
    
    print(f"Added biometric templates: {fingerprint_template_id}, {face_template_id}, {voice_template_id}")
    
    # Add behavioral profiles for the user
    authenticator.add_behavioral_profile(
        user_id=user_id,
        profile_type="typing",
        profile_data=os.urandom(512)  # Simulated typing profile
    )
    
    authenticator.add_behavioral_profile(
        user_id=user_id,
        profile_type="mouse",
        profile_data=os.urandom(512)  # Simulated mouse profile
    )
    
    print(f"Added behavioral profiles: typing, mouse")
    
    # Verify biometric samples
    fingerprint_success, fingerprint_confidence = authenticator.verify_biometric(
        user_id=user_id,
        modality="fingerprint",
        sample_data=os.urandom(1024)  # Simulated fingerprint sample
    )
    
    face_success, face_confidence = authenticator.verify_biometric(
        user_id=user_id,
        modality="face",
        sample_data=os.urandom(2048)  # Simulated face sample
    )
    
    print(f"Fingerprint verification: {fingerprint_success} ({fingerprint_confidence:.2f})")
    print(f"Face verification: {face_success} ({face_confidence:.2f})")
    
    # Perform multi-factor authentication
    multi_factor_success, multi_factor_results = authenticator.multi_factor_authenticate(
        user_id=user_id,
        modality_samples={
            "fingerprint": os.urandom(1024),  # Simulated fingerprint sample
            "face": os.urandom(2048),  # Simulated face sample
            "voice": os.urandom(4096)  # Simulated voice sample
        },
        required_modalities=["fingerprint", "face"],
        min_modalities=2
    )
    
    print(f"Multi-factor authentication: {multi_factor_success}")
    for modality, (success, confidence) in multi_factor_results.items():
        print(f"  {modality}: {success} ({confidence:.2f})")
        
    # Perform continuous authentication
    continuous_success, continuous_confidence = authenticator.continuous_authentication(
        user_id=user_id,
        behavioral_samples={
            "typing": os.urandom(512),  # Simulated typing sample
            "mouse": os.urandom(512)  # Simulated mouse sample
        }
    )
    
    print(f"Continuous authentication: {continuous_success} ({continuous_confidence:.2f})")
    
    # Create a decentralized biometric identity
    decentralized = DecentralizedBiometricIdentity()
    
    identity_id = decentralized.create_identity(
        biometric_data={
            "fingerprint": os.urandom(1024),  # Simulated fingerprint data
            "face": os.urandom(2048),  # Simulated face data
            "voice": os.urandom(4096)  # Simulated voice data
        },
        metadata={
            "name": "Alice Smith",
            "email": "alice@example.com",
            "phone": "+1234567890",
            "country": "US"
        }
    )
    
    print(f"Created decentralized identity: {identity_id}")
    
    # Verify the decentralized identity
    identity_success, identity_results = decentralized.verify_identity(
        identity_id=identity_id,
        biometric_data={
            "fingerprint": os.urandom(1024),  # Simulated fingerprint data
            "face": os.urandom(2048),  # Simulated face data
            "voice": os.urandom(4096)  # Simulated voice data
        }
    )
    
    print(f"Decentralized identity verification: {identity_success}")
    for modality, success in identity_results.items():
        print(f"  {modality}: {success}")


if __name__ == "__main__":
    example_usage()