# github_config.py

import os
import json
from cryptography.fernet import Fernet

class GithubConfig:
    def __init__(self):
        self.config_file = 'github_config.json'
        self.credential_file = 'github_credentials.json'
        self.encrypted_file = 'github_credentials.encrypted'
        self.api_url = 'https://api.github.com'
        self.api_version = 'v3'
        self.user_agent = 'Pi-Nexus-Company-Analyzer'
        self.rate_limit = 5000  # 5000 requests per hour
        self.rate_limit_window = 3600  # 1 hour window

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        else:
            return {}

    def save_config(self, config):
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def load_credentials(self):
        if os.path.exists(self.credential_file):
            with open(self.credential_file, 'r') as f:
                credentials = json.load(f)
            return credentials
        else:
            return {}

    def save_credentials(self, credentials):
        with open(self.credential_file, 'w') as f:
            json.dump(credentials, f, indent=4)

    def encrypt_credentials(self, credentials):
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        encrypted_credentials = cipher_suite.encrypt(json.dumps(credentials).encode())
        with open(self.encrypted_file, 'wb') as f:
            f.write(encrypted_credentials)
        return key

    def decrypt_credentials(self, key):
        cipher_suite = Fernet(key)
        with open(self.encrypted_file, 'rb') as f:
            encrypted_credentials = f.read()
        decrypted_credentials = cipher_suite.decrypt(encrypted_credentials)
        return json.loads(decrypted_credentials.decode())

    def get_api_token(self):
        config = self.load_config()
        if 'api_token' in config:
            return config['api_token']
        else:
            return None

    def set_api_token(self, api_token):
        config = self.load_config()
        config['api_token'] = api_token
        self.save_config(config)

    def get_username(self):
        config = self.load_config()
        if 'username' in config:
            return config['username']
        else:
            return None

    def set_username(self, username):
        config = self.load_config()
        config['username'] = username
        self.save_config(config)

    def get_password(self):
        config = self.load_config()
        if 'password' in config:
            return config['password']
        else:
            return None

    def set_password(self, password):
        config = self.load_config()
        config['password'] = password
        self.save_config(config)

    def get_oauth_token(self):
        config = self.load_config()
        if 'oauth_token' in config:
            return config['oauth_token']
        else:
            return None

    def set_oauth_token(self, oauth_token):
        config = self.load_config()
        config['oauth_token'] = oauth_token
        self.save_config(config)

github_config = GithubConfig()
