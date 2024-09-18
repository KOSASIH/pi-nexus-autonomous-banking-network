import os
import json

class SecretsManager:
    def __init__(self, secrets_file):
        self.secrets_file = secrets_file
        self.secrets = self.load_secrets()

    def load_secrets(self):
        if os.path.exists(self.secrets_file):
            with open(self.secrets_file, 'r') as file:
                return json.load(file)
        else:
            return {}

    def save_secrets(self):
        with open(self.secrets_file, 'w') as file:
            json.dump(self.secrets, file)

    def get_secret(self, key):
        return self.secrets.get(key)

    def set_secret(self, key, value):
        self.secrets[key] = value
        self.save_secrets()
