import os
import json

class Config:
    def __init__(self):
        self.config_file = 'config.json'
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'api_url': 'https://example.com/api',
                'api_key': 'YOUR_API_KEY',
                'database_url': 'sqlite:///database.db',
                'logging_level': 'INFO'
            }
            self.save_config()

    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get(self, key):
        return self.config.get(key)

config = Config()
