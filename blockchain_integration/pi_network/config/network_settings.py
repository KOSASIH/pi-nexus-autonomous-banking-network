import json
from pi_network.utils.cryptographic_helpers import encrypt_data, decrypt_data

class NetworkSettings:
    def __init__(self):
        self.settings_file = os.path.join(os.path.dirname(__file__), 'network_settings.json')
        self.settings_data = self.load_settings_data()

    def load_settings_data(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                settings_data = json.load(f)
                return settings_data
        else:
            return {}

    def get_setting(self, key: str):
        if key in self.settings_data:
            encrypted_value = self.settings_data[key]
            decrypted_value = decrypt_data(encrypted_value)
            return decrypted_value
        else:
            return None

    def set_setting(self, key: str, value: str):
        encrypted_value = encrypt_data(value)
        self.settings_data[key] = encrypted_value
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings_data, f)

    def delete_setting(self, key: str):
        if key in self.settings_data:
            del self.settings_data[key]
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings_data, f)

network_settings = NetworkSettings()

def get_network_setting(key: str):
    return network_settings.get_setting(key)

def set_network_setting(key: str, value: str):
    network_settings.set_setting(key, value)

def delete_network_setting(key: str):
    network_settings.delete_setting(key)
