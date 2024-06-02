import json
import os

from cryptography.fernet import Fernet


class EnvironmentVariables:
    def __init__(self):
        self.env_file = os.path.join(os.path.dirname(__file__), "env.json")
        self.env_data = self.load_env_data()

    def load_env_data(self):
        if os.path.exists(self.env_file):
            with open(self.env_file, "r") as f:
                env_data = json.load(f)
                return env_data
        else:
            return {}

    def get_variable(self, key: str):
        if key in self.env_data:
            encrypted_value = self.env_data[key]
            fernet = Fernet(os.environ["FERNET_KEY"])
            decrypted_value = fernet.decrypt(encrypted_value.encode()).decode()
            return decrypted_value
        else:
            return None

    def set_variable(self, key: str, value: str):
        fernet = Fernet(os.environ["FERNET_KEY"])
        encrypted_value = fernet.encrypt(value.encode()).decode()
        self.env_data[key] = encrypted_value
        with open(self.env_file, "w") as f:
            json.dump(self.env_data, f)

    def delete_variable(self, key: str):
        if key in self.env_data:
            del self.env_data[key]
            with open(self.env_file, "w") as f:
                json.dump(self.env_data, f)


env_vars = EnvironmentVariables()


def get_env_var(key: str):
    return env_vars.get_variable(key)


def set_env_var(key: str, value: str):
    env_vars.set_variable(key, value)


def delete_env_var(key: str):
    env_vars.delete_variable(key)
