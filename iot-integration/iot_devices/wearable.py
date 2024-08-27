import json
import os
import socket
import time

from cryptography.fernet import Fernet


class Wearable:

    def __init__(self, wearable_id, wearable_secret, home_id, home_secret):
        self.wearable_id = wearable_id
        self.wearable_secret = wearable_secret
        self.home_id = home_id
        self.home_secret = home_secret
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(("localhost", 8080))

    def send_data(self, data):
        self.socket.send(data.encode())

    def receive_command(self):
        command = self.socket.recv(1024)
        return command.decode()

    def authenticate(self):
        cipher_suite = Fernet(self.wearable_secret)
        encrypted_home_secret = cipher_suite.encrypt(self.home_secret.encode())
        self.socket.send(self.wearable_id.encode())
        self.socket.send(encrypted_home_secret)

    def start(self):
        self.authenticate()
        while True:
            data = "Heart rate: 100 bpm"
            self.send_data(data)
            time.sleep(1)


if __name__ == "__main__":
    wearable_id = "wearable123"
    wearable_secret = "secret123"
    home_id = "home123"
    home_secret = "secret123"
    wearable = Wearable(wearable_id, wearable_secret, home_id, home_secret)
    wearable.start()
